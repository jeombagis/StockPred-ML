import logging
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from stockpred.config import TrainingConfig, SAVED_MODELS_DIR
from stockpred.feature_engine import FeatureEngine, get_feature_description
from stockpred.model_zoo import build_models

logger = logging.getLogger(__name__)

# 미국 증시 주요 고정 공휴일 (월, 일)
_US_MARKET_HOLIDAYS_FIXED = {
    (1, 1),    # New Year's Day
    (6, 19),   # Juneteenth National Independence Day
    (7, 4),    # Independence Day
    (12, 25),  # Christmas Day
}


def _easter_date(year: int) -> date:
    """Butcher's Algorithm을 사용한 부활절 일요일 계산"""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _is_us_market_holiday(d: date) -> bool:
    """미국 증시(NYSE/NASDAQ) 공휴일 여부 판정"""
    if d.weekday() >= 5:
        return True
    good_friday = _easter_date(d.year) - timedelta(days=2)
    if d == good_friday:
        return True

    for m, day in _US_MARKET_HOLIDAYS_FIXED:
        h_date = date(d.year, m, day)
        if h_date.weekday() < 5 and d == h_date:
            return True
        elif h_date.weekday() == 5 and d == h_date - timedelta(days=1):
            return True
        elif h_date.weekday() == 6 and d == h_date + timedelta(days=1):
            return True

    if d.month == 1 and d.weekday() == 0 and 15 <= d.day <= 21:
        return True
    if d.month == 2 and d.weekday() == 0 and 15 <= d.day <= 21:
        return True
    if d.month == 5 and d.weekday() == 0 and d.day >= 25:
        return True
    if d.month == 9 and d.weekday() == 0 and d.day <= 7:
        return True
    if d.month == 11 and d.weekday() == 3 and 22 <= d.day <= 28:
        return True

    return False


def next_trading_day(from_date: date) -> date:
    """다음 거래일 계산 (주말 + 미국 공휴일 건너뛰기)"""
    candidate = from_date + timedelta(days=1)
    for _ in range(10):
        if not _is_us_market_holiday(candidate):
            return candidate
        candidate += timedelta(days=1)
    return candidate


class ModelTrainer:
    """
    머신러닝 모델 학습, Validation 기반 최적 임계값(Threshold) 탐색,
    Out-of-Sample 평가, 익일 예측 및 본 프로젝트 호환 `.pkl` 번들 저장을 담당합니다.
    """

    def __init__(self):
        self.asset_key: Optional[str] = None
        self.asset_name: str = ""
        self.models: Dict[str, Any] = {}
        self.best_thresholds: Dict[str, float] = {}
        self.eval_metrics: Dict[str, Dict[str, float]] = {}
        self.features: List[str] = []
        self.scaler: Any = None
        self.window_size: int = TrainingConfig.WINDOW_SIZE
        self.threshold: float = TrainingConfig.THRESHOLD
        self.train_split: float = TrainingConfig.TRAIN_SPLIT
        self.train_date: str = ""
        self.total_records: int = 0
        self.X_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.Series] = None
        self.df_test: Optional[pd.DataFrame] = None

    @staticmethod
    def compute_sample_weights(y_train: pd.Series) -> np.ndarray:
        """
        [크루원 튜닝 영역 ④] 시계열 샘플 가중치(Time-Decay) 계산
        오래된 과거 데이터보다 최근 시장 트렌드에 더 높은 가중치를 부여합니다.
        """
        return np.linspace(TrainingConfig.TIME_DECAY_MIN_WEIGHT, 1.0, len(y_train))

    @staticmethod
    def optimize_threshold(y_val: pd.Series, probs_val: np.ndarray) -> float:
        """
        [크루원 튜닝 영역 ④] Validation 세트 기반 최적 결정 임계값(Threshold) 탐색
        기본은 Macro F1-Score 최대화 기준입니다.
        """
        best_th = 0.50
        if len(y_val) > 0 and len(np.unique(y_val)) > 1:
            best_score = 0.0
            thresholds = np.arange(
                TrainingConfig.THRESHOLD_SEARCH_MIN,
                TrainingConfig.THRESHOLD_SEARCH_MAX,
                TrainingConfig.THRESHOLD_SEARCH_STEP,
            )
            for th in thresholds:
                preds_th = (probs_val > th).astype(int)
                sc = f1_score(y_val, preds_th, average="macro", zero_division=0)
                if sc > best_score:
                    best_score = sc
                    best_th = float(th)
        return round(best_th, 4)

    @classmethod
    def train(
        cls,
        asset_key: str,
        raw_df: pd.DataFrame,
        window_size: int = TrainingConfig.WINDOW_SIZE,
        threshold: float = TrainingConfig.THRESHOLD,
        train_split: float = TrainingConfig.TRAIN_SPLIT,
        fast_mode: bool = False,
        custom_params: Optional[Dict[str, Dict[str, Any]]] = None,
        save_artifacts: bool = True,
        output_dir: Path = SAVED_MODELS_DIR,
    ) -> "ModelTrainer":
        """
        단일 자산에 대해 피처 생성 -> 시계열 분할 -> 모델 학습 -> 임계값 최적화 -> Out-of-Sample 평가를 수행합니다.
        """
        norm_key = TrainingConfig.resolve_asset_key(asset_key) or asset_key
        asset_info = TrainingConfig.PRESET_ASSETS.get(norm_key, {"name": norm_key})
        asset_name = asset_info["name"]

        feat_engine = FeatureEngine(window_size=window_size, threshold=threshold)
        processed_df = feat_engine.compute_technical_indicators(raw_df)
        features = feat_engine.features

        # 익일 수익률이 확정된 완료 거래일 기준 학습/테스트 분할
        completed_df = processed_df.iloc[:-1] if len(processed_df) > 1 else processed_df
        X_train, X_test, y_train, y_test, df_test = feat_engine.split_and_scale(
            completed_df, train_split=train_split
        )

        neg_count = int(sum(y_train == 0))
        pos_count = int(sum(y_train == 1))
        scale_pos = (neg_count / pos_count) if pos_count > 0 else 1.0
        sample_weights = cls.compute_sample_weights(y_train)

        models_dict = build_models(
            scale_pos_weight=scale_pos,
            fast_mode=fast_mode,
            custom_params=custom_params,
        )

        # Validation 세트 분할 (In-Sample 과적합 차단)
        val_split = max(int(len(X_train) * TrainingConfig.VAL_SPLIT_IN_TRAIN), 10)
        X_tr, y_tr = X_train.iloc[:val_split], y_train.iloc[:val_split]
        X_val, y_val = X_train.iloc[val_split:], y_train.iloc[val_split:]
        sw_tr = sample_weights[:val_split]

        best_thresholds: Dict[str, float] = {}
        eval_metrics: Dict[str, Dict[str, float]] = {}

        for name, model in models_dict.items():
            try:
                model.fit(X_tr, y_tr, sample_weight=sw_tr)
            except Exception:
                model.fit(X_tr, y_tr)

            probs_val = model.predict_proba(X_val)[:, 1] if len(X_val) > 0 else np.array([])
            best_th = cls.optimize_threshold(y_val, probs_val)
            best_thresholds[name] = best_th

            # 전체 Train 데이터셋으로 최종 재학습
            try:
                model.fit(X_train, y_train, sample_weight=sample_weights)
            except Exception:
                model.fit(X_train, y_train)

            # Out-of-Sample Test Set 평가
            probs_test = model.predict_proba(X_test)[:, 1]
            test_preds = (probs_test > best_th).astype(int)
            acc = float(accuracy_score(y_test, test_preds))
            macro_f1 = float(f1_score(y_test, test_preds, average="macro", zero_division=0))

            eval_metrics[name] = {
                "accuracy": round(acc, 4),
                "f1_score": round(macro_f1, 4),
                "best_threshold": best_th,
            }

        trainer = cls()
        trainer.asset_key = norm_key
        trainer.asset_name = asset_name
        trainer.models = models_dict
        trainer.best_thresholds = best_thresholds
        trainer.eval_metrics = eval_metrics
        trainer.features = features
        trainer.scaler = feat_engine.scaler
        trainer.window_size = window_size
        trainer.threshold = threshold
        trainer.train_split = train_split
        trainer.train_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        trainer.total_records = len(processed_df)
        trainer.X_test = X_test
        trainer.y_test = y_test
        trainer.df_test = df_test

        if save_artifacts:
            trainer.save_bundle(output_dir=output_dir)

        return trainer

    def save_bundle(self, output_dir: Path = SAVED_MODELS_DIR) -> Dict[str, Path]:
        """
        본 프로젝트(Stock-price-prediction) 백엔드 `ModelEngine.get_pretrained_engine()`과
        100% 호환되는 `.pkl` 파일 3종을 생성합니다.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        scaler_file = output_dir / f"scaler_{self.asset_key}.pkl"
        ensemble_file = output_dir / f"ensemble_{self.asset_key}.pkl"
        bundle_file = output_dir / f"models_{self.asset_key}.pkl"

        joblib.dump(self.scaler, scaler_file)
        if "Ensemble" in self.models:
            joblib.dump(self.models["Ensemble"], ensemble_file)

        bundle = {
            "asset_key": self.asset_key,
            "asset_name": self.asset_name,
            "models": self.models,
            "best_thresholds": self.best_thresholds,
            "eval_metrics": self.eval_metrics,
            "features": self.features,
            "train_date": self.train_date,
            "total_records": self.total_records,
            "X_test": self.X_test,
            "y_test": self.y_test,
            "df_test": self.df_test,
            "parameters": {
                "window_size": self.window_size,
                "threshold": self.threshold,
                "train_split": self.train_split,
            },
        }
        joblib.dump(bundle, bundle_file)
        return {
            "scaler": scaler_file,
            "ensemble": ensemble_file,
            "bundle": bundle_file,
        }

    def scale_features(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        if self.scaler is None:
            raise ValueError("스케일러가 초기화되지 않았습니다.")
        cols = self.features if self.features else X_raw.columns.tolist()
        scaled = self.scaler.transform(X_raw[cols])
        return pd.DataFrame(scaled, columns=cols, index=X_raw.index)

    def predict_next_day(self, processed_df: pd.DataFrame) -> Dict[str, Any]:
        """가장 최신 거래일 데이터를 바탕으로 익일 주가 등락 예측 수행"""
        last_row = processed_df.iloc[-1]
        X_last_raw = processed_df[self.features].iloc[[-1]]
        X_last = self.scale_features(X_last_raw)

        last_date_str = str(last_row["date"]).replace("-", "")
        try:
            last_date_obj = datetime.strptime(last_date_str, "%Y%m%d")
        except Exception:
            last_date_obj = datetime.now()

        target_date_obj = next_trading_day(last_date_obj.date())

        model_details = {}
        for name, model in self.models.items():
            prob = float(model.predict_proba(X_last)[0][1])
            thresh = self.best_thresholds.get(name, 0.5)
            signal = 1 if prob >= thresh else 0
            direction = "상승" if signal == 1 else "하락/보합"
            metrics = self.eval_metrics.get(name, {})
            model_details[name] = {
                "model_name": name,
                "signal": signal,
                "direction": direction,
                "probability": round(prob, 4),
                "threshold": round(thresh, 4),
                "f1_score": metrics.get("f1_score"),
                "accuracy": metrics.get("accuracy"),
            }

        ens_detail = model_details.get("Ensemble")
        ens_thresh = self.best_thresholds.get("Ensemble", 0.5)
        ensemble_prob = (
            ens_detail["probability"]
            if ens_detail
            else float(np.mean([d["probability"] for d in model_details.values()]))
        )
        overall_signal = 1 if ensemble_prob >= ens_thresh else 0

        diff = ensemble_prob - ens_thresh
        if diff >= 0.01:
            overall_direction = "상승 우세 ↑"
        elif diff <= -0.01:
            overall_direction = "하락/보합 우세 ↓"
        else:
            overall_direction = "중립 / 관망 ↔"

        return {
            "base_date": last_date_obj.strftime("%Y-%m-%d"),
            "target_date": target_date_obj.strftime("%Y-%m-%d"),
            "overall_direction": overall_direction,
            "overall_signal": overall_signal,
            "avg_probability": round(ensemble_prob, 4),
            "confidence_score": round(abs(diff) * 200, 1),
            "models": model_details,
        }

    def get_feature_importances(self, model_name: str = "XGB", top_k: int = 15) -> List[Dict[str, Any]]:
        """상위 K개 피처 중요도 반환"""
        model = self.models.get(model_name)
        importances = None
        if model is not None and hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif model is not None and hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            fallback = self.models.get("XGB")
            if fallback is not None and hasattr(fallback, "feature_importances_"):
                importances = fallback.feature_importances_

        if importances is None:
            return []

        total = float(np.sum(importances))
        normalized = (importances / total) if total > 0 else importances

        items = [
            {
                "feature": feat,
                "importance": round(float(imp), 4),
                "description": get_feature_description(feat),
            }
            for feat, imp in zip(self.features, normalized)
        ]
        return sorted(items, key=lambda x: x["importance"], reverse=True)[:top_k]
