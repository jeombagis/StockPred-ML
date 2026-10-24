from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
from stockpred.trainer import ModelTrainer, next_trading_day


class Backtester:
    """
    최근 N거래일(Lookback) 동안의 모델 예측과 실제 등락 결과를 비교하여
    적중률(Hit Ratio) 및 백테스트 히스토리를 산출하는 클래스 (웹 스키마 의존성 없음).
    """

    @staticmethod
    def evaluate(
        trainer: ModelTrainer,
        lookback_days: int = 20,
        processed_df: Optional[pd.DataFrame] = None,
        target_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        effective_thresh = (
            target_threshold if target_threshold is not None else getattr(trainer, "threshold", 0.002)
        )

        if processed_df is not None and not processed_df.empty:
            completed_df = processed_df.iloc[:-1] if len(processed_df) > 1 else processed_df
            actual_lookback = min(lookback_days, len(completed_df))
            recent_df = completed_df.tail(actual_lookback).copy().reset_index(drop=True)
            recent_X = trainer.scale_features(recent_df[trainer.features]).reset_index(drop=True)
        elif trainer.df_test is not None and trainer.X_test is not None:
            actual_lookback = min(lookback_days, len(trainer.df_test))
            recent_df = trainer.df_test.tail(actual_lookback).copy().reset_index(drop=True)
            recent_X = trainer.X_test.tail(actual_lookback).copy().reset_index(drop=True)
        else:
            return {
                "asset_name": trainer.asset_name,
                "total_days": 0,
                "hit_count": 0,
                "hit_ratio_pct": 0.0,
                "out_of_sample_acc": 0.0,
                "out_of_sample_f1": 0.0,
                "history": [],
            }

        if actual_lookback == 0:
            return {
                "asset_name": trainer.asset_name,
                "total_days": 0,
                "hit_count": 0,
                "hit_ratio_pct": 0.0,
                "out_of_sample_acc": 0.0,
                "out_of_sample_f1": 0.0,
                "history": [],
            }

        # 모든 등록된 모델에 대해 배치 확률 추론
        model_probs: Dict[str, Any] = {}
        for m_name, model in trainer.models.items():
            model_probs[m_name] = model.predict_proba(recent_X)[:, 1]

        ens_probs = model_probs.get("Ensemble", model_probs.get("XGB"))
        ens_thresh = trainer.best_thresholds.get("Ensemble", 0.5)

        records: List[Dict[str, Any]] = []
        hit_count = 0

        for i in range(actual_lookback):
            row = recent_df.iloc[i]
            ensemble_prob = float(ens_probs[i])
            ensemble_label = "상승" if ensemble_prob >= ens_thresh else "하락"

            raw_date = str(row["date"]).replace("-", "")
            try:
                base_dt = datetime.strptime(raw_date, "%Y%m%d")
            except Exception:
                base_dt = datetime.now()

            target_date_str = None
            if "target_date" in row and pd.notnull(row["target_date"]):
                raw_target = str(row["target_date"]).replace("-", "").split(".")[0]
                if len(raw_target) == 8:
                    target_date_str = f"{raw_target[:4]}-{raw_target[4:6]}-{raw_target[6:]}"

            if not target_date_str:
                target_dt = next_trading_day(base_dt.date())
                target_date_str = target_dt.strftime("%Y-%m-%d")

            actual_ret = float(row.get("Next_Return", 0.0))
            if "target" in row and pd.notnull(row["target"]):
                actual_is_up = bool(int(row["target"]) == 1)
            else:
                actual_is_up = bool(actual_ret > effective_thresh)

            actual_dir = "상승" if actual_is_up else "하락/보합"
            pred_is_up = ensemble_prob >= ens_thresh
            is_hit = pred_is_up == actual_is_up
            if is_hit:
                hit_count += 1

            record = {
                "base_date": base_dt.strftime("%Y-%m-%d"),
                "target_date": target_date_str,
                "ensemble_label": ensemble_label,
                "ensemble_prob": round(ensemble_prob, 4),
                "actual_return_pct": round(actual_ret * 100, 2),
                "actual_direction": actual_dir,
                "is_hit": is_hit,
            }
            records.append(record)

        hit_ratio = round((hit_count / actual_lookback) * 100, 1) if actual_lookback > 0 else 0.0
        ensemble_metrics = trainer.eval_metrics.get("Ensemble", {})

        return {
            "asset_name": trainer.asset_name,
            "total_days": actual_lookback,
            "hit_count": hit_count,
            "hit_ratio_pct": hit_ratio,
            "out_of_sample_acc": ensemble_metrics.get("accuracy", 0.0),
            "out_of_sample_f1": ensemble_metrics.get("f1_score", 0.0),
            "thresholds": {k: round(float(v), 4) for k, v in trainer.best_thresholds.items()},
            "history": list(reversed(records)),
        }
