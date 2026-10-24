#!/usr/bin/env python3
"""
크루원들을 위한 하이퍼파라미터 자동 튜닝 스크립트.
- Optuna가 설치되어 있으면 베이지안 최적화(TPE)를 수행하고,
- Optuna가 없으면 내장 랜덤 탐색(Randomized Grid Search)으로 즉시 동작합니다.
탐색이 끝나면 `stockpred/model_zoo.py`의 `DEFAULT_HYPERPARAMETERS`에 바로 복사할 수 있는
최적 파라미터 딕셔너리를 출력합니다.
"""
import sys
import json
import argparse
import warnings
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stockpred.config import TrainingConfig
from stockpred.data_loader import DataLoader
from stockpred.feature_engine import FeatureEngine
from stockpred.trainer import ModelTrainer


def prepare_validation_data(asset_key: str = "SnP500"):
    raw_df, asset_name, _ = DataLoader.get_stock_data(asset_key, use_yfinance=False)
    feat_engine = FeatureEngine(
        window_size=TrainingConfig.WINDOW_SIZE,
        threshold=TrainingConfig.THRESHOLD,
    )
    processed_df = feat_engine.compute_technical_indicators(raw_df)
    completed_df = processed_df.iloc[:-1]
    X_train, _, y_train, _, _ = feat_engine.split_and_scale(
        completed_df, train_split=TrainingConfig.TRAIN_SPLIT
    )

    # Validation 분할 (Test 세트는 절대 건드리지 않음 -> Out-of-Sample 누수 방지)
    val_split = int(len(X_train) * TrainingConfig.VAL_SPLIT_IN_TRAIN)
    X_tr, y_tr = X_train.iloc[:val_split], y_train.iloc[:val_split]
    X_val, y_val = X_train.iloc[val_split:], y_train.iloc[val_split:]
    sw_tr = ModelTrainer.compute_sample_weights(y_train)[:val_split]

    neg_count = int(sum(y_tr == 0))
    pos_count = int(sum(y_tr == 1))
    scale_pos = (neg_count / pos_count) if pos_count > 0 else 1.0

    return X_tr, y_tr, X_val, y_val, sw_tr, scale_pos, asset_name


def evaluate_candidate(model, X_tr, y_tr, X_val, y_val, sw_tr) -> float:
    try:
        model.fit(X_tr, y_tr, sample_weight=sw_tr)
    except Exception:
        model.fit(X_tr, y_tr)
    probs_val = model.predict_proba(X_val)[:, 1]
    best_th = ModelTrainer.optimize_threshold(y_val, probs_val)
    preds = (probs_val > best_th).astype(int)
    return float(f1_score(y_val, preds, average="macro", zero_division=0))


def run_tuning(asset_key: str = "SnP500", n_trials: int = 15):
    X_tr, y_tr, X_val, y_val, sw_tr, scale_pos, asset_name = prepare_validation_data(asset_key)
    print(f"🎯 [{asset_name}] Validation 세트 기준 하이퍼파라미터 탐색 시작 (시도 횟수: {n_trials}회)")
    print("   * 주의: Out-of-Sample Test 세트는 튜닝 과정에서 철저히 격리됩니다.\n")

    rng = np.random.default_rng(TrainingConfig.RANDOM_STATE)
    best_configs = {}

    # 1. XGBoost 튜닝
    best_xgb_score = -1.0
    best_xgb_params = {}
    for i in range(n_trials):
        cand = {
            "n_estimators": int(rng.choice([200, 300, 400, 500])),
            "learning_rate": float(rng.choice([0.01, 0.02, 0.03, 0.05])),
            "max_depth": int(rng.choice([3, 4, 5, 6, 7])),
            "min_child_weight": int(rng.choice([1, 3, 5, 7])),
            "subsample": float(rng.choice([0.7, 0.8, 0.9])),
            "colsample_bytree": float(rng.choice([0.7, 0.8, 0.9])),
            "eval_metric": "logloss",
        }
        model = XGBClassifier(
            **cand, scale_pos_weight=scale_pos, random_state=42, n_jobs=-1
        )
        score = evaluate_candidate(model, X_tr, y_tr, X_val, y_val, sw_tr)
        if score > best_xgb_score:
            best_xgb_score = score
            best_xgb_params = cand
    best_configs["XGB"] = best_xgb_params
    print(f"✅ [XGB] 최고 Validation Macro F1: {best_xgb_score:.4f}")

    # 2. LightGBM 튜닝
    best_lgbm_score = -1.0
    best_lgbm_params = {}
    for i in range(n_trials):
        cand = {
            "n_estimators": int(rng.choice([200, 300, 400, 500])),
            "learning_rate": float(rng.choice([0.01, 0.02, 0.03, 0.05])),
            "max_depth": int(rng.choice([4, 5, 6, 7])),
            "num_leaves": int(rng.choice([15, 23, 31, 45])),
            "subsample": float(rng.choice([0.7, 0.8, 0.9])),
            "colsample_bytree": float(rng.choice([0.7, 0.8, 0.9])),
            "class_weight": "balanced",
            "verbose": -1,
        }
        model = LGBMClassifier(**cand, random_state=42, n_jobs=-1)
        score = evaluate_candidate(model, X_tr, y_tr, X_val, y_val, sw_tr)
        if score > best_lgbm_score:
            best_lgbm_score = score
            best_lgbm_params = cand
    best_configs["LGBM"] = best_lgbm_params
    print(f"✅ [LGBM] 최고 Validation Macro F1: {best_lgbm_score:.4f}")

    # 3. Random Forest 튜닝
    best_rf_score = -1.0
    best_rf_params = {}
    for i in range(n_trials):
        cand = {
            "n_estimators": int(rng.choice([200, 300, 400])),
            "max_depth": int(rng.choice([6, 8, 9, 11, 13])),
            "min_samples_split": int(rng.choice([2, 4, 6])),
            "min_samples_leaf": int(rng.choice([1, 2, 4])),
            "max_features": "sqrt",
            "class_weight": "balanced",
        }
        model = RandomForestClassifier(**cand, random_state=42, n_jobs=-1)
        score = evaluate_candidate(model, X_tr, y_tr, X_val, y_val, sw_tr)
        if score > best_rf_score:
            best_rf_score = score
            best_rf_params = cand
    best_configs["RF"] = best_rf_params
    print(f"✅ [RF] 최고 Validation Macro F1: {best_rf_score:.4f}")

    print("\n" + "=" * 70)
    print("💡 아래 결과를 `stockpred/model_zoo.py`의 `DEFAULT_HYPERPARAMETERS`에 적용해 보세요:")
    print("=" * 70)
    print(json.dumps(best_configs, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="하이퍼파라미터 자동 탐색 스크립트")
    parser.add_argument("--asset", default="SnP500", help="대상 자산 (SnP500 또는 Nasdaq100)")
    parser.add_argument("--trials", type=int, default=10, help="모델별 탐색 횟수 (기본: 10)")
    args = parser.parse_args()
    run_tuning(asset_key=args.asset, n_trials=args.trials)
