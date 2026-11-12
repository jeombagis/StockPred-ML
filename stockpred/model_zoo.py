import os
from typing import Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from stockpred.config import TrainingConfig


# ===========================================================================
# [크루원 튜닝 영역 ② & ③] 모델 하이퍼파라미터 튜닝 & 신규 알고리즘/앙상블 구성
# ===========================================================================
# 일간 지수 등락은 신호 대 잡음비가 매우 낮아, 깊고 유연한 트리는 잡음을 외웁니다.
# 워크포워드 검증(5폴드)에서 '얕은 트리 + 큰 리프 + 낮은 학습률' 조합이 기존 설정 대비
# XGB/LGBM 단독 AUC를 약 +0.014(S&P 500) 개선했습니다. 값을 바꿀 때는 반드시 워크포워드로 확인하세요.
DEFAULT_HYPERPARAMETERS: Dict[str, Dict[str, Any]] = {
    "XGB": {
        "n_estimators": 300,
        "learning_rate": 0.01,
        "max_depth": 3,
        "min_child_weight": 50,
        "subsample": 0.7,
        "colsample_bytree": 0.5,
        "reg_lambda": 10.0,
        "gamma": 1.0,
        "eval_metric": "logloss",
    },
    "RF": {
        "n_estimators": 500,
        "max_depth": 5,
        "min_samples_leaf": 100,
        "max_features": 0.3,
        "class_weight": "balanced",
    },
    "LGBM": {
        "n_estimators": 300,
        "learning_rate": 0.01,
        "max_depth": 3,
        "num_leaves": 8,
        "min_child_samples": 200,
        "subsample": 0.7,
        "subsample_freq": 1,
        "colsample_bytree": 0.5,
        "reg_lambda": 10.0,
        "class_weight": "balanced",
        "verbose": -1,
    },
    "LR": {
        "C": 0.01,
        "class_weight": "balanced",
        "max_iter": 1000,
        "solver": "liblinear",
    },
}


def build_models(
    scale_pos_weight: float = 1.0,
    fast_mode: bool = False,
    custom_params: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    개별 분류 모델 및 마스터 앙상블(Ensemble) 모델 딕셔너리를 생성하여 반환합니다.

    [크루원 가이드]
    1. 하이퍼파라미터 튜닝: 상단 `DEFAULT_HYPERPARAMETERS`의 값을 수정하거나,
       Optuna(`scripts/tune_optuna.py`)로 찾은 최적 파라미터를 반영하세요.
    2. 새로운 알고리즘 추가:
       - Scikit-Learn 인터페이스(`fit`, `predict_proba`)를 지원하는 어떤 분류기든 추가 가능합니다.
         (예: `ExtraTreesClassifier`, `MLPClassifier`, `CatBoostClassifier`, `GradientBoostingClassifier` 등)
       - 단, 본 프로젝트 웹 대시보드와의 완벽한 호환을 위해 최종 반환 딕셔너리에
         `'Ensemble'` 키는 반드시 포함되어야 하며, 기본 4대 키(`'XGB'`, `'RF'`, `'LGBM'`, `'LR'`)도 유지하는 것을 권장합니다.
    """
    default_n_jobs = "1" if os.environ.get("ENV", "").lower() == "production" else "-1"
    n_jobs = int(os.environ.get("MODEL_N_JOBS", default_n_jobs))
    seed = TrainingConfig.RANDOM_STATE

    params = {k: dict(v) for k, v in DEFAULT_HYPERPARAMETERS.items()}
    if custom_params:
        for model_key, overrides in custom_params.items():
            if model_key in params:
                params[model_key].update(overrides)
            else:
                params[model_key] = dict(overrides)

    if fast_mode:
        # 트리 수를 줄이는 대신 학습률을 높여 비슷한 적합 강도를 유지합니다.
        params["XGB"].update({"n_estimators": 100, "learning_rate": 0.03})
        params["RF"].update({"n_estimators": 150})
        params["LGBM"].update({"n_estimators": 100, "learning_rate": 0.03})
        params["LR"].update({"max_iter": 500})

    best_xgb = XGBClassifier(
        **params["XGB"],
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=n_jobs,
    )

    best_rf = RandomForestClassifier(
        **params["RF"],
        random_state=seed,
        n_jobs=n_jobs,
    )

    best_lgbm = LGBMClassifier(
        **params["LGBM"],
        random_state=seed,
        n_jobs=n_jobs,
    )

    best_lr = LogisticRegression(
        **params["LR"],
        random_state=seed,
    )

    # --- [신규 모델 추가 예시] ---
    # from sklearn.neural_network import MLPClassifier
    # best_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=seed)

    estimators_list = [
        ("XGB", best_xgb),
        ("RF", best_rf),
        ("LGBM", best_lgbm),
        ("LR", best_lr),
        # ("MLP", best_mlp),  # 앙상블에 신규 모델 포함 시 주석 해제
    ]

    # Soft Voting 앙상블 (필요 시 weights=[1.2, 1.0, 1.2, 0.8] 등 가중치 보팅 튜닝 가능)
    ensemble = VotingClassifier(
        estimators=estimators_list,
        voting="soft",
    )

    models_dict: Dict[str, Any] = {
        name: model for name, model in estimators_list
    }
    models_dict["Ensemble"] = ensemble

    return models_dict
