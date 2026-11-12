# 🤝 크루원 기여 가이드 (CONTRIBUTING.md)

`StockPred-ML`에 기여해 주셔서 감사합니다! 크루원 누구나 새로운 피처, 하이퍼파라미터, 머신러닝 알고리즘을 실험하고 **Pull Request(PR)**로 제안할 수 있습니다.

---

## 📌 영역별 코드 수정 방법

### 1️⃣ 새로운 기술적 지표(Feature) 추가하기
[`stockpred/feature_engine.py`](./stockpred/feature_engine.py)를 열고 두 곳만 수정하면 됩니다.

1. `CUSTOM_FEATURE_DESCRIPTIONS`에 피처 이름과 한글 설명을 등록합니다:
```python
CUSTOM_FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "close_open_gap": "당일 시가 대비 종가 등락 비율 (캔들 몸통 비율)",
    "vol_ma5_ratio": "5일 평균 거래량 대비 당일 거래량 비율",
}
```
2. `FeatureEngine.compute_custom_features()` 메서드 안에 계산 로직을 추가합니다:
```python
def compute_custom_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    custom_feature_names: List[str] = []

    df["close_open_gap"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
    df["vol_ma5_ratio"] = df["volume"] / df["volume"].rolling(5).mean().replace(0, np.nan)

    custom_feature_names.extend(["close_open_gap", "vol_ma5_ratio"])
    return df, custom_feature_names
```
> ⚠️ **주의 (메인 서버 동기화)**: 메인 서버는 예측 시 자체 `feature_engine.py`로 피처를 계산합니다. 피처를 추가·변경한 PR이 머지되면 관리자는 `.pkl`과 함께 `python scripts/sync_to_main.py --sync-code`로 피처 엔진 코드도 반드시 동기화해야 합니다. 또한 피처는 OHLCV와 날짜만으로 계산되어야 하며(외부 데이터 불가), 장기 추세 피처 때문에 최소 251거래일의 데이터가 필요합니다.
>
> ⚠️ **주의 (Data Leakage 금지)**: 피처 계산 시 `shift(-1)`과 같이 미래 시점의 데이터를 참조하면 `pytest`의 `test_no_future_data_leakage_in_features` 테스트에서 자동으로 차단됩니다.

---

### 2️⃣ 하이퍼파라미터 튜닝하기
자동 탐색 스크립트를 먼저 실행해 볼 수 있습니다:
```bash
conda activate stock_pred
python scripts/tune_optuna.py --asset SnP500 --trials 20
```
출력된 최적 파라미터를 [`stockpred/model_zoo.py`](./stockpred/model_zoo.py) 상단의 `DEFAULT_HYPERPARAMETERS`에 반영합니다.

---

### 3️⃣ 새로운 알고리즘 추가 및 앙상블 조합 변경하기
[`stockpred/model_zoo.py`](./stockpred/model_zoo.py)의 `build_models()` 함수에서 `predict_proba()`를 지원하는 모든 Scikit-Learn 호환 모델을 추가할 수 있습니다.

```python
from sklearn.ensemble import ExtraTreesClassifier

best_et = ExtraTreesClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=seed,
    n_jobs=n_jobs,
)

estimators_list = [
    ("XGB", best_xgb),
    ("RF", best_rf),
    ("LGBM", best_lgbm),
    ("LR", best_lr),
    ("ET", best_et),  # 신규 모델 추가
]
```
> ⚠️ **주의 (확률 보정)**: 새 모델에 `class_weight="balanced"`나 `scale_pos_weight`를 직접 넣지 마세요. 출력 확률이 실제 상승 빈도보다 높아져 벤치마크 표의 `Brier`가 기준선보다 나빠집니다. 불균형 보정이 필요하면 `TrainingConfig.BALANCE_CLASSES`로 전체를 한 번에 켜고 결과를 비교하세요.
>
> 💡 **필수 호환성 규칙**: 본 프로젝트 웹 대시보드와의 호환을 위해 최종 반환 딕셔너리에 `'Ensemble'` 키는 반드시 유지되어야 합니다.

---

### 4️⃣ 타겟 정의 및 학습 전략 수정하기
- [`stockpred/config.py`](./stockpred/config.py): `WINDOW_SIZE`(시차 피처 일수), `THRESHOLD`(상승 판단 기준 수익률), `TIME_DECAY_MIN_WEIGHT`(과거 데이터 가중치)를 조정할 수 있습니다.
- [`stockpred/config.py`](./stockpred/config.py): `TRAIN_WINDOW_ROWS`(학습에 쓸 최근 거래일 수, 기본 약 20년), `BALANCE_CLASSES`(클래스 불균형 보정, 기본 꺼짐)와 `POOLED_TRAINING_ASSETS`(다른 지수와 합쳐 학습할지)를 조정할 수 있습니다.
- [`stockpred/trainer.py`](./stockpred/trainer.py): `prepare_datasets()`(학습 데이터 구성), `compute_sample_weights()` 또는 `optimize_threshold()` 함수를 수정하여 임계값 탐색 전략을 고도화할 수 있습니다.

---

## ✅ PR 제출 전 체크리스트

1. **자동 검증 테스트 통과 확인**
   ```bash
   conda activate stock_pred
   pytest tests -v
   ```
   모든 테스트(`13 passed`)가 통과해야 합니다.

2. **벤치마크 성적표 생성 및 PR 본문 첨부**
   ```bash
   python scripts/train_and_evaluate.py
   ```
   터미널 하단에 출력되는 `📋 [PR 본문 복사/붙여넣기용 벤치마크 리포트 (Markdown)]` 표를 복사하여 PR 설명란에 붙여넣어 주세요.
   표의 `Baseline` 행은 Train 다수 클래스를 항상 예측한 결과입니다. **`Δ Acc vs 기준선`이 양수이고 ROC-AUC가 0.5보다 높아야** 실제로 의미 있는 개선으로 볼 수 있습니다. `Brier`(확률 정확도, 낮을수록 좋음)도 `Baseline` 행보다 낮아야 합니다.
   *(주의: `.pkl` 가중치 파일은 용량 및 충돌 방지를 위해 Git에 커밋하지 않으며, 코드가 머지된 후 메인 서버에서 재학습됩니다.)*
