# 🧠 StockPred-ML (Open-Source Machine Learning Core)

> **S&P 500 (`^GSPC`) 및 NASDAQ-100 (`^NDX`) 익일 주가 등락 예측 머신러닝 오픈소스 엔진**  
> 본 레포지토리는 풀스택 금융 플랫폼의 백엔드 머신러닝 파이프라인을 독립 분리한 프로젝트로, 크루원들이 자유롭게 **파생변수(Feature) 추가, 하이퍼파라미터 튜닝, 신규 알고리즘 및 앙상블 실험, 학습 전략 개선**에 참여할 수 있도록 설계되었습니다.

---

## 🚀 빠른 시작 (Quick Start)

### 1. 미니콘다 `stock_pred` 가상환경 준비
모든 작업은 `stock_pred` 가상환경에서 진행합니다.

```bash
# Mac에서 XGBoost/LightGBM 구동용 OpenMP 설치 (최초 1회)
brew install libomp

# 가상환경 생성 및 활성화
conda create -n stock_pred python=3.11 -y
conda activate stock_pred

# 필수 패키지 설치
pip install -r requirements.txt
```

### 2. 정규 학습 및 벤치마크 성적표 출력
아래 명령어를 실행하면 S&P 500과 NASDAQ-100 모델을 학습하고, PR 본문에 바로 붙여넣을 수 있는 **Markdown 벤치마크 성적표**를 출력합니다.

```bash
python scripts/train_and_evaluate.py
```
- 빠른 실험 모드(트리 수 축소): `python scripts/train_and_evaluate.py --fast`
- Yahoo Finance 실시간 최신 데이터 반영: `python scripts/train_and_evaluate.py --yfinance`

### 3. 무결성 및 호환성 테스트 (PR 제출 전 필수)
```bash
pytest tests -v
```
- 결측치(`NaN`)/무한대(`Inf`) 유무, **미래 데이터 참조(Data Leakage) 여부**, 본 프로젝트 서버와의 `.pkl` 호환성을 자동으로 검증합니다.

---

## 🛠️ 크루원 4대 튜닝 영역 가이드

크루원 여러분은 복잡한 서버 코드를 건드릴 필요 없이 아래 **4개 핵심 파일**만 수정하여 성능을 개선할 수 있습니다. 자세한 예시는 [`CONTRIBUTING.md`](./CONTRIBUTING.md)를 참고하세요.

| 튜닝 영역 | 수정 대상 파일 | 핵심 함수 / 클래스 |
| :--- | :--- | :--- |
| **① 기술적 지표 / 파생변수 추가** | [`stockpred/feature_engine.py`](./stockpred/feature_engine.py) | `FeatureEngine.compute_custom_features()` 및 `CUSTOM_FEATURE_DESCRIPTIONS` |
| **② 하이퍼파라미터 튜닝** | [`stockpred/model_zoo.py`](./stockpred/model_zoo.py) | `DEFAULT_HYPERPARAMETERS` (`scripts/tune_optuna.py`로 자동 탐색 가능) |
| **③ 신규 알고리즘 및 앙상블 변경** | [`stockpred/model_zoo.py`](./stockpred/model_zoo.py) | `build_models()` (`MLP`, `ExtraTrees`, 가중치 보팅 등) |
| **④ 타겟 정의 및 학습/검증 전략** | [`stockpred/config.py`](./stockpred/config.py), [`stockpred/trainer.py`](./stockpred/trainer.py) | `TrainingConfig` (`WINDOW_SIZE`, `THRESHOLD`), `optimize_threshold()` |

---

## 📁 프로젝트 구조

```text
StockPred-ML/
├── stockpred/                     # 핵심 ML 코어 패키지
│   ├── config.py                  # [영역 ④] 타겟 Threshold, Window Size, 분할 비율 설정
│   ├── data_loader.py             # 로컬 시드 CSV 로드 및 yfinance 데이터 수집
│   ├── feature_engine.py          # [영역 ①] 기본 28개 지표 + 크루원 커스텀 피처 엔진
│   ├── model_zoo.py               # [영역 ②, ③] 모델 정의, 하이퍼파라미터 및 앙상블 구성
│   ├── trainer.py                 # [영역 ④] 시계열 학습, 임계값 최적화, 평가 및 .pkl 번들 저장
│   └── backtester.py              # 최근 20/60거래일 적중률(Hit Ratio) 백테스트 엔진
├── scripts/
│   ├── train_and_evaluate.py      # 전체 학습 + PR용 마크다운 벤치마크 리포트 출력 CLI
│   ├── tune_optuna.py             # 하이퍼파라미터 자동 탐색 도구
│   └── sync_to_main.py            # (관리자용) 검증된 모델/코드를 본 프로젝트 웹 서버로 동기화
├── notebooks/                     # 탐색적 데이터 분석(EDA) 및 모델 실험용 Jupyter 노트북
├── tests/                         # Data Leakage 및 메인 서버 호환성 자동 검증 테스트
├── csv/                           # S&P 500 & NASDAQ-100 시드 데이터셋 (*.csv)
├── CONTRIBUTING.md                # 크루원 코드 수정 예시 및 PR 제출 규칙
└── requirements.txt
```
