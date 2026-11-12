import glob
from pathlib import Path
from typing import Optional

# 프로젝트 루트 및 데이터/모델 저장 경로
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = PROJECT_ROOT / "csv"
CSV_DIR.mkdir(parents=True, exist_ok=True)
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = PROJECT_ROOT / "data_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class TrainingConfig:
    """
    ===========================================================================
    [크루원 튜닝 영역 ④] 타겟 정의, 윈도우 크기 및 학습/검증 분할 설정
    ===========================================================================
    - WINDOW_SIZE: 과거 며칠간의 수익률/거래량 시차(Lag) 피처를 생성할지 결정 (기본: 5거래일)
    - THRESHOLD: 익일 종가 수익률이 이 값보다 클 때만 '상승(1)'으로 분류 (기본: 0.002 = +0.2%)
    - TRAIN_SPLIT: 전체 시계열 데이터 중 학습(Train) 데이터 비율 (기본: 0.8 = 80%)
    - VAL_SPLIT_IN_TRAIN: Train 데이터 내부에서 최적 결정 임계값(Threshold) 튜닝에 사용할 비율 (기본: 0.8)
    - TIME_DECAY_MIN_WEIGHT: 과거 데이터의 최소 가중치 (오래된 데이터 0.1 -> 최신 데이터 1.0 선형 증가)
    - TRAIN_WINDOW_ROWS: 학습에 사용할 최근 거래일 수 (기본: 5040 = 약 20년, None이면 전체 기간 사용).
      1960~80년대 데이터는 현재 시장과 성격이 달라, 최근 20년만 쓰는 편이 워크포워드 AUC가 높았습니다.
    - BALANCE_CLASSES: True면 클래스 불균형 보정(class_weight='balanced', scale_pos_weight)을 켭니다 (기본: False).
      보정을 켜면 출력 확률이 0.5 근처로 끌려가 실제 상승 비율(약 0.42~0.46)보다 높게 나오고, 확률 정확도(Brier)가
      '항상 기저율만 답하는' 기준선보다 나빠집니다. 끄면 AUC는 그대로이고 확률이 실제 빈도와 맞습니다.
      상승/하락 판정은 어차피 Validation에서 고른 결정 임계값으로 하므로 보정이 없어도 한쪽으로 쏠리지 않습니다.
    - POOLED_TRAINING_ASSETS: {대상 자산: [함께 학습할 보조 자산]}. 보조 자산의 '테스트 시작일 이전' 행을
      학습 데이터에 합칩니다. Nasdaq 100은 S&P 500과 합쳐 학습할 때 AUC가 올랐고, S&P 500은 반대였습니다.
    """
    WINDOW_SIZE: int = 5
    THRESHOLD: float = 0.002
    TRAIN_SPLIT: float = 0.8
    VAL_SPLIT_IN_TRAIN: float = 0.8
    TIME_DECAY_MIN_WEIGHT: float = 0.1
    RANDOM_STATE: int = 42
    TRAIN_WINDOW_ROWS: Optional[int] = 5040
    BALANCE_CLASSES: bool = False
    POOLED_TRAINING_ASSETS = {
        "Nasdaq100": ["SnP500"],
    }

    # 임계값(Decision Threshold) 탐색 범위
    # 불균형 보정을 끄면 확률이 기저율(약 0.42~0.46) 주변에 분포하므로 하한을 0.30까지 넓혀 둡니다.
    THRESHOLD_SEARCH_MIN: float = 0.30
    THRESHOLD_SEARCH_MAX: float = 0.60
    THRESHOLD_SEARCH_STEP: float = 0.01

    # 지원 자산 프리셋 (S&P 500 및 NASDAQ-100)
    PRESET_ASSETS = {
        "SnP500": {
            "name": "S&P 500",
            "ticker": "^GSPC",
            "description": "미국 대형주 500개 기업 지수",
            "csv_pattern": "SnP500_*.csv",
        },
        "Nasdaq100": {
            "name": "Nasdaq 100",
            "ticker": "^NDX",
            "description": "미국 기술주 100개 기업 지수",
            "csv_pattern": "Nasdaq100_*.csv",
        },
    }

    @classmethod
    def resolve_asset_key(cls, identifier: str) -> Optional[str]:
        """자산 식별자(키 또는 티커)를 정규화된 자산 키('SnP500' 또는 'Nasdaq100')로 변환"""
        if not identifier:
            return None
        cleaned = identifier.strip()
        if cleaned in cls.PRESET_ASSETS:
            return cleaned

        lower_cleaned = cleaned.lower()
        if lower_cleaned in ["snp500", "s&p500", "s&p 500", "^gspc", "gspc"]:
            return "SnP500"
        if lower_cleaned in ["nasdaq100", "nasdaq 100", "ndx", "^ndx", "qqq"]:
            return "Nasdaq100"
        return None

    @classmethod
    def get_csv_for_preset(cls, asset_key: str) -> Optional[str]:
        """csv/ 디렉토리에서 가장 최신 시드 CSV 파일 경로를 반환"""
        norm_key = cls.resolve_asset_key(asset_key)
        if norm_key and norm_key in cls.PRESET_ASSETS:
            pattern = str(CSV_DIR / cls.PRESET_ASSETS[norm_key]["csv_pattern"])
            matches = glob.glob(pattern)
            if matches:
                return sorted(matches)[-1]
        return None
