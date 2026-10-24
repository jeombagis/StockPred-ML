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
    """
    WINDOW_SIZE: int = 5
    THRESHOLD: float = 0.002
    TRAIN_SPLIT: float = 0.8
    VAL_SPLIT_IN_TRAIN: float = 0.8
    TIME_DECAY_MIN_WEIGHT: float = 0.1
    RANDOM_STATE: int = 42

    # 임계값(Decision Threshold) 탐색 범위
    THRESHOLD_SEARCH_MIN: float = 0.40
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
