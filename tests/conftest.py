import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_stock_df(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2021-01-01", periods=n, freq="B")

    close_prices = 5000 + np.cumsum(rng.randn(n) * 20)

    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y%m%d"),
            "open": close_prices + rng.randn(n) * 5,
            "high": close_prices + abs(rng.randn(n) * 15),
            "low": close_prices - abs(rng.randn(n) * 15),
            "close": close_prices,
            "volume": rng.randint(1_000_000, 5_000_000, size=n).astype(float),
        }
    )

    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

    return df


@pytest.fixture
def sample_stock_df():
    """테스트용 샘플 주가 DataFrame 생성 (750일치 영업일, 장기 추세 피처는 최소 251일 필요)"""
    return _make_stock_df(750, seed=42)


@pytest.fixture
def aux_stock_df():
    """합동 학습(Pooled Training) 테스트용 보조 자산 DataFrame (같은 날짜, 다른 가격 경로)"""
    return _make_stock_df(750, seed=7)
