import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def sample_stock_df():
    """테스트용 샘플 주가 DataFrame 생성 (250일치 영업일)"""
    np.random.seed(42)
    n = 250
    dates = pd.date_range("2024-01-01", periods=n, freq="B")

    close_prices = 5000 + np.cumsum(np.random.randn(n) * 20)

    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y%m%d"),
            "open": close_prices + np.random.randn(n) * 5,
            "high": close_prices + abs(np.random.randn(n) * 15),
            "low": close_prices - abs(np.random.randn(n) * 15),
            "close": close_prices,
            "volume": np.random.randint(1_000_000, 5_000_000, size=n).astype(float),
        }
    )

    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

    return df
