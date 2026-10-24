import os
import logging
from datetime import datetime, timedelta
from typing import Tuple, Optional
import pandas as pd
import yfinance as yf
from stockpred.config import TrainingConfig, CACHE_DIR

logger = logging.getLogger(__name__)


class DataLoader:
    """
    로컬 시드 CSV 파일 로드 및 Yahoo Finance 최신 시세 동기화를 담당하는 데이터 로더.
    웹 서버 의존성 없이 단독으로 실행 가능합니다.
    """

    @staticmethod
    def get_stock_data(
        identifier: str,
        use_yfinance: bool = False,
        force_download: bool = False
    ) -> Tuple[pd.DataFrame, str, str]:
        """
        주식 데이터를 반환합니다.
        - 기본값(use_yfinance=False): 재현 가능한 실험(Reproducible Experiment)을 위해 로컬 시드 CSV를 우선 로드합니다.
        - use_yfinance=True 또는 시드 CSV가 없을 때: Yahoo Finance에서 최신 시세를 다운로드하여 병합합니다.

        Returns:
            df (pd.DataFrame): 'date', 'open', 'high', 'low', 'close', 'volume' 컬럼 DataFrame (오름차순 정렬)
            asset_name (str): 표시용 자산 이름
            ticker (str): 자산 티커
        """
        asset_key = TrainingConfig.resolve_asset_key(identifier)
        if not asset_key or asset_key not in TrainingConfig.PRESET_ASSETS:
            raise ValueError(
                f"지원하지 않는 자산입니다: '{identifier}'. "
                f"지원 자산: {list(TrainingConfig.PRESET_ASSETS.keys())}"
            )

        preset_info = TrainingConfig.PRESET_ASSETS[asset_key]
        asset_name = preset_info["name"]
        ticker = preset_info["ticker"]
        csv_path = TrainingConfig.get_csv_for_preset(asset_key)
        cache_file = CACHE_DIR / f"{asset_key}_daily.csv"

        # 1. 재현성 중심 모드: 로컬 시드 CSV 우선 반환
        if not use_yfinance and not force_download and csv_path and os.path.exists(csv_path):
            df = DataLoader._load_local_csv(csv_path)
            return df, asset_name, ticker

        # 2. 캐시 또는 시드 CSV 베이스라인 확보
        existing_df = None
        if not force_download and cache_file.exists():
            existing_df = DataLoader._load_local_csv(str(cache_file))
        if existing_df is None and csv_path and os.path.exists(csv_path):
            existing_df = DataLoader._load_local_csv(csv_path)

        # 3. Yahoo Finance 다운로드
        try:
            start_date = "1970-01-01"
            if existing_df is not None and len(existing_df) > 0 and not force_download:
                last_date_str = str(existing_df["date"].iloc[-1])
                try:
                    last_dt = datetime.strptime(last_date_str, "%Y%m%d")
                    start_date = (last_dt - timedelta(days=5)).strftime("%Y-%m-%d")
                except ValueError:
                    pass

            new_df = DataLoader._download_from_yfinance(ticker, start_date=start_date)
            if new_df is not None and not new_df.empty:
                if existing_df is not None and not force_download:
                    combined = pd.concat([existing_df, new_df], ignore_index=True)
                    combined = combined.drop_duplicates(subset=["date"], keep="last")
                    df = combined.sort_values(by="date", ascending=True).reset_index(drop=True)
                else:
                    df = new_df

                if len(df) >= 50:
                    df.to_csv(cache_file, index=False)
                    return df, asset_name, ticker
        except Exception as err:
            logger.warning(f"[{asset_key}] Yahoo Finance 다운로드 실패, 로컬 데이터로 대체합니다: {err}")

        if existing_df is not None and len(existing_df) >= 50:
            return existing_df, asset_name, ticker

        raise ValueError(f"데이터를 가져올 수 없습니다: {identifier} ({ticker})")

    @staticmethod
    def _load_local_csv(file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        df.columns = [c.lower() for c in df.columns]

        df["date"] = df["date"].astype(str).str.replace("-", "")
        df = df.sort_values(by="date", ascending=True).reset_index(drop=True)

        cols = ["open", "high", "low", "close", "volume"]
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=cols).reset_index(drop=True)
        return df

    @staticmethod
    def _download_from_yfinance(ticker: str, start_date: str = "1970-01-01") -> Optional[pd.DataFrame]:
        end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df is None or df.empty:
                return None

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            df.rename(
                columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                },
                inplace=True,
            )

            df = df[["date", "open", "high", "low", "close", "volume"]].dropna()
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
            df = df.sort_values(by="date", ascending=True).reset_index(drop=True)
            return df
        except Exception as e:
            logger.error(f"Yahoo Finance 다운로드 에러 ({ticker}): {e}")
            return None
