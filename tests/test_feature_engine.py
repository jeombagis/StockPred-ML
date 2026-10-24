import numpy as np
import pandas as pd
from stockpred.feature_engine import FeatureEngine, BASE_FEATURE_DESCRIPTIONS


class TestFeatureEngine:
    """크루원의 피처 엔지니어링 무결성 및 Data Leakage(미래 데이터 누수) 자동 검증 테스트"""

    def test_no_nan_or_inf_in_features(self, sample_stock_df):
        """생성된 모든 피처에 결측치(NaN)나 무한대(Inf)가 없어야 함"""
        engine = FeatureEngine()
        result = engine.compute_technical_indicators(sample_stock_df)

        assert len(engine.features) >= 20, "최소 20개 이상의 유효 피처가 생성되어야 합니다."
        for feat in engine.features:
            assert feat in result.columns, f"피처 '{feat}'가 DataFrame 컬럼에 존재하지 않습니다."

        assert not result[engine.features].isna().any().any(), "피처 컬럼에 결측치(NaN)가 포함되어 있습니다!"
        assert not np.isinf(result[engine.features].values).any(), "피처 컬럼에 무한대(Inf) 값이 포함되어 있습니다!"

    def test_no_future_data_leakage_in_features(self, sample_stock_df):
        """
        [핵심 검증] 미래 데이터 참조(Look-ahead Bias / Data Leakage) 방지 테스트:
        마지막 거래일(미래)의 주가/거래량을 극단적으로 변경하더라도,
        그 이전 과거 거래일들의 피처 값은 단 1개도 변해서는 안 됨 (shift(-1) 오용 탐지).
        """
        engine1 = FeatureEngine()
        res1 = engine1.compute_technical_indicators(sample_stock_df)

        modified_df = sample_stock_df.copy()
        modified_df.loc[modified_df.index[-1], "close"] *= 2.5
        modified_df.loc[modified_df.index[-1], "high"] *= 3.0
        modified_df.loc[modified_df.index[-1], "volume"] *= 10.0

        engine2 = FeatureEngine()
        res2 = engine2.compute_technical_indicators(modified_df)

        # 마지막 행을 제외한 과거 모든 행의 피처 값이 완전히 동일해야 함
        past_features_1 = res1[engine1.features].iloc[:-1]
        past_features_2 = res2[engine2.features].iloc[:-1]
        pd.testing.assert_frame_equal(
            past_features_1,
            past_features_2,
            obj="과거 거래일 피처가 미래 데이터 변경에 영향을 받았습니다 (Data Leakage 발생!)",
        )

    def test_split_and_scale_no_leakage(self, sample_stock_df):
        """Train/Test 시계열 분할 시 인덱스가 겹치지 않고 시간 순서가 보장되어야 함"""
        engine = FeatureEngine()
        result = engine.compute_technical_indicators(sample_stock_df)
        X_train, X_test, y_train, y_test, df_test = engine.split_and_scale(result, train_split=0.8)

        train_indices = set(X_train.index.tolist())
        test_indices = set(X_test.index.tolist())
        assert train_indices.isdisjoint(test_indices), "Train과 Test 인덱스가 중복됩니다!"
        assert X_train.index.max() < X_test.index.min(), "Train 데이터가 Test 데이터보다 시간적으로 앞서야 합니다!"

    def test_latest_row_preserved(self, sample_stock_df):
        """가장 최신 거래일(마지막 행)이 익일 예측을 위해 유실되지 않고 보존되어야 함"""
        engine = FeatureEngine()
        result = engine.compute_technical_indicators(sample_stock_df)

        assert result["date"].iloc[-1] == sample_stock_df["date"].iloc[-1], (
            "최신 거래일 행이 dropna 과정에서 유실되었습니다!"
        )

    def test_custom_feature_hook_extensibility(self, sample_stock_df):
        """크루원이 compute_custom_features를 오버라이드했을 때 정상 병합되는지 확인"""

        class CustomEngine(FeatureEngine):
            def compute_custom_features(self, df):
                df["custom_momentum_5"] = df["close"].pct_change(5)
                return df, ["custom_momentum_5"]

        engine = CustomEngine(window_size=5, threshold=0.002)
        result = engine.compute_technical_indicators(sample_stock_df)
        assert "custom_momentum_5" in engine.features
        assert "custom_momentum_5" in result.columns
        assert not result["custom_momentum_5"].isna().any()

    def test_does_not_mutate_global_state(self, sample_stock_df):
        """전역 BASE_FEATURE_DESCRIPTIONS가 오염되지 않는지 확인"""
        original_keys = set(BASE_FEATURE_DESCRIPTIONS.keys())
        engine = FeatureEngine(window_size=5, threshold=0.002)
        engine.compute_technical_indicators(sample_stock_df)
        assert set(BASE_FEATURE_DESCRIPTIONS.keys()) == original_keys
