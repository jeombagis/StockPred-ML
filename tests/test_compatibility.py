import joblib
from stockpred.trainer import ModelTrainer
from stockpred.feature_engine import FeatureEngine
from stockpred.backtester import Backtester


class TestMainProjectCompatibility:
    """
    크루원이 튜닝한 모델과 `.pkl` 번들이 본 프로젝트(Stock-price-prediction) 백엔드와
    100% 호환되는지 검증하는 계약(Contract) 테스트
    """

    def test_trainer_and_bundle_contract(self, sample_stock_df, tmp_path):
        """학습 파이프라인 및 생성된 .pkl 번들이 메인 서버 필수 스키마를 모두 충족하는지 검증"""
        trainer = ModelTrainer.train(
            asset_key="SnP500",
            raw_df=sample_stock_df,
            fast_mode=True,
            save_artifacts=True,
            output_dir=tmp_path,
        )

        # 1. 필수 파일 3종 존재 확인
        bundle_path = tmp_path / "models_SnP500.pkl"
        scaler_path = tmp_path / "scaler_SnP500.pkl"
        ensemble_path = tmp_path / "ensemble_SnP500.pkl"

        assert bundle_path.exists(), "models_SnP500.pkl 파일이 생성되지 않았습니다."
        assert scaler_path.exists(), "scaler_SnP500.pkl 파일이 생성되지 않았습니다."
        assert ensemble_path.exists(), "ensemble_SnP500.pkl 파일이 생성되지 않았습니다."

        # 2. 번들 내부 필수 키 검증 (본 프로젝트 ModelEngine.get_pretrained_engine 규격)
        bundle = joblib.load(bundle_path)
        required_keys = {
            "asset_key",
            "asset_name",
            "models",
            "best_thresholds",
            "eval_metrics",
            "features",
            "train_date",
            "total_records",
            "X_test",
            "y_test",
            "df_test",
            "parameters",
        }
        assert required_keys.issubset(set(bundle.keys())), (
            f"번들에 필수 키가 누락되었습니다: {required_keys - set(bundle.keys())}"
        )

        # 3. 마스터 앙상블('Ensemble') 포함 및 predict_proba 인터페이스 검증
        models = bundle["models"]
        assert "Ensemble" in models, "models 딕셔너리에 'Ensemble' 키가 반드시 포함되어야 합니다."
        for m_name, model in models.items():
            assert hasattr(model, "predict_proba"), f"'{m_name}' 모델이 predict_proba()를 지원하지 않습니다."
            assert m_name in bundle["best_thresholds"]
            assert m_name in bundle["eval_metrics"]
            assert "accuracy" in bundle["eval_metrics"][m_name]
            assert "f1_score" in bundle["eval_metrics"][m_name]

        # 4. 익일 예측 및 백테스터 정상 작동 확인
        fe = FeatureEngine(window_size=trainer.window_size, threshold=trainer.threshold)
        processed_df = fe.compute_technical_indicators(sample_stock_df)

        pred = trainer.predict_next_day(processed_df)
        assert "overall_direction" in pred
        assert 0.0 <= pred["avg_probability"] <= 1.0

        bt = Backtester.evaluate(trainer, lookback_days=20, processed_df=processed_df)
        assert bt["total_days"] == 20
        assert 0.0 <= bt["hit_ratio_pct"] <= 100.0
