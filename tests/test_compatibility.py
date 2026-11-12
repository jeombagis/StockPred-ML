import joblib
import numpy as np
import pandas as pd
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

    def test_baseline_metrics_reported(self, sample_stock_df, tmp_path):
        """다수 클래스 기준선 및 추가 지표(Balanced Accuracy, ROC-AUC)가 계산되어 번들에 저장되는지 검증"""
        trainer = ModelTrainer.train(
            asset_key="SnP500",
            raw_df=sample_stock_df,
            fast_mode=True,
            save_artifacts=True,
            output_dir=tmp_path,
        )

        baseline = trainer.baseline_metrics
        assert baseline["majority_class"] in (0, 1)
        expected_acc = round(float((trainer.y_test == baseline["majority_class"]).mean()), 4)
        assert baseline["accuracy"] == expected_acc
        assert baseline["balanced_accuracy"] == 0.5
        assert baseline["roc_auc"] == 0.5

        for metrics in trainer.eval_metrics.values():
            assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
            assert metrics["roc_auc"] is None or 0.0 <= metrics["roc_auc"] <= 1.0

        bundle = joblib.load(tmp_path / "models_SnP500.pkl")
        assert bundle["baseline_metrics"] == baseline

    def test_train_window_and_pooled_training(self, sample_stock_df, aux_stock_df, tmp_path):
        """학습 구간 제한(max_train_rows)과 합동 학습(aux_raw_dfs)이 누수 없이 동작하고 번들 규격을 유지하는지 검증"""
        trainer = ModelTrainer.train(
            asset_key="Nasdaq100",
            raw_df=sample_stock_df,
            fast_mode=True,
            save_artifacts=True,
            output_dir=tmp_path,
            max_train_rows=150,
            aux_raw_dfs={"SnP500": aux_stock_df},
        )

        # 대상 자산 150행 + 보조 자산 150행
        assert trainer.train_rows == 300
        assert trainer.pooled_assets == ["SnP500"]

        bundle = joblib.load(tmp_path / "models_Nasdaq100.pkl")
        assert bundle["parameters"]["max_train_rows"] == 150
        assert bundle["parameters"]["pooled_assets"] == ["SnP500"]
        assert "Ensemble" in bundle["models"]
        # 테스트셋은 대상 자산만으로 구성되고 피처 컬럼 순서가 번들과 일치해야 함
        assert list(bundle["X_test"].columns) == bundle["features"]
        assert len(bundle["X_test"]) == len(bundle["df_test"]) == len(bundle["y_test"])

    def test_prepare_datasets_no_leakage(self, sample_stock_df, aux_stock_df):
        """합동 학습 시 스케일러와 Train이 'Test 시작일 이전' 행(대상 + 보조 자산)으로만 구성되어야 함"""
        fe = FeatureEngine()
        completed = fe.compute_technical_indicators(sample_stock_df).iloc[:-1]
        aux = FeatureEngine().compute_technical_indicators(aux_stock_df).iloc[:-1]

        split_idx = int(len(completed) * 0.8)
        test_start = completed["date"].iloc[split_idx]

        X_train, X_test, y_train, y_test, df_test = ModelTrainer.prepare_datasets(
            fe, completed, train_split=0.8, max_train_rows=100, aux_completed_dfs=[aux]
        )

        expected_train = pd.concat(
            [
                completed.iloc[split_idx - 100:split_idx],
                aux[aux["date"] < test_start].iloc[-100:],
            ]
        )
        assert len(X_train) == len(y_train) == 200
        np.testing.assert_allclose(fe.scaler.mean_, expected_train[fe.features].mean().to_numpy())
        assert int(y_train.sum()) == int(expected_train["target"].sum())

        assert df_test["date"].iloc[0] == test_start
        assert len(X_test) == len(y_test) == len(completed) - split_idx
