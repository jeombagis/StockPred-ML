#!/usr/bin/env python3
"""
전체 자산(S&P 500, NASDAQ-100)에 대해 머신러닝 파이프라인을 실행하고,
PR 제출 시 본문에 바로 붙여넣을 수 있는 Markdown 벤치마크 성적표를 출력합니다.
"""
import sys
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stockpred.config import TrainingConfig, SAVED_MODELS_DIR
from stockpred.data_loader import DataLoader
from stockpred.feature_engine import FeatureEngine
from stockpred.trainer import ModelTrainer
from stockpred.backtester import Backtester


def _baseline_hit_ratio(targets, majority: int, lookback: int):
    """최근 N거래일 동안 항상 다수 클래스를 예측했을 때의 적중률(%) (Backtester 적중률과 비교용)"""
    recent = targets.tail(lookback)
    if len(recent) == 0:
        return None
    return round(float((recent == majority).mean()) * 100, 1)


def run_benchmark(use_yfinance: bool = False, fast_mode: bool = False):
    print("=" * 72)
    print("📈 StockPred-ML 정규 학습 및 벤치마크 평가 파이프라인")
    print("=" * 72)

    summary_rows = []
    feature_top5_map = {}

    # 합동 학습(POOLED_TRAINING_ASSETS)에 쓰기 위해 전 자산 데이터를 먼저 로드
    loaded = {
        key: DataLoader.get_stock_data(key, use_yfinance=use_yfinance)
        for key in TrainingConfig.PRESET_ASSETS
    }

    for asset_key, info in TrainingConfig.PRESET_ASSETS.items():
        print(f"\n🚀 [{info['name']} ({asset_key})] 데이터 로드 및 피처 엔지니어링 중...")
        raw_df, asset_name, ticker = loaded[asset_key]
        aux_raw_dfs = {
            aux_key: loaded[aux_key][0]
            for aux_key in TrainingConfig.POOLED_TRAINING_ASSETS.get(asset_key, [])
            if aux_key in loaded
        }

        trainer = ModelTrainer.train(
            asset_key=asset_key,
            raw_df=raw_df,
            window_size=TrainingConfig.WINDOW_SIZE,
            threshold=TrainingConfig.THRESHOLD,
            train_split=TrainingConfig.TRAIN_SPLIT,
            fast_mode=fast_mode,
            save_artifacts=True,
            aux_raw_dfs=aux_raw_dfs,
        )

        feat_engine = FeatureEngine(
            window_size=TrainingConfig.WINDOW_SIZE,
            threshold=TrainingConfig.THRESHOLD,
        )
        processed_df = feat_engine.compute_technical_indicators(raw_df)

        bt_20 = Backtester.evaluate(trainer, lookback_days=20, processed_df=processed_df)
        bt_60 = Backtester.evaluate(trainer, lookback_days=60, processed_df=processed_df)
        next_pred = trainer.predict_next_day(processed_df)
        top_features = trainer.get_feature_importances(model_name="XGB", top_k=5)
        feature_top5_map[asset_name] = top_features

        print(f" - 총 거래일 레코드: {trainer.total_records:,}건 | 사용 피처 수: {len(trainer.features)}개")
        pooled_label = f" (합동 학습: {', '.join(trainer.pooled_assets)})" if trainer.pooled_assets else ""
        print(f" - 학습 행 수: {trainer.train_rows:,}건{pooled_label}")
        print(
            f" - 최근 20거래일 적중률: {bt_20['hit_ratio_pct']:.1f}% ({bt_20['hit_count']}/{bt_20['total_days']})"
        )
        print(
            f" - 최근 60거래일 적중률: {bt_60['hit_ratio_pct']:.1f}% ({bt_60['hit_count']}/{bt_60['total_days']})"
        )
        print(
            f" - 최신 예측 ({next_pred['base_date']} -> {next_pred['target_date']}): "
            f"{next_pred['overall_direction']} (Ensemble 확률: {next_pred['avg_probability']*100:.2f}%)"
        )

        baseline = trainer.baseline_metrics
        majority = baseline["majority_class"]
        completed_targets = processed_df["target"].iloc[:-1]
        print(
            f" - 기준선(항상 '{'상승' if majority == 1 else '하락/보합'}' 예측) Test Accuracy: "
            f"{baseline['accuracy']:.4f} (Test 상승 비율 {baseline['test_positive_rate']*100:.1f}%)"
        )

        for model_name, metrics in trainer.eval_metrics.items():
            summary_rows.append(
                {
                    "asset": asset_name,
                    "model": model_name,
                    "accuracy": metrics["accuracy"],
                    "acc_delta": metrics["accuracy"] - baseline["accuracy"],
                    "f1_score": metrics["f1_score"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "roc_auc": metrics["roc_auc"],
                    "threshold": metrics["best_threshold"],
                    "hit_20d": bt_20["hit_ratio_pct"] if model_name == "Ensemble" else None,
                    "hit_60d": bt_60["hit_ratio_pct"] if model_name == "Ensemble" else None,
                }
            )

        summary_rows.append(
            {
                "asset": asset_name,
                "model": f"Baseline (항상 {majority})",
                "accuracy": baseline["accuracy"],
                "acc_delta": None,
                "f1_score": baseline["f1_score"],
                "balanced_accuracy": baseline["balanced_accuracy"],
                "roc_auc": baseline["roc_auc"],
                "threshold": None,
                "hit_20d": _baseline_hit_ratio(completed_targets, majority, 20),
                "hit_60d": _baseline_hit_ratio(completed_targets, majority, 60),
            }
        )

    # PR 첨부용 Markdown 리포트 출력
    print("\n" + "=" * 72)
    print("📋 [PR 본문 복사/붙여넣기용 벤치마크 리포트 (Markdown)]")
    print("=" * 72 + "\n")

    print("### 📊 벤치마크 성능 요약 (Out-of-Sample Test Set)")
    print(
        f"- **실험 설정**: `WINDOW_SIZE={TrainingConfig.WINDOW_SIZE}`, "
        f"`THRESHOLD={TrainingConfig.THRESHOLD}`, `TRAIN_SPLIT={TrainingConfig.TRAIN_SPLIT}`, "
        f"`TRAIN_WINDOW_ROWS={TrainingConfig.TRAIN_WINDOW_ROWS}`, "
        f"`POOLED_TRAINING_ASSETS={TrainingConfig.POOLED_TRAINING_ASSETS}`, "
        f"`총 피처 수={len(trainer.features)}개`\n"
    )
    print(
        "> 기준선(Baseline)은 Train 다수 클래스를 Test 전 기간 동일하게 예측한 결과입니다. "
        "`Δ Acc`가 음수면 해당 모델은 '항상 같은 답'보다 정확도가 낮습니다.\n"
    )
    print(
        "| 지수 (Asset) | 모델 (Model) | Accuracy | Δ Acc vs 기준선 | Macro F1 | Balanced Acc | ROC-AUC "
        "| 최적 Threshold | 최근 20일 적중률 | 최근 60일 적중률 |"
    )
    print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in summary_rows:
        h20 = f"**{r['hit_20d']:.1f}%**" if r["hit_20d"] is not None else "-"
        h60 = f"**{r['hit_60d']:.1f}%**" if r["hit_60d"] is not None else "-"
        m_label = f"**{r['model']}**" if r["model"] == "Ensemble" else r["model"]
        if r["model"].startswith("Baseline"):
            m_label = f"_{r['model']}_"
        delta = f"{r['acc_delta']:+.4f}" if r["acc_delta"] is not None else "-"
        auc = f"{r['roc_auc']:.4f}" if r["roc_auc"] is not None else "-"
        th = f"{r['threshold']:.4f}" if r["threshold"] is not None else "-"
        print(
            f"| {r['asset']} | {m_label} | {r['accuracy']:.4f} | {delta} | {r['f1_score']:.4f} "
            f"| {r['balanced_accuracy']:.4f} | {auc} | {th} | {h20} | {h60} |"
        )

    below = [f"{r['asset']} {r['model']}" for r in summary_rows if r["acc_delta"] is not None and r["acc_delta"] < 0]
    if below:
        print(f"\n⚠️ 기준선보다 Accuracy가 낮은 모델: {', '.join(below)}")

    print("\n### 🔍 상위 5개 핵심 피처 중요도 (XGBoost 기준)")
    for asset_name, feats in feature_top5_map.items():
        feat_str = ", ".join([f"`{f['feature']}` ({f['importance']*100:.1f}%)" for f in feats])
        print(f"- **{asset_name}**: {feat_str}")

    print(f"\n💾 학습된 가중치(.pkl) 저장 경로: `{SAVED_MODELS_DIR}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StockPred-ML 학습 및 벤치마크 리포트 생성기")
    parser.add_argument(
        "--yfinance",
        action="store_true",
        help="Yahoo Finance에서 최신 실시간 시세를 다운로드하여 병합합니다 (기본값: 로컬 시드 CSV 사용)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="빠른 실험을 위해 트리 개수(n_estimators=100)를 축소한 고속 모드로 학습합니다",
    )
    args = parser.parse_args()
    run_benchmark(use_yfinance=args.yfinance, fast_mode=args.fast)
