#!/usr/bin/env python3
"""
[관리자 전용] StockPred-ML에서 크루원의 PR이 머지 및 검증 완료된 후,
본 프로젝트(Stock-price-prediction)에 변경된 피처 엔진 및 학습된 가중치(.pkl)를 동기화하는 스크립트입니다.
"""
import argparse
import shutil
from pathlib import Path

ML_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAIN_REPO = ML_REPO_ROOT.parent / "Stock-price-prediction"


def sync_to_main(main_repo_path: Path, sync_code: bool = False):
    if not main_repo_path.exists():
        raise FileNotFoundError(f"본 프로젝트 경로를 찾을 수 없습니다: {main_repo_path}")

    # 1. 사전 학습 가중치(.pkl) 파일 복사
    src_models_dir = ML_REPO_ROOT / "saved_models"
    dst_models_dir = main_repo_path / "saved_models"
    dst_models_dir.mkdir(parents=True, exist_ok=True)

    pkl_files = list(src_models_dir.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(
            f"{src_models_dir}에 생성된 .pkl 파일이 없습니다. 먼저 `python scripts/train_and_evaluate.py`를 실행하세요."
        )

    for pkl in pkl_files:
        shutil.copy2(pkl, dst_models_dir / pkl.name)
        print(f"✅ 가중치 복사 완료: {pkl.name} -> {dst_models_dir / pkl.name}")

    # 2. 선택 사항: feature_engine.py 동기화 안내
    if sync_code:
        src_fe = ML_REPO_ROOT / "stockpred" / "feature_engine.py"
        dst_fe = main_repo_path / "backend" / "app" / "services" / "feature_engine.py"
        content = src_fe.read_text(encoding="utf-8")
        # import 경로를 본 프로젝트 백엔드 구조(app.config)에 맞게 치환
        adapted = content.replace(
            "from stockpred.config import TrainingConfig",
            "from app.config import AppConfig as TrainingConfig",
        )
        dst_fe.write_text(adapted, encoding="utf-8")
        print(f"✅ 피처 엔진 코드 동기화 완료: {dst_fe}")

    print("\n🎉 본 프로젝트(Stock-price-prediction)로의 반영이 완료되었습니다!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="본 프로젝트(Stock-price-prediction) 동기화 도구")
    parser.add_argument(
        "--main-repo",
        type=Path,
        default=DEFAULT_MAIN_REPO,
        help="본 프로젝트 루트 디렉토리 경로",
    )
    parser.add_argument(
        "--sync-code",
        action="store_true",
        help=".pkl 파일뿐만 아니라 feature_engine.py 코드도 본 프로젝트 백엔드에 자동 변환하여 덮어씁니다",
    )
    args = parser.parse_args()
    sync_to_main(args.main_repo, sync_code=args.sync_code)
