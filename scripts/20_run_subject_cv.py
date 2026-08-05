from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.training.subject_cv import run_subject_cv


def main() -> int:
    parser = argparse.ArgumentParser(description="Run subject-level cross-validation.")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--target-label", default="alert")
    parser.add_argument("--out-json", default="outputs/reports/subject_cv_real_v2_metrics.json")
    parser.add_argument("--out-table", default="outputs/reports/subject_cv_real_v2_table.md")
    parser.add_argument("--strategy", choices=["leave_one_subject_out", "grouped_k_fold"], default="leave_one_subject_out")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--modalities", default="eeg,wear,audio,face")
    parser.add_argument("--model", choices=["concat_mlp", "learnable_cross_attention"], default="concat_mlp")
    parser.add_argument("--min-available-modalities", type=int, default=2)
    parser.add_argument("--device")
    parser.add_argument("--fusion-spec", help="Fusion matrix config path for learnable_cross_attention branch experiments.")
    parser.add_argument("--fusion-experiment", help="Experiment name inside --fusion-spec.")
    args = parser.parse_args()

    result = run_subject_cv(
        embeddings=args.embeddings,
        target_label=args.target_label,
        out_json=args.out_json,
        out_table=args.out_table,
        strategy=args.strategy,
        n_splits=args.n_splits,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        seed=args.seed,
        modalities=tuple(item.strip() for item in args.modalities.split(",") if item.strip()),
        model=args.model,
        min_available_modalities=args.min_available_modalities,
        device=args.device,
        fusion_spec=args.fusion_spec,
        fusion_experiment=args.fusion_experiment,
    )
    print(f"fold_count={result['fold_count']}")
    print(f"subject_leakage={result['subject_leakage']}")
    print(f"rmse_mean={result['rmse_mean']}")
    print(f"rmse_std={result['rmse_std']}")
    print(f"pearson_r_mean={result['pearson_r_mean']}")
    print(f"pearson_r_std={result['pearson_r_std']}")
    print(f"json_path={args.out_json}")
    print(f"table_path={args.out_table}")
    return 0 if not result["subject_leakage"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
