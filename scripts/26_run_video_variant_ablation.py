from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.training.video_variant_ablation import run_video_variant_ablation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run video-only V1/V2 face-slot ablation.")
    parser.add_argument("--target-label", required=True)
    parser.add_argument("--mode", default="video_only", choices=["video_only"])
    parser.add_argument("--sample-mode", required=True, choices=["strict_aligned", "behavior_retained"])
    parser.add_argument("--variants", nargs="+", required=True, help="Variant specs such as V1=path.npz V0=mean_baseline")
    parser.add_argument("--bucket-flags", help="Optional behavior flags JSONL used for bucketed test-set metrics.")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-table", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--fold-strategy",
        default="leave_one_subject_out",
        choices=[
            "leave_one_subject_out",
            "grouped_k_fold",
            "within_subject_event_split",
            "within_subject_session_leave_out",
            "within_subject_chronological_split",
            "random_window_split",
        ],
    )
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    result = run_video_variant_ablation(
        variants=args.variants,
        target_label=args.target_label,
        mode=args.mode,
        sample_mode=args.sample_mode,
        out_json=args.out_json,
        out_table=args.out_table,
        bucket_flags_path=args.bucket_flags,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        seed=args.seed,
        fold_strategy=args.fold_strategy,
        n_splits=args.n_splits,
    )
    print(f"experiment_count={len(result['experiments'])}")
    for name, experiment in result["experiments"].items():
        print(
            "{name}: rows={rows} rmse_mean={rmse} pearson_r_mean={pearson} pred_std_mean={pred_std}".format(
                name=name,
                rows=experiment.get("row_count", 0),
                rmse=experiment.get("rmse_mean"),
                pearson=experiment.get("pearson_r_mean"),
                pred_std=experiment.get("pred_std_mean"),
            )
        )
    print(f"paired_delta_count={len(result.get('paired_fold_deltas', {}).get('V2_vs_V1', []))}")
    print(f"json_path={args.out_json}")
    print(f"table_path={args.out_table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
