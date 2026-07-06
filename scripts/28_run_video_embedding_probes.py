from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.training.video_embedding_probes import run_video_embedding_probes


def main() -> int:
    parser = argparse.ArgumentParser(description="Run subject/session/fatigue probes on one video embedding bundle.")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--train-embeddings", help="Optional train-fold-only embedding bundle; validation/test use --embeddings.")
    parser.add_argument("--target-label", default="fatigue")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-table", required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--fold-strategy",
        "--p3-fold-strategy",
        dest="fold_strategy",
        default="shuffled_k_fold",
        choices=[
            "shuffled_k_fold",
            "leave_one_subject_out",
            "grouped_k_fold",
            "within_subject_event_split",
            "within_subject_session_leave_out",
            "within_subject_chronological_split",
            "random_window_split",
        ],
    )
    args = parser.parse_args()

    result = run_video_embedding_probes(
        embeddings=args.embeddings,
        train_embeddings=args.train_embeddings,
        target_label=args.target_label,
        out_json=args.out_json,
        out_table=args.out_table,
        seed=args.seed,
        n_splits=args.n_splits,
        fold_strategy=args.fold_strategy,
    )
    print(f"row_count={result['row_count']}")
    for name, probe in result["probes"].items():
        print(f"{name}: {probe}")
    print(f"json_path={args.out_json}")
    print(f"table_path={args.out_table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
