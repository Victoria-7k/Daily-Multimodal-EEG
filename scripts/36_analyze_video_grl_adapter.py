from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.training.video_grl_adapter_analysis import (
    audit_grl_representations,
    summarize_grl_repeat_stability,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze video adapter/GRL repeat stability and representations.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stability = subparsers.add_parser("repeat-stability", help="Summarize repeated B1/B2 fold-level reports.")
    stability.add_argument("--report-root", required=True, help="Directory containing seed_* report subdirectories.")
    stability.add_argument("--variants", nargs="+", required=True)
    stability.add_argument("--splits", nargs="+", default=["LOSO", "S4", "S2"])
    stability.add_argument("--out-json", required=True)
    stability.add_argument("--out-table", required=True)

    audit = subparsers.add_parser("representation-audit", help="Probe exported out-of-fold adapter representations.")
    audit.add_argument("--representations", required=True, help=".npz produced by 35_run_video_grl_adapter_ablation.py.")
    audit.add_argument("--variants", nargs="+", required=True)
    audit.add_argument("--out-json", required=True)
    audit.add_argument("--out-table", required=True)
    audit.add_argument("--seed", type=int, default=41)
    audit.add_argument("--n-splits", type=int, default=5)

    args = parser.parse_args()
    if args.command == "repeat-stability":
        result = summarize_grl_repeat_stability(
            report_root=args.report_root,
            variants=args.variants,
            splits=args.splits,
            out_json=args.out_json,
            out_table=args.out_table,
        )
    else:
        result = audit_grl_representations(
            representations=args.representations,
            variants=args.variants,
            out_json=args.out_json,
            out_table=args.out_table,
            seed=args.seed,
            n_splits=args.n_splits,
        )
    print(f"analysis={args.command}")
    print(f"json_path={args.out_json}")
    print(f"table_path={args.out_table}")
    print(f"keys={','.join(result.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
