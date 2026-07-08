from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.training.video_loso_diagnostics import analyze_loso_failure


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze video LOSO failure by subject labels and prediction signs.")
    parser.add_argument("--representations", required=True)
    parser.add_argument("--fold-report", help="Optional fold-level JSON from video ablation or GRL adapter run.")
    parser.add_argument("--variant", default="B1")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-table", required=True)
    args = parser.parse_args()

    result = analyze_loso_failure(
        representations=args.representations,
        fold_report=args.fold_report,
        variant=args.variant,
        out_json=args.out_json,
        out_table=args.out_table,
    )
    print(f"subject_count={len(result['subject_label_distribution'])}")
    print(f"positive_subjects={result['prediction_group_summary']['positive']['subject_count']}")
    print(f"negative_subjects={result['prediction_group_summary']['negative']['subject_count']}")
    print(f"json_path={args.out_json}")
    print(f"table_path={args.out_table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
