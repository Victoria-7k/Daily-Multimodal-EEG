from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.training.video_personalization import run_video_personalization


def main() -> int:
    parser = argparse.ArgumentParser(description="Run residual-bias few-shot personalization on video LOSO fold predictions.")
    parser.add_argument("--fold-report", required=True)
    parser.add_argument("--variant", default="B1")
    parser.add_argument("--k-events", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--include-affine", action="store_true", help="Also run regularized affine calibration y'=a*y+b.")
    parser.add_argument("--affine-slope-penalty", type=float, default=10.0)
    parser.add_argument("--affine-bias-penalty", type=float, default=10.0)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-table", required=True)
    args = parser.parse_args()

    result = run_video_personalization(
        fold_report=args.fold_report,
        variant=args.variant,
        out_json=args.out_json,
        out_table=args.out_table,
        k_events=args.k_events,
        include_affine=args.include_affine,
        affine_slope_penalty=args.affine_slope_penalty,
        affine_bias_penalty=args.affine_bias_penalty,
    )
    print(f"event_count={result['event_count']}")
    print(f"subject_count={result['subject_count']}")
    for row in result["protocols"]:
        print(
            "{protocol}: eligible={eligible} rmse_mean={rmse} pearson_r_mean={r}".format(
                protocol=row["protocol"],
                eligible=row["eligible_subjects"],
                rmse=row["rmse_mean"],
                r=row["pearson_r_mean"],
            )
        )
    print(f"json_path={args.out_json}")
    print(f"table_path={args.out_table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
