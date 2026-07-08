from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.training.video_transfer_matrix import analyze_cross_subject_transfer


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze cross-subject transfer of centered B1 fatigue mappings.")
    parser.add_argument("--representations", required=True)
    parser.add_argument("--variant", default="B1")
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-table", required=True)
    args = parser.parse_args()

    result = analyze_cross_subject_transfer(
        representations=args.representations,
        variant=args.variant,
        ridge_alpha=args.ridge_alpha,
        out_json=args.out_json,
        out_table=args.out_table,
    )
    print(f"subject_count={result['subject_count']}")
    print(f"positive_pairs={result['sign_summary']['positive_pairs']}")
    print(f"negative_pairs={result['sign_summary']['negative_pairs']}")
    print(f"diagonal_r_mean={result['sign_summary']['diagonal_r_mean']}")
    print(f"off_diagonal_r_mean={result['sign_summary']['off_diagonal_r_mean']}")
    print(f"json_path={args.out_json}")
    print(f"table_path={args.out_table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
