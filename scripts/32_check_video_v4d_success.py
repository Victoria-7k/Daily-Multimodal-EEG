from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.training.video_v4d_success import evaluate_v4d_success_from_files, write_v4d_success_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the V4d success gate against V4a/V4d probe and fatigue reports.")
    parser.add_argument("--baseline-probe", required=True)
    parser.add_argument("--candidate-probe", required=True)
    parser.add_argument("--baseline-variant", action="append", required=True, help="Split=path JSON, e.g. LOSO=...")
    parser.add_argument("--candidate-variant", action="append", required=True, help="Split=path JSON, e.g. LOSO=...")
    parser.add_argument("--variant-name", default="V4d")
    parser.add_argument("--required-splits", nargs="+", default=["LOSO", "S4", "S2"])
    parser.add_argument("--min-probe-drop", type=float, default=0.0)
    parser.add_argument("--rmse-tolerance", type=float, default=0.0)
    parser.add_argument("--pearson-tolerance", type=float, default=0.0)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-table", required=True)
    args = parser.parse_args()

    result = evaluate_v4d_success_from_files(
        baseline_probe_path=args.baseline_probe,
        candidate_probe_path=args.candidate_probe,
        baseline_variant_paths=_split_specs(args.baseline_variant),
        candidate_variant_paths=_split_specs(args.candidate_variant),
        variant_name=args.variant_name,
        required_splits=tuple(args.required_splits),
        min_probe_drop=args.min_probe_drop,
        rmse_tolerance=args.rmse_tolerance,
        pearson_tolerance=args.pearson_tolerance,
    )
    write_v4d_success_report(result, out_json=args.out_json, out_table=args.out_table)
    print(f"passed={result['passed']}")
    print(f"json_path={args.out_json}")
    print(f"table_path={args.out_table}")
    return 0 if result["passed"] else 2


def _split_specs(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"split spec must be Split=path, got: {value}")
        split, path = value.split("=", 1)
        out[split] = path
    return out


if __name__ == "__main__":
    raise SystemExit(main())
