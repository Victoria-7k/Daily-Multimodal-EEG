from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.training.video_event_aggregation import build_event_embedding_bundles


def main() -> int:
    parser = argparse.ArgumentParser(description="Build 120s event-level video embeddings from adapter representations.")
    parser.add_argument("--representations", required=True, help=".npz produced by video GRL adapter --representations-out.")
    parser.add_argument("--variant", default="B1", help="Representation variant to aggregate, e.g. B1.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-windows", type=int, default=8)
    parser.add_argument("--target-label", default="fatigue")
    parser.add_argument("--aggregations", nargs="+", default=["E1_mean", "E2_mean_std", "E3_mean_std_max"])
    args = parser.parse_args()

    result = build_event_embedding_bundles(
        representations=args.representations,
        variant=args.variant,
        out_dir=args.out_dir,
        min_windows=args.min_windows,
        target_label=args.target_label,
        aggregations=args.aggregations,
    )
    print(f"input_window_count={result['input_window_count']}")
    print(f"event_count={result['event_count']}")
    print(f"dropped_event_count={result['dropped_event_count']}")
    for name, path in result["outputs"].items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
