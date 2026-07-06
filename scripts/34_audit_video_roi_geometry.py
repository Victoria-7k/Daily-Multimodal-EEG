from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.embeddings.video_roi_audit import run_video_roi_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit upper-body ROI geometry by subject/session.")
    parser.add_argument("--window-index", required=True)
    parser.add_argument("--region-cache-root", default="outputs/cache/video_regions")
    parser.add_argument("--video-region", default="upper_body", choices=["upper_body"])
    parser.add_argument("--out-csv", default="outputs/reports/roi_audit/roi_session_summary.csv")
    parser.add_argument("--out-probe-json", default="outputs/reports/roi_audit/geometry_session_probe.json")
    parser.add_argument("--out-summary-md", default="outputs/reports/roi_audit/roi_audit_summary.md")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-probe-windows-per-session", type=int, default=2)
    args = parser.parse_args()

    result = run_video_roi_audit(
        window_index_path=args.window_index,
        region_cache_root=args.region_cache_root,
        video_region=args.video_region,
        out_csv=args.out_csv,
        out_probe_json=args.out_probe_json,
        out_summary_md=args.out_summary_md,
        seed=args.seed,
        n_splits=args.n_splits,
        min_probe_windows_per_session=args.min_probe_windows_per_session,
    )
    print(f"window_count={result['window_count']}")
    print(f"session_count={result['session_count']}")
    print(f"probe_subject_count={result['probe']['subject_count']}")
    print(f"probe_accuracy_mean={result['probe']['accuracy_mean']}")
    print(f"out_csv={result['out_csv']}")
    print(f"out_probe_json={result['out_probe_json']}")
    print(f"out_summary_md={result['out_summary_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
