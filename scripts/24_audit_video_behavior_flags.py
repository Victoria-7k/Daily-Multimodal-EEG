from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.embeddings.video_behavior_flags import (
    audit_behavior_flags,
    write_behavior_audit_markdown,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit video behavior flag JSONL before embedding or ablation use."
    )
    parser.add_argument(
        "--flags",
        required=True,
        help="Window-level behavior flags JSONL from scripts/23_extract_video_behavior_flags.py.",
    )
    parser.add_argument(
        "--openface-embeddings",
        help="Optional OpenFace/face embedding npz to join mask and quality flags by sample_id.",
    )
    parser.add_argument("--out-json", required=True, help="Output JSON audit report.")
    parser.add_argument("--out-table", required=True, help="Output Markdown review table.")
    parser.add_argument("--top-k", type=int, default=20, help="Rows per top/random review set.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic random_windows review rows.",
    )
    args = parser.parse_args()

    report = audit_behavior_flags(
        args.flags,
        openface_embeddings=args.openface_embeddings,
        top_k=args.top_k,
        random_seed=args.seed,
    )
    write_json(report, args.out_json)
    write_behavior_audit_markdown(report, args.out_table)
    print(f"audit_json_path={args.out_json}")
    print(f"audit_table_path={args.out_table}")
    print(f"window_count={report['window_count']}")
    print(f"success_count={report['success_count']}")
    print(f"missing_count={report['missing_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
