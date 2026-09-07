#!/usr/bin/env python3
"""Build the EQL-CAF 5x2s temporal-token index from the canonical window index."""

from __future__ import annotations

import argparse
from pathlib import Path

from daily_multimodal.temporal.window_slicing import (
    build_temporal_token_index,
    read_jsonl,
    write_json,
    write_jsonl,
)


DEFAULT_INDEX = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/index/eeg_aligned_window_index.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--token-seconds", type=float, default=2.0)
    parser.add_argument("--window-seconds", type=float)
    parser.add_argument("--max-rows", type=int, help="Optional smoke limit.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.index_path, max_rows=args.max_rows)
    temporal_rows, audit = build_temporal_token_index(
        rows,
        token_seconds=args.token_seconds,
        window_seconds=args.window_seconds,
    )
    audit.update(
        {
            "source_index": str(args.index_path),
            "output_index": str(args.out),
            "max_rows": args.max_rows,
        }
    )
    write_jsonl(temporal_rows, args.out)
    write_json(audit, args.audit_out)
    print(f"row_count={audit['row_count']}")
    print(f"token_count_distribution={audit['token_count_distribution']}")
    print(f"out={args.out}")
    print(f"audit_out={args.audit_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
