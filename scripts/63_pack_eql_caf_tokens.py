#!/usr/bin/env python3
"""Pack four temporal-token modality files into one EQL-CAF training NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

from daily_multimodal.temporal.pack_temporal_tokens import pack_temporal_token_files, write_pack_audit
from daily_multimodal.temporal.window_slicing import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-index", type=Path, required=True)
    parser.add_argument("--eeg", type=Path, required=True)
    parser.add_argument("--wear", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, help="Optional smoke limit for the temporal index.")
    args = parser.parse_args()

    rows = read_jsonl(args.temporal_index, max_rows=args.max_rows)
    report = pack_temporal_token_files(
        temporal_index_rows=rows,
        modality_paths={
            "eeg": args.eeg,
            "wear": args.wear,
            "video": args.video,
            "audio": args.audio,
        },
        output_path=args.out,
    )
    report.update({"temporal_index": str(args.temporal_index), "out": str(args.out)})
    write_pack_audit(report, args.audit_out)
    print(f"packed_row_count={report['packed']['row_count']}")
    print(f"available_token_ratio={report['packed']['available_token_ratio']:.6f}")
    print(f"out={args.out}")
    print(f"audit_out={args.audit_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
