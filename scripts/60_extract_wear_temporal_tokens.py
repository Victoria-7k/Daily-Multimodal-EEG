#!/usr/bin/env python3
"""Build Wear temporal-token NPZ for EQL-CAF pipeline smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from daily_multimodal.temporal.global_repeat_tokens import build_global_repeat_temporal_tokens
from daily_multimodal.temporal.window_slicing import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-index", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="Existing Wear window-level token NPZ.")
    parser.add_argument("--emb-key", default="wear_emb")
    parser.add_argument("--mask-key")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    report = build_global_repeat_temporal_tokens(
        temporal_index_rows=read_jsonl(args.temporal_index, max_rows=args.max_rows),
        source_npz=args.source,
        output_npz=args.out,
        modality="wear",
        emb_key=args.emb_key,
        mask_key=args.mask_key,
    )
    _write_audit(report, args.audit_out)
    print(f"row_count={report['row_count']}")
    print(f"out={args.out}")
    return 0


def _write_audit(report: dict, path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
