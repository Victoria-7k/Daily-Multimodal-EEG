from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.manifest.validate_manifest import (
    load_jsonl_manifest,
    summarize_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate manifest summary counts.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--summary-out")
    args = parser.parse_args()

    rows = load_jsonl_manifest(args.manifest)
    summary = summarize_manifest(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_out).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

