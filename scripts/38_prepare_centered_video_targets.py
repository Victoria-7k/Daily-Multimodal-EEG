from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.training.video_loso_diagnostics import add_subject_centered_label


def main() -> int:
    parser = argparse.ArgumentParser(description="Add diagnostic subject-centered fatigue labels to a video embedding bundle.")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-label", default="fatigue")
    parser.add_argument("--centered-label", default="fatigue_subject_centered")
    parser.add_argument("--summary-out")
    args = parser.parse_args()

    result = add_subject_centered_label(
        embeddings=args.embeddings,
        out=args.out,
        target_label=args.target_label,
        centered_label=args.centered_label,
    )
    if args.summary_out:
        summary = Path(args.summary_out)
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"row_count={result['row_count']}")
    print(f"subject_count={len(result['subject_means'])}")
    print(f"output={result['output']}")
    print(result["warning"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
