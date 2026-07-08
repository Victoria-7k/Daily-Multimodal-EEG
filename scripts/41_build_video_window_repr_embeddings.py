from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.training.video_window_repr import build_window_repr_embedding_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Wrap window-level adapter representations as a face_emb-compatible video bundle.")
    parser.add_argument("--representations", required=True)
    parser.add_argument("--variant", default="B1")
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-label", default="fatigue")
    parser.add_argument("--summary-out")
    args = parser.parse_args()

    result = build_window_repr_embedding_bundle(
        representations=args.representations,
        variant=args.variant,
        out=args.out,
        target_label=args.target_label,
    )
    if args.summary_out:
        summary = Path(args.summary_out)
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"row_count={result['row_count']}")
    print(f"event_count={result['event_count']}")
    print(f"subject_count={result['subject_count']}")
    print(f"input_dim={result['input_dim']}")
    print(f"output={result['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
