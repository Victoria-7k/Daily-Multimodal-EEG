from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.embeddings.video_temporal import build_video_temporal_embeddings


def main() -> int:
    parser = argparse.ArgumentParser(description="Build V4b temporal video embeddings from DINOv2 frame sequences.")
    parser.add_argument("--frame-sequences", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--temporal-encoder", required=True, choices=["tcn", "temporal_transformer"])
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    summary = build_video_temporal_embeddings(
        frame_sequences=args.frame_sequences,
        out_path=args.out,
        temporal_encoder=args.temporal_encoder,
        seed=args.seed,
    )
    print(f"temporal_encoder={summary['temporal_encoder']}")
    print(f"encoder_version={summary['encoder_version']}")
    print(f"row_count={summary['row_count']}")
    print(f"face_mask_sum={summary['mask_sum']}")
    print(f"out_path={summary['out_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
