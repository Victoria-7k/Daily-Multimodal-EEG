from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.embeddings.video_variants import build_video_variant_embeddings


def main() -> int:
    parser = argparse.ArgumentParser(description="Build V1/V2 face_emb video variant bundles.")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--window-index", required=True)
    parser.add_argument("--behavior-flags")
    parser.add_argument("--openface-embeddings")
    parser.add_argument("--sample-mode", default="behavior_retained", choices=["strict_aligned", "behavior_retained"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    summary = build_video_variant_embeddings(
        variant=args.variant,
        window_index_path=args.window_index,
        behavior_flags_path=args.behavior_flags,
        openface_embeddings_path=args.openface_embeddings,
        sample_mode=args.sample_mode,
        out_path=args.out,
    )
    print(f"variant={summary['variant']}")
    print(f"sample_mode={summary['sample_mode']}")
    print(f"row_count={summary['row_count']}")
    print(f"face_mask_sum={summary['mask_sum']}")
    print(f"out_path={summary['out_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

