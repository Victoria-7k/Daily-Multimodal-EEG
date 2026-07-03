from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.alignment.event_windows import load_window_index
from daily_multimodal.embeddings.wear_real import (
    extract_wear_real_embeddings,
    write_wear_quality_summary,
)


REQUIRED_FLAGS = ("has_eeg", "has_ppg", "has_gsr", "has_acc", "has_face", "has_audio")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract stage-16 real wearable sequence embeddings.")
    parser.add_argument("--window-index", required=True)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--require-all-modalities", action="store_true")
    parser.add_argument("--cache-root", default="outputs/cache")
    parser.add_argument("--encoder-profile", default="wear_sequence_v1")
    parser.add_argument(
        "--mask-low-quality-wear",
        action="store_true",
        help="Set wear mask to 0 for windows graded as low-quality (grade C).",
    )
    parser.add_argument("--out", default="outputs/embeddings/wear_real_embeddings.npz")
    parser.add_argument("--failures-out", default="outputs/reports/wear_real_failures.json")
    parser.add_argument("--summary-out", default="outputs/reports/wear_real_quality_summary.json")
    args = parser.parse_args()

    windows = load_window_index(args.window_index)
    if args.require_all_modalities:
        windows = [
            window for window in windows if all(bool(window.get(flag)) for flag in REQUIRED_FLAGS)
        ]
    selected = windows[: args.max_windows] if args.max_windows is not None else windows
    summary = extract_wear_real_embeddings(
        selected,
        cache_root=args.cache_root,
        output_npz=args.out,
        failures_out=args.failures_out,
        encoder_profile=args.encoder_profile,
        mask_low_quality_wear=args.mask_low_quality_wear,
    )
    write_wear_quality_summary(summary, args.summary_out)

    print(f"embedding_path={args.out}")
    print(f"failures_path={args.failures_out}")
    print(f"summary_path={args.summary_out}")
    print(f"embedded_count={summary['embedded_count']}")
    print(f"success_count={summary['success_count']}")
    print(f"failure_count={summary['failure_count']}")
    print(f"nan_count={summary['nan_count']}")
    return 0 if summary["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
