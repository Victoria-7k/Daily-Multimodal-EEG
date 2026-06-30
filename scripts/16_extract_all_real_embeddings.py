from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.embeddings.real_pipeline import pack_real_embeddings


def main() -> int:
    parser = argparse.ArgumentParser(description="Pack real single-modality embeddings into one training-compatible NPZ.")
    parser.add_argument("--window-index", required=True, help="Window index JSONL used as the sample/order/label table.")
    parser.add_argument("--eeg", required=True, help="EEG real embedding NPZ.")
    parser.add_argument("--wear", required=True, help="Wear real embedding NPZ.")
    parser.add_argument("--face", required=True, help="Face real embedding NPZ.")
    parser.add_argument("--audio", required=True, help="Audio real embedding NPZ.")
    parser.add_argument("--out", default="outputs/embeddings/all_complete_real_embeddings.npz")
    parser.add_argument("--report-out", default="outputs/reports/all_complete_real_embedding_report.json")
    parser.add_argument("--failures-out", default="outputs/reports/all_complete_real_embedding_failures.json")
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--require-all-modalities", action="store_true")
    args = parser.parse_args()

    summary = pack_real_embeddings(
        window_index=args.window_index,
        eeg_embeddings=args.eeg,
        wear_embeddings=args.wear,
        face_embeddings=args.face,
        audio_embeddings=args.audio,
        output_npz=args.out,
        report_out=args.report_out,
        failures_out=args.failures_out,
        require_all_modalities=args.require_all_modalities,
        max_windows=args.max_windows,
    )
    print(f"embedding_path={args.out}")
    print(f"report_path={args.report_out}")
    print(f"failures_path={args.failures_out}")
    print(f"selected_windows={summary['selected_windows']}")
    print(f"failure_count={summary['failure_count']}")
    for modality in ("eeg", "wear", "face", "audio"):
        stats = summary["modalities"][modality]
        print(
            f"{modality}_success_count={stats['success_count']} "
            f"{modality}_missing_count={stats['missing_count']} "
            f"{modality}_masked_count={stats['masked_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
