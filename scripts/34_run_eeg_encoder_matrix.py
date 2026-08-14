#!/usr/bin/env python3
"""Run EEG encoder comparisons on the EEG-aligned fatigue dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.training.eeg_encoder_matrix import (  # noqa: E402
    DEFAULT_ALIGNED_ROOT,
    DEFAULT_DATA_ROOT,
    DEFAULT_EEGPT_FROZEN_EMBEDDINGS,
    DEFAULT_INDEX_PATH,
    DEFAULT_PROTOCOLS,
    DEFAULT_SEEDS,
    DEFAULT_SPLITS_ROOT,
    MatrixRuntime,
    run_eeg_encoder_matrix,
    run_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--aligned-root", type=Path, default=DEFAULT_ALIGNED_ROOT)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument(
        "--protocols",
        default=",".join(DEFAULT_PROTOCOLS),
        help="Comma-separated existing split protocols; default uses all three: cross_subject,cross_day,within_subject_day.",
    )
    parser.add_argument(
        "--profiles",
        default="eegpt_frozen_v1,eeg_de_5band_1s_avg_v1,cbramod_frozen_v1,eegpt_partial_ft_v1,cbramod_partial_ft_v1",
    )
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--target-label", default="fatigue")
    parser.add_argument("--eegpt-frozen-embeddings", type=Path, default=DEFAULT_EEGPT_FROZEN_EMBEDDINGS)
    parser.add_argument("--eegpt-checkpoint", type=Path)
    parser.add_argument("--cbramod-checkpoint", type=Path)
    parser.add_argument(
        "--allow-cbramod-download",
        action="store_true",
        help="Allow CBraMod.from_pretrained to download braindecode/cbramod-pretrained. Without this, use a local checkpoint/cache.",
    )
    parser.add_argument("--max-rows", type=int, help="Smoke cap per protocol while preserving train/val/test rows.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fallback-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--partial-encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--full-encoder-learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--partial-last-n-blocks", type=int, default=2)
    parser.add_argument("--de-feature-cache", type=Path)
    parser.add_argument("--predictions-dir", type=Path)
    parser.add_argument("--embeddings-dir", type=Path, help="Write fusion-compatible full 256D EEG embeddings per protocol/profile/seed.")
    parser.add_argument("--out-json", type=Path, default=Path("outputs/reports/eeg_encoder_matrix_metrics.json"))
    parser.add_argument("--out-md", type=Path, default=Path("outputs/reports/eeg_encoder_matrix_table.md"))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--full-gate", action="store_true", help="Summarize whether partial fine-tune passes the full fine-tune gate.")
    args = parser.parse_args()

    protocols = _split_csv(args.protocols)
    profiles = _split_csv(args.profiles)
    seeds = tuple(int(value) for value in _split_csv(args.seeds))
    runtime = MatrixRuntime(
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        head_learning_rate=args.head_learning_rate,
        partial_encoder_learning_rate=args.partial_encoder_learning_rate,
        full_encoder_learning_rate=args.full_encoder_learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        grad_clip=args.grad_clip,
        fallback_batch_size=args.fallback_batch_size,
        device=args.device,
        torch_threads=args.torch_threads,
        amp=not args.disable_amp,
        partial_last_n_blocks=args.partial_last_n_blocks,
    )
    if args.preflight_only:
        result = run_preflight(
            data_root=args.data_root,
            splits_root=args.splits_root,
            index_path=args.index_path,
            protocols=protocols,
            target_label=args.target_label,
            eegpt_frozen_embeddings=args.eegpt_frozen_embeddings,
            cbramod_checkpoint=args.cbramod_checkpoint,
            out_json=args.out_json,
            out_md=args.out_md,
        )
    else:
        result = run_eeg_encoder_matrix(
            data_root=args.data_root,
            splits_root=args.splits_root,
            index_path=args.index_path,
            profiles=profiles,
            protocols=protocols,
            seeds=seeds,
            target_label=args.target_label,
            eegpt_frozen_embeddings=args.eegpt_frozen_embeddings,
            cbramod_checkpoint=args.cbramod_checkpoint,
            eegpt_checkpoint=args.eegpt_checkpoint,
            allow_cbramod_download=args.allow_cbramod_download,
            max_rows=args.max_rows,
            runtime=runtime,
            out_json=args.out_json,
            out_md=args.out_md,
            predictions_dir=args.predictions_dir,
            embeddings_dir=args.embeddings_dir,
            de_feature_cache=args.de_feature_cache,
            full_gate=args.full_gate,
        )
    print(f"task={result['task']}")
    print(f"protocols={','.join(protocols)}")
    print(f"out_json={args.out_json}")
    print(f"out_md={args.out_md}")
    if "run_count" in result:
        print(f"run_count={result['run_count']}")
    if "ok" in result:
        print(f"ok={result['ok']}")
    failed_runs = [
        row for row in result.get("results", [])
        if row.get("status") not in {None, "ok"}
    ]
    return 0 if result.get("ok", True) and not result.get("profile_errors") and not failed_runs else 1


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


if __name__ == "__main__":
    raise SystemExit(main())
