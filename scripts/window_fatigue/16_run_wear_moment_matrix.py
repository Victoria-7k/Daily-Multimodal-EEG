#!/usr/bin/env python3
"""Train MOMENT-based wear 256D tokens and write fusion-compatible per-protocol/profile/seed npz."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.training.wear_moment_matrix import (  # noqa: E402
    DEFAULT_INDEX_PATH,
    DEFAULT_MOMENT_CHECKPOINT,
    DEFAULT_MOMENT_EMBEDDING_CACHE,
    DEFAULT_PROTOCOLS,
    DEFAULT_SEEDS,
    DEFAULT_SPLITS_ROOT,
    DEFAULT_WEAR_META_NPZ,
    DEFAULT_WEAR_RAW_ROOT,
    WearMomentRuntime,
    run_preflight,
    run_wear_moment_matrix,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument("--wear-meta-npz", type=Path, default=DEFAULT_WEAR_META_NPZ)
    parser.add_argument("--wear-raw-root", type=Path, default=DEFAULT_WEAR_RAW_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MOMENT_CHECKPOINT)
    parser.add_argument("--profiles", default="wear_moment_frozen_v1,wear_moment_partial_ft_v1")
    parser.add_argument("--protocols", default=",".join(DEFAULT_PROTOCOLS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--target-label", default="fatigue")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--partial-last-n-blocks", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--embeddings-dir", type=Path, help="Write wear_tokens/{protocol}/{profile}/seed_{seed}.npz")
    parser.add_argument("--matrix-cache", type=Path, help="Optional cached (N,5,320) wear matrix npz.")
    parser.add_argument("--moment-embedding-cache", type=Path, default=DEFAULT_MOMENT_EMBEDDING_CACHE)
    parser.add_argument("--max-rows", type=int, help="Smoke cap on index rows.")
    parser.add_argument("--out-json", type=Path, default=Path("outputs/reports/wear_moment_matrix_metrics.json"))
    parser.add_argument("--out-md", type=Path, default=Path("outputs/reports/wear_moment_matrix_table.md"))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    protocols = tuple(item.strip() for item in args.protocols.split(",") if item.strip())
    profiles = tuple(item.strip() for item in args.profiles.split(",") if item.strip())
    seeds = tuple(int(item) for item in args.seeds.split(",") if item.strip())
    runtime = WearMomentRuntime(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        encoder_learning_rate=args.encoder_learning_rate,
        partial_last_n_blocks=args.partial_last_n_blocks,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        grad_clip=args.grad_clip,
        device=args.device,
        torch_threads=args.torch_threads,
        amp=not args.disable_amp,
    )
    if args.preflight_only:
        result = run_preflight(
            index_path=args.index_path,
            wear_meta_npz=args.wear_meta_npz,
            wear_raw_root=args.wear_raw_root,
            checkpoint=args.checkpoint,
            matrix_cache=args.matrix_cache,
            max_rows=args.max_rows,
            out_json=args.out_json,
            out_md=args.out_md,
        )
        print(f"preflight_ok={result['ok']}")
        print(f"row_count={result['row_count']} mask_sum={result['mask_sum']} coverage={result['coverage']:.4f}")
        print(f"moment={result['moment']}")
    else:
        result = run_wear_moment_matrix(
            index_path=args.index_path,
            splits_root=args.splits_root,
            wear_meta_npz=args.wear_meta_npz,
            wear_raw_root=args.wear_raw_root,
            checkpoint=args.checkpoint,
            profiles=profiles,
            protocols=protocols,
            seeds=seeds,
            runtime=runtime,
            embeddings_dir=args.embeddings_dir,
            matrix_cache=args.matrix_cache,
            moment_embedding_cache=args.moment_embedding_cache,
            max_rows=args.max_rows,
            out_json=args.out_json,
            out_md=args.out_md,
            target_label=args.target_label,
        )
        print(f"run_count={result['run_count']}")
        print(f"elapsed_seconds={result['elapsed_seconds']:.1f}")
        print(f"out_json={args.out_json}")
        print(f"out_md={args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
