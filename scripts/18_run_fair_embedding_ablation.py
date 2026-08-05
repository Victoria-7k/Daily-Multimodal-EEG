from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.training.fair_embedding_ablation import run_fair_embedding_ablation, run_fusion_spec_fair_ablation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run aligned fair embedding leakage controls.")
    parser.add_argument("--basic-embeddings")
    parser.add_argument("--real-embeddings")
    parser.add_argument("--fusion-spec")
    parser.add_argument("--fusion-experiment")
    parser.add_argument("--target-label", default="alert")
    parser.add_argument("--out-json", default="outputs/reports/fair_embedding_ablation_metrics.json")
    parser.add_argument("--out-table", default="outputs/reports/fair_embedding_ablation_table.md")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--overfit-limit", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--modalities", default="eeg,wear,face,audio")
    parser.add_argument("--model", choices=["concat_mlp", "learnable_cross_attention"], default="concat_mlp")
    parser.add_argument("--min-available-modalities", type=int, default=2)
    parser.add_argument("--device")
    args = parser.parse_args()

    if args.fusion_spec:
        if not args.fusion_experiment:
            parser.error("--fusion-experiment is required with --fusion-spec")
        result = run_fusion_spec_fair_ablation(
            fusion_spec=args.fusion_spec,
            fusion_experiment=args.fusion_experiment,
            basic_embeddings=args.basic_embeddings,
            target_label=args.target_label,
            out_json=args.out_json,
            out_table=args.out_table,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            seed=args.seed,
            min_available_modalities=args.min_available_modalities,
            device=args.device,
        )
    else:
        if not args.basic_embeddings or not args.real_embeddings:
            parser.error("--basic-embeddings and --real-embeddings are required unless --fusion-spec is used")
        result = run_fair_embedding_ablation(
            basic_embeddings=args.basic_embeddings,
            real_embeddings=args.real_embeddings,
            target_label=args.target_label,
            out_json=args.out_json,
            out_table=args.out_table,
            epochs=args.epochs,
            overfit_limit=args.overfit_limit,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            seed=args.seed,
            modalities=tuple(item.strip() for item in args.modalities.split(",") if item.strip()),
            model=args.model,
            min_available_modalities=args.min_available_modalities,
            device=args.device,
        )
    print(f"row_count={result['row_count']}")
    print(f"sample_id_aligned={result['sample_id_aligned']}")
    print(f"experiment_count={len(result['experiments'])}")
    print(f"failure_count={result['failure_count']}")
    print(f"metrics_path={args.out_json}")
    print(f"table_path={args.out_table}")
    return 0 if result["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
