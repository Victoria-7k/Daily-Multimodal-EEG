from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.training.real_embedding_ablation import run_real_embedding_ablation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run stage-18 real embedding ablation experiments.")
    parser.add_argument("--basic-embeddings", required=True)
    parser.add_argument("--real-embeddings", required=True)
    parser.add_argument("--baseline", required=True, help="Baseline reference metrics JSON.")
    parser.add_argument("--stage10-metrics", help="Stage-10 modality token metrics JSON.")
    parser.add_argument("--target-label", default="alert")
    parser.add_argument("--out-table", default="outputs/reports/real_embedding_ablation_table.md")
    parser.add_argument("--metrics-out", default="outputs/reports/real_embedding_ablation_metrics.json")
    parser.add_argument("--failures-out", default="outputs/reports/real_embedding_ablation_failures.json")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--overfit-limit", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--seeds",
        default="17,29,41,53,67",
        help="Comma-separated seeds for face repeated runs.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    result = run_real_embedding_ablation(
        basic_embeddings=args.basic_embeddings,
        real_embeddings=args.real_embeddings,
        baseline_metrics=args.baseline,
        stage10_metrics=args.stage10_metrics,
        target_label=args.target_label,
        out_table=args.out_table,
        metrics_out=args.metrics_out,
        failures_out=args.failures_out,
        epochs=args.epochs,
        overfit_limit=args.overfit_limit,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        seeds=seeds,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    print(f"table_path={args.out_table}")
    print(f"metrics_path={args.metrics_out}")
    print(f"failures_path={args.failures_out}")
    print(f"experiment_count={len(result['experiments'])}")
    print(f"failure_count={len(result['failures'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
