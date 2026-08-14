from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.training.baseline_mlp import run_baseline_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description="Train stage-9 lightweight concat MLP baselines.")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--split", default="subject", choices=["subject"])
    parser.add_argument("--target-label")
    parser.add_argument("--out-dir", default="outputs/models")
    parser.add_argument("--model-out")
    parser.add_argument("--metrics-out", default="outputs/reports/baseline_mlp_metrics.json")
    parser.add_argument("--table-out", default="outputs/reports/baseline_mlp_table.md")
    parser.add_argument("--overfit-limit", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    model_out = args.model_out or str(out_dir / "baseline_mlp.pt")
    result = run_baseline_experiment(
        embeddings_path=args.embeddings,
        model_out=model_out,
        metrics_out=args.metrics_out,
        table_out=args.table_out,
        target_label=args.target_label,
        overfit_limit=args.overfit_limit,
        epochs=args.epochs,
        seed=args.seed,
    )
    print(f"model_path={model_out}")
    print(f"metrics_path={args.metrics_out}")
    print(f"table_path={args.table_out}")
    print(f"target_label={result['target_label']}")
    print(f"overfit_passed={result['overfit_check']['passed']}")
    return 0 if result["overfit_check"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
