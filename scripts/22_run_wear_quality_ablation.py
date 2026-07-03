from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.training.wear_quality_ablation import run_wear_quality_ablation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Wear-only quality ablation W1-W7 plus W5a-W5d.")
    parser.add_argument("--window-index", required=True)
    parser.add_argument("--physio-embeddings", required=True)
    parser.add_argument("--deep-embeddings", required=True)
    parser.add_argument("--target-label", default="fatigue")
    parser.add_argument("--out-json", default="outputs/reports/wear_quality_ablation_metrics.json")
    parser.add_argument("--out-table", default="outputs/reports/wear_quality_ablation_table.md")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    result = run_wear_quality_ablation(
        window_index=args.window_index,
        physio_embeddings=args.physio_embeddings,
        deep_embeddings=args.deep_embeddings,
        target_label=args.target_label,
        out_json=args.out_json,
        out_table=args.out_table,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(f"experiment_count={len(result['experiments'])}")
    for name, experiment in result["experiments"].items():
        print(
            "{name}: rows={rows} rmse_mean={rmse} pearson_r_mean={pearson} pred_std_mean={pred_std}".format(
                name=name,
                rows=experiment.get("row_count", 0),
                rmse=experiment.get("rmse_mean"),
                pearson=experiment.get("pearson_r_mean"),
                pred_std=experiment.get("pred_std_mean"),
            )
        )
    print(f"json_path={args.out_json}")
    print(f"table_path={args.out_table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
