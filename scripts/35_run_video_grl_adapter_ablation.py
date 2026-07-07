from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.training.video_grl_adapter import run_video_grl_adapter_ablation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run video A0/A1 adapter + GRL ablations for V4d.")
    parser.add_argument("--eval-embeddings", required=True, help="Deterministic validation/test embedding bundle, usually A0.")
    parser.add_argument(
        "--train-embedding",
        action="append",
        default=[],
        help="Optional KEY=path train-fold-only embedding, e.g. A1=outputs/embeddings/video_v4d_A1_upper_body_train_embeddings.npz.",
    )
    parser.add_argument("--target-label", default="fatigue")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-table", required=True)
    parser.add_argument(
        "--representations-out",
        help="Optional .npz path for out-of-fold adapter representations and predictions.",
    )
    parser.add_argument("--variants", nargs="+", help="Optional variant names, e.g. B0 B1 B2_lam0.001.")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.001, 0.005, 0.01, 0.05])
    parser.add_argument("--b5-lambdas", type=float, nargs="+", default=[0.001, 0.005, 0.01])
    parser.add_argument(
        "--fold-strategy",
        default="leave_one_subject_out",
        choices=[
            "leave_one_subject_out",
            "grouped_k_fold",
            "within_subject_event_split",
            "within_subject_session_leave_out",
            "within_subject_chronological_split",
            "random_window_split",
        ],
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--adapter-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device")
    parser.add_argument("--compute-domain-probes", action="store_true")
    args = parser.parse_args()

    train_embeddings = _parse_train_embeddings(args.train_embedding)
    result = run_video_grl_adapter_ablation(
        eval_embeddings=args.eval_embeddings,
        train_embeddings=train_embeddings,
        target_label=args.target_label,
        out_json=args.out_json,
        out_table=args.out_table,
        representation_out=args.representations_out,
        variants=args.variants,
        lambdas=args.lambdas,
        b5_lambdas=args.b5_lambdas,
        fold_strategy=args.fold_strategy,
        n_splits=args.n_splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        adapter_dim=args.adapter_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        compute_domain_probes=args.compute_domain_probes,
    )
    print(f"experiment_count={len(result['experiments'])}")
    for name, experiment in result["experiments"].items():
        print(
            "{name}: rows={rows} lambda={lam} rmse_mean={rmse} pearson_r_mean={r}".format(
                name=name,
                rows=experiment.get("row_count", 0),
                lam=experiment.get("grl_lambda"),
                rmse=experiment.get("rmse_mean"),
                r=experiment.get("pearson_r_mean"),
            )
        )
    print(f"json_path={args.out_json}")
    print(f"table_path={args.out_table}")
    return 0


def _parse_train_embeddings(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--train-embedding must use KEY=path form: {value}")
        key, path = value.split("=", 1)
        if not key or not path:
            raise ValueError(f"--train-embedding must use KEY=path form: {value}")
        out[key] = path
    return out


if __name__ == "__main__":
    raise SystemExit(main())
