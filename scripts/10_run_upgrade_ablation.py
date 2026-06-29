from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.training.upgrade_ablation import run_upgrade_ablation, snapshot_baseline_reference


def main() -> int:
    parser = argparse.ArgumentParser(description="Run stage-10 upgrade ablation experiments.")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--baseline", required=True, help="Baseline metrics JSON.")
    parser.add_argument("--baseline-table", default="outputs/reports/baseline_mlp_table.md")
    parser.add_argument("--stage8-report", default="outputs/reports/all_complete_basic_embedding_report.json")
    parser.add_argument("--target-label", default="alert")
    parser.add_argument("--snapshot-baseline", action="store_true")
    parser.add_argument("--upgrade")
    parser.add_argument("--one-upgrade-at-a-time", action="store_true")
    parser.add_argument("--out-table", default="outputs/reports/model_upgrade_ablation_table.md")
    parser.add_argument("--failures-out", default="outputs/reports/model_upgrade_failures.json")
    parser.add_argument("--metrics-out", default="outputs/reports/modality_token_fusion_metrics.json")
    parser.add_argument("--model-out", default="outputs/models/modality_token_fusion.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--overfit-limit", type=int, default=128)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--reference-metrics-out", default="outputs/reports/baseline_reference_metrics.json")
    parser.add_argument("--reference-table-out", default="outputs/reports/baseline_reference_table.md")
    parser.add_argument("--reference-manifest-out", default="outputs/reports/baseline_reference_manifest.json")
    args = parser.parse_args()

    if args.snapshot_baseline:
        snapshot_baseline_reference(
            embeddings_path=args.embeddings,
            baseline_metrics_path=args.baseline,
            baseline_table_path=args.baseline_table,
            stage8_report_path=args.stage8_report,
            metrics_out=args.reference_metrics_out,
            table_out=args.reference_table_out,
            manifest_out=args.reference_manifest_out,
        )
        print("baseline reference snapshot saved")
        print(f"metrics_path={args.reference_metrics_out}")
        print(f"table_path={args.reference_table_out}")
        print(f"manifest_path={args.reference_manifest_out}")
        return 0

    if args.upgrade:
        result = run_upgrade_ablation(
            embeddings_path=args.embeddings,
            baseline_metrics_path=args.baseline,
            upgrade=args.upgrade,
            target_label=args.target_label,
            out_table=args.out_table,
            failures_out=args.failures_out,
            metrics_out=args.metrics_out,
            model_out=args.model_out,
            epochs=args.epochs,
            overfit_limit=args.overfit_limit,
            seed=args.seed,
        )
        print(f"experiment={result['experiment']}")
        print(f"decision={result['decision']}")
        print(f"reason={result['reason']}")
        print(f"table_path={args.out_table}")
        print(f"failures_path={args.failures_out}")
        return 0

    parser.error("Provide --snapshot-baseline or --upgrade.")


if __name__ == "__main__":
    raise SystemExit(main())
