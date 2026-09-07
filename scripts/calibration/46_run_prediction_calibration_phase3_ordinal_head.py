"""Run the Phase 3 ordinal/distributional-head paired screen."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SEEDS = (240800, 240801, 240802)

ROUTES = (
    {"protocol": "cross_day", "experiment": "B0_Wphysio_full", "role": "primary"},
    {"protocol": "cross_day", "experiment": "B0_Wphysio_no_audio", "role": "primary"},
    {"protocol": "within_subject_day", "experiment": "B0_Wdeep_no_audio", "role": "primary"},
    {"protocol": "within_subject_day", "experiment": "A2_Wdeep_full", "role": "primary"},
)

HEAD_CANDIDATES = (
    {"head": "ordinal_cumulative", "head_lambda": 0.0, "candidate_id": "ordinal_cumulative_head"},
    {"head": "classification_expectation", "head_lambda": 0.0, "candidate_id": "classification_expectation_head"},
    {"head": "hybrid_regression_ordinal", "head_lambda": 0.1, "candidate_id": "hybrid_regression_ordinal_head_0p1"},
    {"head": "hybrid_regression_ordinal", "head_lambda": 0.3, "candidate_id": "hybrid_regression_ordinal_head_0p3"},
    {"head": "hybrid_regression_ordinal", "head_lambda": 1.0, "candidate_id": "hybrid_regression_ordinal_head_1p0"},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
    parser.add_argument("--splits-root", default="/vePFS-0x0d/DailyEEG/splits_new")
    parser.add_argument("--embeddings-root", default="/vePFS-0x0d/DailyEEG_multimodal/embeddings")
    parser.add_argument("--out-root", default="outputs/server_sync/fatigue_calibration_20260816/phase3_ordinal_head")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--eeg-token-seed", type=int, default=240800)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sampler", default="subject_balanced", choices=("default", "subject_balanced", "label_balanced", "label_subject_balanced"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args, seeds)
    write_manifest(out_root, manifest)
    if args.execute:
        for record in manifest:
            report_path = Path(record["report_json"])
            if args.skip_existing and report_path.exists():
                record["status"] = "skipped_existing"
                write_manifest(out_root, manifest)
                continue
            report_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"starting run_id={record['run_id']}", flush=True)
            result = subprocess.run(record["command"], text=True)
            record["status"] = "completed" if result.returncode == 0 else "failed"
            record["returncode"] = int(result.returncode)
            write_manifest(out_root, manifest)
            if result.returncode != 0:
                return int(result.returncode)
    print(json.dumps({"out_root": str(out_root), "run_count": len(manifest), "execute": bool(args.execute)}, indent=2))
    return 0


def build_manifest(args: argparse.Namespace, seeds: list[int]) -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    manifest: list[dict[str, Any]] = []
    for seed in seeds:
        for route in ROUTES:
            protocol = str(route["protocol"])
            experiment = str(route["experiment"])
            pair_id = f"{protocol}__{experiment}__seed_{seed}"
            manifest.append(
                make_record(
                    args,
                    run_id=f"raw_baseline__{protocol}__{experiment}__seed_{seed}",
                    pair_id=pair_id,
                    candidate_id="raw_baseline",
                    candidate_role="baseline",
                    protocol=protocol,
                    experiment=experiment,
                    head="regression",
                    head_lambda=0.0,
                    seed=seed,
                    out_root=out_root,
                )
            )
            for candidate in HEAD_CANDIDATES:
                manifest.append(
                    make_record(
                        args,
                        run_id=f"{candidate['candidate_id']}__{protocol}__{experiment}__seed_{seed}",
                        pair_id=pair_id,
                        candidate_id=str(candidate["candidate_id"]),
                        candidate_role=str(route["role"]),
                        protocol=protocol,
                        experiment=experiment,
                        head=str(candidate["head"]),
                        head_lambda=float(candidate["head_lambda"]),
                        seed=seed,
                        out_root=out_root,
                    )
                )
    return manifest


def make_record(
    args: argparse.Namespace,
    *,
    run_id: str,
    pair_id: str,
    candidate_id: str,
    candidate_role: str,
    protocol: str,
    experiment: str,
    head: str,
    head_lambda: float,
    seed: int,
    out_root: Path,
) -> dict[str, Any]:
    report_json = out_root / "reports" / f"{run_id}.json"
    report_md = out_root / "reports" / f"{run_id}.md"
    predictions_dir = out_root / "predictions" / run_id
    command = [
        args.python,
        "scripts/window_fatigue/32_run_eegpt_centered_loss.py",
        "--root",
        args.root,
        "--splits-root",
        args.splits_root,
        "--embeddings-root",
        args.embeddings_root,
        "--experiments",
        f"{protocol}:{experiment}",
        "--eeg-branches",
        "eeg_eegpt_partial_ft_v1",
        "--eeg-token-root",
        "eeg_encoder_256d_tokens",
        "--eeg-token-seed",
        str(args.eeg_token_seed),
        "--loss-modes",
        "raw",
        "--lambdas",
        "0",
        "--heads",
        head,
        "--head-lambdas",
        f"{head_lambda:g}",
        "--no-raw-baseline",
        "--sampler",
        args.sampler,
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--out-json",
        str(report_json),
        "--out-md",
        str(report_md),
        "--predictions-dir",
        str(predictions_dir),
    ]
    return {
        "run_id": run_id,
        "pair_id": pair_id,
        "candidate_id": candidate_id,
        "candidate_role": candidate_role,
        "protocol": protocol,
        "experiment": experiment,
        "eeg_branch": "eeg_eegpt_partial_ft_v1",
        "loss_mode": "raw",
        "lambda": 0.0,
        "head": head,
        "head_lambda": float(head_lambda),
        "sampler": args.sampler,
        "seed": int(seed),
        "report_json": str(report_json),
        "report_md": str(report_md),
        "predictions_dir": str(predictions_dir),
        "command": command,
        "status": "pending",
    }


def write_manifest(out_root: Path, manifest: list[dict[str, Any]]) -> None:
    config_dir = out_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Phase 3 Ordinal/Distributional Head Manifest",
        "",
        "| run_id | role | protocol | experiment | head | head lambda | sampler | seed | status |",
        "| --- | --- | --- | --- | --- | ---: | --- | ---: | --- |",
    ]
    for row in manifest:
        lines.append(
            f"| {row['run_id']} | {row['candidate_role']} | {row['protocol']} | {row['experiment']} | "
            f"{row['head']} | {float(row['head_lambda']):.3g} | {row['sampler']} | {row['seed']} | {row['status']} |"
        )
    (config_dir / "manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
