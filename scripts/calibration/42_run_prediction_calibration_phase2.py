"""Prepare or run Phase 2 fatigue loss/sampler experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROUTES = (
    "cross_day:B0_Wphysio_full",
    "cross_day:B0_Wphysio_no_audio",
    "within_subject_day:B0_Wdeep_no_audio",
    "within_subject_day:A2_Wdeep_full",
)

PHASE2_CONFIGS = (
    {
        "name": "weighted_mse_label_bins_subject_balanced",
        "loss_modes": "weighted_mse_label_bins",
        "lambdas": "0",
        "sampler": "subject_balanced",
    },
    {
        "name": "huber_extreme_weight_subject_balanced",
        "loss_modes": "huber_extreme_weight",
        "lambdas": "0",
        "sampler": "subject_balanced",
    },
    {
        "name": "mse_variance_reg_subject_balanced",
        "loss_modes": "mse_variance_reg",
        "lambdas": "0.05,0.1,0.2",
        "sampler": "subject_balanced",
    },
    {
        "name": "raw_label_balanced",
        "loss_modes": "raw",
        "lambdas": "0",
        "sampler": "label_balanced",
    },
    {
        "name": "raw_label_subject_balanced",
        "loss_modes": "raw",
        "lambdas": "0",
        "sampler": "label_subject_balanced",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
    parser.add_argument("--splits-root", default="/vePFS-0x0d/DailyEEG/splits_new")
    parser.add_argument("--embeddings-root", default="/vePFS-0x0d/DailyEEG_multimodal/embeddings")
    parser.add_argument("--out-root", default="outputs/server_sync/fatigue_calibration_20260816/phase2_loss_sampler")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=240800)
    parser.add_argument("--eeg-token-seed", type=int, default=240800)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--execute", action="store_true", help="Run the generated commands instead of only writing the manifest.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip configs whose metrics JSON already exists when executing.")
    return parser.parse_args()


def command_for(args: argparse.Namespace, config: dict[str, str]) -> list[str]:
    out_root = Path(args.out_root)
    name = config["name"]
    return [
        args.python,
        "scripts/window_fatigue/32_run_eegpt_centered_loss.py",
        "--root",
        args.root,
        "--splits-root",
        args.splits_root,
        "--embeddings-root",
        args.embeddings_root,
        "--experiments",
        ",".join(ROUTES),
        "--eeg-branches",
        "eeg_eegpt_partial_ft_v1",
        "--eeg-token-root",
        "eeg_encoder_256d_tokens",
        "--eeg-token-seed",
        str(args.eeg_token_seed),
        "--loss-modes",
        config["loss_modes"],
        "--lambdas",
        config["lambdas"],
        "--no-raw-baseline",
        "--sampler",
        config["sampler"],
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--out-json",
        str(out_root / "reports" / f"{name}.json"),
        "--out-md",
        str(out_root / "reports" / f"{name}.md"),
        "--predictions-dir",
        str(out_root / "predictions" / name),
    ]


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root)
    config_dir = out_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for config in PHASE2_CONFIGS:
        command = command_for(args, config)
        record = {
            "name": config["name"],
            "loss_modes": config["loss_modes"],
            "lambdas": config["lambdas"],
            "sampler": config["sampler"],
            "routes": list(ROUTES),
            "seed": int(args.seed),
            "command": command,
            "status": "pending",
        }
        report_json = out_root / "reports" / f"{config['name']}.json"
        if args.execute and args.skip_existing and report_json.exists():
            record["status"] = "skipped_existing"
        elif args.execute:
            report_json.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(command, text=True)
            record["status"] = "completed" if result.returncode == 0 else "failed"
            record["returncode"] = int(result.returncode)
            if result.returncode != 0:
                manifest.append(record)
                write_manifest(config_dir, out_root, manifest)
                return int(result.returncode)
        manifest.append(record)
        write_json(config_dir / f"{config['name']}.json", record)
    write_manifest(config_dir, out_root, manifest)
    write_phase2_summary(out_root, manifest)
    print(json.dumps({"out_root": str(out_root), "config_count": len(manifest), "execute": bool(args.execute)}, indent=2))
    return 0


def write_manifest(config_dir: Path, out_root: Path, manifest: list[dict[str, Any]]) -> None:
    write_json(config_dir / "manifest.json", manifest)
    lines = [
        "# Phase 2 Loss/Sampler Manifest",
        "",
        "| config | loss_modes | lambdas | sampler | status | command |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in manifest:
        command = " ".join(quote_arg(part) for part in record["command"])
        lines.append(
            f"| {record['name']} | {record['loss_modes']} | {record['lambdas']} | {record['sampler']} | {record['status']} | `{command}` |"
        )
    (out_root / "paired_delta_summary.md").parent.mkdir(parents=True, exist_ok=True)
    (config_dir / "manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_phase2_summary(out_root: Path, manifest: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for record in manifest:
        report = out_root / "reports" / f"{record['name']}.json"
        if not report.exists():
            continue
        payload = json.loads(report.read_text(encoding="utf-8"))
        for row in payload.get("results", []):
            rows.append(
                {
                    "config": record["name"],
                    "protocol": row["protocol"],
                    "experiment": row["experiment"],
                    "loss_mode": row["loss_mode"],
                    "lambda": row["centered_lambda"],
                    "sampler": record["sampler"],
                    "val_rmse": row["val"]["rmse"],
                    "val_mae": row["val"]["mae"],
                    "val_raw_r": row["val"]["raw_r"],
                    "val_centered_r": row["val"]["within_subject_centered_r"],
                    "test_rmse": row["test"]["rmse"],
                    "test_raw_r": row["test"]["raw_r"],
                    "test_centered_r": row["test"]["within_subject_centered_r"],
                    "prediction_path": row.get("prediction_path", ""),
                }
            )
    lines = [
        "# Phase 2 Loss/Sampler Summary",
        "",
        "This summary is populated after running `scripts/calibration/42_run_prediction_calibration_phase2.py --execute` in an environment with the canonical EEG-aligned index, splits, embeddings, and CUDA runtime.",
        "",
    ]
    if rows:
        lines.extend(
            [
                "| config | protocol | experiment | loss | lambda | sampler | val RMSE | val raw r | val centered r |",
                "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['config']} | {row['protocol']} | {row['experiment']} | {row['loss_mode']} | {float(row['lambda']):.3g} | "
                f"{row['sampler']} | {float(row['val_rmse']):.4f} | {float(row['val_raw_r']):.4f} | {float(row['val_centered_r']):.4f} |"
            )
    else:
        lines.append("No completed Phase 2 report JSON files were found yet.")
    (out_root / "paired_delta_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out_root / "metrics_val.json", rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def quote_arg(value: str) -> str:
    if not value:
        return "''"
    if any(ch.isspace() for ch in value):
        return '"' + value.replace('"', '\\"') + '"'
    return value


if __name__ == "__main__":
    raise SystemExit(main())
