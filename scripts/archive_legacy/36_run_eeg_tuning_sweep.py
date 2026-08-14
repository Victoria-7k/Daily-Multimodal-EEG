#!/usr/bin/env python3
"""Run the focused EEG hyperparameter sweep from the loss-history audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / "runtime/envs/eegpt-gpu-min/bin/python"


@dataclass(frozen=True)
class SweepConfig:
    name: str
    profile: str
    hidden_dim: int = 128
    learning_rate: float = 1e-3
    head_learning_rate: float = 1e-3
    partial_encoder_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    dropout: float = 0.1
    partial_last_n_blocks: int = 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="eeg_tuning_sweep_main_seed240800_20260813")
    parser.add_argument("--protocols", default="cross_day,within_subject_day")
    parser.add_argument("--seed", type=int, default=240800)
    parser.add_argument("--splits-root", default="/vePFS-0x0d/DailyEEG/splits_new")
    parser.add_argument("--eegpt-checkpoint", default="outputs/checkpoints/eegpt-pretrained")
    parser.add_argument("--cbramod-checkpoint", default="outputs/checkpoints/cbramod-pretrained")
    parser.add_argument("--de-feature-cache", default="outputs/cache/eeg_de_5band_1s_avg_v1_full.npz")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Run only the first N pending configs.")
    args = parser.parse_args()

    configs = build_configs()
    out_root = Path("outputs/reports") / args.run_id
    pred_root = Path("outputs/predictions") / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)
    pred_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.jsonl"
    summary_path = out_root / "summary.json"

    started = time.time()
    completed = 0
    skipped = 0
    failures = 0
    for config in configs:
        out_json = out_root / f"{config.name}.json"
        out_md = out_root / f"{config.name}.md"
        if _is_complete(out_json):
            skipped += 1
            continue
        if args.limit is not None and completed >= int(args.limit):
            break
        command = [
            str(PYTHON if PYTHON.exists() else sys.executable),
            "scripts/34_run_eeg_encoder_matrix.py",
            "--splits-root",
            args.splits_root,
            "--protocols",
            args.protocols,
            "--profiles",
            config.profile,
            "--seeds",
            str(args.seed),
            "--eegpt-checkpoint",
            args.eegpt_checkpoint,
            "--cbramod-checkpoint",
            args.cbramod_checkpoint,
            "--de-feature-cache",
            args.de_feature_cache,
            "--predictions-dir",
            str(pred_root / config.name),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
            "--epochs",
            str(args.epochs),
            "--hidden-dim",
            str(config.hidden_dim),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            _float_arg(config.learning_rate),
            "--head-learning-rate",
            _float_arg(config.head_learning_rate),
            "--partial-encoder-learning-rate",
            _float_arg(config.partial_encoder_learning_rate),
            "--weight-decay",
            _float_arg(config.weight_decay),
            "--dropout",
            _float_arg(config.dropout),
            "--patience",
            str(args.patience),
            "--torch-threads",
            str(args.torch_threads),
            "--device",
            args.device,
            "--partial-last-n-blocks",
            str(config.partial_last_n_blocks),
        ]
        record = {
            "name": config.name,
            "profile": config.profile,
            "status": "running",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "command": command,
            "out_json": str(out_json),
            "config": config.__dict__,
        }
        _append_jsonl(manifest_path, record)
        print(f"START {config.name}", flush=True)
        run_started = time.time()
        proc = subprocess.run(command, cwd=PROJECT_ROOT)
        duration = time.time() - run_started
        status = "ok" if proc.returncode == 0 and _is_complete(out_json) else "failed"
        if status == "ok":
            completed += 1
        else:
            failures += 1
        _append_jsonl(
            manifest_path,
            {
                "name": config.name,
                "profile": config.profile,
                "status": status,
                "returncode": int(proc.returncode),
                "duration_seconds": float(duration),
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "out_json": str(out_json),
            },
        )
        print(f"DONE {config.name} status={status} duration_min={duration / 60:.2f}", flush=True)
    summary = {
        "run_id": args.run_id,
        "config_count": len(configs),
        "completed_this_invocation": completed,
        "skipped_existing": skipped,
        "failures": failures,
        "duration_seconds": float(time.time() - started),
        "manifest": str(manifest_path),
        "out_root": str(out_root),
        "predictions_root": str(pred_root),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if failures == 0 else 1


def build_configs() -> list[SweepConfig]:
    configs: list[SweepConfig] = []
    for encoder_lr in (3e-6, 5e-6, 1e-5):
        for head_lr in (5e-4, 1e-3):
            for dropout in (0.1, 0.2):
                for wd in (1e-4, 5e-4):
                    configs.append(
                        SweepConfig(
                            name=f"eegpt_lr_enc{_tag(encoder_lr)}_head{_tag(head_lr)}_drop{_tag(dropout)}_wd{_tag(wd)}_blocks2",
                            profile="eegpt_partial_ft_v1",
                            head_learning_rate=head_lr,
                            partial_encoder_learning_rate=encoder_lr,
                            dropout=dropout,
                            weight_decay=wd,
                            partial_last_n_blocks=2,
                        )
                    )
    for blocks in (1, 4):
        configs.append(
            SweepConfig(
                name=f"eegpt_depth_blocks{blocks}_enc1e-05_head1e-03_drop0p1_wd1e-04",
                profile="eegpt_partial_ft_v1",
                partial_last_n_blocks=blocks,
            )
        )
    for encoder_lr in (1e-6, 3e-6, 5e-6):
        for dropout in (0.2, 0.3):
            for wd in (5e-4, 1e-3):
                configs.append(
                    SweepConfig(
                        name=f"cbramod_enc{_tag(encoder_lr)}_drop{_tag(dropout)}_wd{_tag(wd)}_blocks2",
                        profile="cbramod_partial_ft_v1",
                        partial_encoder_learning_rate=encoder_lr,
                        dropout=dropout,
                        weight_decay=wd,
                    )
                )
    for hidden_dim in (64, 128, 256, 512):
        for lr in (3e-4, 1e-3, 3e-3):
            for wd in (1e-5, 1e-4, 1e-3):
                configs.append(
                    SweepConfig(
                        name=f"de_hidden{hidden_dim}_lr{_tag(lr)}_wd{_tag(wd)}",
                        profile="eeg_de_5band_1s_avg_v1",
                        hidden_dim=hidden_dim,
                        learning_rate=lr,
                        weight_decay=wd,
                    )
                )
    return configs


def _is_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return int(payload.get("run_count", 0)) > 0 and not payload.get("profile_errors")


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _tag(value: float) -> str:
    text = f"{value:g}".replace(".", "p")
    return text


def _float_arg(value: float) -> str:
    return f"{value:.12g}"


if __name__ == "__main__":
    raise SystemExit(main())
