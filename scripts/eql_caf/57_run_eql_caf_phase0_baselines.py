#!/usr/bin/env python3
"""Prepare or execute the EQL-CAF Phase 0 fixed baseline command matrix."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_EMBEDDINGS_ROOT = Path("/vePFS-0x0d/DailyEEG_multimodal/embeddings")
DEFAULT_SPLITS_ROOT = DEFAULT_ROOT / "outputs/splits"
DEFAULT_PROTOCOLS = ("cross_day", "within_subject_day")
DEFAULT_SEEDS = (240800, 240801, 240802)
DEFAULT_EXPERIMENTS = ("B0_Wphysio_full", "B0_Wphysio_no_audio", "B0_Wdeep_full", "B0_Wdeep_no_audio")
DEFAULT_VARIANTS = ("attention", "concat", "eeg_anchor")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--embeddings-root", type=Path, default=DEFAULT_EMBEDDINGS_ROOT)
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument("--protocols", default=",".join(DEFAULT_PROTOCOLS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--experiments", default=",".join(DEFAULT_EXPERIMENTS))
    parser.add_argument("--fusion-variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--eeg-branch", default="eeg_eegpt_partial_ft_v1")
    parser.add_argument("--eeg-token-root", default="eeg_encoder_256d_tokens")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Run commands instead of only writing the manifest.")
    args = parser.parse_args()

    protocols = _split_csv(args.protocols)
    seeds = [int(value) for value in _split_csv(args.seeds)]
    experiments = _split_csv(args.experiments)
    variants = _split_csv(args.fusion_variants)
    commands = []
    for protocol in protocols:
        for experiment in experiments:
            for variant in variants:
                for seed in seeds:
                    run_dir = args.out_root / protocol / experiment / variant / f"seed_{seed}"
                    command = [
                        sys.executable,
                        "scripts/window_fatigue/32_run_eegpt_centered_loss.py",
                        "--root",
                        str(args.root),
                        "--embeddings-root",
                        str(args.embeddings_root),
                        "--splits-root",
                        str(args.splits_root),
                        "--experiments",
                        f"{protocol}:{experiment}",
                        "--eeg-branches",
                        args.eeg_branch,
                        "--eeg-token-root",
                        args.eeg_token_root,
                        "--eeg-token-seed",
                        str(seed),
                        "--seed",
                        str(seed),
                        "--experiment-seed-fixed",
                        "--loss-modes",
                        "raw",
                        "--fusion-variant",
                        variant,
                        "--device",
                        args.device,
                        "--out-json",
                        str(run_dir / "report.json"),
                        "--out-md",
                        str(run_dir / "report.md"),
                        "--predictions-dir",
                        str(run_dir / "predictions"),
                    ]
                    commands.append(
                        {
                            "protocol": protocol,
                            "experiment": experiment,
                            "fusion_variant": variant,
                            "seed": seed,
                            "command": command,
                            "shell_command": " ".join(shlex.quote(part) for part in command),
                        }
                    )
    manifest = {
        "phase": "EQL-CAF Phase 0 baseline reproduction",
        "row_count": len(commands),
        "execute": bool(args.execute),
        "commands": commands,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_root / "baseline_reproduction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.execute:
        for item in commands:
            Path(item["command"][-3]).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(item["command"], check=True)
    print(f"command_count={len(commands)}")
    print(f"manifest={manifest_path}")
    return 0


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


if __name__ == "__main__":
    raise SystemExit(main())
