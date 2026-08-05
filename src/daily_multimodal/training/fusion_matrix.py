from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daily_multimodal.training.cross_attention_fusion import FusionBranchSpec, FusionExperimentSpec


@dataclass(frozen=True)
class FusionMatrixConfig:
    target_label: str
    eeg: FusionBranchSpec
    wear: dict[str, FusionBranchSpec]
    video: dict[str, FusionBranchSpec]
    audio: FusionBranchSpec
    metadata_source: FusionBranchSpec | None = None
    min_available_modalities: int = 2


def load_fusion_matrix_config(path: Path | str) -> FusionMatrixConfig:
    """Load the fusion matrix config.

    The default config file uses JSON syntax in a `.yaml` file so the repository
    does not need a hard dependency on PyYAML.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    branches = raw["branches"]
    return FusionMatrixConfig(
        target_label=str(raw.get("target_label", "fatigue")),
        eeg=_branch(branches["eeg"]),
        wear={name: _branch(value) for name, value in branches["wear"].items()},
        video={name: _branch(value) for name, value in branches["video"].items()},
        audio=_branch(branches["audio"]),
        metadata_source=_branch(raw["metadata_source"]) if raw.get("metadata_source") else None,
        min_available_modalities=int(raw.get("min_available_modalities", 2)),
    )


def matrix_experiment_specs(config: FusionMatrixConfig) -> list[FusionExperimentSpec]:
    specs: list[FusionExperimentSpec] = []
    for wear_name in config.wear:
        for video_name in config.video:
            specs.extend(
                [
                    FusionExperimentSpec(
                        name=f"fusion_{wear_name}_{video_name}_full",
                        enabled_modalities=("eeg", "wear", "video", "audio"),
                        target_label=config.target_label,
                        min_available_modalities=config.min_available_modalities,
                    ),
                    FusionExperimentSpec(
                        name=f"fusion_{wear_name}_{video_name}_no_audio",
                        enabled_modalities=("eeg", "wear", "video"),
                        target_label=config.target_label,
                        min_available_modalities=config.min_available_modalities,
                    ),
                ]
            )
        specs.extend(
            [
                FusionExperimentSpec(
                    name=f"fusion_{wear_name}_no_video",
                    enabled_modalities=("eeg", "wear", "audio"),
                    target_label=config.target_label,
                    min_available_modalities=config.min_available_modalities,
                ),
                FusionExperimentSpec(
                    name=f"fusion_{wear_name}_bio_only",
                    enabled_modalities=("eeg", "wear"),
                    target_label=config.target_label,
                    min_available_modalities=config.min_available_modalities,
                ),
            ]
        )
    return specs


def branches_for_experiment(config: FusionMatrixConfig, experiment_name: str) -> dict[str, FusionBranchSpec]:
    parts = experiment_name.split("_")
    if len(parts) < 3 or parts[0] != "fusion":
        raise ValueError(f"unsupported fusion experiment name: {experiment_name}")
    wear_name = parts[1]
    if wear_name not in config.wear:
        raise ValueError(f"unknown wear branch in experiment name: {wear_name}")
    branches = {"eeg": config.eeg, "wear": config.wear[wear_name]}
    if "no_video" not in experiment_name and "bio_only" not in experiment_name:
        video_name = parts[2]
        if video_name not in config.video:
            raise ValueError(f"unknown video branch in experiment name: {video_name}")
        branches["video"] = config.video[video_name]
    if "no_audio" not in experiment_name and "bio_only" not in experiment_name:
        branches["audio"] = config.audio
    return branches


def _branch(raw: dict[str, Any]) -> FusionBranchSpec:
    return FusionBranchSpec(
        path=raw["path"],
        modality=str(raw["modality"]),
        profile=str(raw["profile"]),
    )
