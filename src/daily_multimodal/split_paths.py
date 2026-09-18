"""Canonical protocol split path resolution."""

from __future__ import annotations

from pathlib import Path


LEGACY_SPLITS_ROOT = Path("/vePFS-0x0d/DailyEEG/splits_new")
CANONICAL_WITHIN_SUBJECT_DAY_ROOT = Path(
    "/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/outputs/splits/within_subject_day"
)


def resolve_protocol_split_root(splits_root: Path | str, protocol: str) -> Path:
    """Resolve the repaired within-subject-day split while preserving other roots."""

    root = Path(splits_root)
    if protocol == "within_subject_day" and root == LEGACY_SPLITS_ROOT:
        return CANONICAL_WITHIN_SUBJECT_DAY_ROOT
    return root / protocol
