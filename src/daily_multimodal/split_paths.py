"""Canonical protocol split path resolution."""

from __future__ import annotations

from pathlib import Path


LEGACY_SPLITS_ROOT = Path("/vePFS-0x0d/DailyEEG/splits_new")
DATE_IN_ORDER_SPLIT_ROOT = Path(
    "/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/outputs/splits/date_in_order"
)


def resolve_protocol_split_root(splits_root: Path | str, protocol: str) -> Path:
    """Resolve the chronological date-in-order split while preserving legacy protocols."""

    root = Path(splits_root)
    if protocol == "date_in_order" and root == LEGACY_SPLITS_ROOT:
        return DATE_IN_ORDER_SPLIT_ROOT
    return root / protocol
