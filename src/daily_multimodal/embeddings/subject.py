from __future__ import annotations

from typing import Any


REQUIRED_MODALITY_FLAGS = ("has_eeg", "has_ppg", "has_gsr", "has_acc", "has_face", "has_audio")


def select_subject_windows(
    windows: list[dict[str, Any]],
    *,
    subject_id: str,
    require_all_modalities: bool = False,
) -> list[dict[str, Any]]:
    """Return windows for one subject, optionally requiring the complete stage-7 modality set."""
    selected = [window for window in windows if window.get("subject_id") == subject_id]
    if require_all_modalities:
        selected = [window for window in selected if _has_all_modalities(window)]
    return selected


def summarize_subject_selection(
    all_windows: list[dict[str, Any]],
    selected_windows: list[dict[str, Any]],
    *,
    subject_id: str,
) -> dict[str, Any]:
    subject_windows = [window for window in all_windows if window.get("subject_id") == subject_id]
    return {
        "subject_id": subject_id,
        "all_windows": len(all_windows),
        "subject_windows": len(subject_windows),
        "selected_windows": len(selected_windows),
        "modality_counts": {
            flag: sum(bool(window.get(flag)) for window in selected_windows)
            for flag in REQUIRED_MODALITY_FLAGS
        },
    }


def _has_all_modalities(window: dict[str, Any]) -> bool:
    return all(bool(window.get(flag)) for flag in REQUIRED_MODALITY_FLAGS)
