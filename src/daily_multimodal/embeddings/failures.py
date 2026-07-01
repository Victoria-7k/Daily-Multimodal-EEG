from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


FAILURE_ERROR_TYPES = {
    "dependency_missing",
    "checkpoint_missing",
    "source_missing",
    "extraction_failed",
    "decode_failed",
    "shape_mismatch",
    "eeg_window_before_recording",
    "eeg_window_after_recording",
    "eeg_window_partial_overlap",
    "eeg_window_shape_mismatch",
    "nan_embedding",
    "quality_threshold_failed",
    "no_face_detected",
    "face_detection_failed",
    "oom",
    "timeout",
    "subject_split_incomplete",
    "unsupported_upgrade",
}


@dataclass
class EmbeddingFailure:
    sample_id: str
    event_id: str
    subject_id: str
    modality: str
    encoder_profile: str
    stage: str
    error_type: str
    error: str
    source_path: str
    recoverable: bool = True

    def __post_init__(self) -> None:
        if self.error_type not in FAILURE_ERROR_TYPES:
            raise ValueError(f"unsupported error_type: {self.error_type}")
        required = {
            "sample_id": self.sample_id,
            "event_id": self.event_id,
            "subject_id": self.subject_id,
            "modality": self.modality,
            "encoder_profile": self.encoder_profile,
            "stage": self.stage,
            "source_path": self.source_path,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"EmbeddingFailure missing required fields: {', '.join(missing)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_failure_list(
    failures: list[EmbeddingFailure | dict[str, Any]],
    path: Path | str,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = [_failure_to_dict(failure) for failure in failures]
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _failure_to_dict(failure: EmbeddingFailure | dict[str, Any]) -> dict[str, Any]:
    if isinstance(failure, EmbeddingFailure):
        return failure.to_dict()
    return EmbeddingFailure(**failure).to_dict()
