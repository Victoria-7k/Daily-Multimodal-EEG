from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from daily_multimodal.embeddings.basic import EmbeddingSample, extract_basic_embedding


@dataclass
class EmbeddingBatch:
    samples: list[EmbeddingSample]
    failures: list[dict[str, Any]]
    summary: dict[str, Any]


def extract_many_basic_embeddings(
    windows: list[dict[str, Any]],
    *,
    max_windows: int | None = None,
) -> EmbeddingBatch:
    selected = windows[:max_windows] if max_windows is not None else windows
    samples: list[EmbeddingSample] = []
    failures: list[dict[str, Any]] = []
    for window in selected:
        try:
            samples.append(extract_basic_embedding(window))
        except Exception as exc:  # pragma: no cover - defensive batch logging
            failures.append(
                {
                    "sample_id": window.get("sample_id", ""),
                    "event_id": window.get("event_id", ""),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return EmbeddingBatch(
        samples=samples,
        failures=failures,
        summary={
            "requested_windows": len(selected),
            "success_count": len(samples),
            "failure_count": len(failures),
        },
    )


def save_embedding_batch(
    batch: EmbeddingBatch,
    output_npz: Path | str,
    report_out: Path | str,
) -> tuple[Path, Path]:
    npz_path = Path(output_npz)
    report_path = Path(report_out)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        sample_id=np.array([sample.sample_id for sample in batch.samples], dtype=object),
        event_id=np.array([sample.event_id for sample in batch.samples], dtype=object),
        subject_id=np.array([sample.subject_id for sample in batch.samples], dtype=object),
        session_id=np.array([sample.session_id for sample in batch.samples], dtype=object),
        eeg_emb=_stack_embeddings(batch.samples, "eeg_emb"),
        wear_emb=_stack_embeddings(batch.samples, "wear_emb"),
        face_emb=_stack_embeddings(batch.samples, "face_emb"),
        audio_emb=_stack_embeddings(batch.samples, "audio_emb"),
        modality_mask=np.stack([sample.modality_mask for sample in batch.samples]).astype(np.int8)
        if batch.samples
        else np.zeros((0, 4), dtype=np.int8),
        labels=np.array([json.dumps(sample.labels, ensure_ascii=False) for sample in batch.samples], dtype=object),
        source_paths=np.array(
            [json.dumps(sample.source_paths, ensure_ascii=False) for sample in batch.samples],
            dtype=object,
        ),
    )
    report = {
        "summary": batch.summary,
        "samples": [_sample_report(sample) for sample in batch.samples],
        "failures": batch.failures,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return npz_path, report_path


def _stack_embeddings(samples: list[EmbeddingSample], attr: str) -> np.ndarray:
    if not samples:
        return np.zeros((0, 256), dtype=np.float32)
    return np.stack([getattr(sample, attr) for sample in samples]).astype(np.float32)


def _sample_report(sample: EmbeddingSample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "event_id": sample.event_id,
        "modality_mask": sample.modality_mask.astype(int).tolist(),
        "encoder_versions": sample.encoder_versions,
        "quality_flags": sample.quality_flags,
    }
