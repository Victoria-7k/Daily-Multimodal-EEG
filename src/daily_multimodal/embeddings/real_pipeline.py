from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from daily_multimodal.embeddings.contracts import EMBEDDING_DIM, REAL_MODALITY_ORDER, validate_embedding_shape
from daily_multimodal.embeddings.failures import EmbeddingFailure, write_failure_list


MODALITY_TO_EMB_KEY = {modality: f"{modality}_emb" for modality in REAL_MODALITY_ORDER}
MODALITY_TO_MASK_INDEX = {"eeg": 0, "wear": 1, "face": 2, "audio": 3}


@dataclass
class ModalityRecord:
    embedding: np.ndarray
    mask_value: int
    quality_flags: dict[str, Any]
    encoder_version: str


@dataclass
class ModalityBundle:
    modality: str
    source_path: Path
    records: dict[str, ModalityRecord]
    encoder_profiles: set[str]


def pack_real_embeddings(
    *,
    window_index: Path | str,
    eeg_embeddings: Path | str,
    wear_embeddings: Path | str,
    face_embeddings: Path | str,
    audio_embeddings: Path | str,
    output_npz: Path | str,
    report_out: Path | str,
    failures_out: Path | str,
    require_all_modalities: bool = False,
    max_windows: int | None = None,
) -> dict[str, Any]:
    windows = _read_jsonl(Path(window_index), max_rows=max_windows)
    bundles = {
        "eeg": _load_modality_bundle("eeg", Path(eeg_embeddings)),
        "wear": _load_modality_bundle("wear", Path(wear_embeddings)),
        "face": _load_modality_bundle("face", Path(face_embeddings)),
        "audio": _load_modality_bundle("audio", Path(audio_embeddings)),
    }

    rows: list[dict[str, Any]] = []
    failures: list[EmbeddingFailure] = []
    modality_stats = _initial_modality_stats(bundles)
    for window in windows:
        row, row_failures = _pack_window(window, bundles, modality_stats)
        failures.extend(row_failures)
        if require_all_modalities and not np.all(row["modality_mask"] == 1):
            continue
        rows.append(row)

    _write_all_real_npz(rows, output_npz)
    summary = {
        "stage": 17,
        "requested_windows": len(windows),
        "selected_windows": len(rows),
        "success_count": len(rows),
        "failure_count": len(failures),
        "require_all_modalities": bool(require_all_modalities),
        "modalities": modality_stats,
        "output_npz": str(output_npz),
        "report_out": str(report_out),
        "failures_out": str(failures_out),
    }
    _write_report(summary, rows, report_out)
    write_failure_list(failures, failures_out)
    return summary


def _read_jsonl(path: Path, *, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def _load_modality_bundle(modality: str, path: Path) -> ModalityBundle:
    records: dict[str, ModalityRecord] = {}
    encoder_profiles: set[str] = set()
    if not path.is_file():
        return ModalityBundle(modality=modality, source_path=path, records=records, encoder_profiles=encoder_profiles)

    emb_key = MODALITY_TO_EMB_KEY[modality]
    mask_index = MODALITY_TO_MASK_INDEX[modality]
    with np.load(path, allow_pickle=True) as loaded:
        sample_ids = loaded["sample_id"].astype(str).tolist()
        embeddings = validate_embedding_shape(emb_key, loaded[emb_key])
        masks = loaded["modality_mask"].astype(np.int8)
        quality_values = loaded["quality_flags"].tolist() if "quality_flags" in loaded.files else ["{}"] * len(sample_ids)
        encoder_values = (
            loaded["encoder_version"].astype(str).tolist()
            if "encoder_version" in loaded.files
            else [path.stem] * len(sample_ids)
        )
        if embeddings.shape[0] != len(sample_ids) or masks.shape[0] != len(sample_ids):
            raise ValueError(f"{path} has inconsistent row counts for {modality}")
        for idx, sample_id in enumerate(sample_ids):
            if sample_id in records:
                raise ValueError(f"{path} contains duplicate sample_id {sample_id!r}")
            encoder_version = str(encoder_values[idx])
            encoder_profiles.add(encoder_version)
            records[sample_id] = ModalityRecord(
                embedding=embeddings[idx].astype(np.float32, copy=False),
                mask_value=int(masks[idx, mask_index]),
                quality_flags=_parse_json_object(quality_values[idx]),
                encoder_version=encoder_version,
            )
    return ModalityBundle(modality=modality, source_path=path, records=records, encoder_profiles=encoder_profiles)


def _pack_window(
    window: dict[str, Any],
    bundles: dict[str, ModalityBundle],
    modality_stats: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[EmbeddingFailure]]:
    sample_id = str(window.get("sample_id", ""))
    failures: list[EmbeddingFailure] = []
    embeddings: dict[str, np.ndarray] = {}
    mask = np.zeros(4, dtype=np.int8)
    quality_flags: dict[str, Any] = {}
    encoder_versions: dict[str, str] = {}

    for modality in REAL_MODALITY_ORDER:
        bundle = bundles[modality]
        stats = modality_stats[modality]
        record = bundle.records.get(sample_id)
        if record is None:
            stats["missing_count"] += 1
            embeddings[modality] = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            quality_flags[modality] = {"missing": True}
            encoder_versions[modality] = ""
            failures.append(_missing_modality_failure(window, modality, bundle.source_path))
            continue

        stats["present_count"] += 1
        quality_flags[modality] = record.quality_flags
        encoder_versions[modality] = record.encoder_version
        if record.mask_value:
            stats["success_count"] += 1
            mask[MODALITY_TO_MASK_INDEX[modality]] = 1
            embeddings[modality] = record.embedding
        else:
            stats["masked_count"] += 1
            embeddings[modality] = np.zeros(EMBEDDING_DIM, dtype=np.float32)

    return (
        {
            "sample_id": sample_id,
            "event_id": str(window.get("event_id", "")),
            "subject_id": str(window.get("subject_id", "")),
            "session_id": str(window.get("session_id", "")),
            "labels": dict(window.get("label_columns") or {}),
            "source_paths": _source_paths_from_window(window),
            "eeg_emb": embeddings["eeg"],
            "wear_emb": embeddings["wear"],
            "face_emb": embeddings["face"],
            "audio_emb": embeddings["audio"],
            "modality_mask": mask,
            "quality_flags": quality_flags,
            "encoder_versions": encoder_versions,
        },
        failures,
    )


def _missing_modality_failure(
    window: dict[str, Any],
    modality: str,
    source_path: Path,
) -> EmbeddingFailure:
    return EmbeddingFailure(
        sample_id=str(window.get("sample_id", "")),
        event_id=str(window.get("event_id", "")),
        subject_id=str(window.get("subject_id", "")),
        modality=modality,
        encoder_profile="unknown",
        stage="pack_real_embeddings",
        error_type="source_missing",
        error=f"{modality} embedding row missing for sample_id",
        source_path=str(source_path),
        recoverable=True,
    )


def _source_paths_from_window(window: dict[str, Any]) -> dict[str, Any]:
    return {
        "eeg": window.get("eeg_bdf_path", ""),
        "wear": {
            "ppg": window.get("wear_ppg_path", ""),
            "gsr": window.get("wear_gsr_path", ""),
            "acc": window.get("wear_acc_path", ""),
        },
        "face": window.get("candidate_mp4_paths") or window.get("video_candidates") or [],
        "audio": window.get("candidate_audio_paths") or window.get("audio_candidates") or [],
    }


def _write_all_real_npz(rows: list[dict[str, Any]], output_npz: Path | str) -> Path:
    out = Path(output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        sample_id=np.array([row["sample_id"] for row in rows], dtype=object),
        event_id=np.array([row["event_id"] for row in rows], dtype=object),
        subject_id=np.array([row["subject_id"] for row in rows], dtype=object),
        session_id=np.array([row["session_id"] for row in rows], dtype=object),
        eeg_emb=_stack_rows(rows, "eeg_emb"),
        wear_emb=_stack_rows(rows, "wear_emb"),
        face_emb=_stack_rows(rows, "face_emb"),
        audio_emb=_stack_rows(rows, "audio_emb"),
        modality_mask=np.stack([row["modality_mask"] for row in rows]).astype(np.int8)
        if rows
        else np.zeros((0, 4), dtype=np.int8),
        labels=np.array([json.dumps(row["labels"], ensure_ascii=False) for row in rows], dtype=object),
        quality_flags=np.array(
            [json.dumps(row["quality_flags"], ensure_ascii=False) for row in rows],
            dtype=object,
        ),
        encoder_versions=np.array(
            [json.dumps(row["encoder_versions"], ensure_ascii=False) for row in rows],
            dtype=object,
        ),
        source_paths=np.array(
            [json.dumps(row["source_paths"], ensure_ascii=False) for row in rows],
            dtype=object,
        ),
    )
    return out


def _stack_rows(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    if not rows:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    return np.stack([row[key] for row in rows]).astype(np.float32)


def _write_report(summary: dict[str, Any], rows: list[dict[str, Any]], report_out: Path | str) -> Path:
    out = Path(report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "stage": 17,
        "summary": summary,
        "modalities": summary["modalities"],
        "samples": [
            {
                "sample_id": row["sample_id"],
                "event_id": row["event_id"],
                "subject_id": row["subject_id"],
                "modality_mask": row["modality_mask"].astype(int).tolist(),
                "encoder_versions": row["encoder_versions"],
                "quality_flags": row["quality_flags"],
            }
            for row in rows
        ],
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _initial_modality_stats(bundles: dict[str, ModalityBundle]) -> dict[str, dict[str, Any]]:
    return {
        modality: {
            "embedding_path": str(bundle.source_path),
            "embedded_count": len(bundle.records),
            "encoder_profiles": sorted(bundle.encoder_profiles),
            "present_count": 0,
            "missing_count": 0,
            "success_count": 0,
            "masked_count": 0,
        }
        for modality, bundle in bundles.items()
    }


def _parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if value is None or str(value) == "":
        return {}
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise ValueError("quality_flags entries must decode to JSON objects")
    return parsed
