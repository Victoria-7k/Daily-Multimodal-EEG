from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


EMBEDDING_DIM = 256
FACE_MASK_INDEX = 2


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _loads_json_array(values: np.ndarray | None, index: int) -> dict[str, Any]:
    if values is None:
        return {}
    raw = values[index]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not raw:
        return {}
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return {"raw": str(raw)}


def _source_video_file(window: dict[str, Any]) -> str:
    candidates = window.get("video_candidates") or []
    if candidates:
        return str(candidates[0].get("mp4_path", ""))
    paths = window.get("candidate_mp4_paths") or []
    if paths:
        return str(paths[0])
    return ""


def _clip_seconds(window: dict[str, Any], key: str) -> float:
    candidates = window.get("video_candidates") or []
    if candidates and candidates[0].get(key) is not None:
        return float(candidates[0][key])
    return float("nan")


def _load_sources(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    by_sample: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            sample_ids = [str(v) for v in data["sample_id"]]
            embeddings = np.asarray(data["face_emb"], dtype=np.float32)
            if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
                raise ValueError(f"{path} face_emb must be (N,{EMBEDDING_DIM}), got {embeddings.shape}")
            modality_mask = np.asarray(data["modality_mask"], dtype=np.int8) if "modality_mask" in data else None
            quality_flags = data["quality_flags"] if "quality_flags" in data else None
            encoder_versions = data["encoder_version"] if "encoder_version" in data else None
            for idx, sample_id in enumerate(sample_ids):
                if sample_id in by_sample:
                    duplicates.append(sample_id)
                    continue
                mask = 1
                if modality_mask is not None:
                    mask = int(modality_mask[idx, FACE_MASK_INDEX])
                by_sample[sample_id] = {
                    "embedding": embeddings[idx],
                    "mask": mask,
                    "quality_flags": _loads_json_array(quality_flags, idx),
                    "encoder_version": str(encoder_versions[idx]) if encoder_versions is not None else "",
                    "source_npz": str(path),
                }
    return by_sample, duplicates


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge DINOv2 video chunks into the EEG-aligned 28819-row layout.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--encoder-version", required=True)
    args = parser.parse_args()

    index_path = Path(args.index)
    output = Path(args.out)
    report_path = Path(args.report)
    source_paths = [Path(p) for p in args.sources]

    windows = _read_jsonl(index_path)
    by_sample, duplicates = _load_sources(source_paths)

    n_rows = len(windows)
    video_emb = np.zeros((n_rows, EMBEDDING_DIM), dtype=np.float32)
    video_mask = np.zeros(n_rows, dtype=np.int8)
    modality_mask = np.zeros((n_rows, 4), dtype=np.int8)
    quality_flags: list[str] = []
    source_npz: list[str] = []
    source_video_file: list[str] = []
    clip_start_seconds = np.full(n_rows, np.nan, dtype=np.float32)
    clip_end_seconds = np.full(n_rows, np.nan, dtype=np.float32)

    for row_index, window in enumerate(windows):
        sample_id = str(window.get("sample_id", ""))
        source_video_file.append(_source_video_file(window))
        clip_start_seconds[row_index] = _clip_seconds(window, "clip_start_seconds")
        clip_end_seconds[row_index] = _clip_seconds(window, "clip_end_seconds")
        record = by_sample.get(sample_id)
        if record is None:
            quality_flags.append(json.dumps({"missing_video_embedding": True}, ensure_ascii=False, allow_nan=False))
            source_npz.append("")
            continue
        video_emb[row_index] = np.asarray(record["embedding"], dtype=np.float32)
        video_mask[row_index] = int(record["mask"])
        modality_mask[row_index, FACE_MASK_INDEX] = int(record["mask"])
        flags = dict(record["quality_flags"])
        flags["source_npz"] = record["source_npz"]
        quality_flags.append(json.dumps(_json_ready(flags), ensure_ascii=False, allow_nan=False))
        source_npz.append(str(record["source_npz"]))

    labels = [
        json.dumps(_json_ready(window.get("label_columns") or window.get("labels") or {}), ensure_ascii=False, allow_nan=False)
        for window in windows
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        sample_id=np.array([str(w.get("sample_id", "")) for w in windows], dtype=object),
        eeg_sample_index=np.array([int(w.get("eeg_sample_index", i)) for i, w in enumerate(windows)], dtype=np.int64),
        subject_id=np.array([str(w.get("subject_id", "")) for w in windows], dtype=object),
        session_id=np.array([str(w.get("session_id", "")) for w in windows], dtype=object),
        day_id=np.array([int(w.get("eeg_day_id", -1)) for w in windows], dtype=np.int64),
        event_id=np.array([str(w.get("event_id", "")) for w in windows], dtype=object),
        eeg_aligned_event_id=np.array([str(w.get("eeg_aligned_event_id", "")) for w in windows], dtype=object),
        event_window_id=np.array([int(w.get("window_id", -1)) for w in windows], dtype=np.int64),
        window_start_seconds=np.array([float(w.get("event_window_start_seconds", np.nan)) for w in windows], dtype=np.float32),
        window_end_seconds=np.array([float(w.get("event_window_end_seconds", np.nan)) for w in windows], dtype=np.float32),
        window_start_time=np.array([str(w.get("window_start_time", "")) for w in windows], dtype=object),
        window_end_time=np.array([str(w.get("window_end_time", "")) for w in windows], dtype=object),
        labels=np.array(labels, dtype=object),
        video_emb=video_emb,
        face_emb=video_emb,
        video_mask=video_mask,
        modality_mask=modality_mask,
        source_video_file=np.array(source_video_file, dtype=object),
        clip_start_seconds=clip_start_seconds,
        clip_end_seconds=clip_end_seconds,
        quality_flags=np.array(quality_flags, dtype=object),
        source_npz=np.array(source_npz, dtype=object),
        encoder_version=np.array([args.encoder_version] * n_rows, dtype=object),
    )

    report = {
        "index": str(index_path),
        "out": str(output),
        "row_count": n_rows,
        "source_files": [str(p) for p in source_paths],
        "source_sample_count": len(by_sample),
        "duplicate_sample_ids": duplicates[:100],
        "duplicate_sample_count": len(duplicates),
        "video_mask_sum": int(video_mask.sum()),
        "missing_video_embedding_count": int(n_rows - int(video_mask.sum())),
        "encoder_version": args.encoder_version,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
