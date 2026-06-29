from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daily_multimodal.alignment.event_windows import build_window_index, load_window_index
from daily_multimodal.embeddings.pipeline import extract_many_basic_embeddings, save_embedding_batch
from daily_multimodal.embeddings.subject import REQUIRED_MODALITY_FLAGS
from daily_multimodal.manifest.validate_manifest import load_jsonl_manifest


def run_full_basic_extraction(
    *,
    output_npz: Path | str,
    manifest_out: Path | str,
    report_out: Path | str,
    failures_out: Path | str,
    manifest: Path | str | None = None,
    window_index: Path | str | None = None,
    require_all_modalities: bool = True,
    max_windows: int | None = None,
) -> dict[str, Any]:
    windows = _load_or_build_windows(manifest=manifest, window_index=window_index)
    selected = _select_complete_windows(windows) if require_all_modalities else list(windows)
    selected_for_embedding = selected[:max_windows] if max_windows is not None else selected
    _write_jsonl(selected_for_embedding, manifest_out)

    batch = extract_many_basic_embeddings(selected_for_embedding)
    batch.summary = {
        "stage": 8,
        "all_windows": len(windows),
        "selected_windows": len(selected),
        "embedded_windows": len(selected_for_embedding),
        "require_all_modalities": require_all_modalities,
        **batch.summary,
        "modality_counts": {
            flag: sum(bool(window.get(flag)) for window in selected_for_embedding)
            for flag in REQUIRED_MODALITY_FLAGS
        },
    }
    _, report_path = save_embedding_batch(batch, output_npz, report_out)
    _append_report_fields(report_path, stage=8, require_all_modalities=require_all_modalities)
    _write_failures(batch.failures, failures_out)
    return batch.summary


def _load_or_build_windows(
    *,
    manifest: Path | str | None,
    window_index: Path | str | None,
) -> list[dict[str, Any]]:
    if window_index:
        return load_window_index(window_index)
    if manifest:
        return build_window_index(
            load_jsonl_manifest(manifest),
            start_seconds=-10,
            end_seconds=0,
            window_size_seconds=10,
            stride_seconds=5,
        )
    raise ValueError("Provide either manifest or window_index.")


def _select_complete_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        window
        for window in windows
        if all(bool(window.get(flag)) for flag in REQUIRED_MODALITY_FLAGS)
    ]


def _write_jsonl(rows: list[dict[str, Any]], output: Path | str) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def _append_report_fields(report_path: Path | str, *, stage: int, require_all_modalities: bool) -> None:
    path = Path(report_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    report["stage"] = stage
    report["require_all_modalities"] = require_all_modalities
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_failures(failures: list[dict[str, Any]], output: Path | str) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
