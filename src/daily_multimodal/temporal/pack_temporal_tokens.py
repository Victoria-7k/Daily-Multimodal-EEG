from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from daily_multimodal.temporal.token_contract import (
    MODALITY_ORDER,
    TOKEN_KEYS,
    load_npz_dict,
    validate_modality_temporal_npz,
    write_packed_temporal_tokens,
)


def pack_temporal_token_files(
    *,
    temporal_index_rows: list[dict[str, Any]],
    modality_paths: dict[str, Path],
    output_path: Path,
) -> dict[str, Any]:
    expected_sample_id = np.asarray([str(row["sample_id"]) for row in temporal_index_rows], dtype=str)
    modality_arrays: dict[str, dict[str, np.ndarray]] = {}
    modality_reports: list[dict[str, Any]] = []
    source_paths: dict[str, str] = {}
    for modality in MODALITY_ORDER:
        path = modality_paths.get(modality)
        if path is None:
            raise ValueError(f"missing path for modality {modality}")
        report = validate_modality_temporal_npz(path, modality=modality, expected_sample_id=expected_sample_id)
        modality_reports.append(report)
        source_paths[modality] = str(path)
        data = load_npz_dict(path)
        if TOKEN_KEYS[modality] not in data:
            raise ValueError(f"{path} missing {TOKEN_KEYS[modality]}")
        modality_arrays[modality] = data
    packed_report = write_packed_temporal_tokens(
        output_path=output_path,
        temporal_index_rows=temporal_index_rows,
        modality_arrays=modality_arrays,
        source_paths=source_paths,
    )
    return {
        "packed": packed_report,
        "modalities": modality_reports,
        "source_paths": source_paths,
    }


def write_pack_audit(report: dict[str, Any], path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
