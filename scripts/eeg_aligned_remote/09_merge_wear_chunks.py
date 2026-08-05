#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


WEAR_MASK_INDEX = 1
EMBEDDING_DIM = 256
LABEL_ORDER = [
    "inspired",
    "alert",
    "determined",
    "attentive",
    "active",
    "hostile",
    "nervous",
    "upset",
    "afraid",
    "ashamed",
    "fatigue",
]


def load_index(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _label_vector(row: dict[str, Any]) -> list[float]:
    labels = row.get("label_columns") or row.get("labels") or {}
    out: list[float] = []
    for key in LABEL_ORDER:
        value = labels.get(key, 0)
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def _source_wear_file(row: dict[str, Any]) -> str:
    paths = {
        "ppg": row.get("wear_ppg_path", ""),
        "gsr": row.get("wear_gsr_path", ""),
        "acc": row.get("wear_acc_path", ""),
    }
    return json.dumps(paths, ensure_ascii=False, allow_nan=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge wear chunks into the EEG-aligned 28819-row layout.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--encoder-version", required=True)
    args = parser.parse_args()

    rows = load_index(Path(args.index))
    n_rows = len(rows)
    emb = np.zeros((n_rows, EMBEDDING_DIM), dtype=np.float32)
    mask = np.zeros(n_rows, dtype=np.int8)
    quality = np.array(["{}"] * n_rows, dtype=object)
    source = np.array([_source_wear_file(row) for row in rows], dtype=object)
    seen = 0
    duplicate_sample_ids: list[str] = []
    seen_samples: set[str] = set()

    for src in args.sources:
        path = Path(src)
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as data:
            sample_ids = [str(value) for value in data["sample_id"]]
            wear_emb = np.asarray(data["wear_emb"], dtype=np.float32)
            if wear_emb.ndim != 2 or wear_emb.shape[1] != EMBEDDING_DIM:
                raise ValueError(f"{path} wear_emb must be (N,{EMBEDDING_DIM}), got {wear_emb.shape}")
            modality_mask = np.asarray(data["modality_mask"], dtype=np.int8) if "modality_mask" in data else None
            quality_flags = data["quality_flags"] if "quality_flags" in data else None
            for j, sample_id in enumerate(sample_ids):
                if not sample_id.startswith("eeg_"):
                    continue
                if sample_id in seen_samples:
                    duplicate_sample_ids.append(sample_id)
                    continue
                seen_samples.add(sample_id)
                row_index = int(sample_id.split("_", 1)[1])
                emb[row_index] = wear_emb[j]
                if modality_mask is not None:
                    mask[row_index] = int(modality_mask[j, WEAR_MASK_INDEX])
                else:
                    mask[row_index] = 1
                if quality_flags is not None:
                    quality[row_index] = str(quality_flags[j])
                seen += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        sample_id=np.array([str(row.get("sample_id", "")) for row in rows], dtype=object),
        eeg_sample_index=np.array([int(row.get("eeg_sample_index", i)) for i, row in enumerate(rows)], dtype=np.int64),
        subject_id=np.array([str(row.get("subject_id", "")) for row in rows], dtype=object),
        session_id=np.array([str(row.get("session_id", "")) for row in rows], dtype=object),
        day_id=np.array([int(row.get("eeg_day_id", row.get("day_id", -1))) for row in rows], dtype=np.int64),
        event_id=np.array([str(row.get("event_id", "")) for row in rows], dtype=object),
        eeg_aligned_event_id=np.array([str(row.get("eeg_aligned_event_id", "")) for row in rows], dtype=object),
        event_window_id=np.array([int(row.get("window_id", row.get("event_window_id", -1))) for row in rows], dtype=np.int64),
        window_start_seconds=np.array([float(row.get("event_window_start_seconds", row.get("window_start_offset_seconds", 0))) for row in rows], dtype=np.float32),
        window_end_seconds=np.array([float(row.get("event_window_end_seconds", row.get("window_end_offset_seconds", 0))) for row in rows], dtype=np.float32),
        window_start_time=np.array([str(row.get("window_start_time", "")) for row in rows], dtype=object),
        window_end_time=np.array([str(row.get("window_end_time", "")) for row in rows], dtype=object),
        labels=np.array([_label_vector(row) for row in rows], dtype=np.float32),
        wear_emb=emb,
        wear_mask=mask,
        source_wear_file=source,
        quality_flags=quality,
        encoder_version=np.array([args.encoder_version] * n_rows, dtype=object),
    )

    report = {
        "rows": n_rows,
        "success_count": int(mask.sum()),
        "missing_count": int(n_rows - int(mask.sum())),
        "raw_seen": seen,
        "duplicate_sample_count": len(duplicate_sample_ids),
        "duplicate_sample_ids": duplicate_sample_ids[:100],
        "out": str(out_path),
        "encoder_version": args.encoder_version,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
