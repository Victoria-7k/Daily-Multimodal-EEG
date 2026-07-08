from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from daily_multimodal.training.video_event_aggregation import build_event_embedding_bundles


def test_build_event_embedding_bundles_writes_one_row_per_event(tmp_path: Path):
    repr_path = tmp_path / "repr.npz"
    _write_representations(repr_path, event_count=2, windows_per_event=12)

    result = build_event_embedding_bundles(
        representations=repr_path,
        variant="B1",
        out_dir=tmp_path / "events",
        min_windows=8,
    )

    assert set(result["outputs"]) == {"E1_mean", "E2_mean_std", "E3_mean_std_max"}
    assert result["event_count"] == 2
    assert result["dropped_event_count"] == 0

    for spec_name, output_path in result["outputs"].items():
        with np.load(output_path, allow_pickle=True) as loaded:
            assert loaded["sample_id"].astype(str).tolist() == ["event-00", "event-01"]
            assert loaded["event_id"].astype(str).tolist() == ["event-00", "event-01"]
            assert loaded["subject_id"].astype(str).tolist() == ["sub-01", "sub-02"]
            assert loaded["face_emb"].shape == (2, 256)
            assert loaded["modality_mask"].tolist() == [[0, 0, 1, 0], [0, 0, 1, 0]]
            assert json.loads(str(loaded["labels"][0])) == {"fatigue": 2.0}
            flags = json.loads(str(loaded["quality_flags"][0]))
            assert flags["source_variant"] == "B1"
            assert flags["aggregation"] == spec_name
            assert flags["window_count"] == 12


def test_build_event_embedding_bundles_drops_short_events(tmp_path: Path):
    repr_path = tmp_path / "repr.npz"
    _write_representations(repr_path, event_count=2, windows_per_event=[12, 7])

    result = build_event_embedding_bundles(
        representations=repr_path,
        variant="B1",
        out_dir=tmp_path / "events",
        min_windows=8,
    )

    assert result["event_count"] == 1
    assert result["dropped_event_count"] == 1
    assert result["dropped_events"][0]["event_id"] == "event-01"
    assert result["dropped_events"][0]["window_count"] == 7


def _write_representations(path: Path, *, event_count: int, windows_per_event: int | list[int]) -> None:
    sample_ids = []
    event_ids = []
    subject_ids = []
    session_ids = []
    targets = []
    rows = []
    for event_index in range(event_count):
        count = windows_per_event if isinstance(windows_per_event, int) else windows_per_event[event_index]
        event_id = f"event-{event_index:02d}"
        subject_id = f"sub-{event_index + 1:02d}"
        for window_index in range(count):
            sample_ids.append(f"{event_id}_win-{window_index:04d}")
            event_ids.append(event_id)
            subject_ids.append(subject_id)
            session_ids.append(f"{subject_id}_ses-01")
            targets.append(float(event_index + 2))
            rows.append(np.full(64, float(event_index + window_index / 100.0), dtype=np.float32))

    np.savez_compressed(
        path,
        sample_id=np.asarray(sample_ids, dtype=object),
        subject_id=np.asarray(subject_ids, dtype=object),
        event_id=np.asarray(event_ids, dtype=object),
        session_id=np.asarray(session_ids, dtype=object),
        target=np.asarray(targets, dtype=np.float32),
        repr__B1=np.stack(rows).astype(np.float32),
    )
