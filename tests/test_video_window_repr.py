from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from daily_multimodal.training.video_window_repr import build_window_repr_embedding_bundle


def test_build_window_repr_embedding_bundle_preserves_window_rows(tmp_path: Path):
    repr_path = tmp_path / "repr.npz"
    _write_repr(repr_path)

    result = build_window_repr_embedding_bundle(
        representations=repr_path,
        variant="B1",
        out=tmp_path / "b1_windows.npz",
    )

    assert result["row_count"] == 3
    assert result["input_dim"] == 64
    with np.load(result["output"], allow_pickle=True) as loaded:
        assert loaded["sample_id"].astype(str).tolist() == ["s0", "s1", "s2"]
        assert loaded["event_id"].astype(str).tolist() == ["e0", "e0", "e1"]
        assert loaded["subject_id"].astype(str).tolist() == ["sub-01", "sub-01", "sub-02"]
        assert loaded["face_emb"].shape == (3, 256)
        assert loaded["modality_mask"].tolist() == [[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]
        assert json.loads(str(loaded["labels"][0])) == {"fatigue": 2.0}
        flags = json.loads(str(loaded["quality_flags"][0]))
        assert flags["source_variant"] == "B1"
        assert flags["representation_source"] == "window_level_adapter_repr"


def _write_repr(path: Path) -> None:
    np.savez_compressed(
        path,
        sample_id=np.asarray(["s0", "s1", "s2"], dtype=object),
        subject_id=np.asarray(["sub-01", "sub-01", "sub-02"], dtype=object),
        event_id=np.asarray(["e0", "e0", "e1"], dtype=object),
        session_id=np.asarray(["sub-01_ses-01", "sub-01_ses-01", "sub-02_ses-01"], dtype=object),
        target=np.asarray([2.0, 2.0, 5.0], dtype=np.float32),
        repr__B1=np.ones((3, 64), dtype=np.float32),
    )
