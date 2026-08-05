import json
from pathlib import Path

import numpy as np

from daily_multimodal.training.cross_attention_fusion import FusionDataset
from daily_multimodal.training.within_subject_video_routes import (
    build_fold_video_tokens,
    load_video_route_registry,
)


def test_fixed_base_route_replaces_static_video_tokens(tmp_path):
    base = tmp_path / "base.npz"
    _write_video(base, offset=4.0)
    registry_path = _write_registry(tmp_path, base=base)
    registry = load_video_route_registry(registry_path, root=tmp_path)
    dataset = _dataset()

    tokens, masks, metadata = build_fold_video_tokens(
        dataset,
        route_registry=registry,
        route_name="FullSweepB0",
        train_indices=[0, 1],
        val_indices=[2],
        test_indices=[3],
        seed=7,
    )

    np.testing.assert_allclose(tokens[:, 2, 0], np.asarray([4, 5, 6, 7], dtype=np.float32))
    assert masks[:, 2].all()
    assert metadata["fit_scope"] == "none"


def test_a2_route_overrides_train_rows_only(tmp_path):
    base = tmp_path / "base.npz"
    a2 = tmp_path / "a2.npz"
    _write_video(base, offset=1.0)
    _write_video(a2, offset=10.0)
    registry_path = _write_registry(tmp_path, base=base, a2=a2)
    registry = load_video_route_registry(registry_path, root=tmp_path)
    dataset = _dataset()

    tokens, _masks, metadata = build_fold_video_tokens(
        dataset,
        route_registry=registry,
        route_name="A1A2TrainOnlyA2",
        train_indices=[0, 2],
        val_indices=[1],
        test_indices=[3],
        seed=7,
    )

    np.testing.assert_allclose(tokens[:, 2, 0], np.asarray([10, 2, 12, 4], dtype=np.float32))
    assert metadata["train_embedding"] == "A2"
    assert metadata["train_sample_count"] == 2


def _dataset() -> FusionDataset:
    return FusionDataset(
        name="route_test",
        modalities=("eeg", "wear", "video", "audio"),
        sample_id=np.asarray([f"sample-{idx}" for idx in range(4)], dtype=str),
        event_id=np.asarray([f"event-{idx}" for idx in range(4)], dtype=str),
        subject_id=np.asarray(["sub-01"] * 4, dtype=str),
        target=np.asarray([0, 1, 2, 3], dtype=np.float32),
        tokens=np.zeros((4, 4, 256), dtype=np.float32),
        token_mask=np.ones((4, 4), dtype=bool),
        branch_profiles={"eeg": "eeg", "wear": "wear", "video": "video", "audio": "audio"},
        target_label="fatigue",
    )


def _write_video(path: Path, *, offset: float) -> None:
    count = 4
    emb = np.zeros((count, 256), dtype=np.float32)
    emb[:, 0] = np.arange(count, dtype=np.float32) + offset
    mask = np.zeros((count, 4), dtype=np.int8)
    mask[:, 2] = 1
    np.savez_compressed(
        path,
        sample_id=np.asarray([f"sample-{idx}" for idx in range(count)], dtype=object),
        face_emb=emb,
        modality_mask=mask,
    )


def _write_registry(tmp_path: Path, *, base: Path, a2: Path | None = None) -> Path:
    payload = {
        "base_embedding_path": str(base),
        "train_augmentation_paths": {} if a2 is None else {"A2": str(a2)},
        "routes": {
            "FullSweepB0": {"source": "full_sweep", "variant": "B0", "mode": "fixed_base"},
            "A1A2TrainOnlyA2": {
                "source": "a1_a2_train_only",
                "variant": "A2",
                "mode": "train_only_embedding_override",
                "train_embedding": "A2",
                "eval_embedding": "base_embedding",
            },
        },
    }
    path = tmp_path / "routes.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
