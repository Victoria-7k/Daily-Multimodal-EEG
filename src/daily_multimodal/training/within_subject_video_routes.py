from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from daily_multimodal.embeddings.contracts import EMBEDDING_DIM, validate_embedding_shape
from daily_multimodal.training.cross_attention_fusion import FusionDataset
from daily_multimodal.training.video_event_aggregation import _project_to_256


VIDEO_MASK_INDEX = 2


@dataclass(frozen=True)
class VideoRouteSpec:
    name: str
    source: str
    variant: str
    mode: str
    adapter_input: str | None = None
    fit_scope: str | None = None
    train_embedding: str | None = None
    eval_embedding: str | None = None


@dataclass(frozen=True)
class VideoRouteRegistry:
    base_embedding_path: Path
    train_augmentation_paths: dict[str, Path]
    routes: dict[str, VideoRouteSpec]


def load_video_route_registry(path: Path | str, *, root: Path | str | None = None) -> VideoRouteRegistry:
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    base = _resolve_path(raw["base_embedding_path"], root=root or config_path.parent.parent)
    augmentations = {
        str(key): _resolve_path(value, root=root or config_path.parent.parent)
        for key, value in raw.get("train_augmentation_paths", {}).items()
    }
    routes: dict[str, VideoRouteSpec] = {}
    for name, value in raw.get("routes", {}).items():
        routes[str(name)] = VideoRouteSpec(
            name=str(name),
            source=str(value["source"]),
            variant=str(value["variant"]),
            mode=str(value["mode"]),
            adapter_input=None if value.get("adapter_input") is None else str(value["adapter_input"]),
            fit_scope=None if value.get("fit_scope") is None else str(value["fit_scope"]),
            train_embedding=None if value.get("train_embedding") is None else str(value["train_embedding"]),
            eval_embedding=None if value.get("eval_embedding") is None else str(value["eval_embedding"]),
        )
    if not routes:
        raise ValueError(f"{config_path} has no video routes")
    if "FullSweepB0" not in routes:
        raise ValueError("video route registry must contain FullSweepB0")
    return VideoRouteRegistry(
        base_embedding_path=base,
        train_augmentation_paths=augmentations,
        routes=routes,
    )


def build_fold_video_tokens(
    dataset: FusionDataset,
    *,
    route_registry: VideoRouteRegistry,
    route_name: str,
    train_indices: Sequence[int] | np.ndarray,
    val_indices: Sequence[int] | np.ndarray,
    test_indices: Sequence[int] | np.ndarray,
    seed: int,
    epochs: int = 80,
    batch_size: int = 256,
    adapter_dim: int = 64,
    hidden_dim: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return fold-specific tokens and masks for one registered video route.

    Static B0 and A2 routes are deterministic. Adapter routes fit their video
    representation on the current training rows and transform every row with
    that fitted model before fusion training starts.
    """
    if "video" not in dataset.modalities:
        raise ValueError("video route requested for a dataset without video modality")
    if route_name not in route_registry.routes:
        raise ValueError(f"unknown video route: {route_name}")
    route = route_registry.routes[route_name]
    train = np.asarray(train_indices, dtype=np.int64)
    val = np.asarray(val_indices, dtype=np.int64)
    test = np.asarray(test_indices, dtype=np.int64)
    if len(train) == 0:
        raise ValueError("video route fitting requires non-empty train indices")
    tokens = np.asarray(dataset.tokens, dtype=np.float32).copy()
    masks = np.asarray(dataset.token_mask, dtype=bool).copy()
    video_position = dataset.modalities.index("video")
    base_emb, base_mask = _load_video_embedding(route_registry.base_embedding_path, dataset.sample_id)

    if route.mode == "fixed_base":
        tokens[:, video_position] = base_emb
        masks[:, video_position] = base_mask
        return tokens, masks, {"route": route_name, "fit_scope": "none"}

    if route.mode == "train_only_embedding_override":
        key = route.train_embedding or ""
        if key not in route_registry.train_augmentation_paths:
            raise ValueError(f"route {route_name} requires missing train embedding {key}")
        train_emb, train_mask = _load_video_embedding(
            route_registry.train_augmentation_paths[key], dataset.sample_id
        )
        tokens[:, video_position] = base_emb
        masks[:, video_position] = base_mask
        tokens[train, video_position] = train_emb[train]
        masks[train, video_position] = train_mask[train]
        return tokens, masks, {
            "route": route_name,
            "fit_scope": "train_only_embedding_override",
            "train_embedding": key,
            "train_sample_count": int(len(train)),
        }

    if route.mode != "fold_fitted_adapter":
        raise ValueError(f"unsupported video route mode: {route.mode}")
    if route.fit_scope != "train_only":
        raise ValueError(f"adapter route {route_name} must declare fit_scope=train_only")
    train_key = "base_embedding" if route.adapter_input == "base_embedding" else route.adapter_input
    if train_key == "base_embedding":
        adapter_train_emb = base_emb
    elif train_key in route_registry.train_augmentation_paths:
        adapter_train_emb, _ = _load_video_embedding(
            route_registry.train_augmentation_paths[train_key], dataset.sample_id
        )
    else:
        raise ValueError(f"route {route_name} requires missing adapter input {train_key}")

    from daily_multimodal.training.video_grl_adapter import (
        GrlVariantSpec,
        _class_lookup,
        _fit_torch_model,
        _predict_with_model,
        _targets_for,
    )

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - server runtime dependency
        raise ImportError("fold-fitted video routes require torch") from exc

    session_id = (
        np.asarray(dataset.session_id).astype(str)
        if dataset.session_id is not None
        else np.asarray(["session-unknown"] * len(dataset.sample_id), dtype=str)
    )
    subject_lookup = _class_lookup(dataset.subject_id.astype(str))
    session_lookup = _class_lookup(session_id)
    if route.variant == "B3_lam0.05":
        grl_spec = GrlVariantSpec(
            route.variant,
            use_adapter=True,
            use_session_grl=True,
            grl_lambda=0.05,
        )
    elif route.variant == "B5_A1_lam0.001":
        grl_spec = GrlVariantSpec(
            route.variant,
            use_adapter=True,
            use_subject_grl=True,
            use_session_grl=True,
            grl_lambda=0.001,
        )
    else:
        raise ValueError(f"unsupported fold-fitted adapter variant: {route.variant}")
    model = _fit_torch_model(
        train_x=adapter_train_emb[train],
        train_y=dataset.target[train],
        train_subject=_targets_for(dataset.subject_id[train], subject_lookup),
        train_session=_targets_for(session_id[train], session_lookup),
        eval_x=base_emb,
        spec=grl_spec,
        input_dim=int(base_emb.shape[1]),
        adapter_dim=int(adapter_dim),
        hidden_dim=int(hidden_dim),
        subject_count=max(1, len(subject_lookup)),
        session_count=max(1, len(session_lookup)),
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        seed=int(seed),
        device=device,
        torch=torch,
    )
    _, representation = _predict_with_model(model, base_emb, torch=torch, device=device)
    projected = np.vstack(
        [_project_to_256(row, salt=f"within_subject_{route_name}") for row in representation]
    ).astype(np.float32)
    tokens[:, video_position] = projected
    masks[:, video_position] = base_mask
    return tokens, masks, {
        "route": route_name,
        "fit_scope": "train_only",
        "adapter_input": train_key,
        "adapter_variant": route.variant,
        "adapter_train_sample_count": int(len(train)),
        "adapter_dim": int(adapter_dim),
    }


def _load_video_embedding(path: Path, sample_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as loaded:
        if "sample_id" not in loaded.files or "face_emb" not in loaded.files:
            raise ValueError(f"{path} must contain sample_id and face_emb")
        source_ids = loaded["sample_id"].astype(str)
        index = {value: idx for idx, value in enumerate(source_ids.tolist())}
        wanted = np.asarray(sample_ids).astype(str)
        missing = [value for value in wanted.tolist() if value not in index]
        if missing:
            raise ValueError(f"{path} missing sample_id values: {missing[:5]}")
        rows = np.asarray([index[value] for value in wanted], dtype=np.int64)
        embedding = validate_embedding_shape("face_emb", loaded["face_emb"][rows], expected_dim=EMBEDDING_DIM)
        if "modality_mask" in loaded.files:
            raw_mask = np.asarray(loaded["modality_mask"])[rows]
            mask = raw_mask[:, VIDEO_MASK_INDEX].astype(bool)
        else:
            mask = np.ones(len(rows), dtype=bool)
    return embedding.astype(np.float32), mask


def _resolve_path(value: str | Path, *, root: Path | str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(root) / path
