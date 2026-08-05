from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from daily_multimodal.embeddings.contracts import EMBEDDING_DIM, validate_embedding_shape


TOKEN_ORDER = ("eeg", "wear", "video", "audio")
MODALITY_TO_EMB_KEY = {
    "eeg": "eeg_emb",
    "wear": "wear_emb",
    "video": "face_emb",
    "audio": "audio_emb",
}
MODALITY_TO_MASK_INDEX = {"eeg": 0, "wear": 1, "video": 2, "audio": 3}


@dataclass(frozen=True)
class FusionBranchSpec:
    path: Path | str
    modality: str
    profile: str


@dataclass(frozen=True)
class FusionExperimentSpec:
    name: str
    enabled_modalities: tuple[str, ...]
    target_label: str
    min_available_modalities: int = 2


@dataclass(frozen=True)
class FusionDataset:
    name: str
    modalities: tuple[str, ...]
    sample_id: np.ndarray
    event_id: np.ndarray
    subject_id: np.ndarray
    target: np.ndarray
    tokens: np.ndarray
    token_mask: np.ndarray
    branch_profiles: dict[str, str]
    target_label: str
    session_id: np.ndarray | None = None


@dataclass(frozen=True)
class LearnableAttentionConfig:
    token_dim: int = 128
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.1
    patience: int = 25
    seed: int = 17
    device: str | None = None


@dataclass(frozen=True)
class TokenNormalization:
    x_mean: np.ndarray
    x_std: np.ndarray
    fit_count: int
    fit_index_sha256: str
    statistics_sha256: str


@dataclass(frozen=True)
class TargetNormalization:
    y_mean: float
    y_std: float
    fit_count: int
    fit_index_sha256: str
    statistics_sha256: str


@dataclass
class LearnableAttentionModel:
    module: Any
    config: LearnableAttentionConfig
    modalities: tuple[str, ...]
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float
    best_val_loss: float
    epoch_count: int
    initial_train_loss: float
    final_train_loss: float
    normalization_fit_index_sha256: str
    feature_statistics_sha256: str
    target_statistics_sha256: str


def build_fusion_dataset(
    *,
    branches: Mapping[str, FusionBranchSpec],
    experiment: FusionExperimentSpec,
    base_sample_ids: Sequence[str] | np.ndarray | None = None,
    metadata_source: FusionBranchSpec | None = None,
) -> FusionDataset:
    """Build aligned modality-token tensors from fixed branch embedding bundles."""
    modalities = _normalize_modalities(experiment.enabled_modalities)
    loaded = {
        modality: _load_branch(branches[modality])
        for modality in modalities
    }
    loaded_metadata_source = _load_branch(metadata_source) if metadata_source is not None else None
    sample_ids = _aligned_sample_ids(loaded, base_sample_ids=base_sample_ids)
    rows_by_modality = {
        modality: _indices_for_sample_ids(data, sample_ids, source=str(branches[modality].path))
        for modality, data in loaded.items()
    }
    metadata = _metadata_with_labels(
        loaded,
        modalities,
        sample_ids=sample_ids,
        metadata_source=loaded_metadata_source,
        metadata_source_path=None if metadata_source is None else str(metadata_source.path),
    )
    metadata_indices = _indices_for_sample_ids(
        metadata,
        sample_ids,
        source=metadata["source"],
    )
    tokens = []
    masks = []
    for modality in modalities:
        data = loaded[modality]
        indices = rows_by_modality[modality]
        tokens.append(data["embedding"][indices])
        masks.append(data["mask"][indices])
    token_tensor = np.stack(tokens, axis=1).astype(np.float32)
    token_mask = np.stack(masks, axis=1).astype(bool)
    available = token_mask.sum(axis=1)
    keep = available >= int(experiment.min_available_modalities)
    if not np.any(keep):
        raise ValueError(
            f"{experiment.name} has no rows with at least "
            f"{experiment.min_available_modalities} available modalities"
        )
    labels = metadata.get("labels")
    target_values = _target_values(labels[metadata_indices], experiment.target_label)
    return FusionDataset(
        name=experiment.name,
        modalities=modalities,
        sample_id=sample_ids[keep].astype(str),
        event_id=metadata["event_id"][metadata_indices][keep].astype(str),
        subject_id=metadata["subject_id"][metadata_indices][keep].astype(str),
        target=target_values[keep].astype(np.float32),
        tokens=token_tensor[keep].astype(np.float32),
        token_mask=token_mask[keep],
        branch_profiles={modality: branches[modality].profile for modality in modalities},
        target_label=experiment.target_label,
        session_id=metadata["session_id"][metadata_indices][keep].astype(str),
    )


def require_torch_for_learnable_cross_attention() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("learnable_cross_attention requires torch") from exc
    return torch


def fit_learnable_cross_attention(
    dataset: FusionDataset,
    *,
    train_indices: Sequence[int] | np.ndarray,
    val_indices: Sequence[int] | np.ndarray,
    config: LearnableAttentionConfig | None = None,
    torch_module: Any | None = None,
) -> LearnableAttentionModel:
    torch = torch_module or require_torch_for_learnable_cross_attention()
    cfg = config or LearnableAttentionConfig()
    train = np.asarray(train_indices, dtype=np.int64)
    val = np.asarray(val_indices, dtype=np.int64)
    if len(train) == 0:
        raise ValueError("learnable_cross_attention requires a non-empty training split")
    if len(val) == 0:
        val = train
    torch.manual_seed(int(cfg.seed))
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    token_norm = fit_token_normalization(dataset.tokens, dataset.token_mask, train)
    target_norm = fit_target_normalization(dataset.target, train)
    x_mean = token_norm.x_mean
    x_std = token_norm.x_std
    y_mean = target_norm.y_mean
    y_std = target_norm.y_std
    module = _build_attention_module(
        torch=torch,
        modality_count=len(dataset.modalities),
        token_dim=int(cfg.token_dim),
        dropout=float(cfg.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )
    loss_fn = torch.nn.MSELoss()
    x_train = torch.as_tensor(_normalize_tokens(dataset.tokens[train], x_mean, x_std), dtype=torch.float32, device=device)
    mask_train = torch.as_tensor(dataset.token_mask[train], dtype=torch.bool, device=device)
    y_train = torch.as_tensor(((dataset.target[train] - y_mean) / y_std).reshape(-1, 1), dtype=torch.float32, device=device)
    x_val = torch.as_tensor(_normalize_tokens(dataset.tokens[val], x_mean, x_std), dtype=torch.float32, device=device)
    mask_val = torch.as_tensor(dataset.token_mask[val], dtype=torch.bool, device=device)
    y_val = torch.as_tensor(((dataset.target[val] - y_mean) / y_std).reshape(-1, 1), dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cfg.seed))
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    batch_size = max(1, min(int(cfg.batch_size), len(train)))
    module.eval()
    with torch.no_grad():
        initial_train_loss = float(loss_fn(module(x_train, mask_train)["prediction"], y_train).detach().cpu().item())
    for epoch in range(max(1, int(cfg.epochs))):
        module.train()
        order = torch.randperm(len(train), generator=generator).to(device)
        for start in range(0, len(train), batch_size):
            batch = order[start : start + batch_size]
            outputs = module(x_train[batch], mask_train[batch])
            loss = loss_fn(outputs["prediction"], y_train[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        module.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(module(x_val, mask_val)["prediction"], y_val).detach().cpu().item())
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
        else:
            stale += 1
            if stale >= int(cfg.patience):
                break
    if best_state is not None:
        module.load_state_dict(best_state)
    module.eval()
    with torch.no_grad():
        final_train_loss = float(loss_fn(module(x_train, mask_train)["prediction"], y_train).detach().cpu().item())
    return LearnableAttentionModel(
        module=module,
        config=cfg,
        modalities=dataset.modalities,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        best_val_loss=best_val,
        epoch_count=best_epoch,
        initial_train_loss=initial_train_loss,
        final_train_loss=final_train_loss,
        normalization_fit_index_sha256=token_norm.fit_index_sha256,
        feature_statistics_sha256=token_norm.statistics_sha256,
        target_statistics_sha256=target_norm.statistics_sha256,
    )


def predict_with_learnable_cross_attention(
    model: LearnableAttentionModel,
    dataset: FusionDataset,
    *,
    indices: Sequence[int] | np.ndarray,
    torch_module: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    torch = torch_module or require_torch_for_learnable_cross_attention()
    idx = np.asarray(indices, dtype=np.int64)
    device = next(model.module.parameters()).device
    x = torch.as_tensor(_normalize_tokens(dataset.tokens[idx], model.x_mean, model.x_std), dtype=torch.float32, device=device)
    mask = torch.as_tensor(dataset.token_mask[idx], dtype=torch.bool, device=device)
    model.module.eval()
    with torch.no_grad():
        outputs = model.module(x, mask)
        pred = outputs["prediction"].detach().cpu().numpy().reshape(-1).astype(np.float32)
        attention = outputs["attention"].detach().cpu().numpy().astype(np.float32)
    return pred * float(model.y_std) + float(model.y_mean), attention


def save_learnable_cross_attention_model(
    model: LearnableAttentionModel,
    path: Path | str,
    *,
    torch_module: Any | None = None,
) -> None:
    torch = torch_module or require_torch_for_learnable_cross_attention()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.module.state_dict(),
            "config": {
                "token_dim": model.config.token_dim,
                "epochs": model.config.epochs,
                "batch_size": model.config.batch_size,
                "learning_rate": model.config.learning_rate,
                "weight_decay": model.config.weight_decay,
                "dropout": model.config.dropout,
                "patience": model.config.patience,
                "seed": model.config.seed,
                "device": model.config.device,
            },
            "modalities": list(model.modalities),
            "x_mean": model.x_mean,
            "x_std": model.x_std,
            "y_mean": model.y_mean,
            "y_std": model.y_std,
            "best_val_loss": model.best_val_loss,
            "epoch_count": model.epoch_count,
            "initial_train_loss": model.initial_train_loss,
            "final_train_loss": model.final_train_loss,
            "normalization_fit_index_sha256": model.normalization_fit_index_sha256,
            "feature_statistics_sha256": model.feature_statistics_sha256,
            "target_statistics_sha256": model.target_statistics_sha256,
        },
        out,
    )


def fit_token_normalization(
    tokens: np.ndarray,
    mask: np.ndarray,
    indices: Sequence[int] | np.ndarray,
) -> TokenNormalization:
    idx = np.asarray(indices, dtype=np.int64)
    if len(idx) == 0:
        raise ValueError("token normalization requires at least one training row")
    x_mean, x_std = _token_normalization(tokens[idx], mask[idx])
    return TokenNormalization(
        x_mean=x_mean,
        x_std=x_std,
        fit_count=int(len(idx)),
        fit_index_sha256=_sha256_array(idx.astype(np.int64)),
        statistics_sha256=_sha256_arrays(x_mean, x_std),
    )


def fit_target_normalization(
    target: np.ndarray,
    indices: Sequence[int] | np.ndarray,
) -> TargetNormalization:
    idx = np.asarray(indices, dtype=np.int64)
    if len(idx) == 0:
        raise ValueError("target normalization requires at least one training row")
    values = np.asarray(target, dtype=np.float32)[idx]
    y_mean = float(values.mean())
    y_std = float(values.std()) or 1.0
    stats = np.asarray([y_mean, y_std], dtype=np.float32)
    return TargetNormalization(
        y_mean=y_mean,
        y_std=y_std,
        fit_count=int(len(idx)),
        fit_index_sha256=_sha256_array(idx.astype(np.int64)),
        statistics_sha256=_sha256_array(stats),
    )


def audit_model_normalization(
    model: LearnableAttentionModel,
    dataset: FusionDataset,
    train_indices: Sequence[int] | np.ndarray,
) -> dict:
    train = np.asarray(train_indices, dtype=np.int64)
    token_norm = fit_token_normalization(dataset.tokens, dataset.token_mask, train)
    target_norm = fit_target_normalization(dataset.target, train)
    np.testing.assert_allclose(model.x_mean, token_norm.x_mean)
    np.testing.assert_allclose(model.x_std, token_norm.x_std)
    np.testing.assert_allclose(
        np.asarray([model.y_mean, model.y_std], dtype=np.float32),
        np.asarray([target_norm.y_mean, target_norm.y_std], dtype=np.float32),
    )
    if model.feature_statistics_sha256 != token_norm.statistics_sha256:
        raise AssertionError("feature normalization statistics hash mismatch")
    if model.target_statistics_sha256 != target_norm.statistics_sha256:
        raise AssertionError("target normalization statistics hash mismatch")
    return {
        "fit_scope": "train_only",
        "fit_sample_count": int(len(train)),
        "fit_sample_id_sha256": _sha256_lines(dataset.sample_id[train].astype(str).tolist()),
        "feature_statistics_sha256": token_norm.statistics_sha256,
        "target_fit_sample_count": int(len(train)),
        "target_statistics_sha256": target_norm.statistics_sha256,
        "verified": True,
    }


def _token_normalization(tokens: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    expanded_mask = np.asarray(mask, dtype=bool)[:, :, None]
    available = np.where(expanded_mask, tokens, np.nan)
    mean = np.nanmean(available, axis=(0, 1), keepdims=True)
    std = np.nanstd(available, axis=(0, 1), keepdims=True)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where((np.isfinite(std)) & (std >= 1e-6), std, 1.0).astype(np.float32)
    return mean, std


def _normalize_tokens(tokens: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((tokens.astype(np.float32) - mean) / std).astype(np.float32)


def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(str(contiguous.shape).encode("utf-8"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    return _sha256_arrays(array)


def _sha256_lines(values: Sequence[str]) -> str:
    digest = sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _build_attention_module(*, torch: Any, modality_count: int, token_dim: int, dropout: float) -> Any:
    class _Module(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = torch.nn.Linear(EMBEDDING_DIM, token_dim)
            self.modality_embedding = torch.nn.Parameter(torch.zeros(1, modality_count, token_dim))
            torch.nn.init.normal_(self.modality_embedding, mean=0.0, std=0.02)
            self.self_attention = torch.nn.MultiheadAttention(
                embed_dim=token_dim,
                num_heads=1,
                dropout=dropout,
                batch_first=True,
            )
            self.query = torch.nn.Parameter(torch.zeros(token_dim))
            torch.nn.init.normal_(self.query, mean=0.0, std=0.02)
            self.dropout = torch.nn.Dropout(dropout)
            self.head = torch.nn.Sequential(
                torch.nn.LayerNorm(token_dim),
                torch.nn.Linear(token_dim, token_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(token_dim, 1),
            )

        def forward(self, tokens: Any, mask: Any) -> dict[str, Any]:
            x = self.input_projection(tokens) + self.modality_embedding
            key_padding_mask = ~mask
            attended, _ = self.self_attention(
                x,
                x,
                x,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            attended = self.dropout(attended)
            scores = torch.matmul(attended, self.query)
            scores = scores.masked_fill(~mask, -1.0e9)
            attention = torch.softmax(scores, dim=1)
            attention = attention * mask.to(dtype=attention.dtype)
            denom = attention.sum(dim=1, keepdim=True).clamp_min(1e-6)
            attention = attention / denom
            pooled = torch.sum(attended * attention.unsqueeze(-1), dim=1)
            return {"prediction": self.head(pooled), "attention": attention}

    return _Module()


def _normalize_modalities(values: Sequence[str]) -> tuple[str, ...]:
    modalities = tuple(str(value).strip() for value in values if str(value).strip())
    if not modalities:
        raise ValueError("fusion experiment requires at least one enabled modality")
    unknown = sorted(set(modalities) - set(TOKEN_ORDER))
    if unknown:
        raise ValueError(f"unsupported fusion modalities: {', '.join(unknown)}")
    return modalities


def _load_branch(spec: FusionBranchSpec) -> dict[str, Any]:
    modality = str(spec.modality)
    if modality not in TOKEN_ORDER:
        raise ValueError(f"unsupported branch modality: {modality}")
    path = Path(spec.path)
    with np.load(path, allow_pickle=True) as loaded:
        emb_key = MODALITY_TO_EMB_KEY[modality]
        if emb_key not in loaded.files:
            raise ValueError(f"{path} missing required array {emb_key}")
        sample_id = loaded["sample_id"].astype(str)
        _validate_unique_sample_ids(sample_id, source=str(path))
        mask = loaded["modality_mask"].astype(np.int8)
        if mask.ndim != 2 or mask.shape[1] <= MODALITY_TO_MASK_INDEX[modality]:
            raise ValueError(f"{path} modality_mask does not include {modality} slot")
        return {
            "sample_id": sample_id,
            "event_id": _optional_array(loaded, "event_id", len(sample_id), default_prefix="event"),
            "subject_id": _optional_array(loaded, "subject_id", len(sample_id), default_prefix="subject"),
            "session_id": _optional_array(loaded, "session_id", len(sample_id), default_prefix="session"),
            "labels": None if "labels" not in loaded.files else np.asarray(loaded["labels"].tolist(), dtype=object),
            "embedding": validate_embedding_shape(emb_key, loaded[emb_key], expected_dim=EMBEDDING_DIM),
            "mask": mask[:, MODALITY_TO_MASK_INDEX[modality]].astype(bool),
        }


def _optional_array(loaded: Any, key: str, row_count: int, *, default_prefix: str) -> np.ndarray:
    if key in loaded.files:
        values = loaded[key].astype(str)
    else:
        values = np.asarray([f"{default_prefix}-{idx}" for idx in range(row_count)], dtype=str)
    if len(values) != row_count:
        raise ValueError(f"{key} row count {len(values)} does not match sample_id row count {row_count}")
    return values


def _validate_unique_sample_ids(sample_id: np.ndarray, *, source: str) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in sample_id.astype(str).tolist():
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"{source} contains duplicate sample_id values: {duplicates[:5]}")


def _aligned_sample_ids(
    loaded: Mapping[str, dict[str, Any]],
    *,
    base_sample_ids: Sequence[str] | np.ndarray | None,
) -> np.ndarray:
    if base_sample_ids is not None:
        sample_ids = np.asarray(base_sample_ids).astype(str)
        if len(set(sample_ids.tolist())) != len(sample_ids):
            raise ValueError("base_sample_ids contains duplicate sample_id values")
        for modality, data in loaded.items():
            missing = sorted(set(sample_ids.tolist()) - set(data["sample_id"].tolist()))
            if missing:
                raise ValueError(f"{modality} branch missing sample_id values: {missing[:5]}")
        return sample_ids
    first = next(iter(loaded.values()))
    sample_ids = first["sample_id"].astype(str)
    common = set(sample_ids.tolist())
    for data in loaded.values():
        common &= set(data["sample_id"].astype(str).tolist())
    aligned = np.asarray([sample_id for sample_id in sample_ids.tolist() if sample_id in common], dtype=str)
    if len(aligned) == 0:
        modalities = ", ".join(loaded.keys())
        raise ValueError(f"no common sample_id values across fusion branches: {modalities}")
    return aligned


def _metadata_with_labels(
    loaded: Mapping[str, dict[str, Any]],
    modalities: tuple[str, ...],
    *,
    sample_ids: np.ndarray,
    metadata_source: dict[str, Any] | None,
    metadata_source_path: str | None,
) -> dict[str, Any]:
    for modality in modalities:
        if loaded[modality].get("labels") is not None:
            data = dict(loaded[modality])
            data["source"] = modality
            return data
    if metadata_source is not None and metadata_source.get("labels") is not None:
        missing = sorted(set(sample_ids.astype(str).tolist()) - set(metadata_source["sample_id"].astype(str).tolist()))
        if missing:
            raise ValueError(f"{metadata_source_path or 'metadata_source'} missing sample_id values for labels: {missing[:5]}")
        data = dict(metadata_source)
        data["source"] = metadata_source_path or "metadata_source"
        return data
    searched = ", ".join(modalities)
    raise ValueError(f"fusion branches have no labels array for target lookup: {searched}")


def _indices_for_sample_ids(data: dict[str, Any], sample_ids: np.ndarray, *, source: str) -> np.ndarray:
    index = {sample_id: idx for idx, sample_id in enumerate(data["sample_id"].astype(str).tolist())}
    try:
        return np.asarray([index[str(sample_id)] for sample_id in sample_ids], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"{source} missing sample_id {exc.args[0]!r}") from exc


def _target_values(labels: np.ndarray, target_label: str) -> np.ndarray:
    values: list[float] = []
    for raw in labels.tolist():
        row = _parse_label(raw)
        if target_label not in row:
            raise ValueError(f"labels missing target label {target_label!r}")
        values.append(float(row[target_label]))
    return np.asarray(values, dtype=np.float32)


def _parse_label(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise ValueError("label JSON must decode to an object")
    return parsed
