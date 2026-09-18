"""Wear × MOMENT token 训练模块（EEG-aligned 疲劳预测主线）。

产出 fatigue-supervised 的 256D wear token，直接接入
``scripts/window_fatigue/32_run_eegpt_centered_loss.py`` 的 BRANCHES：

- ``wear_moment_frozen_v1``:     MOMENT-1 encoder 冻结 + 可学习 256D 投影头
- ``wear_moment_partial_ft_v1``: 解冻最后 N 个 transformer block + final norm + 投影头

监督边界与 EEG 侧（``eeg_encoder_matrix``）一致：train = pretrain + finetune，
val 早停选模型，test 冻结评估；输出 token 属于 fatigue-supervised representation。

数据事实（2026-08-20 服务器核验）：
- aligned index 的 sample_id 为 ``eeg_000000`` 风格，本身不含 wear 字段；
- wear 元数据来自
  ``/vePFS-0x0d/DailyEEG_multimodal/embeddings/wear/wear_physio_preprocessed_eeg23win_embeddings.npz``：
  ``wear_mask``、``source_wear_file``（JSON：ppg/gsr/acc 文件名）、``window_start_time/end_time``；
- 源 CSV 实际位于 ``/vePFS-0x0d/DailyEEG_multimodal/raw/wear/out/<basename>``
  （npz 记录的是旧机器 /mnt/dataset0 路径，按 basename 重新定位）；
- 372 个被引用文件（124 PPG + 124 GSR + 124 ACC）全部存在，列结构与
  ``wear_real.TARGET_COLUMNS`` 一致；mask 覆盖 24127/28819。
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from daily_multimodal.split_paths import resolve_protocol_split_root

from daily_multimodal.alignment.time_utils import parse_absolute_time
from daily_multimodal.embeddings.wear_real import (
    TARGET_SAMPLE_RATES_HZ,
    _aligned_sequence_matrix,
    _read_wear_series,
    _resample_series,
)

EMBEDDING_DIM = 256
MATRIX_STEPS = 320
WEAR_MOMENT_PROFILES = {"wear_moment_frozen_v1", "wear_moment_partial_ft_v1"}
DEFAULT_PROTOCOLS = ("cross_subject", "cross_day", "within_subject_day")
DEFAULT_SEEDS = (240800, 240801, 240802)
DEFAULT_INDEX_PATH = "/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/index/eeg_aligned_window_index.jsonl"
DEFAULT_SPLITS_ROOT = "/vePFS-0x0d/DailyEEG/splits_new"
DEFAULT_WEAR_META_NPZ = "/vePFS-0x0d/DailyEEG_multimodal/embeddings/wear/wear_physio_preprocessed_eeg23win_embeddings.npz"
DEFAULT_WEAR_RAW_ROOT = "/vePFS-0x0d/DailyEEG_multimodal/raw/wear/out"
DEFAULT_MOMENT_CHECKPOINT = "outputs/checkpoints/moment-1-small"
DEFAULT_MOMENT_EMBEDDING_CACHE = "outputs/cache/wear_moment_frozen_embeddings.npz"


@dataclass(frozen=True)
class WearMomentDataset:
    sample_id: np.ndarray
    subject_id: np.ndarray
    day_id: np.ndarray
    target: np.ndarray
    matrices: np.ndarray  # (N, 5, 320) float32，逐通道 z-norm；mask=0 的行全零
    mask: np.ndarray  # (N,) bool，复用 wear_physio npz 的 wear_mask
    label_names: tuple[str, ...] = ("inspired", "alert", "determined", "attentive", "active", "hostile", "nervous", "upset", "afraid", "ashamed", "fatigue")

    @property
    def row_count(self) -> int:
        return int(self.sample_id.shape[0])


@dataclass(frozen=True)
class SplitProtocol:
    name: str
    pretrain: np.ndarray
    finetune: np.ndarray
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    source_root: Path


@dataclass(frozen=True)
class WearMomentRuntime:
    epochs: int = 80
    batch_size: int = 256
    learning_rate: float = 1e-3
    encoder_learning_rate: float = 1e-5
    partial_last_n_blocks: int = 2
    weight_decay: float = 1e-4
    dropout: float = 0.1
    patience: int = 15
    grad_clip: float = 1.0
    device: str = "cuda"
    torch_threads: int = 4
    amp: bool = True


def _strategy_for_profile(profile: str) -> str:
    if profile == "wear_moment_frozen_v1":
        return "frozen"
    if profile == "wear_moment_partial_ft_v1":
        return "partial"
    raise ValueError(f"unsupported wear moment profile: {profile}")


def _embedding_supervision_for_profile(profile: str) -> str:
    if profile == "wear_moment_frozen_v1":
        return "frozen_encoder_trainable_projection_head"
    if profile == "wear_moment_partial_ft_v1":
        return "partial_ft_last_blocks_trainable_projection_head"
    return "unknown"


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_wear_meta(npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as loaded:
        return {
            "sample_id": loaded["sample_id"].astype(str),
            "wear_mask": loaded["wear_mask"].astype(bool),
            "source_wear_file": loaded["source_wear_file"],
            "window_start_time": loaded["window_start_time"],
            "window_end_time": loaded["window_end_time"],
        }


def _parse_source_wear_file(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {k: str(v or "") for k, v in value.items()}
    try:
        parsed = json.loads(str(value))
        if isinstance(parsed, dict):
            return {k: str(v or "") for k, v in parsed.items()}
    except (TypeError, ValueError):
        pass
    return {}


def _relocate_wear_path(recorded_path: str, wear_raw_root: Path) -> Path:
    """npz 记录的是旧机器 /mnt/dataset0 路径；按 basename 在 raw root 下重新定位。"""
    name = Path(str(recorded_path)).name
    if not name:
        raise ValueError(f"empty wear source file name from {recorded_path!r}")
    return wear_raw_root / name


def _load_window_sequences(
    recorded_paths: Any,
    *,
    wear_raw_root: Path,
    window_start,
    window_end,
) -> dict[str, np.ndarray] | None:
    """读一个窗口的 PPG/GSR/ACC 重采样序列；任一源缺失返回 None。"""
    sources = _parse_source_wear_file(recorded_paths)
    duration = (window_end - window_start).total_seconds()
    if duration <= 0:
        return None
    sequences: dict[str, np.ndarray] = {}
    for modality in ("ppg", "gsr", "acc"):
        recorded = sources.get(modality, "")
        if not recorded:
            return None
        source_path = _relocate_wear_path(recorded, wear_raw_root)
        if not source_path.is_file():
            return None
        series = _read_wear_series(source_path, modality=modality, window_start=window_start, window_end=window_end)
        sequences[modality] = _resample_series(
            series,
            duration_seconds=duration,
            target_rate_hz=TARGET_SAMPLE_RATES_HZ[modality],
        )
    return sequences


def meta_label_names() -> tuple[str, ...]:
    return ("inspired", "alert", "determined", "attentive", "active", "hostile", "nervous", "upset", "afraid", "ashamed", "fatigue")


def load_wear_moment_dataset(
    index_path: Path | str,
    wear_meta_npz: Path | str,
    *,
    wear_raw_root: Path | str = DEFAULT_WEAR_RAW_ROOT,
    matrix_cache: Path | str | None = None,
    max_rows: int | None = None,
    target_label: str = "fatigue",
    progress_interval: int = 2000,
) -> WearMomentDataset:
    """按 canonical 顺序加载 28,819 窗口的 320×5 wear 矩阵。

    矩阵缓存（--matrix-cache）存在且行数/顺序一致时直接复用，避免重复解析 CSV。
    """
    rows = _load_jsonl(Path(index_path))
    if max_rows is not None:
        rows = rows[:max_rows]
    meta = _load_wear_meta(Path(wear_meta_npz))
    if meta["sample_id"].shape[0] < len(rows):
        raise ValueError("wear meta npz has fewer rows than requested index rows")
    sample_id = np.asarray([str(row["sample_id"]) for row in rows], dtype=str)
    meta_ids = meta["sample_id"][: len(rows)]
    if not np.array_equal(meta_ids, sample_id):
        raise ValueError("wear meta sample_id order does not match canonical index")
    mask = meta["wear_mask"][: len(rows)]

    subject_id = np.asarray([str(row.get("subject_id", "")) for row in rows], dtype=object)
    day_id = np.asarray([str(row.get("day_id", "")) for row in rows], dtype=object)
    label_names = tuple(str(name) for name in (rows[0].get("label_names") or meta_label_names()))
    label_index = label_names.index(target_label) if target_label in label_names else -1
    if label_index < 0:
        label_names = meta_label_names()
        label_index = label_names.index(target_label)
    target = np.asarray([float(row["labels"][label_index]) for row in rows], dtype=np.float32)

    matrices = np.zeros((len(rows), 5, MATRIX_STEPS), dtype=np.float32)
    if matrix_cache is not None and Path(matrix_cache).is_file():
        with np.load(matrix_cache, allow_pickle=True) as cached:
            if int(cached["sample_id"].shape[0]) != len(rows) or not np.array_equal(
                cached["sample_id"].astype(str), sample_id
            ):
                raise ValueError("matrix cache sample_id order does not match canonical index")
            matrices = cached["matrices"].astype(np.float32)
            print(f"loaded wear matrix cache: {Path(matrix_cache)}")
            return WearMomentDataset(
                sample_id=sample_id,
                subject_id=subject_id,
                day_id=day_id,
                target=target,
                matrices=matrices,
                mask=mask,
                label_names=label_names,
            )

    raw_root = Path(wear_raw_root)
    start = time.time()
    loaded_count = 0
    for index, row in enumerate(rows):
        if not mask[index]:
            continue
        window_start = parse_absolute_time(str(meta["window_start_time"][index]))
        window_end = parse_absolute_time(str(meta["window_end_time"][index]))
        sequences = _load_window_sequences(
            meta["source_wear_file"][index],
            wear_raw_root=raw_root,
            window_start=window_start,
            window_end=window_end,
        )
        if sequences is None:
            continue
        matrices[index] = _aligned_sequence_matrix(sequences, target_steps=MATRIX_STEPS).T  # (5, 320) 通道在前
        loaded_count += 1
        if (index + 1) % progress_interval == 0:
            elapsed = time.time() - start
            print(f"wear sequences: {index + 1}/{len(rows)} loaded={loaded_count} elapsed={elapsed:.1f}s", flush=True)

    if matrix_cache is not None:
        cache_path = Path(matrix_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, sample_id=sample_id, matrices=matrices)
        print(f"wrote wear matrix cache: {cache_path}")

    print(f"wear matrix build done: rows={len(rows)} loaded={loaded_count} mask_sum={int(mask.sum())} "
          f"elapsed={time.time() - start:.1f}s")
    return WearMomentDataset(
        sample_id=sample_id,
        subject_id=subject_id,
        day_id=day_id,
        target=target,
        matrices=matrices,
        mask=mask,
        label_names=label_names,
    )


def load_split_protocols(
    splits_root: Path | str,
    protocols: tuple[str, ...],
    dataset: WearMomentDataset,
) -> dict[str, SplitProtocol]:
    root = Path(splits_root)
    result: dict[str, SplitProtocol] = {}
    for protocol in protocols:
        protocol_root = resolve_protocol_split_root(root, protocol)
        pretrain = _load_indices(protocol_root / "pretrain.json", dataset.row_count)
        finetune = _load_indices(protocol_root / "finetune.json", dataset.row_count)
        val = _load_indices(protocol_root / "val.json", dataset.row_count)
        test = _load_indices(protocol_root / "test.json", dataset.row_count)
        train = np.concatenate([pretrain, finetune]).astype(np.int64)
        _validate_split_no_overlap(protocol, train=train, val=val, test=test)
        result[protocol] = SplitProtocol(
            name=protocol,
            pretrain=pretrain,
            finetune=finetune,
            train=train,
            val=val,
            test=test,
            source_root=protocol_root,
        )
    return result


def _load_indices(path: Path, row_count: int) -> np.ndarray:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("indices", value.get("index", value.get("rows")))
    values = np.asarray(value, dtype=np.int64).reshape(-1)
    if values.size and (values.min() < 0 or values.max() >= row_count):
        raise ValueError(f"{path} contains out-of-range indices")
    return values


def _validate_split_no_overlap(protocol: str, *, train: np.ndarray, val: np.ndarray, test: np.ndarray) -> None:
    overlaps = {
        "train/val": int(np.intersect1d(train, val).size),
        "train/test": int(np.intersect1d(train, test).size),
        "val/test": int(np.intersect1d(val, test).size),
    }
    if any(overlaps.values()):
        raise ValueError(f"split overlap in {protocol}: {overlaps}")


# ---------------------------------------------------------------------------
# Torch 模型与训练
# ---------------------------------------------------------------------------


def build_moment_encoder(checkpoint: Path | str, torch: Any) -> Any:
    """构建 MOMENT-1 embedding encoder；本地 checkpoint 下禁止访问 HF。"""
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from momentfm import MOMENTPipeline

    model = MOMENTPipeline.from_pretrained(str(checkpoint), model_kwargs={"task_name": "embedding"})
    model.init()
    return model


def moment_embedding_dim(model: Any) -> int:
    config = getattr(model, "config", None)
    d_model = getattr(config, "d_model", None)
    if d_model is not None:
        return int(d_model)
    return int(model.d_model)


def encode_moment_batch(model: Any, x: Any, torch: Any) -> Any:
    """MOMENT-1 (momentfm 0.1.4) 推理；返回 (B, d_model) 的序列级表征。

    ``MOMENT.forward`` 只接受 keyword-only 的 ``x_enc``；embedding 任务的输出是
    ``TimeseriesOutputs(embeddings=...)``，reduction="mean" 时可能为 (B, d_model)
    或 (B, C, d_model)，统一压到 (B, d_model)。
    """
    output = model(x_enc=x)
    embeddings = output.embeddings
    if embeddings.dim() == 3:
        embeddings = embeddings.mean(dim=1)
    return embeddings


def build_wear_moment_module(encoder: Any, d_model: int, torch: Any, *, dropout: float = 0.1) -> Any:
    class Module(torch.nn.Module):
        def __init__(self, encoder_: Any) -> None:
            super().__init__()
            self.encoder = encoder_
            self.projection = torch.nn.Sequential(
                torch.nn.Linear(d_model, EMBEDDING_DIM),
                torch.nn.Dropout(dropout),
            )
            self.regression_head = torch.nn.Linear(EMBEDDING_DIM, 1)

        def forward(self, x: Any) -> Any:
            embedding = self.projection(encode_moment_batch(self.encoder, x, torch))
            return embedding, self.regression_head(embedding).reshape(-1)

        def token_embedding(self, x: Any) -> Any:
            embedding, _ = self.forward(x)
            return embedding

    return Module(encoder)


def configure_encoder_trainability(module: Any, strategy: str, torch: Any, *, last_n_blocks: int = 2) -> dict[str, Any]:
    """frozen：encoder 全冻结；partial：解冻最后 last_n_blocks 个 transformer block + final norm。"""
    block_pattern = re.compile(r"\.encoder\.block\.(\d+)\.")
    block_indices: set[int] = set()
    for name, _param in module.encoder.named_parameters():
        match = block_pattern.search(name)
        if match:
            block_indices.add(int(match.group(1)))
    trainable_count = 0
    if strategy == "frozen":
        for name, param in module.encoder.named_parameters():
            param.requires_grad = False
    elif strategy == "partial":
        n_blocks = max(block_indices) + 1 if block_indices else 0
        cutoff = max(0, n_blocks - last_n_blocks)
        for name, param in module.encoder.named_parameters():
            match = block_pattern.search(name)
            if match and int(match.group(1)) >= cutoff:
                param.requires_grad = True
                trainable_count += 1
            elif "final_layer_norm" in name:
                param.requires_grad = True
                trainable_count += 1
            else:
                param.requires_grad = False
    else:
        raise ValueError(f"unsupported trainability strategy: {strategy}")
    return {
        "strategy": strategy,
        "trainable_count": trainable_count,
        "block_count": len(block_indices),
        "last_n_blocks": last_n_blocks,
    }


def _seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _torch_optimizer_for_strategy(module: Any, strategy: str, runtime: WearMomentRuntime, torch: Any) -> Any:
    if strategy == "frozen":
        return torch.optim.AdamW(
            [param for param in module.parameters() if param.requires_grad],
            lr=runtime.learning_rate,
            weight_decay=runtime.weight_decay,
        )
    encoder_params = [
        param for name, param in module.named_parameters() if param.requires_grad and "encoder." in name
    ]
    other_params = [
        param
        for param in module.parameters()
        if param.requires_grad and all(param is not other for other in encoder_params)
    ]
    return torch.optim.AdamW(
        [
            {"params": other_params, "lr": runtime.learning_rate},
            {"params": encoder_params, "lr": runtime.encoder_learning_rate},
        ],
        weight_decay=runtime.weight_decay,
    )


def train_wear_moment_tokens(
    dataset: WearMomentDataset,
    split: SplitProtocol,
    *,
    profile: str,
    seed: int,
    runtime: WearMomentRuntime,
    torch: Any,
    checkpoint: Path | str = DEFAULT_MOMENT_CHECKPOINT,
    moment_embedding_cache: Path | str | None = DEFAULT_MOMENT_EMBEDDING_CACHE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """训练 frozen/partial_ft 档并产出全量 (N,256) wear token。"""
    strategy = _strategy_for_profile(profile)
    _seed_everything(seed, torch)
    device = torch.device(runtime.device)
    mask = dataset.mask
    train_idx = split.train[mask[split.train]]
    val_idx = split.val[mask[split.val]]
    if train_idx.size < 10 or val_idx.size < 2:
        raise RuntimeError(f"{profile} {split.name}: not enough unmasked train/val rows "
                           f"(train={train_idx.size}, val={val_idx.size})")
    y_mean = float(dataset.target[train_idx].mean())
    y_std = float(dataset.target[train_idx].std()) or 1.0

    cached_embeddings: np.ndarray | None = None
    if strategy == "frozen" and moment_embedding_cache is not None and Path(moment_embedding_cache).is_file():
        with np.load(moment_embedding_cache, allow_pickle=True) as cached:
            if not np.array_equal(cached["sample_id"].astype(str), dataset.sample_id):
                raise ValueError("moment embedding cache sample_id mismatch")
            cached_embeddings = cached["embeddings"].astype(np.float32)
        d_model = int(cached_embeddings.shape[1])

        class HeadOnly(torch.nn.Module):
            def __init__(self, d_model_: int) -> None:
                super().__init__()
                self.encoder = None
                self.projection = torch.nn.Sequential(
                    torch.nn.Linear(d_model_, EMBEDDING_DIM),
                    torch.nn.Dropout(runtime.dropout),
                )
                self.regression_head = torch.nn.Linear(EMBEDDING_DIM, 1)

        module = HeadOnly(d_model)
        trainable_config = {"strategy": "frozen", "trainable_count": 0, "block_count": 0, "last_n_blocks": 0}
    else:
        encoder = build_moment_encoder(checkpoint, torch)
        d_model = moment_embedding_dim(encoder)
        module = build_wear_moment_module(encoder, d_model, torch, dropout=runtime.dropout)
        trainable_config = configure_encoder_trainability(module, strategy, torch, last_n_blocks=runtime.partial_last_n_blocks)
    module = module.to(device)
    optimizer = _torch_optimizer_for_strategy(module, strategy, runtime, torch)

    use_amp = bool(runtime.amp and runtime.device.startswith("cuda"))

    def raw_predict(inputs: Any) -> Any:
        if cached_embeddings is not None:
            embedding = module.projection(inputs)
            return module.regression_head(embedding).reshape(-1)
        embedding, prediction = module(inputs)
        return prediction

    def token_embedding(inputs: Any) -> Any:
        if cached_embeddings is not None:
            return module.projection(inputs)
        return module.token_embedding(inputs)

    if cached_embeddings is not None:
        train_inputs = torch.as_tensor(cached_embeddings[train_idx], dtype=torch.float32, device=device)
        val_inputs = torch.as_tensor(cached_embeddings[val_idx], dtype=torch.float32, device=device)
    else:
        train_inputs = torch.as_tensor(dataset.matrices[train_idx], dtype=torch.float32, device=device)
        val_inputs = torch.as_tensor(dataset.matrices[val_idx], dtype=torch.float32, device=device)
    y_train = torch.as_tensor((dataset.target[train_idx] - y_mean) / y_std, dtype=torch.float32, device=device)
    y_val_raw = dataset.target[val_idx].astype(np.float32)

    best_state = None
    best_val_rmse = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(max(1, runtime.epochs)):
        module.train()
        permutation = torch.randperm(train_idx.size, device=device)
        epoch_losses: list[float] = []
        for start in range(0, train_idx.size, runtime.batch_size):
            batch = permutation[start : start + runtime.batch_size]
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    prediction = raw_predict(train_inputs[batch])
                    loss = torch.mean((prediction - y_train[batch]) ** 2)
            else:
                prediction = raw_predict(train_inputs[batch])
                loss = torch.mean((prediction - y_train[batch]) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), runtime.grad_clip)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        module.eval()
        with torch.no_grad():
            val_predictions: list[np.ndarray] = []
            for start in range(0, val_idx.size, 1024):
                x_batch = val_inputs[start : start + 1024]
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        prediction = raw_predict(x_batch)
                else:
                    prediction = raw_predict(x_batch)
                val_predictions.append((prediction.detach().cpu().numpy() * y_std + y_mean).astype(np.float32))
            val_pred = np.concatenate(val_predictions) if val_predictions else np.zeros((0,), dtype=np.float32)
        val_rmse = float(np.sqrt(np.mean((val_pred - y_val_raw) ** 2))) if val_pred.size else float("inf")
        history.append({"epoch": int(epoch + 1), "train_loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan"), "val_rmse": val_rmse})
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch + 1
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
        else:
            stale += 1
            if stale >= runtime.patience:
                break
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[{profile} {split.name} seed={seed}] epoch={epoch + 1} train_loss={history[-1]['train_loss']:.4f} val_rmse={val_rmse:.4f}", flush=True)

    if best_state is None:
        raise RuntimeError(f"{profile} {split.name}: training produced no valid epoch")
    module.load_state_dict(best_state)
    module.eval()

    tokens = np.zeros((dataset.row_count, EMBEDDING_DIM), dtype=np.float32)
    unmasked = np.flatnonzero(mask)
    with torch.no_grad():
        for start in range(0, unmasked.size, 512):
            indices = unmasked[start : start + 512]
            if cached_embeddings is not None:
                inputs = torch.as_tensor(cached_embeddings[indices], dtype=torch.float32, device=device)
            else:
                inputs = torch.as_tensor(dataset.matrices[indices], dtype=torch.float32, device=device)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    embedding = token_embedding(inputs)
            else:
                embedding = token_embedding(inputs)
            tokens[indices] = embedding.detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(tokens).all():
        raise RuntimeError(f"{profile} {split.name}: tokens contain NaN or infinite values")

    audit = {
        "profile": profile,
        "protocol": split.name,
        "seed": int(seed),
        "strategy": strategy,
        "trainable_config": trainable_config,
        "train_count": int(train_idx.size),
        "val_count": int(val_idx.size),
        "best_epoch": int(best_epoch),
        "best_val_rmse": float(best_val_rmse),
        "epoch_count": len(history),
        "history": history,
        "train_supervision": _embedding_supervision_for_profile(profile),
        "normalization": "train_only",
        "token_nan_count": int(np.isnan(tokens).sum()),
        "token_mask_sum": int(mask.sum()),
        "token_coverage": float(mask.mean()),
    }
    return tokens, audit


def write_wear_token_npz(
    tokens: np.ndarray,
    path: Path,
    *,
    dataset: WearMomentDataset,
    profile: str,
    protocol: str,
    seed: int,
    split: SplitProtocol,
    train_supervision: str,
    source_checkpoint: str,
    checkpoint_sha256: str,
) -> str:
    if tokens.shape != (dataset.row_count, EMBEDDING_DIM):
        raise ValueError(f"expected wear tokens shape {(dataset.row_count, EMBEDDING_DIM)}, got {tokens.shape}")
    if not np.isfinite(tokens).all():
        raise ValueError("wear tokens contain NaN or infinite values")
    path.parent.mkdir(parents=True, exist_ok=True)
    modality_mask = np.zeros((dataset.row_count, 4), dtype=np.int8)
    modality_mask[:, 1] = dataset.mask.astype(np.int8)
    np.savez_compressed(
        path,
        sample_id=dataset.sample_id,
        subject_id=dataset.subject_id,
        day_id=dataset.day_id,
        wear_emb=tokens,
        wear_mask=dataset.mask.astype(np.int8),
        modality_mask=modality_mask,
        encoder_profile=np.asarray([profile] * dataset.row_count, dtype=object),
        encoder_version=np.asarray(["wear_moment_256d_supervised_v1"] * dataset.row_count, dtype=object),
        protocol=np.asarray([protocol], dtype=object),
        seed=np.asarray([int(seed)], dtype=np.int64),
        train_index=split.train,
        val_index=split.val,
        test_index=split.test,
        train_supervision=np.asarray([train_supervision], dtype=object),
        source_checkpoint=np.asarray([source_checkpoint], dtype=object),
        checkpoint_sha256=np.asarray([checkpoint_sha256], dtype=object),
    )
    return str(path)


def checkpoint_sha256(path: Path | str, *, sample_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(sample_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# 矩阵编排
# ---------------------------------------------------------------------------


class _null_context:
    def __enter__(self):
        return self

    def __exit__(self, *args: Any) -> None:
        return None


def run_wear_moment_matrix(
    index_path: Path | str,
    splits_root: Path | str,
    wear_meta_npz: Path | str,
    *,
    wear_raw_root: Path | str = DEFAULT_WEAR_RAW_ROOT,
    checkpoint: Path | str = DEFAULT_MOMENT_CHECKPOINT,
    profiles: tuple[str, ...] = ("wear_moment_frozen_v1", "wear_moment_partial_ft_v1"),
    protocols: tuple[str, ...] = DEFAULT_PROTOCOLS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    runtime: WearMomentRuntime | None = None,
    embeddings_dir: Path | str | None = None,
    matrix_cache: Path | str | None = None,
    moment_embedding_cache: Path | str | None = DEFAULT_MOMENT_EMBEDDING_CACHE,
    max_rows: int | None = None,
    out_json: Path | str | None = None,
    out_md: Path | str | None = None,
    target_label: str = "fatigue",
) -> dict[str, Any]:
    unknown = sorted(set(profiles) - WEAR_MOMENT_PROFILES)
    if unknown:
        raise ValueError(f"unsupported wear moment profiles: {', '.join(unknown)}")
    runtime = runtime or WearMomentRuntime()
    started = time.time()
    dataset = load_wear_moment_dataset(
        index_path,
        wear_meta_npz,
        wear_raw_root=wear_raw_root,
        matrix_cache=matrix_cache,
        max_rows=max_rows,
        target_label=target_label,
    )
    splits = load_split_protocols(splits_root, protocols, dataset)

    torch = _import_torch()
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file() and not (checkpoint_path / "config.json").is_file():
        raise FileNotFoundError(f"MOMENT checkpoint not found: {checkpoint_path}")

    if "wear_moment_frozen_v1" in profiles and moment_embedding_cache is not None:
        cache_path = Path(moment_embedding_cache)
        if not cache_path.is_file():
            encoder = build_moment_encoder(checkpoint_path, torch)
            encoder.to(torch.device(runtime.device))
            encoder.eval()
            unmasked = np.flatnonzero(dataset.mask)
            d_model = moment_embedding_dim(encoder)
            embeddings = np.zeros((dataset.row_count, d_model), dtype=np.float32)
            with torch.no_grad():
                for start in range(0, unmasked.size, 256):
                    indices = unmasked[start : start + 256]
                    x_batch = torch.as_tensor(dataset.matrices[indices], dtype=torch.float32, device=torch.device(runtime.device))
                    ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if runtime.amp and runtime.device.startswith("cuda") else _null_context()
                    with ctx:
                        emb = encode_moment_batch(encoder, x_batch, torch)
                    embeddings[indices] = emb.detach().cpu().numpy().astype(np.float32)
                    if (start // 256 + 1) % 20 == 0:
                        print(f"moment encode: {start + indices.size}/{unmasked.size}", flush=True)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, sample_id=dataset.sample_id, embeddings=embeddings)
            print(f"moment embedding cache written: {cache_path} {embeddings.shape}")
        else:
            with np.load(cache_path, allow_pickle=True) as cached:
                print(f"moment embedding cache loaded: {cache_path} {cached['embeddings'].shape}")

    sha = checkpoint_sha256(checkpoint_path / "model.safetensors") if (checkpoint_path / "model.safetensors").is_file() else ""
    results: list[dict[str, Any]] = []
    for protocol in protocols:
        for profile in profiles:
            for seed in seeds:
                try:
                    tokens, audit = train_wear_moment_tokens(
                        dataset,
                        splits[protocol],
                        profile=profile,
                        seed=seed,
                        runtime=runtime,
                        torch=torch,
                        checkpoint=checkpoint_path,
                        moment_embedding_cache=moment_embedding_cache,
                    )
                except Exception as exc:  # noqa: BLE001 - 单 run 失败不中断矩阵
                    results.append(
                        {
                            "protocol": protocol,
                            "profile": profile,
                            "seed": int(seed),
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    print(f"FAILED {protocol}/{profile}/seed_{seed}: {exc}", flush=True)
                    continue
                row: dict[str, Any] = {
                    "protocol": protocol,
                    "profile": profile,
                    "seed": int(seed),
                    "status": "ok",
                    "train_audit": audit,
                    "token_path": None,
                }
                if embeddings_dir is not None:
                    token_path = Path(embeddings_dir) / protocol / profile / f"seed_{seed}.npz"
                    row["token_path"] = write_wear_token_npz(
                        tokens,
                        token_path,
                        dataset=dataset,
                        profile=profile,
                        protocol=protocol,
                        seed=seed,
                        split=splits[protocol],
                        train_supervision=audit["train_supervision"],
                        source_checkpoint=str(checkpoint_path),
                        checkpoint_sha256=sha,
                    )
                results.append(row)
                print(f"completed {protocol}/{profile}/seed_{seed} val_rmse={audit['best_val_rmse']:.4f}", flush=True)

    output: dict[str, Any] = {
        "stage": "wear_moment_matrix",
        "target_label": target_label,
        "index_path": str(index_path),
        "splits_root": str(splits_root),
        "wear_meta_npz": str(wear_meta_npz),
        "wear_raw_root": str(wear_raw_root),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha,
        "profiles": list(profiles),
        "protocols": list(protocols),
        "seeds": [int(seed) for seed in seeds],
        "runtime": {
            "epochs": runtime.epochs,
            "batch_size": runtime.batch_size,
            "learning_rate": runtime.learning_rate,
            "encoder_learning_rate": runtime.encoder_learning_rate,
            "partial_last_n_blocks": runtime.partial_last_n_blocks,
            "weight_decay": runtime.weight_decay,
            "dropout": runtime.dropout,
            "patience": runtime.patience,
            "device": runtime.device,
            "amp": runtime.amp,
            "normalization": "train_only",
            "train_rule": "pretrain + finetune from splits_new",
        },
        "dataset": {
            "row_count": dataset.row_count,
            "mask_sum": int(dataset.mask.sum()),
            "coverage": float(dataset.mask.mean()),
        },
        "elapsed_seconds": float(time.time() - started),
        "run_count": len(results),
        "results": results,
    }
    if out_json is not None:
        _write_json(output, out_json)
    if out_md is not None:
        _write_matrix_markdown(output, out_md)
    return output


def run_preflight(
    index_path: Path | str,
    wear_meta_npz: Path | str,
    *,
    wear_raw_root: Path | str = DEFAULT_WEAR_RAW_ROOT,
    checkpoint: Path | str = DEFAULT_MOMENT_CHECKPOINT,
    matrix_cache: Path | str | None = None,
    max_rows: int | None = 200,
    out_json: Path | str | None = None,
    out_md: Path | str | None = None,
) -> dict[str, Any]:
    started = time.time()
    dataset = load_wear_moment_dataset(
        index_path,
        wear_meta_npz,
        wear_raw_root=wear_raw_root,
        matrix_cache=matrix_cache,
        max_rows=max_rows,
    )
    moment_info: dict[str, Any] = {"checked": False}
    try:
        torch = _import_torch()
    except RuntimeError as exc:
        moment_info = {"checked": False, "error": str(exc)}
        torch = None

    if torch is not None and torch.cuda.is_available():
        try:
            checkpoint_path = Path(checkpoint)
            model_start = time.time()
            encoder = build_moment_encoder(checkpoint_path, torch)
            encoder.to(torch.device("cuda"))
            encoder.eval()
            load_seconds = time.time() - model_start
            unmasked = np.flatnonzero(dataset.mask)[:64]
            infer_start = time.time()
            with torch.no_grad():
                for start in range(0, unmasked.size, 16):
                    x_batch = torch.as_tensor(dataset.matrices[unmasked[start : start + 16]], dtype=torch.float32, device="cuda")
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        encode_moment_batch(encoder, x_batch, torch)
            inference_seconds = time.time() - infer_start
            moment_info = {
                "checked": True,
                "load_seconds": float(load_seconds),
                "inference_seconds_64_windows": float(inference_seconds),
                "per_window_seconds": float(inference_seconds / max(1, unmasked.size)),
            }
        except Exception as exc:  # noqa: BLE001 - momentfm/checkpoint 缺失时只报告，不阻塞数据侧 preflight
            moment_info = {"checked": False, "error_type": type(exc).__name__, "error": str(exc)}

    result = {
        "ok": True,
        "row_count": dataset.row_count,
        "mask_sum": int(dataset.mask.sum()),
        "coverage": float(dataset.mask.mean()),
        "sample_id_order_verified": True,
        "elapsed_seconds": float(time.time() - started),
        "moment": moment_info,
    }
    if out_json is not None:
        _write_json(result, out_json)
    if out_md is not None:
        lines = [
            "# Wear × MOMENT Preflight",
            "",
            f"- row_count: `{result['row_count']}`",
            f"- wear mask sum / coverage: `{result['mask_sum']}` / `{result['coverage']:.4f}`",
            f"- sample_id order verified: `{result['sample_id_order_verified']}`",
            f"- elapsed: `{result['elapsed_seconds']:.1f}s`",
            f"- MOMENT: `{moment_info}`",
            "",
        ]
        Path(out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(out_md).write_text("\n".join(lines), encoding="utf-8")
    return result


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for wear moment profiles") from exc
    return torch


def _write_json(value: dict[str, Any], path: Path | str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_matrix_markdown(output: dict[str, Any], path: Path | str) -> None:
    lines = [
        "# Wear × MOMENT Token Matrix",
        "",
        f"checkpoint: `{output['checkpoint']}`",
        f"checkpoint_sha256: `{output['checkpoint_sha256'][:16]}...`",
        f"row_count / mask coverage: `{output['dataset']['row_count']}` / `{output['dataset']['coverage']:.4f}`",
        "",
        "| protocol | profile | seed | status | best epoch | best val RMSE | train count |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in output["results"]:
        audit = row.get("train_audit") or {}
        lines.append(
            f"| {row['protocol']} | {row['profile']} | {row.get('seed', '')} | {row.get('status')} | "
            f"{audit.get('best_epoch', '')} | {audit.get('best_val_rmse', '')} | {audit.get('train_count', '')} |"
        )
    lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
