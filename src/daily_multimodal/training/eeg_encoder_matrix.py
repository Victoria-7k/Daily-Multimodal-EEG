"""EEG encoder comparison matrix for EEG-aligned fatigue experiments."""

from __future__ import annotations

import json
import math
import random
import time
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from daily_multimodal.training.centered_metrics import evaluate_regression_with_centered


LABEL_NAMES = [
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
DEFAULT_PROTOCOLS = ("cross_subject", "cross_day", "within_subject_day")
MAIN_PROTOCOLS = ("cross_day", "within_subject_day")
DEFAULT_SEEDS = (240800, 240801, 240802, 240803, 240804)
DEFAULT_DATA_ROOT = Path("/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new")
DEFAULT_ALIGNED_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_SPLITS_ROOT = DEFAULT_ALIGNED_ROOT / "outputs/splits"
DEFAULT_INDEX_PATH = DEFAULT_ALIGNED_ROOT / "index/eeg_aligned_window_index.jsonl"
DEFAULT_EEGPT_FROZEN_EMBEDDINGS = Path(
    "/vePFS-0x0d/DailyEEG_multimodal/embeddings/eeg/eeg_eegpt_eeg23win_embeddings.npz"
)

EEG_PROFILES = {
    "eegpt_frozen_v1",
    "eeg_de_5band_1s_avg_v1",
    "eegpt_partial_ft_v1",
    "eegpt_full_ft_v1",
    "cbramod_frozen_v1",
    "cbramod_partial_ft_v1",
    "cbramod_full_ft_v1",
}
TORCH_PROFILES = {
    "eegpt_partial_ft_v1",
    "eegpt_full_ft_v1",
    "cbramod_frozen_v1",
    "cbramod_partial_ft_v1",
    "cbramod_full_ft_v1",
}
FULL_PROFILE_BY_PARTIAL = {
    "eegpt_partial_ft_v1": "eegpt_full_ft_v1",
    "cbramod_partial_ft_v1": "cbramod_full_ft_v1",
}
FROZEN_PROFILE_BY_PARTIAL = {
    "eegpt_partial_ft_v1": "eegpt_frozen_v1",
    "cbramod_partial_ft_v1": "cbramod_frozen_v1",
}
DE_BANDS_HZ = ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))


@dataclass(frozen=True)
class EEGAlignedDataset:
    x_path: Path
    y_path: Path
    sub_path: Path
    day_path: Path
    x: np.ndarray
    y: np.ndarray
    subject_id: np.ndarray
    day_id: np.ndarray
    sample_id: np.ndarray
    label_names: tuple[str, ...] = tuple(LABEL_NAMES)

    @property
    def row_count(self) -> int:
        return int(self.x.shape[0])

    @property
    def sample_count(self) -> int:
        return int(self.x.shape[1])

    @property
    def channel_count(self) -> int:
        return int(self.x.shape[2])


@dataclass(frozen=True)
class SplitProtocol:
    name: str
    pretrain: np.ndarray
    finetune: np.ndarray
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    source_root: Path

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "pretrain": self.pretrain,
            "finetune": self.finetune,
            "train": self.train,
            "val": self.val,
            "test": self.test,
        }


@dataclass(frozen=True)
class MatrixRuntime:
    epochs: int = 80
    hidden_dim: int = 128
    batch_size: int = 256
    learning_rate: float = 1e-3
    head_learning_rate: float = 1e-3
    partial_encoder_learning_rate: float = 1e-5
    full_encoder_learning_rate: float = 5e-6
    weight_decay: float = 1e-4
    dropout: float = 0.1
    patience: int = 15
    grad_clip: float = 1.0
    fallback_batch_size: int = 64
    device: str = "cuda"
    torch_threads: int = 4
    amp: bool = True
    partial_last_n_blocks: int = 2


@dataclass(frozen=True)
class CBraModAcquisitionPlan:
    primary_source: str
    install_command: str
    download_command: str
    local_checkpoint_argument: str
    original_repo_fallback: str
    default_repo_id: str = "braindecode/cbramod-pretrained"
    explicit_download_required_by_default: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_source": self.primary_source,
            "default_repo_id": self.default_repo_id,
            "install_command": self.install_command,
            "download_command": self.download_command,
            "local_checkpoint_argument": self.local_checkpoint_argument,
            "original_repo_fallback": self.original_repo_fallback,
            "explicit_download_required_by_default": self.explicit_download_required_by_default,
        }


def cbramod_acquisition_plan(cache_dir: Path | str = "outputs/checkpoints/cbramod-pretrained") -> CBraModAcquisitionPlan:
    cache = Path(cache_dir)
    return CBraModAcquisitionPlan(
        primary_source="Braindecode/Hugging Face: CBraMod.from_pretrained('braindecode/cbramod-pretrained', return_encoder_output=True)",
        install_command="python -m pip install 'braindecode[hub]'",
        download_command=f"huggingface-cli download braindecode/cbramod-pretrained --local-dir {cache.as_posix()}",
        local_checkpoint_argument=f"--cbramod-checkpoint {cache.as_posix()}",
        original_repo_fallback="If the Braindecode hub path is unavailable, manually place CBraMod pretrained_weights.pth from https://github.com/wjq-learning/CBraMod and pass it via --cbramod-checkpoint.",
    )


def load_eeg_aligned_dataset(
    *,
    data_root: Path | str,
    index_path: Path | str | None = None,
    mmap_mode: str | None = "r",
) -> EEGAlignedDataset:
    root = Path(data_root)
    x_path = root / "X.npy"
    y_path = root / "y.npy"
    sub_path = root / "sub.npy"
    day_path = root / "d.npy"
    for path in (x_path, y_path, sub_path, day_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing EEG aligned array: {path}")
    x = np.load(x_path, mmap_mode=mmap_mode)
    y = np.load(y_path, mmap_mode=mmap_mode)
    sub = np.load(sub_path, mmap_mode=mmap_mode)
    day = np.load(day_path, mmap_mode=mmap_mode)
    if x.ndim != 3:
        raise ValueError(f"expected X shape [rows, samples, channels], got {x.shape}")
    row_count = int(x.shape[0])
    if y.shape[0] != row_count or sub.shape[0] != row_count or day.shape[0] != row_count:
        raise ValueError("X, y, sub, and d row counts must match")
    sample_id = _sample_ids_from_index(index_path, row_count)
    return EEGAlignedDataset(
        x_path=x_path,
        y_path=y_path,
        sub_path=sub_path,
        day_path=day_path,
        x=x,
        y=y,
        subject_id=np.asarray([_norm_subject(value) for value in np.asarray(sub).reshape(-1)], dtype=str),
        day_id=np.asarray(day).reshape(-1).astype(str),
        sample_id=sample_id,
    )


def load_split_protocols(
    splits_root: Path | str,
    protocols: tuple[str, ...] | list[str] = DEFAULT_PROTOCOLS,
    *,
    row_count: int,
) -> dict[str, SplitProtocol]:
    root = Path(splits_root)
    loaded: dict[str, SplitProtocol] = {}
    for protocol in protocols:
        protocol_root = root / protocol
        split = {name: _load_indices(protocol_root / f"{name}.json", row_count) for name in ("pretrain", "finetune", "val", "test")}
        train = np.asarray(split["pretrain"].tolist() + split["finetune"].tolist(), dtype=np.int64)
        _validate_split_no_overlap(protocol, train=train, val=split["val"], test=split["test"])
        loaded[protocol] = SplitProtocol(
            name=protocol,
            pretrain=split["pretrain"],
            finetune=split["finetune"],
            train=train,
            val=split["val"],
            test=split["test"],
            source_root=protocol_root,
        )
    return loaded


def run_eeg_encoder_matrix(
    *,
    data_root: Path | str = DEFAULT_DATA_ROOT,
    splits_root: Path | str = DEFAULT_SPLITS_ROOT,
    index_path: Path | str | None = DEFAULT_INDEX_PATH,
    profiles: tuple[str, ...] = ("eegpt_frozen_v1", "eeg_de_5band_1s_avg_v1", "cbramod_frozen_v1", "eegpt_partial_ft_v1", "cbramod_partial_ft_v1"),
    protocols: tuple[str, ...] = DEFAULT_PROTOCOLS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    target_label: str = "fatigue",
    eegpt_frozen_embeddings: Path | str | None = DEFAULT_EEGPT_FROZEN_EMBEDDINGS,
    cbramod_checkpoint: Path | str | None = None,
    eegpt_checkpoint: Path | str | None = None,
    allow_cbramod_download: bool = False,
    max_rows: int | None = None,
    runtime: MatrixRuntime = MatrixRuntime(),
    out_json: Path | str | None = None,
    out_md: Path | str | None = None,
    predictions_dir: Path | str | None = None,
    embeddings_dir: Path | str | None = None,
    de_feature_cache: Path | str | None = None,
    full_gate: bool = False,
) -> dict[str, Any]:
    unknown_profiles = sorted(set(profiles) - EEG_PROFILES)
    if unknown_profiles:
        raise ValueError(f"unsupported EEG profiles: {', '.join(unknown_profiles)}")
    full_dataset = load_eeg_aligned_dataset(data_root=data_root, index_path=index_path)
    full_splits = load_split_protocols(splits_root, protocols, row_count=full_dataset.row_count)
    row_filter: np.ndarray | None = None
    if max_rows is None:
        dataset = full_dataset
        splits = full_splits
    else:
        capped = {name: _cap_split_for_smoke(split, max_rows=max_rows) for name, split in full_splits.items()}
        row_filter = np.asarray(
            sorted({int(value) for split in capped.values() for values in split.as_dict().values() for value in values.tolist()}),
            dtype=np.int64,
        )
        dataset = _filtered_dataset(full_dataset, row_filter)
        splits = _remap_protocols(capped, row_filter=row_filter)
    target = target_array(dataset.y, target_label)
    profile_list = list(profiles)
    results: list[dict[str, Any]] = []
    feature_cache: dict[str, np.ndarray] = {}
    profile_errors: dict[str, str] = {}

    for profile in profile_list:
        try:
            if profile == "eeg_de_5band_1s_avg_v1":
                feature_cache[profile] = load_or_compute_de_features(
                    dataset.x,
                    cache_path=de_feature_cache,
                    sample_rate_hz=200.0,
                    seconds_per_window=10,
                    batch_size=runtime.batch_size,
                )
            elif profile == "eegpt_frozen_v1":
                if eegpt_frozen_embeddings is None:
                    raise ValueError("eegpt_frozen_v1 requires --eegpt-frozen-embeddings")
                feature_cache[profile] = load_frozen_eeg_embeddings(
                    eegpt_frozen_embeddings,
                    expected_sample_id=dataset.sample_id,
                    row_filter=row_filter,
                )
        except Exception as exc:
            profile_errors[profile] = str(exc)

    for protocol_name, split in splits.items():
        for profile in profile_list:
            if profile in profile_errors:
                results.append(_skipped_result(protocol_name, profile, reason=profile_errors[profile]))
                continue
            for seed in seeds:
                start = time.time()
                try:
                    if profile in feature_cache:
                        run = run_numpy_probe(
                            features=feature_cache[profile],
                            target=target,
                            subjects=dataset.subject_id,
                            split=split,
                            protocol=protocol_name,
                            profile=profile,
                            seed=seed,
                            runtime=runtime,
                        )
                    else:
                        run = run_torch_eeg_profile(
                            x=dataset.x,
                            target=target,
                            subjects=dataset.subject_id,
                            split=split,
                            protocol=protocol_name,
                            profile=profile,
                            seed=seed,
                            runtime=runtime,
                            cbramod_checkpoint=cbramod_checkpoint,
                            eegpt_checkpoint=eegpt_checkpoint,
                            allow_cbramod_download=allow_cbramod_download,
                        )
                    run["duration_seconds"] = float(time.time() - start)
                    if "predictions" in run:
                        if predictions_dir:
                            run["prediction_path"] = write_prediction_npz(
                                run.pop("predictions"),
                                Path(predictions_dir) / protocol_name / profile / f"seed_{seed}.npz",
                                dataset=dataset,
                                split=split,
                                target=target,
                            )
                        else:
                            run.pop("predictions")
                    if "embeddings" in run:
                        if embeddings_dir:
                            run["embedding_path"] = write_eeg_embedding_npz(
                                run.pop("embeddings"),
                                Path(embeddings_dir) / protocol_name / profile / f"seed_{seed}.npz",
                                dataset=dataset,
                                profile=profile,
                                protocol=protocol_name,
                                seed=seed,
                                split=split,
                                train_supervision=_embedding_supervision_for_profile(profile),
                                source_prediction_path=run.get("prediction_path"),
                            )
                        else:
                            run.pop("embeddings")
                except Exception as exc:
                    run = {
                        "protocol": protocol_name,
                        "profile": profile,
                        "seed": int(seed),
                        "status": "failed",
                        "error_type": _error_type(exc),
                        "error": str(exc),
                        "duration_seconds": float(time.time() - start),
                    }
                results.append(run)

    paired_summary = summarize_paired_results(results)
    gate_summary = full_finetune_gate(paired_summary) if full_gate else {"enabled": False}
    output = {
        "stage": 34,
        "task": "eeg_encoder_matrix",
        "target_label": target_label,
        "data": {
            "data_root": str(data_root),
            "index_path": None if index_path is None else str(index_path),
            "row_count": int(dataset.row_count),
            "x_shape": [int(value) for value in dataset.x.shape],
            "sampling_rate_hz": 200,
            "window_seconds": 10,
            "channel_count": int(dataset.channel_count),
        },
        "protocols": list(protocols),
        "profiles": list(profile_list),
        "seeds": [int(seed) for seed in seeds],
        "runtime": runtime.__dict__,
        "cbramod_acquisition": cbramod_acquisition_plan(
            cbramod_checkpoint or "outputs/checkpoints/cbramod-pretrained"
        ).to_dict(),
        "profile_errors": profile_errors,
        "run_count": len(results),
        "results": results,
        "paired_summary": paired_summary,
        "full_finetune_gate": gate_summary,
    }
    if out_json:
        write_json(output, out_json)
    if out_md:
        write_matrix_markdown(output, out_md)
    return output


def run_preflight(
    *,
    data_root: Path | str = DEFAULT_DATA_ROOT,
    splits_root: Path | str = DEFAULT_SPLITS_ROOT,
    index_path: Path | str | None = DEFAULT_INDEX_PATH,
    protocols: tuple[str, ...] = DEFAULT_PROTOCOLS,
    target_label: str = "fatigue",
    eegpt_frozen_embeddings: Path | str | None = DEFAULT_EEGPT_FROZEN_EMBEDDINGS,
    cbramod_checkpoint: Path | str | None = None,
    out_json: Path | str | None = None,
    out_md: Path | str | None = None,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    dataset: EEGAlignedDataset | None = None
    split_report: dict[str, Any] = {}
    try:
        dataset = load_eeg_aligned_dataset(data_root=data_root, index_path=index_path)
        _ = target_array(dataset.y, target_label)
    except Exception as exc:
        errors.append({"stage": "load_dataset", "error_type": _error_type(exc), "error": str(exc)})
    if dataset is not None:
        try:
            protocols_loaded = load_split_protocols(splits_root, protocols, row_count=dataset.row_count)
            split_report = {
                name: {
                    "source_root": str(split.source_root),
                    "pretrain": int(len(split.pretrain)),
                    "finetune": int(len(split.finetune)),
                    "train": int(len(split.train)),
                    "val": int(len(split.val)),
                    "test": int(len(split.test)),
                    "index_overlap": _split_overlap_report(split),
                }
                for name, split in protocols_loaded.items()
            }
        except Exception as exc:
            errors.append({"stage": "load_splits", "error_type": _error_type(exc), "error": str(exc)})
        if eegpt_frozen_embeddings:
            try:
                embeddings = load_frozen_eeg_embeddings(eegpt_frozen_embeddings, expected_sample_id=dataset.sample_id)
                frozen_report = {
                    "path": str(eegpt_frozen_embeddings),
                    "shape": [int(value) for value in embeddings.shape],
                    "nan_count": int(np.isnan(embeddings).sum()),
                }
            except Exception as exc:
                frozen_report = {"path": str(eegpt_frozen_embeddings), "error_type": _error_type(exc), "error": str(exc)}
        else:
            frozen_report = {"path": None, "error": "not configured"}
    else:
        frozen_report = {"path": str(eegpt_frozen_embeddings) if eegpt_frozen_embeddings else None, "error": "dataset unavailable"}
    result = {
        "stage": 34,
        "task": "eeg_encoder_matrix_preflight",
        "ok": not errors,
        "errors": errors,
        "data": None
        if dataset is None
        else {
            "data_root": str(data_root),
            "index_path": None if index_path is None else str(index_path),
            "row_count": int(dataset.row_count),
            "x_shape": [int(value) for value in dataset.x.shape],
            "y_shape": [int(value) for value in dataset.y.shape],
            "subject_count": int(len(np.unique(dataset.subject_id))),
            "day_count": int(len(np.unique(dataset.day_id))),
            "target_label": target_label,
            "label_names": list(dataset.label_names),
        },
        "protocols": split_report,
        "eegpt_frozen_baseline": frozen_report,
        "cbramod_acquisition": cbramod_acquisition_plan(
            cbramod_checkpoint or "outputs/checkpoints/cbramod-pretrained"
        ).to_dict(),
    }
    if out_json:
        write_json(result, out_json)
    if out_md:
        write_preflight_markdown(result, out_md)
    return result


def target_array(y: np.ndarray, target_label: str) -> np.ndarray:
    labels = np.asarray(y)
    if labels.ndim == 1:
        if target_label not in {"target", "fatigue"}:
            raise ValueError(f"1D y supports target/fatigue only, got {target_label}")
        return labels.astype(np.float32)
    if labels.ndim != 2:
        raise ValueError(f"expected y shape [rows, labels], got {labels.shape}")
    try:
        index = LABEL_NAMES.index(target_label)
    except ValueError as exc:
        raise ValueError(f"unknown target label: {target_label}") from exc
    if index >= labels.shape[1]:
        raise ValueError(f"target label {target_label!r} is not available in y shape {labels.shape}")
    return labels[:, index].astype(np.float32)


def compute_de_features(
    x: np.ndarray,
    *,
    sample_rate_hz: float = 200.0,
    seconds_per_window: int = 10,
    bands_hz: tuple[tuple[float, float], ...] = DE_BANDS_HZ,
    batch_size: int = 256,
    eps: float = 1e-8,
) -> np.ndarray:
    values = np.asarray(x)
    if values.ndim != 3:
        raise ValueError(f"expected X shape [rows, samples, channels], got {values.shape}")
    row_count, sample_count, channel_count = values.shape
    second_samples = int(round(float(sample_rate_hz)))
    expected_samples = int(seconds_per_window) * second_samples
    if sample_count != expected_samples:
        raise ValueError(f"expected {expected_samples} samples for {seconds_per_window}s at {sample_rate_hz}Hz, got {sample_count}")
    features = np.zeros((row_count, channel_count * len(bands_hz)), dtype=np.float32)
    for start in range(0, row_count, max(1, int(batch_size))):
        stop = min(row_count, start + max(1, int(batch_size)))
        batch = np.asarray(values[start:stop], dtype=np.float32)
        batch_features: list[np.ndarray] = []
        for low_hz, high_hz in bands_hz:
            filtered = _fft_band_limited_signal(batch, sample_rate_hz=sample_rate_hz, low_hz=low_hz, high_hz=high_hz)
            per_second = filtered.reshape(filtered.shape[0], seconds_per_window, second_samples, channel_count)
            variance = np.var(per_second, axis=2)
            de = 0.5 * np.log(2.0 * np.pi * np.e * np.maximum(variance, eps))
            batch_features.append(de.mean(axis=1))
        features[start:stop] = np.concatenate(batch_features, axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("DE feature matrix contains NaN or infinite values")
    return features


def load_or_compute_de_features(
    x: np.ndarray,
    *,
    cache_path: Path | str | None,
    sample_rate_hz: float,
    seconds_per_window: int,
    batch_size: int,
) -> np.ndarray:
    if cache_path is not None:
        path = Path(cache_path)
        if path.is_file():
            with np.load(path, allow_pickle=True) as loaded:
                features = loaded["features"].astype(np.float32)
                if features.shape[0] == x.shape[0]:
                    return features
    features = compute_de_features(
        x,
        sample_rate_hz=sample_rate_hz,
        seconds_per_window=seconds_per_window,
        batch_size=batch_size,
    )
    if cache_path is not None:
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=features,
            profile=np.array(["eeg_de_5band_1s_avg_v1"], dtype=object),
            bands_hz=np.asarray(DE_BANDS_HZ, dtype=np.float32),
            sample_rate_hz=np.asarray([sample_rate_hz], dtype=np.float32),
        )
    return features


def run_numpy_probe(
    *,
    features: np.ndarray,
    target: np.ndarray,
    subjects: np.ndarray,
    split: SplitProtocol,
    protocol: str,
    profile: str,
    seed: int,
    runtime: MatrixRuntime,
) -> dict[str, Any]:
    model, audit = fit_numpy_mlp(
        features=features,
        target=target,
        train_idx=split.train,
        val_idx=split.val,
        seed=seed,
        runtime=runtime,
    )
    predictions = {
        "train": predict_numpy_mlp(model, features[split.train]),
        "val": predict_numpy_mlp(model, features[split.val]),
        "test": predict_numpy_mlp(model, features[split.test]),
    }
    result = {
        "protocol": protocol,
        "profile": profile,
        "seed": int(seed),
        "status": "ok",
        "backend": "numpy_mlp_probe",
        "feature_dim": int(features.shape[1]),
        "split_counts": _split_counts(split),
        "train": _metric_aliases(evaluate_regression_with_centered(target[split.train], predictions["train"], subjects[split.train])),
        "val": _metric_aliases(evaluate_regression_with_centered(target[split.val], predictions["val"], subjects[split.val])),
        "test": _metric_aliases(evaluate_regression_with_centered(target[split.test], predictions["test"], subjects[split.test])),
        "train_audit": audit,
        "predictions": predictions,
    }
    if profile == "eegpt_frozen_v1":
        if int(features.shape[1]) == 256:
            result["embeddings"] = _as_existing_256d_embeddings(features)
    else:
        result["embeddings"] = extract_numpy_mlp_embeddings(model, features)
    return result


def fit_numpy_mlp(
    *,
    features: np.ndarray,
    target: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
    runtime: MatrixRuntime,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(target, dtype=np.float32).reshape(-1)
    x_mean = x[train_idx].mean(axis=0, keepdims=True)
    x_std = x[train_idx].std(axis=0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    y_mean = float(y[train_idx].mean())
    y_std = float(y[train_idx].std()) or 1.0
    train_x = (x[train_idx] - x_mean) / x_std
    val_x = (x[val_idx] - x_mean) / x_std
    train_y = ((y[train_idx] - y_mean) / y_std).reshape(-1, 1)
    val_y = ((y[val_idx] - y_mean) / y_std).reshape(-1, 1)
    input_dim = train_x.shape[1]
    hidden_dim = int(runtime.hidden_dim)
    weights1 = rng.normal(0.0, 0.05, size=(input_dim, hidden_dim)).astype(np.float32)
    bias1 = np.zeros((1, hidden_dim), dtype=np.float32)
    embedding_dim = 256
    weights_emb = rng.normal(0.0, 0.05, size=(hidden_dim, embedding_dim)).astype(np.float32)
    bias_emb = np.zeros((1, embedding_dim), dtype=np.float32)
    weights2 = rng.normal(0.0, 0.05, size=(embedding_dim, 1)).astype(np.float32)
    bias2 = np.zeros((1, 1), dtype=np.float32)
    optimizer = _AdamWState([weights1, bias1, weights_emb, bias_emb, weights2, bias2])
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    initial_train_loss = None
    batch_size = max(1, int(runtime.batch_size))
    epoch_audits: list[dict[str, Any]] = []
    for epoch in range(max(1, int(runtime.epochs))):
        order = rng.permutation(len(train_y))
        losses = []
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            loss, grads = _numpy_mlp_loss_and_grads(
                train_x[rows],
                train_y[rows],
                weights1,
                bias1,
                weights_emb,
                bias_emb,
                weights2,
                bias2,
            )
            optimizer.step(
                params=[weights1, bias1, weights_emb, bias_emb, weights2, bias2],
                grads=grads,
                lr=float(runtime.learning_rate),
                weight_decay=float(runtime.weight_decay),
            )
            losses.append(float(loss))
        train_loss = float(np.mean(losses)) if losses else math.nan
        if initial_train_loss is None:
            initial_train_loss = train_loss
        val_loss = _numpy_mlp_loss(val_x, val_y, weights1, bias1, weights_emb, bias_emb, weights2, bias2)
        epoch_audits.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
        )
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            stale = 0
            best_state = [value.copy() for value in (weights1, bias1, weights_emb, bias_emb, weights2, bias2)]
        else:
            stale += 1
            if stale >= int(runtime.patience):
                break
    if best_state is not None:
        weights1[:], bias1[:], weights_emb[:], bias_emb[:], weights2[:], bias2[:] = best_state
    model = {
        "weights1": weights1,
        "bias1": bias1,
        "weights_emb": weights_emb,
        "bias_emb": bias_emb,
        "weights2": weights2,
        "bias2": bias2,
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mean": y_mean,
        "y_std": y_std,
    }
    audit = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "initial_train_loss": float(initial_train_loss) if initial_train_loss is not None else None,
        "normalization": "train_only",
        "embedding_dim": int(embedding_dim),
        "embedding_source": "supervised_de_mlp_penultimate_projection",
        "optimizer": "numpy_adamw",
        "train_count": int(len(train_idx)),
        "val_count": int(len(val_idx)),
        "epoch_count": len(epoch_audits),
        "history": epoch_audits,
    }
    return model, audit


def predict_numpy_mlp(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    embedding = extract_numpy_mlp_embeddings(model, features)
    pred = embedding @ model["weights2"] + model["bias2"]
    return (pred.reshape(-1) * float(model["y_std"]) + float(model["y_mean"])).astype(np.float32)


def extract_numpy_mlp_embeddings(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    x = (np.asarray(features, dtype=np.float32) - model["x_mean"]) / model["x_std"]
    hidden = np.maximum(0.0, x @ model["weights1"] + model["bias1"])
    embedding = np.maximum(0.0, hidden @ model["weights_emb"] + model["bias_emb"])
    if embedding.shape[1] != 256:
        raise ValueError(f"expected DE MLP embedding dim 256, got {embedding.shape}")
    if not np.isfinite(embedding).all():
        raise ValueError("DE MLP embedding contains NaN or infinite values")
    return embedding.astype(np.float32)


def _as_existing_256d_embeddings(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 256:
        raise ValueError(f"expected existing frozen EEG embedding shape [N,256], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("existing frozen EEG embedding contains NaN or infinite values")
    return values.copy()


def load_frozen_eeg_embeddings(
    path: Path | str,
    *,
    expected_sample_id: np.ndarray | None = None,
    row_filter: np.ndarray | None = None,
) -> np.ndarray:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"missing frozen EEG embeddings: {source}")
    with np.load(source, allow_pickle=True) as loaded:
        key = "eeg_emb" if "eeg_emb" in loaded.files else "embedding"
        if key not in loaded.files:
            raise ValueError(f"{source} does not contain eeg_emb or embedding")
        embedding = loaded[key].astype(np.float32)
        if row_filter is not None:
            embedding = embedding[row_filter]
        if expected_sample_id is not None and "sample_id" in loaded.files:
            sample_id = loaded["sample_id"].astype(str)
            if row_filter is not None:
                sample_id = sample_id[row_filter]
            if not np.array_equal(sample_id, expected_sample_id.astype(str)):
                raise ValueError("frozen EEG sample_id order does not match canonical index")
    if embedding.ndim != 2:
        raise ValueError(f"expected frozen EEG embedding matrix, got {embedding.shape}")
    if not np.isfinite(embedding).all():
        raise ValueError("frozen EEG embedding contains NaN or infinite values")
    return embedding


def run_torch_eeg_profile(
    *,
    x: np.ndarray,
    target: np.ndarray,
    subjects: np.ndarray,
    split: SplitProtocol,
    protocol: str,
    profile: str,
    seed: int,
    runtime: MatrixRuntime,
    cbramod_checkpoint: Path | str | None,
    eegpt_checkpoint: Path | str | None,
    allow_cbramod_download: bool,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on server runtime
        raise RuntimeError("PyTorch is required for EEGPT/CBraMod fine-tuning profiles") from exc
    torch.set_num_threads(max(1, int(runtime.torch_threads)))
    if runtime.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    try:
        return _run_torch_eeg_profile_once(
            x=x,
            target=target,
            subjects=subjects,
            split=split,
            protocol=protocol,
            profile=profile,
            seed=seed,
            runtime=runtime,
            cbramod_checkpoint=cbramod_checkpoint,
            eegpt_checkpoint=eegpt_checkpoint,
            allow_cbramod_download=allow_cbramod_download,
            torch=torch,
            batch_size=runtime.batch_size,
            gradient_accumulation_steps=1,
        )
    except RuntimeError as exc:  # pragma: no cover - depends on server runtime
        if "out of memory" not in str(exc).lower() and "oom" not in str(exc).lower():
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        fallback_batch = max(1, int(runtime.fallback_batch_size))
        accumulation = max(1, int(math.ceil(float(runtime.batch_size) / float(fallback_batch))))
        result = _run_torch_eeg_profile_once(
            x=x,
            target=target,
            subjects=subjects,
            split=split,
            protocol=protocol,
            profile=profile,
            seed=seed,
            runtime=runtime,
            cbramod_checkpoint=cbramod_checkpoint,
            eegpt_checkpoint=eegpt_checkpoint,
            allow_cbramod_download=allow_cbramod_download,
            torch=torch,
            batch_size=fallback_batch,
            gradient_accumulation_steps=accumulation,
        )
        result["oom_recovered"] = True
        result["effective_batch_size"] = int(fallback_batch * accumulation)
        return result


def _run_torch_eeg_profile_once(
    *,
    x: np.ndarray,
    target: np.ndarray,
    subjects: np.ndarray,
    split: SplitProtocol,
    protocol: str,
    profile: str,
    seed: int,
    runtime: MatrixRuntime,
    cbramod_checkpoint: Path | str | None,
    eegpt_checkpoint: Path | str | None,
    allow_cbramod_download: bool,
    torch: Any,
    batch_size: int,
    gradient_accumulation_steps: int,
) -> dict[str, Any]:  # pragma: no cover - depends on server runtime
    _seed_everything(seed, torch=torch)
    device = torch.device(runtime.device)
    encoder, backend_report = build_torch_encoder(
        profile=profile,
        n_channels=int(x.shape[2]),
        n_times=int(x.shape[1]),
        sample_rate_hz=200.0,
        cbramod_checkpoint=cbramod_checkpoint,
        eegpt_checkpoint=eegpt_checkpoint,
        allow_cbramod_download=allow_cbramod_download,
        torch=torch,
    )
    trainability = configure_encoder_trainability(
        encoder,
        _strategy_for_profile(profile),
        last_n_blocks=int(runtime.partial_last_n_blocks),
    )
    model = _TorchEEGRegressor(encoder=encoder, dropout=float(runtime.dropout)).to(device)
    channel_mean, channel_std = _fit_channel_normalization(x, split.train)
    y_mean = float(target[split.train].mean())
    y_std = float(target[split.train].std()) or 1.0
    optimizer = _torch_optimizer_for_profile(model, profile, runtime, torch)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(runtime.amp and device.type == "cuda"))
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    rng = np.random.default_rng(seed)
    epoch_audits: list[dict[str, Any]] = []
    for epoch in range(max(1, int(runtime.epochs))):
        model.train()
        order = rng.permutation(split.train)
        train_losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        batch_starts = list(range(0, len(order), max(1, int(batch_size))))
        for offset, start in enumerate(batch_starts):
            batch_idx = order[start : start + max(1, int(batch_size))]
            batch_x = _torch_eeg_batch(x, batch_idx, channel_mean, channel_std, torch=torch, device=device)
            batch_y = torch.as_tensor((target[batch_idx] - y_mean) / y_std, dtype=torch.float32, device=device)
            with torch.cuda.amp.autocast(enabled=bool(runtime.amp and device.type == "cuda")):
                prediction = model(batch_x)
                loss = torch.mean((prediction - batch_y) ** 2) / max(1, int(gradient_accumulation_steps))
            scaler.scale(loss).backward()
            is_accumulation_boundary = (offset + 1) % max(1, int(gradient_accumulation_steps)) == 0
            is_last_batch = offset + 1 == len(batch_starts)
            if is_accumulation_boundary or is_last_batch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(runtime.grad_clip))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_losses.append(float(loss.detach().cpu().item()) * max(1, int(gradient_accumulation_steps)))
        val_pred = _torch_predict(model, x, split.val, channel_mean, channel_std, y_mean, y_std, torch=torch, device=device)
        val_metrics = evaluate_regression_with_centered(target[split.val], val_pred, subjects[split.val])
        val_loss = float(val_metrics["rmse"] or float("inf"))
        audit = {
            "epoch": int(epoch + 1),
            "train_loss": float(np.mean(train_losses)) if train_losses else math.nan,
            "val_rmse": val_loss,
        }
        epoch_audits.append(audit)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(runtime.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    predictions = {
        "train": _torch_predict(model, x, split.train, channel_mean, channel_std, y_mean, y_std, torch=torch, device=device),
        "val": _torch_predict(model, x, split.val, channel_mean, channel_std, y_mean, y_std, torch=torch, device=device),
        "test": _torch_predict(model, x, split.test, channel_mean, channel_std, y_mean, y_std, torch=torch, device=device),
    }
    embeddings = _torch_extract_embeddings(
        model,
        x,
        np.arange(int(x.shape[0]), dtype=np.int64),
        channel_mean,
        channel_std,
        torch=torch,
        device=device,
    )
    return {
        "protocol": protocol,
        "profile": profile,
        "seed": int(seed),
        "status": "ok",
        "backend": backend_report["backend"],
        "split_counts": _split_counts(split),
        "train": _metric_aliases(evaluate_regression_with_centered(target[split.train], predictions["train"], subjects[split.train])),
        "val": _metric_aliases(evaluate_regression_with_centered(target[split.val], predictions["val"], subjects[split.val])),
        "test": _metric_aliases(evaluate_regression_with_centered(target[split.test], predictions["test"], subjects[split.test])),
        "trainability": trainability,
        "backend_report": backend_report,
        "train_audit": {
            "best_epoch": int(best_epoch),
            "best_val_rmse": float(best_val),
            "normalization": "train_only_channel_mean_std",
            "embedding_dim": 256,
            "embedding_source": "supervised_encoder_pooled_projection",
            "epoch_count": len(epoch_audits),
            "batch_size": int(batch_size),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "partial_last_n_blocks": int(runtime.partial_last_n_blocks),
            "history": epoch_audits,
        },
        "predictions": predictions,
        "embeddings": embeddings,
    }


def build_torch_encoder(
    *,
    profile: str,
    n_channels: int,
    n_times: int,
    sample_rate_hz: float,
    cbramod_checkpoint: Path | str | None,
    eegpt_checkpoint: Path | str | None,
    allow_cbramod_download: bool,
    torch: Any,
) -> tuple[Any, dict[str, Any]]:  # pragma: no cover - depends on server runtime
    if profile.startswith("cbramod"):
        try:
            from braindecode.models import CBraMod
        except ImportError as exc:
            raise RuntimeError("CBraMod requires braindecode[hub]; install it before running cbramod profiles") from exc
        checkpoint = Path(cbramod_checkpoint) if cbramod_checkpoint else None
        if checkpoint is None and not allow_cbramod_download:
            raise RuntimeError(
                "CBraMod requires a local --cbramod-checkpoint/cache unless --allow-cbramod-download is set; "
                "prepare it with: huggingface-cli download braindecode/cbramod-pretrained --local-dir outputs/checkpoints/cbramod-pretrained"
            )
        local_only = not bool(allow_cbramod_download)
        if hasattr(CBraMod, "from_pretrained"):
            kwargs = {"return_encoder_output": True}
            if checkpoint is not None:
                repo_or_path = str(checkpoint)
                kwargs["local_files_only"] = True
            else:
                repo_or_path = "braindecode/cbramod-pretrained"
                kwargs["local_files_only"] = local_only
            try:
                model = CBraMod.from_pretrained(repo_or_path, **kwargs)
            except TypeError:
                kwargs.pop("local_files_only", None)
                model = CBraMod.from_pretrained(repo_or_path, **kwargs)
            return model, {
                "backend": "braindecode_cbramod",
                "source": repo_or_path,
                "allow_download": bool(allow_cbramod_download),
                "n_channels": int(n_channels),
                "n_times": int(n_times),
                "sample_rate_hz": float(sample_rate_hz),
            }
        raise RuntimeError("installed braindecode does not expose CBraMod.from_pretrained")
    if profile.startswith("eegpt"):
        try:
            from braindecode.models import EEGPT
        except ImportError as exc:
            raise RuntimeError("EEGPT fine-tuning requires braindecode with EEGPT") from exc
        signature = inspect.signature(EEGPT)
        kwargs = {
            "n_outputs": 1,
            "n_chans": int(n_channels),
            "n_times": int(n_times),
            "sfreq": float(sample_rate_hz),
        }
        if "return_encoder_output" in signature.parameters:
            kwargs["return_encoder_output"] = True
        model = EEGPT(**kwargs)
        report = {
            "backend": "braindecode_eegpt",
            "source": None if eegpt_checkpoint is None else str(eegpt_checkpoint),
            "n_channels": int(n_channels),
            "n_times": int(n_times),
            "sample_rate_hz": float(sample_rate_hz),
        }
        if eegpt_checkpoint:
            load_report = _load_torch_state_if_present(model, Path(eegpt_checkpoint), torch=torch)
            report["load_report"] = load_report
        return model, report
    raise ValueError(f"unsupported torch EEG profile: {profile}")


def configure_encoder_trainability(encoder: Any, strategy: str, *, last_n_blocks: int = 2) -> dict[str, Any]:
    named_params = list(encoder.named_parameters())
    if strategy == "frozen":
        for _, param in named_params:
            param.requires_grad = False
    elif strategy == "full":
        for _, param in named_params:
            param.requires_grad = True
    elif strategy == "partial":
        block_indices = _block_indices(named_params)
        keep_blocks = set(sorted(block_indices)[-max(0, int(last_n_blocks)) :])
        for name, param in named_params:
            lower = name.lower()
            block_index = _block_index_from_name(name)
            param.requires_grad = (
                block_index in keep_blocks
                or "norm" in lower
                or "ln" in lower
                or "projection" in lower
                or "proj" in lower
                or "head" in lower
                or "classifier" in lower
            )
    else:
        raise ValueError(f"unsupported fine-tune strategy: {strategy}")
    trainable = [name for name, param in named_params if bool(param.requires_grad)]
    frozen = [name for name, param in named_params if not bool(param.requires_grad)]
    return {
        "strategy": strategy,
        "parameter_count": int(len(named_params)),
        "trainable_count": int(len(trainable)),
        "frozen_count": int(len(frozen)),
        "trainable_preview": trainable[:20],
        "last_n_blocks": int(last_n_blocks) if strategy == "partial" else None,
    }


def summarize_paired_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in results if row.get("status") == "ok"]
    summary: dict[str, Any] = {"by_profile": {}, "paired_deltas": {}}
    for profile in sorted({row["profile"] for row in ok}):
        rows = [row for row in ok if row["profile"] == profile]
        summary["by_profile"][profile] = {
            "run_count": len(rows),
            "test_rmse_mean": _mean_metric(rows, "test", "rmse"),
            "test_raw_r_mean": _mean_metric(rows, "test", "raw_r"),
            "test_centered_r_mean": _mean_metric(rows, "test", "within_subject_centered_r"),
        }
    for partial, frozen in FROZEN_PROFILE_BY_PARTIAL.items():
        deltas = []
        for row in ok:
            if row["profile"] != partial:
                continue
            match = next(
                (
                    ref
                    for ref in ok
                    if ref["profile"] == frozen
                    and ref["protocol"] == row["protocol"]
                    and int(ref["seed"]) == int(row["seed"])
                ),
                None,
            )
            if match is None:
                continue
            deltas.append(_paired_delta(row, match))
        summary["paired_deltas"][partial] = _summarize_deltas(deltas)
    return summary


def full_finetune_gate(paired_summary: dict[str, Any]) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    for partial, full in FULL_PROFILE_BY_PARTIAL.items():
        delta_summary = paired_summary.get("paired_deltas", {}).get(partial, {})
        protocol_decisions = {}
        for protocol, row in delta_summary.get("by_protocol", {}).items():
            if protocol not in MAIN_PROTOCOLS:
                continue
            passes = (
                (
                    row.get("val_delta_rmse_mean") is not None
                    and float(row["val_delta_rmse_mean"]) <= -0.010
                )
                or (
                    row.get("val_delta_raw_r_mean") is not None
                    and float(row["val_delta_raw_r_mean"]) >= 0.020
                )
            ) and (
                row.get("val_delta_centered_r_mean") is None
                or float(row["val_delta_centered_r_mean"]) >= -0.020
            )
            protocol_decisions[protocol] = {
                "passes": bool(passes),
                "full_profile": full,
                **row,
            }
        decisions[partial] = {
            "passes_any_main_protocol": any(item["passes"] for item in protocol_decisions.values()),
            "full_profile": full,
            "protocols": protocol_decisions,
        }
    return {"enabled": True, "decisions": decisions}


def write_prediction_npz(
    predictions: dict[str, np.ndarray],
    path: Path,
    *,
    dataset: EEGAlignedDataset,
    split: SplitProtocol,
    target: np.ndarray,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train_index=split.train,
        val_index=split.val,
        test_index=split.test,
        train_prediction=predictions["train"],
        val_prediction=predictions["val"],
        test_prediction=predictions["test"],
        target=target,
        sample_id=dataset.sample_id,
        subject_id=dataset.subject_id,
        day_id=dataset.day_id,
    )
    return str(path)


def write_eeg_embedding_npz(
    embeddings: np.ndarray,
    path: Path,
    *,
    dataset: EEGAlignedDataset,
    profile: str,
    protocol: str,
    seed: int,
    split: SplitProtocol,
    train_supervision: str,
    source_prediction_path: str | None = None,
) -> str:
    values = np.asarray(embeddings, dtype=np.float32)
    expected_shape = (dataset.row_count, 256)
    if values.shape != expected_shape:
        raise ValueError(f"expected EEG embeddings shape {expected_shape}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("EEG embeddings contain NaN or infinite values")
    path.parent.mkdir(parents=True, exist_ok=True)
    modality_mask = np.zeros((dataset.row_count, 4), dtype=np.int8)
    modality_mask[:, 0] = 1
    np.savez_compressed(
        path,
        sample_id=dataset.sample_id,
        subject_id=dataset.subject_id,
        day_id=dataset.day_id,
        eeg_emb=values,
        eeg_mask=np.ones((dataset.row_count,), dtype=np.int8),
        modality_mask=modality_mask,
        encoder_profile=np.asarray([profile] * dataset.row_count, dtype=object),
        encoder_version=np.asarray(["eeg_encoder_256d_supervised_v1"] * dataset.row_count, dtype=object),
        protocol=np.asarray([protocol], dtype=object),
        seed=np.asarray([int(seed)], dtype=np.int64),
        train_index=split.train,
        val_index=split.val,
        test_index=split.test,
        train_supervision=np.asarray([train_supervision], dtype=object),
        source_prediction_npz=np.asarray(["" if source_prediction_path is None else source_prediction_path], dtype=object),
    )
    return str(path)


def write_json(value: dict[str, Any], path: Path | str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_preflight_markdown(result: dict[str, Any], path: Path | str) -> None:
    lines = ["# EEG Encoder Matrix Preflight", "", f"ok: `{result['ok']}`", ""]
    data = result.get("data") or {}
    if data:
        lines.extend([
            "## Data",
            "",
            f"- rows: `{data['row_count']}`",
            f"- X shape: `{data['x_shape']}`",
            f"- target: `{data['target_label']}`",
            "",
        ])
    lines.extend(["## Protocols", "", "| protocol | train | val | test | overlap ok |", "| --- | ---: | ---: | ---: | --- |"])
    for name, row in result.get("protocols", {}).items():
        overlap = row["index_overlap"]
        ok = not overlap["train_val"] and not overlap["train_test"] and not overlap["val_test"]
        lines.append(f"| {name} | {row['train']} | {row['val']} | {row['test']} | {ok} |")
    lines.extend(["", "## CBraMod Acquisition", ""])
    plan = result["cbramod_acquisition"]
    lines.extend([f"- primary: `{plan['primary_source']}`", f"- install: `{plan['install_command']}`", f"- download: `{plan['download_command']}`", ""])
    if result.get("errors"):
        lines.extend(["## Errors", ""])
        for error in result["errors"]:
            lines.append(f"- `{error['stage']}` `{error['error_type']}`: {error['error']}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_matrix_markdown(output: dict[str, Any], path: Path | str) -> None:
    lines = [
        "# EEG Encoder Matrix",
        "",
        f"target_label: `{output['target_label']}`",
        f"run_count: `{output['run_count']}`",
        f"protocols: `{', '.join(output['protocols'])}`",
        "",
        "| protocol | profile | seed | status | test RMSE | test raw r | test centered r | val RMSE |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in output["results"]:
        test = row.get("test", {})
        val = row.get("val", {})
        lines.append(
            f"| {row.get('protocol')} | {row.get('profile')} | {row.get('seed', '')} | {row.get('status')} | "
            f"{_fmt(test.get('rmse'))} | {_fmt(test.get('raw_r'))} | {_fmt(test.get('within_subject_centered_r'))} | {_fmt(val.get('rmse'))} |"
        )
    profile_summary = output.get("paired_summary", {}).get("by_profile", {})
    if profile_summary:
        lines.extend([
            "",
            "## Profile Means",
            "",
            "| profile | runs | test RMSE mean | test raw r mean | test centered r mean |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for profile, row in sorted(profile_summary.items()):
            lines.append(
                f"| {profile} | {row.get('run_count', 0)} | {_fmt(row.get('test_rmse_mean'))} | "
                f"{_fmt(row.get('test_raw_r_mean'))} | {_fmt(row.get('test_centered_r_mean'))} |"
            )
    delta_summary = output.get("paired_summary", {}).get("paired_deltas", {})
    if delta_summary:
        lines.extend([
            "",
            "## Paired Deltas",
            "",
            "| profile | protocol | pairs | val dRMSE | val draw r | val dcentered r | test dRMSE | test draw r | test dcentered r |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for profile, summary in sorted(delta_summary.items()):
            for protocol, row in sorted(summary.get("by_protocol", {}).items()):
                lines.append(
                    f"| {profile} | {protocol} | {row.get('pair_count', 0)} | {_fmt(row.get('val_delta_rmse_mean'))} | "
                    f"{_fmt(row.get('val_delta_raw_r_mean'))} | {_fmt(row.get('val_delta_centered_r_mean'))} | "
                    f"{_fmt(row.get('test_delta_rmse_mean'))} | {_fmt(row.get('test_delta_raw_r_mean'))} | "
                    f"{_fmt(row.get('test_delta_centered_r_mean'))} |"
                )
    gate = output.get("full_finetune_gate", {})
    if gate.get("enabled"):
        lines.extend([
            "",
            "## Full Fine-Tune Gate",
            "",
            "| partial profile | passes any main protocol | full profile |",
            "| --- | --- | --- |",
        ])
        for profile, row in sorted(gate.get("decisions", {}).items()):
            lines.append(f"| {profile} | {bool(row.get('passes_any_main_protocol'))} | {row.get('full_profile')} |")
    lines.extend(["", "## CBraMod Acquisition", ""])
    plan = output["cbramod_acquisition"]
    lines.extend([f"- Install: `{plan['install_command']}`", f"- Download: `{plan['download_command']}`", f"- Use local checkpoint: `{plan['local_checkpoint_argument']}`", ""])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


class _AdamWState:
    def __init__(self, params: list[np.ndarray], *, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.m = [np.zeros_like(param, dtype=np.float32) for param in params]
        self.v = [np.zeros_like(param, dtype=np.float32) for param in params]
        self.t = 0
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)

    def step(self, *, params: list[np.ndarray], grads: list[np.ndarray], lr: float, weight_decay: float) -> None:
        self.t += 1
        for index, (param, grad) in enumerate(zip(params, grads)):
            self.m[index] = self.beta1 * self.m[index] + (1.0 - self.beta1) * grad
            self.v[index] = self.beta2 * self.v[index] + (1.0 - self.beta2) * (grad * grad)
            m_hat = self.m[index] / (1.0 - self.beta1**self.t)
            v_hat = self.v[index] / (1.0 - self.beta2**self.t)
            param *= 1.0 - lr * weight_decay
            param -= lr * m_hat / (np.sqrt(v_hat) + self.eps)


def _fft_band_limited_signal(batch: np.ndarray, *, sample_rate_hz: float, low_hz: float, high_hz: float) -> np.ndarray:
    freqs = np.fft.rfftfreq(batch.shape[1], d=1.0 / float(sample_rate_hz))
    spectrum = np.fft.rfft(batch, axis=1)
    mask = (freqs >= float(low_hz)) & (freqs < float(high_hz))
    spectrum[:, ~mask, :] = 0.0
    return np.fft.irfft(spectrum, n=batch.shape[1], axis=1).astype(np.float32)


def _numpy_mlp_loss_and_grads(
    x: np.ndarray,
    y: np.ndarray,
    weights1: np.ndarray,
    bias1: np.ndarray,
    weights_emb: np.ndarray,
    bias_emb: np.ndarray,
    weights2: np.ndarray,
    bias2: np.ndarray,
) -> tuple[float, list[np.ndarray]]:
    z1 = x @ weights1 + bias1
    hidden = np.maximum(0.0, z1)
    z_emb = hidden @ weights_emb + bias_emb
    embedding = np.maximum(0.0, z_emb)
    pred = embedding @ weights2 + bias2
    error = pred - y
    loss = float(np.mean(error * error))
    grad_pred = (2.0 / max(1, len(y))) * error
    grad_w2 = embedding.T @ grad_pred
    grad_b2 = grad_pred.sum(axis=0, keepdims=True)
    grad_embedding = grad_pred @ weights2.T
    grad_z_emb = grad_embedding * (z_emb > 0.0)
    grad_w_emb = hidden.T @ grad_z_emb
    grad_b_emb = grad_z_emb.sum(axis=0, keepdims=True)
    grad_hidden = grad_z_emb @ weights_emb.T
    grad_z1 = grad_hidden * (z1 > 0.0)
    grad_w1 = x.T @ grad_z1
    grad_b1 = grad_z1.sum(axis=0, keepdims=True)
    return loss, [
        grad_w1.astype(np.float32),
        grad_b1.astype(np.float32),
        grad_w_emb.astype(np.float32),
        grad_b_emb.astype(np.float32),
        grad_w2.astype(np.float32),
        grad_b2.astype(np.float32),
    ]


def _numpy_mlp_loss(
    x: np.ndarray,
    y: np.ndarray,
    weights1: np.ndarray,
    bias1: np.ndarray,
    weights_emb: np.ndarray,
    bias_emb: np.ndarray,
    weights2: np.ndarray,
    bias2: np.ndarray,
) -> float:
    hidden = np.maximum(0.0, x @ weights1 + bias1)
    embedding = np.maximum(0.0, hidden @ weights_emb + bias_emb)
    pred = embedding @ weights2 + bias2
    return float(np.mean((pred - y) ** 2))


def _sample_ids_from_index(index_path: Path | str | None, row_count: int) -> np.ndarray:
    if index_path is None or not Path(index_path).is_file():
        return np.asarray([f"eeg_{idx:06d}" for idx in range(row_count)], dtype=str)
    rows = _load_jsonl(Path(index_path))
    if len(rows) != row_count:
        raise ValueError(f"index row count {len(rows)} does not match EEG rows {row_count}")
    return np.asarray([str(row.get("sample_id", f"eeg_{idx:06d}")) for idx, row in enumerate(rows)], dtype=str)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_indices(path: Path, row_count: int) -> np.ndarray:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("indices", value.get("index", value.get("rows")))
    values = np.asarray(value, dtype=np.int64).reshape(-1)
    if values.size and (int(values.min()) < 0 or int(values.max()) >= row_count):
        raise ValueError(f"{path} contains out-of-range indices")
    return values


def _validate_split_no_overlap(protocol: str, *, train: np.ndarray, val: np.ndarray, test: np.ndarray) -> None:
    train_set = set(train.tolist())
    val_set = set(val.tolist())
    test_set = set(test.tolist())
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError(f"{protocol} split has train/val/test index overlap")


def _split_overlap_report(split: SplitProtocol) -> dict[str, list[int]]:
    train = set(split.train.tolist())
    val = set(split.val.tolist())
    test = set(split.test.tolist())
    return {
        "train_val": sorted(train & val)[:20],
        "train_test": sorted(train & test)[:20],
        "val_test": sorted(val & test)[:20],
    }


def _filtered_dataset(dataset: EEGAlignedDataset, row_filter: np.ndarray) -> EEGAlignedDataset:
    if len(row_filter) == dataset.row_count:
        return dataset
    return EEGAlignedDataset(
        x_path=dataset.x_path,
        y_path=dataset.y_path,
        sub_path=dataset.sub_path,
        day_path=dataset.day_path,
        x=np.asarray(dataset.x[row_filter]),
        y=np.asarray(dataset.y[row_filter]),
        subject_id=dataset.subject_id[row_filter],
        day_id=dataset.day_id[row_filter],
        sample_id=dataset.sample_id[row_filter],
        label_names=dataset.label_names,
    )


def _cap_split_for_smoke(split: SplitProtocol, *, max_rows: int) -> SplitProtocol:
    total = max(3, int(max_rows))
    train_cap = max(1, int(round(total * 0.6)))
    val_cap = max(1, int(round(total * 0.2)))
    test_cap = max(1, total - train_cap - val_cap)
    if train_cap + val_cap + test_cap > total:
        train_cap = max(1, total - val_cap - test_cap)
    pretrain_cap = min(len(split.pretrain), max(1, train_cap // 2))
    finetune_cap = min(len(split.finetune), max(1, train_cap - pretrain_cap))
    if pretrain_cap + finetune_cap < train_cap and len(split.pretrain) > pretrain_cap:
        pretrain_cap = min(len(split.pretrain), train_cap - finetune_cap)
    pretrain = split.pretrain[:pretrain_cap]
    finetune = split.finetune[:finetune_cap]
    val = split.val[: min(len(split.val), val_cap)]
    test = split.test[: min(len(split.test), test_cap)]
    train = np.asarray(pretrain.tolist() + finetune.tolist(), dtype=np.int64)
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError(f"smoke split {split.name} has an empty train/val/test cap; increase --max-rows")
    _validate_split_no_overlap(split.name, train=train, val=val, test=test)
    return SplitProtocol(split.name, pretrain, finetune, train, val, test, split.source_root)


def _remap_protocols(protocols: dict[str, SplitProtocol], *, row_filter: np.ndarray) -> dict[str, SplitProtocol]:
    original_to_new = {int(original): new for new, original in enumerate(row_filter.tolist())}

    def remap(indices: np.ndarray) -> np.ndarray:
        return np.asarray([original_to_new[int(value)] for value in indices.tolist()], dtype=np.int64)

    return {
        name: SplitProtocol(
            name=split.name,
            pretrain=remap(split.pretrain),
            finetune=remap(split.finetune),
            train=remap(split.train),
            val=remap(split.val),
            test=remap(split.test),
            source_root=split.source_root,
        )
        for name, split in protocols.items()
    }


def _norm_subject(value: Any) -> str:
    text = str(value)
    if text.startswith("sub-"):
        return text
    try:
        return f"sub-{int(float(text)):02d}"
    except (TypeError, ValueError):
        return text


def _strategy_for_profile(profile: str) -> str:
    if profile.endswith("_frozen_v1"):
        return "frozen"
    if profile.endswith("_partial_ft_v1"):
        return "partial"
    if profile.endswith("_full_ft_v1"):
        return "full"
    raise ValueError(f"profile has no fine-tune strategy: {profile}")


def _split_counts(split: SplitProtocol) -> dict[str, int]:
    return {name: int(len(values)) for name, values in split.as_dict().items()}


def _metric_aliases(metrics: dict[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    result["pearson_r"] = result.get("raw_r")
    result["per_subject_r_mean"] = result.get("per_subject_r", {}).get("mean")
    result["per_subject_r_std"] = result.get("per_subject_r", {}).get("std")
    return result


def _skipped_result(protocol: str, profile: str, *, reason: str) -> dict[str, Any]:
    return {
        "protocol": protocol,
        "profile": profile,
        "seed": None,
        "status": "skipped",
        "error_type": "profile_unavailable",
        "error": reason,
    }


def _embedding_supervision_for_profile(profile: str) -> str:
    if profile == "eegpt_frozen_v1":
        return "label_free_frozen_eegpt_existing_256d"
    return "fatigue_supervised_train_val_selected"


def _paired_delta(row: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol": row["protocol"],
        "seed": int(row["seed"]),
        "profile": row["profile"],
        "reference_profile": ref["profile"],
        "val_delta_rmse": _subtract(row["val"].get("rmse"), ref["val"].get("rmse")),
        "val_delta_raw_r": _subtract(row["val"].get("raw_r"), ref["val"].get("raw_r")),
        "val_delta_centered_r": _subtract(row["val"].get("within_subject_centered_r"), ref["val"].get("within_subject_centered_r")),
        "test_delta_rmse": _subtract(row["test"].get("rmse"), ref["test"].get("rmse")),
        "test_delta_raw_r": _subtract(row["test"].get("raw_r"), ref["test"].get("raw_r")),
        "test_delta_centered_r": _subtract(row["test"].get("within_subject_centered_r"), ref["test"].get("within_subject_centered_r")),
    }


def _summarize_deltas(deltas: list[dict[str, Any]]) -> dict[str, Any]:
    by_protocol: dict[str, list[dict[str, Any]]] = {}
    for row in deltas:
        by_protocol.setdefault(row["protocol"], []).append(row)
    return {
        "pair_count": int(len(deltas)),
        "by_protocol": {
            protocol: {
                "pair_count": int(len(rows)),
                "val_delta_rmse_mean": _mean_values([row["val_delta_rmse"] for row in rows]),
                "val_delta_raw_r_mean": _mean_values([row["val_delta_raw_r"] for row in rows]),
                "val_delta_centered_r_mean": _mean_values([row["val_delta_centered_r"] for row in rows]),
                "test_delta_rmse_mean": _mean_values([row["test_delta_rmse"] for row in rows]),
                "test_delta_raw_r_mean": _mean_values([row["test_delta_raw_r"] for row in rows]),
                "test_delta_centered_r_mean": _mean_values([row["test_delta_centered_r"] for row in rows]),
            }
            for protocol, rows in sorted(by_protocol.items())
        },
        "pairs": deltas,
    }


def _mean_metric(rows: list[dict[str, Any]], split_name: str, metric: str) -> float | None:
    return _mean_values([row.get(split_name, {}).get(metric) for row in rows])


def _mean_values(values: list[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return None if not finite else float(np.mean(finite))


def _subtract(value: Any, reference: Any) -> float | None:
    if value is None or reference is None:
        return None
    return float(value) - float(reference)


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def _error_type(exc: Exception) -> str:
    message = str(exc).lower()
    if isinstance(exc, FileNotFoundError):
        return "source_missing"
    if "out of memory" in message or "oom" in message:
        return "oom"
    if "braindecode" in message or "torch" in message:
        return "dependency_missing"
    return "runtime_error"


def _block_indices(named_params: list[tuple[str, Any]]) -> set[int]:
    indices = {_block_index_from_name(name) for name, _ in named_params}
    return {index for index in indices if index is not None}


def _block_index_from_name(name: str) -> int | None:
    parts = name.replace("[", ".").replace("]", ".").split(".")
    for left, right in zip(parts, parts[1:]):
        if left in {"blocks", "block", "layers", "layer", "encoder_layers"}:
            try:
                return int(right)
            except ValueError:
                continue
    return None


def _fit_channel_normalization(x: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
    train = np.asarray(x[train_idx], dtype=np.float32)
    mean = train.mean(axis=(0, 1)).astype(np.float32)
    std = train.std(axis=(0, 1)).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def _torch_eeg_batch(
    x: np.ndarray,
    indices: np.ndarray,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    *,
    torch: Any,
    device: Any,
) -> Any:  # pragma: no cover
    values = (np.asarray(x[indices], dtype=np.float32) - channel_mean.reshape(1, 1, -1)) / channel_std.reshape(1, 1, -1)
    values = np.transpose(values, (0, 2, 1))
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _torch_predict(
    model: Any,
    x: np.ndarray,
    indices: np.ndarray,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    y_mean: float,
    y_std: float,
    *,
    torch: Any,
    device: Any,
) -> np.ndarray:  # pragma: no cover
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(indices), 512):
            batch_idx = indices[start : start + 512]
            batch_x = _torch_eeg_batch(x, batch_idx, channel_mean, channel_std, torch=torch, device=device)
            pred = model(batch_x).detach().cpu().numpy().reshape(-1)
            values.append((pred * float(y_std) + float(y_mean)).astype(np.float32))
    return np.concatenate(values) if values else np.zeros((0,), dtype=np.float32)


def _torch_extract_embeddings(
    model: Any,
    x: np.ndarray,
    indices: np.ndarray,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    *,
    torch: Any,
    device: Any,
) -> np.ndarray:  # pragma: no cover
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(indices), 512):
            batch_idx = indices[start : start + 512]
            batch_x = _torch_eeg_batch(x, batch_idx, channel_mean, channel_std, torch=torch, device=device)
            emb = model.embedding(batch_x).detach().cpu().numpy()
            values.append(emb.astype(np.float32))
    if not values:
        return np.zeros((0, 256), dtype=np.float32)
    result = np.concatenate(values, axis=0).astype(np.float32)
    if result.shape != (len(indices), 256):
        raise ValueError(f"expected torch EEG embedding shape {(len(indices), 256)}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("torch EEG embedding contains NaN or infinite values")
    return result


def _torch_optimizer_for_profile(model: Any, profile: str, runtime: MatrixRuntime, torch: Any) -> Any:  # pragma: no cover
    strategy = _strategy_for_profile(profile)
    if strategy == "frozen":
        return torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=float(runtime.head_learning_rate),
            weight_decay=float(runtime.weight_decay),
        )
    encoder_lr = runtime.partial_encoder_learning_rate if strategy == "partial" else runtime.full_encoder_learning_rate
    encoder_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(param)
        else:
            head_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": float(encoder_lr)},
            {"params": head_params, "lr": float(runtime.head_learning_rate)},
        ],
        weight_decay=float(runtime.weight_decay),
    )


class _TorchEEGRegressor:  # pragma: no cover - constructed only when torch is installed
    def __new__(cls, *, encoder: Any, dropout: float) -> Any:
        import torch

        class Module(torch.nn.Module):
            def __init__(self, encoder_: Any, dropout_: float) -> None:
                super().__init__()
                self.encoder = encoder_
                self.projection = torch.nn.Sequential(
                    torch.nn.LazyLinear(256),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(float(dropout_)),
                )
                self.head = torch.nn.Sequential(
                    torch.nn.LayerNorm(256),
                    torch.nn.Linear(256, 1),
                )

            def pooled_feature(self, x: Any) -> Any:
                output = self.encoder(x)
                if isinstance(output, dict):
                    output = output.get("features", output.get("encoder_output", output.get("cls_token")))
                if isinstance(output, (tuple, list)):
                    output = output[0]
                if output.ndim > 2:
                    output = output.reshape(output.shape[0], -1, output.shape[-1]).mean(dim=1)
                if output.ndim == 1:
                    output = output.unsqueeze(0)
                return output

            def embedding(self, x: Any) -> Any:
                return self.projection(self.pooled_feature(x))

            def forward(self, x: Any) -> Any:
                return self.head(self.embedding(x)).reshape(-1)

        return Module(encoder, dropout)


def _load_torch_state_if_present(model: Any, checkpoint: Path, *, torch: Any) -> dict[str, Any]:  # pragma: no cover
    if checkpoint.is_dir():
        candidates = [checkpoint / name for name in ("model.safetensors", "pytorch_model.bin", "model.pt", "checkpoint.pt")]
    else:
        candidates = [checkpoint]
    state_path = next((path for path in candidates if path.is_file()), None)
    if state_path is None:
        raise FileNotFoundError(f"checkpoint state file not found under {checkpoint}")
    if state_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(state_path), device="cpu")
    else:
        state = torch.load(str(state_path), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
    model_state = model.state_dict()
    matched = {}
    skipped: list[str] = []
    for key, value in state.items():
        normalized_key = key[7:] if key.startswith("module.") else key
        if normalized_key not in model_state:
            skipped.append(normalized_key)
            continue
        if tuple(model_state[normalized_key].shape) != tuple(value.shape):
            skipped.append(normalized_key)
            continue
        matched[normalized_key] = value
    missing, unexpected = model.load_state_dict(matched, strict=False)
    return {
        "state_path": str(state_path),
        "loaded_key_count": len(matched),
        "skipped_key_count": len(skipped),
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
        "skipped_keys_preview": sorted(skipped)[:10],
    }


def _seed_everything(seed: int, *, torch: Any | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
