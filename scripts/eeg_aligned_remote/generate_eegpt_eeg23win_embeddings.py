from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import types
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import signal


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

STANDARD_1020_NAMES = [
    "FP1",
    "FPZ",
    "FP2",
    "AF7",
    "AF3",
    "AFZ",
    "AF4",
    "AF8",
    "F7",
    "F5",
    "F3",
    "F1",
    "FZ",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCZ",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "T7",
    "C5",
    "C3",
    "C1",
    "CZ",
    "C2",
    "C4",
    "C6",
    "T8",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPZ",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "P7",
    "P5",
    "P3",
    "P1",
    "PZ",
    "P2",
    "P4",
    "P6",
    "P8",
    "PO7",
    "PO5",
    "PO3",
    "POZ",
    "PO4",
    "PO6",
    "PO8",
    "O1",
    "OZ",
    "O2",
]

DEFAULT_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_EEG_ROOT = Path("/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new")
DEFAULT_OUT = Path("/vePFS-0x0d/DailyEEG_multimodal/embeddings/eeg/eeg_eegpt_eeg23win_embeddings.npz")
DEFAULT_CHECKPOINT = DEFAULT_ROOT / "outputs/checkpoints/eegpt-pretrained"
EMBEDDING_DIM = 256
TARGET_SAMPLE_RATE_HZ = 250.0
TARGET_WINDOW_SECONDS = 10.0
TARGET_WINDOW_SAMPLES = 2500
ENCODER_PROFILE = "eeg_deep_frozen_v1"
PROJECTION_SEED = 15015


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate EEGPT EEG embeddings for the EEG-aligned 23-window dataset.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--eeg-root", type=Path, default=DEFAULT_EEG_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report-md", type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--checksum-out", type=Path)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--download-checkpoint", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.torch_threads)))
    if args.out.exists() and not args.force:
        print(f"exists={args.out}")
        print("use --force to regenerate")
        return 0

    report_md = args.report_md or args.root / "reports/eeg_eegpt_eeg23win_embedding_report.md"
    report_json = args.report_json or args.root / "reports/eeg_eegpt_eeg23win_embedding_report.json"
    checksum_out = args.checksum_out or args.root / "checksums/eeg_eegpt_eeg23win_sha256.txt"

    rows = _load_index(args.root / "index/eeg_aligned_window_index.jsonl")
    X = np.load(args.eeg_root / "X.npy", mmap_mode="r", allow_pickle=True)
    y = np.load(args.eeg_root / "y.npy", mmap_mode="r", allow_pickle=True)
    sub = np.load(args.eeg_root / "sub.npy", mmap_mode="r", allow_pickle=True)
    day = np.load(args.eeg_root / "d.npy", mmap_mode="r", allow_pickle=True)
    ts = np.load(args.eeg_root / "ts.npy", mmap_mode="r", allow_pickle=True)
    _validate_inputs(rows, X, y, sub, day, ts)

    n_total = len(rows)
    n_rows = n_total if args.max_rows is None else min(n_total, max(0, int(args.max_rows)))
    print(f"row_count={n_rows} source_rows={n_total} X_shape={X.shape} X_dtype={X.dtype}", flush=True)

    checkpoint_report = _ensure_checkpoint(args.checkpoint, download=bool(args.download_checkpoint))
    EEGPT = _load_eegpt_class()
    channel_names = STANDARD_1020_NAMES[: int(X.shape[2])]
    model, load_report = _load_eegpt_model(
        EEGPT,
        checkpoint_path=args.checkpoint,
        n_chans=int(X.shape[2]),
        channel_names=channel_names,
        device=args.device,
    )
    if int(load_report.get("loaded_key_count", 0)) <= 0:
        raise RuntimeError(f"EEGPT checkpoint did not load any matching keys: {load_report}")
    model.eval()

    eeg_emb = np.zeros((n_rows, EMBEDDING_DIM), dtype=np.float32)
    eeg_mask = np.zeros(n_rows, dtype=np.int8)
    failures: list[dict[str, Any]] = []
    nan_count = 0
    feature_dim: int | None = None
    projection_cache: dict[int, np.ndarray] = {}
    started = time.time()

    for start in range(0, n_rows, max(1, int(args.batch_size))):
        end = min(n_rows, start + max(1, int(args.batch_size)))
        batch_indices = list(range(start, end))
        try:
            batch = [_preprocess_window(np.asarray(X[idx], dtype=np.float32)) for idx in batch_indices]
        except Exception:
            for idx in batch_indices:
                try:
                    batch = [_preprocess_window(np.asarray(X[idx], dtype=np.float32))]
                    features = _eegpt_features(model, batch, device=args.device)
                    if feature_dim is None:
                        feature_dim = int(features.shape[1])
                    eeg_emb[idx : idx + 1] = _project_batch_to_256(features, projection_cache=projection_cache)
                    eeg_mask[idx] = 1
                except Exception as exc:
                    failures.append(_failure(rows[idx], idx, exc))
            continue

        try:
            features = _eegpt_features(model, batch, device=args.device)
            if feature_dim is None:
                feature_dim = int(features.shape[1])
                print(f"deep_feature_dim={feature_dim}", flush=True)
            values = _project_batch_to_256(features, projection_cache=projection_cache)
            if not np.isfinite(values).all():
                raise ValueError("projected EEG embedding contains NaN or infinite values")
            eeg_emb[start:end] = values
            eeg_mask[start:end] = 1
        except Exception:
            for local_offset, idx in enumerate(batch_indices):
                try:
                    features = _eegpt_features(model, [batch[local_offset]], device=args.device)
                    if feature_dim is None:
                        feature_dim = int(features.shape[1])
                    values = _project_batch_to_256(features, projection_cache=projection_cache)
                    if not np.isfinite(values).all():
                        raise ValueError("projected EEG embedding contains NaN or infinite values")
                    eeg_emb[idx : idx + 1] = values
                    eeg_mask[idx] = 1
                except Exception as exc:
                    failures.append(_failure(rows[idx], idx, exc))

        nan_count = int(np.isnan(eeg_emb[:end]).sum())
        if end == n_rows or end % max(1, int(args.progress_every)) == 0:
            elapsed = time.time() - started
            rate = end / elapsed if elapsed > 0 else 0.0
            print(
                f"processed={end}/{n_rows} eeg_mask_sum={int(eeg_mask.sum())} failures={len(failures)} rate={rate:.3f}/s",
                flush=True,
            )

    modality_mask = np.zeros((n_rows, 4), dtype=np.int8)
    modality_mask[:, 0] = eeg_mask
    labels = np.asarray(y[:n_rows], dtype=np.float32)
    quality_flags = np.asarray(
        [
            json.dumps(
                {
                    "encoder_profile": ENCODER_PROFILE,
                    "backend": "braindecode EEGPT",
                    "checkpoint_path": str(args.checkpoint),
                    "target_sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
                    "target_samples": TARGET_WINDOW_SAMPLES,
                    "notch_hz": 50.0,
                    "bandpass_hz": [1.0, 45.0],
                    "source_shape": [int(X.shape[1]), int(X.shape[2])],
                    "channel_names": channel_names,
                    "deep_feature_dim": int(feature_dim or 0),
                    "success": bool(eeg_mask[idx]),
                },
                separators=(",", ":"),
            )
            for idx in range(n_rows)
        ],
        dtype=object,
    )
    out = args.out
    tmp_out = out.with_name(f"{out.stem}.tmp.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        tmp_out,
        sample_id=np.asarray([row["sample_id"] for row in rows[:n_rows]], dtype=object),
        eeg_sample_index=np.arange(n_rows, dtype=np.int64),
        subject_id=np.asarray([_norm_subject(value) for value in np.asarray(sub[:n_rows]).tolist()], dtype=object),
        day_id=np.asarray(day[:n_rows], dtype=np.int64),
        event_id=np.asarray([row["event_id"] for row in rows[:n_rows]], dtype=object),
        event_window_id=np.asarray([int(row["event_window_id"]) for row in rows[:n_rows]], dtype=np.int64),
        window_start_seconds=np.asarray([float(row["window_start_seconds"]) for row in rows[:n_rows]], dtype=np.float32),
        window_end_seconds=np.asarray([float(row["window_end_seconds"]) for row in rows[:n_rows]], dtype=np.float32),
        eeg_window_start_seconds=np.asarray(ts[:n_rows], dtype=np.float32),
        labels=labels,
        eeg_emb=eeg_emb,
        eeg_mask=eeg_mask,
        modality_mask=modality_mask,
        quality_flags=quality_flags,
        encoder_version=np.asarray([ENCODER_PROFILE] * n_rows, dtype=object),
    )
    tmp_out.replace(out)
    sha256 = _sha256_file(out)
    checksum_out.parent.mkdir(parents=True, exist_ok=True)
    checksum_out.write_text(f"{sha256}  {out}\n", encoding="utf-8")

    report = {
        "path": str(out),
        "row_count": n_rows,
        "source_row_count": n_total,
        "eeg_emb_shape": [int(value) for value in eeg_emb.shape],
        "eeg_mask_shape": [int(value) for value in eeg_mask.shape],
        "eeg_mask_sum": int(eeg_mask.sum()),
        "failure_count": int(len(failures)),
        "nan_count": int(np.isnan(eeg_emb).sum()),
        "checksum_sha256": sha256,
        "encoder_profile": ENCODER_PROFILE,
        "backend": "braindecode EEGPT",
        "checkpoint": checkpoint_report,
        "load_report": load_report,
        "target_sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
        "target_samples": TARGET_WINDOW_SAMPLES,
        "preprocessing": {"notch_hz": 50.0, "bandpass_hz": [1.0, 45.0]},
        "deep_feature_dim": int(feature_dim or 0),
        "failures_preview": failures[:20],
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md.write_text(_markdown_report(report), encoding="utf-8")
    print(f"out={out}", flush=True)
    print(f"report_md={report_md}", flush=True)
    print(f"checksum_out={checksum_out}", flush=True)
    return 0


def _load_index(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != 28819:
        raise ValueError(f"expected 28819 index rows, got {len(rows)}")
    for idx, row in enumerate(rows):
        if row.get("sample_id") != f"eeg_{idx:06d}":
            raise ValueError(f"index sample_id mismatch at row {idx}")
        if int(row.get("eeg_sample_index", -1)) != idx:
            raise ValueError(f"index eeg_sample_index mismatch at row {idx}")
    return rows


def _validate_inputs(rows: list[dict[str, Any]], X: np.ndarray, y: np.ndarray, sub: np.ndarray, day: np.ndarray, ts: np.ndarray) -> None:
    n_rows = len(rows)
    if X.shape[0] != n_rows or X.ndim != 3:
        raise ValueError(f"expected X shape [28819, samples, channels], got {X.shape}")
    if y.shape != (n_rows, len(LABEL_NAMES)):
        raise ValueError(f"expected y shape {(n_rows, len(LABEL_NAMES))}, got {y.shape}")
    if sub.shape[0] != n_rows or day.shape[0] != n_rows or ts.shape[0] != n_rows:
        raise ValueError("sub/d/ts row count does not match index")
    index_labels = np.asarray([row["labels"] for row in rows], dtype=np.float32)
    if not np.allclose(np.asarray(y, dtype=np.float32), index_labels):
        raise ValueError("y.npy labels do not match eeg_aligned_window_index labels")
    if int(X.shape[2]) > len(STANDARD_1020_NAMES):
        raise ValueError(f"X has {X.shape[2]} channels but only {len(STANDARD_1020_NAMES)} channel names are defined")


def _ensure_checkpoint(checkpoint: Path, *, download: bool) -> dict[str, Any]:
    checkpoint.mkdir(parents=True, exist_ok=True)
    existing = _checkpoint_state_files(checkpoint)
    if existing:
        return {"path": str(checkpoint), "downloaded": False, "state_files": [str(path) for path in existing]}
    if not download:
        raise RuntimeError(f"EEGPT checkpoint state file not found under {checkpoint}; rerun with --download-checkpoint")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="braindecode/eegpt-pretrained",
        local_dir=str(checkpoint),
        resume_download=True,
    )
    existing = _checkpoint_state_files(checkpoint)
    if not existing:
        raise RuntimeError(f"download completed but no checkpoint state file found under {checkpoint}")
    return {"path": str(checkpoint), "downloaded": True, "state_files": [str(path) for path in existing]}


def _checkpoint_state_files(checkpoint: Path) -> list[Path]:
    names = ["model.safetensors", "pytorch_model.bin", "model.pt", "checkpoint.pt"]
    return [checkpoint / name for name in names if (checkpoint / name).is_file()]


def _load_eegpt_model(EEGPT: Any, *, checkpoint_path: Path, n_chans: int, channel_names: list[str], device: str) -> tuple[Any, dict[str, Any]]:
    model = EEGPT(
        n_outputs=1,
        n_chans=n_chans,
        n_times=TARGET_WINDOW_SAMPLES,
        sfreq=TARGET_SAMPLE_RATE_HZ,
        chs_info=_chs_info(channel_names),
        return_encoder_output=True,
    )
    load_report = _load_matching_torch_weights(model, checkpoint_path)
    model.to(torch.device(device))
    model.eval()
    return model, load_report


def _load_matching_torch_weights(model: Any, checkpoint_path: Path) -> dict[str, Any]:
    state_path = _select_checkpoint_state_file(checkpoint_path)
    model_state = model.state_dict()
    if state_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        checkpoint_state = load_file(str(state_path), device="cpu")
    else:
        try:
            checkpoint_state = torch.load(str(state_path), map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint_state = torch.load(str(state_path), map_location="cpu")
        if isinstance(checkpoint_state, dict) and "state_dict" in checkpoint_state:
            checkpoint_state = checkpoint_state["state_dict"]
    matched = {}
    skipped = []
    for key, value in checkpoint_state.items():
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
        "checkpoint_key_count": len(checkpoint_state),
        "loaded_key_count": len(matched),
        "skipped_key_count": len(skipped),
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
        "skipped_keys_preview": sorted(skipped)[:20],
    }


def _select_checkpoint_state_file(checkpoint_path: Path) -> Path:
    if checkpoint_path.is_file():
        return checkpoint_path
    for path in _checkpoint_state_files(checkpoint_path):
        return path
    raise RuntimeError(f"EEGPT checkpoint state file was not found under {checkpoint_path}")


def _preprocess_window(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected one EEG window with 2 dims, got {values.shape}")
    # DailyEEG X.npy is [samples, channels]. EEGPT expects [channels, samples].
    data = values.T if values.shape[0] >= values.shape[1] else values
    if not np.isfinite(data).all():
        raise ValueError("EEG window contains NaN or infinite values")
    source_samples = int(data.shape[1])
    source_sfreq = float(source_samples / TARGET_WINDOW_SECONDS)
    filtered = _filter_window(data, sfreq=source_sfreq)
    if source_samples != TARGET_WINDOW_SAMPLES:
        gcd = math.gcd(source_samples, TARGET_WINDOW_SAMPLES)
        filtered = signal.resample_poly(
            filtered,
            up=TARGET_WINDOW_SAMPLES // gcd,
            down=source_samples // gcd,
            axis=1,
        ).astype(np.float32, copy=False)
    if filtered.shape[1] > TARGET_WINDOW_SAMPLES:
        filtered = filtered[:, :TARGET_WINDOW_SAMPLES]
    elif filtered.shape[1] < TARGET_WINDOW_SAMPLES:
        pad = TARGET_WINDOW_SAMPLES - filtered.shape[1]
        filtered = np.pad(filtered, ((0, 0), (0, pad)), mode="edge")
    if filtered.shape[1] != TARGET_WINDOW_SAMPLES:
        raise ValueError(f"expected target samples {TARGET_WINDOW_SAMPLES}, got {filtered.shape[1]}")
    if not np.isfinite(filtered).all():
        raise ValueError("preprocessed EEG window contains NaN or infinite values")
    return filtered.astype(np.float32, copy=False)


def _filter_window(data: np.ndarray, *, sfreq: float) -> np.ndarray:
    filtered = np.asarray(data, dtype=np.float32)
    if sfreq > 100.0:
        b_notch, a_notch = signal.iirnotch(50.0, Q=30.0, fs=sfreq)
        filtered = signal.filtfilt(b_notch, a_notch, filtered, axis=1).astype(np.float32, copy=False)
    high = min(45.0, sfreq / 2.0 - 1.0)
    if high <= 1.0:
        raise ValueError(f"source sampling frequency too low for 1-45Hz bandpass: {sfreq}")
    sos = signal.butter(4, [1.0, high], btype="bandpass", fs=sfreq, output="sos")
    return signal.sosfiltfilt(sos, filtered, axis=1).astype(np.float32, copy=False)


def _eegpt_features(model: Any, batch: list[np.ndarray], *, device: str) -> np.ndarray:
    tensor = torch.as_tensor(np.stack(batch, axis=0), dtype=torch.float32, device=torch.device(device))
    with torch.no_grad():
        output = model(tensor)
    if isinstance(output, dict):
        output = output.get("features", output.get("encoder_output", output.get("cls_token")))
    if isinstance(output, (tuple, list)):
        output = output[0]
    if hasattr(output, "detach"):
        values = output.detach().cpu().numpy()
    else:
        values = np.asarray(output)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    elif values.ndim > 2:
        values = values.reshape(values.shape[0], -1, values.shape[-1]).mean(axis=1)
    if values.ndim != 2 or values.shape[0] != len(batch):
        raise ValueError(f"unexpected EEGPT feature shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("EEGPT feature vector contains NaN or infinite values")
    return values.astype(np.float32, copy=False)


def _project_batch_to_256(features: np.ndarray, *, projection_cache: dict[int, np.ndarray]) -> np.ndarray:
    feature_dim = int(features.shape[1])
    if feature_dim not in projection_cache:
        projection_cache[feature_dim] = _projection_matrix(feature_dim)
    weights = projection_cache[feature_dim]
    row_mean = features.mean(axis=1, keepdims=True)
    row_std = features.std(axis=1, keepdims=True)
    row_std = np.where(row_std >= 1e-6, row_std, 1.0)
    normalized = (features - row_mean) / row_std
    return np.tanh(normalized @ weights).astype(np.float32)


def _projection_matrix(feature_dim: int) -> np.ndarray:
    rng_seed = PROJECTION_SEED + int(hashlib.sha256(ENCODER_PROFILE.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(rng_seed)
    scale = 1.0 / max(1.0, float(np.sqrt(feature_dim)))
    return rng.normal(0.0, scale, size=(feature_dim, EMBEDDING_DIM)).astype(np.float32)


def _failure(row: dict[str, Any], idx: int, exc: Exception) -> dict[str, Any]:
    return {
        "row_index": int(idx),
        "sample_id": str(row.get("sample_id", f"eeg_{idx:06d}")),
        "eeg_sample_index": int(row.get("eeg_sample_index", idx)),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _norm_subject(value: Any) -> str:
    text = str(value)
    if text.startswith("sub-"):
        return text
    return f"sub-{int(float(text)):02d}"


def _chs_info(channel_names: list[str]) -> list[dict[str, object]]:
    return [{"ch_name": name, "kind": 2, "loc": [0.0] * 12} for name in channel_names]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# EEGPT EEG23Win Embedding Report",
        "",
        f"row_count: `{report['row_count']}`",
        f"eeg_emb shape: `{report['eeg_emb_shape']}`",
        f"eeg_mask shape: `{report['eeg_mask_shape']}`",
        f"eeg_mask_sum: `{report['eeg_mask_sum']}`",
        f"failure_count: `{report['failure_count']}`",
        f"nan_count: `{report['nan_count']}`",
        f"sha256: `{report['checksum_sha256']}`",
        f"encoder_profile: `{report['encoder_profile']}`",
        f"backend: `{report['backend']}`",
        f"deep_feature_dim: `{report['deep_feature_dim']}`",
        "",
        "## Checkpoint",
        "",
        f"path: `{report['checkpoint']['path']}`",
        f"downloaded: `{report['checkpoint']['downloaded']}`",
        f"state_files: `{report['checkpoint']['state_files']}`",
        f"loaded_key_count: `{report['load_report']['loaded_key_count']}`",
        f"skipped_key_count: `{report['load_report']['skipped_key_count']}`",
    ]
    if report["failures_preview"]:
        lines.extend(["", "## Failure Preview", ""])
        lines.extend(f"- `{row}`" for row in report["failures_preview"])
    return "\n".join(lines) + "\n"


def _load_eegpt_class():
    import importlib.util
    import site

    _install_mne_stub()
    candidates = [Path(base) / "braindecode" for base in site.getsitepackages()]
    candidates.append(Path(sys.prefix) / "lib/python3.11/site-packages/braindecode")
    package_root = next(path for path in candidates if (path / "models/eegpt.py").is_file())

    bd_pkg = types.ModuleType("braindecode")
    bd_pkg.__path__ = [str(package_root)]
    sys.modules["braindecode"] = bd_pkg
    _install_braindecode_util_stub()
    _install_braindecode_modules_stub()

    models_pkg = types.ModuleType("braindecode.models")
    models_pkg.__path__ = [str(package_root / "models")]
    sys.modules["braindecode.models"] = models_pkg
    _install_interpolated_model_stub()

    spec = importlib.util.spec_from_file_location("braindecode.models.eegpt", package_root / "models/eegpt.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to locate braindecode.models.eegpt")
    module = importlib.util.module_from_spec(spec)
    sys.modules["braindecode.models.eegpt"] = module
    spec.loader.exec_module(module)
    return module.EEGPT


def _install_mne_stub() -> None:
    mne = types.ModuleType("mne")
    channels = types.ModuleType("mne.channels")
    utils = types.ModuleType("mne.utils")

    class Montage:
        ch_names = STANDARD_1020_NAMES

        def get_positions(self):
            return {"ch_pos": {name: [0.0, 0.0, 0.0] for name in self.ch_names}}

    def make_standard_montage(name: str) -> Montage:
        if name != "standard_1020":
            raise ValueError(f"unsupported montage {name}")
        return Montage()

    def _soft_import(name: str, purpose: str, strict: bool = False):
        del purpose
        try:
            return __import__(name)
        except Exception:
            if strict:
                raise
            return False

    def warn(message: str, *args, **kwargs) -> None:
        del args, kwargs
        warnings.warn(message, stacklevel=2)

    channels.make_standard_montage = make_standard_montage
    utils._soft_import = _soft_import
    utils.warn = warn
    mne.channels = channels
    mne.utils = utils
    sys.modules["mne"] = mne
    sys.modules["mne.channels"] = channels
    sys.modules["mne.utils"] = utils


def _install_braindecode_util_stub() -> None:
    util = types.ModuleType("braindecode.util")

    def np_to_th(X, requires_grad=False, dtype=None, pin_memory=False, **tensor_kwargs):
        if not hasattr(X, "__len__"):
            X = [X]
        values = np.asarray(X)
        if dtype is not None:
            values = values.astype(dtype)
        tensor = torch.tensor(values, requires_grad=requires_grad, **tensor_kwargs)
        if pin_memory:
            tensor = tensor.pin_memory()
        return tensor

    util.np_to_th = np_to_th
    sys.modules["braindecode.util"] = util


def _install_braindecode_modules_stub() -> None:
    import torch.nn.functional as F
    from torch import nn

    modules = types.ModuleType("braindecode.modules")
    conv = types.ModuleType("braindecode.modules.convolution")
    linear = types.ModuleType("braindecode.modules.linear")

    class DropPath(nn.Module):
        def __init__(self, drop_prob=None):
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, x):
            drop_prob = float(self.drop_prob or 0.0)
            if drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1.0 - drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return x.div(keep_prob) * random_tensor

    class Conv1dWithConstraint(nn.Conv1d):
        def __init__(self, *args, max_norm=1, **kwargs):
            super().__init__(*args, **kwargs)
            self.max_norm = max_norm

        def forward(self, input):
            weight = self.weight
            if self.max_norm is not None:
                weight = torch.renorm(weight, p=2, dim=0, maxnorm=float(self.max_norm))
            return F.conv1d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    class LinearWithConstraint(nn.Linear):
        def __init__(self, *args, max_norm=1, **kwargs):
            super().__init__(*args, **kwargs)
            self.max_norm = max_norm

        def forward(self, input):
            weight = self.weight
            if self.max_norm is not None:
                weight = torch.renorm(weight, p=2, dim=0, maxnorm=float(self.max_norm))
            return F.linear(input, weight, self.bias)

    modules.DropPath = DropPath
    modules.Conv1dWithConstraint = Conv1dWithConstraint
    modules.LinearWithConstraint = LinearWithConstraint
    conv.Conv1dWithConstraint = Conv1dWithConstraint
    linear.LinearWithConstraint = LinearWithConstraint
    sys.modules["braindecode.modules"] = modules
    sys.modules["braindecode.modules.convolution"] = conv
    sys.modules["braindecode.modules.linear"] = linear


def _install_interpolated_model_stub() -> None:
    interpolated = types.ModuleType("braindecode.models.interpolated")

    def InterpolatedModel(model_cls, target_chs_info, name="InterpolatedModel", **kwargs):
        del target_chs_info, kwargs
        return type(str(name), (model_cls,), {})

    interpolated.InterpolatedModel = InterpolatedModel
    sys.modules["braindecode.models.interpolated"] = interpolated


if __name__ == "__main__":
    raise SystemExit(main())
