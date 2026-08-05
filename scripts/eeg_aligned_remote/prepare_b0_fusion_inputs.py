from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


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

DEFAULT_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_EEG_ROOT = Path("/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new")
DEFAULT_OUT = Path("/vePFS-0x0d/DailyEEG_multimodal/embeddings/eeg/eeg_statfft_eeg23win_embeddings.npz")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare B0 EEG-aligned fusion inputs.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--eeg-root", type=Path, default=DEFAULT_EEG_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--sample-rate-hz", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=240729)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    summary_out = args.summary_out or args.root / "reports/b0_eeg_embedding_summary.json"
    if args.out.exists() and not args.force:
        _write_existing_summary(args.out, summary_out)
        print(f"exists={args.out}")
        print("use --force to regenerate")
        return 0

    rows = _load_index(args.root / "index/eeg_aligned_window_index.jsonl")
    X = np.load(args.eeg_root / "X.npy", mmap_mode="r", allow_pickle=True)
    y = np.load(args.eeg_root / "y.npy", mmap_mode="r", allow_pickle=True)
    sub = np.load(args.eeg_root / "sub.npy", mmap_mode="r", allow_pickle=True)
    day = np.load(args.eeg_root / "d.npy", mmap_mode="r", allow_pickle=True)
    ts = np.load(args.eeg_root / "ts.npy", mmap_mode="r", allow_pickle=True)
    if X.shape[0] != len(rows):
        raise ValueError(f"X row count {X.shape[0]} != index row count {len(rows)}")
    if X.ndim != 3:
        raise ValueError(f"expected X shape [rows, samples, channels], got {X.shape}")
    if y.shape != (len(rows), len(LABEL_NAMES)):
        raise ValueError(f"expected y shape {(len(rows), len(LABEL_NAMES))}, got {y.shape}")

    out = args.out
    tmp_out = out.with_name(f"{out.stem}.tmp.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_out.parent.mkdir(parents=True, exist_ok=True)

    n_rows = int(X.shape[0])
    emb = np.zeros((n_rows, 256), dtype=np.float32)
    eeg_mask = np.zeros(n_rows, dtype=np.int8)
    feature_dim = None
    projection = None
    finite_failures = 0
    print(f"rows={n_rows} samples={X.shape[1]} channels={X.shape[2]} chunk_size={args.chunk_size}")
    for start in range(0, n_rows, max(1, int(args.chunk_size))):
        end = min(n_rows, start + max(1, int(args.chunk_size)))
        chunk = np.asarray(X[start:end], dtype=np.float32)
        finite = np.isfinite(chunk).all(axis=(1, 2))
        finite_failures += int((~finite).sum())
        chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
        features = _chunk_features(chunk, sfreq=float(args.sample_rate_hz))
        if feature_dim is None:
            feature_dim = int(features.shape[1])
            projection = _projection_matrix(feature_dim, seed=int(args.seed))
            print(f"feature_dim={feature_dim}")
        row_mean = features.mean(axis=1, keepdims=True)
        row_std = features.std(axis=1, keepdims=True)
        row_std = np.where(row_std >= 1e-6, row_std, 1.0)
        normalized = (features - row_mean) / row_std
        values = np.tanh(normalized @ projection).astype(np.float32)
        values[~finite] = 0.0
        emb[start:end] = values
        eeg_mask[start:end] = finite.astype(np.int8)
        print(f"processed={end}/{n_rows} mask_sum={int(eeg_mask[:end].sum())}", flush=True)

    modality_mask = np.zeros((n_rows, 4), dtype=np.int8)
    modality_mask[:, 0] = eeg_mask
    labels_json = np.asarray([_label_json(y[idx]) for idx in range(n_rows)], dtype=object)
    quality_flags = np.asarray(
        [
            json.dumps(
                {
                    "encoder": "eeg_statfft_eeg23win",
                    "sample_rate_hz": float(args.sample_rate_hz),
                    "source_shape": [int(X.shape[1]), int(X.shape[2])],
                    "finite": bool(eeg_mask[idx]),
                    "feature_dim": int(feature_dim or 0),
                },
                separators=(",", ":"),
            )
            for idx in range(n_rows)
        ],
        dtype=object,
    )
    np.savez_compressed(
        tmp_out,
        sample_id=np.asarray([row["sample_id"] for row in rows], dtype=object),
        eeg_sample_index=np.arange(n_rows, dtype=np.int64),
        subject_id=np.asarray([_norm_subject(value) for value in np.asarray(sub).tolist()], dtype=object),
        day_id=np.asarray(day, dtype=np.int64),
        event_id=np.asarray([row["event_id"] for row in rows], dtype=object),
        event_window_id=np.asarray([row["event_window_id"] for row in rows], dtype=np.int64),
        window_start_seconds=np.asarray([row["window_start_seconds"] for row in rows], dtype=np.float32),
        window_end_seconds=np.asarray([row["window_end_seconds"] for row in rows], dtype=np.float32),
        eeg_window_start_seconds=np.asarray(ts, dtype=np.float32),
        labels=labels_json,
        eeg_emb=emb,
        eeg_mask=eeg_mask,
        modality_mask=modality_mask,
        quality_flags=quality_flags,
        encoder_version=np.asarray(["eeg_statfft_eeg23win_v1"] * n_rows, dtype=object),
    )
    tmp_out.replace(out)

    summary = {
        "path": str(out),
        "row_count": n_rows,
        "x_shape": [int(value) for value in X.shape],
        "embedding_shape": [int(value) for value in emb.shape],
        "mask_sum": int(eeg_mask.sum()),
        "finite_failures": finite_failures,
        "feature_dim": int(feature_dim or 0),
        "encoder_version": "eeg_statfft_eeg23win_v1",
        "decision": "Generated a deterministic 256-dimensional EEG token from per-window time statistics and FFT bandpower features because no EEGPT checkpoint/runtime was available in the B0 handoff.",
    }
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"out={out}")
    print(f"summary_out={summary_out}")
    return 0


def _load_index(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _chunk_features(chunk: np.ndarray, *, sfreq: float) -> np.ndarray:
    # Input shape is [batch, samples, channels].
    means = chunk.mean(axis=1)
    stds = chunk.std(axis=1)
    rms = np.sqrt(np.mean(np.square(chunk), axis=1))
    mins = chunk.min(axis=1)
    maxs = chunk.max(axis=1)
    ptp = maxs - mins
    band_features = _bandpower_features(chunk, sfreq=sfreq)
    global_stats = np.stack(
        [
            chunk.mean(axis=(1, 2)),
            chunk.std(axis=(1, 2)),
            np.sqrt(np.mean(np.square(chunk), axis=(1, 2))),
            chunk.min(axis=(1, 2)),
            chunk.max(axis=(1, 2)),
        ],
        axis=1,
    )
    return np.concatenate([means, stds, rms, mins, maxs, ptp, band_features, global_stats], axis=1).astype(np.float32)


def _bandpower_features(chunk: np.ndarray, *, sfreq: float) -> np.ndarray:
    freqs = np.fft.rfftfreq(chunk.shape[1], d=1.0 / float(sfreq))
    spectrum = np.abs(np.fft.rfft(chunk, axis=1)) ** 2
    features = []
    for low_hz, high_hz in ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45)):
        mask = (freqs >= low_hz) & (freqs < high_hz)
        if mask.any():
            power = spectrum[:, mask, :].mean(axis=1)
        else:
            power = np.zeros((chunk.shape[0], chunk.shape[2]), dtype=np.float32)
        features.append(np.log1p(power).astype(np.float32))
    return np.concatenate(features, axis=1)


def _projection_matrix(feature_dim: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scale = 1.0 / max(1.0, float(np.sqrt(feature_dim)))
    return rng.normal(0.0, scale, size=(feature_dim, 256)).astype(np.float32)


def _write_existing_summary(path: Path, summary_out: Path) -> None:
    with np.load(path, allow_pickle=True) as loaded:
        emb = loaded["eeg_emb"]
        mask = loaded["eeg_mask"].astype(np.int8)
        summary = {
            "path": str(path),
            "row_count": int(emb.shape[0]),
            "embedding_shape": [int(value) for value in emb.shape],
            "mask_sum": int(mask.sum()),
            "finite_failures": int((mask == 0).sum()),
            "encoder_version": str(loaded["encoder_version"][0]) if "encoder_version" in loaded.files and len(loaded["encoder_version"]) else "unknown",
            "decision": "Existing EEG embedding file was reused.",
        }
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _label_json(row: np.ndarray) -> str:
    payload = {name: float(row[idx]) for idx, name in enumerate(LABEL_NAMES)}
    return json.dumps(payload, separators=(",", ":"))


def _norm_subject(value: Any) -> str:
    text = str(value)
    if text.startswith("sub-"):
        return text
    return f"sub-{int(float(text)):02d}"


if __name__ == "__main__":
    raise SystemExit(main())
