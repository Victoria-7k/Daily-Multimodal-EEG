"""Label-free temporal masked-autoencoder components for aligned windows.

The module deliberately contains no emotion target or downstream fusion logic.
Callers supply only raw window signals, train/validation indices, and a seed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

MAE_TRAINING_VERSION = "position_preserved_fixed_validation_v2"
MAE_PREPROCESSING_VERSION = "per_second_zscore_v1"
MAE_FIXED_PREPROCESSING_VERSION = "train_channel_zscore_v1"
MAE_ROBUST_PREPROCESSING_VERSION = "train_channel_robust_zscore_v1"
MAE_BALANCED_TRAINING_VERSION = "position_preserved_balanced_channel_eeg_v4"


def _fit_channel_statistics(source: np.ndarray, indices: np.ndarray, channel_axis: int | None, batch_size: int) -> dict[str, Any]:
    """Streaming population moments over the supplied training rows only."""
    count = 0
    mean = m2 = None
    for start in range(0, len(indices), batch_size):
        values = np.asarray(source[indices[start:start + batch_size]], dtype=np.float64)
        values = values.reshape(-1, 1) if channel_axis is None else np.moveaxis(values, channel_axis, -1).reshape(-1, values.shape[channel_axis])
        if not np.isfinite(values).all():
            raise ValueError("non-finite raw signal in normalization training rows")
        n = len(values)
        batch_mean = values.mean(axis=0)
        batch_m2 = np.square(values - batch_mean).sum(axis=0)
        if count == 0:
            mean, m2 = batch_mean, batch_m2
        else:
            delta = batch_mean - mean
            m2 = m2 + batch_m2 + delta ** 2 * (count * n / (count + n))
            mean = mean + delta * (n / (count + n))
        count += n
    std = np.sqrt(m2 / count)
    return {"mean": mean.tolist(), "std": std.tolist(), "scale": np.maximum(std, 1e-6).tolist(),
            "sample_count_per_channel": count, "scale_floor": 1e-6}


def _fit_robust_channel_statistics(source: np.ndarray, indices: np.ndarray, channel_axis: int | None, batch_size: int) -> dict[str, Any]:
    """Fixed channel scales with equal window influence, retaining every raw row."""
    means, variances = [], []
    samples_per_window = None
    for start in range(0, len(indices), batch_size):
        values = np.asarray(source[indices[start:start + batch_size]], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("non-finite raw signal in normalization training rows")
        channels = 1 if channel_axis is None else values.shape[channel_axis]
        values = values.reshape(len(values), -1, 1) if channel_axis is None else np.moveaxis(values, channel_axis, -1).reshape(len(values), -1, channels)
        samples_per_window = values.shape[1]
        means.append(values.mean(axis=1))
        variances.append(values.var(axis=1))
    window_mean, window_variance = np.concatenate(means), np.concatenate(variances)
    center = np.median(window_mean, axis=0)
    window_rms = np.sqrt(window_variance + (window_mean - center) ** 2)
    scale = np.median(window_rms, axis=0)
    return {"mean": center.tolist(), "std": scale.tolist(), "scale": np.maximum(scale, 1e-6).tolist(),
            "sample_count_per_channel": len(indices) * samples_per_window, "scale_floor": 1e-6,
            "center_estimator": "median_window_mean", "scale_estimator": "median_window_rms_about_fixed_center"}


def fit_signal_normalization(modality: str, sources: tuple[np.ndarray, ...], train_idx: np.ndarray, *, batch_size: int = 128, robust: bool = False) -> dict[str, Any]:
    """Fit fixed per-channel affine transforms without accessing val/test rows."""
    indices = np.asarray(train_idx, dtype=np.int64)
    branches = {"eeg": (("eeg", -1),), "wear": (("ppg", None), ("eda", 1), ("acc", 1))}
    if modality not in branches or len(sources) != len(branches[modality]):
        raise ValueError("normalization modality/source count mismatch")
    if indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices) or batch_size <= 0:
        raise ValueError("normalization requires nonempty unique training indices and a positive batch size")
    if any(indices.min() < 0 or indices.max() >= len(source) for source in sources):
        raise ValueError("normalization training index outside source")
    fit = _fit_robust_channel_statistics if robust else _fit_channel_statistics
    return {"mode": "train_channel_robust" if robust else "train_channel",
            "version": MAE_ROBUST_PREPROCESSING_VERSION if robust else MAE_FIXED_PREPROCESSING_VERSION, "modality": modality,
            "fit_row_count": len(indices), "fit_indices_sha256": hashlib.sha256(np.sort(indices).astype("<i8").tobytes()).hexdigest(),
            "fit_scope": "supplied_training_rows_only", "branches": {
                name: fit(source, indices, axis, batch_size)
                for source, (name, axis) in zip(sources, branches[modality])}}


def _fixed_normalize(values: np.ndarray, normalization: dict[str, Any], branch: str, channel_axis: int | None) -> np.ndarray:
    versions = {"train_channel": MAE_FIXED_PREPROCESSING_VERSION, "train_channel_robust": MAE_ROBUST_PREPROCESSING_VERSION}
    if normalization.get("mode") not in versions or normalization.get("version") != versions[normalization["mode"]]:
        raise ValueError("unsupported fixed signal normalization")
    stats = normalization["branches"][branch]
    mean, scale = (np.asarray(stats[key], dtype=np.float32) for key in ("mean", "scale"))
    channels = 1 if channel_axis is None else values.shape[channel_axis]
    if mean.shape != (channels,) or scale.shape != (channels,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("invalid fixed channel statistics")
    shape = [1] * values.ndim
    if channel_axis is not None:
        shape[channel_axis] = channels
    return ((values - mean.reshape(shape)) / scale.reshape(shape)).astype(np.float32)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _masked_patch_mse(prediction: torch.Tensor, target: torch.Tensor, masked: torch.Tensor) -> torch.Tensor:
    if not bool(masked.any()):
        raise ValueError("each MAE batch must contain at least one masked patch")
    if masked.ndim == target.ndim:
        return ((prediction - target) ** 2)[masked].mean()
    error = torch.mean((prediction - target) ** 2, dim=-1)
    return error[masked].mean()


class TemporalMaskedAutoencoder(torch.nn.Module):
    """MAE for one normalized temporal patch vector per time step."""

    def __init__(self, *, patch_dim: int, embedding_dim: int, encoder_layers: int, decoder_layers: int, heads: int) -> None:
        super().__init__()
        self.patch_dim = int(patch_dim)
        self.patch_embed = torch.nn.Linear(patch_dim, embedding_dim)
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position = torch.nn.Parameter(torch.zeros(1, 10, embedding_dim))
        enc_layer = torch.nn.TransformerEncoderLayer(embedding_dim, heads, 4 * embedding_dim, batch_first=True, activation="gelu")
        dec_layer = torch.nn.TransformerEncoderLayer(embedding_dim, heads, 4 * embedding_dim, batch_first=True, activation="gelu")
        self.encoder = torch.nn.TransformerEncoder(enc_layer, encoder_layers)
        self.decoder = torch.nn.TransformerEncoder(dec_layer, decoder_layers)
        self.reconstruction = torch.nn.Linear(embedding_dim, patch_dim)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        torch.nn.init.normal_(self.position, std=0.02)

    def encode(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 3 or patches.shape[1] != 10 or patches.shape[2] != self.patch_dim:
            raise ValueError(f"expected [batch,10,{self.patch_dim}], got {tuple(patches.shape)}")
        return self.encoder(self.patch_embed(patches) + self.position)

    def forward(self, patches: torch.Tensor, masked: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if masked.shape == patches.shape:
            # Channel-masked EEG retains only observed channels at every time step.
            hidden = self.patch_embed(patches.masked_fill(masked, 0)) + self.position
        elif masked.shape == patches.shape[:2]:
            content = self.patch_embed(patches)
            hidden = torch.where(masked[:, :, None], self.mask_token.expand_as(content), content) + self.position
        else:
            raise ValueError("EEG mask must match temporal tokens or patch elements")
        encoded = self.encoder(hidden)
        return self.reconstruction(self.decoder(encoded)), encoded


class WearMaskedAutoencoder(torch.nn.Module):
    """Wear MAE with independent PPG, EDA and ACC stems and reconstructions."""

    def __init__(self, *, embedding_dim: int, encoder_layers: int, decoder_layers: int, heads: int) -> None:
        super().__init__()
        self.ppg_stem = torch.nn.Linear(125, embedding_dim)
        self.eda_stem = torch.nn.Linear(40, embedding_dim)
        self.acc_stem = torch.nn.Linear(90, embedding_dim)
        self.merge = torch.nn.Linear(3 * embedding_dim, embedding_dim)
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position = torch.nn.Parameter(torch.zeros(1, 10, embedding_dim))
        enc_layer = torch.nn.TransformerEncoderLayer(embedding_dim, heads, 4 * embedding_dim, batch_first=True, activation="gelu")
        dec_layer = torch.nn.TransformerEncoderLayer(embedding_dim, heads, 4 * embedding_dim, batch_first=True, activation="gelu")
        self.encoder = torch.nn.TransformerEncoder(enc_layer, encoder_layers)
        self.decoder = torch.nn.TransformerEncoder(dec_layer, decoder_layers)
        self.ppg_decoder = torch.nn.Linear(embedding_dim, 125)
        self.eda_decoder = torch.nn.Linear(embedding_dim, 40)
        self.acc_decoder = torch.nn.Linear(embedding_dim, 90)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        torch.nn.init.normal_(self.position, std=0.02)

    def _content(self, ppg: torch.Tensor, eda: torch.Tensor, acc: torch.Tensor) -> torch.Tensor:
        return self.merge(torch.cat((self.ppg_stem(ppg), self.eda_stem(eda), self.acc_stem(acc)), dim=-1))

    def _embed(self, ppg: torch.Tensor, eda: torch.Tensor, acc: torch.Tensor) -> torch.Tensor:
        return self._content(ppg, eda, acc) + self.position

    def encode(self, ppg: torch.Tensor, eda: torch.Tensor, acc: torch.Tensor) -> torch.Tensor:
        return self.encoder(self._embed(ppg, eda, acc))

    def forward(self, ppg: torch.Tensor, eda: torch.Tensor, acc: torch.Tensor, masked: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        content = self._content(ppg, eda, acc)
        hidden = torch.where(masked[:, :, None], self.mask_token.expand_as(content), content) + self.position
        encoded = self.encoder(hidden)
        decoded = self.decoder(encoded)
        return (self.ppg_decoder(decoded), self.eda_decoder(decoded), self.acc_decoder(decoded)), encoded


def eeg_to_patches(values: np.ndarray, normalization: dict[str, Any] | None = None) -> np.ndarray:
    """Return 10 EEG patches; legacy per-second or fixed train-channel scaling."""
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (2000, 59):
        raise ValueError(f"expected EEG [batch,2000,59], got {x.shape}")
    if normalization is not None:
        if normalization.get("modality") != "eeg":
            raise ValueError("EEG normalization modality mismatch")
        x = _fixed_normalize(x, normalization, "eeg", -1)
        return x.reshape(len(x), 10, 200, 59).transpose(0, 1, 3, 2).reshape(len(x), 10, -1)
    patch = x.reshape(len(x), 10, 200, 59).transpose(0, 1, 3, 2)
    mean = patch.mean(axis=-1, keepdims=True)
    std = patch.std(axis=-1, keepdims=True)
    patch = (patch - mean) / np.maximum(std, 1e-6)
    return patch.reshape(len(x), 10, -1).astype(np.float32)


def wear_to_patches(ppg: np.ndarray, eda: np.ndarray, acc: np.ndarray, normalization: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return independently normalized 1-second PPG, EDA and tri-axial ACC patches."""
    p = np.asarray(ppg, dtype=np.float32)
    e = np.asarray(eda, dtype=np.float32)
    a = np.asarray(acc, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 1250 or e.shape != (len(p), 1, 400) or a.shape != (len(p), 3, 300):
        raise ValueError(f"unexpected wear shapes: ppg={p.shape}, eda={e.shape}, acc={a.shape}")
    if normalization is not None:
        if normalization.get("modality") != "wear":
            raise ValueError("Wear normalization modality mismatch")
        p = _fixed_normalize(p, normalization, "ppg", None)
        e = _fixed_normalize(e, normalization, "eda", 1)
        a = _fixed_normalize(a, normalization, "acc", 1)
    p = p.reshape(len(p), 10, 125)
    e = e.reshape(len(p), 1, 10, 40).transpose(0, 2, 1, 3).reshape(len(p), 10, 40)
    a = a.reshape(len(p), 3, 10, 30).transpose(0, 2, 1, 3).reshape(len(p), 10, 90)
    if normalization is not None:
        return p, e, a
    def normalize(x: np.ndarray) -> np.ndarray:
        return ((x - x.mean(axis=-1, keepdims=True)) / np.maximum(x.std(axis=-1, keepdims=True), 1e-6)).astype(np.float32)
    return normalize(p), normalize(e), normalize(a)


@dataclass(frozen=True)
class MAERuntime:
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 15
    mask_ratio: float = 0.7
    seed: int = 240800
    device: str = "cuda"
    health_probe_count: int = 256
    min_relative_variation: float = 1e-3
    reconstruction_loss: str = "masked_mse"
    robust_health: bool = False
    eeg_masking: str = "temporal"


def reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor, masked: torch.Tensor, runtime: MAERuntime) -> torch.Tensor:
    """Equal-window loss; large-amplitude targets cannot dominate other windows."""
    if runtime.reconstruction_loss == "masked_mse":
        return _masked_patch_mse(prediction, target, masked)
    if runtime.reconstruction_loss != "window_energy_balanced_mse_v1":
        raise ValueError("unsupported MAE reconstruction loss")
    if masked.shape not in (target.shape, target.shape[:2]):
        raise ValueError("reconstruction mask must match tokens or target elements")
    axes = (1, 2) if masked.ndim == target.ndim else (1,)
    counts = masked.sum(dim=axes)
    if not bool((counts > 0).all()):
        raise ValueError("each MAE window must contain at least one masked patch")
    error = (prediction - target).square()
    if masked.ndim != target.ndim:
        error = error.mean(dim=-1)
    energy = target.detach().square().mean(dim=(1, 2)).clamp_min(1.0)
    return ((error * masked).sum(dim=axes) / counts / energy).mean()


def representative_indices(indices: np.ndarray, count: int) -> np.ndarray:
    """Deterministic canonical-order coverage; never mix train and val rows."""
    values = np.asarray(indices, dtype=np.int64)
    if count <= 0 or len(values) <= count:
        return values
    return np.sort(values)[np.linspace(0, len(values) - 1, count, dtype=np.int64)]


def fixed_validation_mask(batch: int, tokens: int, ratio: float, generator: np.random.Generator, device: torch.device) -> torch.Tensor:
    count = max(1, min(tokens - 1, int(round(tokens * ratio))))
    noise = generator.random((batch, tokens))
    return torch.as_tensor(noise.argsort(axis=1).argsort(axis=1) < count, device=device)


def require_finite(value: torch.Tensor, context: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"non-finite tensor: {context}")


def guarded_optimizer_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss: torch.Tensor, context: str) -> None:
    require_finite(loss, f"{context} loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    try:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    except RuntimeError as exc:
        raise RuntimeError(f"non-finite gradients: {context}") from exc
    optimizer.step()
    if not bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()):
        raise RuntimeError(f"non-finite parameters: {context}")


def embedding_health(values: np.ndarray, *, min_relative_variation: float = 1e-3, robust: bool = False) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or len(x) < 4 or not np.isfinite(x).all():
        raise RuntimeError("embedding health requires at least four finite feature vectors")
    centered = x - x.mean(axis=0, keepdims=True)
    rms = float(np.sqrt(np.mean(x ** 2)))
    centered_rms = float(np.sqrt(np.mean(centered ** 2)))
    relative = centered_rms / max(rms, 1e-12)
    median_center = np.median(x, axis=0)
    median_rms = float(np.median(np.sqrt(np.mean(x ** 2, axis=1))))
    median_centered_rms = float(np.median(np.sqrt(np.mean((x - median_center) ** 2, axis=1))))
    median_relative = median_centered_rms / max(median_rms, 1e-12)
    eigen = np.linalg.eigvalsh(centered.T @ centered / len(x)).clip(min=0)
    total = float(eigen.sum())
    weights = eigen / max(total, 1e-300)
    rank = float(np.exp(-(weights * np.log(np.maximum(weights, 1e-300))).sum())) if total > 0 else 0.0
    return {"probe_count": len(x), "feature_rms": rms, "centered_rms": centered_rms,
            "relative_variation": relative, "std_median": float(np.median(x.std(axis=0))),
            "effective_rank": rank, "min_relative_variation": min_relative_variation,
            "median_relative_variation": median_relative, "robust_health": robust,
            "collapsed": bool(relative < min_relative_variation or (robust and median_relative < min_relative_variation))}


def require_healthy(health: dict[str, Any], context: str) -> None:
    if health["collapsed"]:
        raise RuntimeError(f"collapsed embeddings: {context}: {json.dumps(health)}")


@torch.no_grad()
def probe_health(model: torch.nn.Module, indices: np.ndarray, batch_size: int, encode: Callable[[np.ndarray], torch.Tensor], *, count: int, minimum: float, robust: bool = False) -> dict[str, Any]:
    model.eval()
    probe = representative_indices(indices, count)
    output = [encode(probe[start:start + batch_size]).cpu().numpy() for start in range(0, len(probe), batch_size)]
    return embedding_health(np.concatenate(output), min_relative_variation=minimum, robust=robust)


def _batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator):
    shuffled = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    for start in range(0, len(shuffled), batch_size):
        yield shuffled[start : start + batch_size]


def _random_mask(batch_size: int, ratio: float, device: torch.device) -> torch.Tensor:
    count = max(1, min(9, int(round(10 * ratio))))
    noise = torch.rand((batch_size, 10), device=device)
    return noise.argsort(dim=1).argsort(dim=1) < count


def eeg_reconstruction_mask(batch: int, ratio: float, device: torch.device, *, kind: str, generator: np.random.Generator | None = None) -> torch.Tensor:
    if kind == "temporal":
        return _random_mask(batch, ratio, device) if generator is None else fixed_validation_mask(batch, 10, ratio, generator, device)
    if kind != "channel":
        raise ValueError("unsupported EEG masking kind")
    if generator is None:
        count = max(1, min(58, int(round(59 * ratio))))
        noise = torch.rand((batch, 59), device=device)
        channels = noise.argsort(dim=1).argsort(dim=1) < count
    else:
        channels = fixed_validation_mask(batch, 59, ratio, generator, device)
    return channels[:, None, :, None].expand(-1, 10, -1, 200).reshape(batch, 10, 11800)


def train_eeg_mae(
    source: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, *, embedding_dim: int, encoder_layers: int, decoder_layers: int, heads: int, runtime: MAERuntime, normalization: dict[str, Any] | None = None, train_source: Any | None = None, recovery_path: Path | None = None, retry_attention_math: bool = False
) -> tuple[TemporalMaskedAutoencoder, dict[str, Any]]:
    _seed_everything(runtime.seed)
    device = torch.device(runtime.device)
    model = TemporalMaskedAutoencoder(patch_dim=59 * 200, embedding_dim=embedding_dim, encoder_layers=encoder_layers, decoder_layers=decoder_layers, heads=heads).to(device)
    import hashlib
    import time
    started = time.perf_counter()
    initial_sha = hashlib.sha256(b''.join(value.detach().cpu().numpy().tobytes() for value in model.state_dict().values())).hexdigest()
    training_source = source if train_source is None else train_source
    optimizer = torch.optim.AdamW(model.parameters(), lr=runtime.learning_rate, weight_decay=runtime.weight_decay)
    rng = np.random.default_rng(runtime.seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_val, best_epoch, waiting = float("inf"), 0, 0
    history: list[dict[str, float | int]] = []
    contract = {"runtime": runtime.__dict__, "normalization": normalization,
                "train_indices_sha256": hashlib.sha256(np.asarray(train_idx, dtype='<i8').tobytes()).hexdigest(),
                "val_indices_sha256": hashlib.sha256(np.asarray(val_idx, dtype='<i8').tobytes()).hexdigest(), "initial_state_sha256": initial_sha}
    start_epoch = 1; previous_wall = 0.0; attention_retry_records = []
    if recovery_path is not None and recovery_path.exists():
        saved = torch.load(recovery_path, map_location='cpu', weights_only=False)
        if saved['contract'] != contract: raise ValueError('recovery checkpoint contract changed')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        best_state, best_val, best_epoch, waiting, history = (saved[k] for k in ('best_state','best_val','best_epoch','waiting','history'))
        rng.bit_generator.state = saved['shuffle_rng']; random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng']) if device.type == 'cuda' else None
        start_epoch = saved['epoch'] + 1; previous_wall = saved['gpu_wall_seconds']
        attention_retry_records = saved.get('attention_retry_records', [])
        if attention_retry_records and not retry_attention_math: raise ValueError('attention recovery must remain enabled after a recorded retry')
        if waiting >= runtime.patience: start_epoch = runtime.epochs + 1
        print(json.dumps({'recovery': 'epoch_boundary', 'completed_epochs': saved['epoch']}), flush=True)
    for epoch in range(start_epoch, runtime.epochs + 1):
        model.train(); train_losses: list[float] = []
        for batch_idx in _batches(train_idx, runtime.batch_size, rng):
            patches = torch.as_tensor(eeg_to_patches(training_source[batch_idx], normalization), device=device)
            masked = eeg_reconstruction_mask(len(batch_idx), runtime.mask_ratio, device, kind=runtime.eeg_masking)
            forward_rng = torch.cuda.get_rng_state_all() if (recovery_path is not None or retry_attention_math) and device.type == 'cuda' else None
            pred, _ = model(patches, masked)
            loss = reconstruction_loss(pred, patches, masked, runtime)
            try:
                guarded_optimizer_step(model, optimizer, loss, f"eeg epoch={epoch} batch={len(train_losses)}")
            except RuntimeError as error:
                if recovery_path is not None:
                    gradients = {key: value.grad.detach().cpu() for key,value in model.named_parameters() if value.grad is not None}
                    torch.save({'model': model.state_dict(), 'patches': patches.detach().cpu(), 'masked': masked.cpu(),
                                'gradients': gradients, 'forward_cuda_rng': forward_rng, 'runtime': runtime.__dict__,
                                'epoch': epoch, 'batch_indices': batch_idx, 'loss': float(loss.detach().cpu())}, recovery_path.with_suffix('.failure.pt'))
                if not (retry_attention_math and device.type=='cuda' and str(error).startswith('non-finite gradients:')):
                    raise
                if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):raise RuntimeError('attention recovery requires unchanged finite parameters') from error
                from torch.nn.attention import sdpa_kernel, SDPBackend
                discarded_loss=float(loss.detach().cpu());after_forward_rng=torch.cuda.get_rng_state_all()
                torch.cuda.set_rng_state_all(forward_rng)
                try:
                    with sdpa_kernel(SDPBackend.MATH):
                        pred,_=model(patches,masked)
                        loss=reconstruction_loss(pred,patches,masked,runtime)
                        guarded_optimizer_step(model,optimizer,loss,f'eeg math-retry epoch={epoch} batch={len(train_losses)}')
                finally:
                    # Later masks/dropout retain the original seeded CUDA stream.
                    torch.cuda.set_rng_state_all(after_forward_rng)
                retry={'epoch':epoch,'batch':len(train_losses),'row_count':len(batch_idx),
                       'discarded_default_loss':discarded_loss,'math_loss':float(loss.detach().cpu()),
                       'rows_skipped':0,'optimizer_updates':1,'rng_after_retry':'discarded_default_forward_stream'}
                attention_retry_records.append(retry)
                print(json.dumps({'attention_numerical_recovery':retry}),flush=True)
            train_losses.append(float(loss.detach().cpu()))
        val_detail = _evaluate_eeg(model, source, val_idx, runtime, device, normalization=normalization, details=True)
        val_loss = val_detail["masked_nmse"]
        health = probe_health(model, val_idx, runtime.batch_size,
            lambda idx: model.encode(torch.as_tensor(eeg_to_patches(source[idx], normalization), device=device)).mean(dim=1),
            count=runtime.health_probe_count, minimum=runtime.min_relative_variation, robust=runtime.robust_health)
        record = {"epoch": epoch, "train_masked_nmse": float(np.mean(train_losses)), "val_masked_nmse": val_loss, "validation": val_detail, "embedding_health": health,
                  "optimizer_steps": epoch * ((len(train_idx) + runtime.batch_size - 1) // runtime.batch_size), "seen_windows": epoch * len(train_idx)}
        history.append(record)
        print(json.dumps({"modality": "eeg", **record}), flush=True)
        require_healthy(health, f"eeg epoch={epoch}")
        if val_loss < best_val:
            best_val, best_epoch, waiting = val_loss, epoch, 0
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        else:
            waiting += 1
        if recovery_path is not None:
            recovery_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = recovery_path.with_suffix('.tmp.pt')
            torch.save({'contract': contract, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'best_state': best_state, 'best_val': best_val, 'best_epoch': best_epoch, 'waiting': waiting, 'history': history,
                        'epoch': epoch, 'shuffle_rng': rng.bit_generator.state, 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                        'gpu_wall_seconds': previous_wall + time.perf_counter()-started,'attention_retry_records':attention_retry_records}, temporary)
            temporary.replace(recovery_path)
        if waiting >= runtime.patience: break
    if best_state is None: raise RuntimeError("EEG MAE did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_val_masked_nmse": best_val, "history": history, "train_count": int(len(train_idx)), "val_count": int(len(val_idx)), "selected_embedding_health": history[best_epoch - 1]["embedding_health"],
                   "initial_state_sha256": initial_sha, "optimizer_steps": history[-1]["optimizer_steps"], "best_checkpoint_steps": history[best_epoch-1]["optimizer_steps"], "seen_windows": history[-1]["seen_windows"], "gpu_wall_seconds": previous_wall + time.perf_counter()-started,
                   "attention_retry_records": attention_retry_records}


@torch.no_grad()
def _evaluate_eeg(model: TemporalMaskedAutoencoder, source: np.ndarray, idx: np.ndarray, runtime: MAERuntime, device: torch.device, *, normalization: dict[str, Any] | None = None, details: bool = False) -> float | dict[str, Any]:
    model.eval(); losses=[]; zeros=[]; weights=[]; generator=np.random.default_rng(runtime.seed + 100003)
    for start in range(0, len(idx), runtime.batch_size):
        batch_idx=idx[start:start+runtime.batch_size]; patches=torch.as_tensor(eeg_to_patches(source[batch_idx], normalization),device=device)
        masked=eeg_reconstruction_mask(len(batch_idx),runtime.mask_ratio,device,kind=runtime.eeg_masking,generator=generator)
        pred,_=model(patches,masked); loss=reconstruction_loss(pred,patches,masked,runtime)
        require_finite(loss, "eeg validation loss")
        losses.append(float(loss.cpu())); weights.append(len(batch_idx))
        zeros.append(float(reconstruction_loss(torch.zeros_like(patches),patches,masked,runtime).cpu()))
    loss, zero = float(np.average(losses, weights=weights)), float(np.average(zeros, weights=weights))
    value = {"masked_nmse": loss, "zero_predictor_nmse": zero, "relative_to_zero": loss / max(zero, 1e-12)}
    return value if details else loss


def train_wear_mae(
    ppg_source: np.ndarray, eda_source: np.ndarray, acc_source: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, *, embedding_dim: int, encoder_layers: int, decoder_layers: int, heads: int, runtime: MAERuntime, normalization: dict[str, Any] | None = None
) -> tuple[WearMaskedAutoencoder, dict[str, Any]]:
    _seed_everything(runtime.seed); device=torch.device(runtime.device)
    model=WearMaskedAutoencoder(embedding_dim=embedding_dim, encoder_layers=encoder_layers, decoder_layers=decoder_layers, heads=heads).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=runtime.learning_rate,weight_decay=runtime.weight_decay); rng=np.random.default_rng(runtime.seed)
    best_state=None; best_val,best_epoch,waiting=float("inf"),0,0; history=[]
    for epoch in range(1,runtime.epochs+1):
        model.train(); losses=[]; branch_losses=[]
        for batch_idx in _batches(train_idx,runtime.batch_size,rng):
            p,e,a=(torch.as_tensor(x,device=device) for x in wear_to_patches(ppg_source[batch_idx],eda_source[batch_idx],acc_source[batch_idx], normalization))
            masked=_random_mask(len(batch_idx),runtime.mask_ratio,device); (pp,pe,pa),_=model(p,e,a,masked)
            branches=torch.stack([reconstruction_loss(pred,target,masked,runtime) for pred,target in zip((pp,pe,pa),(p,e,a))])
            loss=branches.mean()
            guarded_optimizer_step(model,optimizer,loss,f"wear epoch={epoch} batch={len(losses)}")
            losses.append(float(loss.detach().cpu())); branch_losses.append(branches.detach().cpu().numpy())
        val_detail=_evaluate_wear(model,ppg_source,eda_source,acc_source,val_idx,runtime,device,details=True, normalization=normalization)
        val_loss=val_detail["masked_nmse"]
        def encode(idx: np.ndarray) -> torch.Tensor:
            values=(torch.as_tensor(x,device=device) for x in wear_to_patches(ppg_source[idx],eda_source[idx],acc_source[idx], normalization))
            return model.encode(*values).mean(dim=1)
        health=probe_health(model,val_idx,runtime.batch_size,encode,count=runtime.health_probe_count,minimum=runtime.min_relative_variation,robust=runtime.robust_health)
        record={"epoch":epoch,"train_masked_nmse":float(np.mean(losses)),"val_masked_nmse":val_loss,
                "train_branch_nmse":dict(zip(("ppg","eda","acc"),np.mean(branch_losses,axis=0).astype(float).tolist())),
                "validation":val_detail,"embedding_health":health}
        history.append(record); print(json.dumps({"modality":"wear",**record}),flush=True)
        require_healthy(health,f"wear epoch={epoch}")
        if val_loss<best_val:
            best_val,best_epoch,waiting=val_loss,epoch,0; best_state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
        else:
            waiting+=1
            if waiting>=runtime.patience: break
    if best_state is None: raise RuntimeError("Wear MAE did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model,{"best_epoch":best_epoch,"best_val_masked_nmse":best_val,"history":history,"train_count":int(len(train_idx)),"val_count":int(len(val_idx)),"selected_embedding_health":history[best_epoch-1]["embedding_health"]}


@torch.no_grad()
def _evaluate_wear(model: WearMaskedAutoencoder, ppg: np.ndarray, eda: np.ndarray, acc: np.ndarray, idx: np.ndarray, runtime: MAERuntime, device: torch.device, *, details: bool = False, normalization: dict[str, Any] | None = None) -> float | dict[str, Any]:
    model.eval(); losses=[]; zeros=[]; weights=[]; generator=np.random.default_rng(runtime.seed+100003)
    for start in range(0,len(idx),runtime.batch_size):
        batch_idx=idx[start:start+runtime.batch_size]; p,e,a=(torch.as_tensor(x,device=device) for x in wear_to_patches(ppg[batch_idx],eda[batch_idx],acc[batch_idx], normalization))
        masked=fixed_validation_mask(len(batch_idx),10,runtime.mask_ratio,generator,device); (pp,pe,pa),_=model(p,e,a,masked)
        branch=torch.stack([reconstruction_loss(pred,target,masked,runtime) for pred,target in zip((pp,pe,pa),(p,e,a))])
        require_finite(branch,"wear validation loss")
        losses.append(branch.cpu().numpy()); weights.append(len(batch_idx))
        zeros.append([float(reconstruction_loss(torch.zeros_like(target),target,masked,runtime).cpu()) for target in (p,e,a)])
    branch=np.average(losses,axis=0,weights=weights); baseline=np.average(zeros,axis=0,weights=weights)
    value={"masked_nmse":float(branch.mean()),"branch_nmse":dict(zip(("ppg","eda","acc"),branch.tolist())),
           "zero_predictor_nmse":dict(zip(("ppg","eda","acc"),baseline.tolist())),
           "relative_to_zero":float(branch.mean()) / max(float(baseline.mean()), 1e-12)}
    return value if details else value["masked_nmse"]


@torch.no_grad()
def export_eeg_embeddings(model: TemporalMaskedAutoencoder, source: np.ndarray, *, batch_size: int, device: str, normalization: dict[str, Any] | None = None) -> np.ndarray:
    model.eval(); dev=torch.device(device); output=[]
    for start in range(0,len(source),batch_size):
        value=model.encode(torch.as_tensor(eeg_to_patches(source[start:start+batch_size], normalization),device=dev)).mean(dim=1)
        require_finite(value,f"eeg export start={start}"); output.append(value.cpu().numpy())
    return np.concatenate(output).astype(np.float32)


@torch.no_grad()
def export_wear_embeddings(model: WearMaskedAutoencoder, ppg: np.ndarray, eda: np.ndarray, acc: np.ndarray, valid: np.ndarray, *, batch_size: int, device: str, normalization: dict[str, Any] | None = None) -> np.ndarray:
    model.eval(); dev=torch.device(device); output=np.zeros((len(ppg),model.position.shape[-1]),dtype=np.float32); valid_idx=np.flatnonzero(valid)
    for start in range(0,len(valid_idx),batch_size):
        idx=valid_idx[start:start+batch_size]; p,e,a=(torch.as_tensor(x,device=dev) for x in wear_to_patches(ppg[idx],eda[idx],acc[idx], normalization)); value=model.encode(p,e,a).mean(dim=1)
        require_finite(value,f"wear export start={start}"); output[idx]=value.cpu().numpy()
    return output
