"""Label-free temporal masked-autoencoder components for aligned windows.

The module deliberately contains no emotion target or downstream fusion logic.
Callers supply only raw window signals, train/validation indices, and a seed.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _masked_patch_mse(prediction: torch.Tensor, target: torch.Tensor, masked: torch.Tensor) -> torch.Tensor:
    if not bool(masked.any()):
        raise ValueError("each MAE batch must contain at least one masked patch")
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
        embedded = self.patch_embed(patches) + self.position
        hidden = torch.where(masked[:, :, None], self.mask_token.expand_as(embedded), embedded)
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

    def _embed(self, ppg: torch.Tensor, eda: torch.Tensor, acc: torch.Tensor) -> torch.Tensor:
        return self.merge(torch.cat((self.ppg_stem(ppg), self.eda_stem(eda), self.acc_stem(acc)), dim=-1)) + self.position

    def encode(self, ppg: torch.Tensor, eda: torch.Tensor, acc: torch.Tensor) -> torch.Tensor:
        return self.encoder(self._embed(ppg, eda, acc))

    def forward(self, ppg: torch.Tensor, eda: torch.Tensor, acc: torch.Tensor, masked: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        embedded = self._embed(ppg, eda, acc)
        hidden = torch.where(masked[:, :, None], self.mask_token.expand_as(embedded), embedded)
        encoded = self.encoder(hidden)
        decoded = self.decoder(encoded)
        return (self.ppg_decoder(decoded), self.eda_decoder(decoded), self.acc_decoder(decoded)), encoded


def eeg_to_patches(values: np.ndarray) -> np.ndarray:
    """Return 10 one-second, per-channel normalized EEG patches."""
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (2000, 59):
        raise ValueError(f"expected EEG [batch,2000,59], got {x.shape}")
    patch = x.reshape(len(x), 10, 200, 59).transpose(0, 1, 3, 2)
    mean = patch.mean(axis=-1, keepdims=True)
    std = patch.std(axis=-1, keepdims=True)
    patch = (patch - mean) / np.maximum(std, 1e-6)
    return patch.reshape(len(x), 10, -1).astype(np.float32)


def wear_to_patches(ppg: np.ndarray, eda: np.ndarray, acc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return independently normalized 1-second PPG, EDA and tri-axial ACC patches."""
    p = np.asarray(ppg, dtype=np.float32)
    e = np.asarray(eda, dtype=np.float32)
    a = np.asarray(acc, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 1250 or e.shape != (len(p), 1, 400) or a.shape != (len(p), 3, 300):
        raise ValueError(f"unexpected wear shapes: ppg={p.shape}, eda={e.shape}, acc={a.shape}")
    p = p.reshape(len(p), 10, 125)
    e = e.reshape(len(p), 1, 10, 40).transpose(0, 2, 1, 3).reshape(len(p), 10, 40)
    a = a.reshape(len(p), 3, 10, 30).transpose(0, 2, 1, 3).reshape(len(p), 10, 90)
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


def _batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator):
    shuffled = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    for start in range(0, len(shuffled), batch_size):
        yield shuffled[start : start + batch_size]


def _random_mask(batch_size: int, ratio: float, device: torch.device) -> torch.Tensor:
    count = max(1, min(9, int(round(10 * ratio))))
    noise = torch.rand((batch_size, 10), device=device)
    return noise.argsort(dim=1).argsort(dim=1) < count


def train_eeg_mae(
    source: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, *, embedding_dim: int, encoder_layers: int, decoder_layers: int, heads: int, runtime: MAERuntime
) -> tuple[TemporalMaskedAutoencoder, dict[str, Any]]:
    _seed_everything(runtime.seed)
    device = torch.device(runtime.device)
    model = TemporalMaskedAutoencoder(patch_dim=59 * 200, embedding_dim=embedding_dim, encoder_layers=encoder_layers, decoder_layers=decoder_layers, heads=heads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=runtime.learning_rate, weight_decay=runtime.weight_decay)
    rng = np.random.default_rng(runtime.seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_val, best_epoch, waiting = float("inf"), 0, 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, runtime.epochs + 1):
        model.train(); train_losses: list[float] = []
        for batch_idx in _batches(train_idx, runtime.batch_size, rng):
            patches = torch.as_tensor(eeg_to_patches(source[batch_idx]), device=device)
            masked = _random_mask(len(batch_idx), runtime.mask_ratio, device)
            pred, _ = model(patches, masked)
            loss = _masked_patch_mse(pred, patches, masked)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        val_loss = _evaluate_eeg(model, source, val_idx, runtime, device)
        history.append({"epoch": epoch, "train_masked_nmse": float(np.mean(train_losses)), "val_masked_nmse": val_loss})
        if val_loss < best_val:
            best_val, best_epoch, waiting = val_loss, epoch, 0
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        else:
            waiting += 1
            if waiting >= runtime.patience: break
    if best_state is None: raise RuntimeError("EEG MAE did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_val_masked_nmse": best_val, "history": history, "train_count": int(len(train_idx)), "val_count": int(len(val_idx))}


@torch.no_grad()
def _evaluate_eeg(model: TemporalMaskedAutoencoder, source: np.ndarray, idx: np.ndarray, runtime: MAERuntime, device: torch.device) -> float:
    model.eval(); losses=[]; generator=np.random.default_rng(runtime.seed + 100003)
    for start in range(0, len(idx), runtime.batch_size):
        batch_idx=idx[start:start+runtime.batch_size]; patches=torch.as_tensor(eeg_to_patches(source[batch_idx]),device=device)
        mask_np=generator.random((len(batch_idx),10)) < runtime.mask_ratio
        for row in mask_np:
            if not row.any(): row[0]=True
        masked=torch.as_tensor(mask_np,device=device)
        pred,_=model(patches,masked); losses.append(float(_masked_patch_mse(pred,patches,masked).cpu()))
    return float(np.mean(losses))


def train_wear_mae(
    ppg_source: np.ndarray, eda_source: np.ndarray, acc_source: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, *, embedding_dim: int, encoder_layers: int, decoder_layers: int, heads: int, runtime: MAERuntime
) -> tuple[WearMaskedAutoencoder, dict[str, Any]]:
    _seed_everything(runtime.seed); device=torch.device(runtime.device)
    model=WearMaskedAutoencoder(embedding_dim=embedding_dim, encoder_layers=encoder_layers, decoder_layers=decoder_layers, heads=heads).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=runtime.learning_rate,weight_decay=runtime.weight_decay); rng=np.random.default_rng(runtime.seed)
    best_state=None; best_val,best_epoch,waiting=float("inf"),0,0; history=[]
    for epoch in range(1,runtime.epochs+1):
        model.train(); losses=[]
        for batch_idx in _batches(train_idx,runtime.batch_size,rng):
            p,e,a=(torch.as_tensor(x,device=device) for x in wear_to_patches(ppg_source[batch_idx],eda_source[batch_idx],acc_source[batch_idx]))
            masked=_random_mask(len(batch_idx),runtime.mask_ratio,device); (pp,pe,pa),_=model(p,e,a,masked)
            loss=(_masked_patch_mse(pp,p,masked)+_masked_patch_mse(pe,e,masked)+_masked_patch_mse(pa,a,masked))/3
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); losses.append(float(loss.detach().cpu()))
        val_loss=_evaluate_wear(model,ppg_source,eda_source,acc_source,val_idx,runtime,device)
        history.append({"epoch":epoch,"train_masked_nmse":float(np.mean(losses)),"val_masked_nmse":val_loss})
        if val_loss<best_val:
            best_val,best_epoch,waiting=val_loss,epoch,0; best_state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
        else:
            waiting+=1
            if waiting>=runtime.patience: break
    if best_state is None: raise RuntimeError("Wear MAE did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model,{"best_epoch":best_epoch,"best_val_masked_nmse":best_val,"history":history,"train_count":int(len(train_idx)),"val_count":int(len(val_idx))}


@torch.no_grad()
def _evaluate_wear(model: WearMaskedAutoencoder, ppg: np.ndarray, eda: np.ndarray, acc: np.ndarray, idx: np.ndarray, runtime: MAERuntime, device: torch.device) -> float:
    model.eval(); losses=[]; generator=np.random.default_rng(runtime.seed+100003)
    for start in range(0,len(idx),runtime.batch_size):
        batch_idx=idx[start:start+runtime.batch_size]; p,e,a=(torch.as_tensor(x,device=device) for x in wear_to_patches(ppg[batch_idx],eda[batch_idx],acc[batch_idx]))
        mask_np=generator.random((len(batch_idx),10))<runtime.mask_ratio
        for row in mask_np:
            if not row.any():row[0]=True
        masked=torch.as_tensor(mask_np,device=device); (pp,pe,pa),_=model(p,e,a,masked)
        losses.append(float((_masked_patch_mse(pp,p,masked)+_masked_patch_mse(pe,e,masked)+_masked_patch_mse(pa,a,masked)).div(3).cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def export_eeg_embeddings(model: TemporalMaskedAutoencoder, source: np.ndarray, *, batch_size: int, device: str) -> np.ndarray:
    model.eval(); dev=torch.device(device); output=[]
    for start in range(0,len(source),batch_size): output.append(model.encode(torch.as_tensor(eeg_to_patches(source[start:start+batch_size]),device=dev)).mean(dim=1).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


@torch.no_grad()
def export_wear_embeddings(model: WearMaskedAutoencoder, ppg: np.ndarray, eda: np.ndarray, acc: np.ndarray, valid: np.ndarray, *, batch_size: int, device: str) -> np.ndarray:
    model.eval(); dev=torch.device(device); output=np.zeros((len(ppg),model.position.shape[-1]),dtype=np.float32); valid_idx=np.flatnonzero(valid)
    for start in range(0,len(valid_idx),batch_size):
        idx=valid_idx[start:start+batch_size]; p,e,a=(torch.as_tensor(x,device=dev) for x in wear_to_patches(ppg[idx],eda[idx],acc[idx])); output[idx]=model.encode(p,e,a).mean(dim=1).cpu().numpy()
    return output
