from __future__ import annotations

import torch
from torch import nn


class DynamicEMAKernel(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        sequence_len: int = 23,
        hop_sec: float = 5.0,
        tau_sec: tuple[float, float, float] = (15.0, 45.0, 120.0),
    ) -> None:
        super().__init__()
        self.sequence_len = int(sequence_len)
        self.hop_sec = float(hop_sec)
        self.tau_sec = tuple(float(value) for value in tau_sec)
        if len(self.tau_sec) != 3 or any(value <= 0.0 for value in self.tau_sec):
            raise ValueError("tau_sec must contain three positive decay constants")
        self.mixture = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 3))
        positions = torch.arange(sequence_len, dtype=torch.float32)
        distance_from_recent = (sequence_len - 1) - positions
        bases = []
        for tau in self.tau_sec:
            tau_hops = tau / self.hop_sec
            weights = torch.exp(-distance_from_recent / tau_hops)
            bases.append(weights / weights.sum().clamp_min(1e-6))
        self.register_buffer("bases", torch.stack(bases, dim=0))

    def forward(self, states: torch.Tensor, window_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if states.ndim != 3 or states.shape[1] != self.sequence_len:
            raise ValueError(f"states expected shape (B,{self.sequence_len},H), got {tuple(states.shape)}")
        valid = window_mask.to(dtype=states.dtype)
        summary = torch.sum(states * valid[:, :, None], dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        mixture = torch.softmax(self.mixture(summary), dim=-1)
        weights = torch.matmul(mixture, self.bases.to(device=states.device, dtype=states.dtype))
        weights = weights * valid
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return weights, mixture


class GlobalEMAKernel(nn.Module):
    """One trainable mixture of the same constrained decay bases for all EMA bags."""

    def __init__(
        self,
        *,
        sequence_len: int = 23,
        hop_sec: float = 5.0,
        tau_sec: tuple[float, float, float] = (15.0, 45.0, 120.0),
    ) -> None:
        super().__init__()
        self.sequence_len = int(sequence_len)
        tau_values = tuple(float(value) for value in tau_sec)
        if len(tau_values) != 3 or any(value <= 0.0 for value in tau_values):
            raise ValueError("tau_sec must contain three positive decay constants")
        positions = torch.arange(sequence_len, dtype=torch.float32)
        distance_from_recent = (sequence_len - 1) - positions
        bases = []
        for tau in tau_values:
            weights = torch.exp(-distance_from_recent / (tau / float(hop_sec)))
            bases.append(weights / weights.sum().clamp_min(1e-6))
        self.register_buffer("bases", torch.stack(bases, dim=0))
        self.mixture_logits = nn.Parameter(torch.zeros(3))

    def forward(self, states: torch.Tensor, window_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if states.ndim != 3 or states.shape[1] != self.sequence_len:
            raise ValueError(f"states expected shape (B,{self.sequence_len},H), got {tuple(states.shape)}")
        mixture = torch.softmax(self.mixture_logits, dim=0).view(1, -1).expand(states.shape[0], -1)
        weights = torch.matmul(mixture, self.bases.to(device=states.device, dtype=states.dtype))
        weights = weights * window_mask.to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return weights, mixture
