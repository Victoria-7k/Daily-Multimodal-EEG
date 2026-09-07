from __future__ import annotations

import torch
from torch import nn


class ModalityAdapters(nn.Module):
    def __init__(self, *, mode: str, input_dim: int = 256, hidden_dim: int = 128, modality_count: int = 4) -> None:
        super().__init__()
        if mode not in {"shared", "per_modality"}:
            raise ValueError(f"unsupported adapter mode: {mode}")
        self.mode = mode
        self.modality_count = int(modality_count)
        if mode == "shared":
            self.shared = nn.Linear(input_dim, hidden_dim)
        else:
            self.per_modality = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(modality_count)])
        self.modality_embedding = nn.Parameter(torch.zeros(1, 1, modality_count, hidden_dim))
        nn.init.normal_(self.modality_embedding, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError(f"tokens expected shape (B,L,M,D), got {tuple(tokens.shape)}")
        if tokens.shape[2] != self.modality_count:
            raise ValueError(f"expected {self.modality_count} modalities, got {tokens.shape[2]}")
        if self.mode == "shared":
            adapted = self.shared(tokens)
        else:
            adapted = torch.stack(
                [layer(tokens[:, :, idx, :]) for idx, layer in enumerate(self.per_modality)],
                dim=2,
            )
        return adapted + self.modality_embedding
