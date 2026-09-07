from __future__ import annotations

import torch
from torch import nn


class AffectStateFilter(nn.Module):
    def __init__(self, *, hidden_dim: int = 128) -> None:
        super().__init__()
        self.cell = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, evidence: torch.Tensor, window_mask: torch.Tensor) -> torch.Tensor:
        if evidence.ndim != 3:
            raise ValueError(f"evidence expected shape (B,L,H), got {tuple(evidence.shape)}")
        state = self.initial_state(evidence)
        states = []
        mask = window_mask.to(dtype=torch.bool)
        for step in range(evidence.shape[1]):
            state = self.update_step(evidence[:, step], state, mask[:, step])
            states.append(state)
        return torch.stack(states, dim=1)

    @staticmethod
    def initial_state(evidence: torch.Tensor) -> torch.Tensor:
        return evidence.new_zeros((evidence.shape[0], evidence.shape[2]))

    def update_step(self, evidence: torch.Tensor, state: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        updated = self.cell(evidence, state)
        return torch.where(valid[:, None], updated, state)
