from __future__ import annotations

import math

import torch
from torch import nn


class CumulativeOrdinalModalityProbe(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        modality_count: int = 4,
        kind: str = "cumulative",
    ) -> None:
        super().__init__()
        if kind not in {"cumulative", "categorical"}:
            raise ValueError(f"unsupported modality probe kind: {kind}")
        self.kind = kind
        output_dim = 4 if kind == "cumulative" else 5
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for _ in range(modality_count)])
        self.register_buffer("class_values", torch.arange(1, 6, dtype=torch.float32))

    def forward(
        self,
        adapted_tokens: torch.Tensor,
        modality_mask: torch.Tensor,
        *,
        temperatures: torch.Tensor | None = None,
        beta_ord: float = 0.25,
    ) -> dict[str, torch.Tensor]:
        valid = modality_mask.to(dtype=torch.bool)
        modality_valid = valid.any(dim=1)
        counts = valid.sum(dim=1).clamp_min(1).to(dtype=adapted_tokens.dtype)
        pooled = (adapted_tokens * valid[:, :, :, None].to(adapted_tokens.dtype)).sum(dim=1) / counts[:, :, None]
        logits = torch.stack([head(pooled[:, idx]) for idx, head in enumerate(self.heads)], dim=1)
        if temperatures is not None:
            expected_shape = (logits.shape[1],)
            if tuple(temperatures.shape) != expected_shape:
                raise ValueError(f"probe temperatures expected shape {expected_shape}, got {tuple(temperatures.shape)}")
            logits = logits / temperatures.to(device=logits.device, dtype=logits.dtype).view(1, -1, 1).clamp_min(1e-6)
        probs = cumulative_logits_to_probs(logits) if self.kind == "cumulative" else torch.softmax(logits, dim=-1)
        entropy = normalized_entropy(probs)
        ordinal_var = normalized_ordinal_variance(probs, self.class_values.to(probs.device, probs.dtype))
        beta = float(beta_ord)
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta_ord must be in [0, 1], got {beta_ord}")
        difficulty = ((1.0 - beta) * entropy + beta * ordinal_var).clamp(0.0, 1.0)
        difficulty = torch.where(modality_valid, difficulty, torch.ones_like(difficulty))
        result = {
            "probe_probs": probs,
            "probe_entropy": entropy,
            "probe_ordinal_var": ordinal_var,
            "modality_difficulty": difficulty,
            "valid_modality_mask": modality_valid,
        }
        result["probe_ordinal_logits" if self.kind == "cumulative" else "probe_class_logits"] = logits
        return result


def cumulative_logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    cumulative = torch.sigmoid(logits)
    cumulative = torch.cummin(cumulative, dim=-1).values
    probs = torch.cat(
        [
            1.0 - cumulative[..., :1],
            cumulative[..., :1] - cumulative[..., 1:2],
            cumulative[..., 1:2] - cumulative[..., 2:3],
            cumulative[..., 2:3] - cumulative[..., 3:4],
            cumulative[..., 3:4],
        ],
        dim=-1,
    ).clamp_min(0.0)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def normalized_entropy(probs: torch.Tensor) -> torch.Tensor:
    entropy = -(probs.clamp_min(1e-8) * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
    return entropy / math.log(probs.shape[-1])


def normalized_ordinal_variance(probs: torch.Tensor, class_values: torch.Tensor) -> torch.Tensor:
    values = class_values.view(*([1] * (probs.ndim - 1)), -1)
    mean = (probs * values).sum(dim=-1, keepdim=True)
    var = (probs * (values - mean) ** 2).sum(dim=-1)
    return (var / 4.0).clamp(0.0, 1.0)
