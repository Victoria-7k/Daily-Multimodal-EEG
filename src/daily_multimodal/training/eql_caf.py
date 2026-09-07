from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class EQLCAFConfig:
    input_dim: int = 256
    hidden_dim: int = 128
    num_tokens: int = 5
    num_heads: int = 4
    dropout: float = 0.1
    quality_feature_dim: int = 1
    use_lag_bias: bool = True
    use_token_quality: bool = True
    use_modality_gate: bool = True
    use_eeg_residual: bool = True


class EQLCAFRegressor(nn.Module):
    """EEG-anchored quality- and lag-aware cross-attention regressor."""

    def __init__(self, config: EQLCAFConfig | None = None):
        super().__init__()
        self.config = config or EQLCAFConfig()
        cfg = self.config
        if cfg.hidden_dim % cfg.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.input_proj = nn.ModuleList([nn.Linear(cfg.input_dim, cfg.hidden_dim) for _ in range(4)])
        self.layer_norm = nn.LayerNorm(cfg.hidden_dim)
        self.modality_embedding = nn.Parameter(torch.zeros(4, cfg.hidden_dim))
        self.position_embedding = nn.Parameter(torch.zeros(cfg.num_tokens, cfg.hidden_dim))
        self.quality_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(max(1, cfg.quality_feature_dim), cfg.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(cfg.hidden_dim // 2, 1),
                )
                for _ in range(4)
            ]
        )
        self.cross = nn.ModuleList(
            [_LagAwareCrossAttention(cfg.hidden_dim, cfg.num_heads, cfg.num_tokens, cfg.dropout) for _ in range(3)]
        )
        self.eeg_pool_score = nn.Linear(cfg.hidden_dim, 1)
        self.aux_pool_score = nn.Linear(cfg.hidden_dim, 1)
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(cfg.hidden_dim * 2 + 2, cfg.hidden_dim), nn.ReLU(), nn.Linear(cfg.hidden_dim, 1)) for _ in range(3)])
        self.aux_out = nn.ModuleList([nn.Linear(cfg.hidden_dim, cfg.hidden_dim) for _ in range(3)])
        self.eeg_head = nn.Sequential(nn.LayerNorm(cfg.hidden_dim), nn.Linear(cfg.hidden_dim, 1))
        self.residual_head = nn.Sequential(nn.LayerNorm(cfg.hidden_dim), nn.Linear(cfg.hidden_dim, 1))
        self.global_gate = nn.Sequential(nn.Linear(6, cfg.hidden_dim // 2), nn.ReLU(), nn.Linear(cfg.hidden_dim // 2, 1))
        self.dropout = nn.Dropout(cfg.dropout)
        self._reset_parameters()

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        quality_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if tokens.ndim != 4 or tokens.shape[1] != 4:
            raise ValueError(f"tokens expected shape (batch,4,time,dim), got {tuple(tokens.shape)}")
        if token_mask.shape != tokens.shape[:3]:
            raise ValueError(f"token_mask expected shape {tuple(tokens.shape[:3])}, got {tuple(token_mask.shape)}")
        batch, _, time_steps, _ = tokens.shape
        if time_steps != self.config.num_tokens:
            raise ValueError(f"expected {self.config.num_tokens} time tokens, got {time_steps}")
        mask = token_mask.to(dtype=torch.bool)
        z = []
        for idx in range(4):
            projected = self.input_proj[idx](tokens[:, idx])
            projected = projected + self.modality_embedding[idx].view(1, 1, -1) + self.position_embedding.view(1, time_steps, -1)
            z.append(self.layer_norm(projected))
        encoded = torch.stack(z, dim=1)
        quality = self._quality_scores(quality_features, mask)
        eeg = encoded[:, 0]
        eeg_mask = mask[:, 0]
        eeg_summary = _masked_pool(eeg, eeg_mask, self.eeg_pool_score)
        fused = eeg_summary
        aux_summaries = []
        gate_values = []
        attn_weights = []
        for aux_index in range(3):
            modality_slot = aux_index + 1
            context, attn = self.cross[aux_index](
                eeg,
                encoded[:, modality_slot],
                query_mask=eeg_mask,
                key_mask=mask[:, modality_slot],
                quality=quality[:, modality_slot] if self.config.use_token_quality else None,
                use_lag_bias=self.config.use_lag_bias,
            )
            summary = _masked_pool(context, eeg_mask, self.aux_pool_score)
            aux_summaries.append(summary)
            available = mask[:, modality_slot].float().mean(dim=1, keepdim=True)
            q_mean = quality[:, modality_slot].mean(dim=1, keepdim=True)
            gate_input = torch.cat([eeg_summary, summary, q_mean, available], dim=1)
            gate = torch.sigmoid(self.gates[aux_index](gate_input))
            if not self.config.use_modality_gate:
                gate = available
            gate = gate * (available > 0).float()
            gate_values.append(gate)
            fused = fused + gate * self.aux_out[aux_index](summary)
            attn_weights.append(attn)
        eeg_prediction = self.eeg_head(eeg_summary).squeeze(-1)
        residual = self.residual_head(self.dropout(fused)).squeeze(-1)
        gates = torch.cat(gate_values, dim=1) if gate_values else tokens.new_zeros((batch, 0))
        global_features = torch.cat(
            [
                gates,
                quality[:, 1:].mean(dim=2),
            ],
            dim=1,
        )
        global_gate = torch.sigmoid(self.global_gate(global_features)).squeeze(-1)
        if not self.config.use_eeg_residual:
            global_gate = torch.ones_like(global_gate)
        if gates.numel():
            global_gate = global_gate * (gates.sum(dim=1) > 0).float()
        prediction = eeg_prediction + global_gate * residual
        return {
            "prediction": prediction,
            "eeg_prediction": eeg_prediction,
            "residual": residual,
            "modality_gates": gates,
            "global_gate": global_gate,
            "attention": torch.stack(attn_weights, dim=1),
        }

    def _quality_scores(self, quality_features: torch.Tensor | None, mask: torch.Tensor) -> torch.Tensor:
        if quality_features is None:
            return mask.to(dtype=torch.float32)
        if quality_features.ndim != 4 or quality_features.shape[:3] != mask.shape:
            raise ValueError(f"quality_features expected shape {tuple(mask.shape)} + (Q,), got {tuple(quality_features.shape)}")
        quality = quality_features.to(dtype=next(self.parameters()).dtype)
        scores = []
        for idx in range(4):
            features = quality[:, idx]
            if features.shape[-1] != self.config.quality_feature_dim:
                raise ValueError(
                    f"quality feature dim expected {self.config.quality_feature_dim}, got {features.shape[-1]}"
                )
            scores.append(torch.sigmoid(self.quality_mlps[idx](features)).squeeze(-1))
        return torch.stack(scores, dim=1) * mask.to(dtype=torch.float32)

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.modality_embedding, mean=0.0, std=0.02)


class _LagAwareCrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_tokens: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_tokens = num_tokens
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.lag_bias = nn.Parameter(torch.zeros(num_heads, 2 * num_tokens - 1))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        *,
        query_mask: torch.Tensor,
        key_mask: torch.Tensor,
        quality: torch.Tensor | None,
        use_lag_bias: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key_value))
        v = self._split_heads(self.v_proj(key_value))
        scores = torch.einsum("bhtd,bhsd->bhts", q, k) / math.sqrt(self.head_dim)
        if use_lag_bias:
            scores = scores + self._lag_bias_matrix(scores.device, scores.dtype).view(1, self.num_heads, self.num_tokens, self.num_tokens)
        scores = scores.masked_fill(~key_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        if quality is not None:
            q_prior = torch.clamp(quality, min=1e-6)
            scores = scores + torch.log(q_prior)[:, None, None, :]
        attn = torch.softmax(scores, dim=-1)
        attn = torch.where(query_mask[:, None, :, None], attn, torch.zeros_like(attn))
        context = torch.einsum("bhts,bhsd->bhtd", self.dropout(attn), v)
        merged = context.transpose(1, 2).contiguous().view(query.shape[0], self.num_tokens, self.hidden_dim)
        return self.out_proj(merged), attn

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = value.shape
        return value.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def _lag_bias_matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        idx = torch.arange(self.num_tokens, device=device)
        relative = idx[:, None] - idx[None, :] + self.num_tokens - 1
        return self.lag_bias[:, relative].to(dtype=dtype)


def _masked_pool(values: torch.Tensor, mask: torch.Tensor, scorer: nn.Linear) -> torch.Tensor:
    scores = scorer(values).squeeze(-1)
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=1)
    weights = torch.where(mask, weights, torch.zeros_like(weights))
    denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    weights = weights / denom
    return torch.einsum("bt,btd->bd", weights, values)


class TemporalConcatRegressor(nn.Module):
    def __init__(self, *, input_dim: int = 256, hidden_dim: int = 128, num_modalities: int = 4, num_tokens: int = 5, dropout: float = 0.1):
        super().__init__()
        self.num_modalities = num_modalities
        self.num_tokens = num_tokens
        self.net = nn.Sequential(
            nn.Linear(num_modalities * num_tokens * input_dim + num_modalities * num_tokens, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor, quality_features: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        masked = tokens * token_mask[:, :, :, None].to(tokens.dtype)
        flat = torch.cat([masked.reshape(tokens.shape[0], -1), token_mask.to(tokens.dtype).reshape(tokens.shape[0], -1)], dim=1)
        prediction = self.net(flat).squeeze(-1)
        return {"prediction": prediction}


class TemporalSelfAttentionRegressor(nn.Module):
    def __init__(self, *, input_dim: int = 256, hidden_dim: int = 128, num_modalities: int = 4, num_tokens: int = 5, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_modalities = num_modalities
        self.num_tokens = num_tokens
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.modality_embedding = nn.Parameter(torch.zeros(num_modalities, hidden_dim))
        self.position_embedding = nn.Parameter(torch.zeros(num_tokens, hidden_dim))
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.pool_score = nn.Linear(hidden_dim, 1)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        nn.init.normal_(self.modality_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor, quality_features: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        batch, modalities, steps, _ = tokens.shape
        x = self.input_projection(tokens)
        x = x + self.modality_embedding.view(1, modalities, 1, -1) + self.position_embedding.view(1, 1, steps, -1)
        x = x.reshape(batch, modalities * steps, -1)
        mask = token_mask.reshape(batch, modalities * steps).to(dtype=torch.bool)
        x = x.masked_fill(~mask[:, :, None], 0.0)
        encoded, attn = self.attention(x, x, x, key_padding_mask=~mask)
        summary = _masked_pool(encoded, mask, self.pool_score)
        return {"prediction": self.head(summary).squeeze(-1), "attention": attn}


def make_eql_caf_variant(
    model_name: str,
    *,
    input_dim: int = 256,
    hidden_dim: int = 128,
    num_tokens: int = 5,
    num_heads: int = 4,
    dropout: float = 0.1,
    quality_feature_dim: int = 1,
) -> nn.Module:
    if model_name == "B1":
        return TemporalConcatRegressor(input_dim=input_dim, hidden_dim=hidden_dim, num_tokens=num_tokens, dropout=dropout)
    if model_name == "B2":
        return TemporalSelfAttentionRegressor(input_dim=input_dim, hidden_dim=hidden_dim, num_tokens=num_tokens, num_heads=num_heads, dropout=dropout)
    configs = {
        "M1": dict(use_lag_bias=False, use_token_quality=False, use_modality_gate=False, use_eeg_residual=False),
        "M2": dict(use_lag_bias=True, use_token_quality=False, use_modality_gate=False, use_eeg_residual=False),
        "M3": dict(use_lag_bias=True, use_token_quality=False, use_modality_gate=True, use_eeg_residual=True),
        "M4": dict(use_lag_bias=True, use_token_quality=True, use_modality_gate=True, use_eeg_residual=True),
        "M5": dict(use_lag_bias=True, use_token_quality=True, use_modality_gate=True, use_eeg_residual=True),
    }
    if model_name not in configs:
        raise ValueError(f"unsupported EQL-CAF model name: {model_name}")
    return EQLCAFRegressor(
        EQLCAFConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_tokens=num_tokens,
            num_heads=num_heads,
            dropout=dropout,
            quality_feature_dim=quality_feature_dim,
            **configs[model_name],
        )
    )
