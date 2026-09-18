"""Scalar-regression EMA-bag models for the Daily-affect comparison route.

The module deliberately keeps the ordinal model untouched.  It reuses the
same token adapters, window fusion, state, prior, and temporal-kernel
structures while exposing one scalar prediction per EMA event.  The
``window_replicated`` model id produces independent per-window predictions and
their fixed event-level mean; every other supported id predicts after the
EMA-bag temporal aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .adapters import ModalityAdapters
from .affect_state_filter import AffectStateFilter
from .dynamic_ema_kernel import DynamicEMAKernel, GlobalEMAKernel
from .model import (
    DYNAMIC_KERNEL_MODEL_IDS,
    GLOBAL_KERNEL_MODEL_IDS,
    ORDINAL_DIFFICULTY_MODEL_IDS,
    PRIOR_MODEL_IDS,
    STATE_MODEL_IDS,
    STATIC_TEMPORAL_POLICIES,
    SUPPORTED_MODEL_IDS,
    WINDOW_ATTENTION_MODEL_IDS,
    _static_temporal_weights,
)


@dataclass(frozen=True)
class DailyAffectRegressionConfig:
    model_id: str = "bag_static"
    input_dim: int = 256
    hidden_dim: int = 128
    sequence_len: int = 23
    modality_count: int = 4
    adapter_mode: str = "shared"
    dropout: float = 0.1
    lambda_d: float = 0.25
    detach_difficulty: bool = True
    temporal_policy: str = "uniform"


class GaussianRegressionModalityProbe(nn.Module):
    """Per-modality EMA regression probe used only by difficulty routes."""

    def __init__(self, *, hidden_dim: int, modality_count: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 2) for _ in range(modality_count)])

    def forward(self, adapted: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        valid = mask.to(dtype=torch.bool)
        modality_valid = valid.any(dim=1)
        counts = valid.sum(dim=1).clamp_min(1).to(dtype=adapted.dtype)
        pooled = (adapted * valid[:, :, :, None].to(dtype=adapted.dtype)).sum(dim=1) / counts[:, :, None]
        parameters = torch.stack([head(pooled[:, index]) for index, head in enumerate(self.heads)], dim=1)
        mean = parameters[..., 0]
        # Keeping the log variance bounded makes the NLL and sigmoid difficulty
        # numerically stable while retaining a useful uncertainty ordering.
        log_variance = parameters[..., 1].clamp(-8.0, 8.0)
        difficulty = torch.sigmoid(log_variance)
        difficulty = torch.where(modality_valid, difficulty, torch.ones_like(difficulty))
        return {
            "probe_mean": mean,
            "probe_log_variance": log_variance,
            "modality_difficulty": difficulty,
            "valid_modality_mask": modality_valid,
        }


class DailyAffectRegressionModel(nn.Module):
    """0906 EMA-bag structures with a scalar regression head."""

    def __init__(self, config: DailyAffectRegressionConfig | None = None) -> None:
        super().__init__()
        self.config = config or DailyAffectRegressionConfig()
        cfg = self.config
        if cfg.model_id not in SUPPORTED_MODEL_IDS:
            raise ValueError(f"unsupported daily-affect model_id: {cfg.model_id}")
        if cfg.temporal_policy not in STATIC_TEMPORAL_POLICIES:
            raise ValueError(f"unsupported static temporal_policy: {cfg.temporal_policy}")
        if (self.uses_dynamic_kernel or self.uses_global_kernel) and cfg.temporal_policy != "uniform":
            raise ValueError("learned-kernel models require temporal_policy=uniform")
        self.adapters = ModalityAdapters(
            mode=cfg.adapter_mode,
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            modality_count=cfg.modality_count,
        )
        self.window_self_attention = (
            nn.MultiheadAttention(cfg.hidden_dim, 1, dropout=cfg.dropout, batch_first=True)
            if self.uses_window_attention
            else None
        )
        self.window_query = nn.Parameter(torch.empty(cfg.hidden_dim)) if self.uses_window_attention else None
        if self.window_query is not None:
            nn.init.normal_(self.window_query, mean=0.0, std=0.02)
        self.evidence_score = nn.Linear(cfg.hidden_dim, 1)
        self.dropout = nn.Dropout(cfg.dropout)
        self.state_filter = AffectStateFilter(hidden_dim=cfg.hidden_dim) if self.uses_state else None
        self.prior_transition = (
            nn.Sequential(
                nn.LayerNorm(cfg.hidden_dim),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            )
            if self.uses_prior_guidance
            else None
        )
        self.prior_norm = nn.LayerNorm(cfg.hidden_dim) if self.uses_prior_guidance else None
        self.prior_query = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False) if self.uses_prior_guidance else None
        self.prior_keys = (
            nn.ModuleList([nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False) for _ in range(cfg.modality_count)])
            if self.uses_prior_guidance
            else None
        )
        self.prior_score = nn.Linear(cfg.hidden_dim, 1, bias=False) if self.uses_prior_guidance else None
        self.probe = (
            GaussianRegressionModalityProbe(hidden_dim=cfg.hidden_dim, modality_count=cfg.modality_count)
            if self.uses_regression_difficulty
            else None
        )
        self.kernel = DynamicEMAKernel(hidden_dim=cfg.hidden_dim, sequence_len=cfg.sequence_len) if self.uses_dynamic_kernel else None
        self.global_kernel = GlobalEMAKernel(sequence_len=cfg.sequence_len) if self.uses_global_kernel else None
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

    @property
    def uses_state(self) -> bool:
        return self.config.model_id in STATE_MODEL_IDS

    @property
    def uses_window_attention(self) -> bool:
        return self.config.model_id in WINDOW_ATTENTION_MODEL_IDS

    @property
    def uses_window_regression(self) -> bool:
        return self.config.model_id == "window_replicated"

    @property
    def uses_prior_guidance(self) -> bool:
        return self.config.model_id in PRIOR_MODEL_IDS

    @property
    def uses_regression_difficulty(self) -> bool:
        return self.config.model_id in ORDINAL_DIFFICULTY_MODEL_IDS

    @property
    def uses_dynamic_kernel(self) -> bool:
        return self.config.model_id in DYNAMIC_KERNEL_MODEL_IDS

    @property
    def uses_global_kernel(self) -> bool:
        return self.config.model_id in GLOBAL_KERNEL_MODEL_IDS

    @property
    def fixed_kernel_index(self) -> int | None:
        return {
            "dynamic_fixed_short": 0,
            "dynamic_fixed_medium": 1,
            "dynamic_fixed_long": 2,
        }.get(self.config.model_id)

    def forward(
        self,
        tokens: torch.Tensor,
        modality_mask: torch.Tensor,
        *,
        lambda_d_override: float | None = None,
        detach_difficulty_override: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        if tokens.ndim != 4:
            raise ValueError(f"tokens expected shape (B,L,M,D), got {tuple(tokens.shape)}")
        if tokens.shape[1] != self.config.sequence_len:
            raise ValueError(f"expected {self.config.sequence_len} windows, got {tokens.shape[1]}")
        if modality_mask.shape != tokens.shape[:3]:
            raise ValueError(f"modality_mask expected shape {tuple(tokens.shape[:3])}, got {tuple(modality_mask.shape)}")
        mask = modality_mask.to(dtype=torch.bool)
        adapted = self.adapters(tokens)
        window_mask = mask.any(dim=2)
        probe_outputs = self._probe_outputs(adapted, mask)
        if self.uses_window_attention:
            evidence, modality_weights, states = self._window_attention_evidence(adapted, mask)
        else:
            evidence, modality_weights, states = self._state_conditioned_evidence(
                adapted,
                mask,
                window_mask,
                probe_outputs["modality_difficulty"],
                lambda_d=float(self.config.lambda_d if lambda_d_override is None else lambda_d_override),
                detach_difficulty=self.config.detach_difficulty
                if detach_difficulty_override is None
                else bool(detach_difficulty_override),
            )
        temporal_weights, kernel_mixture = self._temporal_weights(states, window_mask)
        if self.uses_window_regression:
            window_prediction = self.head(self.dropout(states))
            prediction = torch.sum(window_prediction * temporal_weights[:, :, None], dim=1)
        else:
            summary = torch.sum(states * temporal_weights[:, :, None], dim=1)
            prediction = self.head(self.dropout(summary))
            window_prediction = None
        if prediction.shape[-1] == 1:
            prediction = prediction.squeeze(-1)
            if window_prediction is not None:
                window_prediction = window_prediction.squeeze(-1)
        outputs: dict[str, torch.Tensor] = {
            "prediction": prediction,
            "modality_weights": modality_weights,
            "states": states,
            "temporal_weights": temporal_weights,
            "kernel_mixture": kernel_mixture,
            "window_mask": window_mask,
            **probe_outputs,
        }
        if window_prediction is not None:
            outputs["window_prediction"] = window_prediction
        return outputs

    def _probe_outputs(self, adapted: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.probe is not None:
            return self.probe(adapted, mask)
        valid = mask.any(dim=1)
        zeros = adapted.new_zeros((adapted.shape[0], adapted.shape[2]))
        return {
            "probe_mean": zeros,
            "probe_log_variance": zeros,
            "modality_difficulty": torch.where(valid, zeros, torch.ones_like(zeros)),
            "valid_modality_mask": valid,
        }

    def _temporal_weights(self, states: torch.Tensor, window_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.uses_dynamic_kernel:
            assert self.kernel is not None
            fixed_index = self.fixed_kernel_index
            if fixed_index is None:
                return self.kernel(states, window_mask)
            base = self.kernel.bases[fixed_index].to(device=states.device, dtype=states.dtype)
            weights = base.view(1, -1).expand(states.shape[0], -1) * window_mask.to(dtype=states.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            mixture = states.new_zeros((states.shape[0], 3))
            mixture[:, fixed_index] = 1.0
            return weights, mixture
        if self.uses_global_kernel:
            assert self.global_kernel is not None
            return self.global_kernel(states, window_mask)
        return (
            _static_temporal_weights(window_mask, policy=self.config.temporal_policy, dtype=states.dtype),
            states.new_zeros((states.shape[0], 3)),
        )

    def _window_attention_evidence(
        self,
        adapted: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.window_self_attention is not None
        assert self.window_query is not None
        batch_size, sequence_len, modality_count, hidden_dim = adapted.shape
        flat_tokens = adapted.reshape(batch_size * sequence_len, modality_count, hidden_dim)
        flat_mask = mask.reshape(batch_size * sequence_len, modality_count)
        safe_mask = flat_mask.clone()
        empty_rows = ~safe_mask.any(dim=1)
        if torch.any(empty_rows):
            safe_mask[empty_rows, 0] = True
        flat_tokens = torch.where(flat_mask[:, :, None], flat_tokens, torch.zeros_like(flat_tokens))
        attended, _ = self.window_self_attention(flat_tokens, flat_tokens, flat_tokens, key_padding_mask=~safe_mask, need_weights=False)
        attended = self.dropout(attended)
        scores = torch.matmul(attended, self.window_query).masked_fill(~flat_mask, torch.finfo(attended.dtype).min)
        weights = _masked_softmax(scores, flat_mask, dim=1)
        evidence = torch.sum(attended * weights[:, :, None], dim=1)
        evidence = evidence.reshape(batch_size, sequence_len, hidden_dim)
        return evidence, weights.reshape(batch_size, sequence_len, modality_count), evidence

    def _state_conditioned_evidence(
        self,
        adapted: torch.Tensor,
        mask: torch.Tensor,
        window_mask: torch.Tensor,
        difficulty: torch.Tensor,
        *,
        lambda_d: float,
        detach_difficulty: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_scores = self.evidence_score(adapted).squeeze(-1)
        if not self.uses_prior_guidance:
            weights = _masked_softmax(base_scores, mask, dim=2)
            evidence = torch.sum(adapted * weights[:, :, :, None], dim=2)
            if self.uses_state:
                assert self.state_filter is not None
                return evidence, weights, self.state_filter(evidence, window_mask)
            return evidence, weights, evidence
        assert self.state_filter is not None
        assert self.prior_transition is not None
        assert self.prior_norm is not None
        assert self.prior_query is not None
        assert self.prior_keys is not None
        assert self.prior_score is not None
        state = self.state_filter.initial_state(adapted[:, :, 0, :])
        evidence_steps: list[torch.Tensor] = []
        weight_steps: list[torch.Tensor] = []
        state_steps: list[torch.Tensor] = []
        for step in range(adapted.shape[1]):
            prior = self.prior_norm(state + self.prior_transition(state))
            prior_query = self.prior_query(prior)[:, None, :]
            keys = torch.stack([layer(adapted[:, step, modality]) for modality, layer in enumerate(self.prior_keys)], dim=1)
            compatibility = self.prior_score(torch.tanh(prior_query + keys)).squeeze(-1)
            scores = base_scores[:, step] + compatibility
            if self.uses_regression_difficulty:
                current_difficulty = difficulty.detach() if detach_difficulty else difficulty
                scores = scores - float(lambda_d) * current_difficulty
            weights = _masked_softmax(scores, mask[:, step], dim=1)
            evidence_step = torch.sum(adapted[:, step] * weights[:, :, None], dim=1)
            state = self.state_filter.update_step(evidence_step, state, window_mask[:, step])
            evidence_steps.append(evidence_step)
            weight_steps.append(weights)
            state_steps.append(state)
        return (
            torch.stack(evidence_steps, dim=1),
            torch.stack(weight_steps, dim=1),
            torch.stack(state_steps, dim=1),
        )


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor, *, dim: int) -> torch.Tensor:
    safe_mask = mask.to(dtype=torch.bool)
    masked = scores.masked_fill(~safe_mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked, dim=dim) * safe_mask.to(dtype=scores.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-6)
