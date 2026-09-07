from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .adapters import ModalityAdapters
from .affect_state_filter import AffectStateFilter
from .dynamic_ema_kernel import DynamicEMAKernel, GlobalEMAKernel
from .modality_probe import CumulativeOrdinalModalityProbe


@dataclass(frozen=True)
class DailyAffectConfig:
    model_id: str = "bag_static"
    input_dim: int = 256
    hidden_dim: int = 128
    sequence_len: int = 23
    modality_count: int = 4
    adapter_mode: str = "shared"
    dropout: float = 0.1
    lambda_d: float = 0.25
    beta_ord: float = 0.25
    probe_kind: str = "cumulative"
    difficulty_mode: str = "auto"
    detach_difficulty: bool = True
    temporal_policy: str = "uniform"


class DailyAffectOrdinalModel(nn.Module):
    def __init__(self, config: DailyAffectConfig | None = None) -> None:
        super().__init__()
        self.config = config or DailyAffectConfig()
        cfg = self.config
        if cfg.model_id not in SUPPORTED_MODEL_IDS:
            raise ValueError(f"unsupported daily-affect model_id: {cfg.model_id}")
        if cfg.temporal_policy not in STATIC_TEMPORAL_POLICIES:
            raise ValueError(f"unsupported static temporal_policy: {cfg.temporal_policy}")
        if cfg.probe_kind not in {"cumulative", "categorical"}:
            raise ValueError(f"unsupported probe_kind: {cfg.probe_kind}")
        if cfg.difficulty_mode not in {"auto", "none", "entropy", "ordinal"}:
            raise ValueError(f"unsupported difficulty_mode: {cfg.difficulty_mode}")
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
        self.probe = CumulativeOrdinalModalityProbe(
            hidden_dim=cfg.hidden_dim,
            modality_count=cfg.modality_count,
            kind=cfg.probe_kind,
        )
        self.register_buffer("probe_temperatures", torch.ones(cfg.modality_count, dtype=torch.float32))
        self.kernel = DynamicEMAKernel(hidden_dim=cfg.hidden_dim, sequence_len=cfg.sequence_len) if self.uses_dynamic_kernel else None
        self.global_kernel = GlobalEMAKernel(sequence_len=cfg.sequence_len) if self.uses_global_kernel else None
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 5),
        )

    @property
    def uses_state(self) -> bool:
        return self.config.model_id in STATE_MODEL_IDS

    @property
    def uses_window_attention(self) -> bool:
        return self.config.model_id in WINDOW_ATTENTION_MODEL_IDS

    @property
    def uses_window_replicated_supervision(self) -> bool:
        return self.config.model_id == "window_replicated"

    @property
    def uses_probe(self) -> bool:
        return self.config.model_id in PROBE_MODEL_IDS

    @property
    def uses_prior_guidance(self) -> bool:
        return self.config.model_id in PRIOR_MODEL_IDS

    @property
    def uses_difficulty_penalty(self) -> bool:
        return self.resolved_difficulty_mode != "none"

    @property
    def resolved_difficulty_mode(self) -> str:
        # The difficulty penalty is defined only inside the state-prior router.
        if not self.uses_prior_guidance:
            return "none"
        if self.config.difficulty_mode != "auto":
            return self.config.difficulty_mode
        return "ordinal" if self.config.model_id in ORDINAL_DIFFICULTY_MODEL_IDS else "none"

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

    def set_probe_temperatures(self, temperatures: torch.Tensor) -> None:
        values = temperatures.detach().to(device=self.probe_temperatures.device, dtype=self.probe_temperatures.dtype)
        if tuple(values.shape) != tuple(self.probe_temperatures.shape):
            raise ValueError(f"probe temperatures expected shape {tuple(self.probe_temperatures.shape)}, got {tuple(values.shape)}")
        self.probe_temperatures.copy_(values.clamp_min(1e-6))

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
        probe_outputs = self.probe(
            adapted,
            mask,
            temperatures=self.probe_temperatures,
            beta_ord=self.config.beta_ord,
        )
        window_mask = mask.any(dim=2)
        if self.uses_window_attention:
            evidence, modality_weights, states = self._window_attention_evidence(adapted, mask)
        else:
            evidence, modality_weights, states = self._state_conditioned_evidence(
                adapted,
                mask,
                window_mask,
                probe_outputs["modality_difficulty"],
                probe_outputs["probe_entropy"],
                lambda_d=float(self.config.lambda_d if lambda_d_override is None else lambda_d_override),
                detach_difficulty=self.config.detach_difficulty if detach_difficulty_override is None else bool(detach_difficulty_override),
            )
        if self.uses_dynamic_kernel:
            assert self.kernel is not None
            fixed_index = self.fixed_kernel_index
            if fixed_index is None:
                temporal_weights, kernel_mixture = self.kernel(states, window_mask)
            else:
                base = self.kernel.bases[fixed_index].to(device=states.device, dtype=states.dtype)
                temporal_weights = base.view(1, -1).expand(states.shape[0], -1) * window_mask.to(dtype=states.dtype)
                temporal_weights = temporal_weights / temporal_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
                kernel_mixture = states.new_zeros((states.shape[0], 3))
                kernel_mixture[:, fixed_index] = 1.0
        elif self.uses_global_kernel:
            assert self.global_kernel is not None
            temporal_weights, kernel_mixture = self.global_kernel(states, window_mask)
        else:
            temporal_weights = _static_temporal_weights(
                window_mask,
                policy=self.config.temporal_policy,
                dtype=states.dtype,
            )
            kernel_mixture = states.new_zeros((states.shape[0], 3))
        if self.uses_window_replicated_supervision:
            window_logits = self.head(self.dropout(states))
            window_probs = torch.softmax(window_logits, dim=2)
            probs = torch.sum(window_probs * temporal_weights[:, :, None], dim=1)
            logits = torch.log(probs.clamp_min(1e-8))
        else:
            summary = torch.sum(states * temporal_weights[:, :, None], dim=1)
            logits = self.head(self.dropout(summary))
            probs = torch.softmax(logits, dim=1)
        expected_score = torch.sum(probs * torch.arange(1, 6, dtype=probs.dtype, device=probs.device).view(1, -1), dim=1)
        outputs = {
            "class_logits": logits,
            "probabilities": probs,
            "expected_score": expected_score,
            "predicted_class": torch.argmax(logits, dim=1),
            "modality_weights": modality_weights,
            "states": states,
            "temporal_weights": temporal_weights,
            "kernel_mixture": kernel_mixture,
            **probe_outputs,
        }
        if self.uses_window_replicated_supervision:
            outputs.update(
                {
                    "window_class_logits": window_logits,
                    "window_probabilities": window_probs,
                    "window_expected_score": torch.sum(
                        window_probs * torch.arange(1, 6, dtype=window_probs.dtype, device=window_probs.device).view(1, 1, -1),
                        dim=2,
                    ),
                    "window_mask": window_mask,
                }
            )
        return outputs

    def _window_attention_evidence(
        self,
        adapted: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fuse each 10-second modality set with the legacy attention/query path."""

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
        return (
            evidence.reshape(batch_size, sequence_len, hidden_dim),
            weights.reshape(batch_size, sequence_len, modality_count),
            evidence.reshape(batch_size, sequence_len, hidden_dim),
        )

    def _state_conditioned_evidence(
        self,
        adapted: torch.Tensor,
        mask: torch.Tensor,
        window_mask: torch.Tensor,
        modality_difficulty: torch.Tensor,
        modality_entropy: torch.Tensor,
        *,
        lambda_d: float,
        detach_difficulty: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_scores = self.evidence_score(adapted).squeeze(-1)
        if not self.uses_prior_guidance:
            modality_weights = _masked_softmax(base_scores, mask, dim=2)
            evidence = torch.sum(adapted * modality_weights[:, :, :, None], dim=2)
            if self.uses_state:
                assert self.state_filter is not None
                return evidence, modality_weights, self.state_filter(evidence, window_mask)
            return evidence, modality_weights, evidence

        assert self.state_filter is not None
        assert self.prior_transition is not None
        assert self.prior_norm is not None
        assert self.prior_query is not None
        assert self.prior_keys is not None
        assert self.prior_score is not None
        state = self.state_filter.initial_state(adapted[:, :, 0, :])
        evidence_steps: list[torch.Tensor] = []
        weight_steps: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        for step in range(adapted.shape[1]):
            prior = self.prior_norm(state + self.prior_transition(state))
            prior_query = self.prior_query(prior)[:, None, :]
            modality_keys = torch.stack(
                [layer(adapted[:, step, modality]) for modality, layer in enumerate(self.prior_keys)],
                dim=1,
            )
            compatibility = self.prior_score(torch.tanh(prior_query + modality_keys)).squeeze(-1)
            scores = base_scores[:, step] + compatibility
            if self.uses_difficulty_penalty:
                difficulty = modality_difficulty
                if self.resolved_difficulty_mode == "entropy":
                    # The P1/P2 baseline suppresses only predictive entropy.
                    difficulty = modality_entropy
                if detach_difficulty:
                    difficulty = difficulty.detach()
                scores = scores - float(lambda_d) * difficulty
            weights = _masked_softmax(scores, mask[:, step], dim=1)
            evidence_step = torch.sum(adapted[:, step] * weights[:, :, None], dim=1)
            state = self.state_filter.update_step(evidence_step, state, window_mask[:, step])
            evidence_steps.append(evidence_step)
            weight_steps.append(weights)
            states.append(state)
        return torch.stack(evidence_steps, dim=1), torch.stack(weight_steps, dim=1), torch.stack(states, dim=1)


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor, *, dim: int) -> torch.Tensor:
    masked = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked, dim=dim)
    weights = torch.where(mask, weights, torch.zeros_like(weights))
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-6)


STATIC_TEMPORAL_POLICIES = {
    "uniform",
    "last_10s",
    "last_30s",
    "last_60s",
    "first_30s",
    "kernel_short",
    "kernel_medium",
    "kernel_long",
}


def _static_temporal_weights(
    window_mask: torch.Tensor,
    *,
    policy: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    sequence_len = window_mask.shape[1]
    valid = window_mask.to(dtype=dtype)
    if policy == "uniform":
        raw = valid
    elif policy in {"last_10s", "last_30s", "last_60s", "first_30s"}:
        ranges = {
            "last_10s": (sequence_len - 1, sequence_len),
            "last_30s": (max(0, sequence_len - 5), sequence_len),
            "last_60s": (max(0, sequence_len - 11), sequence_len),
            "first_30s": (0, min(sequence_len, 5)),
        }
        start, end = ranges[policy]
        selected = torch.zeros((sequence_len,), device=window_mask.device, dtype=dtype)
        selected[start:end] = 1.0
        raw = valid * selected.view(1, -1)
    else:
        tau = {"kernel_short": 15.0 / 5.0, "kernel_medium": 45.0 / 5.0, "kernel_long": 120.0 / 5.0}[policy]
        positions = torch.arange(sequence_len, device=window_mask.device, dtype=dtype)
        distance_from_recent = (sequence_len - 1) - positions
        raw = valid * torch.exp(-distance_from_recent / tau).view(1, -1)
    return raw / raw.sum(dim=1, keepdim=True).clamp_min(1e-6)


SUPPORTED_MODEL_IDS = {
    "window_replicated",
    "bag_static",
    "state_uniform",
    "prior_uniform",
    "prior_ordD_uniform",
    "dynamic_kernel",
    "dynamic_kernel_no_prior",
    "dynamic_kernel_prior_uniform",
    "dynamic_fixed_short",
    "dynamic_fixed_medium",
    "dynamic_fixed_long",
    "global_kernel_no_prior",
}
WINDOW_ATTENTION_MODEL_IDS = {"window_replicated", "bag_static"}
STATE_MODEL_IDS = SUPPORTED_MODEL_IDS - {"window_replicated", "bag_static"}
PROBE_MODEL_IDS = SUPPORTED_MODEL_IDS - {"window_replicated", "bag_static", "state_uniform"}
PRIOR_MODEL_IDS = {
    "prior_uniform",
    "prior_ordD_uniform",
    "dynamic_kernel_prior_uniform",
    "dynamic_kernel",
    "dynamic_fixed_short",
    "dynamic_fixed_medium",
    "dynamic_fixed_long",
}
ORDINAL_DIFFICULTY_MODEL_IDS = {"prior_ordD_uniform", "dynamic_kernel", "dynamic_fixed_short", "dynamic_fixed_medium", "dynamic_fixed_long"}
DYNAMIC_KERNEL_MODEL_IDS = {
    "dynamic_kernel",
    "dynamic_kernel_no_prior",
    "dynamic_kernel_prior_uniform",
    "dynamic_fixed_short",
    "dynamic_fixed_medium",
    "dynamic_fixed_long",
}
GLOBAL_KERNEL_MODEL_IDS = {"global_kernel_no_prior"}
