from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


IGNORE_INDEX = -100


@dataclass(frozen=True)
class DomainTargets:
    subject_targets: np.ndarray
    session_targets: np.ndarray
    subject_classes: list[str]
    session_classes: list[str]


def encode_domain_targets(
    *,
    subject_ids: Iterable[Any],
    session_ids: Iterable[Any],
    ignore_index: int = IGNORE_INDEX,
) -> DomainTargets:
    subject_values = [_normalize_label(value) for value in subject_ids]
    session_values = [_normalize_label(value) for value in session_ids]
    subject_classes = sorted({value for value in subject_values if value})
    session_classes = sorted({value for value in session_values if value})
    subject_lookup = {value: index for index, value in enumerate(subject_classes)}
    session_lookup = {value: index for index, value in enumerate(session_classes)}
    return DomainTargets(
        subject_targets=_target_array(subject_values, subject_lookup, ignore_index=ignore_index),
        session_targets=_target_array(session_values, session_lookup, ignore_index=ignore_index),
        subject_classes=subject_classes,
        session_classes=session_classes,
    )


def gradient_reverse(embeddings: Any, *, lambda_: float = 1.0) -> Any:
    torch = _require_torch()
    return _GradientReverse.apply(embeddings, float(lambda_))


class _GradientReverse:
    @staticmethod
    def apply(embeddings: Any, lambda_: float) -> Any:
        torch = _require_torch()

        class _GradientReverseFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, values: Any, scale: float) -> Any:
                ctx.scale = float(scale)
                return values.view_as(values)

            @staticmethod
            def backward(ctx: Any, grad_output: Any) -> tuple[Any, None]:
                return -ctx.scale * grad_output, None

        return _GradientReverseFn.apply(embeddings, float(lambda_))


class DomainAdversarialHeads:
    def __init__(
        self,
        *,
        embedding_dim: int,
        subject_count: int,
        session_count: int,
        hidden_dim: int = 128,
        grl_lambda: float = 1.0,
        subject_weight: float = 1.0,
        session_weight: float = 1.0,
    ) -> None:
        torch = _require_torch()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if subject_count <= 0:
            raise ValueError("subject_count must be positive")
        if session_count <= 0:
            raise ValueError("session_count must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.embedding_dim = int(embedding_dim)
        self.subject_count = int(subject_count)
        self.session_count = int(session_count)
        self.hidden_dim = int(hidden_dim)
        self.grl_lambda = float(grl_lambda)
        self.subject_weight = float(subject_weight)
        self.session_weight = float(session_weight)
        self._module = _DomainAdversarialTorchModule(
            torch=torch,
            embedding_dim=self.embedding_dim,
            subject_count=self.subject_count,
            session_count=self.session_count,
            hidden_dim=self.hidden_dim,
            grl_lambda=self.grl_lambda,
        )

    def __call__(self, embeddings: Any) -> dict[str, Any]:
        return self._module(embeddings)

    def parameters(self) -> Any:
        return self._module.parameters()

    def train(self, mode: bool = True) -> "DomainAdversarialHeads":
        self._module.train(mode)
        return self

    def eval(self) -> "DomainAdversarialHeads":
        self._module.eval()
        return self

    def loss(
        self,
        outputs: dict[str, Any],
        *,
        subject_targets: Any,
        session_targets: Any,
        ignore_index: int = IGNORE_INDEX,
    ) -> tuple[Any, dict[str, float]]:
        torch = _require_torch()
        subject_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=int(ignore_index))
        session_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=int(ignore_index))
        subject_loss = subject_loss_fn(outputs["subject_logits"], subject_targets)
        session_loss = session_loss_fn(outputs["session_logits"], session_targets)
        total = self.subject_weight * subject_loss + self.session_weight * session_loss
        metrics = {
            "subject_adversarial_loss": float(subject_loss.detach().cpu().item()),
            "session_adversarial_loss": float(session_loss.detach().cpu().item()),
            "subject_adversarial_weight": float(self.subject_weight),
            "session_adversarial_weight": float(self.session_weight),
            "gradient_reversal_lambda": float(self.grl_lambda),
        }
        return total, metrics


def _normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _target_array(values: list[str | None], lookup: dict[str, int], *, ignore_index: int) -> np.ndarray:
    return np.asarray([lookup[value] if value in lookup else int(ignore_index) for value in values], dtype=np.int64)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised on environments without torch.
        raise ImportError("video domain robustness requires PyTorch") from exc
    return torch


class _DomainAdversarialTorchModule:
    def __new__(
        cls,
        *,
        torch: Any,
        embedding_dim: int,
        subject_count: int,
        session_count: int,
        hidden_dim: int,
        grl_lambda: float,
    ) -> Any:
        class _Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.grl_lambda = float(grl_lambda)
                self.subject_head = _make_head(embedding_dim, hidden_dim, subject_count, torch=torch)
                self.session_head = _make_head(embedding_dim, hidden_dim, session_count, torch=torch)

            def forward(self, embeddings: Any) -> dict[str, Any]:
                reversed_embeddings = gradient_reverse(embeddings, lambda_=self.grl_lambda)
                return {
                    "subject_logits": self.subject_head(reversed_embeddings),
                    "session_logits": self.session_head(reversed_embeddings),
                }

        return _Module()


def _make_head(input_dim: int, hidden_dim: int, output_dim: int, *, torch: Any) -> Any:
    return torch.nn.Sequential(
        torch.nn.Linear(int(input_dim), int(hidden_dim)),
        torch.nn.ReLU(),
        torch.nn.Linear(int(hidden_dim), int(output_dim)),
    )
