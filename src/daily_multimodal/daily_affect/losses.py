from __future__ import annotations

import torch
from torch.nn import functional as F


def ordinal_probe_loss(
    logits: torch.Tensor,
    raw_labels: torch.Tensor,
    valid_modality_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.new_zeros(())
    labels = torch.clamp(torch.round(raw_labels), 1, 5)
    thresholds = torch.arange(1, 5, dtype=raw_labels.dtype, device=raw_labels.device).view(1, 1, 4)
    target = (labels.view(-1, 1, 1) > thresholds).to(dtype=logits.dtype)
    target = target.expand_as(logits)
    valid = valid_modality_mask.to(dtype=torch.bool)
    if not bool(valid.any()):
        return logits.new_zeros(())
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dim=-1)
    return loss[valid].mean()


def classification_loss(
    logits: torch.Tensor,
    labels_zero_based: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    return F.cross_entropy(logits, labels_zero_based.to(dtype=torch.long), weight=class_weights)


def cumulative_probability_ordinal_loss(
    logits: torch.Tensor,
    labels_zero_based: torch.Tensor,
) -> torch.Tensor:
    """Penalize probability mass that crosses more ordinal thresholds.

    The final head remains a five-class softmax classifier.  This term compares
    its four cumulative probabilities against the corresponding ordinal target,
    so a distant error contributes at more thresholds than an adjacent error.
    """

    if logits.numel() == 0:
        return logits.new_zeros(())
    probabilities = torch.softmax(logits, dim=-1)
    cumulative = torch.cumsum(probabilities, dim=-1)[..., :-1]
    thresholds = torch.arange(4, device=logits.device, dtype=labels_zero_based.dtype)
    target = (labels_zero_based.view(-1, 1) <= thresholds.view(1, -1)).to(dtype=logits.dtype)
    return torch.mean((cumulative - target) ** 2)


def within_subject_ordinal_ranking_loss(
    expected_score: torch.Tensor,
    raw_labels: torch.Tensor,
    subject_codes: torch.Tensor,
) -> torch.Tensor:
    """Pairwise ordinal ranking for unequal-label EMA bags from one subject."""

    if expected_score.numel() < 2:
        return expected_score.new_zeros(())
    label_delta = raw_labels[:, None] - raw_labels[None, :]
    same_subject = subject_codes[:, None] == subject_codes[None, :]
    upper = torch.triu(torch.ones_like(same_subject, dtype=torch.bool), diagonal=1)
    valid = same_subject & (label_delta != 0) & upper
    if not bool(valid.any()):
        return expected_score.new_zeros(())
    score_delta = expected_score[:, None] - expected_score[None, :]
    signed_margin = torch.sign(label_delta) * score_delta
    return F.softplus(-signed_margin[valid]).mean()


def categorical_probe_loss(
    logits: torch.Tensor,
    labels_zero_based: torch.Tensor,
    valid_modality_mask: torch.Tensor,
) -> torch.Tensor:
    """EMA-level five-class probe loss used by the P1 entropy baseline."""

    if logits.numel() == 0:
        return logits.new_zeros(())
    valid = valid_modality_mask.to(dtype=torch.bool)
    if not bool(valid.any()):
        return logits.new_zeros(())
    targets = labels_zero_based.view(-1, 1).expand(valid.shape)
    return F.cross_entropy(logits[valid], targets[valid].to(dtype=torch.long))
