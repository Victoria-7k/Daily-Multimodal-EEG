from __future__ import annotations

import pytest

from daily_multimodal.embeddings.video_domain_robust import encode_domain_targets


def test_encode_domain_targets_is_stable_and_keeps_unknown_as_ignore_index():
    targets = encode_domain_targets(
        subject_ids=["sub-02", "sub-01", "sub-02", ""],
        session_ids=["2025-07-02", "2025-07-01", "2025-07-02", None],
    )

    assert targets.subject_classes == ["sub-01", "sub-02"]
    assert targets.session_classes == ["2025-07-01", "2025-07-02"]
    assert targets.subject_targets.tolist() == [1, 0, 1, -100]
    assert targets.session_targets.tolist() == [1, 0, 1, -100]


def test_gradient_reversal_flips_embedding_gradient_sign():
    torch = pytest.importorskip("torch")
    from daily_multimodal.embeddings.video_domain_robust import gradient_reverse

    x = torch.tensor([[1.0, -2.0]], requires_grad=True)
    y = gradient_reverse(x, lambda_=0.5)
    loss = (y**2).sum()
    loss.backward()

    assert torch.allclose(x.grad, torch.tensor([[-1.0, 2.0]]))


def test_domain_adversarial_heads_return_subject_and_session_logits():
    torch = pytest.importorskip("torch")
    from daily_multimodal.embeddings.video_domain_robust import DomainAdversarialHeads

    heads = DomainAdversarialHeads(
        embedding_dim=256,
        subject_count=3,
        session_count=4,
        hidden_dim=8,
        grl_lambda=0.25,
    )
    embeddings = torch.randn(5, 256, requires_grad=True)

    outputs = heads(embeddings)
    total_loss, metrics = heads.loss(
        outputs,
        subject_targets=torch.tensor([0, 1, 2, -100, 1]),
        session_targets=torch.tensor([3, 2, 1, 0, -100]),
    )
    total_loss.backward()

    assert outputs["subject_logits"].shape == (5, 3)
    assert outputs["session_logits"].shape == (5, 4)
    assert metrics["subject_adversarial_weight"] == 1.0
    assert metrics["session_adversarial_weight"] == 1.0
    assert embeddings.grad is not None
