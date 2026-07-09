import numpy as np
import pytest

from daily_multimodal.training.cross_attention_fusion import (
    FusionDataset,
    LearnableAttentionConfig,
    audit_model_normalization,
    fit_learnable_cross_attention,
    fit_target_normalization,
    fit_token_normalization,
)


def test_normalization_is_unchanged_by_validation_and_test_outliers():
    dataset = _dataset_with_train_rows_and_outlier_holdout()
    train = np.asarray([0, 1, 2, 3])
    first = fit_token_normalization(dataset.tokens, dataset.token_mask, train)
    changed = dataset.tokens.copy()
    changed[4:] = 1.0e9
    second = fit_token_normalization(changed, dataset.token_mask, train)
    np.testing.assert_allclose(first.x_mean, second.x_mean)
    np.testing.assert_allclose(first.x_std, second.x_std)


def test_target_normalization_is_unchanged_by_validation_and_test_outliers():
    dataset = _dataset_with_train_rows_and_outlier_holdout()
    train = np.asarray([0, 1, 2, 3])
    first = fit_target_normalization(dataset.target, train)
    changed = dataset.target.copy()
    changed[4:] = 1.0e9
    second = fit_target_normalization(changed, train)
    assert first.y_mean == pytest.approx(second.y_mean)
    assert first.y_std == pytest.approx(second.y_std)


def test_attention_fit_records_training_only_normalization_hash():
    torch = pytest.importorskip("torch")
    dataset = _dataset_with_train_rows_and_outlier_holdout()
    train = np.asarray([0, 1, 2, 3])
    val = np.asarray([4, 5])
    model = fit_learnable_cross_attention(
        dataset,
        train_indices=train,
        val_indices=val,
        config=LearnableAttentionConfig(
            token_dim=8,
            epochs=2,
            batch_size=2,
            patience=2,
            seed=3,
        ),
        torch_module=torch,
    )
    audit = audit_model_normalization(model, dataset, train)
    assert audit["fit_scope"] == "train_only"
    assert audit["fit_sample_count"] == len(train)
    assert audit["target_fit_sample_count"] == len(train)
    assert audit["fit_sample_id_sha256"]
    assert audit["feature_statistics_sha256"]
    assert audit["target_statistics_sha256"]
    assert audit["verified"] is True


def _dataset_with_train_rows_and_outlier_holdout():
    tokens = np.zeros((6, 4, 256), dtype=np.float32)
    for row in range(6):
        tokens[row, :, 0] = float(row)
        tokens[row, :, 1] = float(row + 1)
    mask = np.ones((6, 4), dtype=bool)
    return FusionDataset(
        name="normalization",
        modalities=("eeg", "wear", "video", "audio"),
        sample_id=np.asarray([f"s{idx}" for idx in range(6)], dtype=str),
        event_id=np.asarray([f"e{idx}" for idx in range(6)], dtype=str),
        subject_id=np.asarray(["sub-01"] * 6, dtype=str),
        target=np.asarray([0.0, 1.0, 2.0, 3.0, 100.0, 200.0], dtype=np.float32),
        tokens=tokens,
        token_mask=mask,
        branch_profiles={"eeg": "eeg", "wear": "wear", "video": "video", "audio": "audio"},
        target_label="fatigue",
    )
