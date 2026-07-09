import numpy as np
import pytest

from daily_multimodal.training.within_subject_metrics import (
    aggregate_event_predictions,
    audit_fold_target_variance,
    fit_predict_concat_ridge,
    fit_predict_train_mean,
    regression_metrics,
    summarize_pooled_oof,
    summarize_subject_oof,
)


def test_subject_metrics_are_recomputed_from_complete_oof_not_fold_mean():
    records = make_records(
        sample_ids=["s0", "s1", "s2", "s3"],
        subjects=["sub-01"] * 4,
        sessions=["ses-01"] * 4,
        events=["e1", "e2", "e3", "e4"],
        folds=[0, 1, 2, 3],
        target=[0.0, 0.0, 10.0, 10.0],
        prediction=[0.0, 0.0, 0.0, 0.0],
    )
    result = summarize_subject_oof(records, expected_sample_ids=["s0", "s1", "s2", "s3"])
    assert result["window"]["rmse"] == pytest.approx(np.sqrt(50.0))
    assert "fold_rmse_mean" not in result["window"]
    assert result["oof_complete"] is True


def test_pooled_pearson_is_centered_within_subject():
    records = make_records(
        subjects=["sub-01", "sub-01", "sub-02", "sub-02"],
        sessions=["ses-01", "ses-01", "ses-01", "ses-01"],
        events=["a", "b", "c", "d"],
        folds=[0, 1, 0, 1],
        target=[0.0, 1.0, 100.0, 101.0],
        prediction=[0.0, 1.0, 200.0, 201.0],
    )
    result = summarize_pooled_oof(records)
    assert result["within_subject_centered_pearson"] == pytest.approx(1.0)
    assert result["pearson_definition"] == "center prediction and target by subject OOF mean"


def test_subject_oof_fails_when_expected_samples_are_missing_or_duplicated():
    records = make_records(sample_ids=["s0", "s1"], target=[0.0, 1.0], prediction=[0.0, 1.0])
    with pytest.raises(ValueError, match="expected OOF sample coverage"):
        summarize_subject_oof(records, expected_sample_ids=["s0", "s1", "s2"])
    duplicated = make_records(sample_ids=["s0", "s0"], target=[0.0, 1.0], prediction=[0.0, 1.0])
    with pytest.raises(ValueError, match="duplicate OOF sample"):
        summarize_subject_oof(duplicated, expected_sample_ids=["s0", "s1"])


def test_subject_oof_fails_when_unexpected_sample_appears():
    records = make_records(sample_ids=["s0", "unexpected"], target=[0.0, 1.0], prediction=[0.0, 1.0])
    with pytest.raises(ValueError, match="unexpected OOF sample"):
        summarize_subject_oof(records, expected_sample_ids=["s0", "s1"])


def test_event_oof_averages_window_predictions():
    records = make_records(
        subjects=["sub-01"] * 4,
        sessions=["ses-01", "ses-01", "ses-01", "ses-01"],
        events=["e1", "e1", "e2", "e2"],
        folds=[0, 0, 1, 1],
        target=[1.0, 1.0, 3.0, 3.0],
        prediction=[0.0, 2.0, 2.0, 4.0],
    )
    event_records = aggregate_event_predictions(records)
    assert event_records.prediction.tolist() == [1.0, 3.0]
    assert regression_metrics(event_records.prediction, event_records.target)["rmse"] == 0.0


def test_event_aggregation_rejects_inconsistent_event_targets():
    records = make_records(
        subjects=["sub-01", "sub-01"],
        sessions=["ses-01", "ses-01"],
        events=["e1", "e1"],
        folds=[0, 0],
        target=[1.0, 2.0],
        prediction=[1.0, 2.0],
    )
    with pytest.raises(ValueError, match="inconsistent target"):
        aggregate_event_predictions(records)


def test_event_aggregation_keeps_same_event_id_separate_across_sessions():
    records = make_records(
        subjects=["sub-01", "sub-01"],
        sessions=["ses-01", "ses-02"],
        events=["e1", "e1"],
        folds=[0, 1],
        target=[1.0, 3.0],
        prediction=[1.0, 3.0],
    )
    event_records = aggregate_event_predictions(records)
    assert event_records.session_id.tolist() == ["ses-01", "ses-02"]
    assert event_records.event_id.tolist() == ["e1", "e1"]
    assert event_records.target.tolist() == [1.0, 3.0]


def test_degenerate_training_target_is_rejected_and_constant_test_pearson_is_null():
    target = np.asarray([1.0, 1.0, 1.0, 2.0, 2.0])
    with pytest.raises(ValueError, match="degenerate_train_target"):
        audit_fold_target_variance(
            target,
            train=np.asarray([0, 1, 2]),
            val=np.asarray([3]),
            test=np.asarray([4]),
        )
    metrics = regression_metrics(prediction=np.asarray([0.5, 0.5]), target=np.asarray([2.0, 2.0]))
    assert metrics["pearson"] is None


def test_baselines_fit_only_training_rows():
    target = np.asarray([1.0, 2.0, 3.0, 100.0], dtype=np.float32)
    train = np.asarray([0, 1, 2])
    test = np.asarray([3])
    tokens = np.zeros((4, 2, 3), dtype=np.float32)
    tokens[:, 0, 0] = np.asarray([1.0, 2.0, 3.0, 1000.0])
    tokens[:, 1, 1] = np.asarray([3.0, 2.0, 1.0, -1000.0])
    mask = np.ones((4, 2), dtype=bool)
    mean_prediction = fit_predict_train_mean(target, train, test)
    ridge_prediction, audit = fit_predict_concat_ridge(tokens, mask, target, train, test, alpha=10.0)
    assert np.all(mean_prediction == np.mean(target[train]))
    assert ridge_prediction.shape == (1,)
    assert audit["normalization_fit_scope"] == "train_only"
    assert audit["alpha"] == 10.0


def make_records(
    *,
    sample_ids=None,
    subjects=None,
    sessions=None,
    events=None,
    folds=None,
    target=None,
    prediction=None,
):
    target = [0.0, 1.0] if target is None else target
    count = len(target)
    prediction = target if prediction is None else prediction
    sample_ids = [f"s{idx}" for idx in range(count)] if sample_ids is None else sample_ids
    subjects = ["sub-01"] * count if subjects is None else subjects
    sessions = ["ses-01"] * count if sessions is None else sessions
    events = [f"e{idx}" for idx in range(count)] if events is None else events
    folds = [0] * count if folds is None else folds
    from daily_multimodal.training.within_subject_metrics import PredictionRecords

    return PredictionRecords(
        sample_id=np.asarray(sample_ids, dtype=str),
        event_id=np.asarray(events, dtype=str),
        subject_id=np.asarray(subjects, dtype=str),
        session_id=np.asarray(sessions, dtype=str),
        fold_id=np.asarray(folds),
        target=np.asarray(target, dtype=np.float32),
        prediction=np.asarray(prediction, dtype=np.float32),
        attention=None,
        model_name="model",
        experiment="experiment",
        protocol="protocol",
    )
