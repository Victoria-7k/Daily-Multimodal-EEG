from __future__ import annotations

import json
from pathlib import Path

from daily_multimodal.training.video_personalization import run_video_personalization


def test_run_video_personalization_calibrates_residual_bias_by_k_events(tmp_path: Path):
    report = tmp_path / "loso.json"
    _write_fold_report(report)

    result = run_video_personalization(
        fold_report=report,
        variant="B1",
        out_json=tmp_path / "personalization.json",
        out_table=tmp_path / "personalization.md",
        k_events=[1, 3, 5],
    )

    by_protocol = {row["protocol"]: row for row in result["protocols"]}
    assert by_protocol["0-shot"]["eligible_subjects"] == 2
    assert by_protocol["k_event_1"]["eligible_subjects"] == 2
    assert by_protocol["k_event_1"]["rmse_mean"] < by_protocol["0-shot"]["rmse_mean"]
    assert by_protocol["k_event_3"]["calibration_count"] == 3
    assert by_protocol["k_event_5"]["skipped_subjects"]
    assert by_protocol["1-session"]["eligible_subjects"] == 2
    assert "calibration/test disjoint" in result["protocol_note"]


def test_run_video_personalization_includes_regularized_affine_protocols(tmp_path: Path):
    report = tmp_path / "loso_affine.json"
    _write_affine_fold_report(report)

    result = run_video_personalization(
        fold_report=report,
        variant="B1",
        out_json=tmp_path / "personalization.json",
        out_table=tmp_path / "personalization.md",
        k_events=[3],
        include_affine=True,
        affine_slope_penalty=0.01,
        affine_bias_penalty=0.01,
    )

    by_protocol = {row["protocol"]: row for row in result["protocols"]}
    assert "affine_k_event_3" in by_protocol
    affine = by_protocol["affine_k_event_3"]
    assert affine["eligible_subjects"] == 2
    assert affine["rmse_mean"] < by_protocol["k_event_3"]["rmse_mean"]
    first_subject = affine["subjects"][0]
    assert first_subject["affine_slope"] < 0.0
    assert not set(first_subject["calibration_event_ids"]) & set(first_subject["test_event_ids"])


def _write_fold_report(path: Path) -> None:
    def rows(subject: str, bias: float) -> tuple[list[str], list[float], list[float]]:
        sample_ids = []
        preds = []
        targets = []
        for event_index in range(4):
            session = "ses-01" if event_index < 2 else "ses-02"
            target = float(event_index + 2)
            pred = target + bias
            for window_index in range(2):
                sample_ids.append(f"{subject}_{session}_row-{event_index:04d}_win-{window_index:04d}")
                preds.append(pred)
                targets.append(target)
        return sample_ids, preds, targets

    s1_ids, s1_pred, s1_target = rows("sub-01", bias=-1.0)
    s2_ids, s2_pred, s2_target = rows("sub-02", bias=1.0)
    payload = {
        "experiments": {
            "B1": {
                "folds": [
                    {
                        "fold": "loso_sub-01",
                        "test_sample_ids": s1_ids,
                        "test_predictions": s1_pred,
                        "test_targets": s1_target,
                    },
                    {
                        "fold": "loso_sub-02",
                        "test_sample_ids": s2_ids,
                        "test_predictions": s2_pred,
                        "test_targets": s2_target,
                    },
                ]
            }
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_affine_fold_report(path: Path) -> None:
    def rows(subject: str) -> tuple[list[str], list[float], list[float]]:
        sample_ids = []
        preds = []
        targets = []
        for event_index in range(6):
            target = float(event_index + 1)
            pred = 7.0 - target
            session = "ses-01" if event_index < 3 else "ses-02"
            sample_ids.append(f"{subject}_{session}_row-{event_index:04d}_win-0000")
            preds.append(pred)
            targets.append(target)
        return sample_ids, preds, targets

    s1_ids, s1_pred, s1_target = rows("sub-01")
    s2_ids, s2_pred, s2_target = rows("sub-02")
    payload = {
        "experiments": {
            "B1": {
                "folds": [
                    {
                        "fold": "loso_sub-01",
                        "test_sample_ids": s1_ids,
                        "test_predictions": s1_pred,
                        "test_targets": s1_target,
                    },
                    {
                        "fold": "loso_sub-02",
                        "test_sample_ids": s2_ids,
                        "test_predictions": s2_pred,
                        "test_targets": s2_target,
                    },
                ]
            }
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
