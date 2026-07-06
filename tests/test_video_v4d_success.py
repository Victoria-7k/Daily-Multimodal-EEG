from __future__ import annotations

import json

from daily_multimodal.training.video_v4d_success import evaluate_v4d_success, write_v4d_success_report


def _probe(subject_acc: float, session_acc: float) -> dict:
    return {
        "probes": {
            "P1_subject_logreg": {"accuracy_mean": subject_acc},
            "P2_within_subject_session_logreg": {"accuracy_mean": session_acc},
        }
    }


def _variant(rmse: float, pearson: float) -> dict:
    return {"experiments": {"V4d": {"rmse_mean": rmse, "pearson_r_mean": pearson}}}


def test_evaluate_v4d_success_passes_when_domain_probe_drops_without_fatigue_regression():
    result = evaluate_v4d_success(
        baseline_probe=_probe(0.98, 0.96),
        candidate_probe=_probe(0.72, 0.70),
        baseline_variants={
            "LOSO": _variant(1.00, 0.10),
            "S4": _variant(0.95, 0.20),
            "S2": _variant(0.97, 0.22),
        },
        candidate_variants={
            "LOSO": _variant(0.99, 0.11),
            "S4": _variant(0.95, 0.20),
            "S2": _variant(0.96, 0.25),
        },
        variant_name="V4d",
    )

    assert result["passed"] is True
    assert result["checks"]["subject_probe_drop"]["passed"] is True
    assert result["checks"]["session_probe_drop"]["passed"] is True
    assert result["checks"]["fatigue_LOSO_no_regression"]["passed"] is True


def test_evaluate_v4d_success_fails_when_probe_drops_but_fatigue_regresses():
    result = evaluate_v4d_success(
        baseline_probe=_probe(0.98, 0.96),
        candidate_probe=_probe(0.72, 0.70),
        baseline_variants={"LOSO": _variant(1.00, 0.10), "S4": _variant(0.95, 0.20), "S2": _variant(0.97, 0.22)},
        candidate_variants={"LOSO": _variant(1.08, 0.11), "S4": _variant(0.95, 0.20), "S2": _variant(0.96, 0.25)},
        variant_name="V4d",
        rmse_tolerance=0.0,
    )

    assert result["passed"] is False
    assert result["checks"]["subject_probe_drop"]["passed"] is True
    assert result["checks"]["fatigue_LOSO_no_regression"]["passed"] is False
    assert result["checks"]["fatigue_LOSO_no_regression"]["rmse_delta"] > 0


def test_evaluate_v4d_success_fails_when_domain_probe_does_not_strictly_drop():
    result = evaluate_v4d_success(
        baseline_probe=_probe(0.98, 0.96),
        candidate_probe=_probe(0.98, 0.96),
        baseline_variants={"LOSO": _variant(1.00, 0.10), "S4": _variant(0.95, 0.20), "S2": _variant(0.97, 0.22)},
        candidate_variants={"LOSO": _variant(1.00, 0.10), "S4": _variant(0.95, 0.20), "S2": _variant(0.97, 0.22)},
        variant_name="V4d",
    )

    assert result["passed"] is False
    assert result["checks"]["subject_probe_drop"]["passed"] is False
    assert result["checks"]["session_probe_drop"]["passed"] is False


def test_write_v4d_success_report_writes_json_and_markdown(tmp_path):
    result = evaluate_v4d_success(
        baseline_probe=_probe(0.98, 0.96),
        candidate_probe=_probe(0.72, 0.70),
        baseline_variants={"LOSO": _variant(1.00, 0.10), "S4": _variant(0.95, 0.20), "S2": _variant(0.97, 0.22)},
        candidate_variants={"LOSO": _variant(0.99, 0.11), "S4": _variant(0.95, 0.20), "S2": _variant(0.96, 0.25)},
        variant_name="V4d",
    )
    out_json = tmp_path / "success.json"
    out_md = tmp_path / "success.md"

    write_v4d_success_report(result, out_json=out_json, out_table=out_md)

    assert json.loads(out_json.read_text(encoding="utf-8"))["passed"] is True
    text = out_md.read_text(encoding="utf-8")
    assert "| check | passed |" in text
    assert "subject_probe_drop" in text
