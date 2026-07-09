from datetime import datetime

import numpy as np

from daily_multimodal.training.within_subject_splits import (
    WindowMetadata,
    build_global_paired_cohort,
    build_overlap_components,
    load_window_metadata,
    write_cohort_manifest,
)


def test_global_cohort_is_ordered_intersection_across_all_experiments():
    sample_ids = {
        f"exp-{index:02d}": np.asarray(
            ["s3", "s1", "s2"] if index == 0 else ["s1", "s2", "extra"]
        )
        for index in range(12)
    }
    cohort = build_global_paired_cohort(
        sample_ids,
        reference_order=sample_ids["exp-00"],
    )
    assert cohort.tolist() == ["s1", "s2"]


def test_overlapping_events_form_one_connected_split_unit():
    rows = [
        WindowMetadata(
            "w1",
            "e1",
            "sub-01",
            "ses-01",
            datetime.fromisoformat("2026-01-01 10:00:00"),
            datetime.fromisoformat("2026-01-01 10:02:00"),
        ),
        WindowMetadata(
            "w2",
            "e2",
            "sub-01",
            "ses-01",
            datetime.fromisoformat("2026-01-01 10:01:00"),
            datetime.fromisoformat("2026-01-01 10:03:00"),
        ),
        WindowMetadata(
            "w3",
            "e3",
            "sub-01",
            "ses-01",
            datetime.fromisoformat("2026-01-01 11:00:00"),
            datetime.fromisoformat("2026-01-01 11:02:00"),
        ),
    ]
    component_by_event, overlaps = build_overlap_components(rows)
    assert component_by_event[("sub-01", "ses-01", "e1")] == component_by_event[
        ("sub-01", "ses-01", "e2")
    ]
    assert component_by_event[("sub-01", "ses-01", "e1")] != component_by_event[
        ("sub-01", "ses-01", "e3")
    ]
    assert overlaps == [
        {
            "subject_id": "sub-01",
            "session_id": "ses-01",
            "event_a": "e1",
            "event_b": "e2",
            "overlap_seconds": 60.0,
        }
    ]


def test_overlap_components_use_collision_safe_event_keys():
    rows = [
        WindowMetadata(
            "w1",
            "local-1",
            "sub-01",
            "ses-01",
            datetime.fromisoformat("2026-01-01 10:00:00"),
            datetime.fromisoformat("2026-01-01 10:02:00"),
        ),
        WindowMetadata(
            "w2",
            "local-1",
            "sub-02",
            "ses-01",
            datetime.fromisoformat("2026-01-01 10:01:00"),
            datetime.fromisoformat("2026-01-01 10:03:00"),
        ),
        WindowMetadata(
            "w3",
            "local-1",
            "sub-01",
            "ses-02",
            datetime.fromisoformat("2026-01-01 10:01:00"),
            datetime.fromisoformat("2026-01-01 10:03:00"),
        ),
    ]
    component_by_event, overlaps = build_overlap_components(rows)
    assert len(component_by_event) == 3
    assert len(set(component_by_event.values())) == 3
    assert overlaps == []


def test_load_window_metadata_returns_required_samples_in_requested_order(tmp_path):
    path = tmp_path / "window_index.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"sample_id":"s2","event_id":"e2","subject_id":"sub-01","session_id":"ses-02","window_start_time":"2026-01-01 11:00:00","window_end_time":"2026-01-01 11:00:10"}',
                '{"sample_id":"s1","event_id":"e1","subject_id":"sub-01","session_id":"ses-01","window_start_time":"2026-01-01 10:00:00","window_end_time":"2026-01-01 10:00:10"}',
                '{"sample_id":"unused","event_id":"e3","subject_id":"sub-02","session_id":"ses-01","window_start_time":"2026-01-01 12:00:00","window_end_time":"2026-01-01 12:00:10"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_window_metadata(path, required_sample_ids=np.asarray(["s1", "s2"]))
    assert [row.sample_id for row in rows] == ["s1", "s2"]
    assert [row.session_id for row in rows] == ["ses-01", "ses-02"]


def test_write_cohort_manifest_records_ordered_hash_and_counts():
    manifest = write_cohort_manifest(
        cohort=np.asarray(["s1", "s2"]),
        native_counts={"exp-00": 3, "exp-01": 2},
        source_hashes={"fusion_config": "abc", "window_index": "def"},
    )
    assert manifest["schema_version"] == 1
    assert manifest["cohort_count"] == 2
    assert manifest["sample_ids"] == ["s1", "s2"]
    assert manifest["sample_id_sha256"]
    assert manifest["native_counts"] == {"exp-00": 3, "exp-01": 2}
    assert manifest["source_hashes"] == {"fusion_config": "abc", "window_index": "def"}
