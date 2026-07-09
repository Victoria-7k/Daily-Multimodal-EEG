from datetime import datetime
import json
import subprocess
import sys

import numpy as np

from daily_multimodal.training.within_subject_splits import (
    WindowMetadata,
    build_global_paired_cohort,
    build_overlap_components,
    build_split_manifest,
    load_window_metadata,
    validate_split_manifest,
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


def test_split_manifest_is_fixed_across_model_seeds_and_blocks_overlap():
    cohort, metadata = _split_fixture()
    first = build_split_manifest(cohort, metadata, split_seed=17)
    second = build_split_manifest(cohort, metadata, split_seed=17)
    assert first == second
    assert "model_seed" not in first
    for protocol in first["protocols"].values():
        for subject in protocol["subjects"]:
            for fold in subject.get("folds", []):
                train = set(map(tuple, fold["train_event_keys"]))
                val = set(map(tuple, fold["val_event_keys"]))
                test = set(map(tuple, fold["test_event_keys"]))
                assert not train & val
                assert not train & test
                assert not val & test
                assert fold["cross_partition_time_overlap_count"] == 0


def test_session_protocol_holds_out_each_session_once():
    cohort, metadata = _split_fixture()
    manifest = build_split_manifest(cohort, metadata, split_seed=17)
    row = manifest["protocols"]["session_held_out"]["subjects"][0]
    held_out = [fold["test_session_ids"][0] for fold in row["folds"]]
    assert sorted(held_out) == sorted(row["session_ids"])


def test_validate_split_manifest_rejects_hash_mismatch():
    cohort, metadata = _split_fixture()
    manifest = build_split_manifest(cohort, metadata, split_seed=17)
    manifest["cohort_sha256"] = "cohort-ok"
    manifest["window_index_sha256"] = "window-ok"
    validate_split_manifest(manifest, cohort_hash="cohort-ok", window_index_hash="window-ok")
    try:
        validate_split_manifest(manifest, cohort_hash="changed", window_index_hash="window-ok")
    except ValueError as exc:
        assert "cohort hash mismatch" in str(exc)
    else:
        raise AssertionError("validate_split_manifest accepted a stale cohort hash")


def test_prepare_within_subject_fusion_splits_dry_run_reports_matrix_without_writing(tmp_path):
    fusion_config, window_index = _write_tiny_fusion_inputs(tmp_path)
    cohort_manifest = tmp_path / "cohort.json"
    split_manifest = tmp_path / "splits.json"
    config = tmp_path / "within_subject.json"
    config.write_text(
        json.dumps(
            {
                "fusion_config": str(fusion_config),
                "window_index": str(window_index),
                "cohort_manifest": str(cohort_manifest),
                "split_manifest": str(split_manifest),
                "split_seed": 17,
                "model_seed": 1701,
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/44_prepare_within_subject_fusion_splits.py",
            "--config",
            str(config),
            "--dry-run",
        ],
        cwd=__import__("pathlib").Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "native_experiment_count=12" in completed.stdout
    assert "paired_cohort_count=6" in completed.stdout
    assert "protocols=event_grouped_5fold,session_held_out" in completed.stdout
    assert not cohort_manifest.exists()
    assert not split_manifest.exists()


def _split_fixture():
    rows = []
    cohort = []
    base = datetime.fromisoformat("2026-01-01 10:00:00")
    for index in range(6):
        session = f"ses-{(index % 3) + 1:02d}"
        sample_id = f"s{index}"
        cohort.append(sample_id)
        rows.append(
            WindowMetadata(
                sample_id,
                f"e{index}",
                "sub-01",
                session,
                base.replace(hour=10 + index),
                base.replace(hour=10 + index, minute=10),
            )
        )
    return np.asarray(cohort), rows


def _write_tiny_fusion_inputs(root):
    sample_ids = [f"s{index}" for index in range(6)]
    labels = np.asarray([json.dumps({"fatigue": float(index)}) for index in range(6)], dtype=object)
    mask = np.ones((6, 4), dtype=np.int8)
    embedding = np.zeros((6, 256), dtype=np.float32)
    embedding[:, 0] = np.arange(6, dtype=np.float32)
    branches = {}
    for modality, key in {
        "eeg": "eeg_emb",
        "wear_physio": "wear_emb",
        "wear_deep": "wear_emb",
        "video_v4a": "face_emb",
        "video_b1": "face_emb",
        "audio": "audio_emb",
    }.items():
        path = root / f"{modality}.npz"
        np.savez_compressed(
            path,
            sample_id=np.asarray(sample_ids, dtype=object),
            event_id=np.asarray([f"e{index}" for index in range(6)], dtype=object),
            subject_id=np.asarray(["sub-01"] * 6, dtype=object),
            labels=labels,
            modality_mask=mask,
            **{key: embedding},
        )
        branches[modality] = path
    fusion_config = root / "fusion_matrix.json"
    fusion_config.write_text(
        json.dumps(
            {
                "target_label": "fatigue",
                "branches": {
                    "eeg": {"path": str(branches["eeg"]), "modality": "eeg", "profile": "eeg"},
                    "wear": {
                        "WphysioPre": {
                            "path": str(branches["wear_physio"]),
                            "modality": "wear",
                            "profile": "wear_physio_features_preprocessed_v1",
                        },
                        "WdeepPre": {
                            "path": str(branches["wear_deep"]),
                            "modality": "wear",
                            "profile": "wear_deep_sequence_preprocessed_v1",
                        },
                    },
                    "video": {
                        "V4aUpper": {
                            "path": str(branches["video_v4a"]),
                            "modality": "video",
                            "profile": "V4a_upper",
                        },
                        "B1": {"path": str(branches["video_b1"]), "modality": "video", "profile": "B1"},
                    },
                    "audio": {"path": str(branches["audio"]), "modality": "audio", "profile": "audio"},
                },
            }
        ),
        encoding="utf-8",
    )
    window_index = root / "window_index.jsonl"
    rows = []
    base = datetime.fromisoformat("2026-01-01 10:00:00")
    for index, sample_id in enumerate(sample_ids):
        rows.append(
            {
                "sample_id": sample_id,
                "event_id": f"e{index}",
                "subject_id": "sub-01",
                "session_id": f"ses-{(index % 3) + 1:02d}",
                "window_start_time": base.replace(hour=10 + index).isoformat(sep=" "),
                "window_end_time": base.replace(hour=10 + index, minute=10).isoformat(sep=" "),
            }
        )
    window_index.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return fusion_config, window_index
