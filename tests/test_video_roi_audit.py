from __future__ import annotations

import csv
import json
from pathlib import Path

from daily_multimodal.embeddings.video_roi_audit import run_video_roi_audit


def test_video_roi_audit_summarizes_subject_session_geometry(tmp_path):
    window_index = tmp_path / "window_index.jsonl"
    region_root = tmp_path / "video_regions"
    out_csv = tmp_path / "roi_session_summary.csv"
    out_json = tmp_path / "geometry_session_probe.json"
    out_md = tmp_path / "roi_audit_summary.md"
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    rows = [
        _window("sub-01_ses-A_win-0000", "sub-01_ses-A_event", "sub-01", source, [40, 40, 40, 40]),
        _window("sub-01_ses-A_win-0001", "sub-01_ses-A_event", "sub-01", source, [40, 40, 40, 40]),
        _window("sub-01_ses-B_win-0000", "sub-01_ses-B_event", "sub-01", source, [40, 40, 40, 40]),
        _window("sub-01_ses-B_win-0001", "sub-01_ses-B_event", "sub-01", source, [40, 40, 40, 40]),
    ]
    _write_jsonl(window_index, rows)
    for sample_id in ["sub-01_ses-A_win-0000", "sub-01_ses-A_win-0001"]:
        _write_sidecar(region_root, sample_id, source, [20, 20, 220, 220], fallback=False)
    for sample_id in ["sub-01_ses-B_win-0000", "sub-01_ses-B_win-0001"]:
        _write_sidecar(region_root, sample_id, source, [20, 20, 420, 420], fallback=False)

    result = run_video_roi_audit(
        window_index_path=window_index,
        region_cache_root=region_root,
        out_csv=out_csv,
        out_probe_json=out_json,
        out_summary_md=out_md,
        frame_size_reader=lambda path: (800, 600),
        brightness_reader=lambda path: 120.0 if "ses-A" in str(path) else 80.0,
        min_probe_windows_per_session=1,
    )

    summary_rows = list(csv.DictReader(out_csv.open("r", encoding="utf-8")))
    assert result["session_count"] == 2
    assert [row["session_id"] for row in summary_rows] == ["sub-01_ses-A", "sub-01_ses-B"]
    assert float(summary_rows[0]["roi_area_ratio_mean"]) < float(summary_rows[1]["roi_area_ratio_mean"])
    assert float(summary_rows[0]["face_roi_ratio_mean"]) > float(summary_rows[1]["face_roi_ratio_mean"])
    assert float(summary_rows[0]["fallback_ratio"]) == 0.0
    assert float(summary_rows[0]["brightness_mean"]) == 120.0
    assert "ROI scale changed" in out_md.read_text(encoding="utf-8")


def test_video_roi_audit_runs_within_subject_geometry_session_probe(tmp_path):
    window_index = tmp_path / "window_index.jsonl"
    region_root = tmp_path / "video_regions"
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    windows = []
    for session, crop in [("ses-A", [10, 10, 210, 210]), ("ses-B", [400, 300, 760, 560])]:
        for index in range(4):
            sample_id = f"sub-01_{session}_win-{index:04d}"
            windows.append(_window(sample_id, f"sub-01_{session}_event", "sub-01", source, [20, 20, 40, 40]))
            _write_sidecar(region_root, sample_id, source, crop, fallback=False)
    _write_jsonl(window_index, windows)

    result = run_video_roi_audit(
        window_index_path=window_index,
        region_cache_root=region_root,
        out_csv=tmp_path / "roi_session_summary.csv",
        out_probe_json=tmp_path / "geometry_session_probe.json",
        out_summary_md=tmp_path / "roi_audit_summary.md",
        frame_size_reader=lambda path: (800, 600),
        brightness_reader=lambda path: 100.0,
        min_probe_windows_per_session=2,
        seed=1,
    )

    probe = json.loads((tmp_path / "geometry_session_probe.json").read_text(encoding="utf-8"))
    assert result["probe"]["subject_count"] == 1
    assert probe["feature_names"] == [
        "roi_area_ratio",
        "roi_center_x",
        "roi_center_y",
        "face_roi_ratio",
        "fallback_ratio",
        "brightness_mean",
    ]
    assert probe["subjects"][0]["subject_id"] == "sub-01"
    assert probe["subjects"][0]["session_count"] == 2
    assert probe["subjects"][0]["accuracy_mean"] >= 0.75


def _window(sample_id: str, event_id: str, subject_id: str, source: Path, face_bbox: list[int]) -> dict:
    return {
        "sample_id": sample_id,
        "event_id": event_id,
        "subject_id": subject_id,
        "face_presence": {"main_face_bbox": face_bbox},
        "video_candidates": [{"mp4_path": str(source), "clip_start_seconds": 0.0, "clip_end_seconds": 10.0}],
    }


def _write_sidecar(region_root: Path, sample_id: str, source: Path, crop_bbox: list[int] | None, *, fallback: bool) -> None:
    out_dir = region_root / "upper_body" / sample_id
    out_dir.mkdir(parents=True)
    (out_dir / "window.mp4").write_bytes(b"clip")
    (out_dir / "region.json").write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "event_id": sample_id,
                "subject_id": sample_id.split("_")[0],
                "region": "upper_body",
                "effective_region": "full_frame" if fallback else "upper_body",
                "source_video_path": str(source),
                "output_video_path": str(out_dir / "window.mp4"),
                "crop_bbox": crop_bbox,
                "upper_body_fallback_full_frame": fallback,
            }
        ),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
