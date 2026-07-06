from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from daily_multimodal.embeddings.video_regions import build_video_region_cache


def test_build_video_region_cache_writes_expected_paths_and_upper_body_fallback(tmp_path):
    window_index = tmp_path / "window_index.jsonl"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_a = source_dir / "a.mp4"
    source_b = source_dir / "b.mp4"
    source_a.write_bytes(b"source-a")
    source_b.write_bytes(b"source-b")
    _write_jsonl(
        window_index,
        [
            _window("sample-a", source_a, start=1.0, end=11.0),
            _window("sample-b", source_b, start=2.0, end=12.0),
        ],
    )
    roi_cache_root = tmp_path / "roi-cache"
    for sample_id in ["sample-a", "sample-b"]:
        roi_dir = roi_cache_root / "openface" / sample_id / "openface_temporal_v1"
        roi_dir.mkdir(parents=True)
        (roi_dir / "window.mp4").write_bytes(f"roi-{sample_id}".encode("utf-8"))

    calls = []

    def fake_writer(*, source_video, output_video, clip_start_seconds, clip_end_seconds, crop_bbox):
        calls.append(
            {
                "source_video": Path(source_video),
                "output_video": Path(output_video),
                "clip_start_seconds": clip_start_seconds,
                "clip_end_seconds": clip_end_seconds,
                "crop_bbox": crop_bbox,
            }
        )
        Path(output_video).parent.mkdir(parents=True, exist_ok=True)
        Path(output_video).write_bytes(Path(source_video).read_bytes())

    def fake_upper_body_localizer(window):
        if window["sample_id"] == "sample-a":
            return [10, 20, 210, 320]
        return None

    summary = build_video_region_cache(
        window_index_path=window_index,
        out_root=tmp_path / "video_regions",
        roi_cache_root=roi_cache_root,
        roi_encoder_profile="openface_temporal_v1",
        upper_body_localizer=fake_upper_body_localizer,
        clip_writer=fake_writer,
    )

    assert summary == {"selected_count": 2, "written_count": 6, "skipped_existing_count": 0, "failure_count": 0}
    for region in ["2x_face_roi", "upper_body", "full_frame"]:
        for sample_id in ["sample-a", "sample-b"]:
            assert (tmp_path / "video_regions" / region / sample_id / "window.mp4").is_file()
            assert (tmp_path / "video_regions" / region / sample_id / "region.json").is_file()

    upper_a = _read_json(tmp_path / "video_regions" / "upper_body" / "sample-a" / "region.json")
    upper_b = _read_json(tmp_path / "video_regions" / "upper_body" / "sample-b" / "region.json")
    assert upper_a["crop_bbox"] == [10, 20, 210, 320]
    assert upper_a["upper_body_fallback_full_frame"] is False
    assert upper_a["effective_region"] == "upper_body"
    assert upper_b["crop_bbox"] is None
    assert upper_b["upper_body_fallback_full_frame"] is True
    assert upper_b["effective_region"] == "full_frame"

    manifest_rows = _read_jsonl(tmp_path / "video_regions" / "video_regions_manifest.jsonl")
    assert len(manifest_rows) == 6
    assert {
        (row["region"], row["sample_id"], row["output_video_path"].endswith("window.mp4"))
        for row in manifest_rows
    } == {
        ("2x_face_roi", "sample-a", True),
        ("2x_face_roi", "sample-b", True),
        ("upper_body", "sample-a", True),
        ("upper_body", "sample-b", True),
        ("full_frame", "sample-a", True),
        ("full_frame", "sample-b", True),
    }
    face_roi_calls = [call for call in calls if "2x_face_roi" in str(call["output_video"])]
    assert [call["source_video"].name for call in face_roi_calls] == ["window.mp4", "window.mp4"]
    upper_b_call = next(call for call in calls if call["output_video"].parent.name == "sample-b" and "upper_body" in str(call["output_video"]))
    assert upper_b_call["source_video"] == source_b
    assert upper_b_call["crop_bbox"] is None


def test_build_video_region_cache_records_missing_source_failure(tmp_path):
    window_index = tmp_path / "window_index.jsonl"
    _write_jsonl(window_index, [_window("missing", tmp_path / "missing.mp4")])

    summary = build_video_region_cache(
        window_index_path=window_index,
        out_root=tmp_path / "video_regions",
    )

    assert summary == {"selected_count": 1, "written_count": 0, "skipped_existing_count": 0, "failure_count": 3}
    failures = _read_json(tmp_path / "video_regions" / "video_regions_failures.json")
    assert {failure["region"] for failure in failures} == {"2x_face_roi", "upper_body", "full_frame"}
    assert {failure["error_type"] for failure in failures} == {"source_video_missing"}


def test_build_video_region_cache_can_write_chunk_manifests(tmp_path):
    window_index = tmp_path / "window_index.jsonl"
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    _write_jsonl(
        window_index,
        [
            _window("sample-a", source),
            _window("sample-b", source),
            _window("sample-c", source),
        ],
    )
    manifest_out = tmp_path / "chunks" / "manifest_0001.jsonl"
    failures_out = tmp_path / "chunks" / "failures_0001.json"

    def fake_writer(*, source_video, output_video, clip_start_seconds, clip_end_seconds, crop_bbox):
        del source_video, clip_start_seconds, clip_end_seconds, crop_bbox
        Path(output_video).parent.mkdir(parents=True, exist_ok=True)
        Path(output_video).write_bytes(b"clip")

    summary = build_video_region_cache(
        window_index_path=window_index,
        out_root=tmp_path / "video_regions",
        regions=["full_frame"],
        start_index=1,
        max_windows=1,
        manifest_out=manifest_out,
        failures_out=failures_out,
        clip_writer=fake_writer,
    )

    assert summary == {"selected_count": 1, "written_count": 1, "skipped_existing_count": 0, "failure_count": 0}
    assert _read_jsonl(manifest_out)[0]["sample_id"] == "sample-b"
    assert _read_json(failures_out) == []
    assert not (tmp_path / "video_regions" / "video_regions_manifest.jsonl").exists()
    assert not (tmp_path / "video_regions" / "video_regions_failures.json").exists()


def test_upper_body_bbox_expands_face_presence_xywh_bbox(tmp_path):
    window_index = tmp_path / "window_index.jsonl"
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    row = _window("sample-a", source)
    row["face_presence"] = {"main_face_bbox": [100, 50, 40, 60]}
    _write_jsonl(window_index, [row])
    calls = []

    def fake_writer(*, source_video, output_video, clip_start_seconds, clip_end_seconds, crop_bbox):
        del source_video, clip_start_seconds, clip_end_seconds
        calls.append(crop_bbox)
        Path(output_video).parent.mkdir(parents=True, exist_ok=True)
        Path(output_video).write_bytes(b"region")

    summary = build_video_region_cache(
        window_index_path=window_index,
        out_root=tmp_path / "video_regions",
        regions=["upper_body"],
        clip_writer=fake_writer,
    )

    assert summary == {"selected_count": 1, "written_count": 1, "skipped_existing_count": 0, "failure_count": 0}
    assert calls == [[60, 20, 180, 290]]
    sidecar = _read_json(tmp_path / "video_regions" / "upper_body" / "sample-a" / "region.json")
    assert sidecar["crop_bbox"] == [60, 20, 180, 290]
    assert sidecar["upper_body_fallback_full_frame"] is False


def test_build_video_region_cache_default_writer_trims_source_window(tmp_path, monkeypatch):
    window_index = tmp_path / "window_index.jsonl"
    source = tmp_path / "long_source.mp4"
    source.write_bytes(b"long-video")
    _write_jsonl(window_index, [_window("sample-a", source, start=12.5, end=22.5)])
    written = []
    seek_positions = []
    read_positions = []

    class FakeCapture:
        def __init__(self, path):
            self.path = path
            self.position = 0

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == fake_cv2.CAP_PROP_FPS:
                return 5.0
            if prop == fake_cv2.CAP_PROP_FRAME_COUNT:
                return 200
            return 0

        def set(self, prop, value):
            if prop == fake_cv2.CAP_PROP_POS_FRAMES:
                self.position = int(value)
                seek_positions.append(self.position)

        def read(self):
            frame = np.full((120, 240, 3), self.position % 255, dtype=np.uint8)
            read_positions.append(self.position)
            self.position += 1
            return True, frame

        def release(self):
            return None

    class FakeWriter:
        def __init__(self, path, fourcc, fps, size):
            self.path = Path(path)
            self.fps = fps
            self.size = size
            self.frames = []

        def isOpened(self):
            return True

        def write(self, frame):
            self.frames.append(frame)
            written.append({"path": self.path, "fps": self.fps, "size": self.size, "shape": frame.shape})

        def release(self):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_bytes(f"frames={len(self.frames)}".encode("utf-8"))

    class FakeCV2:
        CAP_PROP_FPS = 1
        CAP_PROP_FRAME_COUNT = 2
        CAP_PROP_POS_FRAMES = 3
        INTER_AREA = 4

        @staticmethod
        def VideoCapture(path):
            return FakeCapture(path)

        @staticmethod
        def VideoWriter(path, fourcc, fps, size):
            return FakeWriter(path, fourcc, fps, size)

        @staticmethod
        def VideoWriter_fourcc(*args):
            return 1234

        @staticmethod
        def resize(frame, size, interpolation=None):
            width, height = size
            return np.zeros((height, width, 3), dtype=frame.dtype)

    fake_cv2 = FakeCV2()
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    summary = build_video_region_cache(
        window_index_path=window_index,
        out_root=tmp_path / "video_regions",
        regions=["full_frame"],
    )

    output = tmp_path / "video_regions" / "full_frame" / "sample-a" / "window.mp4"
    assert summary == {"selected_count": 1, "written_count": 1, "skipped_existing_count": 0, "failure_count": 0}
    assert output.read_bytes() == b"frames=16"
    assert len(written) == 16
    assert {item["path"] for item in written} == {output}
    assert {item["fps"] for item in written} == {2.0}
    assert {item["size"] for item in written} == {(640, 320)}
    assert {item["shape"] for item in written} == {(320, 640, 3)}
    assert seek_positions == [62]
    assert len(read_positions) == 50


def test_default_writer_reuses_sampled_frames_for_upper_body_and_full_frame(tmp_path, monkeypatch):
    window_index = tmp_path / "window_index.jsonl"
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    row = _window("sample-a", source, start=1.0, end=3.0)
    row["face_presence"] = {"main_face_bbox": [10, 10, 10, 10]}
    _write_jsonl(window_index, [row])
    capture_count = 0

    class FakeCapture:
        def __init__(self, path):
            nonlocal capture_count
            capture_count += 1
            self.position = 0

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == fake_cv2.CAP_PROP_FPS:
                return 5.0
            if prop == fake_cv2.CAP_PROP_FRAME_COUNT:
                return 100
            return 0

        def set(self, prop, value):
            if prop == fake_cv2.CAP_PROP_POS_FRAMES:
                self.position = int(value)

        def read(self):
            frame = np.full((120, 240, 3), self.position % 255, dtype=np.uint8)
            self.position += 1
            return True, frame

        def release(self):
            return None

    class FakeWriter:
        def __init__(self, path, fourcc, fps, size):
            self.path = Path(path)
            self.frames = []

        def isOpened(self):
            return True

        def write(self, frame):
            self.frames.append(frame)

        def release(self):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_bytes(f"frames={len(self.frames)}".encode("utf-8"))

    class FakeCV2:
        CAP_PROP_FPS = 1
        CAP_PROP_FRAME_COUNT = 2
        CAP_PROP_POS_FRAMES = 3
        INTER_AREA = 4

        @staticmethod
        def VideoCapture(path):
            return FakeCapture(path)

        @staticmethod
        def VideoWriter(path, fourcc, fps, size):
            return FakeWriter(path, fourcc, fps, size)

        @staticmethod
        def VideoWriter_fourcc(*args):
            return 1234

        @staticmethod
        def resize(frame, size, interpolation=None):
            width, height = size
            return np.zeros((height, width, 3), dtype=frame.dtype)

    fake_cv2 = FakeCV2()
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    summary = build_video_region_cache(
        window_index_path=window_index,
        out_root=tmp_path / "video_regions",
        regions=["upper_body", "full_frame"],
    )

    assert summary == {"selected_count": 1, "written_count": 2, "skipped_existing_count": 0, "failure_count": 0}
    assert capture_count == 1
    assert (tmp_path / "video_regions" / "upper_body" / "sample-a" / "window.mp4").read_bytes() == b"frames=16"
    assert (tmp_path / "video_regions" / "full_frame" / "sample-a" / "window.mp4").read_bytes() == b"frames=16"


def test_default_writer_frame_cache_is_scoped_to_one_window(tmp_path, monkeypatch):
    window_index = tmp_path / "window_index.jsonl"
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    rows = [_window("sample-a", source, start=1.0, end=3.0), _window("sample-b", source, start=1.0, end=3.0)]
    for row in rows:
        row["face_presence"] = {"main_face_bbox": [10, 10, 10, 10]}
    _write_jsonl(window_index, rows)
    capture_count = 0

    class FakeCapture:
        def __init__(self, path):
            nonlocal capture_count
            del path
            capture_count += 1
            self.position = 0

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == fake_cv2.CAP_PROP_FPS:
                return 5.0
            if prop == fake_cv2.CAP_PROP_FRAME_COUNT:
                return 100
            return 0

        def set(self, prop, value):
            if prop == fake_cv2.CAP_PROP_POS_FRAMES:
                self.position = int(value)

        def read(self):
            frame = np.full((120, 240, 3), self.position % 255, dtype=np.uint8)
            self.position += 1
            return True, frame

        def release(self):
            return None

    class FakeWriter:
        def __init__(self, path, fourcc, fps, size):
            del fourcc, fps, size
            self.path = Path(path)
            self.frames = []

        def isOpened(self):
            return True

        def write(self, frame):
            self.frames.append(frame)

        def release(self):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_bytes(f"frames={len(self.frames)}".encode("utf-8"))

    class FakeCV2:
        CAP_PROP_FPS = 1
        CAP_PROP_FRAME_COUNT = 2
        CAP_PROP_POS_FRAMES = 3
        INTER_AREA = 4

        @staticmethod
        def VideoCapture(path):
            return FakeCapture(path)

        @staticmethod
        def VideoWriter(path, fourcc, fps, size):
            return FakeWriter(path, fourcc, fps, size)

        @staticmethod
        def VideoWriter_fourcc(*args):
            return 1234

        @staticmethod
        def resize(frame, size, interpolation=None):
            width, height = size
            return np.zeros((height, width, 3), dtype=frame.dtype)

    fake_cv2 = FakeCV2()
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    summary = build_video_region_cache(
        window_index_path=window_index,
        out_root=tmp_path / "video_regions",
        regions=["upper_body", "full_frame"],
    )

    assert summary == {"selected_count": 2, "written_count": 4, "skipped_existing_count": 0, "failure_count": 0}
    assert capture_count == 2


def test_default_writer_groups_same_event_source_windows(tmp_path, monkeypatch):
    window_index = tmp_path / "window_index.jsonl"
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    rows = [_window("sample-a", source, start=1.0, end=3.0), _window("sample-b", source, start=3.0, end=5.0)]
    for row in rows:
        row["event_id"] = "same-event"
        row["face_presence"] = {"main_face_bbox": [10, 10, 10, 10]}
    _write_jsonl(window_index, rows)
    capture_count = 0

    class FakeCapture:
        def __init__(self, path):
            nonlocal capture_count
            del path
            capture_count += 1
            self.position = 0

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == fake_cv2.CAP_PROP_FPS:
                return 5.0
            if prop == fake_cv2.CAP_PROP_FRAME_COUNT:
                return 100
            return 0

        def set(self, prop, value):
            if prop == fake_cv2.CAP_PROP_POS_FRAMES:
                self.position = int(value)

        def read(self):
            frame = np.full((120, 240, 3), self.position % 255, dtype=np.uint8)
            self.position += 1
            return True, frame

        def release(self):
            return None

    class FakeWriter:
        def __init__(self, path, fourcc, fps, size):
            del fourcc, fps, size
            self.path = Path(path)
            self.frames = []

        def isOpened(self):
            return True

        def write(self, frame):
            self.frames.append(frame)

        def release(self):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_bytes(f"frames={len(self.frames)}".encode("utf-8"))

    class FakeCV2:
        CAP_PROP_FPS = 1
        CAP_PROP_FRAME_COUNT = 2
        CAP_PROP_POS_FRAMES = 3
        INTER_AREA = 4

        @staticmethod
        def VideoCapture(path):
            return FakeCapture(path)

        @staticmethod
        def VideoWriter(path, fourcc, fps, size):
            return FakeWriter(path, fourcc, fps, size)

        @staticmethod
        def VideoWriter_fourcc(*args):
            return 1234

        @staticmethod
        def resize(frame, size, interpolation=None):
            width, height = size
            return np.zeros((height, width, 3), dtype=frame.dtype)

    fake_cv2 = FakeCV2()
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    summary = build_video_region_cache(
        window_index_path=window_index,
        out_root=tmp_path / "video_regions",
        regions=["upper_body", "full_frame"],
    )

    assert summary == {"selected_count": 2, "written_count": 4, "skipped_existing_count": 0, "failure_count": 0}
    assert capture_count == 1
    assert (tmp_path / "video_regions" / "upper_body" / "sample-a" / "window.mp4").read_bytes() == b"frames=16"
    assert (tmp_path / "video_regions" / "full_frame" / "sample-a" / "window.mp4").read_bytes() == b"frames=16"
    row = _read_json(tmp_path / "video_regions" / "upper_body" / "sample-a" / "region.json")
    assert row["region_source"] == "source_video_event_group"


def test_build_video_region_cache_skips_existing_region_clip_and_sidecar(tmp_path):
    window_index = tmp_path / "window_index.jsonl"
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    _write_jsonl(window_index, [_window("sample-a", source)])
    region_dir = tmp_path / "video_regions" / "full_frame" / "sample-a"
    region_dir.mkdir(parents=True)
    (region_dir / "window.mp4").write_bytes(b"existing")
    existing_row = {
        "sample_id": "sample-a",
        "event_id": "sample-a-event",
        "subject_id": "sub-01",
        "region": "full_frame",
        "effective_region": "full_frame",
        "source_video_path": str(source),
        "output_video_path": str(region_dir / "window.mp4"),
        "clip_start_seconds": 0,
        "clip_end_seconds": 10,
        "crop_bbox": None,
        "region_source": "source_video",
        "upper_body_fallback_full_frame": False,
    }
    (region_dir / "region.json").write_text(json.dumps(existing_row), encoding="utf-8")
    calls = []

    def fake_writer(*, source_video, output_video, clip_start_seconds, clip_end_seconds, crop_bbox):
        del source_video, output_video, clip_start_seconds, clip_end_seconds, crop_bbox
        calls.append("called")

    summary = build_video_region_cache(
        window_index_path=window_index,
        out_root=tmp_path / "video_regions",
        regions=["full_frame"],
        clip_writer=fake_writer,
    )

    assert summary == {"selected_count": 1, "written_count": 1, "skipped_existing_count": 1, "failure_count": 0}
    assert calls == []
    assert (region_dir / "window.mp4").read_bytes() == b"existing"
    manifest_row = _read_jsonl(tmp_path / "video_regions" / "video_regions_manifest.jsonl")[0]
    assert manifest_row["cache_status"] == "existing"


def _window(sample_id: str, source_path: Path, *, start: float = 0.0, end: float = 10.0) -> dict:
    return {
        "sample_id": sample_id,
        "event_id": f"{sample_id}-event",
        "subject_id": "sub-01",
        "video_candidates": [
            {
                "mp4_path": str(source_path),
                "clip_start_seconds": start,
                "clip_end_seconds": end,
            }
        ],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
