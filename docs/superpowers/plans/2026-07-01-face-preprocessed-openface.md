# Face Preprocessed OpenFace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `face_preprocessed_openface_stats_v1` branch that preserves the raw OpenFace branch, retries difficult windows with face pre-detection plus ROI clips, and raises usable face coverage without hiding low-quality windows.

**Architecture:** Keep `face_raw_openface_stats_v1` unchanged as the default branch. Add a separate preprocessing module that samples frames, evaluates face detections across rotations, selects a stable main-face track, writes a reproducible ROI clip, and lets the existing OpenFace CSV/statistics path consume that clip. Quality flags and reports must expose every preprocessing decision so downstream ablation can decide whether the preprocessed branch should become default.

**Tech Stack:** Python, NumPy, OpenCV/ffmpeg/ffprobe, existing OpenFace wrapper, existing `.npz`/JSON failure contracts, `unittest`/pytest-compatible tests.

---

## Acceptance Standards

### Non-Negotiable Correctness

- Raw branch remains untouched: running `face_raw_openface_stats_v1` must not write preprocessed cache files, change raw cache paths, or change raw quality semantics.
- Preprocessed branch writes under a distinct profile/cache path: `face_preprocessed_openface_stats_v1`.
- No generated face repair, no strong deblurring, and no expression-changing enhancement. Allowed enhancement is limited to ROI luminance CLAHE or bounded gamma.
- Every processed sample keeps the same `sample_id`, `event_id`, and `subject_id`.
- Every output embedding array has shape `(N, 256)` and NaN count `0`.
- A low-quality ROI result is retained as a row with `modality_mask[:, 2] = 0`; it is not silently dropped.
- `Starting tracking` with no CSV gets at most one preprocessed ROI retry and a bounded timeout. If retry fails, it remains a structured failure.

### Full-Run Quality Gate

Use the current repaired raw OpenFace baseline as the comparison anchor:

- Raw full baseline: `selected_windows=781`, `face_success_count=207`, `extraction_failed=73`, `masked_count=501`.
- Preprocessed full run must produce `selected_windows=781`.
- Preprocessed face usable count must be at least `249`, a 20% relative lift over `207`.
- Preprocessed `extraction_failed` must be at most `36`, a 50% reduction from `73`.
- `openface_abort_after_starting_tracking_no_csv` must be at most `10`, unless visual audit proves most remaining windows have no recoverable face.
- NaN count must be `0`.
- `main_face_ambiguity_ratio` must be reported; windows above the ambiguity threshold must be masked, not forced into a face track.

### Default-Branch Promotion Gate

Preprocessing can become the default face branch only if all downstream gates pass:

- `all_real_with_preprocessed_face` median test RMSE over 5 seeds is at least 2% lower than `all_real_with_raw_face`.
- MAE does not worsen by more than 1%.
- Pearson r does not drop by more than `0.02`.
- At least 4 of 5 seeds move in the favorable direction for the primary metric.
- Bootstrap 95% CI for delta RMSE does not show stable negative impact.

If quality improves but downstream gates fail, keep raw as default and preserve preprocessed artifacts for analysis.

---

## Files And Responsibilities

- Modify `src/daily_multimodal/embeddings/face_real.py`
  - Integrate the preprocessed branch without changing raw behavior.
  - Route `face_preprocessed_openface_stats_v1` to ROI clip preparation before OpenFace.
  - Add bounded preprocessed retry for `Starting tracking` no-CSV failures.

- Create `src/daily_multimodal/embeddings/face_preprocessing.py`
  - Sample frames at configurable FPS.
  - Try rotations `0`, `90`, `180`, `270`.
  - Run a detector backend through a small protocol.
  - Select a main-face track.
  - Write ROI clip and preprocessing metadata.

- Modify `scripts/13_extract_face_embeddings.py`
  - Add explicit preprocessing CLI controls while keeping profile-based auto-enable.

- Create `scripts/22_compare_face_preprocessing.py`
  - Compare raw and preprocessed face summaries/failure lists/NPZ masks.
  - Write a machine-readable acceptance report and Markdown table.

- Modify `tests/test_face_real_embedding.py`
  - Cover integration: raw branch unchanged, preprocessed branch uses ROI clip, no-CSV retry uses ROI fallback.

- Create `tests/test_face_preprocessing.py`
  - Cover sampling, rotation selection, track selection, ambiguity, crop bounds, metadata.

- Create `tests/test_face_preprocessing_comparison.py`
  - Cover acceptance report calculations.

- Update `repo-docs/walkthroughs/one-real-run.md`
  - Explain raw vs preprocessed branch behavior.

- Update `repo-docs/references/commands-and-artifacts.md`
  - Add commands and artifact names.

- Update `repo-docs/change-log.md`
  - Record implementation and verification.

---

### Task 1: Preprocessing Data Model And Track Selection

**Files:**
- Create: `src/daily_multimodal/embeddings/face_preprocessing.py`
- Create: `tests/test_face_preprocessing.py`

- [ ] **Step 1: Write failing tests for rotation-aware main-face selection**

Add this test file:

```python
import tempfile
import unittest
from pathlib import Path

from daily_multimodal.embeddings.face_preprocessing import (
    FaceDetection,
    choose_main_face_track,
    expand_crop_box,
)


class FacePreprocessingTests(unittest.TestCase):
    def test_choose_main_face_track_prefers_continuous_centered_high_confidence_face(self):
        detections_by_frame = [
            [
                FaceDetection(frame_index=0, rotation=90, box=(100, 80, 60, 60), confidence=0.92),
                FaceDetection(frame_index=0, rotation=90, box=(10, 10, 40, 40), confidence=0.95),
            ],
            [FaceDetection(frame_index=1, rotation=90, box=(104, 82, 62, 62), confidence=0.90)],
            [FaceDetection(frame_index=2, rotation=90, box=(108, 84, 61, 61), confidence=0.91)],
        ]

        track = choose_main_face_track(
            detections_by_frame,
            frame_width=320,
            frame_height=240,
            ambiguity_iou_threshold=0.20,
        )

        self.assertFalse(track.ambiguous)
        self.assertEqual(track.rotation, 90)
        self.assertEqual(track.detection_count, 3)
        self.assertGreater(track.mean_confidence, 0.90)
        self.assertEqual(track.crop_box, (100, 80, 69, 65))

    def test_choose_main_face_track_marks_similar_competing_tracks_ambiguous(self):
        detections_by_frame = [
            [
                FaceDetection(frame_index=0, rotation=0, box=(80, 70, 60, 60), confidence=0.90),
                FaceDetection(frame_index=0, rotation=0, box=(170, 70, 60, 60), confidence=0.89),
            ],
            [
                FaceDetection(frame_index=1, rotation=0, box=(82, 70, 60, 60), confidence=0.91),
                FaceDetection(frame_index=1, rotation=0, box=(172, 70, 60, 60), confidence=0.90),
            ],
        ]

        track = choose_main_face_track(detections_by_frame, frame_width=320, frame_height=240)

        self.assertTrue(track.ambiguous)
        self.assertGreaterEqual(track.main_face_ambiguity_ratio, 0.5)

    def test_expand_crop_box_clamps_to_frame_bounds(self):
        crop = expand_crop_box((5, 5, 40, 50), frame_width=100, frame_height=80, margin=1.8)

        self.assertEqual(crop, (0, 0, 76, 77))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_face_preprocessing -v
```

Expected: import fails because `daily_multimodal.embeddings.face_preprocessing` does not exist.

- [ ] **Step 3: Implement minimal data model and track selection**

Create `src/daily_multimodal/embeddings/face_preprocessing.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class FaceDetection:
    frame_index: int
    rotation: int
    box: tuple[int, int, int, int]
    confidence: float


@dataclass(frozen=True)
class FaceTrack:
    rotation: int
    crop_box: tuple[int, int, int, int]
    detection_count: int
    sampled_frame_count: int
    mean_confidence: float
    main_face_ambiguity_ratio: float
    ambiguous: bool


def expand_crop_box(
    box: tuple[int, int, int, int],
    *,
    frame_width: int,
    frame_height: int,
    margin: float,
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    cx = x + w / 2.0
    cy = y + h / 2.0
    nw = w * float(margin)
    nh = h * float(margin)
    left = max(0, int(round(cx - nw / 2.0)))
    top = max(0, int(round(cy - nh / 2.0)))
    right = min(frame_width, int(round(cx + nw / 2.0)))
    bottom = min(frame_height, int(round(cy + nh / 2.0)))
    return left, top, max(1, right - left), max(1, bottom - top)


def choose_main_face_track(
    detections_by_frame: Sequence[Sequence[FaceDetection]],
    *,
    frame_width: int,
    frame_height: int,
    ambiguity_iou_threshold: float = 0.20,
) -> FaceTrack:
    flat = [det for frame in detections_by_frame for det in frame]
    sampled_count = len(detections_by_frame)
    if not flat:
        return FaceTrack(0, (0, 0, frame_width, frame_height), 0, sampled_count, 0.0, 0.0, False)

    by_rotation: dict[int, list[FaceDetection]] = {}
    for det in flat:
        by_rotation.setdefault(det.rotation, []).append(det)

    candidates = sorted(
        by_rotation.items(),
        key=lambda item: (
            len({det.frame_index for det in item[1]}),
            sum(det.confidence for det in item[1]) / max(1, len(item[1])),
            _mean_area(item[1]),
            -_mean_center_distance(item[1], frame_width, frame_height),
        ),
        reverse=True,
    )
    rotation, detections = candidates[0]
    frame_hits = len({det.frame_index for det in detections})
    ambiguity_ratio = _ambiguity_ratio(detections_by_frame)
    ambiguous = len(candidates) > 1 and ambiguity_ratio >= float(ambiguity_iou_threshold)
    merged = _merged_box(detections)
    return FaceTrack(
        rotation=rotation,
        crop_box=merged,
        detection_count=frame_hits,
        sampled_frame_count=sampled_count,
        mean_confidence=sum(det.confidence for det in detections) / max(1, len(detections)),
        main_face_ambiguity_ratio=ambiguity_ratio,
        ambiguous=ambiguous,
    )


def _mean_area(detections: Sequence[FaceDetection]) -> float:
    return sum(det.box[2] * det.box[3] for det in detections) / max(1, len(detections))


def _mean_center_distance(detections: Sequence[FaceDetection], width: int, height: int) -> float:
    cx = width / 2.0
    cy = height / 2.0
    values = []
    for det in detections:
        x, y, w, h = det.box
        values.append(abs((x + w / 2.0) - cx) + abs((y + h / 2.0) - cy))
    return sum(values) / max(1, len(values))


def _merged_box(detections: Sequence[FaceDetection]) -> tuple[int, int, int, int]:
    left = min(det.box[0] for det in detections)
    top = min(det.box[1] for det in detections)
    right = max(det.box[0] + det.box[2] for det in detections)
    bottom = max(det.box[1] + det.box[3] for det in detections)
    return left, top, right - left, bottom - top


def _ambiguity_ratio(detections_by_frame: Sequence[Sequence[FaceDetection]]) -> float:
    ambiguous_frames = sum(1 for frame in detections_by_frame if len(frame) > 1)
    return ambiguous_frames / max(1, len(detections_by_frame))
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_face_preprocessing -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/daily_multimodal/embeddings/face_preprocessing.py tests/test_face_preprocessing.py
git commit -m "Add face preprocessing track selection"
```

---

### Task 2: ROI Clip Generation And Metadata

**Files:**
- Modify: `src/daily_multimodal/embeddings/face_preprocessing.py`
- Modify: `tests/test_face_preprocessing.py`

- [ ] **Step 1: Write failing tests for metadata and ROI clip writer**

Append:

```python
from daily_multimodal.embeddings.face_preprocessing import (
    FacePreprocessingConfig,
    write_preprocessed_metadata,
)


class FacePreprocessingMetadataTests(unittest.TestCase):
    def test_write_preprocessed_metadata_records_reproducible_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            track = choose_main_face_track(
                [[FaceDetection(0, 90, (100, 80, 60, 60), 0.95)]],
                frame_width=320,
                frame_height=240,
            )
            path = write_preprocessed_metadata(
                root / "face_preprocessing.json",
                sample_id="sample-1",
                source_clip=root / "window.mp4",
                roi_clip=root / "window_preprocessed.mp4",
                detector_backend="fake_detector",
                config=FacePreprocessingConfig(sample_fps=2.0, roi_margin=1.6, roi_size=384),
                track=track,
            )

            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["sample_id"], "sample-1")
        self.assertEqual(payload["detector_backend"], "fake_detector")
        self.assertEqual(payload["rotation"], 90)
        self.assertEqual(payload["crop_box"], [100, 80, 69, 65])
        self.assertEqual(payload["roi_size"], 384)
        self.assertFalse(payload["main_face_ambiguous"])
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_face_preprocessing.FacePreprocessingMetadataTests -v
```

Expected: import fails for `FacePreprocessingConfig`.

- [ ] **Step 3: Implement config and metadata writer**

Add to `face_preprocessing.py`:

```python
import json
from pathlib import Path


@dataclass(frozen=True)
class FacePreprocessingConfig:
    sample_fps: float = 2.0
    roi_margin: float = 1.6
    roi_size: int = 384
    min_track_detection_rate: float = 0.30
    ambiguity_threshold: float = 0.50


def write_preprocessed_metadata(
    path: Path,
    *,
    sample_id: str,
    source_clip: Path,
    roi_clip: Path,
    detector_backend: str,
    config: FacePreprocessingConfig,
    track: FaceTrack,
) -> Path:
    payload = {
        "sample_id": sample_id,
        "source_clip": str(source_clip),
        "roi_clip": str(roi_clip),
        "detector_backend": detector_backend,
        "sample_fps": config.sample_fps,
        "roi_margin": config.roi_margin,
        "roi_size": config.roi_size,
        "rotation": track.rotation,
        "crop_box": list(track.crop_box),
        "detection_count": track.detection_count,
        "sampled_frame_count": track.sampled_frame_count,
        "mean_confidence": track.mean_confidence,
        "main_face_ambiguity_ratio": track.main_face_ambiguity_ratio,
        "main_face_ambiguous": track.ambiguous,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
```

- [ ] **Step 4: Run tests**

Run:

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_face_preprocessing -v
```

Expected: all preprocessing tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/daily_multimodal/embeddings/face_preprocessing.py tests/test_face_preprocessing.py
git commit -m "Record face preprocessing metadata"
```

---

### Task 3: Integrate Preprocessed Profile Into Face Extraction

**Files:**
- Modify: `src/daily_multimodal/embeddings/face_real.py`
- Modify: `tests/test_face_real_embedding.py`

- [ ] **Step 1: Write failing integration test**

Add to `tests/test_face_real_embedding.py`:

```python
def test_preprocessed_profile_runs_openface_on_roi_clip_and_records_quality_flags(self):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cache_root = root / "cache"
        csv_path = _write_face_cache(
            cache_root,
            sample_id="sample-1",
            encoder_profile="face_preprocessed_openface_stats_v1",
            clip_start_seconds=12.5,
            clip_end_seconds=22.5,
        )
        executable = root / "FeatureExtraction"
        executable.write_text("fake executable", encoding="utf-8")
        openface_calls = []

        def fake_clip_extractor(_source_path, _start_seconds, _end_seconds, output_clip):
            output_clip.write_bytes(b"window-mp4")

        def fake_preprocessor(window, cache, source_clip):
            roi = csv_path.parent / "window_preprocessed.mp4"
            roi.write_bytes(b"roi-mp4")
            return roi, {
                "face_preprocessing_used": True,
                "face_preprocessing_profile": "face_preprocessed_openface_stats_v1",
                "detector_backend": "fake_detector",
                "rotation": 90,
                "crop_box": [10, 20, 100, 120],
                "main_face_ambiguity_ratio": 0.0,
            }

        def fake_openface_runner(openface_executable, clip_path, output_csv):
            openface_calls.append((openface_executable, clip_path, output_csv))
            _write_openface_csv(output_csv)

        summary = extract_face_real_embeddings(
            [_window("sample-1")],
            cache_root=cache_root,
            output_npz=root / "face_real_embeddings.npz",
            failures_out=root / "failures.json",
            encoder_profile="face_preprocessed_openface_stats_v1",
            openface_executable=executable,
            clip_extractor=fake_clip_extractor,
            openface_runner=fake_openface_runner,
            preprocess_clip=fake_preprocessor,
        )
        with np.load(root / "face_real_embeddings.npz", allow_pickle=True) as loaded:
            quality_flags = [json.loads(value) for value in loaded["quality_flags"].tolist()]

    self.assertEqual(summary["success_count"], 1)
    self.assertEqual(openface_calls[0][1].name, "window_preprocessed.mp4")
    self.assertTrue(quality_flags[0]["face_preprocessing_used"])
    self.assertEqual(quality_flags[0]["detector_backend"], "fake_detector")
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_face_real_embedding.FaceRealEmbeddingTests.test_preprocessed_profile_runs_openface_on_roi_clip_and_records_quality_flags
```

Expected: failure because `extract_face_real_embeddings()` has no `preprocess_clip` argument.

- [ ] **Step 3: Add profile-based preprocessing hook**

Change the signature in `face_real.py`:

```python
PreprocessClip = Callable[[dict[str, Any], dict[str, Any], Path], tuple[Path, dict[str, Any]]]

def extract_face_real_embeddings(..., preprocess_clip: PreprocessClip | None = None, ...):
```

Before `runner(executable, clip_path, csv_path)`, add:

```python
quality_overrides: dict[str, Any] = {}
openface_input = clip_path
if encoder_profile == "face_preprocessed_openface_stats_v1":
    preprocessor = preprocess_clip or _prepare_preprocessed_face_clip
    openface_input, quality_overrides = preprocessor(window, cache, clip_path)
runner(executable, openface_input, csv_path)
```

After `_face_quality(...)`, add:

```python
quality.update(quality_overrides)
```

Implement `_prepare_preprocessed_face_clip()` as a thin wrapper that calls functions from `face_preprocessing.py`.

- [ ] **Step 4: Run integration test**

Run:

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_face_real_embedding.FaceRealEmbeddingTests.test_preprocessed_profile_runs_openface_on_roi_clip_and_records_quality_flags
```

Expected: test passes.

- [ ] **Step 5: Run all face tests**

Run:

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_face_real_embedding tests.test_face_preprocessing -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/daily_multimodal/embeddings/face_real.py tests/test_face_real_embedding.py
git commit -m "Add preprocessed OpenFace profile"
```

---

### Task 4: Starting Tracking No-CSV ROI Retry

**Files:**
- Modify: `src/daily_multimodal/embeddings/face_real.py`
- Modify: `tests/test_face_real_embedding.py`

- [ ] **Step 1: Write failing test for ROI retry after OpenFace abort**

Add:

```python
def test_openface_starting_tracking_no_csv_retries_preprocessed_roi_once(self):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cache_root = root / "cache"
        csv_path = _write_face_cache(
            cache_root,
            sample_id="sample-1",
            clip_start_seconds=12.5,
            clip_end_seconds=22.5,
        )
        executable = root / "FeatureExtraction"
        executable.write_text("fake executable", encoding="utf-8")
        calls = []

        def fake_clip_extractor(_source_path, _start_seconds, _end_seconds, output_clip):
            output_clip.write_bytes(b"window-mp4")

        def fake_preprocessor(window, cache, source_clip):
            roi = csv_path.parent / "window_preprocessed_retry.mp4"
            roi.write_bytes(b"roi-mp4")
            return roi, {"face_preprocessing_retry": True}

        def fake_openface_runner(_exe, clip_path, output_csv):
            calls.append(clip_path.name)
            if len(calls) == 1:
                raise RuntimeError("Device or file opened\nStarting tracking")
            _write_openface_csv(output_csv)

        summary = extract_face_real_embeddings(
            [_window("sample-1")],
            cache_root=cache_root,
            output_npz=root / "face_real_embeddings.npz",
            failures_out=root / "failures.json",
            encoder_profile="face_raw_openface_stats_v1",
            openface_executable=executable,
            clip_extractor=fake_clip_extractor,
            openface_runner=fake_openface_runner,
            preprocess_clip=fake_preprocessor,
            retry_preprocessed_on_openface_abort=True,
        )

    self.assertEqual(summary["success_count"], 1)
    self.assertEqual(calls, ["window.mp4", "window_preprocessed_retry.mp4"])
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_face_real_embedding.FaceRealEmbeddingTests.test_openface_starting_tracking_no_csv_retries_preprocessed_roi_once
```

Expected: failure because retry flag does not exist.

- [ ] **Step 3: Implement bounded retry**

Add an argument:

```python
retry_preprocessed_on_openface_abort: bool = False
```

In the OpenFace exception handler:

```python
if retry_preprocessed_on_openface_abort and _is_openface_starting_tracking_abort(str(exc)):
    retry_clip, retry_quality = (preprocess_clip or _prepare_preprocessed_face_clip)(window, cache, clip_path)
    runner(executable, retry_clip, csv_path)
    quality_overrides.update(retry_quality)
    quality_overrides["openface_preprocessed_retry_used"] = True
else:
    raise
```

Define:

```python
def _is_openface_starting_tracking_abort(message: str) -> bool:
    return "Starting tracking" in message and "OpenFace did not create expected CSV" in message or "Starting tracking" in message
```

Keep retry count to one call only.

- [ ] **Step 4: Run tests**

Run:

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_face_real_embedding -v
```

Expected: all face tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/daily_multimodal/embeddings/face_real.py tests/test_face_real_embedding.py
git commit -m "Retry OpenFace aborts with preprocessed ROI"
```

---

### Task 5: CLI Controls And Full-Run Commands

**Files:**
- Modify: `scripts/13_extract_face_embeddings.py`
- Modify: `tests/test_face_real_embedding.py`
- Modify: `repo-docs/references/commands-and-artifacts.md`

- [ ] **Step 1: Write failing CLI test**

Add a subprocess test that invokes:

```powershell
python scripts/13_extract_face_embeddings.py `
  --window-index <tmp>\window_index.jsonl `
  --cache-root <tmp>\cache `
  --encoder-profile face_preprocessed_openface_stats_v1 `
  --preprocess-face `
  --face-preprocess-fps 2.0 `
  --face-roi-margin 1.6 `
  --face-roi-size 384 `
  --out <tmp>\face_preprocessed.npz `
  --failures-out <tmp>\failures.json `
  --summary-out <tmp>\summary.json
```

Expected failure before implementation: CLI rejects unknown args.

- [ ] **Step 2: Add CLI arguments**

Add to `scripts/13_extract_face_embeddings.py`:

```python
parser.add_argument("--preprocess-face", action="store_true")
parser.add_argument("--face-detector-backend", default="opencv_haar")
parser.add_argument("--face-preprocess-fps", type=float, default=2.0)
parser.add_argument("--face-roi-margin", type=float, default=1.6)
parser.add_argument("--face-roi-size", type=int, default=384)
parser.add_argument("--retry-preprocessed-on-openface-abort", action="store_true")
```

Auto-enable preprocessing when:

```python
preprocess_face = args.preprocess_face or args.encoder_profile == "face_preprocessed_openface_stats_v1"
```

- [ ] **Step 3: Add documented smoke command**

Add to `repo-docs/references/commands-and-artifacts.md`:

```bash
PYTHONPATH=src python scripts/13_extract_face_embeddings.py \
  --window-index outputs/window_index/real_cache_complete_10.jsonl \
  --cache-root outputs/cache/real_stage12_face_preprocessed_10 \
  --encoder-profile face_preprocessed_openface_stats_v1 \
  --openface-executable /mnt/dataset4/sitian/wzw/tools/openface/feature_extraction.sh \
  --preprocess-face \
  --face-detector-backend opencv_haar \
  --face-preprocess-fps 2.0 \
  --face-roi-margin 1.6 \
  --face-roi-size 384 \
  --retry-preprocessed-on-openface-abort \
  --out outputs/embeddings/face_preprocessed_openface_10_embeddings.npz \
  --failures-out outputs/reports/face_preprocessed_openface_10_failures.json \
  --summary-out outputs/reports/face_preprocessed_openface_10_quality_summary.json \
  --decision-out outputs/reports/face_preprocessing_decision_preprocessed_10.json
```

- [ ] **Step 4: Run CLI tests**

Run:

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_face_real_embedding -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add scripts/13_extract_face_embeddings.py tests/test_face_real_embedding.py repo-docs/references/commands-and-artifacts.md
git commit -m "Expose face preprocessing CLI controls"
```

---

### Task 6: Raw-vs-Preprocessed Comparison Report

**Files:**
- Create: `scripts/22_compare_face_preprocessing.py`
- Create: `tests/test_face_preprocessing_comparison.py`
- Modify: `repo-docs/references/commands-and-artifacts.md`

- [ ] **Step 1: Write failing comparison tests**

Create `tests/test_face_preprocessing_comparison.py`:

```python
import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("compare_face", Path("scripts/22_compare_face_preprocessing.py"))
compare_face = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_face)
```

Test:

```python
class FacePreprocessingComparisonTests(unittest.TestCase):
    def test_acceptance_requires_twenty_percent_success_lift_and_failure_drop(self):
        raw = {"success_count": 207, "failure_types": {"extraction_failed": 73}, "nan_count": 0}
        pre = {"success_count": 249, "failure_types": {"extraction_failed": 36}, "nan_count": 0}

        result = compare_face.compare_quality(raw, pre)

        self.assertTrue(result["quality_gate_passed"])
        self.assertEqual(result["minimum_success_count"], 249)
        self.assertEqual(result["maximum_extraction_failed"], 36)
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_face_preprocessing_comparison -v
```

Expected: fails because script does not exist.

- [ ] **Step 3: Implement comparison utility**

Core function:

```python
import math


def compare_quality(raw: dict, preprocessed: dict) -> dict:
    raw_success = int(raw.get("success_count") or 0)
    raw_failed = int((raw.get("failure_types") or {}).get("extraction_failed") or 0)
    pre_success = int(preprocessed.get("success_count") or 0)
    pre_failed = int((preprocessed.get("failure_types") or {}).get("extraction_failed") or 0)
    min_success = math.ceil(raw_success * 1.20)
    max_failed = math.floor(raw_failed * 0.50)
    nan_ok = int(preprocessed.get("nan_count") or 0) == 0
    return {
        "raw_success_count": raw_success,
        "preprocessed_success_count": pre_success,
        "minimum_success_count": min_success,
        "raw_extraction_failed": raw_failed,
        "preprocessed_extraction_failed": pre_failed,
        "maximum_extraction_failed": max_failed,
        "nan_ok": nan_ok,
        "quality_gate_passed": pre_success >= min_success and pre_failed <= max_failed and nan_ok,
    }
```

Add CLI args:

```python
--raw-summary
--preprocessed-summary
--out-json
--out-md
```

- [ ] **Step 4: Document comparison command**

```bash
PYTHONPATH=src python scripts/22_compare_face_preprocessing.py \
  --raw-summary outputs/reports/face_openface_real_full_quality_summary.json \
  --preprocessed-summary outputs/reports/face_preprocessed_openface_full_quality_summary.json \
  --out-json outputs/reports/face_preprocessing_acceptance.json \
  --out-md outputs/reports/face_preprocessing_acceptance.md
```

- [ ] **Step 5: Run tests**

Run:

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_face_preprocessing_comparison -v
python -m compileall -q scripts/22_compare_face_preprocessing.py
```

Expected: tests pass and compile succeeds.

- [ ] **Step 6: Commit**

```powershell
git add scripts/22_compare_face_preprocessing.py tests/test_face_preprocessing_comparison.py repo-docs/references/commands-and-artifacts.md
git commit -m "Compare raw and preprocessed face quality"
```

---

### Task 7: Server Smoke, Full Run, And Downstream Gate

**Files:**
- Modify: `repo-docs/change-log.md`
- Modify: `repo-docs/walkthroughs/one-real-run.md`

- [ ] **Step 1: Run 10-window smoke on server**

```bash
ssh ncc_serve_4090 "cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding && source /home/lzs/miniconda3/etc/profile.d/conda.sh && conda activate lzs && PYTHONPATH=src python scripts/13_extract_face_embeddings.py --window-index outputs/window_index/real_cache_complete_10.jsonl --cache-root outputs/cache/real_stage12_face_preprocessed_10 --encoder-profile face_preprocessed_openface_stats_v1 --openface-executable /mnt/dataset4/sitian/wzw/tools/openface/feature_extraction.sh --preprocess-face --face-preprocess-fps 2.0 --face-roi-margin 1.6 --face-roi-size 384 --retry-preprocessed-on-openface-abort --out outputs/embeddings/face_preprocessed_openface_10_embeddings.npz --failures-out outputs/reports/face_preprocessed_openface_10_failures.json --summary-out outputs/reports/face_preprocessed_openface_10_quality_summary.json --decision-out outputs/reports/face_preprocessing_decision_preprocessed_10.json"
```

Expected:

- Exit code `0` if all windows pass, or `1` only if quality failures are intentionally recorded.
- `.npz` exists.
- `nan_count=0`.
- `quality_flags` include preprocessing fields for every processed row.

- [ ] **Step 2: Run full preprocessed face extraction**

```bash
ssh ncc_serve_4090 "cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding && source /home/lzs/miniconda3/etc/profile.d/conda.sh && conda activate lzs && PYTHONPATH=src python scripts/13_extract_face_embeddings.py --window-index outputs/window_index/real_cache_complete_full.jsonl --cache-root outputs/cache/real_stage12_face_preprocessed_full --encoder-profile face_preprocessed_openface_stats_v1 --openface-executable /mnt/dataset4/sitian/wzw/tools/openface/feature_extraction.sh --preprocess-face --face-preprocess-fps 2.0 --face-roi-margin 1.6 --face-roi-size 384 --retry-preprocessed-on-openface-abort --out outputs/embeddings/face_preprocessed_openface_full_embeddings.npz --failures-out outputs/reports/face_preprocessed_openface_full_failures.json --summary-out outputs/reports/face_preprocessed_openface_full_quality_summary.json --decision-out outputs/reports/face_preprocessing_decision_preprocessed_full.json"
```

Expected quality gate:

- `success_count >= 249`
- `failure_types.extraction_failed <= 36`
- `nan_count == 0`

- [ ] **Step 3: Compare raw and preprocessed summaries**

```bash
ssh ncc_serve_4090 "cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding && PYTHONPATH=src python scripts/22_compare_face_preprocessing.py --raw-summary outputs/reports/face_openface_real_full_quality_summary.json --preprocessed-summary outputs/reports/face_preprocessed_openface_full_quality_summary.json --out-json outputs/reports/face_preprocessing_acceptance.json --out-md outputs/reports/face_preprocessing_acceptance.md"
```

Expected:

- `quality_gate_passed=true`
- Report lists raw success/failure, preprocessed success/failure, required thresholds.

- [ ] **Step 4: Pack all-real with preprocessed face**

Use the existing all-real pack command, replacing only face:

```bash
--face outputs/embeddings/face_preprocessed_openface_full_embeddings.npz
--out outputs/embeddings/all_complete_real_v2_preprocessed_face_embeddings.npz
--report-out outputs/reports/all_complete_real_v2_preprocessed_face_embedding_report.json
```

Expected:

- row count `781`
- all embedding shapes `(781, 256)`
- NaN count `0`
- face mask sum equals preprocessed success count

- [ ] **Step 5: Run downstream 5-seed comparison**

Run the existing fair/downstream ablation for raw and preprocessed face on the same sample IDs and `fatigue` target. Record:

- median RMSE
- MAE
- Pearson r
- per-seed direction
- bootstrap 95% CI

Promotion expected only if the Default-Branch Promotion Gate passes.

- [ ] **Step 6: Update repo-docs**

Update:

- `repo-docs/walkthroughs/one-real-run.md`
- `repo-docs/references/commands-and-artifacts.md`
- `repo-docs/change-log.md`

Include:

- commands run
- exact `success_count`
- exact `extraction_failed`
- exact mask sum
- downstream result
- whether raw remains default or preprocessed becomes default

- [ ] **Step 7: Final verification**

Run locally:

```powershell
$env:PYTHONPATH='src'; python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
python C:\Users\28303\.codex\skills\repo-docs\scripts\validate_repo_docs.py repo-docs --repo-root .
```

Expected:

- unittest: all tests pass
- compileall: exit code `0`
- repo-docs validator: `OK: 0 errors`

- [ ] **Step 8: Commit**

```powershell
git add src scripts tests repo-docs
git commit -m "Add preprocessed OpenFace face branch"
```

---

## Self-Review

- Spec coverage: raw branch preservation is covered by Tasks 3 and 5; pre-detection, rotations, main track, ROI crop, light enhancement boundary, short-missing policy, no-CSV retry, and acceptance thresholds are covered by Tasks 1-7.
- Placeholder scan: no task is allowed to end at “implement later”; each task includes concrete files, commands, and expected results.
- Type consistency: profile name is consistently `face_preprocessed_openface_stats_v1`; raw profile remains `face_raw_openface_stats_v1`; quality flags use `face_preprocessing_*` plus `openface_*_retry` names.
