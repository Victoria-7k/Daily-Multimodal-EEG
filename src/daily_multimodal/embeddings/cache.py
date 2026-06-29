from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from daily_multimodal.alignment.time_utils import parse_absolute_time
from daily_multimodal.embeddings.failures import EmbeddingFailure, write_failure_list


AudioExtractor = Callable[[Path, float, float, Path], None]


@dataclass(frozen=True)
class RealCacheProfiles:
    eeg: str = "eeg_real_frozen_v1"
    wear: str = "wear_sequence_v1"
    face: str = "openface_temporal_v1"
    audio: str = "wavlm_frozen_v1"

    def for_modality(self, modality: str) -> str:
        return getattr(self, modality)


class DependencyMissingError(RuntimeError):
    pass


def build_cache_key(sample_id: str, modality: str, encoder_profile: str) -> str:
    parts = [sample_id, modality, encoder_profile]
    if any(not part or _unsafe_path_part(part) for part in parts):
        raise ValueError(f"cache key parts must not be empty, absolute, or path-traversing: {parts!r}")
    return "/".join(parts)


def prepare_real_embedding_cache(
    windows: list[dict[str, Any]],
    *,
    cache_root: Path | str,
    report_out: Path | str,
    failures_out: Path | str,
    profiles: RealCacheProfiles | None = None,
    audio_extractor: AudioExtractor | None = None,
) -> dict[str, Any]:
    profiles = profiles or RealCacheProfiles()
    extractor = audio_extractor or extract_audio_clip_ffmpeg
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    failures: list[EmbeddingFailure] = []
    summary: dict[str, Any] = {
        "stage": 12,
        "window_count": len(windows),
        "cache_root": str(root),
        "failures_path": str(failures_out),
        "modalities": {
            "eeg": _empty_modality_summary(),
            "wear": _empty_modality_summary(),
            "face": _empty_modality_summary(),
            "audio": _empty_modality_summary(),
        },
    }

    for window in windows:
        _prepare_audio_cache(window, root, profiles, extractor, summary, failures)
        _prepare_face_cache(window, root, profiles, summary, failures)
        _prepare_eeg_cache(window, root, profiles, summary, failures)
        _prepare_wear_cache(window, root, profiles, summary, failures)

    for modality, modality_summary in summary["modalities"].items():
        modality_summary["failure_count"] = sum(
            1 for failure in failures if failure.modality == modality
        )
    write_failure_list(failures, failures_out)
    _write_readiness_report(summary, report_out)
    return summary


def extract_audio_clip_ffmpeg(
    source: Path,
    start_seconds: float,
    end_seconds: float,
    output: Path,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise DependencyMissingError("ffmpeg executable was not found on PATH")
    duration = end_seconds - start_seconds
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_seconds:.6f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.6f}",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "ffmpeg failed"
        raise RuntimeError(message)


def _prepare_audio_cache(
    window: dict[str, Any],
    cache_root: Path,
    profiles: RealCacheProfiles,
    extractor: AudioExtractor,
    summary: dict[str, Any],
    failures: list[EmbeddingFailure],
) -> None:
    modality = "audio"
    encoder_profile = profiles.audio
    candidate = _select_video_candidate(window)
    if candidate is None:
        _record_missing(summary, failures, window, modality, encoder_profile, "select_audio_source", "<missing-video-candidate>")
        return
    source = Path(str(candidate.get("mp4_path", "")))
    if not source.is_file():
        _record_missing(summary, failures, window, modality, encoder_profile, "select_audio_source", str(source))
        return
    start_seconds, end_seconds = _candidate_clip_bounds(candidate, window)
    if end_seconds <= start_seconds:
        _record_failure(
            summary,
            failures,
            window,
            modality,
            encoder_profile,
            "extract_audio_clip",
            "extraction_failed",
            f"invalid clip bounds: {start_seconds} >= {end_seconds}",
            str(source),
        )
        return
    output = _sample_cache_dir(cache_root, "audio_clips", window, encoder_profile) / "audio.wav"
    metadata_out = output.with_suffix(".json")
    try:
        if not output.is_file():
            extractor(source, start_seconds, end_seconds, output)
        if not output.is_file():
            raise RuntimeError("audio extractor did not create output wav")
    except DependencyMissingError as exc:
        _record_failure(
            summary,
            failures,
            window,
            modality,
            encoder_profile,
            "extract_audio_clip",
            "dependency_missing",
            str(exc),
            str(source),
        )
        return
    except Exception as exc:
        _record_failure(
            summary,
            failures,
            window,
            modality,
            encoder_profile,
            "extract_audio_clip",
            "extraction_failed",
            str(exc),
            str(source),
        )
        return
    _write_json(
        metadata_out,
        {
            **_base_cache_record(window, modality, encoder_profile),
            "source_path": str(source),
            "wav_path": str(output),
            "clip_start_seconds": start_seconds,
            "clip_end_seconds": end_seconds,
            "target_sample_rate_hz": 16000,
            "target_channels": 1,
        },
    )
    _record_ready(summary, modality, metadata_out)


def _prepare_face_cache(
    window: dict[str, Any],
    cache_root: Path,
    profiles: RealCacheProfiles,
    summary: dict[str, Any],
    failures: list[EmbeddingFailure],
) -> None:
    modality = "face"
    encoder_profile = profiles.face
    candidate = _select_video_candidate(window)
    if candidate is None:
        _record_missing(summary, failures, window, modality, encoder_profile, "select_face_source", "<missing-video-candidate>")
        return
    source = Path(str(candidate.get("mp4_path", "")))
    if not source.is_file():
        _record_missing(summary, failures, window, modality, encoder_profile, "select_face_source", str(source))
        return
    output_dir = _sample_cache_dir(cache_root, "openface", window, encoder_profile)
    record_out = output_dir / "openface_target.json"
    _write_json(
        record_out,
        {
            **_base_cache_record(window, modality, encoder_profile),
            "source_path": str(source),
            "target_csv_path": str(output_dir / "openface.csv"),
            "openface_required": False,
            "note": "Stage 12 records the target CSV path; OpenFace runs in stage 14.",
        },
    )
    _record_ready(summary, modality, record_out)


def _prepare_eeg_cache(
    window: dict[str, Any],
    cache_root: Path,
    profiles: RealCacheProfiles,
    summary: dict[str, Any],
    failures: list[EmbeddingFailure],
) -> None:
    modality = "eeg"
    encoder_profile = profiles.eeg
    source = Path(str(window.get("eeg_bdf_path") or ""))
    if not source.is_file():
        _record_missing(summary, failures, window, modality, encoder_profile, "prepare_eeg_window", str(source))
        return
    record_out = _sample_cache_dir(cache_root, "eeg_windows", window, encoder_profile) / "window.json"
    _write_json(
        record_out,
        {
            **_base_cache_record(window, modality, encoder_profile),
            "source_path": str(source),
            "window_start_time": window.get("window_start_time", ""),
            "window_end_time": window.get("window_end_time", ""),
            "source_sampling_frequency_hz": window.get("eeg_sampling_frequency"),
            "target_resample_hz": 250,
            "target_window_samples": 2500,
        },
    )
    _record_ready(summary, modality, record_out)


def _prepare_wear_cache(
    window: dict[str, Any],
    cache_root: Path,
    profiles: RealCacheProfiles,
    summary: dict[str, Any],
    failures: list[EmbeddingFailure],
) -> None:
    modality = "wear"
    encoder_profile = profiles.wear
    source_paths = {
        "ppg": Path(str(window.get("wear_ppg_path") or "")),
        "gsr": Path(str(window.get("wear_gsr_path") or "")),
        "acc": Path(str(window.get("wear_acc_path") or "")),
    }
    missing = {name: path for name, path in source_paths.items() if not path.is_file()}
    if missing:
        source_text = ";".join(f"{name}={path}" for name, path in missing.items()) or "<missing-wear-source>"
        _record_missing(summary, failures, window, modality, encoder_profile, "prepare_wear_window", source_text)
        return
    record_out = _sample_cache_dir(cache_root, "wear_windows", window, encoder_profile) / "window.json"
    _write_json(
        record_out,
        {
            **_base_cache_record(window, modality, encoder_profile),
            "source_paths": {name: str(path) for name, path in source_paths.items()},
            "window_start_time": window.get("window_start_time", ""),
            "window_end_time": window.get("window_end_time", ""),
            "target_sample_rates_hz": {"ppg": 64, "gsr": 32, "acc": 32},
        },
    )
    _record_ready(summary, modality, record_out)


def _select_video_candidate(window: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [candidate for candidate in window.get("video_candidates", []) or [] if candidate.get("mp4_path")]
    if not candidates:
        for path in window.get("candidate_mp4_paths", []) or []:
            if path:
                return {"mp4_path": path, "clip_start_seconds": 0.0, "clip_end_seconds": _window_duration_seconds(window)}
        return None
    for candidate in candidates:
        if candidate.get("covers_window"):
            return candidate
    return candidates[0]


def _candidate_clip_bounds(candidate: dict[str, Any], window: dict[str, Any]) -> tuple[float, float]:
    start = _as_float(candidate.get("clip_start_seconds"), 0.0)
    end = candidate.get("clip_end_seconds")
    if end is None:
        end = start + _window_duration_seconds(window)
    return start, _as_float(end, start)


def _window_duration_seconds(window: dict[str, Any]) -> float:
    if window.get("window_size_seconds") is not None:
        return float(window["window_size_seconds"])
    try:
        start = parse_absolute_time(str(window["window_start_time"]))
        end = parse_absolute_time(str(window["window_end_time"]))
        return (end - start).total_seconds()
    except Exception:
        return 0.0


def _base_cache_record(window: dict[str, Any], modality: str, encoder_profile: str) -> dict[str, Any]:
    sample_id = str(window.get("sample_id", ""))
    return {
        "sample_id": sample_id,
        "event_id": window.get("event_id", ""),
        "subject_id": window.get("subject_id", ""),
        "modality": modality,
        "encoder_profile": encoder_profile,
        "cache_key": build_cache_key(sample_id, modality, encoder_profile),
    }


def _sample_cache_dir(
    cache_root: Path,
    modality_dir: str,
    window: dict[str, Any],
    encoder_profile: str,
) -> Path:
    sample_id = str(window.get("sample_id", ""))
    if _unsafe_path_part(sample_id) or _unsafe_path_part(encoder_profile):
        raise ValueError(f"cache key parts must be safe: {sample_id!r}, {encoder_profile!r}")
    return cache_root / modality_dir / sample_id / encoder_profile


def _record_ready(summary: dict[str, Any], modality: str, cache_record_path: Path) -> None:
    modality_summary = summary["modalities"][modality]
    modality_summary["ready_count"] += 1
    modality_summary["cache_entries"].append(str(cache_record_path))


def _record_missing(
    summary: dict[str, Any],
    failures: list[EmbeddingFailure],
    window: dict[str, Any],
    modality: str,
    encoder_profile: str,
    stage: str,
    source_path: str,
) -> None:
    _record_failure(
        summary,
        failures,
        window,
        modality,
        encoder_profile,
        stage,
        "source_missing",
        "required source file is missing",
        source_path or "<missing-source-path>",
    )


def _record_failure(
    summary: dict[str, Any],
    failures: list[EmbeddingFailure],
    window: dict[str, Any],
    modality: str,
    encoder_profile: str,
    stage: str,
    error_type: str,
    error: str,
    source_path: str,
) -> None:
    summary["modalities"][modality]["missing_count"] += 1
    failures.append(
        EmbeddingFailure(
            sample_id=str(window.get("sample_id") or "<missing-sample-id>"),
            event_id=str(window.get("event_id") or "<missing-event-id>"),
            subject_id=str(window.get("subject_id") or "<missing-subject-id>"),
            modality=modality,
            encoder_profile=encoder_profile,
            stage=stage,
            error_type=error_type,
            error=error,
            source_path=source_path or "<missing-source-path>",
            recoverable=True,
        )
    )


def _write_readiness_report(summary: dict[str, Any], report_out: Path | str) -> Path:
    out = Path(report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Real Embedding Readiness Report",
        "",
        f"Window count: {summary['window_count']}",
        f"Cache root: {summary['cache_root']}",
        f"Failures: {summary['failures_path']}",
        "",
        "## Modality readiness",
        "",
    ]
    for modality in ("EEG", "Wear", "Face", "Audio"):
        values = summary["modalities"][modality.lower()]
        lines.append(
            f"- {modality} ready: {values['ready_count']}, "
            f"missing: {values['missing_count']}, failures: {values['failure_count']}"
        )
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _empty_modality_summary() -> dict[str, Any]:
    return {
        "ready_count": 0,
        "missing_count": 0,
        "failure_count": 0,
        "cache_entries": [],
    }


def _unsafe_path_part(value: str) -> bool:
    return bool(
        value in {".", ".."}
        or "/" in value
        or "\\" in value
        or Path(value).is_absolute()
        or re.search(r"(^|[._-])\.\.($|[._-])", value)
    )


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
