from __future__ import annotations

import copy
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from daily_multimodal.alignment.event_windows import DEFAULT_END_SECONDS, DEFAULT_START_SECONDS
from daily_multimodal.alignment.time_utils import parse_absolute_time


ProbeFunc = Callable[[str], dict[str, Any]]
TimedProbeFunc = Callable[..., dict[str, Any]]


def align_video_audio_rows(
    rows: list[dict[str, Any]],
    *,
    start_seconds: int | float = DEFAULT_START_SECONDS,
    end_seconds: int | float = DEFAULT_END_SECONDS,
    timezone_name: str = "Asia/Shanghai",
    ffprobe_timeout_seconds: int | float | None = 10,
    ffprobe_func: ProbeFunc | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach precise MP4 clip candidates to event manifest rows."""
    probe = ffprobe_func or (lambda path: run_ffprobe(path, timeout_seconds=ffprobe_timeout_seconds))
    cache: dict[str, dict[str, Any] | Exception] = {}
    enriched: list[dict[str, Any]] = []
    failed_files: dict[str, str] = {}
    probed_files: set[str] = set()

    for row in rows:
        output_row = copy.deepcopy(row)
        event_time = parse_absolute_time(str(row["absolute_onset_time"]))
        window_start = event_time + timedelta(seconds=float(start_seconds))
        window_end = event_time + timedelta(seconds=float(end_seconds))
        candidates: list[dict[str, Any]] = []

        for path in _as_list(row.get("candidate_mp4_paths")):
            if path not in cache:
                try:
                    cache[path] = probe(path)
                    probed_files.add(path)
                except Exception as exc:  # pragma: no cover - exercised on real ffprobe failures
                    cache[path] = exc
                    failed_files[path] = str(exc)
            metadata = cache[path]
            if isinstance(metadata, Exception):
                continue
            try:
                candidate = _candidate_from_probe(
                    path,
                    metadata,
                    window_start=window_start,
                    window_end=window_end,
                    timezone_name=timezone_name,
                )
            except Exception as exc:  # pragma: no cover - defensive metadata parsing
                failed_files[path] = str(exc)
                continue
            if candidate["overlap_seconds"] > 0:
                candidates.append(candidate)

        output_row["video_candidates"] = candidates
        output_row["has_precise_video"] = bool(candidates)
        output_row["has_precise_audio"] = any(candidate["has_audio_stream"] for candidate in candidates)
        enriched.append(output_row)

    report = {
        "events_total": len(enriched),
        "events_with_video_day": sum(bool(row.get("has_video") or row.get("candidate_mp4_paths")) for row in enriched),
        "events_with_precise_video_overlap": sum(bool(row.get("has_precise_video")) for row in enriched),
        "events_with_precise_audio_overlap": sum(bool(row.get("has_precise_audio")) for row in enriched),
        "events_with_full_window_video_coverage": sum(
            any(candidate.get("covers_window") for candidate in row.get("video_candidates", []))
            for row in enriched
        ),
        "unique_mp4_probed": len(probed_files),
        "ffprobe_failed_files": len(failed_files),
        "failed_files": failed_files,
        "window_start_seconds": _clean_number(start_seconds),
        "window_end_seconds": _clean_number(end_seconds),
        "ffprobe_timeout_seconds": _clean_number(ffprobe_timeout_seconds),
        "timezone": timezone_name,
    }
    return enriched, report


def run_ffprobe(path: str, *, timeout_seconds: int | float | None = 10) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_entries",
        "format=duration:format_tags=creation_time:stream=codec_type,codec_name,sample_rate,channels:stream_tags=creation_time",
        "-show_format",
        "-show_streams",
        path,
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=_subprocess_timeout(timeout_seconds),
    )
    return json.loads(completed.stdout)


def probe_many_mp4_paths(
    paths: list[str],
    *,
    timeout_seconds: int | float | None = 10,
    max_workers: int = 8,
    cache_path: Path | str | None = None,
    progress_every: int = 25,
    retry_failed: bool = False,
    ffprobe_func: TimedProbeFunc | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Probe unique MP4 paths with incremental caching and progress reporting."""
    unique_paths = sorted({path for path in paths if path})
    cached = _load_probe_cache(cache_path)
    pending = [
        path
        for path in unique_paths
        if path not in cached or (retry_failed and not cached[path].get("ok"))
    ]
    report = {
        "unique_mp4_total": len(unique_paths),
        "cache_hits": len(unique_paths) - len(pending),
        "newly_probed": 0,
        "ffprobe_success_files": 0,
        "ffprobe_failed_files": 0,
        "ffprobe_timeout_seconds": _clean_number(timeout_seconds),
        "ffprobe_max_workers": max_workers,
        "retry_failed": retry_failed,
    }
    if not pending:
        _refresh_probe_report_counts(report, unique_paths, cached)
        return cached, report

    probe = ffprobe_func or run_ffprobe
    timeout_label = "none" if timeout_seconds is None else f"{timeout_seconds}s"
    _print_progress(
        f"ffprobe start: unique={len(unique_paths)} cached={report['cache_hits']} "
        f"pending={len(pending)} workers={max_workers} timeout={timeout_label}"
    )
    completed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_probe_one_for_cache, path, probe, timeout_seconds): path
            for path in pending
        }
        for future in as_completed(futures):
            path = futures[future]
            result = future.result()
            cached[path] = result
            _append_probe_cache_record(cache_path, path, result)
            completed_count += 1
            report["newly_probed"] += 1
            _refresh_probe_report_counts(report, unique_paths, cached)
            if completed_count == 1 or completed_count % progress_every == 0 or completed_count == len(pending):
                _print_progress(
                    f"ffprobe progress: {completed_count}/{len(pending)} new, "
                    f"success={report['ffprobe_success_files']} failed={report['ffprobe_failed_files']}"
                )
    _refresh_probe_report_counts(report, unique_paths, cached)
    return cached, report


def save_aligned_manifest(rows: list[dict[str, Any]], output: Path | str) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def save_alignment_report(report: dict[str, Any], output: Path | str) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def cached_probe_func(cache: dict[str, dict[str, Any]]) -> ProbeFunc:
    def _probe(path: str) -> dict[str, Any]:
        result = cache.get(path)
        if not result:
            raise RuntimeError("ffprobe cache missing path")
        if result.get("ok"):
            return result["metadata"]
        raise RuntimeError(str(result.get("error", "ffprobe failed")))

    return _probe


def _candidate_from_probe(
    path: str,
    metadata: dict[str, Any],
    *,
    window_start: datetime,
    window_end: datetime,
    timezone_name: str,
) -> dict[str, Any]:
    duration = float(metadata.get("format", {}).get("duration") or 0.0)
    mp4_start = _parse_creation_time(metadata).astimezone(ZoneInfo(timezone_name)).replace(tzinfo=None)
    mp4_end = mp4_start + timedelta(seconds=duration)
    overlap_start = max(window_start, mp4_start)
    overlap_end = min(window_end, mp4_end)
    overlap_seconds = max(0.0, (overlap_end - overlap_start).total_seconds())
    audio = _first_audio_stream(metadata)
    return {
        "mp4_path": path,
        "mp4_start_time": _format_time(mp4_start),
        "mp4_end_time": _format_time(mp4_end),
        "duration_seconds": round(duration, 6),
        "clip_start_seconds": round(max(0.0, (window_start - mp4_start).total_seconds()), 6),
        "clip_end_seconds": round(min(duration, (window_end - mp4_start).total_seconds()), 6),
        "overlap_seconds": round(overlap_seconds, 6),
        "covers_window": bool(mp4_start <= window_start and mp4_end >= window_end),
        "has_audio_stream": audio is not None,
        "audio_codec": str(audio.get("codec_name", "")) if audio else "",
        "audio_sample_rate": _int_or_none(audio.get("sample_rate")) if audio else None,
        "audio_channels": _int_or_none(audio.get("channels")) if audio else None,
    }


def _parse_creation_time(metadata: dict[str, Any]) -> datetime:
    tags = metadata.get("format", {}).get("tags", {}) or {}
    value = tags.get("creation_time")
    if not value:
        for stream in metadata.get("streams", []) or []:
            value = (stream.get("tags") or {}).get("creation_time")
            if value:
                break
    if not value:
        raise ValueError("ffprobe metadata has no creation_time")
    text = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _first_audio_stream(metadata: dict[str, Any]) -> dict[str, Any] | None:
    for stream in metadata.get("streams", []) or []:
        if stream.get("codec_type") == "audio":
            return stream
    return None


def _format_time(value: datetime) -> str:
    if value.microsecond:
        return value.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip("0").rstrip(".")
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, str) and value:
        return [value]
    return []


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_number(value: int | float | None) -> int | float | None:
    if value is None:
        return None
    number = float(value)
    if number.is_integer():
        return int(number)
    return number


def _probe_one_for_cache(
    path: str,
    probe: TimedProbeFunc,
    timeout_seconds: int | float | None,
) -> dict[str, Any]:
    try:
        metadata = probe(path, timeout_seconds=timeout_seconds)
        return {"ok": True, "metadata": metadata}
    except Exception as exc:  # pragma: no cover - exact subprocess failures vary by platform
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _load_probe_cache(cache_path: Path | str | None) -> dict[str, dict[str, Any]]:
    if not cache_path:
        return {}
    path = Path(cache_path)
    if not path.is_file():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            mp4_path = record.get("mp4_path")
            if mp4_path:
                cache[str(mp4_path)] = {key: value for key, value in record.items() if key != "mp4_path"}
    return cache


def _append_probe_cache_record(
    cache_path: Path | str | None,
    mp4_path: str,
    result: dict[str, Any],
) -> None:
    if not cache_path:
        return
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"mp4_path": mp4_path, **result}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _print_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _refresh_probe_report_counts(
    report: dict[str, Any],
    unique_paths: list[str],
    cached: dict[str, dict[str, Any]],
) -> None:
    report["ffprobe_success_files"] = sum(1 for path in unique_paths if cached.get(path, {}).get("ok"))
    report["ffprobe_failed_files"] = sum(1 for path in unique_paths if path in cached and not cached[path].get("ok"))


def _subprocess_timeout(timeout_seconds: int | float | None) -> float | None:
    if timeout_seconds is None:
        return None
    timeout = float(timeout_seconds)
    if timeout <= 0:
        return None
    return timeout
