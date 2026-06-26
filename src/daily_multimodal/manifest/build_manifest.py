from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from daily_multimodal.alignment.time_utils import (
    parse_absolute_time,
    subject_to_video_subject,
    time_to_video_day,
)
from daily_multimodal.config import load_simple_yaml
from daily_multimodal.io.wear import WearFile, discover_wear_files
from daily_multimodal.schema import CONTEXT_COLUMNS, EMOTION_LABEL_COLUMNS


def build_manifest(
    *,
    eeg_dataset: Path | str,
    video_root: Path | str,
    wear_root: Path | str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build an event-level manifest from sidecar metadata without reading BDF/MP4 payloads."""
    eeg_root = Path(eeg_dataset)
    video_path = Path(video_root)
    wear_path = Path(wear_root)
    wear_index = _build_wear_index(discover_wear_files(wear_path))

    rows: list[dict[str, Any]] = []
    sampling_counter: Counter[str] = Counter()
    for beh_tsv in sorted(eeg_root.glob("sub-*/ses-*/beh/*.tsv")):
        subject_id = beh_tsv.parts[-4]
        session_id = beh_tsv.parts[-3]
        segment_id = _segment_id_from_beh(beh_tsv)
        eeg_json = _matching_eeg_sidecar(beh_tsv, ".json")
        eeg_bdf = _matching_eeg_sidecar(beh_tsv, ".bdf")
        eeg_meta = _read_eeg_json(eeg_json) if eeg_json else {}
        recording_start = eeg_meta.get("RecordingStartTime")
        recording_duration = eeg_meta.get("RecordingDuration")
        sampling_frequency = eeg_meta.get("SamplingFrequency")
        if sampling_frequency is not None:
            sampling_counter[str(float(sampling_frequency))] += 1

        with beh_tsv.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row_index, label_row in enumerate(reader, start=1):
                if not any(label_row.values()):
                    continue
                event_time = parse_absolute_time(label_row["absolute_onset_time"])
                wear_match = _match_wear(wear_index, event_time)
                video_day_dir = (
                    video_path
                    / subject_to_video_subject(subject_id)
                    / time_to_video_day(event_time)
                )
                mp4_paths = _media_files(video_day_dir, {".mp4"})
                aac_paths = _media_files(video_day_dir, {".aac", ".m4a", ".wav"})
                labels = {name: label_row.get(name, "") for name in EMOTION_LABEL_COLUMNS}
                context = {name: label_row.get(name, "") for name in CONTEXT_COLUMNS}

                has_eeg = bool(eeg_json and eeg_bdf)
                has_ppg = "PPG" in wear_match
                has_gsr = "GSR" in wear_match
                has_acc = "ACC" in wear_match
                has_video = bool(mp4_paths)
                has_audio = has_video or bool(aac_paths)

                event_id = f"{subject_id}_{session_id}_{segment_id}_row-{row_index:04d}"
                rows.append(
                    {
                        "event_id": event_id,
                        "subject_id": subject_id,
                        "session_id": session_id,
                        "segment_id": segment_id,
                        "rating_row_index": row_index,
                        "absolute_onset_time": event_time.isoformat(sep=" "),
                        "eeg_recording_start_time": recording_start,
                        "eeg_recording_duration": recording_duration,
                        "eeg_onset_seconds": _float_or_none(label_row.get("onset")),
                        "rating_duration_seconds": _float_or_none(label_row.get("duration")),
                        "eeg_sampling_frequency": sampling_frequency,
                        "eeg_bdf_path": str(eeg_bdf) if eeg_bdf else "",
                        "eeg_json_path": str(eeg_json) if eeg_json else "",
                        "beh_tsv_path": str(beh_tsv),
                        "wear_ppg_path": str(wear_match.get("PPG", "")),
                        "wear_gsr_path": str(wear_match.get("GSR", "")),
                        "wear_acc_path": str(wear_match.get("ACC", "")),
                        "video_day_dir": str(video_day_dir),
                        "candidate_mp4_paths": [str(path) for path in mp4_paths],
                        "candidate_audio_paths": [str(path) for path in aac_paths],
                        "has_eeg": has_eeg,
                        "has_ppg": has_ppg,
                        "has_gsr": has_gsr,
                        "has_acc": has_acc,
                        "has_video": has_video,
                        "has_audio": has_audio,
                        "is_complete_wear_event": has_ppg and has_gsr and has_acc,
                        "is_complete_multimodal_candidate": (
                            has_eeg and has_ppg and has_gsr and has_acc and has_video and has_audio
                        ),
                        "labels": labels,
                        **context,
                    }
                )

    coverage = _coverage(rows)
    coverage["eeg_sampling_frequency_counts"] = dict(sorted(sampling_counter.items()))
    return rows, coverage


def save_manifest(rows: list[dict[str, Any]], output: Path | str) -> Path:
    """Save manifest as JSONL or CSV. Parquet requires optional dependencies and is not default."""
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        _save_csv(rows, out)
        return out
    if out.suffix.lower() == ".parquet":
        try:
            import pandas as pd  # type: ignore

            pd.DataFrame(rows).to_parquet(out, index=False)
            return out
        except Exception:
            out = out.with_suffix(".jsonl")
    _save_jsonl(rows, out)
    return out


def save_coverage(coverage: dict[str, Any], output: Path | str) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build event-level Daily Multimodal manifest.")
    parser.add_argument("--config", required=True, help="Path to paths.server.yaml.")
    parser.add_argument("--out", required=True, help="Manifest output path (.jsonl, .csv, or .parquet if available).")
    parser.add_argument("--coverage-out", help="Coverage JSON output path.")
    args = parser.parse_args(argv)

    config = load_simple_yaml(args.config)
    roots = config["data_roots"]
    rows, coverage = build_manifest(
        eeg_dataset=roots["eeg_dataset"],
        video_root=roots["video"],
        wear_root=roots["wear"],
    )
    manifest_path = save_manifest(rows, args.out)
    coverage_path = save_coverage(
        coverage,
        args.coverage_out
        or Path(config["outputs"]["reports"]) / "manifest_coverage.json",
    )
    print(f"manifest_path={manifest_path}")
    print(f"coverage_path={coverage_path}")
    for key in [
        "events_total",
        "complete_wear_events",
        "video_day_events",
        "complete_multimodal_candidates",
    ]:
        print(f"{key}={coverage[key]}")
    return 0


def _build_wear_index(files: list[WearFile]) -> list[tuple[datetime, datetime, dict[str, Path]]]:
    grouped: dict[tuple[datetime, datetime], dict[str, Path]] = defaultdict(dict)
    for item in files:
        if item.modality in {"ACC", "GSR", "PPG", "SUMMARY", "SUMMARY_MAT", "MIN"}:
            grouped[(item.start_time, item.end_time)][item.modality] = item.path
    return [(start, end, paths) for (start, end), paths in sorted(grouped.items())]


def _match_wear(
    index: list[tuple[datetime, datetime, dict[str, Path]]],
    event_time: datetime,
) -> dict[str, Path]:
    matches = [
        (end - start, paths)
        for start, end, paths in index
        if start <= event_time <= end
    ]
    if not matches:
        return {}
    return sorted(matches, key=lambda item: item[0])[0][1]


def _segment_id_from_beh(path: Path) -> str:
    stem = path.stem
    if "_emotion_beh-" in stem:
        return stem.rsplit("_emotion_beh-", 1)[1]
    return "00"


def _matching_eeg_sidecar(beh_path: Path, suffix: str) -> Path | None:
    segment_id = _segment_id_from_beh(beh_path)
    eeg_dir = beh_path.parent.parent / "eeg"
    if segment_id == "00":
        pattern = f"*task-dailylife_eeg{suffix}"
    else:
        pattern = f"*task-dailylife_eeg-{segment_id}{suffix}"
    matches = sorted(eeg_dir.glob(pattern))
    return matches[0] if matches else None


def _read_eeg_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    return {
        "RecordingStartTime": data.get("RecordingStartTime"),
        "RecordingDuration": data.get("RecordingDuration"),
        "SamplingFrequency": data.get("SamplingFrequency"),
    }


def _media_files(directory: Path, suffixes: set[str]) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete_wear = sum(bool(row["is_complete_wear_event"]) for row in rows)
    video_day = sum(bool(row["has_video"]) for row in rows)
    complete_multi = sum(bool(row["is_complete_multimodal_candidate"]) for row in rows)
    per_subject: dict[str, dict[str, int]] = defaultdict(lambda: {
        "events_total": 0,
        "complete_wear_events": 0,
        "video_day_events": 0,
        "complete_multimodal_candidates": 0,
    })
    for row in rows:
        subject = row["subject_id"]
        per_subject[subject]["events_total"] += 1
        per_subject[subject]["complete_wear_events"] += int(row["is_complete_wear_event"])
        per_subject[subject]["video_day_events"] += int(row["has_video"])
        per_subject[subject]["complete_multimodal_candidates"] += int(
            row["is_complete_multimodal_candidate"]
        )
    return {
        "events_total": len(rows),
        "complete_wear_events": complete_wear,
        "video_day_events": video_day,
        "complete_multimodal_candidates": complete_multi,
        "per_subject": dict(sorted(per_subject.items())),
    }


def _save_jsonl(rows: list[dict[str, Any]], out: Path) -> None:
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_csv(rows: list[dict[str, Any]], out: Path) -> None:
    if not rows:
        out.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })


def _float_or_none(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())

