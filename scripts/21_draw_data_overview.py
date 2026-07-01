from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

BLUE = {"xlight": "#EAF1FE", "light": "#CEDFFE", "base": "#A3BEFA", "dark": "#2E4780"}
GOLD = {"xlight": "#FFF4C2", "base": "#FFE15B", "dark": "#736422"}
ORANGE = {"xlight": "#FFEDDE", "light": "#FFBDA1", "base": "#F0986E", "dark": "#804126"}
OLIVE = {"base": "#A3D576", "dark": "#386411"}
PINK = {"base": "#F390CA", "dark": "#8A3A6F"}
NEUTRAL = {"xlight": "#F4F5F7", "base": "#C5CAD3", "dark": "#464C55"}

WEAR_RE = re.compile(
    r"_(?P<start>\d{14})_(?P<end>\d{14})(?:_(?P<modality>ACC|GSR|PPG|min))?\.(?P<extension>csv|mat)$",
    re.IGNORECASE,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw local Daily Multimodal data overview figures.")
    parser.add_argument("--manifest", default="outputs/manifests/events_manifest.jsonl")
    parser.add_argument("--window-index", default="outputs/window_index/window_index.jsonl")
    parser.add_argument("--quality-summary", default="outputs/reports/real_embedding_quality_summary.json")
    parser.add_argument("--out-dir", default="outputs/figures")
    parser.add_argument("--subject-id", help="Complete subject to use for the modality timeline.")
    parser.add_argument("--date", help="Optional YYYY-MM-DD date for a single-day subject timeline.")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    window_path = Path(args.window_index)
    quality_path = Path(args.quality_summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_jsonl(manifest_path)
    windows = load_jsonl(window_path)
    quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}

    completeness_rows, missing_counts, quality_issues = summarize_completeness(manifest, quality)
    csv_path = out_dir / "data_completeness_by_subject.csv"
    write_completeness_csv(csv_path, completeness_rows)

    overview_svg = out_dir / "subject_modality_completeness.svg"
    overview_svg.write_text(
        draw_completeness_svg(completeness_rows, manifest, missing_counts, quality_issues),
        encoding="utf-8",
    )

    subject_id = args.subject_id or choose_complete_subject(completeness_rows)
    timeline_summary, intervals, event_rows, start_time, end_time = build_timeline(
        manifest=manifest,
        windows=windows,
        subject_id=subject_id,
        date=args.date,
    )
    timeline_svg = (
        out_dir / f"{subject_id}_{args.date}_precise_timeline.svg"
        if args.date
        else out_dir / f"{subject_id}_modality_timeline.svg"
    )
    timeline_svg.write_text(
        draw_timeline_svg(subject_id, intervals, event_rows, start_time, end_time, timeline_summary),
        encoding="utf-8",
    )

    summary_path = out_dir / "data_overview_figures_summary.json"
    summary = {
        "generated_files": [str(overview_svg), str(timeline_svg), str(csv_path)],
        "source_files": [str(manifest_path), str(window_path), str(quality_path)],
        "subject_overview": {
            "events_total": len(manifest),
            "subjects": len(completeness_rows),
            "complete_multimodal_candidates": sum(
                int(bool(row.get("is_complete_multimodal_candidate"))) for row in manifest
            ),
            "missing_counts": missing_counts,
            "quality_issues": quality_issues,
        },
        "timeline_subject": timeline_summary,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"overview_svg={overview_svg}")
    print(f"timeline_svg={timeline_svg}")
    print(f"summary_json={summary_path}")
    print(f"completeness_csv={csv_path}")
    return 0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_time(value: str) -> datetime:
    value = value.replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(value)


def parse_wear_interval(path: str) -> tuple[datetime, datetime] | None:
    match = WEAR_RE.search(Path(path).name)
    if not match:
        return None
    return (
        datetime.strptime(match.group("start"), "%Y%m%d%H%M%S"),
        datetime.strptime(match.group("end"), "%Y%m%d%H%M%S"),
    )


def summarize_completeness(
    manifest: list[dict[str, Any]],
    quality: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    modalities = modality_columns()
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        by_subject[str(row["subject_id"])].append(row)

    rows: list[dict[str, Any]] = []
    for subject_id in sorted(by_subject):
        subject_rows = by_subject[subject_id]
        total = len(subject_rows)
        record: dict[str, Any] = {"subject_id": subject_id, "events_total": total}
        for label, field in modalities:
            count = sum(int(bool(row.get(field))) for row in subject_rows)
            record[f"{label}_count"] = count
            record[f"{label}_rate"] = count / total if total else 0.0
        rows.append(record)

    missing_counts = {
        label: len(manifest) - sum(int(bool(row.get(field))) for row in manifest)
        for label, field in modalities[:-1]
    }
    missing_counts["Wear complete"] = len(manifest) - sum(
        int(bool(row.get("is_complete_wear_event"))) for row in manifest
    )
    missing_counts["All candidate"] = len(manifest) - sum(
        int(bool(row.get("is_complete_multimodal_candidate"))) for row in manifest
    )

    quality_modalities = quality.get("modalities", {})
    quality_issues = {
        "EEG": int(quality_modalities.get("eeg", {}).get("failure_count", 0) or 0),
        "Wear": int(quality_modalities.get("wear", {}).get("failure_count", 0) or 0),
        "Video/Face": int(
            quality_modalities.get("face", {}).get(
                "masked_count",
                quality_modalities.get("face", {}).get("failure_count", 0),
            )
            or 0
        ),
        "Audio": int(quality_modalities.get("audio", {}).get("failure_count", 0) or 0),
    }
    return rows, missing_counts, quality_issues


def modality_columns() -> list[tuple[str, str]]:
    return [
        ("EEG", "has_eeg"),
        ("PPG", "has_ppg"),
        ("GSR", "has_gsr"),
        ("ACC", "has_acc"),
        ("Video/Face", "has_video"),
        ("Audio", "has_audio"),
        ("All candidate", "is_complete_multimodal_candidate"),
    ]


def write_completeness_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["subject_id", "events_total"]
    for label, _ in modality_columns():
        fieldnames.extend([f"{label}_count", f"{label}_rate"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose_complete_subject(rows: list[dict[str, Any]]) -> str:
    candidates = [
        row for row in rows
        if int(row["All candidate_count"]) == int(row["events_total"])
    ]
    if not candidates:
        candidates = rows
    candidates = sorted(
        candidates,
        key=lambda row: (-int(row["All candidate_count"]), -int(row["events_total"]), str(row["subject_id"])),
    )
    return str(candidates[0]["subject_id"])


def build_timeline(
    *,
    manifest: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    subject_id: str,
    date: str | None = None,
) -> tuple[dict[str, Any], dict[str, list[tuple[datetime, datetime, str]]], list[dict[str, Any]], datetime, datetime]:
    subject_rows = sorted(
        [
            row for row in manifest
            if row["subject_id"] == subject_id and (date is None or str(row.get("absolute_onset_time", "")).startswith(date))
        ],
        key=lambda row: parse_time(row["absolute_onset_time"]),
    )
    subject_windows = sorted(
        [
            row for row in windows
            if row["subject_id"] == subject_id and (date is None or str(row.get("absolute_onset_time", "")).startswith(date))
        ],
        key=lambda row: parse_time(row["window_start_time"]),
    )
    if not subject_rows:
        suffix = f" on {date}" if date else ""
        raise ValueError(f"No manifest rows found for {subject_id}{suffix}")

    intervals: dict[str, list[tuple[datetime, datetime, str]]] = defaultdict(list)
    seen = set()
    for row in subject_rows:
        key = (
            row.get("session_id"),
            row.get("segment_id"),
            row.get("eeg_recording_start_time"),
            row.get("eeg_recording_duration"),
        )
        if key in seen or not row.get("eeg_recording_start_time") or not row.get("eeg_recording_duration"):
            continue
        seen.add(key)
        start = parse_time(str(row["eeg_recording_start_time"]))
        end = start + timedelta(seconds=float(row["eeg_recording_duration"]))
        intervals["EEG"].append((start, end, str(row.get("session_id", ""))))

    for modality, field in [("PPG", "wear_ppg_path"), ("GSR", "wear_gsr_path"), ("ACC", "wear_acc_path")]:
        seen_paths = set()
        for row in subject_rows:
            source_path = str(row.get(field, ""))
            if not source_path or source_path in seen_paths:
                continue
            seen_paths.add(source_path)
            parsed = parse_wear_interval(source_path)
            if parsed:
                intervals[modality].append((parsed[0], parsed[1], str(row.get("session_id", ""))))

    window_by_event = {_window_event_key(row): row for row in subject_windows}
    precise_video_full = 0
    precise_audio_full = 0
    precise_paths: set[tuple[str, str, str]] = set()
    precise_audio_paths: set[tuple[str, str, str]] = set()
    for row in subject_windows:
        candidates = row.get("video_candidates") or []
        if any(bool(candidate.get("covers_window")) for candidate in candidates):
            precise_video_full += 1
        if any(bool(candidate.get("covers_window") and candidate.get("has_audio_stream")) for candidate in candidates):
            precise_audio_full += 1
        for candidate in candidates:
            start_text = str(candidate.get("mp4_start_time") or "")
            end_text = str(candidate.get("mp4_end_time") or "")
            mp4_path = str(candidate.get("mp4_path") or "")
            if not (start_text and end_text and mp4_path):
                continue
            key = (mp4_path, start_text, end_text)
            if key not in precise_paths:
                precise_paths.add(key)
                intervals["Precise MP4"].append((parse_time(start_text), parse_time(end_text), Path(mp4_path).name))
            if candidate.get("has_audio_stream") and key not in precise_audio_paths:
                precise_audio_paths.add(key)
                intervals["Audio stream"].append((parse_time(start_text), parse_time(end_text), Path(mp4_path).name))

    if not intervals.get("Precise MP4"):
        for modality, field in [("Video/Face day candidate", "has_face"), ("Audio day candidate", "has_audio")]:
            by_day: dict[Any, list[tuple[datetime, datetime]]] = defaultdict(list)
            for row in subject_windows:
                if row.get(field):
                    day = parse_time(str(row["absolute_onset_time"])).date()
                    by_day[day].append((parse_time(str(row["window_start_time"])), parse_time(str(row["window_end_time"]))))
            for day, spans in sorted(by_day.items()):
                intervals[modality].append((min(span[0] for span in spans), max(span[1] for span in spans), str(day)))

    event_rows = []
    for row in subject_rows:
        event = dict(row)
        event_id = _event_key(event)
        window = window_by_event.get(event_id)
        candidates = (window or {}).get("video_candidates") or []
        primary = next((candidate for candidate in candidates if candidate.get("covers_window")), candidates[0] if candidates else {})
        mp4_name = Path(str(primary.get("mp4_path") or "")).name if primary else ""
        fatigue = str((event.get("labels") or event.get("label_columns") or {}).get("fatigue", ""))
        event_time = parse_time(str(event["absolute_onset_time"]))
        compact_label = f"{event_time:%H:%M} f={fatigue}"
        if mp4_name:
            compact_label = f"{compact_label} {mp4_name}"
        event["timeline_label"] = compact_label
        event["timeline_mp4_name"] = mp4_name
        event_rows.append(event)

    all_times: list[datetime] = []
    for spans in intervals.values():
        for start, end, _ in spans:
            all_times.extend([start, end])
    all_times.extend(parse_time(str(row["absolute_onset_time"])) for row in event_rows)
    start_time = min(all_times) - timedelta(hours=1)
    end_time = max(all_times) + timedelta(hours=1)

    summary = {
        "subject_id": subject_id,
        "date": date,
        "events_total": len(event_rows),
        "complete_multimodal_candidates": sum(
            int(bool(row.get("is_complete_multimodal_candidate"))) for row in event_rows
        ),
        "precise_video_full_coverage": precise_video_full,
        "precise_audio_full_coverage": precise_audio_full,
        "time_start": start_time.isoformat(sep=" "),
        "time_end": end_time.isoformat(sep=" "),
        "interval_counts": {key: len(value) for key, value in intervals.items()},
        "note": "Precise MP4 and Audio stream intervals come from video_candidates when present.",
    }
    return summary, intervals, event_rows, start_time, end_time


def draw_completeness_svg(
    rows: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    missing_counts: dict[str, int],
    quality_issues: dict[str, int],
) -> str:
    width, height = 2220, 1550
    left, top = 170, 220
    label_width, cell_width, cell_height = 120, 185, 64
    modalities = modality_columns()
    parts = svg_root_start(width, height)
    parts.append(text_svg(70, 58, "所有被试的数据完整性概况", 42, weight=700))
    parts.append(
        text_svg(
            70,
            112,
            "每个单元格为事件级可用比例；右侧汇总 manifest 缺失和本地 real embedding 质量问题。",
            24,
            fill=TOKENS["muted"],
        )
    )
    parts.append(
        text_svg(
            70,
            146,
            "数据源：outputs/manifests/events_manifest.jsonl、outputs/reports/real_embedding_quality_summary.json",
            20,
            fill=TOKENS["muted"],
        )
    )

    for j, (label, _) in enumerate(modalities):
        x = left + label_width + j * cell_width + cell_width / 2
        parts.append(text_svg(x, top - 42, label, 18, anchor="middle"))

    for i, row in enumerate(rows):
        y = top + i * cell_height
        parts.append(text_svg(left, y + cell_height / 2 + 7, str(row["subject_id"]), 22))
        parts.append(text_svg(left + 74, y + cell_height / 2 + 7, f"n={row['events_total']}", 16, fill=TOKENS["muted"]))
        for j, (label, _) in enumerate(modalities):
            rate = float(row[f"{label}_rate"])
            count = int(row[f"{label}_count"])
            total = int(row["events_total"])
            x = left + label_width + j * cell_width
            color = completeness_color(rate)
            parts.append(rect_svg(x + 3, y + 4, cell_width - 8, cell_height - 9, fill=color, stroke=TOKENS["panel"]))
            parts.append(text_svg(x + cell_width / 2, y + cell_height / 2 + 7, f"{count}/{total}", 19, anchor="middle"))

    grid_x0 = left + label_width
    grid_x1 = grid_x0 + len(modalities) * cell_width
    grid_y1 = top + len(rows) * cell_height
    for i in range(len(rows) + 1):
        y = top + i * cell_height
        parts.append(line_svg(grid_x0, y, grid_x1, y, TOKENS["grid"]))
    for j in range(len(modalities) + 1):
        x = grid_x0 + j * cell_width
        parts.append(line_svg(x, top, x, grid_y1, TOKENS["grid"]))

    legend_y = grid_y1 + 52
    parts.append(text_svg(grid_x0, legend_y + 10, "完整率", 18, fill=TOKENS["muted"]))
    for k, (label, color) in enumerate(
        [("0%", ORANGE["xlight"]), ("50%", blend(ORANGE["light"], BLUE["base"], 0.5)), ("100%", OLIVE["base"])]
    ):
        x = grid_x0 + 90 + k * 150
        parts.append(rect_svg(x, legend_y - 10, 55, 25, fill=color, stroke=TOKENS["axis"]))
        parts.append(text_svg(x + 65, legend_y + 10, label, 18, fill=TOKENS["muted"]))

    panel_x = grid_x1 + 170
    panel_y = 214
    panel_width = width - panel_x - 80
    panel_height = 1110
    parts.append(rect_svg(panel_x, panel_y, panel_width, panel_height, fill=TOKENS["panel"], stroke=TOKENS["grid"], radius=12))
    parts.append(text_svg(panel_x + 34, panel_y + 58, "缺失与问题摘要", 30, weight=700))
    complete = sum(int(bool(row.get("is_complete_multimodal_candidate"))) for row in manifest)
    parts.append(
        text_svg(
            panel_x + 34,
            panel_y + 102,
            f"总事件数：{len(manifest)}；完整多模态候选：{complete}",
            18,
            fill=TOKENS["muted"],
        )
    )

    issue_rows = [
        ("EEG", missing_counts["EEG"], quality_issues["EEG"], "shape mismatch"),
        ("PPG", missing_counts["PPG"], 0, "manifest missing"),
        ("GSR", missing_counts["GSR"], 0, "manifest missing"),
        ("ACC", missing_counts["ACC"], 0, "manifest missing"),
        ("Wear complete", missing_counts["Wear complete"], quality_issues["Wear"], "PPG+GSR+ACC"),
        ("Video/Face", missing_counts["Video/Face"], quality_issues["Video/Face"], "quality masked"),
        ("Audio", missing_counts["Audio"], quality_issues["Audio"], "real failures"),
        ("All candidate", missing_counts["All candidate"], 0, "all modalities"),
    ]
    max_issue = max(missing + issue for _, missing, issue, _ in issue_rows) or 1
    bar_x = panel_x + 220
    bar_y = panel_y + 145
    bar_width = panel_width - 300
    for i, (label, missing, issue, note) in enumerate(issue_rows):
        y = bar_y + i * 86
        parts.append(text_svg(panel_x + 34, y + 22, label, 21))
        parts.append(text_svg(panel_x + 34, y + 50, note, 16, fill=TOKENS["muted"]))
        parts.append(rect_svg(bar_x, y, bar_width, 28, fill=NEUTRAL["xlight"], stroke=TOKENS["grid"]))
        missing_width = bar_width * missing / max_issue
        issue_width = bar_width * issue / max_issue
        if missing_width:
            parts.append(rect_svg(bar_x, y, missing_width, 28, fill=ORANGE["base"], stroke=ORANGE["dark"]))
        if issue_width:
            parts.append(rect_svg(bar_x + missing_width, y, issue_width, 28, fill=PINK["base"], stroke=PINK["dark"]))
        parts.append(text_svg(bar_x + bar_width + 15, y + 21, f"缺失 {missing}", 16))
        if issue:
            parts.append(text_svg(bar_x + bar_width + 15, y + 45, f"质量/失败 {issue}", 16, fill=PINK["dark"]))

    legend_y2 = panel_y + panel_height - 72
    parts.append(rect_svg(panel_x + 34, legend_y2, 24, 20, fill=ORANGE["base"], stroke=ORANGE["dark"]))
    parts.append(text_svg(panel_x + 70, legend_y2 + 16, "manifest/日期候选缺失事件", 16, fill=TOKENS["muted"]))
    parts.append(rect_svg(panel_x + 310, legend_y2, 24, 20, fill=PINK["base"], stroke=PINK["dark"]))
    parts.append(text_svg(panel_x + 346, legend_y2 + 16, "本地 real embedding 质量 mask 或失败", 16, fill=TOKENS["muted"]))

    parts.append("</svg>")
    return "\n".join(parts)


def draw_timeline_svg(
    subject_id: str,
    intervals: dict[str, list[tuple[datetime, datetime, str]]],
    event_rows: list[dict[str, Any]],
    start_time: datetime,
    end_time: datetime,
    summary: dict[str, Any],
) -> str:
    width, height = 2400, 1450
    plot_x, plot_y = 250, 245
    plot_width, row_height = 1980, 96
    video_lane = "Precise MP4" if intervals.get("Precise MP4") else "Video/Face day candidate"
    audio_lane = "Audio stream" if intervals.get("Audio stream") else "Audio day candidate"
    lanes = ["EEG", "PPG", "GSR", "ACC", video_lane, audio_lane, "Rating events"]
    span_seconds = (end_time - start_time).total_seconds()

    def x_of(value: datetime) -> float:
        return plot_x + ((value - start_time).total_seconds() / span_seconds) * plot_width

    colors = {
        "EEG": (BLUE["base"], BLUE["dark"]),
        "PPG": (OLIVE["base"], OLIVE["dark"]),
        "GSR": (GOLD["base"], GOLD["dark"]),
        "ACC": (ORANGE["base"], ORANGE["dark"]),
        "Precise MP4": (PINK["base"], PINK["dark"]),
        "Audio stream": (BLUE["light"], BLUE["dark"]),
        "Video/Face day candidate": (PINK["base"], PINK["dark"]),
        "Audio day candidate": (BLUE["light"], BLUE["dark"]),
    }
    fatigue_palette = {
        "1": (BLUE["xlight"], BLUE["dark"]),
        "2": (BLUE["base"], BLUE["dark"]),
        "3": (GOLD["base"], GOLD["dark"]),
        "4": (ORANGE["base"], ORANGE["dark"]),
        "5": (PINK["base"], PINK["dark"]),
    }

    parts = svg_root_start(width, height)
    parts.append(text_svg(74, 58, f"{subject_id} 多模态覆盖时间轴", 42, weight=700))
    parts.append(
        text_svg(
            74,
            112,
            "横条为模态覆盖区间；媒体为本地 manifest 的日期级候选/事件窗口跨度；竖线为情绪评分事件，颜色表示 fatigue 评分。",
            24,
            fill=TOKENS["muted"],
        )
    )

    day = datetime(start_time.year, start_time.month, start_time.day)
    while day <= end_time:
        x = x_of(day)
        parts.append(line_svg(x, plot_y - 38, x, plot_y + len(lanes) * row_height - 18, TOKENS["grid"]))
        if day >= start_time:
            parts.append(text_svg(x + 6, plot_y - 66, day.strftime("%m-%d"), 16, fill=TOKENS["muted"]))
        day += timedelta(days=1)

    for i, lane in enumerate(lanes):
        y_center = plot_y + i * row_height + row_height / 2
        parts.append(text_svg(74, y_center + 7, lane, 21))
        parts.append(line_svg(plot_x, y_center, plot_x + plot_width, y_center, TOKENS["grid"]))
        if lane == "Rating events":
            for row in event_rows:
                event_time = parse_time(str(row["absolute_onset_time"]))
                x = x_of(event_time)
                score = str(row.get("labels", {}).get("fatigue", ""))
                fill, stroke = fatigue_palette.get(score, (NEUTRAL["base"], NEUTRAL["dark"]))
                parts.append(line_svg(x, y_center - 34, x, y_center + 28, stroke, width=2))
                parts.append(circle_svg(x, y_center + 31, 5, fill, stroke))
                if summary.get("date"):
                    label = str(row.get("timeline_label") or event_time.strftime("%H:%M"))
                    label_y = y_center - 42 - (18 if event_rows.index(row) % 2 else 0)
                    parts.append(text_svg(x + 6, label_y, label, 13, fill=TOKENS["muted"]))
        else:
            fill, stroke = colors[lane]
            for start, end, _ in intervals.get(lane, []):
                x0 = x_of(start)
                x1 = max(x0 + 3, x_of(end))
                parts.append(rect_svg(x0, y_center - 18, x1 - x0, 36, fill=fill, stroke=stroke, radius=8))

    axis_y = plot_y + len(lanes) * row_height - 18
    parts.append(line_svg(plot_x, axis_y, plot_x + plot_width, axis_y, TOKENS["axis"], width=2))
    parts.append(text_svg(plot_x, axis_y + 45, start_time.strftime("%Y-%m-%d %H:%M"), 16, fill=TOKENS["muted"]))
    parts.append(
        text_svg(plot_x + plot_width, axis_y + 45, end_time.strftime("%Y-%m-%d %H:%M"), 16, fill=TOKENS["muted"], anchor="end")
    )

    cards = [
        ("事件数", str(summary["events_total"])),
        ("完整候选事件", str(summary["complete_multimodal_candidates"])),
        ("EEG 记录段", str(summary["interval_counts"].get("EEG", 0))),
        (
            "Wear 文件段/模态",
            "PPG {PPG} / GSR {GSR} / ACC {ACC}".format(
                PPG=summary["interval_counts"].get("PPG", 0),
                GSR=summary["interval_counts"].get("GSR", 0),
                ACC=summary["interval_counts"].get("ACC", 0),
            ),
        ),
        (
            "精确媒体覆盖",
            f"V {summary.get('precise_video_full_coverage', 0)} / A {summary.get('precise_audio_full_coverage', 0)}",
        ),
    ]
    card_y, card_width = 1170, 410
    for i, (label, value) in enumerate(cards):
        x = 74 + i * (card_width + 36)
        parts.append(rect_svg(x, card_y, card_width, 110, fill=TOKENS["panel"], stroke=TOKENS["grid"], radius=10))
        parts.append(text_svg(x + 24, card_y + 44, label, 18, fill=TOKENS["muted"]))
        parts.append(text_svg(x + 24, card_y + 88, value, 28, weight=700))

    legend_y = 1325
    parts.append(text_svg(74, legend_y + 16, "Fatigue 评分颜色", 18, fill=TOKENS["muted"]))
    for i, score in enumerate(["1", "2", "3", "4", "5"]):
        x = 240 + i * 90
        fill, stroke = fatigue_palette[score]
        parts.append(circle_svg(x + 13, legend_y + 8, 13, fill, stroke))
        parts.append(text_svg(x + 34, legend_y + 16, score, 18, fill=TOKENS["muted"]))
    parts.append(
        text_svg(
            780,
            legend_y + 16,
            _timeline_note(start_time, end_time, summary),
            18,
            fill=TOKENS["muted"],
        )
    )

    parts.append("</svg>")
    return "\n".join(parts)


def _event_key(row: dict[str, Any]) -> str:
    event_id = str(row.get("event_id") or "")
    if event_id:
        return event_id
    sample_id = str(row.get("sample_id") or "")
    if sample_id:
        return sample_id.rsplit("_win-", 1)[0]
    parts = [
        str(row.get("subject_id", "")),
        str(row.get("session_id", "")),
        str(row.get("segment_id", "00")),
        f"row-{int(row.get('rating_row_index', 0)):04d}",
    ]
    return "_".join(parts)


def _window_event_key(row: dict[str, Any]) -> str:
    event_id = str(row.get("event_id") or "")
    if event_id:
        return event_id
    sample_id = str(row.get("sample_id") or "")
    return sample_id.rsplit("_win-", 1)[0] if sample_id else ""


def _timeline_note(start_time: datetime, end_time: datetime, summary: dict[str, Any]) -> str:
    date = summary.get("date")
    date_text = str(date) if date else f"{start_time:%Y-%m-%d} 至 {end_time:%Y-%m-%d}"
    if summary.get("precise_video_full_coverage") or summary.get("precise_audio_full_coverage"):
        return f"时间范围：{date_text}；媒体条来自 ffprobe 精确 video_candidates，V/A 为完整覆盖事件数。"
    return f"时间范围：{date_text}；媒体条基于日期候选，不代表 ffprobe 精确秒级覆盖。"


def svg_root_start(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif; letter-spacing: 0; }",
        ".mono { font-family: Consolas, 'Microsoft YaHei', monospace; }",
        "</style>",
        rect_svg(0, 0, width, height, fill=TOKENS["surface"]),
    ]


def text_svg(
    x: float,
    y: float,
    value: str,
    size: int,
    *,
    fill: str = TOKENS["ink"],
    anchor: str = "start",
    weight: int = 400,
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}">{html.escape(str(value))}</text>'
    )


def rect_svg(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    stroke: str | None = None,
    radius: int = 0,
) -> str:
    stroke_attr = f' stroke="{stroke}" stroke-width="1"' if stroke else ""
    radius_attr = f' rx="{radius}" ry="{radius}"' if radius else ""
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}"'
        f'{radius_attr} fill="{fill}"{stroke_attr}/>'
    )


def line_svg(x1: float, y1: float, x2: float, y2: float, stroke: str, *, width: int = 1) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"/>'


def circle_svg(x: float, y: float, radius: float, fill: str, stroke: str) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1"/>'


def completeness_color(rate: float) -> str:
    if rate >= 0.999:
        return OLIVE["base"]
    if rate <= 0:
        return ORANGE["xlight"]
    return blend(ORANGE["light"], BLUE["base"], rate)


def blend(color_a: str, color_b: str, t: float) -> str:
    a = hex_to_rgb(color_a)
    b = hex_to_rgb(color_b)
    t = max(0.0, min(1.0, t))
    return "#%02x%02x%02x" % tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


if __name__ == "__main__":
    raise SystemExit(main())
