from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.alignment.event_windows import load_window_index
from daily_multimodal.embeddings.face_real import (
    extract_face_real_embeddings,
    write_face_preprocessing_decision,
    write_face_quality_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit raw face/OpenFace quality without preprocessing frames.")
    parser.add_argument("--window-index", required=True)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--cache-root", default="outputs/cache")
    parser.add_argument("--encoder-profile", default="face_raw_openface_stats_v1")
    parser.add_argument("--openface-executable")
    parser.add_argument("--allow-opencv-fallback", action="store_true")
    parser.add_argument("--min-success-rate", type=float, default=0.50)
    parser.add_argument("--scratch-out", default="outputs/embeddings/face_quality_audit_scratch.npz")
    parser.add_argument("--failures-out", default="outputs/reports/face_quality_audit_failures.json")
    parser.add_argument("--summary-out", default="outputs/reports/face_quality_audit.json")
    parser.add_argument("--report-out", default="outputs/reports/face_quality_audit.md")
    parser.add_argument("--decision-out", default="outputs/reports/face_preprocessing_decision.json")
    args = parser.parse_args()

    windows = load_window_index(args.window_index)
    selected = windows[: args.max_windows] if args.max_windows is not None else windows
    summary = extract_face_real_embeddings(
        selected,
        cache_root=args.cache_root,
        output_npz=args.scratch_out,
        failures_out=args.failures_out,
        encoder_profile=args.encoder_profile,
        openface_executable=args.openface_executable,
        allow_opencv_fallback=args.allow_opencv_fallback,
        min_success_rate=args.min_success_rate,
    )
    write_face_quality_summary(summary, args.summary_out)
    decision_path = write_face_preprocessing_decision(summary, args.decision_out)
    _write_markdown_report(summary, args.report_out, decision_path=decision_path)

    print(f"summary_path={args.summary_out}")
    print(f"report_path={args.report_out}")
    print(f"decision_path={args.decision_out}")
    print(f"embedded_count={summary['embedded_count']}")
    print(f"success_count={summary['success_count']}")
    print(f"failure_count={summary['failure_count']}")
    return 0 if summary["failure_count"] == 0 else 1


def _write_markdown_report(summary: dict, path: Path | str, *, decision_path: Path) -> Path:
    lines = [
        "# Face Quality Audit",
        "",
        f"- Encoder profile: `{summary.get('encoder_profile')}`",
        f"- Embedded windows: {summary.get('embedded_count')}",
        f"- Usable windows: {summary.get('success_count')}",
        f"- Masked windows: {summary.get('masked_count')}",
        f"- Failure count: {summary.get('failure_count')}",
        f"- Mean detection success rate: {summary.get('mean_face_detection_success_rate')}",
        f"- Mean OpenFace confidence: {summary.get('mean_openface_confidence')}",
        f"- Mean low confidence ratio: {summary.get('mean_low_confidence_ratio')}",
        f"- Mean pose bad ratio: {summary.get('mean_pose_bad_ratio')}",
        f"- Mean dark frame ratio: {summary.get('mean_dark_frame_ratio')}",
        f"- Mean blur frame ratio: {summary.get('mean_blur_frame_ratio')}",
        f"- Mean multi-face ratio: {summary.get('mean_multi_face_ratio')}",
        f"- Decision JSON: `{decision_path}`",
        "",
        "```json",
        json.dumps(summary.get("failure_types", {}), ensure_ascii=False, indent=2),
        "```",
    ]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
