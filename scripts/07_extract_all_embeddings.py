from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.embeddings.full_extract import run_full_basic_extraction


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract stage-8 complete-candidate basic embeddings.")
    parser.add_argument("--manifest", help="Event manifest JSONL.")
    parser.add_argument("--window-index", help="Existing window index JSONL.")
    parser.add_argument("--require-all-modalities", action="store_true", default=True)
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--encoder-profile", default="basic", choices=["basic"])
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--out", default="outputs/embeddings/all_complete_basic_embeddings.npz")
    parser.add_argument("--manifest-out", default="outputs/manifests/all_complete_multimodal_manifest.jsonl")
    parser.add_argument("--report-out", default="outputs/reports/all_complete_basic_embedding_report.json")
    parser.add_argument("--failures-out", default="outputs/reports/all_complete_basic_embedding_failures.json")
    args = parser.parse_args()
    if not args.manifest and not args.window_index:
        parser.error("Provide either --manifest or --window-index.")

    summary = run_full_basic_extraction(
        manifest=args.manifest,
        window_index=args.window_index,
        output_npz=args.out,
        manifest_out=args.manifest_out,
        report_out=args.report_out,
        failures_out=args.failures_out,
        require_all_modalities=not args.include_incomplete and args.require_all_modalities,
        max_windows=args.max_windows,
    )
    print(f"embedding_path={args.out}")
    print(f"manifest_path={args.manifest_out}")
    print(f"report_path={args.report_out}")
    print(f"failures_path={args.failures_out}")
    print(f"selected_windows={summary['selected_windows']}")
    print(f"success_count={summary['success_count']}")
    print(f"failure_count={summary['failure_count']}")
    return 0 if summary["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
