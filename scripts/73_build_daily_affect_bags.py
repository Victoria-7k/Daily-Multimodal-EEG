#!/usr/bin/env python3
"""Build EEG-aligned daily-affect EMA bags from window-level modality embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from daily_multimodal.daily_affect.ema_bags import build_daily_affect_bags


DEFAULT_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_EMBEDDINGS_ROOT = Path("/vePFS-0x0d/DailyEEG_multimodal/embeddings")
DEFAULT_SPLITS_ROOT = Path("/vePFS-0x0d/DailyEEG/splits_new")
DEFAULT_OUT_ROOT = DEFAULT_ROOT / "outputs/daily_affect_ordinal_20260903/bags"
DEFAULT_PROTOCOLS = ("cross_subject", "cross_day", "within_subject_day")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--index-path", type=Path)
    parser.add_argument("--embeddings-root", type=Path, default=DEFAULT_EMBEDDINGS_ROOT)
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument("--protocols", default=",".join(DEFAULT_PROTOCOLS))
    parser.add_argument("--route-id", default="B0_Wphysio_full")
    parser.add_argument("--route-ids", help="Comma-separated route ids. Overrides --route-id.")
    parser.add_argument("--branches", help="Comma-separated explicit branch names for a custom route.")
    parser.add_argument("--target-label", default="fatigue")
    parser.add_argument("--eeg-branch", default="eeg")
    parser.add_argument("--eeg-token-root", default="eeg_encoder_256d_tokens")
    parser.add_argument("--eeg-token-seed", type=int)
    parser.add_argument("--wear-token-seed", type=int)
    parser.add_argument("--seeds", default="240800")
    parser.add_argument("--max-events", type=int, help="Build only the first N complete EMA events for smoke tests.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="Root directory for EMA bag artifacts.")
    args = parser.parse_args()

    index_path = args.index_path or args.root / "index/eeg_aligned_window_index.jsonl"
    protocols = split_csv(args.protocols)
    route_ids = split_csv(args.route_ids) if args.route_ids else (args.route_id,)
    explicit_branches = split_csv(args.branches) if args.branches else None
    seeds = [int(value) for value in split_csv(args.seeds)]
    results: list[dict[str, Any]] = []
    for protocol in protocols:
        for route_id in route_ids:
            for seed in seeds:
                eeg_seed = int(args.eeg_token_seed) if args.eeg_token_seed is not None else seed
                wear_seed = int(args.wear_token_seed) if args.wear_token_seed is not None else seed
                print(f"building protocol={protocol} route={route_id} seed={seed} eeg_branch={args.eeg_branch}", flush=True)
                out_dir = args.out_root / protocol / effective_output_route(route_id, args.eeg_branch, explicit_branches) / f"seed_{seed}"
                result = build_daily_affect_bags(
                    index_path=index_path,
                    splits_root=args.splits_root,
                    embeddings_root=args.embeddings_root,
                    protocol=protocol,
                    route_id=route_id,
                    out_dir=out_dir,
                    target_label=args.target_label,
                    eeg_branch=args.eeg_branch,
                    eeg_seed=eeg_seed,
                    wear_seed=wear_seed,
                    eeg_token_root=args.eeg_token_root,
                    explicit_branches=explicit_branches,
                    max_events=args.max_events,
                )
                result["seed"] = int(seed)
                result["eeg_token_seed"] = int(eeg_seed)
                result["wear_token_seed"] = int(wear_seed)
                results.append(result)
                print(f"wrote {result['bag_path']} bags={result['row_count']}", flush=True)
    output = {
        "script": Path(__file__).name,
        "target_label": args.target_label,
        "index_path": str(index_path),
        "splits_root": str(args.splits_root),
        "embeddings_root": str(args.embeddings_root),
        "out_root": str(args.out_root),
        "run_count": len(results),
        "results": results,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    out_json = args.out_root / "daily_affect_bag_build_report.json"
    out_md = args.out_root / "daily_affect_bag_build_report.md"
    out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(output, out_md)
    print(f"run_count={len(results)}")
    print(f"out_json={out_json}")
    print(f"out_md={out_md}")
    return 0


def effective_output_route(route_id: str, eeg_branch: str, explicit_branches: tuple[str, ...] | None) -> str:
    if explicit_branches is not None:
        return route_id
    return route_id if eeg_branch == "eeg" else f"{route_id}__{eeg_branch}"


def split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def write_markdown(output: dict[str, Any], path: Path) -> None:
    lines = [
        "# Daily-Affect EMA Bag Build Report",
        "",
        f"- run_count: `{output['run_count']}`",
        f"- index_path: `{output['index_path']}`",
        "",
        "| protocol | route | seed | bags | tokens shape | split counts | bag path |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in output["results"]:
        lines.append(
            f"| {row['protocol']} | {row['route_id']} | {row['seed']} | {row['row_count']} | "
            f"{row['tokens_shape']} | {row['split_counts']} | `{row['bag_path']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
