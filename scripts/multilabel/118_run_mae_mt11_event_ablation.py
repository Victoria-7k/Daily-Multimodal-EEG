#!/usr/bin/env python3
"""Run frozen MAE replacements against the archived A1+MT11 0814 baseline.

The B0 reference is read from the completed MT11 structure-matrix run rather
than retrained.  Each candidate copies its exact EMA bags and replaces exactly
selected modality slots with frozen label-free Stage-A MAE tokens.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from daily_multimodal.daily_affect.ema_bags import LABEL_NAMES, load_jsonl
from daily_multimodal.daily_affect.npz_compat import install_numpy_core_pickle_aliases
from daily_multimodal.daily_affect.training import load_bag_dataset
from daily_multimodal.training.structure_emotion import conditions, event_targets, run_condition

REFERENCE_ROUTE = "A1_Wphysio_no_audio__eeg_eegpt_partial_ft_multitask_11label_v1"
REFERENCE_ROOT = ROOT / "outputs/multiemotion_20260913/structure_matrix_A1"
MAE_ROOT = ROOT / "outputs/mae_20260924"


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _reference_bag(root: Path, protocol: str) -> Path:
    return root / "bags" / protocol / REFERENCE_ROUTE / "seed_240800" / "ema_bags.npz"


def _reference_metric(root: Path, protocol: str, seed: int) -> Path:
    return root / "runs" / protocol / "window_attention_regression_full_mean" / f"seed_{seed}" / "metrics.json"


def _mae_path(mae_root: Path, protocol: str, modality: str) -> Path:
    if modality not in {"eeg", "wear", "video"}:
        raise ValueError(f"unsupported MAE modality: {modality}")
    return mae_root / protocol / f"{modality}_seed_240800" / "window_embeddings.npz"


def _replacement_bag(out_root: Path, protocol: str, condition: str) -> Path:
    return out_root / "bags" / protocol / condition / "seed_240800" / "ema_bags.npz"


def _replacements(condition: str) -> tuple[tuple[str, int, str, str], ...]:
    if condition == 'B0_VIDEO_COMMON':return ()
    if condition == 'E_POOL':return _replacements('E1')
    if condition in ('V_ROI','V_FULL_MATCH'):return _replacements('V1')
    if condition in {"M3", "M4", "M5-F"}:
        slots = {"M3": ("E1", "V1"), "M4": ("W1", "V1"), "M5-F": ("E1", "W1", "V1")}
        return tuple(item for single in slots[condition] for item in _replacements(single))
    if condition == "E1":
        return (("eeg", 0, "eeg_eegpt_partial_ft_multitask_11label_v1", "eeg_mae_stage_a_label_free"),)
    if condition == "W1":
        return (("wear", 1, "wear_physio", "wear_mae_stage_a_label_free"),)
    if condition == "V1":
        return (("video", 2, "video_A1", "video_mae_stage_a_label_free"),)
    if condition == "M2":
        return (
            ("eeg", 0, "eeg_eegpt_partial_ft_multitask_11label_v1", "eeg_mae_stage_a_label_free"),
            ("wear", 1, "wear_physio", "wear_mae_stage_a_label_free"),
        )
    raise ValueError(f"unsupported MAE replacement condition: {condition}")


def _candidate_route(condition: str) -> str:
    if condition=='B0_VIDEO_COMMON':return 'A1_MT11_reference__B0_VIDEO_COMMON_matched_video_availability'
    kind = "single" if len(_replacements(condition)) == 1 else "multiple"
    return f"A1_MT11_reference__{condition}_{kind}_modality_replacement"


def _supervision(condition: str) -> str:
    if condition=='B0_VIDEO_COMMON':return 'MT11_EEG_retained__DINO_A1_common_mask__11label_supervised_downstream'
    retained = "MT11_EEG_retained" if all(slot != 0 for _, slot, _, _ in _replacements(condition)) else "label_free_EEG_MAE"
    return f"{retained}__frozen_MAE_replacements__11label_supervised_downstream"


def _write_replaced_bag(*, reference: Path, mae_paths: dict[str, Path], destination: Path, condition: str, video_common_mask: Path | None = None) -> None:
    install_numpy_core_pickle_aliases()
    with np.load(reference, allow_pickle=True) as loaded:
        payload = {name: loaded[name] for name in loaded.files}
    matrix = payload["sample_id_matrix"].astype(str)
    tokens = payload["tokens"].astype(np.float32, copy=True)
    masks = payload["modality_mask"].astype(np.int8, copy=True)
    sources = json.loads(str(payload["source_npz_json"].item()))
    flat = matrix.reshape(-1)
    for modality, slot, old_key, new_key in _replacements(condition):
        mae = mae_paths[modality]
        with np.load(mae, allow_pickle=True) as loaded:
            embedding = loaded["embedding"].astype(np.float32)
            valid = loaded["valid_mask"].astype(bool)
            sample_id = loaded["sample_id"].astype(str)
        if embedding.shape != (len(sample_id), 256) or valid.shape != (len(sample_id),):
            raise ValueError(f"invalid MAE shape: {mae}")
        if not np.isfinite(embedding).all() or np.any(embedding[~valid] != 0):
            raise ValueError(f"invalid MAE values or unavailable rows: {mae}")
        position = {value: index for index, value in enumerate(sample_id.tolist())}
        if len(position) != len(sample_id) or any(value not in position for value in flat.tolist()):
            raise ValueError(f"MAE token cannot align to reference bag: {mae}")
        indices = np.asarray([position[value] for value in flat.tolist()], dtype=np.int64).reshape(matrix.shape)
        tokens[:, :, slot, :] = embedding[indices]
        masks[:, :, slot] = valid[indices].astype(np.int8)
        if old_key not in sources:
            raise ValueError(f"reference bag misses expected source {old_key}")
        sources[new_key] = str(mae)
        del sources[old_key]
    if video_common_mask is not None:
        with np.load(video_common_mask,allow_pickle=False) as common:
            sid=common['sample_id'].astype(str);mask=common['valid_mask'].astype(bool)
        position={value:i for i,value in enumerate(sid)}
        if len(position)!=28819 or mask.shape!=(28819,) or any(value not in position for value in flat):raise ValueError('invalid video common identity')
        indices=np.asarray([position[value] for value in flat]).reshape(matrix.shape)
        common_event=mask[indices]
        if condition in ('V_ROI','V_FULL_MATCH') and not np.array_equal(masks[:,:,2].astype(bool),common_event):raise ValueError('candidate video mask differs from common')
        if np.any(common_event & ~masks[:,:,2].astype(bool)):raise ValueError('common mask adds unavailable video rows')
        masks[:,:,2]=common_event.astype(np.int8)
        tokens[:,:,2,:][~common_event]=0
        payload['video_common_mask_source']=np.asarray(str(video_common_mask))
    payload["tokens"] = tokens
    payload["modality_mask"] = masks
    payload["route_id"] = np.asarray(_candidate_route(condition))
    payload["source_npz_json"] = np.asarray(json.dumps(sources, ensure_ascii=False, sort_keys=True))
    payload["supervision_boundary"] = np.asarray(_supervision(condition))
    payload["mae_embedding_seed"] = np.asarray(240800, dtype=np.int64)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)


def _read_reference(root: Path, protocol: str, seed: int) -> dict[str, Any]:
    path = _reference_metric(root, protocol, seed)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "ok" or value.get("route_id") != REFERENCE_ROUTE:
        raise ValueError(f"invalid archived B0 reference: {path}")
    if value.get("condition_id") != "window_attention_regression_full_mean":
        raise ValueError(f"wrong archived B0 condition: {path}")
    return value


def _patch_metrics(path: Path, *, condition: str, reference_bag: Path, mae_paths: dict[str, Path]) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    value.update({
        "route_id": _candidate_route(condition),
        "candidate_condition": condition,
        "reference_route": REFERENCE_ROUTE,
        "reference_bag": str(reference_bag),
        "mae_token_paths": {name: str(value) for name, value in mae_paths.items()},
        "mae_embedding_seed": 240800,
        "supervision_boundary": _supervision(condition),
    })
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return value


def _table(output: dict[str, Any], path: Path) -> None:
    lines = ["# A1+MT11 baseline MAE replacement results", "", f"Test raw Pearson r; each cell is mean ± sample SD over {len(output['seeds'])} downstream seeds.", ""]
    for protocol in output["protocols"]:
        conditions = output["table_conditions"]
        headings = ["B0 A1+MT11" if condition == "B0" else condition for condition in conditions]
        lines.extend([f"## {protocol}", "", "| emotion | " + " | ".join(headings) + " |", "| --- | " + " | ".join(["---:"] * len(conditions)) + " |"])
        rows = output["summary"][protocol]
        for label in LABEL_NAMES:
            lines.append("| " + label + " | " + " | ".join(rows[condition][label] for condition in conditions) + " |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--mae-root", type=Path, default=MAE_ROOT)
    parser.add_argument("--video-mae-root", type=Path)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--protocols", default="cross_day,within_subject_day")
    parser.add_argument("--seeds", default="240800,240801,240802")
    parser.add_argument("--conditions", default="E1,W1")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--modality-dropout-prob", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument('--video-common-mask',type=Path)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.torch_threads))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but CUDA is unavailable")
    protocols, selected = _csv(args.protocols), _csv(args.conditions)
    seeds = tuple(int(value) for value in _csv(args.seeds))
    if set(protocols) - {"cross_day", "within_subject_day"} or set(selected) - {"E1", "W1", "V1", "M2", "M3", "M4", "M5-F",'E_POOL','V_ROI','V_FULL_MATCH','B0_VIDEO_COMMON'} or not seeds:
        raise ValueError("unsupported protocol, MAE replacement condition, or empty seeds")
    if set(selected).intersection(('V_ROI','V_FULL_MATCH','B0_VIDEO_COMMON')) and not args.video_common_mask:raise ValueError('round1 video routes require common mask')
    index_rows = load_jsonl(args.root / "index/eeg_aligned_window_index.jsonl")
    all_results: list[dict[str, Any]] = []
    for protocol in protocols:
        reference_bag = _reference_bag(args.reference_root, protocol)
        if not reference_bag.is_file():
            raise FileNotFoundError(reference_bag)
        reference_values = {seed: _read_reference(args.reference_root, protocol, seed) for seed in seeds}
        for condition in selected:
            mae_paths = {
                modality: _mae_path(args.video_mae_root if modality == "video" and args.video_mae_root else args.mae_root, protocol, modality)
                for modality, *_ in _replacements(condition)
            }
            for mae in mae_paths.values():
                if not mae.is_file():
                    raise FileNotFoundError(mae)
            bag_path = _replacement_bag(args.out_root, protocol, condition)
            _write_replaced_bag(reference=reference_bag, mae_paths=mae_paths, destination=bag_path, condition=condition,video_common_mask=args.video_common_mask)
            dataset = load_bag_dataset(bag_path)
            targets = event_targets(dataset, index_rows)
            if dataset.tokens.shape != (1253, 23, 4, 256) or targets.shape != (1253, 11):
                raise ValueError(f"invalid candidate bag contract: {bag_path}")
            if dataset.modality_mask[:, :, 3].any():
                raise ValueError("audio must remain disabled")
            model_id, policy = conditions()["window_attention_regression_full_mean"]
            for seed in seeds:
                out_dir = args.out_root / "runs" / protocol / condition / f"seed_{seed}"
                if (out_dir/'metrics.json').exists():
                    result=json.loads((out_dir/'metrics.json').read_text())
                    if result.get('status')!='ok' or result.get('candidate_condition')!=condition:raise ValueError('invalid retained cell')
                    all_results.append({'protocol':protocol,'condition':condition,'seed':seed,'metrics':result})
                    print(f'retaining protocol={protocol} condition={condition} seed={seed}',flush=True)
                    continue
                print(f"starting protocol={protocol} condition={condition} seed={seed}", flush=True)
                run_condition(
                    dataset=dataset, targets=targets, protocol=protocol,
                    condition_id="window_attention_regression_full_mean", model_id=model_id,
                    temporal_policy=policy, seed=seed, out_dir=out_dir, epochs=args.epochs,
                    batch_size=args.batch_size, hidden_dim=args.hidden_dim, learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay, dropout=args.dropout, patience=args.patience,
                    modality_dropout_prob=args.modality_dropout_prob, device=args.device,
                )
                result = _patch_metrics(out_dir / "metrics.json", condition=condition, reference_bag=reference_bag, mae_paths=mae_paths)
                all_results.append({"protocol": protocol, "condition": condition, "seed": seed, "metrics": result})
                print(f"completed protocol={protocol} condition={condition} seed={seed}", flush=True)
        for seed, metric in reference_values.items():
            all_results.append({"protocol": protocol, "condition": "B0", "seed": seed, "metrics": metric})
    table_conditions = ("B0",) + selected
    summary: dict[str, dict[str, dict[str, str]]] = {}
    for protocol in protocols:
        summary[protocol] = {}
        for condition in table_conditions:
            rows = [r["metrics"] for r in all_results if r["protocol"] == protocol and r["condition"] == condition]
            if not rows:
                continue
            summary[protocol][condition] = {}
            for label in LABEL_NAMES:
                values = np.asarray([row["test"]["per_label"][label]["raw_r"] for row in rows], dtype=float)
                summary[protocol][condition][label] = f"{values.mean():.4f} ± {values.std(ddof=1):.4f}"
    output = {"reference_route": REFERENCE_ROUTE, "protocols": list(protocols), "seeds": list(seeds), "table_conditions": list(table_conditions), "results": all_results, "summary": summary}
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    _table(output, args.out_root / "raw_r_tables.md")
    print(f"run_count={len(all_results)}", flush=True)
    print(f"out={args.out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
