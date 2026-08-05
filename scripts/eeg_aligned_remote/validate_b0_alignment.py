from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


LABEL_NAMES = [
    "inspired",
    "alert",
    "determined",
    "attentive",
    "active",
    "hostile",
    "nervous",
    "upset",
    "afraid",
    "ashamed",
    "fatigue",
]

DEFAULT_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_EEG_ROOT = Path("/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new")
DEFAULT_SPLITS_ROOT = Path("/vePFS-0x0d/DailyEEG/splits_new")
DEFAULT_EMB_ROOT = Path("/vePFS-0x0d/DailyEEG_multimodal/embeddings")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate B0 EEG-aligned multimodal inputs.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--eeg-root", type=Path, default=DEFAULT_EEG_ROOT)
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument("--emb-root", type=Path, default=DEFAULT_EMB_ROOT)
    parser.add_argument("--require-eeg-embedding", action="store_true")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args()

    root = args.root
    json_out = args.json_out or root / "reports/b0_alignment_preflight.json"
    md_out = args.md_out or root / "reports/b0_alignment_preflight.md"

    index_path = root / "index/eeg_aligned_window_index.jsonl"
    rows = _load_index(index_path)
    canonical = _canonical_arrays(rows)
    expected_n = len(rows)
    errors: list[str] = []
    report: dict[str, Any] = {
        "index_path": str(index_path),
        "row_count": expected_n,
        "label_names": LABEL_NAMES,
        "checks": {},
        "branches": {},
        "splits": {},
        "errors": errors,
    }

    _check_eeg_arrays(args.eeg_root, canonical, report, errors)

    branches = {
        "video_B0": {
            "path": args.emb_root / "video/video_B0_2xroi_eeg23win_embeddings.npz",
            "emb_key": "video_emb",
            "mask_key": "video_mask",
            "modality_index": 2,
        },
        "audio": {
            "path": args.emb_root / "audio/audio_opensmile_eeg23win_embeddings.npz",
            "emb_key": "audio_emb",
            "mask_key": "audio_mask",
            "modality_index": 3,
        },
        "wear_physio": {
            "path": args.emb_root / "wear/wear_physio_preprocessed_eeg23win_embeddings.npz",
            "emb_key": "wear_emb",
            "mask_key": "wear_mask",
            "modality_index": 1,
        },
        "wear_deep": {
            "path": args.emb_root / "wear/wear_deep_sequence_preprocessed_eeg23win_embeddings.npz",
            "emb_key": "wear_emb",
            "mask_key": "wear_mask",
            "modality_index": 1,
        },
    }
    eeg_path = args.emb_root / "eeg/eeg_statfft_eeg23win_embeddings.npz"
    if eeg_path.exists() or args.require_eeg_embedding:
        branches["eeg"] = {
            "path": eeg_path,
            "emb_key": "eeg_emb",
            "mask_key": "eeg_mask",
            "modality_index": 0,
        }

    for name, spec in branches.items():
        if not spec["path"].exists():
            errors.append(f"{name} missing: {spec['path']}")
            continue
        report["branches"][name] = _check_branch(name, spec, canonical, errors)

    report["splits"] = _check_splits(args.splits_root, expected_n, errors)
    report["ok"] = not errors

    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_out.write_text(_markdown_report(report), encoding="utf-8")
    print(f"ok={report['ok']}")
    print(f"json_out={json_out}")
    print(f"md_out={md_out}")
    if errors:
        for error in errors:
            print(f"ERROR {error}")
        return 1
    return 0


def _load_index(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != 28819:
        raise ValueError(f"expected 28819 index rows, got {len(rows)}")
    for idx, row in enumerate(rows):
        expected_sample = f"eeg_{idx:06d}"
        if row.get("sample_id") != expected_sample:
            raise ValueError(f"index row {idx} sample_id mismatch: {row.get('sample_id')} != {expected_sample}")
        if int(row.get("eeg_sample_index", -1)) != idx:
            raise ValueError(f"index row {idx} eeg_sample_index mismatch")
    return rows


def _canonical_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    labels = np.asarray([row["labels"] for row in rows], dtype=np.float32)
    return {
        "sample_id": np.asarray([row["sample_id"] for row in rows], dtype=str),
        "eeg_sample_index": np.arange(len(rows), dtype=np.int64),
        "subject_id": np.asarray([_norm_subject(row["subject_id"]) for row in rows], dtype=str),
        "day_id": np.asarray([int(row["day_id"]) for row in rows], dtype=np.int64),
        "event_id": np.asarray([str(row["event_id"]) for row in rows], dtype=str),
        "event_window_id": np.asarray([int(row["event_window_id"]) for row in rows], dtype=np.int64),
        "labels": labels,
    }


def _check_eeg_arrays(eeg_root: Path, canonical: dict[str, np.ndarray], report: dict[str, Any], errors: list[str]) -> None:
    expected_n = len(canonical["sample_id"])
    checks: dict[str, Any] = {}
    for name in ("X", "y", "sub", "d", "ts"):
        path = eeg_root / f"{name}.npy"
        if not path.exists():
            errors.append(f"missing EEG array {path}")
            continue
        arr = np.load(path, mmap_mode="r", allow_pickle=True)
        checks[name] = {"path": str(path), "shape": list(arr.shape), "dtype": str(arr.dtype)}
        if arr.shape[0] != expected_n:
            errors.append(f"{path} row count {arr.shape[0]} != {expected_n}")
        if name == "y" and not np.allclose(np.asarray(arr), canonical["labels"]):
            errors.append("y.npy labels do not match index labels")
        if name == "sub":
            subjects = np.asarray([_norm_subject(v) for v in np.asarray(arr).tolist()], dtype=str)
            if not np.array_equal(subjects, canonical["subject_id"]):
                errors.append("sub.npy subject ids do not match index subject_id")
        if name == "d" and not np.array_equal(np.asarray(arr, dtype=np.int64), canonical["day_id"]):
            errors.append("d.npy day ids do not match index day_id")
    report["checks"]["eeg_arrays"] = checks


def _check_branch(
    name: str,
    spec: dict[str, Any],
    canonical: dict[str, np.ndarray],
    errors: list[str],
) -> dict[str, Any]:
    path = Path(spec["path"])
    emb_key = str(spec["emb_key"])
    mask_key = str(spec["mask_key"])
    modality_index = int(spec["modality_index"])
    expected_n = len(canonical["sample_id"])
    branch_report: dict[str, Any] = {
        "path": str(path),
        "sha256": _sha256_file(path),
    }
    with np.load(path, allow_pickle=True) as loaded:
        branch_report["keys"] = loaded.files
        sample_id = loaded["sample_id"].astype(str)
        branch_report["row_count"] = int(len(sample_id))
        if len(sample_id) != expected_n:
            errors.append(f"{name} row count {len(sample_id)} != {expected_n}")
        if not np.array_equal(sample_id, canonical["sample_id"]):
            errors.append(f"{name} sample_id order does not match canonical index")
        if "eeg_sample_index" in loaded.files:
            eeg_idx = loaded["eeg_sample_index"].astype(np.int64)
            if not np.array_equal(eeg_idx, canonical["eeg_sample_index"]):
                errors.append(f"{name} eeg_sample_index is not 0..{expected_n - 1}")
        if "subject_id" in loaded.files:
            subjects = np.asarray([_norm_subject(v) for v in loaded["subject_id"].tolist()], dtype=str)
            if not np.array_equal(subjects, canonical["subject_id"]):
                errors.append(f"{name} subject_id does not match canonical index")
        if emb_key not in loaded.files:
            errors.append(f"{name} missing {emb_key}")
        else:
            emb = loaded[emb_key]
            branch_report["embedding_shape"] = list(emb.shape)
            branch_report["embedding_dtype"] = str(emb.dtype)
            if emb.shape != (expected_n, 256):
                errors.append(f"{name} {emb_key} shape {emb.shape} != ({expected_n}, 256)")
            if not np.isfinite(emb).all():
                errors.append(f"{name} {emb_key} contains NaN or infinite values")
        mask = _load_mask(loaded, mask_key, modality_index)
        if mask is None:
            errors.append(f"{name} missing {mask_key} and compatible modality_mask")
            mask = np.zeros(expected_n, dtype=np.int8)
        branch_report["mask_sum"] = int(mask.sum())
        branch_report["mask_shape"] = list(mask.shape)
        if mask.shape != (expected_n,):
            errors.append(f"{name} mask shape {mask.shape} != ({expected_n},)")
        if not set(np.unique(mask).astype(int).tolist()).issubset({0, 1}):
            errors.append(f"{name} mask contains values outside 0/1")
        if "modality_mask" in loaded.files and mask_key in loaded.files:
            modality_mask = loaded["modality_mask"].astype(np.int8)
            if modality_mask.ndim == 2 and modality_mask.shape[1] > modality_index:
                if not np.array_equal(modality_mask[:, modality_index], loaded[mask_key].astype(np.int8)):
                    errors.append(f"{name} {mask_key} differs from modality_mask column {modality_index}")
        if "labels" in loaded.files:
            labels = _labels_to_matrix(loaded["labels"])
            if labels.shape == canonical["labels"].shape and not np.allclose(labels, canonical["labels"]):
                errors.append(f"{name} labels do not match canonical y/index labels")
            branch_report["labels_shape"] = list(labels.shape)
    return branch_report


def _load_mask(loaded: Any, mask_key: str, modality_index: int) -> np.ndarray | None:
    if mask_key in loaded.files:
        return loaded[mask_key].astype(np.int8)
    if "modality_mask" in loaded.files:
        mask = loaded["modality_mask"].astype(np.int8)
        if mask.ndim == 2 and mask.shape[1] > modality_index:
            return mask[:, modality_index].astype(np.int8)
    return None


def _labels_to_matrix(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    rows = []
    for raw in arr.tolist():
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if isinstance(raw, str):
            parsed = json.loads(raw)
        else:
            parsed = raw
        if isinstance(parsed, dict):
            rows.append([float(parsed[name]) for name in LABEL_NAMES])
        else:
            rows.append([float(value) for value in parsed])
    return np.asarray(rows, dtype=np.float32)


def _check_splits(path: Path, expected_n: int, errors: list[str]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for protocol_dir in sorted(p for p in path.iterdir() if p.is_dir()):
        protocol = protocol_dir.name
        split_report: dict[str, Any] = {}
        split_sets: dict[str, set[int]] = {}
        for split_name in ("pretrain", "finetune", "val", "test"):
            split_path = protocol_dir / f"{split_name}.json"
            if not split_path.exists():
                errors.append(f"{protocol} missing {split_name}.json")
                continue
            values = json.loads(split_path.read_text(encoding="utf-8"))
            indices = [int(value) for value in values]
            split_sets[split_name] = set(indices)
            split_report[split_name] = {
                "count": len(indices),
                "min": min(indices) if indices else None,
                "max": max(indices) if indices else None,
            }
            if len(indices) != len(split_sets[split_name]):
                errors.append(f"{protocol}/{split_name} contains duplicate indices")
            if indices and (min(indices) < 0 or max(indices) >= expected_n):
                errors.append(f"{protocol}/{split_name} has out-of-range indices")
        for left_name, left in split_sets.items():
            for right_name, right in split_sets.items():
                if left_name >= right_name:
                    continue
                overlap = left & right
                if overlap:
                    errors.append(f"{protocol} splits overlap: {left_name}/{right_name} overlap_count={len(overlap)}")
        union = set().union(*split_sets.values()) if split_sets else set()
        split_report["union_count"] = len(union)
        if len(union) != expected_n:
            errors.append(f"{protocol} split union {len(union)} != {expected_n}")
        info_path = protocol_dir / "split_info.json"
        if info_path.exists():
            split_report["split_info"] = json.loads(info_path.read_text(encoding="utf-8"))
        report[protocol] = split_report
    return report


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _norm_subject(value: Any) -> str:
    text = str(value)
    if text.startswith("sub-"):
        return text
    try:
        return f"sub-{int(float(text)):02d}"
    except ValueError:
        return text


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# B0 Alignment Preflight",
        "",
        f"ok: `{report['ok']}`",
        f"row_count: `{report['row_count']}`",
        "",
        "## Branches",
        "",
        "| branch | rows | emb_shape | mask_sum | sha256 |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for name, branch in report["branches"].items():
        lines.append(
            "| {name} | {rows} | {shape} | {mask_sum} | `{sha}` |".format(
                name=name,
                rows=branch.get("row_count"),
                shape=branch.get("embedding_shape"),
                mask_sum=branch.get("mask_sum"),
                sha=str(branch.get("sha256", ""))[:16],
            )
        )
    lines.extend(["", "## Splits", "", "| protocol | pretrain | finetune | val | test | union |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for protocol, split in report["splits"].items():
        lines.append(
            "| {protocol} | {pretrain} | {finetune} | {val} | {test} | {union} |".format(
                protocol=protocol,
                pretrain=split.get("pretrain", {}).get("count"),
                finetune=split.get("finetune", {}).get("count"),
                val=split.get("val", {}).get("count"),
                test=split.get("test", {}).get("count"),
                union=split.get("union_count"),
            )
        )
    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
