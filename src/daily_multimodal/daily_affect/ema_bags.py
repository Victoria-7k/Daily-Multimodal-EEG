from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .npz_compat import install_numpy_core_pickle_aliases


LABEL_NAMES = (
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
)
MODALITY_ORDER = ("eeg", "wear", "video", "audio")
WINDOWS_PER_EVENT = 23
TRAIN_LEAVES = frozenset(("pretrain", "finetune"))


@dataclass(frozen=True)
class Branch:
    name: str
    modality: str
    filename: str
    emb_key: str
    mask_key: str
    modality_index: int
    supervision: str = "label_free_or_fixed"


BRANCHES = {
    "eeg": Branch("eeg", "eeg", "eeg/eeg_eegpt_eeg23win_embeddings.npz", "eeg_emb", "eeg_mask", 0),
    "eeg_eegpt_frozen_v1": Branch(
        "eeg_eegpt_frozen_v1",
        "eeg",
        "{eeg_token_root}/{protocol}/eegpt_frozen_v1/seed_{eeg_seed}.npz",
        "eeg_emb",
        "eeg_mask",
        0,
    ),
    "eeg_eegpt_partial_ft_v1": Branch(
        "eeg_eegpt_partial_ft_v1",
        "eeg",
        "{eeg_token_root}/{protocol}/eegpt_partial_ft_v1/seed_{eeg_seed}.npz",
        "eeg_emb",
        "eeg_mask",
        0,
        "fatigue_supervised_control",
    ),
    "eeg_cbramod_frozen_v1": Branch(
        "eeg_cbramod_frozen_v1",
        "eeg",
        "{eeg_token_root}/{protocol}/cbramod_frozen_v1/seed_{eeg_seed}.npz",
        "eeg_emb",
        "eeg_mask",
        0,
        "fatigue_supervised_control",
    ),
    "eeg_cbramod_partial_ft_v1": Branch(
        "eeg_cbramod_partial_ft_v1",
        "eeg",
        "{eeg_token_root}/{protocol}/cbramod_partial_ft_v1/seed_{eeg_seed}.npz",
        "eeg_emb",
        "eeg_mask",
        0,
        "fatigue_supervised_control",
    ),
    "eeg_de_5band_1s_avg_v1": Branch(
        "eeg_de_5band_1s_avg_v1",
        "eeg",
        "{eeg_token_root}/{protocol}/eeg_de_5band_1s_avg_v1/seed_{eeg_seed}.npz",
        "eeg_emb",
        "eeg_mask",
        0,
        "fatigue_supervised_control",
    ),
    "wear_physio": Branch("wear_physio", "wear", "wear/wear_physio_preprocessed_eeg23win_embeddings.npz", "wear_emb", "wear_mask", 1),
    "wear_deep": Branch("wear_deep", "wear", "wear/wear_deep_sequence_preprocessed_eeg23win_embeddings.npz", "wear_emb", "wear_mask", 1),
    "wear_moment_frozen_v1": Branch(
        "wear_moment_frozen_v1",
        "wear",
        "wear_tokens/{protocol}/wear_moment_frozen_v1/seed_{wear_seed}.npz",
        "wear_emb",
        "wear_mask",
        1,
        "fatigue_supervised_control",
    ),
    "wear_moment_partial_ft_v1": Branch(
        "wear_moment_partial_ft_v1",
        "wear",
        "wear_tokens/{protocol}/wear_moment_partial_ft_v1/seed_{wear_seed}.npz",
        "wear_emb",
        "wear_mask",
        1,
        "fatigue_supervised_control",
    ),
    "video_B0": Branch("video_B0", "video", "video/video_B0_2xroi_eeg23win_embeddings.npz", "video_emb", "video_mask", 2),
    "video_A1": Branch("video_A1", "video", "video/video_A1_2xroi_eeg23win_embeddings.npz", "video_emb", "video_mask", 2),
    "video_A2": Branch("video_A2", "video", "video/video_A2_2xroi_eeg23win_embeddings.npz", "video_emb", "video_mask", 2),
    "audio": Branch("audio", "audio", "audio/audio_opensmile_eeg23win_embeddings.npz", "audio_emb", "audio_mask", 3),
}

for _label in LABEL_NAMES:
    _name = f"eeg_eegpt_partial_ft_single_{_label}_v1"
    BRANCHES[_name] = Branch(
        _name, "eeg",
        f"{{eeg_token_root}}/single_task/{{protocol}}/{_label}/seed_{{eeg_seed}}.npz",
        "eeg_emb", "eeg_mask", 0, f"{_label}_supervised_partial_ft",
    )


EXPERIMENT_BRANCHES = {
    "B0_Wphysio_full": ("eeg", "wear_physio", "video_B0", "audio"),
    "B0_Wphysio_no_audio": ("eeg", "wear_physio", "video_B0"),
    "B0_Wphysio_no_video": ("eeg", "wear_physio", "audio"),
    "B0_Wphysio_bio_only": ("eeg", "wear_physio"),
    "B0_Wdeep_full": ("eeg", "wear_deep", "video_B0", "audio"),
    "B0_Wdeep_no_audio": ("eeg", "wear_deep", "video_B0"),
    "B0_Wdeep_no_video": ("eeg", "wear_deep", "audio"),
    "B0_Wdeep_bio_only": ("eeg", "wear_deep"),
    "A1_Wphysio_full": ("eeg", "wear_physio", "video_A1", "audio"),
    "A1_Wphysio_no_audio": ("eeg", "wear_physio", "video_A1"),
    "A1_Wdeep_full": ("eeg", "wear_deep", "video_A1", "audio"),
    "A1_Wdeep_no_audio": ("eeg", "wear_deep", "video_A1"),
    "A2_Wphysio_full": ("eeg", "wear_physio", "video_A2", "audio"),
    "A2_Wphysio_no_audio": ("eeg", "wear_physio", "video_A2"),
    "A2_Wdeep_full": ("eeg", "wear_deep", "video_A2", "audio"),
    "A2_Wdeep_no_audio": ("eeg", "wear_deep", "video_A2"),
    "A1_Wmoment_frozen_full": ("eeg", "wear_moment_frozen_v1", "video_A1", "audio"),
    "A1_Wmoment_ft_full": ("eeg", "wear_moment_partial_ft_v1", "video_A1", "audio"),
    "A1_Wmoment_frozen_no_audio": ("eeg", "wear_moment_frozen_v1", "video_A1"),
    "A1_Wmoment_ft_no_audio": ("eeg", "wear_moment_partial_ft_v1", "video_A1"),
    "B0_Wmoment_frozen_full": ("eeg", "wear_moment_frozen_v1", "video_B0", "audio"),
    "B0_Wmoment_frozen_no_audio": ("eeg", "wear_moment_frozen_v1", "video_B0"),
    "B0_Wmoment_ft_full": ("eeg", "wear_moment_partial_ft_v1", "video_B0", "audio"),
    "B0_Wmoment_ft_no_audio": ("eeg", "wear_moment_partial_ft_v1", "video_B0"),
    "A2_Wmoment_frozen_full": ("eeg", "wear_moment_frozen_v1", "video_A2", "audio"),
    "A2_Wmoment_frozen_no_audio": ("eeg", "wear_moment_frozen_v1", "video_A2"),
    "A2_Wmoment_ft_full": ("eeg", "wear_moment_partial_ft_v1", "video_A2", "audio"),
    "A2_Wmoment_ft_no_audio": ("eeg", "wear_moment_partial_ft_v1", "video_A2"),
}


def resolve_route_branches(route_id: str, *, eeg_branch: str = "eeg", explicit_branches: tuple[str, ...] | None = None) -> tuple[str, ...]:
    names = explicit_branches if explicit_branches is not None else EXPERIMENT_BRANCHES.get(route_id)
    if names is None:
        raise ValueError(f"unknown route_id {route_id!r}; pass --branches for a custom route")
    if eeg_branch not in BRANCHES or BRANCHES[eeg_branch].modality != "eeg":
        raise ValueError(f"unsupported EEG branch: {eeg_branch}")
    resolved = tuple(eeg_branch if name == "eeg" else name for name in names)
    seen: dict[str, str] = {}
    for name in resolved:
        if name not in BRANCHES:
            raise ValueError(f"unsupported branch: {name}")
        modality = BRANCHES[name].modality
        if modality in seen:
            raise ValueError(f"route contains two {modality} branches: {seen[modality]} and {name}")
        seen[modality] = name
    return resolved


def effective_route_id(route_id: str, *, eeg_branch: str = "eeg", explicit_branches: tuple[str, ...] | None = None) -> str:
    if explicit_branches is not None:
        return route_id
    return route_id if eeg_branch == "eeg" else f"{route_id}__{eeg_branch}"


def build_daily_affect_bags(
    *,
    index_path: Path,
    splits_root: Path,
    embeddings_root: Path,
    protocol: str,
    route_id: str,
    out_dir: Path,
    target_label: str = "fatigue",
    eeg_branch: str = "eeg",
    eeg_seed: int = 240800,
    wear_seed: int | None = None,
    eeg_token_root: str = "eeg_encoder_256d_tokens",
    explicit_branches: tuple[str, ...] | None = None,
    max_events: int | None = None,
) -> dict[str, Any]:
    rows = load_jsonl(index_path)
    if not rows:
        raise ValueError(f"{index_path} contains no rows")
    sample_id = np.asarray([str(row["sample_id"]) for row in rows], dtype=str)
    branches = resolve_route_branches(route_id, eeg_branch=eeg_branch, explicit_branches=explicit_branches)
    route_name = effective_route_id(route_id, eeg_branch=eeg_branch, explicit_branches=explicit_branches)
    branch_data = load_branch_arrays(
        embeddings_root,
        sample_id,
        protocol=protocol,
        branches=branches,
        eeg_seed=eeg_seed,
        wear_seed=eeg_seed if wear_seed is None else wear_seed,
        eeg_token_root=eeg_token_root,
    )
    groups = grouped_event_indices(rows)
    if max_events is not None:
        groups = groups[: max(0, int(max_events))]
    split_by_window = load_window_split(splits_root / protocol, len(rows))
    bag_payload = assemble_bags(
        rows,
        sample_id,
        groups,
        split_by_window,
        branch_data,
        branches=branches,
        route_id=route_name,
        target_label=target_label,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    bag_path = out_dir / "ema_bags.npz"
    manifest_path = out_dir / "ema_bag_manifest.jsonl"
    report_path = out_dir / "bag_build_report.json"
    save_bag_npz(bag_payload, bag_path)
    write_manifest(bag_payload, manifest_path)
    report = build_report(
        bag_payload,
        protocol=protocol,
        index_path=index_path,
        splits_root=splits_root,
        embeddings_root=embeddings_root,
        branches=branches,
        branch_data=branch_data,
        route_id=route_name,
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "protocol": protocol,
        "route_id": route_name,
        "branches": list(branches),
        "bag_path": str(bag_path),
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
        **report,
    }


def assemble_bags(
    rows: list[dict[str, Any]],
    sample_id: np.ndarray,
    groups: list[np.ndarray],
    split_by_window: np.ndarray,
    branch_data: dict[str, dict[str, Any]],
    *,
    branches: tuple[str, ...],
    route_id: str,
    target_label: str,
) -> dict[str, Any]:
    n_events = len(groups)
    tokens = np.zeros((n_events, WINDOWS_PER_EVENT, len(MODALITY_ORDER), 256), dtype=np.float32)
    modality_mask = np.zeros((n_events, WINDOWS_PER_EVENT, len(MODALITY_ORDER)), dtype=bool)
    labels = np.zeros((n_events,), dtype=np.float32)
    event_ids = np.empty((n_events,), dtype=object)
    subject_ids = np.empty((n_events,), dtype=object)
    day_ids = np.empty((n_events,), dtype=object)
    sample_id_matrix = np.empty((n_events, WINDOWS_PER_EVENT), dtype=object)
    split = np.empty((n_events,), dtype=object)
    window_leaf_split_counts_json = np.empty((n_events,), dtype=object)
    mixed_train_leaf_event_count = 0
    projected_boundary_event_count = 0
    source_npz = {name: branch_data[name]["path"] for name in branches}
    branch_by_modality = {BRANCHES[name].modality: name for name in branches}
    for event_index, indices in enumerate(groups):
        if len(indices) != WINDOWS_PER_EVENT:
            raise ValueError(f"event group {event_index} has {len(indices)} windows, expected {WINDOWS_PER_EVENT}")
        ordered = sorted(indices.tolist(), key=lambda idx: int(rows[idx]["event_window_id"]))
        window_ids = [int(rows[idx]["event_window_id"]) for idx in ordered]
        if window_ids != list(range(WINDOWS_PER_EVENT)):
            raise ValueError(f"event {rows[ordered[0]].get('event_id')} has event_window_id {window_ids}")
        leaf_values = [str(split_by_window[idx]) for idx in ordered]
        leaf_splits = set(leaf_values)
        event_split, projection_kind = resolve_event_split(leaf_values, str(rows[ordered[0]].get("event_id")))
        if projection_kind == "train_leaf_merge":
            mixed_train_leaf_event_count += 1
        elif projection_kind == "window_majority":
            projected_boundary_event_count += 1
        for modality_index, modality in enumerate(MODALITY_ORDER):
            branch_name = branch_by_modality.get(modality)
            if branch_name is None:
                continue
            data = branch_data[branch_name]
            tokens[event_index, :, modality_index] = data["embedding"][ordered]
            modality_mask[event_index, :, modality_index] = data["mask"][ordered]
        first = rows[ordered[0]]
        label = target_value(first, target_label)
        for idx in ordered[1:]:
            other = target_value(rows[idx], target_label)
            if abs(float(other) - float(label)) > 1e-6:
                raise ValueError(f"event {first.get('event_id')} has inconsistent {target_label} labels")
        labels[event_index] = float(label)
        event_ids[event_index] = str(first.get("event_id", f"event_{event_index:06d}"))
        subject_ids[event_index] = norm_subject(first.get("subject_id", ""))
        day_ids[event_index] = day_id(first)
        sample_id_matrix[event_index] = sample_id[ordered]
        split[event_index] = event_split
        window_leaf_split_counts_json[event_index] = json.dumps(dict(sorted(Counter(leaf_values).items())), ensure_ascii=False)
    label_zero = np.clip(np.rint(labels).astype(np.int64) - 1, 0, 4)
    split_indices = split_indices_from_labels(split)
    return {
        "tokens": tokens,
        "modality_mask": modality_mask.astype(np.int8),
        "label": labels,
        "label_zero_based": label_zero,
        "event_id": event_ids,
        "subject_id": subject_ids,
        "day_id": day_ids,
        "sample_id_matrix": sample_id_matrix,
        "event_window_id": np.arange(WINDOWS_PER_EVENT, dtype=np.int64),
        "split": split,
        "window_leaf_split_counts_json": window_leaf_split_counts_json,
        "mixed_train_leaf_event_count": int(mixed_train_leaf_event_count),
        "projected_boundary_event_count": int(projected_boundary_event_count),
        "event_split_policy": "single_leaf_or_train_leaf_merge_or_window_majority",
        "route_id": route_id,
        "target_label": target_label,
        "source_npz_json": json.dumps(source_npz, ensure_ascii=False, sort_keys=True),
        "supervision_boundary": supervision_boundary(branches),
        **split_indices,
    }


def save_bag_npz(payload: dict[str, Any], path: Path) -> None:
    np.savez_compressed(
        path,
        tokens=payload["tokens"],
        modality_mask=payload["modality_mask"],
        label=payload["label"],
        label_zero_based=payload["label_zero_based"],
        event_id=payload["event_id"],
        subject_id=payload["subject_id"],
        day_id=payload["day_id"],
        sample_id_matrix=payload["sample_id_matrix"],
        event_window_id=payload["event_window_id"],
        split=payload["split"],
        window_leaf_split_counts_json=payload["window_leaf_split_counts_json"],
        mixed_train_leaf_event_count=np.asarray(int(payload.get("mixed_train_leaf_event_count", 0))),
        projected_boundary_event_count=np.asarray(int(payload.get("projected_boundary_event_count", 0))),
        event_split_policy=np.asarray(str(payload.get("event_split_policy", ""))),
        route_id=np.asarray(str(payload["route_id"])),
        target_label=np.asarray(str(payload["target_label"])),
        source_npz_json=np.asarray(str(payload["source_npz_json"])),
        supervision_boundary=np.asarray(str(payload["supervision_boundary"])),
        pretrain_index=payload["pretrain_index"],
        finetune_index=payload["finetune_index"],
        train_index=payload["train_index"],
        val_index=payload["val_index"],
        test_index=payload["test_index"],
    )


def write_manifest(payload: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for idx in range(payload["label"].shape[0]):
            row = {
                "bag_index": int(idx),
                "event_id": str(payload["event_id"][idx]),
                "subject_id": str(payload["subject_id"][idx]),
                "day_id": str(payload["day_id"][idx]),
                "split": str(payload["split"][idx]),
                "window_leaf_split_counts": json.loads(str(payload["window_leaf_split_counts_json"][idx])),
                "label": float(payload["label"][idx]),
                "sample_ids": [str(value) for value in payload["sample_id_matrix"][idx].tolist()],
                "modality_valid_counts": payload["modality_mask"][idx].sum(axis=0).astype(int).tolist(),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_report(
    payload: dict[str, Any],
    *,
    protocol: str,
    index_path: Path,
    splits_root: Path,
    embeddings_root: Path,
    branches: tuple[str, ...],
    branch_data: dict[str, dict[str, Any]],
    route_id: str,
) -> dict[str, Any]:
    labels, counts = np.unique(payload["label_zero_based"] + 1, return_counts=True)
    return {
        "protocol": protocol,
        "route_id": route_id,
        "row_count": int(payload["tokens"].shape[0]),
        "tokens_shape": list(payload["tokens"].shape),
        "modality_mask_shape": list(payload["modality_mask"].shape),
        "windows_per_event": WINDOWS_PER_EVENT,
        "event_window_id": payload["event_window_id"].astype(int).tolist(),
        "label_distribution": {str(int(label)): int(count) for label, count in zip(labels.tolist(), counts.tolist())},
        "split_counts": {
            name: int(len(payload[f"{name}_index"]))
            for name in ("pretrain", "finetune", "train", "val", "test")
        },
        "mixed_train_leaf_event_count": int(payload.get("mixed_train_leaf_event_count", 0)),
        "projected_boundary_event_count": int(payload.get("projected_boundary_event_count", 0)),
        "event_split_policy": str(payload.get("event_split_policy", "")),
        "index_path": str(index_path),
        "splits_root": str(splits_root),
        "embeddings_root": str(embeddings_root),
        "branches": list(branches),
        "branch_report": {
            name: {
                "path": branch_data[name]["path"],
                "mask_sum": int(branch_data[name]["mask_sum"]),
                "modality": BRANCHES[name].modality,
                "supervision": BRANCHES[name].supervision,
            }
            for name in branches
        },
        "supervision_boundary": payload["supervision_boundary"],
        "sample_id_order_verified": True,
    }


def load_branch_arrays(
    embedding_root: Path,
    sample_id: np.ndarray,
    *,
    protocol: str,
    branches: tuple[str, ...],
    eeg_seed: int,
    wear_seed: int,
    eeg_token_root: str,
) -> dict[str, dict[str, Any]]:
    install_numpy_core_pickle_aliases()
    result: dict[str, dict[str, Any]] = {}
    for name in branches:
        branch = BRANCHES[name]
        filename = branch.filename.format(
            protocol=protocol,
            eeg_seed=int(eeg_seed),
            wear_seed=int(wear_seed),
            eeg_token_root=eeg_token_root,
        )
        path = embedding_root / filename
        with np.load(path, allow_pickle=True) as loaded:
            if "sample_id" not in loaded.files:
                raise ValueError(f"{path} missing sample_id")
            loaded_ids = loaded["sample_id"].astype(str)
            if not np.array_equal(loaded_ids, sample_id):
                raise ValueError(f"{name} sample_id order does not match canonical index")
            emb_key = resolve_embedding_key(loaded.files, branch.emb_key)
            embedding = loaded[emb_key].astype(np.float32)
            mask = load_mask(loaded, branch).astype(bool)
            if embedding.shape != (len(sample_id), 256) or mask.shape != (len(sample_id),):
                raise ValueError(f"invalid shape for {name}: {embedding.shape}, {mask.shape}")
            if not np.isfinite(embedding).all():
                raise ValueError(f"{path} contains non-finite values")
        result[name] = {
            "embedding": embedding,
            "mask": mask,
            "path": str(path),
            "mask_sum": int(mask.sum()),
        }
    return result


def resolve_embedding_key(keys: list[str], preferred: str) -> str:
    if preferred in keys:
        return preferred
    if preferred == "video_emb" and "face_emb" in keys:
        return "face_emb"
    raise ValueError(f"embedding key {preferred!r} not found; available keys={keys}")


def load_mask(loaded: Any, branch: Branch) -> np.ndarray:
    if branch.mask_key in loaded.files:
        return loaded[branch.mask_key]
    if branch.modality == "video" and "face_mask" in loaded.files:
        return loaded["face_mask"]
    if "modality_mask" in loaded.files:
        return loaded["modality_mask"][:, branch.modality_index]
    raise ValueError(f"missing mask for branch {branch.name}")


def grouped_event_indices(rows: list[dict[str, Any]]) -> list[np.ndarray]:
    groups: dict[str, list[int]] = {}
    order: list[str] = []
    for index, row in enumerate(rows):
        if "event_window_id" not in row:
            raise ValueError("canonical daily-affect index requires event_window_id")
        event_id = str(row.get("event_id", ""))
        if not event_id:
            raise ValueError(f"row {index} missing event_id")
        if event_id not in groups:
            groups[event_id] = []
            order.append(event_id)
        groups[event_id].append(index)
    result = [np.asarray(groups[event_id], dtype=np.int64) for event_id in order]
    for event_id, indices in zip(order, result):
        if len(indices) != WINDOWS_PER_EVENT:
            raise ValueError(f"event {event_id} has {len(indices)} windows, expected {WINDOWS_PER_EVENT}")
    return result


def load_window_split(protocol_root: Path, n_rows: int) -> np.ndarray:
    split = np.full((n_rows,), "", dtype=object)
    for name in ("pretrain", "finetune", "val", "test"):
        indices = load_indices(protocol_root / f"{name}.json", n_rows)
        if indices.size == 0:
            continue
        occupied = split[indices] != ""
        if bool(occupied.any()):
            raise ValueError(f"{protocol_root} has overlapping split indices for {name}")
        split[indices] = name
    if bool(np.any(split == "")):
        missing = int(np.sum(split == ""))
        raise ValueError(f"{protocol_root} does not assign {missing} windows to pretrain/finetune/val/test")
    return split


def split_indices_from_labels(split: np.ndarray) -> dict[str, np.ndarray]:
    result = {
        name: np.flatnonzero(split == name).astype(np.int64)
        for name in ("pretrain", "finetune", "val", "test")
    }
    result["train"] = np.flatnonzero(np.isin(split, ("pretrain", "finetune", "train"))).astype(np.int64)
    for name in ("train", "val", "test"):
        if len(result[name]) == 0:
            raise ValueError(f"bag-level {name} split is empty")
    return {f"{name}_index": values for name, values in result.items()}


def resolve_event_split(leaf_values: list[str], event_id: str) -> tuple[str, str]:
    leaf_splits = set(leaf_values)
    if "" in leaf_splits:
        raise ValueError(f"event {event_id} has unassigned split leaves: {sorted(leaf_splits)}")
    if len(leaf_splits) == 1:
        return next(iter(leaf_splits)), "single_leaf"
    if leaf_splits.issubset(TRAIN_LEAVES):
        return "train", "train_leaf_merge"
    counts = Counter("train" if value in TRAIN_LEAVES else value for value in leaf_values)
    priority = {"train": 0, "val": 1, "test": 2}
    winner = max(counts.items(), key=lambda item: (item[1], -priority.get(item[0], 99)))[0]
    if winner not in {"train", "val", "test"}:
        raise ValueError(f"event {event_id} has unsupported split leaves: {sorted(leaf_splits)}")
    return winner, "window_majority"


def load_indices(path: Path, n_rows: int) -> np.ndarray:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("indices", value.get("index", value.get("rows")))
    values = np.asarray(value, dtype=np.int64).reshape(-1)
    if values.size and (values.min() < 0 or values.max() >= n_rows):
        raise ValueError(f"{path} contains out-of-range indices")
    return values


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def target_value(row: dict[str, Any], target_label: str) -> float:
    if "label_columns" in row and target_label in row["label_columns"]:
        return float(row["label_columns"][target_label])
    labels = row.get("labels")
    if isinstance(labels, dict):
        return float(labels[target_label])
    label_names = tuple(str(value) for value in row.get("label_names") or LABEL_NAMES)
    if isinstance(labels, list):
        return float(labels[label_names.index(target_label)])
    if target_label in row:
        return float(row[target_label])
    raise ValueError(f"row missing target label {target_label!r}")


def day_id(row: dict[str, Any]) -> str:
    for key in ("day_id", "date", "session_date"):
        if row.get(key):
            return str(row[key])
    onset = str(row.get("absolute_onset_time", ""))
    if len(onset) >= 10:
        return onset[:10]
    session = str(row.get("session_id", ""))
    return session or "unknown_day"


def norm_subject(value: Any) -> str:
    text = str(value)
    if text.startswith("sub-"):
        return text
    try:
        return f"sub-{int(float(text)):02d}"
    except (TypeError, ValueError):
        return text


def supervision_boundary(branches: tuple[str, ...]) -> str:
    controls = [name for name in branches if BRANCHES[name].supervision != "label_free_or_fixed"]
    if controls:
        prefix = "mixed_with_label_supervised_controls:" if any("_single_" in name for name in controls) else "mixed_with_fatigue_supervised_controls:"
        return prefix + ",".join(controls)
    return "label_free_or_fixed_embeddings_only"
