#!/usr/bin/env python3
"""Render compact, seed-aware diagnostic atlases for daily-affect runs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_RUN_ROOT = DEFAULT_ROOT / "outputs/daily_affect_ordinal_routefix_20260906"
MODALITIES = ("EEG", "Wear", "Video", "Audio")
LEGACY_FIGURE_PREFIXES = (
    "confusion_",
    "temporal_kernel_",
    "modality_weights_",
    "probe_reliability_",
)
ATLAS_FIGURE_PREFIXES = (
    "daily_affect_diagnostics_atlas_",
    "daily_affect_probe_reliability_atlas_",
    "daily_affect_confusion_atlas",
)
PROBE_RELIABILITY_METRICS = ("Accuracy", "NLL", "ECE", "RPS")


@dataclass(frozen=True)
class RunDiagnostic:
    protocol: str
    route_id: str
    model_id: str
    experiment_id: str
    normalization: str
    adapter_mode: str
    objective_id: str
    routing_id: str
    seed: int | None
    metrics_path: Path
    qwk: float | None
    modality_weights: np.ndarray | None
    modality_difficulty: np.ndarray | None
    temporal_weights: np.ndarray | None
    confusion: np.ndarray | None
    probe_reliability: np.ndarray | None = None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RUN_ROOT / "figures")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="remove only legacy per-run 77 figures and prior atlas figures under --out-root before rendering",
    )
    args = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("77_plot_daily_affect_diagnostics.py requires matplotlib for atlas rendering") from exc

    runs = load_run_diagnostics(args.run_root)
    if not runs:
        raise SystemExit(f"no usable daily-affect diagnostics found under {args.run_root}")
    groups = aggregate_groups(runs)
    if args.clean:
        remove_generated_figures(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    figure_entries: list[dict[str, Any]] = []
    for protocol in sorted({group["protocol"] for group in groups}):
        protocol_groups = [group for group in groups if group["protocol"] == protocol and group_has_atlas_data(group)]
        if not protocol_groups:
            continue
        path = args.out_root / f"daily_affect_diagnostics_atlas_{protocol}.png"
        plot_protocol_atlas(plt, protocol_groups, path, protocol=protocol)
        figure_entries.append(
            {
                "path": str(path),
                "kind": "diagnostics_atlas",
                "protocol": protocol,
                "group_count": len(protocol_groups),
                "groups": [manifest_group(group) for group in protocol_groups],
            }
        )
        reliability_groups = [group for group in groups if group["protocol"] == protocol and group_has_probe_reliability(group)]
        if reliability_groups:
            reliability_path = args.out_root / f"daily_affect_probe_reliability_atlas_{protocol}.png"
            plot_probe_reliability_atlas(plt, reliability_groups, reliability_path, protocol=protocol)
            figure_entries.append(
                {
                    "path": str(reliability_path),
                    "kind": "probe_reliability_atlas",
                    "protocol": protocol,
                    "metrics": list(PROBE_RELIABILITY_METRICS),
                    "group_count": len(reliability_groups),
                    "groups": [manifest_group(group) for group in reliability_groups],
                }
            )

    selected_confusions = select_confusion_groups(groups)
    if selected_confusions:
        path = args.out_root / "daily_affect_confusion_atlas.png"
        plot_confusion_atlas(plt, selected_confusions, path)
        figure_entries.append(
            {
                "path": str(path),
                "kind": "best_mean_qwk_confusion_atlas",
                "selection": "highest mean test QWK within each protocol and route; visualization only, not a promotion decision",
                "group_count": len(selected_confusions),
                "groups": [manifest_group(group) for group in selected_confusions],
            }
        )

    manifest_path = args.out_root / "daily_affect_figure_manifest.json"
    manifest = {
        "figure_count": len(figure_entries),
        "renderer": "matplotlib",
        "aggregation": {
            "unit": "protocol / route_id / model_id / experiment_id / normalization / adapter_mode / objective_id / routing_id",
            "seed_rule": "compute a test-event mean within each seed, then take an equal-weight mean across seeds",
            "modality_weight_rule": "mean routing weight over test events and 23 event windows",
            "difficulty_rule": "mean event-level probe reliability difficulty over test events; categorical entropy and cumulative ordinal routes remain separate groups",
            "probe_reliability_rule": "per modality Accuracy, NLL, ECE, and RPS use valid test events; values are then equal-weight means across seeds",
            "temporal_weight_rule": "mean temporal kernel over test events, then equal-weight mean across seeds",
            "confusion_rule": "sum seed-level test confusion counts; displayed as row-normalized percentages",
        },
        "figures": figure_entries,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"run_count={len(runs)}")
    print(f"group_count={len(groups)}")
    print(f"figure_count={len(figure_entries)}")
    print(f"manifest={manifest_path}")
    return 0


def load_run_diagnostics(run_root: Path) -> list[RunDiagnostic]:
    records: list[RunDiagnostic] = []
    for metrics_path in sorted(run_root.rglob("metrics.json")):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        protocol = str(metrics.get("protocol", "unknown_protocol"))
        route_id = str(metrics.get("route_id", "unknown_route"))
        model_id = str(metrics.get("model_id", "unknown_model"))
        experiment_id = str(metrics.get("experiment_id", "legacy_unversioned"))
        normalization = str(metrics.get("normalization", "unknown_normalization"))
        adapter_mode = adapter_mode_label(metrics)
        objective_id = str(metrics.get("objective_id", "legacy_unversioned"))
        routing_id = str(metrics.get("routing_id", "legacy_unversioned"))
        seed = as_int_or_none(metrics.get("seed"))
        test_metrics = metrics.get("test", {})
        qwk = as_float_or_none(test_metrics.get("qwk") if isinstance(test_metrics, dict) else None)
        diagnostics_path = resolve_artifact_path(metrics_path, metrics.get("diagnostics_path"), "diagnostics.npz")
        prediction_path = resolve_artifact_path(metrics_path, metrics.get("prediction_path"), "predictions.npz")
        test_context = load_test_prediction_context(prediction_path)
        modality_weights = None
        modality_difficulty = None
        temporal_weights = None
        probe_reliability = None
        if diagnostics_path.is_file():
            try:
                with np.load(diagnostics_path, allow_pickle=False) as diagnostics:
                    modality_weights = summarize_last_dimension(npz_optional(diagnostics, "modality_weights"), expected_size=len(MODALITIES))
                    modality_difficulty = summarize_last_dimension(npz_optional(diagnostics, "modality_difficulty"), expected_size=len(MODALITIES))
                    temporal_weights = summarize_last_dimension(npz_optional(diagnostics, "temporal_weights"), expected_size=None)
                    if test_context is not None:
                        probe_reliability = probe_reliability_metrics(
                            npz_optional(diagnostics, "probe_probs"),
                            npz_optional(diagnostics, "valid_modality_mask"),
                            test_context["test_index"],
                            test_context["truth"],
                        )
            except (OSError, ValueError, KeyError):
                pass
        confusion = confusion_from_context(test_context)
        if all(value is None for value in (modality_weights, modality_difficulty, temporal_weights, confusion, probe_reliability)):
            continue
        records.append(
            RunDiagnostic(
                protocol=protocol,
                route_id=route_id,
                model_id=model_id,
                experiment_id=experiment_id,
                normalization=normalization,
                adapter_mode=adapter_mode,
                objective_id=objective_id,
                routing_id=routing_id,
                seed=seed,
                metrics_path=metrics_path,
                qwk=qwk,
                modality_weights=modality_weights,
                modality_difficulty=modality_difficulty,
                temporal_weights=temporal_weights,
                confusion=confusion,
                probe_reliability=probe_reliability,
            )
        )
    return records


def resolve_artifact_path(metrics_path: Path, configured: Any, fallback_name: str) -> Path:
    candidate = Path(str(configured)) if configured else metrics_path.parent / fallback_name
    if not candidate.is_absolute():
        candidate = metrics_path.parent / candidate
    if candidate.is_file():
        return candidate
    return metrics_path.parent / fallback_name


def npz_optional(archive: Any, key: str) -> np.ndarray | None:
    return archive[key] if key in archive.files else None


def summarize_last_dimension(values: Any, *, expected_size: int | None) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0 or (expected_size is not None and array.shape[-1] != expected_size):
        return None
    return np.nanmean(array.reshape(-1, array.shape[-1]), axis=0)


def load_test_prediction_context(prediction_path: Path) -> dict[str, np.ndarray] | None:
    if not prediction_path.is_file():
        return None
    try:
        with np.load(prediction_path, allow_pickle=False) as predictions:
            test_index = predictions["test_index"].astype(np.int64)
            truth = predictions["label_zero_based"][test_index].astype(np.int64)
            predicted = predictions["test_predicted_class"].astype(np.int64)
    except (OSError, ValueError, KeyError):
        return None
    return {"test_index": test_index, "truth": truth, "predicted": predicted}


def confusion_from_context(context: dict[str, np.ndarray] | None) -> np.ndarray | None:
    if context is None:
        return None
    return confusion_matrix(context["truth"], context["predicted"])


def probe_reliability_metrics(
    probe_probs: Any,
    valid_modality_mask: Any,
    test_index: np.ndarray,
    truth: np.ndarray,
) -> np.ndarray | None:
    if probe_probs is None or valid_modality_mask is None:
        return None
    probs = np.asarray(probe_probs, dtype=np.float64)
    valid = np.asarray(valid_modality_mask, dtype=bool)
    indices = np.asarray(test_index, dtype=np.int64)
    labels = np.asarray(truth, dtype=np.int64)
    if probs.ndim != 3 or probs.shape[1:] != (len(MODALITIES), 5):
        return None
    if valid.shape != probs.shape[:2] or indices.ndim != 1 or labels.shape != indices.shape:
        return None
    if indices.size == 0 or indices.min() < 0 or indices.max() >= probs.shape[0]:
        return None
    selected_probs = probs[indices]
    selected_valid = valid[indices]
    result = np.full((len(MODALITIES), len(PROBE_RELIABILITY_METRICS)), np.nan, dtype=np.float64)
    for modality in range(len(MODALITIES)):
        select = selected_valid[:, modality]
        if not bool(select.any()):
            continue
        current = selected_probs[select, modality]
        current = current / current.sum(axis=1, keepdims=True).clip(min=1e-8)
        current_truth = labels[select]
        predicted = current.argmax(axis=1)
        probability_true = current[np.arange(len(current_truth)), current_truth].clip(min=1e-12, max=1.0)
        cumulative_probs = np.cumsum(current, axis=1)[:, :-1]
        cumulative_truth = (current_truth[:, None] <= np.arange(4)[None, :]).astype(np.float64)
        result[modality] = (
            float(np.mean(predicted == current_truth)),
            float(-np.mean(np.log(probability_true))),
            expected_calibration_error(current, current_truth),
            float(np.mean(np.sum((cumulative_probs - cumulative_truth) ** 2, axis=1) / 4.0)),
        )
    return result


def expected_calibration_error(probabilities: np.ndarray, labels_zero_based: np.ndarray, *, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correctness = (predicted == labels_zero_based).astype(np.float64)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    value = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        select = (confidence >= left) & (confidence < right if right < 1.0 else confidence <= right)
        if bool(select.any()):
            value += float(select.mean()) * abs(float(correctness[select].mean()) - float(confidence[select].mean()))
    return float(value)


def aggregate_groups(records: list[RunDiagnostic]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str, str, str, str, str], list[RunDiagnostic]] = defaultdict(list)
    for record in records:
        buckets[
            (
                record.protocol,
                record.route_id,
                record.model_id,
                record.experiment_id,
                record.normalization,
                record.adapter_mode,
                record.objective_id,
                record.routing_id,
            )
        ].append(record)
    groups: list[dict[str, Any]] = []
    for key in sorted(buckets):
        members = sorted(buckets[key], key=lambda item: (item.seed is None, item.seed, str(item.metrics_path)))
        groups.append(
            {
                "protocol": key[0],
                "route_id": key[1],
                "model_id": key[2],
                "experiment_id": key[3],
                "normalization": key[4],
                "adapter_mode": key[5],
                "objective_id": key[6],
                "routing_id": key[7],
                "seed_count": len(members),
                "seeds": [member.seed for member in members if member.seed is not None],
                "qwk_mean": mean_scalar([member.qwk for member in members]),
                "modality_weights": mean_array([member.modality_weights for member in members]),
                "modality_difficulty": mean_array([member.modality_difficulty for member in members]),
                "probe_reliability": mean_array([member.probe_reliability for member in members]),
                "temporal_weights": mean_array([member.temporal_weights for member in members]),
                "confusion": sum_arrays([member.confusion for member in members]),
                "sources": [str(member.metrics_path) for member in members],
            }
        )
    return groups


def mean_scalar(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def mean_array(values: list[np.ndarray | None]) -> np.ndarray | None:
    arrays = [np.asarray(value, dtype=np.float64) for value in values if value is not None]
    return np.mean(np.stack(arrays, axis=0), axis=0) if arrays else None


def sum_arrays(values: list[np.ndarray | None]) -> np.ndarray | None:
    arrays = [np.asarray(value, dtype=np.int64) for value in values if value is not None]
    return np.sum(np.stack(arrays, axis=0), axis=0) if arrays else None


def group_has_atlas_data(group: dict[str, Any]) -> bool:
    return any(group[name] is not None for name in ("modality_weights", "modality_difficulty", "temporal_weights"))


def group_has_probe_reliability(group: dict[str, Any]) -> bool:
    return group["probe_reliability"] is not None


def manifest_group(group: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol": group["protocol"],
        "route_id": group["route_id"],
        "model_id": group["model_id"],
        "experiment_id": group["experiment_id"],
        "normalization": group["normalization"],
        "adapter_mode": group["adapter_mode"],
        "objective_id": group["objective_id"],
        "routing_id": group["routing_id"],
        "seed_count": group["seed_count"],
        "seeds": group["seeds"],
        "qwk_mean": group["qwk_mean"],
        "sources": group["sources"],
    }


def plot_protocol_atlas(plt: Any, groups: list[dict[str, Any]], path: Path, *, protocol: str) -> None:
    row_count = len(groups)
    labels = [group_label(group) for group in groups]
    weights = matrix_for_groups(groups, "modality_weights", len(MODALITIES))
    difficulty = matrix_for_groups(groups, "modality_difficulty", len(MODALITIES))
    temporal_width = max((len(group["temporal_weights"]) for group in groups if group["temporal_weights"] is not None), default=23)
    temporal = matrix_for_groups(groups, "temporal_weights", temporal_width)
    height = max(8.0, 0.48 * row_count + 3.5)
    fig, axes = plt.subplots(1, 3, figsize=(22.0, height), gridspec_kw={"width_ratios": [1.25, 1.25, 3.3]})
    fig.suptitle(
        f"Daily-affect diagnostic atlas: {protocol}\n"
        "Each row is a route/model/normalization group; values are equal-weight means across seeds.",
        fontsize=16,
        y=0.995,
    )
    plot_heatmap(plt, axes[0], weights, MODALITIES, labels, "Mean routing weight", vmin=0.0, vmax=1.0, annotate=True)
    plot_heatmap(plt, axes[1], difficulty, MODALITIES, labels, "Mean probe reliability difficulty", vmin=0.0, vmax=1.0, annotate=True)
    temporal_labels = [str(index) for index in range(temporal_width)]
    temporal_vmax = max(0.05, finite_percentile(temporal, 99.0))
    plot_heatmap(
        plt,
        axes[2],
        temporal,
        temporal_labels,
        labels,
        "Mean temporal weight by event_window_id",
        vmin=0.0,
        vmax=temporal_vmax,
        annotate=False,
    )
    for axis in axes[1:]:
        axis.set_yticklabels([])
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_probe_reliability_atlas(plt: Any, groups: list[dict[str, Any]], path: Path, *, protocol: str) -> None:
    labels = [group_label(group) for group in groups]
    values = np.stack([np.asarray(group["probe_reliability"], dtype=np.float64) for group in groups], axis=0)
    height = max(8.0, 0.48 * len(groups) + 3.5)
    fig, axes = plt.subplots(1, len(PROBE_RELIABILITY_METRICS), figsize=(21.0, height))
    fig.suptitle(
        f"Daily-affect modality Probe reliability: {protocol}\n"
        "Each cell is the test-event mean, then an equal-weight mean across seeds.",
        fontsize=16,
        y=0.995,
    )
    for index, metric in enumerate(PROBE_RELIABILITY_METRICS):
        matrix = values[:, :, index]
        vmax = 1.0 if metric == "Accuracy" else max(0.05, finite_percentile(matrix, 99.0))
        plot_heatmap(plt, axes[index], matrix, MODALITIES, labels, metric, vmin=0.0, vmax=vmax, annotate=True)
        if index:
            axes[index].tick_params(axis="y", labelleft=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def matrix_for_groups(groups: list[dict[str, Any]], field: str, width: int) -> np.ndarray:
    matrix = np.full((len(groups), width), np.nan, dtype=np.float64)
    for row, group in enumerate(groups):
        values = group[field]
        if values is None:
            continue
        array = np.asarray(values, dtype=np.float64)
        matrix[row, : min(width, array.size)] = array[:width]
    return matrix


def group_label(group: dict[str, Any]) -> str:
    qwk = group["qwk_mean"]
    qwk_label = "QWK=NA" if qwk is None else f"QWK={qwk:.3f}"
    return (
        f"{group['route_id']} | {group['model_id']} | {group['experiment_id']}\n"
        f"norm={group['normalization']} | adapter={group['adapter_mode']} | seeds={group['seed_count']} | {qwk_label}"
    )


def plot_heatmap(
    plt: Any,
    axis: Any,
    values: np.ndarray,
    x_labels: tuple[str, ...] | list[str],
    y_labels: list[str],
    title: str,
    *,
    vmin: float,
    vmax: float,
    annotate: bool,
) -> None:
    image = axis.imshow(np.ma.masked_invalid(values), aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axis.set_title(title, fontsize=11)
    axis.set_xticks(range(len(x_labels)), labels=x_labels)
    axis.set_yticks(range(len(y_labels)), labels=y_labels)
    axis.tick_params(axis="x", labelrotation=0, labelsize=8)
    axis.tick_params(axis="y", labelsize=7)
    axis.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
    axis.set_yticks(np.arange(-0.5, values.shape[0], 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.45)
    axis.tick_params(which="minor", bottom=False, left=False)
    if annotate:
        midpoint = (vmin + vmax) / 2.0
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                if not np.isfinite(value):
                    continue
                color = "white" if value < midpoint else "black"
                axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)
    plt.colorbar(image, ax=axis, fraction=0.032, pad=0.02)


def finite_percentile(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, percentile)) if finite.size else 1.0


def select_confusion_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        if group["confusion"] is not None and group["qwk_mean"] is not None:
            buckets[(group["protocol"], group["route_id"])].append(group)
    selected: list[dict[str, Any]] = []
    for key in sorted(buckets):
        selected.append(
            sorted(
                buckets[key],
                key=lambda group: (-float(group["qwk_mean"]), group["model_id"], group["normalization"]),
            )[0]
        )
    return selected


def plot_confusion_atlas(plt: Any, groups: list[dict[str, Any]], path: Path) -> None:
    protocols = sorted({group["protocol"] for group in groups})
    routes = sorted({group["route_id"] for group in groups})
    lookup = {(group["protocol"], group["route_id"]): group for group in groups}
    fig, axes = plt.subplots(len(protocols), len(routes), figsize=(4.2 * len(routes), 4.0 * len(protocols)), squeeze=False)
    fig.suptitle(
        "Daily-affect confusion atlas\n"
        "One highest-mean-test-QWK group per protocol and route; selection is visualization-only.",
        fontsize=15,
        y=0.995,
    )
    image = None
    for row, protocol in enumerate(protocols):
        for column, route in enumerate(routes):
            axis = axes[row, column]
            group = lookup.get((protocol, route))
            if group is None:
                axis.set_axis_off()
                continue
            counts = np.asarray(group["confusion"], dtype=np.float64)
            normalized = counts / counts.sum(axis=1, keepdims=True).clip(min=1.0)
            image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
            for truth in range(normalized.shape[0]):
                for predicted in range(normalized.shape[1]):
                    axis.text(
                        predicted,
                        truth,
                        f"{normalized[truth, predicted]:.0%}\n{int(counts[truth, predicted])}",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )
            axis.set_title(
                f"{protocol} | {route}\n{group['model_id']} | {group['normalization']} | QWK={group['qwk_mean']:.3f}",
                fontsize=9,
            )
            axis.set_xticks(range(5), labels=[str(index) for index in range(1, 6)])
            axis.set_yticks(range(5), labels=[str(index) for index in range(1, 6)])
            axis.set_xlabel("Predicted fatigue")
            axis.set_ylabel("True fatigue")
    if image is not None:
        colorbar_axis = fig.add_axes([0.91, 0.12, 0.018, 0.68])
        fig.colorbar(image, cax=colorbar_axis, label="Row-normalized share")
    fig.subplots_adjust(left=0.06, right=0.87, bottom=0.06, top=0.86, wspace=0.65, hspace=0.48)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def confusion_matrix(truth: np.ndarray, predicted: np.ndarray, num_classes: int = 5) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for left, right in zip(truth.tolist(), predicted.tolist()):
        if 0 <= left < num_classes and 0 <= right < num_classes:
            matrix[left, right] += 1
    return matrix


def as_float_or_none(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None


def adapter_mode_label(metrics: dict[str, Any]) -> str:
    value = metrics.get("adapter_mode")
    if value in {"shared", "per_modality"}:
        return str(value)
    return "legacy_coupled_to_normalization"


def as_int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def remove_generated_figures(out_root: Path) -> None:
    if not out_root.exists():
        return
    prefixes = LEGACY_FIGURE_PREFIXES + ATLAS_FIGURE_PREFIXES
    for path in sorted(out_root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if not path.is_file():
            continue
        if path.name == "daily_affect_figure_manifest.json" or path.name.startswith(prefixes):
            path.unlink()
    for path in sorted((item for item in out_root.rglob("*") if item.is_dir()), key=lambda item: len(item.parts), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
