#!/usr/bin/env python3
"""Freeze A's validation decision and report A; test disclosure awaits B/C."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from daily_multimodal.daily_affect.ema_bags import LABEL_NAMES

SEEDS = (240800, 240801, 240802)
METRICS = ("raw_r", "within_subject_centered_r", "rmse", "standardized_rmse", "mae")
ROUTES = ("F_C", "F_P", "F_A_C", "F_A_P")
CONTRASTS = {"adapt_C": (0, 0, 1, 0), "adapt_P": (0, 0, 0, 1)}
# Coefficients apply to metric values, preserving all four predictions during
# bootstrap; the interaction is a difference of paired metric differences.
COEFFICIENTS = {"F_A_C-F_C": (-1, 0, 1, 0), "F_A_P-F_P": (0, -1, 0, 1),
                "input_adaptation_interaction": (1, -1, -1, 1),
                "F_A_P-F_A_C": (0, 0, -1, 1)}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_write(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def scores(y, p, subjects, scale):
    y, p = y.astype(np.float64), p.astype(np.float64)
    def corr(a, b):
        a, b = a - a.mean(axis=0), b - b.mean(axis=0)
        den = np.sqrt((a*a).sum(axis=0) * (b*b).sum(axis=0))
        return np.divide((a*b).sum(axis=0), den, out=np.zeros(11), where=den > 1e-12)
    cy, cp = y.copy(), p.copy()
    for s in np.unique(subjects):
        mask = subjects == s
        cy[mask] -= cy[mask].mean(axis=0)
        cp[mask] -= cp[mask].mean(axis=0)
    error = p - y
    rmse = np.sqrt((error*error).mean(axis=0))
    return np.stack((corr(y, p), corr(cy, cp), rmse, rmse / scale, np.abs(error).mean(axis=0)), axis=1)


def read_val(directory, expected_count=269):
    # This reader deliberately never loads a test_* array or metric leaf.
    m = json.loads((directory / "metrics.json").read_text())
    if m["status"] != "ok" or m["protocol"] != "cross_day":
        raise ValueError("incomplete A cell: " + str(directory))
    with np.load(directory / "event_predictions.npz", allow_pickle=False) as z:
        labels = tuple(z["label_names"].astype(str))
        a = {k: z["val_" + k] for k in ("event_id", "subject_id", "day_id", "target", "prediction")}
    if labels != LABEL_NAMES or a["target"].shape != (expected_count, 11) or not np.isfinite(a["prediction"]).all():
        raise ValueError("A prediction contract failed")
    scale = np.asarray(m["train_target_std"], np.float64)
    v = scores(a["target"], a["prediction"], a["subject_id"], scale)
    stored = np.asarray([[m["val"]["per_label"][label][metric] for metric in METRICS] for label in LABEL_NAMES])
    if not np.isfinite(stored).all() or np.max(np.abs(v - stored)) > 2e-6:
        raise ValueError("stored validation metric differs from saved predictions")
    return m, a, stored


def paired(a, b):
    for k in ("event_id", "subject_id", "day_id", "target"):
        if not np.array_equal(a[k], b[k]):
            raise ValueError("paired event membership differs: " + k)


def bootstraps(arrays, scale, coefficients, iterations, seed):
    first = arrays[0]
    for a in arrays[1:]:
        paired(first, a)
    keys = np.char.add(np.char.add(first["subject_id"].astype(str), "/"), first["day_id"].astype(str))
    blocks = [np.flatnonzero(keys == k) for k in np.unique(keys)]
    rng = np.random.default_rng(seed)
    values = {name: np.empty((iterations, 11, 5)) for name in coefficients}
    for i in range(iterations):
        idx = np.concatenate([blocks[k] for k in rng.integers(0, len(blocks), len(blocks))])
        metric = np.stack([scores(a["target"][idx], a["prediction"][idx], a["subject_id"][idx], scale) for a in arrays])
        for name, c in coefficients.items():
            values[name][i] = np.tensordot(c, metric, axes=(0, 0))
    return {name: {"per_label": np.quantile(v, (0.025, 0.975), axis=0),
                   "macro": np.quantile(v.mean(axis=1), (0.025, 0.975), axis=0)} for name, v in values.items()}, len(blocks)


def gate(deltas):
    means = deltas.mean(axis=0)
    return {"passed": bool(means[0] > 0 and (deltas[:, 0] > 0).sum() >= 2 and means[1] >= -1e-6 and means[3] <= 1e-6),
            "raw_r_delta": float(means[0]), "raw_r_positive_seeds": int((deltas[:, 0] > 0).sum()),
            "centered_r_delta": float(means[1]), "standardized_rmse_delta": float(means[3]),
            "rule": "mean_raw_r>0; positive>=2/3; mean_centered>=-1e-6; mean_sRMSE<=1e-6"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--bootstrap-iters", type=int, default=2000)
    a = p.parse_args()
    if a.bootstrap_iters != 2000:
        raise ValueError("experiment A fixes 2000 bootstrap draws")
    out = a.root / "reports"
    out.mkdir(exist_ok=True)
    for marker in ("STAGE0_COMPLETE", "A_ADAPTATION_SMOKE_COMPLETE", "A_ADAPTATION_FORMAL_COMPLETE", "A_FUSION_SMOKE_COMPLETE", "A_FUSION_FORMAL_COMPLETE"):
        if not (a.root / marker).exists():
            raise ValueError("missing gate: " + marker)
    registry = json.loads((a.root / "inputs/source_registry.json").read_text())
    reuse = json.loads((a.root / "inputs/reuse_audit.json").read_text())
    for v in registry.values():
        for name, h in v["sha256"].items():
            if sha(Path(v["directory"]) / name) != h:
                raise ValueError("SSL source changed after Stage0")
    paths = {}
    cells, provenance = [], {}
    for route in ROUTES:
        for seed in SEEDS:
            if route in ("F_C", "F_P"):
                r = next(r for r in reuse["cells"] if r["source"] == route[-1] and r["seed"] == seed)
                d = Path(r["directory"])
                if sha(d / "metrics.json") != r["metrics_sha256"] or sha(d / "event_predictions.npz") != r["predictions_sha256"]:
                    raise ValueError("reused control changed")
            else:
                d = a.root / f"fusion/{route}/formal/cross_day/seed_{seed}"
                if not (d / "FUSION_COMPLETE").exists():
                    raise ValueError("fusion incomplete")
            paths[route, seed] = d
            cells.append({"protocol": "cross_day", "cell": route, "seed": seed,
                          "status": "completed", "reused": route in ("F_C", "F_P"), "directory": str(d)})
            provenance[str(d)] = {name: sha(d / name) for name in ("metrics.json", "event_predictions.npz")}
    metrics, arrays, values = {}, {}, {}
    for key, d in paths.items():
        metrics[key], arrays[key], values[key] = read_val(d)
    # Freeze val selection before bootstrap/reporting. B/C remain unexecuted;
    # A's test results stay saved and undisclosed until the full Stage-D record.
    all_values = np.stack([np.stack([values[route, seed] for seed in SEEDS]) for route in ROUTES])
    gates = {name: gate(np.tensordot(c, all_values, axes=(0, 0)).mean(axis=1))
             for name, c in COEFFICIENTS.items() if name.startswith("F_A_") and name not in ("F_A_P-F_A_C",)}
    selection = {"scope": "experiment_A", "leaf": "val_only", "seeds": list(SEEDS), "gates": gates,
                 "B_C": "pending", "test_disclosure": "deferred_until_full_round2_Stage_D",
                 "upstream_adaptation_seed_count": 1, "provenance": provenance}
    selection_path = out / "selection_val_only.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("frozen A selection changed")
    write(selection_path, selection)
    seed_rows, macro_rows, delta_rows, boot_rows, interaction = [], [], [], [], []
    for route in ROUTES:
        for seed in SEEDS:
            for j, label in enumerate(LABEL_NAMES):
                seed_rows.append({"protocol": "cross_day", "route": route, "seed": seed, "split": "val", "label": label,
                                  **{k: float(values[route, seed][j, i]) for i, k in enumerate(METRICS)}})
        val = np.stack([values[route, seed].mean(axis=0) for seed in SEEDS])
        for i, k in enumerate(METRICS):
            macro_rows.append({"split": "val", "route": route, "metric": k, "mean": float(val[:, i].mean()),
                               "seed_sd": float(val[:, i].std(ddof=1)), "seed_count": 3})
    for seed in SEEDS:
        ms = [metrics[r, seed] for r in ROUTES]
        for m in ms[1:]:
            for field in ("condition_id", "model_id", "temporal_policy", "normalization", "adapter_mode", "head_variant",
                          "selection_metric", "train_target_mean", "train_target_std", "downstream_seed", "trainable_params"):
                if m[field] != ms[0][field]:
                    raise ValueError("fusion pairing mismatch: " + field)
        ci, blocks = bootstraps([arrays[r, seed] for r in ROUTES], np.asarray(ms[0]["train_target_std"]),
                               COEFFICIENTS, a.bootstrap_iters, seed + 90210)
        vv = np.stack([values[r, seed] for r in ROUTES])
        for name, c in COEFFICIENTS.items():
            delta = np.tensordot(c, vv, axes=(0, 0))
            for j, label in enumerate(LABEL_NAMES):
                for i, k in enumerate(METRICS):
                    delta_rows.append({"split": "val", "contrast": name, "seed": seed, "label": label,
                                       "metric": k, "delta": float(delta[j, i])})
                    boot_rows.append({"split": "val", "contrast": name, "seed": seed, "label": label, "metric": k,
                                      "delta": float(delta[j, i]), "low": float(ci[name]["per_label"][0, j, i]),
                                      "high": float(ci[name]["per_label"][1, j, i]), "iterations": 2000, "subject_day_blocks": blocks})
            for i, k in enumerate(METRICS):
                boot_rows.append({"split": "val", "contrast": name, "seed": seed, "label": "macro", "metric": k,
                                  "delta": float(delta[:, i].mean()), "low": float(ci[name]["macro"][0, i]),
                                  "high": float(ci[name]["macro"][1, i]), "iterations": 2000, "subject_day_blocks": blocks})
        print(f"paired_bootstrap_complete seed={seed}", flush=True)
    for name, c in COEFFICIENTS.items():
        deltas = np.tensordot(c, all_values, axes=(0, 0)).mean(axis=1)
        for i, k in enumerate(METRICS):
            interaction.append({"split": "val", "contrast": name, "metric": k,
                                "mean_delta": float(deltas[:, i].mean()), "seed_sd": float(deltas[:, i].std(ddof=1)),
                                "positive_seeds": int((deltas[:, i] > 0).sum()), "seed_count": 3})
    eeg_rows, eeg_delta, eeg_boot = [], [], []
    for source in ("C", "P"):
        group = []
        for mode in ("PROBE", "ADAPT"):
            d = a.root / f"adaptation/A_{source}_{mode}/formal/cross_day/seed_240800"
            if not (d / "ADAPTATION_COMPLETE").exists():
                raise ValueError("adaptation incomplete")
            m, ar, v = read_val(d)
            if mode == "ADAPT":
                with np.load(d / "window_embeddings.npz", allow_pickle=False) as z:
                    with np.load(Path(registry[source]["directory"]) / "window_embeddings.npz", allow_pickle=False) as original:
                        if not np.array_equal(z["sample_id"], original["sample_id"]) or z["embedding"].shape != (28819, 256) or not z["valid_mask"].all() or not np.isfinite(z["embedding"]).all():
                            raise ValueError("adapted token completion failed")
            group.append((m, ar, v))
            cells.append({"protocol": "cross_day", "cell": f"A_{source}_{mode}", "seed": 240800, "status": "completed", "reused": False, "directory": str(d)})
            for j, label in enumerate(LABEL_NAMES):
                eeg_rows.append({"source": source, "mode": mode, "split": "val", "seed": 240800, "label": label,
                                 **{k: float(v[j, i]) for i, k in enumerate(METRICS)}})
        ci, blocks = bootstraps([g[1] for g in group], np.asarray(group[0][0]["train_target_std"]),
                               {"ADAPT-PROBE": (-1, 1)}, 2000, 240800 + 77)
        delta = group[1][2] - group[0][2]
        for i, k in enumerate(METRICS):
            eeg_delta.append({"source": source, "split": "val", "metric": k, "delta": float(delta[:, i].mean()), "upstream_seed_count": 1})
            for j, label in enumerate(LABEL_NAMES):
                eeg_boot.append({"source": source, "split": "val", "seed": 240800, "label": label, "metric": k,
                                 "delta": float(delta[j, i]), "low": float(ci["ADAPT-PROBE"]["per_label"][0, j, i]),
                                 "high": float(ci["ADAPT-PROBE"]["per_label"][1, j, i]), "iterations": 2000, "subject_day_blocks": blocks})
    for cell in list(cells):
        cells.append({**cell, "protocol": "within_subject_day", "status": "stopped_protocol_overlap", "directory": None,
                      "reason": "107 canonical train windows overlap holdout; pending_protocol_revision"})
    for filename, table in (("per_label_seed_metrics.csv", seed_rows), ("macro_paired_metrics.csv", macro_rows),
                            ("per_label_paired_deltas.csv", delta_rows), ("per_seed_paired_bootstrap.csv", boot_rows),
                            ("interaction_metrics.csv", interaction), ("eeg_only_metrics.csv", eeg_rows),
                            ("eeg_only_deltas.csv", eeg_delta), ("eeg_only_bootstrap.csv", eeg_boot)):
        csv_write(out / filename, table)
    write(out / "cell_status.json", {"scope": "A", "cells": cells, "B": "planned", "C": "planned"})
    result = {"scope": "A", "status": "completed_with_protocol_stop", "split": "val", "gates": gates,
              "fusion_macro": macro_rows, "contrasts": interaction, "eeg_only_deltas": eeg_delta,
              "test_report": "deferred_until_full_round2_Stage_D", "upstream_adaptation_seeds": [240800],
              "within_subject_day": "stopped_protocol_overlap", "bootstrap_iterations_per_seed": 2000}
    write(out / "results.json", result)
    lines = ["# MAE 第二轮实验 A：EEG 单模态情绪适配", "", "cross_day 实验 A 已完成：4 个 EEG-only 模型、2 份适配 token、6 个新融合 cell 和 6 个匹配复用 cell。以下报告 val；test 预测已保存，统一披露遵循完整第二轮 Stage D。", "",
             "每个融合 seed 先平均 11 标签，表中为三 seed 均值 ± 样本 SD。上游适配 seed 固定 240800；融合 SD 描述下游训练波动。", "",
             "| 路线 | raw r | centered r | RMSE | sRMSE | MAE |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for route in ROUTES:
        v = np.stack([values[route, s].mean(axis=0) for s in SEEDS])
        lines.append("| " + route + " | " + " | ".join(f"{v[:, i].mean():.6f} ± {v[:, i].std(ddof=1):.6f}" for i in range(5)) + " |")
    lines.extend(["", "## 预声明的适配收益", "", "| 配对差 | raw r delta | centered r delta | sRMSE delta | raw r 正向 seed | 联合门槛 |",
                  "| --- | ---: | ---: | ---: | ---: | --- |"])
    for name, g in gates.items():
        lines.append(f"| {name} | {g['raw_r_delta']:+.6f} | {g['centered_r_delta']:+.6f} | {g['standardized_rmse_delta']:+.6f} | {g['raw_r_positive_seeds']}/3 | {g['passed']} |")
    lines.extend(["", "输入×适配交互为 `(F_A_P−F_P)−(F_A_C−F_C)`，每 seed 独立计算，并与四路线共同的 subject-day bootstrap 重采样配对。逐情绪、每 seed 和 95% delta 区间见同目录 CSV。", "",
                  "EEG-only ADAPT−PROBE 单列在 eeg_only_deltas.csv；其上游训练 seed 数为 1。P 保留第一轮 gate_passed=false 的来源状态。", "",
                  "within_subject_day 全部预定 A cell 保留 stopped_protocol_overlap / pending_protocol_revision：107 个 canonical train 窗与 holdout 信号相交，整 session 预处理上下文共享。", "",
                  "B/C 继续保持计划状态；本次完成标记限定实验 A。完整第二轮 test 报告和联合确认沿用原计划的冻结及授权顺序。"])
    (out / "experiment_a_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (a.root / "EXPERIMENT_A_COMPLETE_WITH_PROTOCOL_STOP").write_text("cross_day A complete; within_subject_day protocol stopped\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
