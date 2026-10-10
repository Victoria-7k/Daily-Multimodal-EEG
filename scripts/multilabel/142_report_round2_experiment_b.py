#!/usr/bin/env python3
"""B's predeclared val-only contrasts, paired bootstrap and practical gate."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from daily_multimodal.daily_affect.ema_bags import LABEL_NAMES
from daily_multimodal.training.mae_eeg_partial_ft import VARIANTS

spec = importlib.util.spec_from_file_location("round2_validation_helpers", Path(__file__).with_name("140_report_round2_experiment_a.py"))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
SEEDS = h.SEEDS
ROUTES = ("F_C", *VARIANTS)
COEFFICIENTS = {"B_T2_LOW-B_T2_STD": (0, -1, 1, 0), "B_T1_LOW-B_T2_LOW": (0, 0, -1, 1),
    "B_T2_STD-F_C": (-1, 1, 0, 0), "B_T2_LOW-F_C": (-1, 0, 1, 0), "B_T1_LOW-F_C": (-1, 0, 0, 1)}

def select_candidate(gates, means):
    eligible = []
    for variant, mechanism in (("B_T2_LOW", "B_T2_LOW-B_T2_STD"), ("B_T1_LOW", "B_T1_LOW-B_T2_LOW")):
        if gates[mechanism]["passed"] and gates[variant + "-F_C"]["passed"]:
            eligible.append(variant)
    if not eligible:
        return None
    minimum = min(means[v] for v in eligible)
    tied = [v for v in eligible if means[v] <= minimum + 1e-6]
    return min(tied, key=lambda v: VARIANTS[v])

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    args = p.parse_args()
    r = args.root
    out = r / "reports/experiment_b"
    out.mkdir(parents=True, exist_ok=True)
    for marker in ("B_STAGE0_COMPLETE", "B_PREFIX_COMPLETE", "B_SMOKE_COMPLETE", "B_FORMAL_COMPLETE"):
        if not (r / marker).exists():
            raise ValueError("B gate missing: " + marker)
    reuse = json.loads((r / "inputs/reuse_audit.json").read_text())
    values, arrays, metrics, provenance, cells = {}, {}, {}, {}, []
    for route in ROUTES:
        for seed in SEEDS:
            if route == "F_C":
                row = next(v for v in reuse["cells"] if v["source"] == "C" and v["seed"] == seed)
                d = Path(row["directory"])
                if h.sha(d / "metrics.json") != row["metrics_sha256"] or h.sha(d / "event_predictions.npz") != row["predictions_sha256"]:
                    raise ValueError("frozen C control changed")
            else:
                d = r / f"fusion/{route}/formal/cross_day/seed_{seed}"
                if not (d / "FT_COMPLETE").exists():
                    raise ValueError("missing B cell")
            metrics[route, seed], arrays[route, seed], values[route, seed] = h.read_val(d)
            provenance[str(d)] = {n: h.sha(d / n) for n in ("metrics.json", "event_predictions.npz")}
            h.paired(arrays["F_C", seed], arrays[route, seed])
            cells.append({"protocol": "cross_day", "route": route, "seed": seed, "status": "completed", "reused": route == "F_C", "directory": str(d)})
    tensor = np.stack([np.stack([values[v, s] for s in SEEDS]) for v in ROUTES])
    contrasts = {n: np.tensordot(c, tensor, axes=(0, 0)) for n, c in COEFFICIENTS.items()}
    gates = {n: h.gate(d.mean(axis=1)) for n, d in contrasts.items()}
    means = {v: float(tensor[i, :, :, 3].mean()) for i, v in enumerate(ROUTES)}
    selection = {"scope": "experiment_B", "leaf": "val_only", "seeds": list(SEEDS), "gates": gates,
        "selected_practical_ft": select_candidate(gates, means), "C": "pending",
        "selection_rule": "T2_LOW_vs_T2_STD_and_F_C; T1_LOW_vs_T2_LOW_and_F_C; minimum_mean_sRMSE; ties_1e-6_fewer_blocks_then_lower_LR",
        "B_T2_STD_role": "new_standard_LR_mechanism_control", "provenance": provenance,
        "test_disclosure": "deferred_until_full_round2_Stage_D"}
    h.write(out / "selection_val_only.json", selection)
    per_label, delta_rows, epoch_rows, bootstrap_rows, summaries, seed_rows = [], [], [], [], {}, []
    for i, route in enumerate(ROUTES):
        macro = tensor[i].mean(axis=1)
        summaries[route] = {m: {"mean": float(macro[:, j].mean()), "sd": float(macro[:, j].std(ddof=1))} for j, m in enumerate(h.METRICS)}
        for si, seed in enumerate(SEEDS):
            seed_rows.append({"route": route, "seed": seed, "best_epoch": metrics[route, seed]["best_epoch"], **dict(zip(h.METRICS, macro[si]))})
            for li, label in enumerate(LABEL_NAMES):
                per_label.append({"route": route, "seed": seed, "label": label, **dict(zip(h.METRICS, tensor[i, si, li]))})
            if route != "F_C":
                for record in metrics[route, seed]["history"]:
                    epoch_rows.append({"route": route, "seed": seed, **record})
    for name, delta in contrasts.items():
        for si, seed in enumerate(SEEDS):
            for li, label in enumerate((*LABEL_NAMES, "macro")):
                v = delta[si, li] if li < 11 else delta[si].mean(axis=0)
                delta_rows.append({"contrast": name, "seed": seed, "label": label, **dict(zip(h.METRICS, v))})
    for seed in SEEDS:
        scale = np.asarray(metrics["F_C", seed]["train_target_std"])
        for route in ROUTES[1:]:
            if not np.array_equal(scale, np.asarray(metrics[route, seed]["train_target_std"])):
                raise ValueError("training label normalization differs from F_C")
        intervals, block_count = h.bootstraps([arrays[v, seed] for v in ROUTES], scale, COEFFICIENTS, 2000, seed + 91000)
        for name, data in intervals.items():
            for li, label in enumerate((*LABEL_NAMES, "macro")):
                ci = data["per_label"][:, li] if li < 11 else data["macro"]
                for j, metric in enumerate(h.METRICS):
                    bootstrap_rows.append({"contrast": name, "seed": seed, "label": label, "metric": metric,
                        "delta": float((contrasts[name][SEEDS.index(seed), li] if li < 11 else contrasts[name][SEEDS.index(seed)].mean(axis=0))[j]),
                        "ci_low": float(ci[0, j]), "ci_high": float(ci[1, j]), "subject_day_blocks": block_count, "iterations": 2000})
    for name, rows in (("per_label_metrics.csv", per_label), ("paired_deltas.csv", delta_rows), ("epoch_diagnostics.csv", epoch_rows),
                       ("paired_subject_day_bootstrap.csv", bootstrap_rows), ("seed_metrics.csv", seed_rows)):
        h.csv_write(out / name, rows)
    for route in VARIANTS:
        for seed in SEEDS:
            cells.append({"protocol": "within_subject_day", "route": route, "seed": seed, "status": "stopped_protocol_overlap", "reason": "canonical_signal_and_preprocessing_context_overlap"})
    h.write(out / "cell_status.json", cells)
    h.write(out / "results.json", {"summaries": summaries, "gates": gates, "selection": selection,
        "bootstrap_unit": "subject_day_within_each_seed", "seed_count": 3, "upstream_ssl_seed_count": 1})
    lines = ["# 实验 B：EEG 尾部学习率与解冻范围", "", "cross_day；每配置三个匹配下游 seed，原始 C SSL 初始化。所有模型由 val event sRMSE 选择。", "",
        "| 配置 | raw r mean±SD | centered r mean±SD | sRMSE mean±SD |", "|---|---:|---:|---:|"]
    for route in ROUTES:
        values_ = summaries[route]
        lines.append("| " + route + " | " + " | ".join(f"{values_[k]['mean']:.6f} ± {values_[k]['sd']:.6f}" for k in (h.METRICS[0], h.METRICS[1], h.METRICS[3])) + " |")
    lines += ["", "| 配对差 | Δraw r | 正向 seed | Δcentered r | ΔsRMSE | 联合门槛 |", "|---|---:|---:|---:|---:|---|"]
    for name, g in gates.items():
        lines.append(f"| {name} | {g['raw_r_delta']:+.6f} | {g['raw_r_positive_seeds']}/3 | {g['centered_r_delta']:+.6f} | {g['standardized_rmse_delta']:+.6f} | {g['passed']} |")
    lines += ["", f"预声明实用 FT 候选：`{selection['selected_practical_ft']}`。", "",
        "`B_T2_STD` 提供本轮标准 LR 控制。低 LR/末一层候选分别须通过机制对照与 F_C 对照。",
        "`epoch_diagnostics.csv` 包含无 dropout 的 train event sRMSE、val 指标、训练随机 window MSE、tail L2 偏移、256 个固定 val 窗口的 token 相对变化与有效秩。有效秩使用中心化 token 的奇异值熵。",
        "`paired_subject_day_bootstrap.csv` 每 seed 对四条路线共同进行 2000 次 subject-day block 重采样，保存逐情绪与宏指标的配对 95% 区间。",
        "", "within_subject_day 延续严格 overlap 停止状态。Test 预测已保存；候选选择与本报告只读取 val，整体 test 报告等待 C 和 Stage D。"]
    (out / "experiment_b_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (r / "EXPERIMENT_B_COMPLETE_WITH_PROTOCOL_STOP").write_text("complete\n")
    print(json.dumps({"status": "complete", "summaries": summaries, "gates": gates, "selected": selection["selected_practical_ft"]}), flush=True)

if __name__ == "__main__":
    main()
