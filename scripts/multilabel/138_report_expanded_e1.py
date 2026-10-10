#!/usr/bin/env python3
"""Report completed expanded-pool E1 against matched canonical-only v4 E1."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from daily_multimodal.daily_affect.ema_bags import LABEL_NAMES
from daily_multimodal.training.centered_metrics import safe_pearsonr, within_subject_centered_arrays

SEEDS = (240800, 240801, 240802)
ROUTES = ('E1_original_v4', 'E1_expanded')
METRICS = ('raw_r', 'within_subject_centered_r', 'rmse', 'standardized_rmse', 'mae')
ZH = ('受鼓舞', '警觉', '坚定', '专注', '活跃', '敌意', '紧张', '心烦', '害怕', '羞愧', '疲劳')


def score(y, p, subjects, scale):
    """Apply the source metric definition, including float32 centered arrays."""
    result = np.empty((len(LABEL_NAMES), len(METRICS)))
    for j in range(len(LABEL_NAMES)):
        true, pred = y[:, j], p[:, j]
        cy, cp = within_subject_centered_arrays(true, pred, subjects)
        error = pred - true
        rmse = float(np.sqrt(np.mean(error * error)))
        result[j] = (safe_pearsonr(true, pred), safe_pearsonr(cy, cp),
                     rmse, rmse / float(scale[j]), float(np.mean(np.abs(error))))
    if not np.isfinite(result).all():
        raise ValueError('undefined or nonfinite metric')
    return result


def bootstrap(y, pa, pb, subjects, days, scale, iterations, seed):
    keys = np.char.add(np.char.add(subjects.astype(str), '/'), days.astype(str))
    blocks = [np.flatnonzero(keys == key) for key in np.unique(keys)]
    rng = np.random.default_rng(seed)
    values = np.empty((iterations, len(LABEL_NAMES), len(METRICS)))
    for i in range(iterations):
        idx = np.concatenate([blocks[k] for k in rng.integers(0, len(blocks), len(blocks))])
        values[i] = score(y[idx], pa[idx], subjects[idx], scale) - score(y[idx], pb[idx], subjects[idx], scale)
    return np.quantile(values, (.025, .975), axis=0), len(blocks)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bootstrap-iters', type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_iters < 1:
        raise ValueError('bootstrap iterations must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    audit = json.loads((args.sources / 'completion_audit.json').read_text())
    if audit['status'] != 'pass' or not audit['paired_membership_verified'] or not audit['non_target_modality_verified']:
        raise ValueError('upstream or bag completion audit failed')
    metrics = {}
    predictions = {}
    hashes = {}
    maximum_error = 0.0
    fields = ('protocol', 'condition_id', 'model_id', 'temporal_policy', 'normalization',
              'adapter_mode', 'head_variant', 'embedding_seed', 'downstream_seed',
              'selection_metric', 'trainable_params', 'train_target_mean', 'train_target_std')
    seed_rows = []
    paired = []
    arrays = {}
    for seed in SEEDS:
        for route in ROUTES:
            directory = args.sources / route / f'seed_{seed}'
            for name in ('metrics.json', 'event_predictions.npz'):
                path = directory / name
                hashes[path.relative_to(args.sources).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            m = json.loads((directory / 'metrics.json').read_text())
            if m['status'] != 'ok' or m['protocol'] != 'cross_day' or m['downstream_seed'] != seed:
                raise ValueError('run status or protocol mismatch')
            if route == 'E1_expanded':
                expected = next(c['metrics_sha256'] for c in audit['downstream_cells'] if c['seed'] == seed)
                if hashes[f'{route}/seed_{seed}/metrics.json'] != expected:
                    raise ValueError('completed expanded run hash changed')
            metrics[route, seed] = m
            with np.load(directory / 'event_predictions.npz', allow_pickle=False) as z:
                predictions[route, seed] = {k: z[k] for k in z.files}
            if tuple(predictions[route, seed]['label_names'].astype(str)) != LABEL_NAMES:
                raise ValueError('prediction label order mismatch')
        old, new = [metrics[r, seed] for r in ROUTES]
        for field in fields:
            if old[field] != new[field]:
                raise ValueError('downstream contract mismatch: ' + field)
        for leaf, expected_count in (('val', 269), ('test', 253)):
            po, pn = [predictions[r, seed] for r in ROUTES]
            for field in ('event_id', 'target', 'subject_id', 'day_id'):
                if not np.array_equal(po[f'{leaf}_{field}'], pn[f'{leaf}_{field}']):
                    raise ValueError('paired membership mismatch: ' + field)
            y = pn[f'{leaf}_target']
            subjects, days = pn[f'{leaf}_subject_id'], pn[f'{leaf}_day_id']
            if y.shape != (expected_count, 11):
                raise ValueError('event shape mismatch')
            scale = np.asarray(new['train_target_std'])
            for route in ROUTES:
                p = predictions[route, seed][f'{leaf}_prediction']
                got = score(y, p, subjects, scale)
                stored = np.asarray([[metrics[route, seed][leaf]['per_label'][label][key]
                                      for key in METRICS] for label in LABEL_NAMES])
                error = float(np.max(np.abs(got - stored)))
                maximum_error = max(maximum_error, error)
                if error > 1e-6:
                    raise ValueError('saved metric disagrees with predictions')
                arrays[route, seed, leaf] = stored
                for j, label in enumerate(LABEL_NAMES):
                    detail = metrics[route, seed][leaf]['per_label'][label]
                    seed_rows.append({'protocol': 'cross_day', 'route': route, 'seed': seed,
                                      'split': leaf, 'label': label, **detail})
            ci, block_count = bootstrap(y, pn[f'{leaf}_prediction'], po[f'{leaf}_prediction'],
                                       subjects, days, scale, args.bootstrap_iters, seed + 90210)
            delta = arrays['E1_expanded', seed, leaf] - arrays['E1_original_v4', seed, leaf]
            for j, label in enumerate(LABEL_NAMES):
                for k, metric in enumerate(METRICS):
                    paired.append({'split': leaf, 'seed': seed, 'label': label, 'metric': metric,
                                   'delta': float(delta[j, k]), 'bootstrap_low': float(ci[0, j, k]),
                                   'bootstrap_high': float(ci[1, j, k]), 'bootstrap_iters': args.bootstrap_iters,
                                   'subject_day_blocks': block_count})
            print(json.dumps({'seed': seed, 'split': leaf, 'verified': True, 'bootstrap_complete': True}), flush=True)
    summary = []
    macro = {}
    for leaf in ('val', 'test'):
        old = np.stack([arrays['E1_original_v4', seed, leaf] for seed in SEEDS])
        new = np.stack([arrays['E1_expanded', seed, leaf] for seed in SEEDS])
        delta = new - old
        macro[leaf] = {}
        for k, metric in enumerate(METRICS):
            o, n, d = old[:, :, k].mean(axis=1), new[:, :, k].mean(axis=1), delta[:, :, k].mean(axis=1)
            macro[leaf][metric] = {'old_mean': float(o.mean()), 'old_sd': float(o.std(ddof=1)),
                                   'new_mean': float(n.mean()), 'new_sd': float(n.std(ddof=1)),
                                   'delta_mean': float(d.mean()), 'delta_sd': float(d.std(ddof=1)),
                                   'positive_seeds': int((d > 0).sum())}
            for j, label in enumerate(LABEL_NAMES):
                row = {'split': leaf, 'label': label, 'metric': metric,
                       'old_mean': float(old[:, j, k].mean()), 'old_sd': float(old[:, j, k].std(ddof=1)),
                       'new_mean': float(new[:, j, k].mean()), 'new_sd': float(new[:, j, k].std(ddof=1)),
                       'delta_mean': float(delta[:, j, k].mean()), 'delta_sd': float(delta[:, j, k].std(ddof=1)),
                       'positive_seeds': int((delta[:, j, k] > 0).sum())}
                for i, seed in enumerate(SEEDS):
                    row[f'old_{seed}'] = float(old[i, j, k])
                    row[f'new_{seed}'] = float(new[i, j, k])
                    row[f'delta_{seed}'] = float(delta[i, j, k])
                summary.append(row)
    val = macro['val']
    gate = (val['raw_r']['delta_mean'] > 0 and val['raw_r']['positive_seeds'] >= 2
            and val['within_subject_centered_r']['delta_mean'] >= -1e-6
            and val['standardized_rmse']['delta_mean'] <= 1e-6)
    prior = json.loads((args.sources / 'validation_comparison.json').read_text())
    expected = next(x for x in prior['pairs'] if x['control'] == 'E_CANON_V4')
    if gate != expected['predeclared_promotion_gate']:
        raise ValueError('validation gate changed')
    for metric in METRICS:
        if abs(val[metric]['delta_mean'] - expected['metrics'][metric]['delta_mean']) > 1e-12:
            raise ValueError('prior validation contrast changed')
    result = {'status': 'complete', 'scope': 'EEG E1 comparison only; video queue remains independent',
              'protocol': 'cross_day', 'original_E1': 'final v4 canonical-only EEG MAE',
              'expanded_E1': 'E_POOL frozen single EEG slot replacement', 'seeds': SEEDS,
              'embedding_seed': 240800, 'seed_sd_ddof': 1, 'source_sha256': hashes,
              'prediction_metric_max_abs_error': maximum_error, 'paired_membership_verified': True,
              'bootstrap_unit': 'subject-day paired within each seed', 'bootstrap_iters': args.bootstrap_iters,
              'validation_gate': bool(gate), 'test_used_for_selection': False,
              'within_subject_day': 'stopped_canonical_signal_overlap; no expanded E1 comparison',
              'summary': summary, 'macro': macro, 'paired_per_label': paired}
    (args.out / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    write_csv(args.out / 'per_label_seed_metrics.csv', seed_rows)
    write_csv(args.out / 'per_label_comparison.csv', summary)
    write_csv(args.out / 'per_label_seed_paired_bootstrap.csv', paired)
    lines = ['# 扩展 EEG MAE 的 E1 逐情绪对照', '',
             '对照：最终 v4 canonical-only E1；新路线：E_POOL，作为扩池后的 E1。', '',
             'EEG SSL train 从 16,813 个 canonical 窗口扩为 60,994 个窗口（新增 raw 44,181 个，新增独有覆盖约 62.028 小时）。'
             '上游 seed 240800；冻结下游 seeds 240800/240801/240802。保留 Wphysio、DINO A1、23 窗口/event、'
             '`window_attention_regression_full_mean`、11-head、split 与优化合同。', '',
             '表中为三 seed 均值 ± 样本 SD（ddof=1）；Δ=新−旧，相关系数越高越好。centered r 在当前评估 split 内'
             '分别减去每个被试的标签均值与预测均值，再计算整体 Pearson r。', '',
             '每个 seed 均核对 event/target/subject/day/label 顺序，独立重算全部五类指标；'
             f'与保存指标的最大误差 {maximum_error:.3g}。每情绪每 seed 的 subject-day 配对 bootstrap 为 {args.bootstrap_iters} 次，'
             '95% percentile 区间保存在 CSV，三 seed 独立报告。', '']
    lookup = {(x['split'], x['label'], x['metric']): x for x in summary}
    for leaf, title in (('test', '测试集：253 个 EMA event'), ('val', '验证集：269 个 EMA event')):
        lines += [f'## {title}', '']
        for metric, title_metric in zip(METRICS, ('raw r', 'centered r', 'RMSE', 'standardized RMSE', 'MAE')):
            lines += [f'### {title_metric}', '',
                      '| 情绪 | 原 E1 v4 ± SD | 扩池 E1 ± SD | Δ ± 配对 SD | 正向 seeds |',
                      '| --- | ---: | ---: | ---: | ---: |']
            for label, zh in zip(LABEL_NAMES, ZH):
                x = lookup[leaf, label, metric]
                lines.append(f"| {zh} ({label}) | {x['old_mean']:.4f} ± {x['old_sd']:.4f} | "
                             f"{x['new_mean']:.4f} ± {x['new_sd']:.4f} | {x['delta_mean']:+.4f} ± {x['delta_sd']:.4f} | {x['positive_seeds']}/3 |")
            lines += ['']
        lines += ['### 各 seed 相关系数', '',
                  '| 情绪 | seed | 原 raw r | 新 raw r | Δ raw r | 原 centered r | 新 centered r | Δ centered r |',
                  '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for label, zh in zip(LABEL_NAMES, ZH):
            raw, centered = lookup[leaf, label, METRICS[0]], lookup[leaf, label, METRICS[1]]
            for seed in SEEDS:
                lines.append(f"| {zh} ({label}) | {seed} | {raw[f'old_{seed}']:.4f} | {raw[f'new_{seed}']:.4f} | "
                             f"{raw[f'delta_{seed}']:+.4f} | {centered[f'old_{seed}']:.4f} | {centered[f'new_{seed}']:.4f} | {centered[f'delta_{seed}']:+.4f} |")
        lines += ['']
    lines += ['## 固定判定与范围', '',
              f'原 val 联合推进门槛：{gate}。val raw r Δ={val[METRICS[0]]["delta_mean"]:+.8f}、'
              f'centered r Δ={val[METRICS[1]]["delta_mean"]:+.8f}、sRMSE Δ={val[METRICS[3]]["delta_mean"]:+.8f}；'
              'sRMSE 超过 +0.000001 容差。测试结果按冻结配置完整报告，未据其调整 checkpoint、seed 或方案。', '',
              '`within_subject_day` 按既定实际信号重叠门槛停止：107 个 canonical train 窗口与 held-out 输入重叠，'
              '扩池 E1 同日结果保持 stopped。历史同日 E1 保留原来源与协议。', '',
              '单个固定上游 seed 的三 seed 下游证据适用于本次输入对照；相关系数逐情绪探索性结果不作多重检验后的显著性声明。', '',
              '完整精度、源文件 SHA256、合同核验和 val gate 见 `results.json`；'
              '每情绪原/新各 seed 全指标含标签/预测 SD、尾部计数见 `per_label_seed_metrics.csv`。', '']
    (args.out / 'expanded_e1_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'status': 'complete', 'macro': macro, 'validation_gate': bool(gate),
                      'per_label_summary_rows': len(summary), 'paired_bootstrap_rows': len(paired)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
