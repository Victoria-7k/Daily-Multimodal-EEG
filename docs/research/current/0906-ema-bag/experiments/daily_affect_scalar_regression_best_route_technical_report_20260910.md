# Daily-affect 标量回归：三协议最佳 raw-r 路线技术报告

> 日期：2026-09-10  
> 结果根：`/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/outputs/daily_affect_scalar_regression_20260908/v2_partialft_noaudio/`  
> 范围：只报告三种数据划分协议中，最终 raw-r 最强且完成 seven-seed 配对验证的路线。所有数字均为 held-out **EMA event-level** 指标。

## 1. 结果总览

| protocol | 协议内最佳路线 | 结构形式 | seeds | raw r | 相对窗口 Δraw r | 正向 seed | RMSE | centered r | 判定 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `within_subject_day` | `window_attention_regression_full_mean` | 独立窗口 attention 回归，23 窗口等权平均为 event | 7 | `0.4314 ± 0.0105` | reference | -- | `0.8952 ± 0.0065` | `0.1964 ± 0.0113` | 保留窗口路线 |
| `cross_day` | `bag_static_reg__temporal_last_30s` | 独立窗口 attention 融合，最后 30 s 的固定时间汇总后回归 | 7 | `0.4091 ± 0.0314` | `+0.0453 ± 0.0291` | `7/7` | `0.8788 ± 0.0188` | `0.2217 ± 0.0225` | EMA-bag 获得支持 |
| `cross_subject` | `prior_uniform_reg` | 独立窗口 attention 融合 + state-prior modality routing + 均匀时间汇总 | 7 | `0.1072 ± 0.0425` | `+0.0631 ± 0.0596` | `6/7` | `0.9405 ± 0.0429` | `0.1286 ± 0.0424` | EMA-bag 获得支持 |

比较基线在每个 `protocol × seed` 内完全匹配：同一个 EMA bag、相同的事件顺序、标签、mask、split、下游随机种子和训练配置。`Δraw r` 恒定义为 `EMA-bag route − matched window full-mean`。

本轮先完成了 `3 protocols × 3 seeds × 19 conditions = 171` 个运行：1 个窗口基线与 18 个 EMA-bag 标量回归条件。通过三 seed 门槛的候选随后补齐 `240732--240735` 四个 seed，最终共有 199 个 `metrics.json` 和 178 条 window-paired 结果。

## 2. 共同数据、embedding 与监督契约

### 2.1 原始事件至 canonical 窗口

原始 EEG、PPG/GSR/ACC 与视频数据以事件记录为起点，由 `index/eeg_aligned_window_index.jsonl` 定义 canonical EEG-aligned 样本。每个 fatigue EMA event 展开为 23 个重叠的 10 秒窗口，stride 为 5 秒，窗口编号为 `event_window_id=0..22`。全体数据含 28,819 个窗口和 1,253 个 events。

关键实现：

| 角色 | 程序 / 模块 | 作用 |
| --- | --- | --- |
| 事件窗口语义 | `src/daily_multimodal/alignment/event_windows.py` | 维护窗口起止、event ID、sample ID 与事件成员关系。 |
| window 索引 | `index/eeg_aligned_window_index.jsonl` | 每个窗口的 sample/event/subject/day、fatigue、模态可用性及窗口编号。 |
| split | `/vePFS-0x0d/DailyEEG/splits_new/{protocol}/` | `cross_day`、`within_subject_day`、`cross_subject` 的窗口 leaf split。 |

### 2.2 三种 256D 上游 token

本轮不重训 encoder；它把下列已冻结的输入读取为固定的 256D token。`eeg_eegpt_partial_ft_v1` 是 fatigue-supervised control：EEGPT partial fine-tuning 的监督严格限制在相应 protocol 的 train/validation 叶，故 EMA-bag metadata 显式记录 `mixed_with_fatigue_supervised_controls:eeg_eegpt_partial_ft_v1`。

| 模态 | 从原始数据到 embedding 的实现 | 本轮读取的产物 |
| --- | --- | --- |
| EEG | `scripts/embeddings/34_run_eeg_encoder_matrix.py` 调用 `src/daily_multimodal/training/eeg_encoder_matrix.py`。EEGPT 从 EEG 10 秒窗口生成 `eegpt_partial_ft_v1` 256D token。 | `/vePFS-0x0d/DailyEEG_multimodal/embeddings/eeg_encoder_256d_tokens/{protocol}/eegpt_partial_ft_v1/seed_240800.npz` |
| Wear | `scripts/embeddings/15_extract_wear_embeddings.py` 调用 `src/daily_multimodal/embeddings/wear_real.py`。它从窗口内 PPG、GSR、ACC 计算 physiologic representation 并投影到 256D。 | `/vePFS-0x0d/DailyEEG_multimodal/embeddings/wear/wear_physio_preprocessed_eeg23win_embeddings.npz` |
| Video | `scripts/embeddings/27_extract_dinov2_roi_embeddings.py` 调用 `src/daily_multimodal/embeddings/dinov2_roi.py`。它读取每个窗口的 2× face ROI video，使用 frozen DINOv2-base 汇总为 256D。 | `/vePFS-0x0d/DailyEEG_multimodal/embeddings/video/video_A1_2xroi_eeg23win_embeddings.npz` 或 `video_B0_2xroi_eeg23win_embeddings.npz` |

本报告的三条路线都是 `no_audio`：audio 的 token 槽保留在统一 `(4,256)` 契约中，但其 `modality_mask=0`，不参与 adapter、attention 或损失。

### 2.3 23-window EMA bag 的构建与审计

`scripts/daily_affect/73_build_daily_affect_bags.py` 调用 `src/daily_multimodal/daily_affect/ema_bags.py`，按 `sample_id` 对齐 EEG、Wear、Video token；按 `event_window_id` 排序；检查同一 event 的 fatigue 标签一致；最后写出：

```text
one event = tokens (23,4,256) + modality_mask (23,4) + one fatigue label
```

每个 protocol × downstream seed 的输出目录为：

```text
outputs/daily_affect_scalar_regression_20260908/
  v2_partialft_noaudio_bags/
    {protocol}/{route}/seed_{seed}/
      ema_bags.npz
      ema_bag_manifest.jsonl
      bag_build_report.json
```

`scripts/daily_affect/90_run_daily_affect_scalar_regression.py --stage preflight` 读取 9 个初始 bag 并写入：

```text
outputs/daily_affect_scalar_regression_20260908/
  v2_partialft_noaudio/preflight/
    event_split_audit.csv
    manifest.json
```

审计确认所有 bag 的 shape 均为 `(1253,23,4,256)`，`modality_mask` 为 `(1253,23,4)`，event ID 无重复，train/validation/test event 无重叠。

## 3. 共同标量回归与融合骨架

### 3.1 训练、选择和评价

`src/daily_multimodal/daily_affect/regression_training.py` 只用 train events 拟合 token normalization 统计量和 fatigue 的均值/标准差。目标为标准化疲劳值 `z`，所有模型以：

```text
L_main = mean((z_hat - z)^2)
```

训练。预测逆变换到原始 1--5 fatigue 量尺后计算 raw r、RMSE、MAE 与 within-subject centered r。最大 80 epoch、patience 15，以 validation **event-level RMSE** 保存 checkpoint；test raw r 不参与候选选择。

### 3.2 0814 窗口融合器的匹配基线

`window_attention_regression_full_mean` 是 0814 窗口路线在 EMA-event 公平比较中的对应实现，代码在 `src/daily_multimodal/daily_affect/regression.py`：

```text
each valid 10-second window
  (up to four 256D modality tokens)
  -> Linear(256,128) + modality embedding
  -> single-head modality self-attention
  -> learnable-query pooling
  -> LayerNorm + MLP + scalar head
  -> window prediction

one EMA event
  -> mean(valid window MSE) during training
  -> mean(valid 23 window predictions) at evaluation
  -> one event prediction
```

窗口之间不传递 hidden state，也不学习 event-adaptive temporal weights。与 0814 历史数字的区别是：0814 报告 window-level raw r；本报告先把同一 EMA 的 23 个窗口预测 `full-mean`，再在 event level 计算 raw r。因此 0814 的单 seed `0.3985` 与本报告的 seven-seed `0.4314` 不可直接相减比较。

### 3.3 EMA-bag 变体的回归实现

EMA-bag 变体也复用上述 window-level adapter 与 modality attention，然后对 23 条 window evidence 做时间级融合。所有结构由 `regression.py` 定义，由 `90_run_daily_affect_scalar_regression.py` 调用。

- `bag_static_reg`：固定 temporal policy 汇总 evidence；不使用 state。
- `prior_uniform_reg`：使用 state-prior compatibility 修正 modality routing score，再做 uniform temporal pooling。
- 其它完整筛选过的条件包括 state、global/dynamic kernel 与 regression-difficulty variants；它们均在三 seed 全矩阵中完成，未在本报告的协议最佳表重复展开。

每个训练 run 保存：

```text
outputs/daily_affect_scalar_regression_20260908/v2_partialft_noaudio/
  runs/{protocol}/{route}/{condition}/seed_{seed}/
    config.json
    best_checkpoint.pt
    val_history.csv
    predictions.npz
    test_predictions.csv
    metrics.json
```

## 4. `within_subject_day`：窗口 full-mean 最强

### 4.1 路线配置和数据划分

```text
route          = A1_Wphysio_no_audio__eeg_eegpt_partial_ft_v1
normalization  = per_modality
adapter        = per_modality
EEG token      = .../eeg_encoder_256d_tokens/within_subject_day/eegpt_partial_ft_v1/seed_240800.npz
video token    = .../video/video_A1_2xroi_eeg23win_embeddings.npz
bag split      = train / val / test = 749 / 246 / 258 events
winner         = window_attention_regression_full_mean
```

三模态 token 先独立窗口融合，再把同一 event 的 23 个有效 window prediction 平均。该路线的 seven-seed raw r 为 `0.4314 ± 0.0105`，RMSE 为 `0.8952 ± 0.0065`。

`prior_uniform_reg` 是三 seed 全矩阵中唯一进入扩展的 EMA-bag 候选；补齐 seed 后 raw r 为 `0.4161 ± 0.0257`，相对窗口 Δraw r 为 `-0.0154 ± 0.0323`，只有 `3/7` 正向，且 ΔRMSE 为 `+0.0256`。因此这里的最佳工作形式是独立窗口 attention + event full-mean。

## 5. `cross_day`：last-30-second static EMA-bag 最强

### 5.1 路线配置和数据划分

```text
route          = A1_Wphysio_no_audio__eeg_eegpt_partial_ft_v1
normalization  = per_modality
adapter        = per_modality
EEG token      = .../eeg_encoder_256d_tokens/cross_day/eegpt_partial_ft_v1/seed_240800.npz
video token    = .../video/video_A1_2xroi_eeg23win_embeddings.npz
bag split      = train / val / test = 731 / 269 / 253 events
winner         = bag_static_reg__temporal_last_30s
```

该路线先对每个窗口做与 0814 相同的 modality attention fusion；随后保留 event 内最后 5 个窗口的有效 evidence（对应最近约 30 秒），将其权重归一化后固定汇总，再用 scalar head 输出一个 event prediction。

它的 raw r 为 `0.4091 ± 0.0314`，相对 matched window full-mean `0.3639 ± 0.0264` 的 Δraw r 为 `+0.0453 ± 0.0291`、7 个 seed 全部正向，RMSE 从 `0.9055 ± 0.0092` 降到 `0.8788 ± 0.0188`。这说明在跨日期泛化时，近期窗口的固定聚合比对完整 23-window 等权平均更有效。

## 6. `cross_subject`：state-prior EMA-bag 最强

### 6.1 路线配置和数据划分

```text
route          = B0_Wphysio_no_audio__eeg_eegpt_partial_ft_v1
normalization  = shared
adapter        = shared
EEG token      = .../eeg_encoder_256d_tokens/cross_subject/eegpt_partial_ft_v1/seed_240800.npz
video token    = .../video/video_B0_2xroi_eeg23win_embeddings.npz
bag split      = train / val / test = 767 / 222 / 264 events
winner         = prior_uniform_reg
```

`prior_uniform_reg` 先用每个窗口的 attention-fused evidence 更新 state，再由 state 与当前模态 token 的 compatibility 调整 routing score；时间层使用所有有效 23-window evidence 的均匀汇总，随后 scalar regression head 给出 event prediction。

它的 raw r 为 `0.1072 ± 0.0425`，相对窗口 full-mean `0.0441 ± 0.0390` 的 Δraw r 为 `+0.0631 ± 0.0596`、`6/7` 正向。RMSE 的增量为 `+0.0173`，仍位于预注册 `+0.02` guardrail 内；centered r 从 `0.0671` 提升至 `0.1286`。该协议的绝对相关仍低于另两个协议，故应保持独立解释。

## 7. 汇总、图形和复核脚本

| 阶段 | 程序 | 输出 |
| --- | --- | --- |
| 训练矩阵 / preflight | `scripts/daily_affect/90_run_daily_affect_scalar_regression.py` | `v2_partialft_noaudio/preflight/` 与 `v2_partialft_noaudio/runs/` |
| 配对汇总 | `scripts/daily_affect/91_summarize_daily_affect_scalar_regression.py` | `v2_partialft_noaudio/summary/run_metrics.csv`、`paired_window_deltas.csv`、`condition_summary.csv`、`summary.json`、`summary.md` |
| 图形 | `scripts/daily_affect/92_plot_daily_affect_scalar_regression.py` | `v2_partialft_noaudio/figures/raw_r_rmse_by_protocol.png`、`paired_delta_raw_r.png` |
| 回归单元测试 | `tests/test_daily_affect_scalar_regression.py` | 本地与远端均通过 `3/3` |

每个 non-window condition 与同一 protocol、同一路线、同一 downstream seed 的窗口 prediction 做身份核验后，再以 `scripts/daily_affect/91_summarize_daily_affect_scalar_regression.py` 执行 2,000 次 paired subject-day bootstrap。任何未来的新 route 应保留这一 event-level 配对和 train-only 标准化契约。
