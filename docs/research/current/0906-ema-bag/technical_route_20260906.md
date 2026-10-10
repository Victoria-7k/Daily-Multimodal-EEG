# Daily-affect 多模态 EMA-bag 标量回归技术路线

> Status: Current parallel route
> Route role: 0906 EMA-bag 标量回归；与 0814 window 并列使用
> Evidence status: Exploratory · test-guided；结构扩展参考 held-out 配对门槛，checkpoint 仅用 validation
> 研究索引：[研究文档索引](../../README.md)
> 文档更新：2026-10-10；回归实验完成快照：2026-09-10；服务器产物复核：2026-10-10
> 本文范围：fatigue 的“窗口架构 × 回归目标”与“EMA-bag 架构 × 回归目标”。三协议分别固定各自选定的 0814 embedding 配置。联合 11 情绪矩阵由[独立入口](../joint-evaluation/multiemotion_eeg_multitask_experiment_plan_20260913.md)维护。

## 1. 当前任务与结果

0906 当前采用一个标量回归 head：读取一次 EMA 评分前的 23 个重叠窗口，预测连续 fatigue score。训练使用 MSE，checkpoint 按 validation event-level RMSE 选择，主要报告 held-out event-level raw Pearson r；RMSE、MAE 与 within-subject centered r 补充描述误差和个体内关联。

本次比较统一了输入 embedding、event 集合、split、mask、下游 seed 和训练预算，检验窗口独立预测与 EMA-bag 时间聚合的区别。原五分类 head、CE/ordinal/ranking 损失和 QWK 结果保留在[五分类 / QWK 历史快照](experiments/technical_route_20260906_ordinal_snapshot_20261010.md)。这份快照中的候选排序和后续步骤属于原序数实验。

### 1.1 三 seed 全变体筛选中的候选结果

下表复现本对话讨论的三 seed 比较，seed 为 240729、240730、240731。每行的 Δraw r 为同 seed 的 EMA-bag 减窗口 full-mean，均值与标准差采用 seed 等权、总体标准差 `ddof=0`。

| 协议 | 窗口 full-mean raw r | EMA-bag 候选 | EMA-bag raw r | Δraw r | 正向 seed |
| --- | ---: | --- | ---: | ---: | ---: |
| cross_day | 0.3849 ± 0.0189 | `bag_static_reg__temporal_last_30s` | 0.4240 ± 0.0105 | +0.0391 | 3/3 |
| cross_subject | 0.0230 ± 0.0302 | `prior_uniform_reg` | 0.1076 ± 0.0228 | +0.0846 | 3/3 |
| within_subject_day | 0.4288 ± 0.0117 | `prior_uniform_reg` | 0.4318 ± 0.0237 | +0.0029 | 2/3 |

三 seed 用于筛选候选；协议级保留决策以随后完成的七 seed 配对验证为依据。单次原 0814 实验的窗口级 r 与本表的 EMA event-level r 使用不同聚合口径，分别记录。

### 1.2 七 seed 配对验证与保留路线

七 seed 为 240729..240735。表中“保留”表示本轮候选验证后的协议内决策，覆盖范围限于本节固定的输入组合和已扩展候选。

| 协议 | 保留工作路线 | raw r | RMSE | centered r | 相对匹配窗口 Δraw r | 正向 seed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| within_subject_day | `window_attention_regression_full_mean` | 0.4314 ± 0.0105 | 0.8952 ± 0.0065 | 0.1964 ± 0.0113 | reference | — |
| cross_day | `bag_static_reg__temporal_last_30s` | 0.4091 ± 0.0314 | 0.8788 ± 0.0188 | 0.2217 ± 0.0225 | +0.0453 ± 0.0291 | 7/7 |
| cross_subject | `prior_uniform_reg` | 0.1072 ± 0.0425 | 0.9405 ± 0.0429 | 0.1286 ± 0.0424 | +0.0631 ± 0.0596 | 6/7 |

匹配窗口基线的七 seed raw r：cross_day 为 0.3639 ± 0.0264，cross_subject 为 0.0441 ± 0.0390。

- cross_day：last-30s 同时提高 raw r、centered r，并使 RMSE 相对窗口下降 0.0267；该配置支持近期证据聚合。
- cross_subject：prior-uniform 的 raw r 与 centered r 提高，RMSE 相对窗口增加 0.0173；保留为相关性优势候选，并同步报告尺度误差。
- within_subject_day：`prior_uniform_reg` 的七 seed raw r 为 0.4161 ± 0.0257，Δraw r = −0.0154 ± 0.0323，3/7 正向，ΔRMSE = +0.0256；按既定配对门槛保留窗口 full-mean。
- cross_day 的 `bag_static_reg__temporal_uniform` 已扩到七 seed，raw r = 0.3484 ± 0.0416，Δraw r = −0.0154 ± 0.0275，2/7 正向；完整两分钟等权 bag 汇总保持为对照。

`91` 对每个 protocol × condition × seed 做 2,000 次 subject-day cluster paired bootstrap。cross_day last-30s 的 bootstrap Δr 均值 7/7 正向，cross_subject prior-uniform 为 6/7；两者各有 1/7 seed 的 95% CI 下界大于零。因此上述结论表述为多 seed 的方向一致性，统计区间仍跨零的 seed 如实保留。`condition_summary.csv` 的 `mean_bootstrap_ci_low/high` 是各 seed 区间端点的平均值，不能用作合并七 seed 的置信区间。

## 2. 输入 embedding 与数据契约

### 2.1 按协议固定的输入组合

本轮复用本对话从 0814 结果中选定的各协议 embedding 路线，整个下游矩阵中冻结 encoder。上游 EEG token seed 固定为 240800，与下游训练 seed 分开记录。

| 协议 | route_id | EEG | Wear | Video | Audio | normalization / adapter |
| --- | --- | --- | --- | --- | --- | --- |
| cross_day | `A1_Wphysio_no_audio__eeg_eegpt_partial_ft_v1` | EEGPT partial FT，256D | Wphysio，256D | A1，256D | 关闭 | per_modality / per_modality |
| within_subject_day | `A1_Wphysio_no_audio__eeg_eegpt_partial_ft_v1` | EEGPT partial FT，256D | Wphysio，256D | A1，256D | 关闭 | per_modality / per_modality |
| cross_subject | `B0_Wphysio_no_audio__eeg_eegpt_partial_ft_v1` | EEGPT partial FT，256D | Wphysio，256D | B0，256D | 关闭 | shared / shared |

A1 为 2× 主脸 ROI 的 DINOv2-base 表征，带轻量颜色/亮度增强；B0 使用基础 ROI 表征。Wear 从窗口内 PPG、GSR、ACC 构建生理特征并投影。EEG 使用真正的 encoder pooled hidden state 经 projection 输出的 256D token。

EEGPT partial FT 属于 fatigue-supervised control，原上游按对应窗口 protocol 的 train/validation 边界训练和选型。本轮 bag metadata 为：

```text
mixed_with_fatigue_supervised_controls:eeg_eegpt_partial_ft_v1
```

该监督来源必须与窗口对照共同保留。三协议输入均为 EEG + Wear + Video；统一张量中的 Audio 槽保留，mask 恒为 0。

### 2.2 从原始窗口到冻结 token

本节路径以远端项目为准：

```text
项目根 P = /vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned
embedding 根 E = /vePFS-0x0d/DailyEEG_multimodal/embeddings
window split 根 S = /vePFS-0x0d/DailyEEG/splits_new
```

| 阶段 | 脚本与实现 | 本轮实际读取的产物 |
| --- | --- | --- |
| 事件 / 窗口定义 | [event_windows.py](../../../../src/daily_multimodal/alignment/event_windows.py) | `P/index/eeg_aligned_window_index.jsonl` |
| EEG encoder 与 256D 导出 | [34_run_eeg_encoder_matrix.py](../../../../scripts/embeddings/34_run_eeg_encoder_matrix.py)；[eeg_encoder_matrix.py](../../../../src/daily_multimodal/training/eeg_encoder_matrix.py) | `E/eeg_encoder_256d_tokens/{protocol}/eegpt_partial_ft_v1/seed_240800.npz` |
| Wear 生理表征 | [15_extract_wear_embeddings.py](../../../../scripts/embeddings/15_extract_wear_embeddings.py)；[wear_real.py](../../../../src/daily_multimodal/embeddings/wear_real.py) | `E/wear/wear_physio_preprocessed_eeg23win_embeddings.npz` |
| Video ROI / DINOv2 表征 | [27_extract_dinov2_roi_embeddings.py](../../../../scripts/embeddings/27_extract_dinov2_roi_embeddings.py)；[dinov2_roi.py](../../../../src/daily_multimodal/embeddings/dinov2_roi.py) | `E/video/video_A1_2xroi_eeg23win_embeddings.npz` 或 `video_B0_2xroi_eeg23win_embeddings.npz` |
| EMA bags | [73_build_daily_affect_bags.py](../../../../scripts/daily_affect/73_build_daily_affect_bags.py)；[ema_bags.py](../../../../src/daily_multimodal/daily_affect/ema_bags.py) | `P/outputs/daily_affect_scalar_regression_20260908/v2_partialft_noaudio_bags/{protocol}/{route_id}/seed_{seed}/ema_bags.npz` |

### 2.3 Canonical EMA bag 与划分边界

全数据有 28,819 个窗口、1,253 个 EMA events。每个 event 包含 23 个重叠的 10 秒窗口，stride 为 5 秒，按 `event_window_id=0..22` 从早到晚排列，覆盖评分前 120 秒。模态顺序固定为 `[EEG, Wear, Video, Audio]`。

```text
tokens: (N_event, 23, 4, 256)
modality_mask: (N_event, 23, 4)
label: (N_event,)，原始 fatigue score 1..5，作为数值回归目标
```

bag builder 按 sample_id 对齐 token，检查 event 标签一致性，并保存 `sample_id_matrix`、`ema_bag_manifest.jsonl` 与 `bag_build_report.json`。mask 决定有效模态；无有效模态的窗口从时间池化中排除。

| 协议 | train events | val events | test events | train 内 leaf 合并 events | window-majority 边界投影 events |
| --- | ---: | ---: | ---: | ---: | ---: |
| cross_day | 731 | 269 | 253 | 42 | 0 |
| within_subject_day | 749 | 246 | 258 | 0 | 214 |
| cross_subject | 767 | 222 | 264 | 42 | 0 |

上述计数来自本轮 `bag_build_report.json`。bag split 使用 `single_leaf_or_train_leaf_merge_or_window_majority`：单一 leaf 直接保留，pretrain/finetune/train 可在训练并集合并，跨 train/val/test 的 event 按窗口数多数票归属；平票优先顺序为 train、val、test。构建后 train/val/test event ID 两两无交集。

`within_subject_day` 对应同一受试者、同一天内的既有窗口 holdout；214 个 event 涉及多数票投影。event ID 无交集这项检查覆盖 bag 成员边界，上游监督仍遵循原窗口叶边界，不能据此推出原始信号或上游标签在投影后也严格隔离。本表结论限定为该现有协议；整日时间顺序泛化需使用另行审计的 `date_in_order`。

原比较计划要求排除跨 train/val/test leaf 的 event；本轮实际 builder 使用上述多数票投影，within_subject_day 因此与原计划的严格单-leaf 合同有差异。本文按实际产物报告该协议，严格排除边界 event 的版本尚未在本轮执行。

## 3. 共同标量目标与训练设置

实现入口为 [regression.py](../../../../src/daily_multimodal/daily_affect/regression.py) 和 [regression_training.py](../../../../src/daily_multimodal/daily_affect/regression_training.py)。

token normalization 与目标标准化均只在 bag train events 拟合。per_modality 分别拟合各有效模态统计量并使用独立 adapter；shared 使用共享统计量和共享 adapter。两者都加入可学习 modality embedding。

```text
z = (y − μ_train) / σ_train
ŷ = μ_train + σ_train × z_hat

EMA-bag 主损失:
L_bag = mean_event[(z_hat_event − z_event)^2]

窗口匹配基线主损失:
L_window = mean_event[mean_valid_window((z_hat_window − z_event)^2)]
```

窗口基线向有效窗口复制 event 标签，先对每 event 的有效窗口 MSE 求平均，再对 events 平均，保持 event 权重一致。EMA-bag 在时间聚合后对每 event 监督一次。两条架构保留各自的监督方式，并共享标量目标量尺。

回归 difficulty 变体另加 `0.1 × Gaussian probe NLL`，`objective_id=scalar_mse__gaussian_probe_nll_0.1`；其余条件为 `scalar_mse`。这项辅助目标只启用在第 4 节注明的 difficulty 模型中。

| 项目 | 本轮配置 |
| --- | --- |
| hidden dimension | 128 |
| 回归 head | LayerNorm → Linear(128,128) → GELU → Dropout → Linear(128,1) |
| optimizer | AdamW，lr = 1e-3，weight_decay = 1e-4 |
| batch / epoch / patience | 128 events / 最多 80 epochs / 15 |
| dropout / modality dropout | 0.1 / 0.1 |
| difficulty schedule | probe warm-up 5 epochs，随后 ramp 5 epochs；λ_D = 0.25，默认 detach |
| checkpoint 选择 | `val_rmse_min`，event-level |
| 评价 | 逆标准化后的 raw r、RMSE、MAE、within-subject centered r；raw 预测保留，不裁剪、取整或做 test calibration |

raw r 是本轮主报告读数；训练目标是 MSE，checkpoint 选择是 validation RMSE，三者分别记录。窗口基线与所有 EMA-bag 条件均在同一 test events 上评价。

## 4. 从窗口融合到 EMA-bag 的结构

### 4.1 窗口 full-mean 与 bag_static

这两种结构共享窗口内融合：

```text
可用 256D modality tokens
→ Linear(256,128) + modality embedding
→ 单头 modality self-attention
→ learnable-query pooling
→ 128D window evidence e_t
```

窗口基线 `window_attention_regression_full_mean` 对应 `model_id=window_replicated`：每个有效窗口先通过标量 head，再把 23 个窗口预测等权平均为 event prediction。

`bag_static_reg` 先按固定时间权重平均 window evidence，再通过同一个形式的标量 head：

```text
窗口 full-mean: ŷ_event = mean_t head(e_t)
EMA-bag static: ŷ_event = head(sum_t β_t e_t)
```

head 含非线性，两种计算顺序形成不同模型。窗口 full-mean 复用 0814 的窗口 attention 架构，同时接入本轮的 23-window EMA bag、event split、目标标准化、event-balanced window MSE 和 event-level 评价。该结果是匹配比较中的窗口参考，原 0814 历史窗口级结果另行保留。

cross_day 保留的 last-30s 使用 index 18..22，共 5 个重叠窗口，其并集覆盖评分前 30 秒；有效 evidence 固定等权汇总，无 GRU、prior 或 learned time kernel。

### 4.2 state 与 prior 路由

state/prior/kernel 变体采用线性 evidence scorer 进行窗口内融合。其原生计算结构为：

```text
a_t,m = adapter(x_t,m) + modality_embedding_m
α_t,m = masked_softmax_m[baseScore(a_t,m)]
e_t = sum_m α_t,m a_t,m
s_t = GRUCell(e_t, s_t−1)，每个 event 的 s_−1 = 0
```

`state_uniform_reg` 均匀平均有效 GRU states。state 在一个 event 内沿 23 个窗口递推，每个 event 重新初始化。

`prior_uniform_reg` 在计算当前模态权重前，引入上一状态：

```text
p_t = LayerNorm(s_t−1 + transition(s_t−1))
c_t,m = g(tanh(Wq p_t + Wm a_t,m))
α_t,m = masked_softmax_m[baseScore(a_t,m) + c_t,m]
e_t → GRU state s_t → 有效 states 等权平均 → scalar head
```

这也是 cross_subject 保留配置的完整融合流程。`prior_uniform_reg` 使用线性 scorer + prior compatibility + GRU，不启用窗口 self-attention/query pooling，也不启用 difficulty probe。state 路线与 bag_static 的对比同时改变窗口融合与时间状态结构，机制解释应覆盖两项变化。

### 4.3 11 个原生 EMA-bag 变体及回归难度

| 原生 model_id | 回归 condition_id | state | prior | Gaussian difficulty | 时间汇总 |
| --- | --- | --- | --- | --- | --- |
| bag_static | `bag_static_reg__temporal_{policy}` | — | — | — | 8 种固定 policy |
| state_uniform | `state_uniform_reg` | GRU | — | — | uniform |
| prior_uniform | `prior_uniform_reg` | GRU | 有 | — | uniform |
| prior_ordD_uniform | `prior_regD_uniform_reg` | GRU | 有 | 有 | uniform |
| global_kernel_no_prior | `global_kernel_no_prior_reg` | GRU | — | — | 全局 learned kernel |
| dynamic_kernel_no_prior | `dynamic_kernel_no_prior_reg` | GRU | — | — | event-adaptive kernel |
| dynamic_kernel_prior_uniform | `dynamic_kernel_prior_uniform_reg` | GRU | 有 | — | event-adaptive kernel |
| dynamic_kernel | `dynamic_kernel_reg` | GRU | 有 | 有 | event-adaptive kernel |
| dynamic_fixed_short | `dynamic_fixed_short_reg` | GRU | 有 | 有 | fixed short |
| dynamic_fixed_medium | `dynamic_fixed_medium_reg` | GRU | 有 | 有 | fixed medium |
| dynamic_fixed_long | `dynamic_fixed_long_reg` | GRU | 有 | 有 | fixed long |

`prior_ordD_uniform` 的底层 model_id 为兼容既有结构而保留。回归版每模态 probe 根据整个 bag 的有效 adapted tokens 均值预测 Gaussian mean 与 log variance，difficulty 为 `sigmoid(log_variance)`，路由分数减去 `λ_D × difficulty`；它替换原五分类的 entropy/ordinal-variance 设计。

固定 policy 为 `uniform,last_10s,last_30s,last_60s,first_30s,kernel_short,kernel_medium,kernel_long`；近期窗口分别选 index 22、18..22、12..22，first-30s 选 0..4。指数核时间常数为 15、45、120 秒，全局核学习共享混合，动态核按 event 状态学习混合。`dynamic_kernel_prior_uniform` 实际使用动态时间核，尾部 `uniform` 是原生路由命名。

## 5. 已执行矩阵与证据范围

初始矩阵全部完成：

```text
1 个窗口基线
+ 11 个原生 EMA-bag 结构（含 bag_static uniform）
+ 7 个额外 bag_static 时间 policy
= 每 protocol × seed 19 条件

3 protocols × 3 seeds × 19 conditions = 171 runs
```

因此第 4.3 节的所有原生变体及 8 个 bag_static 固定时间规则都完成了三协议三 seed 标量回归测试。

扩展到七 seed 的组合为 cross_day 的 bag_static uniform / last-30s、within_subject_day 的 prior-uniform、cross_subject 的 prior-uniform，以及三个协议的匹配窗口基线。扩展新增 28 runs，总计 199 个 `metrics.json`、178 条 window-paired 比较。其余条件保持三 seed；七 seed 表给出已扩展候选的验证结果。

原计划的三 seed gate 关注 Δraw r > 0、至少 2/3 正向、平均 ΔRMSE ≤ +0.02、centered r / prediction scale 和审计项；超过候选数量上限时按 validation raw r 排序。七 seed gate 关注 Δraw r > 0、至少 5/7 正向、RMSE guard 和 paired bootstrap。结构的扩展和保留参考了 held-out 配对表现，结果定位为 exploratory/test-guided；新增 seed 验证训练随机性，保持原 test events。第 1 节另列 bootstrap 区间事实，避免把方向性通过等同于多数 seed 的区间排除零。

## 6. 执行入口与产物路径

### 6.1 脚本导航

| 角色 | 入口 |
| --- | --- |
| 构建、对齐与审计 bags | [73_build_daily_affect_bags.py](../../../../scripts/daily_affect/73_build_daily_affect_bags.py) |
| preflight / smoke / 全结构回归矩阵 | [90_run_daily_affect_scalar_regression.py](../../../../scripts/daily_affect/90_run_daily_affect_scalar_regression.py) |
| metrics 汇总与 subject-day paired bootstrap | [91_summarize_daily_affect_scalar_regression.py](../../../../scripts/daily_affect/91_summarize_daily_affect_scalar_regression.py) |
| 回归结果可视化 | [92_plot_daily_affect_scalar_regression.py](../../../../scripts/daily_affect/92_plot_daily_affect_scalar_regression.py) |
| 模型 / state / time kernels | [regression.py](../../../../src/daily_multimodal/daily_affect/regression.py)；[affect_state_filter.py](../../../../src/daily_multimodal/daily_affect/affect_state_filter.py)；[dynamic_ema_kernel.py](../../../../src/daily_multimodal/daily_affect/dynamic_ema_kernel.py) |
| 训练、checkpoint、指标与预测保存 | [regression_training.py](../../../../src/daily_multimodal/daily_affect/regression_training.py) |
| 回归契约测试 | [test_daily_affect_scalar_regression.py](../../../../tests/test_daily_affect_scalar_regression.py) |
| 原实验合同 | [scalar regression comparison plan](experiments/daily_affect_scalar_regression_comparison_plan_20260908.md) |

当前 evidence root 与初始 full/label-free screen 分开：

```text
P/outputs/daily_affect_scalar_regression_20260908/
  v2_partialft_noaudio_bags/
    {protocol}/{route_id}/seed_{seed}/
      ema_bags.npz
      ema_bag_manifest.jsonl
      bag_build_report.json
  v2_partialft_noaudio/
    preflight/
      event_split_audit.csv
      manifest.json
    runs/{protocol}/{route_id}/{condition_id}/seed_{seed}/
      config.json
      metrics.json
      best_checkpoint.pt
      predictions.npz
      test_predictions.csv
    summary/
      run_metrics.csv
      paired_window_deltas.csv
      condition_summary.csv
      summary.json
      summary.md
```

2026-10-10 直接登录 `huoshan_TriDim` 核验上述 bag report、config 与 metrics，并重算第 1 节三/七 seed 指标。早期 `outputs/daily_affect_scalar_regression_20260908/runs` 使用另一套 full/label-free 输入，与本文 v2 结果分别解释。

### 6.2 本轮三 seed 矩阵的显式参数

以下为远端项目根中的 Bash 复现示例，本次文档更新没有启动训练。已有 bags 保持只读。先检查 `--stage preflight --device cpu`，需要执行矩阵时才改为 `--stage matrix --device cuda --skip-existing`。

```bash
cd /vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned
PYTHONPATH=src runtime/envs/eegpt-gpu-min/bin/python \
  scripts/daily_affect/90_run_daily_affect_scalar_regression.py \
  --stage preflight --device cpu \
  --bags-root outputs/daily_affect_scalar_regression_20260908/v2_partialft_noaudio_bags \
  --out-root outputs/daily_affect_scalar_regression_20260908/v2_partialft_noaudio \
  --protocols cross_day,within_subject_day,cross_subject \
  --seeds 240729,240730,240731 \
  --route-map cross_day=A1_Wphysio_no_audio__eeg_eegpt_partial_ft_v1,within_subject_day=A1_Wphysio_no_audio__eeg_eegpt_partial_ft_v1,cross_subject=B0_Wphysio_no_audio__eeg_eegpt_partial_ft_v1 \
  --normalization-map cross_day=per_modality,within_subject_day=per_modality,cross_subject=shared
```

`90` 的无参数默认值仍指向早期 full 输入和 `date_in_order`，复现本文应完整传入上述 bags/out/protocol/route/normalization 参数。七 seed 补齐只对第 5 节列出的候选执行，避免把尚未完成的全变体七 seed 写成已有结果。

汇总与画图示例：

```bash
PYTHONPATH=src runtime/envs/eegpt-gpu-min/bin/python \
  scripts/daily_affect/91_summarize_daily_affect_scalar_regression.py \
  --runs-root outputs/daily_affect_scalar_regression_20260908/v2_partialft_noaudio/runs \
  --out-dir outputs/daily_affect_scalar_regression_20260908/v2_partialft_noaudio/summary \
  --bootstrap-replicates 2000
PYTHONPATH=src runtime/envs/eegpt-gpu-min/bin/python \
  scripts/daily_affect/92_plot_daily_affect_scalar_regression.py \
  --summary-dir outputs/daily_affect_scalar_regression_20260908/v2_partialft_noaudio/summary
```

本轮决策：within_subject_day 使用匹配窗口 full-mean，cross_day 使用静态 last-30s EMA-bag，cross_subject 保留 prior-uniform 的相关性优势候选。所有结果保留各自协议、上游监督、输入组合、seed 数量和 event-level 评价边界。
