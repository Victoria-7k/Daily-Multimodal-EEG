# 当前多模态疲劳预测技术路线

> 版本：2026-09-06
>
> 本文是当前可执行路线与证据状态的总览。`technical_route_20260814.md` 保留为窗口级融合矩阵的历史快照；本文以当前源码、已同步产物和已完成的配对实验为准。

## 1. 技术路线总览

项目目前同时维护三条严格分离的工作线：

| 工作线 | 监督单位与目标 | 当前地位 | 核心产物 |
| --- | --- | --- | --- |
| 窗口级疲劳回归主线 | 一个 EEG-aligned 10 秒窗口，预测连续 `fatigue` | 主线 | `scripts/window_fatigue/32_run_eegpt_centered_loss.py` |
| Daily-affect ordinal 候选线 | 一个 EMA event 的 23 个窗口，预测 `1..5` 级疲劳 | 已完成 focused 7-seed 候选证据 | `scripts/73` 至 `83` |
| EQL-CAF temporal-token 探索线 | 一个窗口内的 5 个 2 秒 temporal token | 真实 token 前置条件尚未满足 | `scripts/57` 至 `64` |

三条线共享 canonical EEG-aligned index 与窗口级四模态 embedding；训练样本单位、标签、模型、输出目录和 promotion gate 分开维护。EQL-CAF 的 `token_mask` 不进入 daily-affect；daily-affect 的 `(N_ema,23,4)` `modality_mask` 不进入 EQL-CAF。

```text
Canonical EEG-aligned windows
  -> 10 s window-level EEG/Wear/Video/Audio 256D tokens
  -> A. Window-level fusion regression
  -> B. 23-window EMA bag ordinal prediction
  -> C. 5 x 2 s temporal-token research when real token encoders are available
```

## 2. 共同数据与实验契约

### 2.1 Canonical index

| 项目 | 当前定义 |
| --- | --- |
| 窗口数 | `28,819` |
| EMA event 数 | `1,253` |
| 窗口 | 10 秒 EEG-aligned 窗口 |
| event 内时间轴 | `event_window_id=0..22`，相邻窗口 stride 为 5 秒 |
| EEG 原始张量 | `X.npy: (28819,2000,59)`，200 Hz、59 通道 |
| 模态顺序 | `[eeg, wear, video, audio]` |
| 窗口级 token | 每个可用模态为 256D，`modality_mask (N,4)` 屏蔽缺失 |
| 划分 | `pretrain + finetune` 训练，`val` 早停/选型，`test` 仅作最终评估 |

主协议始终独立报告 `cross_subject`、`cross_day`、`within_subject_day`。窗口级回归把 `cross_day` 与 `within_subject_day` 作为主要证据场景；daily-affect 在三种协议上使用同一 EMA-bag 构建契约。

### 2.2 表征监督边界

`eegpt_frozen_v1` 与 frozen 外部 encoder 是表征对照。`eegpt_partial_ft_v1`、`Wmoment_*` 的 projection 或部分微调均只使用对应 protocol 的 train/val fatigue 标签，因此在文档、路径和比较中标注为 fatigue-supervised representation。融合器与 daily-affect head 的监督范围同样止于 train/val。

## 3. 工作线 A：窗口级连续疲劳回归主线

### 3.1 表征与融合

| 模态 | 当前路线 | 状态 |
| --- | --- | --- |
| EEG | `eeg_eegpt_partial_ft_v1`：EEGPT 后部 block、norm 与 projection/head 的 partial fine-tune，导出 256D token | 当前主线 EEG 表征 |
| Wear | `Wphysio`：HR/HRV、SCR/slope、运动等可解释特征；`Wdeep`：固定序列特征；`Wmoment_frozen`：MOMENT-1-small frozen 加监督投影 | Wmoment frozen 为已验证的表示候选，仍需在当前主线分支上做匹配确认 |
| Video | `B0/A1/A2`：2 倍主脸 ROI 的 DINOv2-Base frozen 256D token；A1/A2 采用不同增强 | 有有效信号，跨日覆盖与漂移需继续控制 |
| Audio | 窗口化音轨的 openSMILE eGeMAPS functionals 投影为 256D | 由 route 的 full/no-audio 配对决定是否保留 |

当前主线 fusion 是 `AttentionRegressor` 的 modality-token attention：每个 256D token 线性投影到 `hidden_dim=128`，加入 modality embedding，经单头 attention 和 learnable-query pooling 后进入 `LayerNorm + MLP` regression head。token normalization 仅在训练 split 拟合；`shared` 与 `per_modality` 都是可复现实验配置，其中跨日的 `per_modality` 保留为候选，不能以单次 screen 替换全局默认。

训练以回归损失与项目既有的 raw/centered 配置运行。评价固定报告 RMSE、raw Pearson r 和 within-subject centered r：raw r 描述总体等级相关，centered r 描述个体内波动相关。

### 3.2 当前已核验的结果锚点

以下为历史 window-level matrix 的可复现参考点，用于说明主线能力边界；它们保留各自 seed 与 route 口径，不能与 daily-affect QWK 合并排名。

| protocol | 选择口径 | `eeg_eegpt_partial_ft_v1` 路线 | RMSE | raw r | centered r |
| --- | --- | --- | ---: | ---: | ---: |
| `cross_day` | 最低 RMSE | `B0_Wphysio_no_audio` | 0.9189 | 0.3301 | 0.1526 |
| `cross_day` | 最高 raw r | `B0_Wphysio_full` | 0.9481 | 0.3504 | 0.1176 |
| `within_subject_day` | 最低 RMSE | `A2_Wdeep_full` | 0.9138 | 0.4046 | 0.1736 |
| `within_subject_day` | 最高 raw r | `B0_Wdeep_no_audio` | 0.9337 | 0.4252 | 0.2122 |

`eeg_eegpt_partial_ft_v1` 的 EEG-only 参考值为：`cross_day` RMSE `0.9272`、raw r `0.2741`、centered r `0.1005`；`within_subject_day` 为 `0.9270`、`0.3749`、`0.1489`。

### 3.3 当前主线决策

`eeg_eegpt_partial_ft_v1 + modality-token attention` 保持为窗口级主线。concat 在跨路线平均上有正向证据，但在主线 EEG 分支上的 matched evidence 尚不足以替代 attention；因此 fusion default 的变更需要在 `eeg_eegpt_partial_ft_v1`、主协议、相同 token 与相同 seed 下完成 paired multi-seed gate。

## 4. 工作线 B：Daily-affect EMA-bag ordinal 候选线

### 4.1 输入、标签与模型

一个 EMA event 是一个监督样本：

```text
23 个 EEG-aligned 10 s 窗口
-> tokens (23,4,256) + modality_mask (23,4)
-> event-level ordinal fatigue label 1..5
```

`73_build_daily_affect_bags.py` 按 event 构建 `tokens (1253,23,4,256)`、窗口级 `modality_mask`、bag-level split 和可回溯的 `sample_id_matrix`。若 event 的窗口 split 出现边界，构建器显式记录其投影规则；模型只以 bag-level train/val 选型。

公平的静态对照由 `74` 成对训练：

- `window_replicated`：对有效窗口复制同一 event 标签训练，测试时汇聚为 event 预测；
- `bag_static`：对 23 个窗口进行时间池化后，以一个 event 标签训练；
- 二者共享窗口 attention/query pooling、adapter、normalization、class-weighted CE、cumulative ordinal loss 与 within-subject ranking loss。

候选 `dynamic_kernel_prior_uniform` 在每个时间步以 latent fatigue state 与当前各模态 token 的 compatibility 形成 routing score，再以 learned short/medium/long EMA kernel 混合 23 个窗口。此模型不启用 ordinal-difficulty penalty。`global_kernel_no_prior` 是无 state-prior 的 global temporal-kernel 对照；无 state-prior 路径的 difficulty routing 统一解析为 `none`。

### 4.2 Focused 7-seed 结果

场景固定为 `cross_day / A1_Wphysio_full / per_modality normalization / per_modality adapter`。所有比较均与同 protocol、route、objective、adapter 和 seed 的 `bag_static` 配对；每 seed 的 bootstrap 以 subject-day 为重采样块。

| 模型 | clean QWK | clean Macro-F1 | clean ordinal MAE | 相对 static 的 ΔQWK | 相对 static 的 ΔMAE | QWK 胜出 seed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bag_static` | 0.1864 | 0.2592 | 0.9571 | - | - | - |
| `dynamic_kernel_prior_uniform` | 0.2515 | 0.2732 | 0.9221 | `+0.0651 +/- 0.0656` | `-0.0350 +/- 0.0650` | 6/7 |
| `global_kernel_no_prior` | 0.2230 | 0.2561 | 0.9526 | `+0.0366 +/- 0.0716` | `-0.0045 +/- 0.0556` | 6/7 |

`dynamic_kernel_prior_uniform` 是 daily-affect 的优先候选。它在 7 个 seed 中建立了相对 static baseline 的一致性优势；当前 direct no-prior dynamic control 仍只有最初 3 个 seed，因此 state-prior compatibility 的独立因果效应需在该对照补至同一 seed 集合后再作正式判定。

### 4.3 与窗口主线的 matched event bridge

为在同一 event 单位上比较窗口回归与 daily-affect，新增 `cross_day / A1_Wphysio_full / eeg_eegpt_partial_ft_v1 / per_modality normalization` 的三 seed bridge。每个测试 event 均要求 `23/23` 个窗口来自 `test`，验证 event 同理要求 `23/23` 个窗口来自 `val`；因此窗口侧的 `full_mean` 聚合和 daily-affect 输出面对同一 `253` 个测试 event。窗口回归先在验证 event 上独立拟合 isotonic 校准，再仅在测试 event 上报告。

| 系统 | event 级 QWK | expected raw r | expected centered r | expected RMSE |
| --- | ---: | ---: | ---: | ---: |
| Window regression + `full_mean` | 0.2093 | 0.3660 | 0.1669 | 0.9092 |
| `bag_static` | 0.2350 | 0.3020 | 0.1368 | 1.0038 |
| `dynamic_kernel_prior_uniform` | 0.1759 | 0.2695 | 0.0937 | 1.0346 |

相对窗口 `full_mean` 的配对结果中，`bag_static` 的 native QWK 差值为 `+0.0257 +/- 0.0868`（`1/3` seed 胜出），expected raw-r 差值为 `-0.0640 +/- 0.0084`（`0/3`），expected RMSE 差值为 `+0.0946 +/- 0.0254`（`0/3`）。`dynamic_kernel_prior_uniform` 的对应差值为 QWK `-0.0334 +/- 0.0381`（`1/3`）、raw r `-0.0965 +/- 0.0367`（`0/3`）和 RMSE `+0.1254 +/- 0.0747`（`0/3`）。各 seed 的 QWK subject-day bootstrap 区间均覆盖零。

这组 bridge 把原先不可直接对照的两条输出放到相同 token、split、seed 和 event 集合上。它表明当前 partial-FT 分支的窗口回归保留了更强的连续评分关联与更低误差；`bag_static` 具有小幅平均 QWK 差异，尚未形成 seed 一致的序数优势。`dynamic_kernel_prior_uniform` 的既有 7-seed 结论来自 frozen EEG token 分支，继续作为该分支内 static 对照的证据；本段 partial-FT bridge 单独记录，三 seed 仅作描述性证据。

同一 bridge 的 dynamic-range audit 量化了图中的范围扩展：窗口、static 和 dynamic 的 prediction P10--P90 分别为 `1.0524 +/- 0.1559`、`1.7484 +/- 0.2530` 和 `1.9567 +/- 0.2408`，daily 相对 window 在每个 seed 的跨度差均为正，subject-day bootstrap 区间也均高于零。raw r 同时从窗口的 `0.3660 +/- 0.0280` 降至 static 的 `0.3020 +/- 0.0214` 和 dynamic 的 `0.2695 +/- 0.0107`。static 的 label-mean span 增加到 `0.8370`，表明 ordinal score 的平均等级分离增强；其 within-label SD 也从窗口的 `0.4106` 升至 `0.6371`，使 label-explained prediction variance 从 `0.1375` 降至 `0.0937`。dynamic 的 within-label SD 为 `0.7025`、label-explained variance 为 `0.0760`。因此范围扩展主要伴随同等级内的 event 波动扩张；它不等价于连续评分精度提升。测试集的 5 级仅有 `3` 个 event，等级尾部的均值曲线仍需在扩展 seed 与尾部样本上复核。

完整记录：`outputs/server_sync/daily_affect_window_event_bridge_20260906/comparison/window_daily_event_comparison.md`。该目录的 `window_daily_affect_matched_event_series_cross_day.png` 按相同 253 个 held-out event 和三组 seed 并列显示窗口主线、`bag_static` 与 dynamic candidate 的 true label、seed 平均 expected score 和 10--90% 区间，用于直观审阅而非新增选型指标。

### 4.3.1 Expected-score Huber 辅助损失 screen

为直接检验 event-level expected score 的数值约束能否降低上述连续评分缺口，在同一 partial-FT bridge bag 上对 `bag_static` 增加 `lambda * Huber(E[p], y)`；其余 class-weighted CE、`0.5` cumulative ordinal、`0.1` within-subject ranking、QWK checkpoint selection、split、seed 和训练预算固定。`88_run_daily_affect_expected_score_huber.py` 对 `lambda={0,0.025,0.05,0.1,0.2}` 与 `delta=1.0` 完成三 seed screen，并且只依据验证集选择权重：非零权重须在至少 `2/3` seed 提升 validation raw r，且平均 validation QWK 相对零权重不低于 `-0.02`。

所有非零权重均通过验证 gate，`lambda=0.2` 以 validation raw-r delta `+0.0142`、`3/3` wins 和 validation QWK delta `+0.0148` 被锁定。测试集随后只审阅锁定候选相对零权重控制：QWK `-0.0029 +/- 0.0368`（`2/3` positive）、expected raw r `+0.0031 +/- 0.0058`（`2/3` positive）、centered r `+0.0093 +/- 0.0075`（`3/3` positive）、RMSE `-0.0106 +/- 0.0170`（`2/3` lower）。预测 P10--P90 从 `1.7484` 小幅到 `1.7054`，within-label SD 从 `0.6371` 到 `0.6361`，说明该 loss 没有主要通过压缩输出范围得到连续指标变化；raw-r 增益也远小于 window-vs-static bridge 的 `0.0640` 差距。此三 seed validation-locked screen 证明 Huber 是可复现的轻量改善候选，但尚不支持替换窗口级主线或将其扩展为 dynamic-kernel 结论。完整产物位于 `outputs/server_sync/daily_affect_expected_score_huber_20260907/`。

### 4.4 Missing/corruption robustness

对 3 个 frozen model、7 个 seed、19 个条件运行了 `399` 个测试评估单元。下表展示最有决策价值的 QWK 切片：

| 条件 | `bag_static` | `dynamic_kernel_prior_uniform` | 读法 |
| --- | ---: | ---: | --- |
| clean | 0.1864 | 0.2515 | 候选有稳定的 clean 优势 |
| missing video | 0.0990 | 0.2144 | 候选在 video 缺失时保持较高性能 |
| shuffle video | 0.0778 | 0.0769 | 错配 video 的时序/身份仍是共同脆弱点 |
| shuffle wear | 0.1890 | 0.1509 | 候选对错误 Wear token 更敏感，需质量感知的降权机制 |

这一结果把 daily-affect 的下一步聚焦到 video 与 wear 的可靠性判断：正常 token 可被使用，缺失或被检测为不可靠时应平稳退回到其余模态。

### 4.5 时间核、模态权重与标签可辨识性

已有 bottleneck audit 显示，EMA 评分更符合近期 30--60 秒的加权状态：`kernel_short` 与 `kernel_medium` 相对完整两分钟均值的 3-seed ΔQWK 分别为 `+0.0677` 与 `+0.0585`，medium 为 `3/3` QWK wins；last-10s 没有稳定优势。dynamic candidate 的 7-seed 诊断中，Wear 与 Video 路由权重通常同量级，未出现 dual-shared 诊断中 Wear 单模态饱和的模式。

## 5. 解释性与稳定性工程

### 5.1 可审计输出

daily-affect 每个 run 都写入 `metrics.json`、`config.json`、checkpoint、test prediction 与 diagnostics。diagnostics 按 EMA event 保存：

- `modality_weights (N,23,4)`；
- `modality_difficulty (N,4)`；
- `temporal_weights (N,23)` 与 `kernel_mixture (N,3)`；
- cumulative Probe 的 ordinal logits 或 categorical Probe logits。

`85_plot_daily_affect_event_series.py` 还从保存的 7-seed test prediction 生成 held-out event sequence 图：同一 event 顺序上显示 true ordinal label、每个模型的 seed 平均 expected score 与 10--90% seed 区间，并以细线标记 subject-day 边界。它服务于对 event-level 预测起伏与 seed 不确定性的直观审阅，横轴不作为连续传感器时间解释。

`76` 汇总 matched seeds、方向一致性和 subject-day bootstrap CI。`77` 将多 seed 结果压缩为 diagnostics、Probe reliability、confusion 三张 atlas，避免逐 run 图的同名覆盖。

### 5.2 已定位的稳定性问题

| 问题 | 已有证据 | 当前工程位置 |
| --- | --- | --- |
| Dual-shared Wear saturation | 共享 normalization 与共享 adapter 同时启用时 Wear weight 高达 0.7363；拆开任一因素后权重回到约 0.28--0.32 | `83_run_daily_affect_routing_factorial.py`，保持两项配置显式分开 |
| Video cross-day stability | video test-event availability 为 0.6319，token centroid shift RMS 为 0.8074 | `82_run_daily_affect_bottleneck_audit.py`，下一步补 quality-aware routing |
| Wear-only FM | W3FM 的正式 Phase 3 gate 已结束，ACC 跨日稳定性与 gate/error association 是主要诊断方向 | `65` 至 `72`，作为表示研究分支 |

## 6. 工作线 C：EQL-CAF temporal-token 探索

EQL-CAF 的目标是使用窗口内 5 个 2 秒 token 建模 EEG-anchored temporal lag、quality 与 residual。它的输入契约为 `(N,4,5,256)` 与 `token_mask (N,4,5)`，和 daily-affect 的 23-window bag 契约不同。

当前 `global_repeat_smoke_v1` 将已有 10 秒 pooled token 重复到 5 个时间片，只验证 57--64 的 index、packing、mask 和训练管线。它不提供真实的窗口内时序变化，因此不参与主线或 daily-affect 的效果结论。真实 EEG/Wear/Video/Audio 2 秒 encoder 完成后，仍复用同一 packed-NPZ 契约运行 B1/B2/M1--M5。

## 7. 评价与 promotion 规则

| 线路 | 主指标 | 补充读数 | promotion 条件 |
| --- | --- | --- | --- |
| 窗口级回归 | 各主协议的 RMSE、raw r、centered r | per-subject r、误差分布 | 同 branch、同 route、同 seed 的 multi-seed paired result |
| Daily-affect | QWK、Macro-F1、ordinal MAE | `expected_score` 的 RMSE/raw r/centered r | 同 protocol/route/objective/normalization/adapter 的 matched 5--7 seed paired gate，逐 seed subject-day bootstrap |
| EQL-CAF | 待真实 2 秒 token 后定义 | smoke 仅验证契约 | 真实 temporal token 与 matched baseline 完整可用 |

单 seed 最优值用于筛选候选，不能替代 promotion。不同训练样本单位、不同标签或不同 protocol 的分数保留在各自工作线内解读。

## 8. 当前优先级与下一步

1. 为 `dynamic_kernel_no_prior` 补齐与现有 7-seed candidate 完全相同的四个 seed，完成 state-prior compatibility 的直接 paired causal comparison。
2. 在 `dynamic_kernel_prior_uniform` 上加入训练期 video/wear modality dropout、质量特征驱动的 routing 降权和 video-only/day-shift 诊断，以改善 shuffle corruption 的稳定性。
3. 给 `kernel_short`、`kernel_medium` 与 full-history static baseline 扩展到 5--7 seeds，确定 EMA 标签的实际 look-back 范围。
4. 在窗口级 `eeg_eegpt_partial_ft_v1` 主线、`cross_day` 与 `within_subject_day` 上完成 attention versus concat 的 matched multi-seed gate，再决定 fusion default。
5. 为 EQL-CAF 生成真实四模态 2 秒 token；在此之前保留 smoke 作为契约验证。

## 9. 脚本与产物导航

| 目的 | 入口 | 主要产物 |
| --- | --- | --- |
| 窗口级 EEG token | `34_run_eeg_encoder_matrix.py` | `eeg_encoder_256d_tokens/{protocol}/{profile}/seed_*.npz` |
| 窗口级融合回归 | `32_run_eegpt_centered_loss.py` | report JSON/MD、prediction、checkpoint |
| EMA bag 构建与矩阵 | `73`--`77` | `outputs/daily_affect_ordinal_routefix_20260906/` |
| Focused daily-affect | `78`--`83` | 远端 `outputs/daily_affect_dynamic_a1_crossday_routefix_20260906/` |
| 窗口与 daily-affect event bridge | `84_compare_window_daily_event_level.py` | `outputs/server_sync/daily_affect_window_event_bridge_20260906/comparison/` |
| Daily-affect 7-seed 本地副本 | 见下列路径 | `outputs/server_sync/daily_affect_dynamic_a1_crossday_routefix_20260906/` |
| EQL-CAF temporal 契约 | `57`--`64` | `src/daily_multimodal/temporal/` 与 packed temporal NPZ |

关键 daily-affect 证据：

- `outputs/server_sync/daily_affect_dynamic_a1_crossday_routefix_20260906/norm_per_modality__adapter_per_modality/reports_7seed/daily_affect_ordinal_summary.md`
- `outputs/server_sync/daily_affect_dynamic_a1_crossday_routefix_20260906/norm_per_modality__adapter_per_modality/robustness_7seed/focused_robustness_report.md`
- `outputs/server_sync/daily_affect_dynamic_a1_crossday_routefix_20260906/norm_per_modality__adapter_per_modality/figures_7seed/daily_affect_diagnostics_atlas_cross_day.png`

窗口级历史矩阵与 matched fusion evidence：

- `outputs/server_sync/eeg_encoder_256d_5route_20260814/`
- `fusion_attention_vs_concat_evidence_20260820.md`
- `wear_moment_results_20260820.md`
