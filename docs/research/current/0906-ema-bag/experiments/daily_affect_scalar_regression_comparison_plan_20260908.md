# Daily-affect 标量回归双路线比较计划

> 版本：2026-09-08  
> 状态：**v2 主比较已完成：171-run 三 seed 全矩阵与 28-run 四 seed 扩展均已落盘；此前 full-route 171-run 与其 Phase 5 扩展保留为独立诊断，不用于主结论。**  
> 对照来源：`technical_route_20260814.md` 的独立窗口回归路线与 `technical_route_20260906.md` 的 Daily-affect EMA-bag 路线

## 1. 目标与唯一比较问题

本实验把两条路线统一为标量疲劳回归，在完全匹配的上游 token、EMA event、split、训练 seed 和回归目标下比较 held-out event-level `raw Pearson r`：

1. **窗口架构 × 回归目标**：每个 10 秒窗口独立完成四模态融合和标量回归，再把同一 EMA event 的 23 个窗口预测固定汇总为一个 event 预测。
2. **EMA-bag 架构 × 回归目标**：一次读取同一 EMA event 的 23 个窗口，运行 0906 路线中的全部 EMA-bag 模型变体，以标量回归 head 输出一个 event 预测。

本计划只比较以上两条回归路线。五分类 head、QWK 训练、序数目标对照、EQL-CAF、上游 encoder 重训和新 token 路线均不进入本实验。

实验最终回答：

- 联合建模 23 个窗口能否比独立窗口预测后汇总获得更高 event-level raw r；
- 状态、prior、固定/全局/动态时间核能否在标量回归任务上提供稳定增益；
- 0906 的收益来自 EMA-bag 时间架构，还是依赖原来的五分类/序数训练目标。

## 2. 公平比较契约

### 2.1 统一监督与评价单位

Canonical 输入保持为：

```text
one EMA event = tokens (23, 4, 256) + modality_mask (23, 4) + one fatigue label
```

- 数据全集沿用 canonical EEG-aligned `28,819` 个窗口、`1,253` 个 EMA events、`event_window_id=0..22`。
- 每个 event 含 23 个重叠的 10 秒窗口，stride 为 5 秒。
- 最终训练审计、validation checkpoint 选择和 test 指标均以 **EMA event** 为单位。
- 一个 test event 的 23 个窗口必须全部来自同一个 test leaf split；validation 同理。
- 任一 event 跨越 train/validation/test 边界时，该 event 不进入对应比较集合，并在 preflight 中单独报告。

### 2.2 匹配原则

同一 `protocol × seed` 内，两条路线必须复用：

- 同一份只读 `ema_bags.npz`；
- 同一 event 顺序、23-window membership、标签和 `modality_mask`；
- 同一 EEG/Wear/Video/Audio token artifact；
- 同一 train-only normalization 统计量；
- 同一 train/validation/test event 集合；
- 同一模型初始化 seed、batch event 顺序、最大 epoch 和 patience；
- 同一标量回归目标和 checkpoint 选择规则；
- 同一 test prediction 行顺序与 paired bootstrap resampling index。

训练侧不得根据 test 指标调整 pooling、loss、模型列表、学习率、epoch、early stopping 或回归难度定义。

### 2.3 三协议锁定配置

每个协议内部做严格配对；不同协议的数值分别解释。

| protocol | 锁定 route | normalization | adapter | 角色 |
| --- | --- | --- | --- | --- |
| `cross_day` | `A1_Wphysio_no_audio` | `per_modality` | `per_modality` | 0814 协议内最佳窗口路线 |
| `within_subject_day` | `A1_Wphysio_no_audio` | `per_modality` | `per_modality` | 0814 协议内最佳窗口路线 |
| `cross_subject` | `B0_Wphysio_no_audio` | `shared` | `shared` | 0814 协议内最佳窗口路线 |

Phase 0 必须从 bag manifest 重新读取 EEG branch 和 `supervision_boundary`。v2 三协议统一读取 `eeg_eegpt_partial_ft_v1`（fatigue-supervised control）及上表的 no-audio route；窗口与 EMA-bag 两条路线在同一协议内必须读取同一 branch。若实际 bag 配置与上表不一致，先修订计划或重建匹配 bag，再开始训练。

## 3. 统一标量回归定义

### 3.1 Target 与主 head

只使用 train events 拟合标签均值和标准差：

```text
z = (fatigue - train_mean) / train_std
```

所有模型的最终 head 输出一个标量 `z_hat`，评价前逆变换为原始 1--5 疲劳量尺：

```text
y_hat = z_hat * train_std + train_mean
```

原生预测不裁剪到 `[1,5]`。如需展示裁剪后的 RMSE，只能作为附加诊断，raw r 和主表始终使用未裁剪预测。

### 3.2 主训练目标

两条路线统一使用：

```text
L_main = mean((z_hat - z)^2)
```

即 train-only 标准化标签上的 MSE。该选择与 0814 窗口回归主线一致，使本实验只改变窗口组织和 EMA-bag 时间架构。

本轮固定：

- 不加入 batch correlation loss；
- 不加入 centered loss；
- 不加入 within-subject ranking loss；
- 不加入 class weight、CE、cumulative ordinal loss 或 expected-score Huber；
- 不扫描 loss 权重。

### 3.3 Checkpoint 选择

所有模型按 **validation event-level RMSE 最小** 保存 checkpoint，patience 固定为 `15`，最多 `80` epochs。

raw r 是最终主要比较指标；validation RMSE 负责稳定选模并约束预测尺度。validation raw r、centered r 和 prediction standard deviation 全程记录，但不参与 epoch 选择。

## 4. 路线 W：窗口架构 × 回归目标

唯一窗口基线 ID：

```text
window_attention_regression_full_mean
```

### 4.1 模型

复用 0814 的独立窗口融合结构：

```text
four modality tokens (4,256)
-> Linear(256,128) + modality embedding
-> single-head modality self-attention
-> learnable query pooling
-> LayerNorm + MLP + scalar regression head
-> one prediction per 10-second window
```

23 个窗口之间不传递 hidden state，不共享 event-adaptive temporal weights。

### 4.2 训练损失的 event 等权

每个 event 内先对有效窗口的 MSE 求均值，再在 batch 内对 event 求均值：

```text
L_window(event) = mean_t_valid((z_hat_t - z_event)^2)
L_batch = mean_event(L_window(event))
```

这样每个 EMA event 对训练目标贡献相同，不因有效窗口数量不同而改变权重。event 标签仍由各窗口独立学习，保持窗口路线的监督含义。

### 4.3 Event-level 输出

主比较预注册为有效窗口预测的等权平均：

```text
y_hat_event = mean_t_valid(y_hat_t)
```

只使用 `full_mean`。`last_30s`、固定指数核或 validation 选择 pooling 不进入窗口主线，避免增加第二套窗口模型搜索空间。

## 5. 路线 B：EMA-bag 架构 × 回归目标

### 5.1 全部模型 ID

0906 当前代码中的 11 个 EMA-bag `model_id` 全部建立标量回归版本：

| 回归实验 ID | 对应 0906 model_id | 保留的结构 |
| --- | --- | --- |
| `bag_static_reg` | `bag_static` | 窗口内 attention/query pooling，固定时间汇总后标量回归 |
| `state_uniform_reg` | `state_uniform` | 线性模态 evidence routing、GRU state、均匀时间汇总 |
| `prior_uniform_reg` | `prior_uniform` | state-prior compatibility 路由、均匀时间汇总 |
| `prior_regD_uniform_reg` | `prior_ordD_uniform` | prior 路由加回归不确定度抑制、均匀时间汇总 |
| `global_kernel_no_prior_reg` | `global_kernel_no_prior` | 所有 events 共用的 learned short/medium/long 核混合 |
| `dynamic_kernel_no_prior_reg` | `dynamic_kernel_no_prior` | 无 prior 的 event-adaptive 时间核 |
| `dynamic_kernel_prior_uniform_reg` | `dynamic_kernel_prior_uniform` | prior-guided 模态路由加 event-adaptive 时间核，无 difficulty penalty |
| `dynamic_kernel_reg` | `dynamic_kernel` | prior 路由、回归不确定度抑制和 event-adaptive 时间核 |
| `dynamic_fixed_short_reg` | `dynamic_fixed_short` | 完整状态/prior/回归难度路由，时间核固定 short |
| `dynamic_fixed_medium_reg` | `dynamic_fixed_medium` | 完整状态/prior/回归难度路由，时间核固定 medium |
| `dynamic_fixed_long_reg` | `dynamic_fixed_long` | 完整状态/prior/回归难度路由，时间核固定 long |

### 5.2 `bag_static_reg` 固定时间策略

除默认 `uniform` 外，0906 的全部静态 temporal policies 均纳入回归矩阵：

```text
uniform
last_10s
last_30s
last_60s
first_30s
kernel_short
kernel_medium
kernel_long
```

因此 EMA-bag 主矩阵包含：

```text
11 native/default model conditions
+ 7 additional bag_static temporal-policy conditions
= 18 EMA-bag regression conditions per protocol/seed
```

### 5.3 分类专属 probe 的回归映射

`prior_ordD_uniform`、`dynamic_kernel` 和 `dynamic_fixed_*` 原本使用五分类/累计序数 probe 的预测难度参与路由。标量回归版本统一改为异方差回归 probe：

```text
per-modality bag representation
-> probe_mean mu_m
-> probe_log_variance s_m

L_probe = 0.5 * mean(exp(-s_m) * (z - mu_m)^2 + s_m)
d_reg(m) = sigmoid(s_m)
```

- `d_reg` 保持在 `[0,1]`，替代原 ordinal difficulty；
- native regression-difficulty 条件默认 `detach_difficulty=true`；
- `L_total = L_main + 0.1 * L_probe`；
- 仅 `prior_regD_uniform_reg`、`dynamic_kernel_reg` 和三个 `dynamic_fixed_*_reg` 使用该 probe；
- `prior_uniform_reg` 和 `dynamic_kernel_prior_uniform_reg` 的 difficulty 为零、probe loss 为零；
- 所有 probe 标签仍为标量回归标签，不引入五分类或序数辅助目标。

原 P0--P5 是分类/累计序数 probe 的 routing profiles。本计划把它们视为训练配置而非独立模型 ID，不逐项迁移；no-prior 结构已由对应 model IDs 覆盖，回归难度只运行锁定的 detached 版本。Calibrated 和 end-to-end regression-difficulty 扩展需另立计划。

## 6. 实验矩阵与运行数

### 6.1 三 seed 全模型矩阵

固定 seeds：

```text
240729, 240730, 240731
```

每个 `protocol × seed`：

- 1 个 `window_attention_regression_full_mean`；
- 18 个 EMA-bag regression conditions。

完整三协议运行数：

```text
3 protocols x 3 seeds x (1 window + 18 EMA-bag) = 171 runs
```

完全相同配置完成的 smoke run 可以复用到 171-run 矩阵，不重复训练。

### 6.2 七 seed 扩展

三 seed 矩阵完成后，每个协议独立选择最多两个 EMA-bag 候选，补齐：

```text
240732, 240733, 240734, 240735
```

同一协议的窗口基线也补齐相同四个 seeds。七 seed 阶段只扩展通过第 9 节 promotion gate 的模型；三 seed 未通过时保留完整描述性结果并停止该模型扩展。

## 7. 评价指标与比较方式

### 7.1 主指标

```text
event-level raw Pearson r
```

每个 EMA-bag condition 与同 `protocol × seed` 的窗口基线做配对：

```text
delta_raw_r = raw_r(EMA-bag) - raw_r(window)
```

### 7.2 必报 guardrails

- event-level RMSE；
- event-level MAE；
- within-subject centered r；
- per-subject r mean 和有效 subject 数；
- prediction standard deviation；
- prediction P10--P90 span 与 IQR；
- best epoch、训练 epoch 数和 train/validation loss；
- 各 split 的 event、subject、subject-day 和标签计数。

QWK、Macro-F1、ordinal MAE 和分类混淆矩阵不进入本计划的模型选择或主结论。

### 7.3 两层配对比较

每个 EMA-bag 模型报告两种 delta：

1. `EMA-bag model - matched window regression`：回答完整两路线比较；
2. `EMA-bag model - bag_static_reg(uniform)`：回答 EMA-bag 内部机制增益。

固定时间策略另以 `bag_static_reg(uniform)` 为共同参照。

### 7.4 Subject-day block bootstrap

每个 seed 内使用相同的 test events 和相同 subject-day block resampling index，执行 `2,000` 次 paired bootstrap。

每个模型必须报告：

- 每 seed point delta；
- 每 seed bootstrap 95% CI；
- seed-level delta mean +/- std；
- 正向 seed 数；
- 三/七 seed 的方向一致性；
- raw r 改善时 RMSE、centered r 和预测范围是否同步稳定。

不同 seeds 的预测不得先平均后做一次 pooled bootstrap。

## 8. 分阶段实施

### Phase 0：只读 preflight

逐协议、逐 seed 验证：

1. bag shape 为 `(N_event,23,4,256)`，mask 为 `(N_event,23,4)`；
2. `event_window_id=0..22` 且每条 event membership 唯一；
3. window 与 bag 两条路线的 train/val/test events 完全一致；
4. validation/test events 的 23 个窗口均来自对应 leaf split；
5. token route、normalization、adapter 和 EEG branch 与锁定表一致；
6. train-only 标签与 token normalization 没有读取 validation/test；
7. `supervision_boundary` 写入 manifest；
8. 输出根目录独立，不覆盖 ordinal/QWK、bridge 或历史窗口回归结果。

**停止门槛**：任一 split、event membership、token route 或 normalization 审计失败，停止所有训练并修复数据契约。

### Phase 1：实现与本地/远端测试

建议新增：

```text
scripts/daily_affect/90_run_daily_affect_scalar_regression_matrix.py
scripts/daily_affect/91_summarize_daily_affect_scalar_regression.py
scripts/daily_affect/92_plot_daily_affect_scalar_regression.py
```

训练代码需要支持：

- `task_type=scalar_regression`；
- scalar main head；
- event-balanced window MSE；
- event-level validation prediction；
- regression uncertainty probe/difficulty；
- RMSE checkpoint selection；
- 保存 window 和 event predictions；
- 独立 objective/model IDs，禁止读取 ordinal checkpoint。

最低测试：

- scalar head 输出 shape；
- train-only target normalization 与 inverse transform；
- window loss 对 event 等权；
- window `full_mean` event prediction；
- bag scalar prediction；
- regression probe NLL 和 bounded `d_reg`；
- detached difficulty 不回传 probe；
- mask 下空窗口与缺失模态处理；
- validation RMSE checkpoint selection；
- prediction/label/event_id 一一对应；
- ordinal 和 scalar-regression 产物路径互不覆盖。

**继续门槛**：本地单元测试、远端单元测试和 CPU synthetic end-to-end smoke 全部通过。

### Phase 2：真实数据 GPU smoke

使用 `cross_day / seed 240729` 运行四个代表条件：

```text
window_attention_regression_full_mean
bag_static_reg__uniform
prior_regD_uniform_reg
dynamic_kernel_reg
```

检查：

- `4/4` run 正常完成；
- train/validation loss finite；
- test prediction 数与匹配 event 数一致；
- prediction SD 大于 `0.05`；
- 最佳 epoch 可审计且 checkpoint 能重新加载复算；
- 两条路线逐 event 标签和 split 索引完全一致；
- difficulty 模型的 `d_reg` finite、位于 `[0,1]` 且存在非零方差。

**停止门槛**：出现 NaN/Inf、常数预测、event 错位、split 污染、checkpoint 复算不一致或 difficulty 塌缩时停止扩展。

### Phase 3：171-run 三协议全矩阵

执行顺序：

```text
cross_day
-> within_subject_day
-> cross_subject
```

每完成一个 protocol，立即运行 completeness audit，确认：

```text
3 seeds x 19 conditions = 57/57 valid runs
```

该阶段覆盖全部模型，不根据中途 test 排名删减后续模型。只有数据、数值稳定性或结果完整性门槛失败才停止。

### Phase 4：三 seed 汇总和候选锁定

生成完整 paired 表、趋势图和 subject-day bootstrap。每个协议最多锁定两个 EMA-bag 候选，候选只能由预注册 gate 产生，锁定后不再改变模型、loss、route 或超参数。

### Phase 5：候选补齐七 seed

对窗口基线和通过 gate 的 EMA-bag 候选补跑四个新 seeds，重新计算七 seed paired delta、方向一致性和 subject-day bootstrap。未过门槛的协议在三 seed 报告处结束。

## 9. Promotion 与停止门槛

### 9.1 三 seed进入七 seed

EMA-bag 候选需同时满足：

- mean `delta_raw_r > 0`；
- raw r 至少 `2/3` seeds 优于 matched window；
- mean `delta_RMSE <= +0.02`；
- centered r 没有 `3/3` 同向恶化；
- prediction SD 不低于窗口基线的 `50%`；
- 所有 event/split/bootstrap 审计通过。

若超过两个模型通过，按 validation raw r mean 排序锁定前两个；test 指标只用于最终描述，不用于候选之间的再调参。

### 9.2 七 seed正式支持 EMA-bag 优势

候选需同时满足：

- mean `delta_raw_r > 0`；
- raw r 至少 `5/7` seeds 胜出；
- seed-level paired delta mean +/- std 明确报告；
- subject-day bootstrap 在多数 seeds 中支持正方向；
- mean `delta_RMSE <= +0.02`；
- centered r 与动态范围没有形成系统性反向证据。

门槛未通过时，结论写为“该回归 EMA-bag 变体未获得稳定优于匹配窗口回归的证据”，并停止该模型扩展。

整个实验遵循“直到有门槛未过再停下”：每一阶段只在上一阶段的契约、完整性和稳定性门槛通过后继续；三 seed 全模型矩阵完成后，性能门槛只控制七 seed 扩展。

## 10. 计划命令与恢复机制

计划中的远端运行入口：

```bash
cd /vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned
export PYTHONPATH=src

runtime/envs/eegpt-gpu-min/bin/python \
  scripts/daily_affect/90_run_daily_affect_scalar_regression.py \
  --stage matrix \
  --protocols cross_day,within_subject_day,cross_subject \
  --seeds 240729,240730,240731 \
  --epochs 80 \
  --patience 15 \
  --skip-existing
```

实际 CLI 以 Phase 1 实现并通过测试后的 `--help` 为准。每个 run 独立落盘并原子写入完成标记；中断后只重启缺失或失败的 run。

## 10.1 执行记录（2026-09-08）

- v2 设计修订：用户要求三个协议都使用 0814 各自的最佳窗口路线，即 `cross_day/within_subject_day=A1_Wphysio_no_audio`、`cross_subject=B0_Wphysio_no_audio`，并统一使用 `eeg_eegpt_partial_ft_v1`。此前 full-route、fixed-token 的 171-run 及其 24-run 扩展不回答该问题，保留为独立诊断。
- v2 Phase 0--2：在独立 `v2_partialft_noaudio_bags/` 下构建 9 个 matched bag；每个均标记 `mixed_with_fatigue_supervised_controls:eeg_eegpt_partial_ft_v1`。v2 preflight 通过 9 个 protocol×seed bag，cross-day 四条件 GPU smoke 已完成 `4/4`。
- v2 Phase 3：以 v2 bags、三条 0814 协议内最佳 route、相同 `240729/240730/240731` seeds 启动新的 `171` 条矩阵，结果根为 `outputs/daily_affect_scalar_regression_20260908/v2_partialft_noaudio/`。
- v2 Phase 3 完成：`171/171` 均写出 checkpoint/metrics，`91` 对 162 条 window-paired 结果完成 2,000 次 subject-day bootstrap。cross-day 最强 test delta 为 `bag_static_reg__temporal_last_30s`（raw-r `+0.0391`，`3/3`）；cross-subject 为 `prior_uniform_reg`（`+0.0846`，`3/3`）；within-subject-day 为 `prior_uniform_reg`（`+0.0029`，`2/3`）。
- v2 Phase 5：按 validation-only 排序锁定 cross-day `bag_static_reg__temporal_uniform` 与 `bag_static_reg__temporal_last_30s`，cross-subject/within-subject-day 各锁定 `prior_uniform_reg`；4 个新增 seeds 的 12 个 bag preflight 通过，28 条 seven-seed run 已启动。
- Phase 0：远端 `90 ... --stage preflight --device cpu` 已检查 9 个 protocol×seed bag，全部通过 `(N,23,4,256)`、唯一 event ID、train/validation/test event 无交集与 metadata 读取。
- Phase 1：新增标量 head、event-balanced replicated-window MSE、EMA-bag 回归结构、异方差 regression-difficulty probe、`90` 矩阵入口、`91` paired summary 和 3 项合成回归单测；本地与远端测试均通过。
- Phase 2：`cross_day / A1_Wphysio_full / seed_240729` 四条件 GPU smoke 已完成，4 个 run 可由 `91` 成功汇总并产生 subject-day bootstrap。
- Phase 3：三协议、三 seed、19 条件的矩阵已完成 `171/171`，所有 run 均有 checkpoint 与 metrics，日志未出现数值或运行错误；`91` 已生成 162 条 window-paired、2,000 次 subject-day bootstrap。
- Phase 4：按预注册门槛，`cross_day` 锁定 `bag_static_reg__temporal_kernel_medium` 和 `bag_static_reg__temporal_last_60s`；`cross_subject` 锁定 `bag_static_reg__temporal_first_30s` 和 `dynamic_kernel_no_prior_reg`；`within_subject_day` 没有条件通过，不进入七 seed 扩展。
- Phase 5：`cross_day` 与 `cross_subject` 的新 seed `240732--240735` bag preflight 已通过，窗口基线和每协议两名锁定候选共 24 条 run 正在同一独立结果根执行。
- 2026-09-10 执行状态校正：v2 Phase 5 已完成，结果根共有 `199` 个 `metrics.json`，无活动训练进程；`91` 已重新以 2,000 次 subject-day paired bootstrap 汇总 `178` 条配对结果。实际扩展候选为 cross-day `bag_static_reg__temporal_uniform` 与 `bag_static_reg__temporal_last_30s`、cross-subject `prior_uniform_reg`、within-subject-day `prior_uniform_reg`，窗口基线同步补齐四 seed。最终 seven-seed delta raw-r 分别为 cross-day last-30s `+0.0453`（`7/7`）、cross-day uniform `-0.0154`（`2/7`）、cross-subject prior-uniform `+0.0631`（`6/7`）、within-subject-day prior-uniform `-0.0154`（`3/7`）。

## 11. 输出目录与交付物

独立结果根：

```text
outputs/daily_affect_scalar_regression_20260908/
|-- preflight/
|   |-- event_split_audit.csv
|   `-- manifest.json
|-- runs/
|   `-- {protocol}/{route}/{condition}/seed_{seed}/
|       |-- config.json
|       |-- best_checkpoint.pt
|       |-- val_history.csv
|       |-- predictions.npz
|       |-- test_predictions.csv
|       `-- metrics.json
|-- summary/
|   |-- run_metrics.csv
|   |-- paired_window_deltas.csv
|   |-- condition_summary.csv
|   `-- summary.md
`-- figures/
|-- raw_r_rmse_by_protocol.png
    `-- paired_delta_raw_r.png
```

最终报告必须明确区分：

- 已完成 runs 与计划 runs；
- 三 seed exploratory screen 与七 seed paired evidence；
- `EMA-bag - window` 和 `EMA-bag - bag_static_reg` 两类 delta；
- 纯回归模型与使用 regression uncertainty probe 的模型；
- 每个 protocol 的独立结论；
- 上游 token 的 label-free/fatigue-supervised 边界；
- 通过、停止及未完成阶段的原因。

## 12. 最终主表模板

| protocol | family | condition | seeds | raw r | delta raw r vs window | raw-r wins | RMSE | delta RMSE vs window | centered r | prediction SD | gate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cross_day | window | `window_attention_regression_full_mean` | 3/7 |  | reference | -- |  | reference |  |  | reference |
| cross_day | EMA-bag | `bag_static_reg__uniform` | 3/7 |  |  |  |  |  |  |  |  |
| cross_day | EMA-bag | `state_uniform_reg` | 3/7 |  |  |  |  |  |  |  |  |
| cross_day | EMA-bag | `dynamic_kernel_reg` | 3/7 |  |  |  |  |  |  |  |  |

附录提供 171-run 完整表、全部静态时间策略、全部 seed 指标、bootstrap 区间和逐 event 预测索引。
