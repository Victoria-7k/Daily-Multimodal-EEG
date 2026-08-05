# EEGPT B0/A1/A2 EEG 对齐多模态交叉注意力完整结果报告

## 技术摘要

本报告汇总 EEGPT EEG 分支接入后的 EEG 对齐多模态 cross-attention 矩阵结果。实验基于 28,819 行 canonical EEG window index，默认预测 `fatigue`，在 `cross_subject`、`cross_day`、`within_subject_day` 三个 `splits_new` 协议下训练和测试。

- **全量 route-aware 结果已经完成。** 本轮共有 48 个 metrics 文件：3 个协议 x 16 个实验。16 个实验由原 B0 的 8 个控制组合，加上 A1/A2 各 4 个真正使用视频的组合构成；`no_video` 和 `bio_only` 是路线无关控制项，只在 B0 名下保留一次。
- **最佳 RMSE 随泛化协议变化。** `cross_subject` 最优为 `B0_Wphysio_bio_only`，RMSE `0.9012`，raw r `-0.0207`；`cross_day` 和 `within_subject_day` 都由 `A1_Wphysio_no_audio` 取得最佳 RMSE，分别为 `0.9298` 和 `0.9348`，raw r 分别为 `0.2594` 和 `0.3032`。
- **A1 是当前跨天和被试内跨天的主候选视频路线。** 两个协议的最佳 RMSE 都来自 A1 + Wphysio + no_audio。A2 在 `within_subject_day` 中很接近，并给出最高 raw r 行：`A2_Wdeep_full`，raw r `0.3161`。
- **音频分支在当前矩阵中整体拉高误差。** 18 个 full-vs-no-audio 配对中，加入音频后 RMSE 平均增加 `0.0151`，raw r 平均变化 `-0.0271`。`cross_day` 和 `within_subject_day` 的顶部候选主要来自 `no_audio`。
- **centered r 显示模型的个体内波动追踪能力弱于总体排序能力。** `cross_day` 的 centered r 最优为 `A1_Wphysio_no_audio`（`0.1107`）；`within_subject_day` 的 centered r 最优为 `B0_Wdeep_no_audio`（`0.1026`）。二者明显低于对应协议的 raw r 前排结果，说明当前模型更擅长总体 fatigue 水平排序，个体内相对波动仍是后续改进重点。
- **跨被试场景更像校准问题，排序信号仍弱。** `cross_subject` 的最低 RMSE 来自 bio-only 控制项，但 raw Pearson r 接近 0；因此它适合作为当前跨被试 RMSE 基线，暂不宜当作跨被试 fatigue ranking 已经有效的证据。

## 协议级最佳结果

| 协议 | RMSE 最优实验 | RMSE | MAE | Raw r | Centered r | Per-subject r mean | Raw r 最优实验 | Raw r | RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `cross_subject` | `B0_Wphysio_bio_only` | 0.9012 | 0.6861 | -0.0207 | -0.0415 | -0.0336 | `B0_Wdeep_bio_only` | 0.0835 | 0.9060 |
| `cross_day` | `A1_Wphysio_no_audio` | 0.9298 | 0.7055 | 0.2594 | 0.1107 | 0.0687 | `A1_Wphysio_no_audio` | 0.2594 | 0.9298 |
| `within_subject_day` | `A1_Wphysio_no_audio` | 0.9348 | 0.7255 | 0.3032 | 0.0778 | 0.0555 | `A2_Wdeep_full` | 0.3161 | 0.9382 |

**解读。** RMSE 衡量预测尺度和校准，raw r 衡量样本排序。`cross_subject` 的 RMSE 最优行和 raw r 最优行并不一致，且相关系数都偏弱；`cross_day` 的 RMSE 和 raw r 最优都集中在 `A1_Wphysio_no_audio`；`within_subject_day` 中 A1 拿到最低 RMSE，A2/Wdeep/full 拿到最高 raw r。后续如果主目标是误差，优先追 A1/Wphysio/no_audio；如果主目标是排序，也要保留 A2/Wdeep 作为比较对象。

## Raw r 与 centered r 的计算和解释

对某个协议的 test split，设第 `i` 个测试样本属于被试 `s_i`，真实 fatigue 为 `y_i`，模型预测为 `ŷ_i`。raw r 直接在全部测试样本上计算：

```text
raw_r = corr({ŷ_i}, {y_i}), i in test
```

within-subject centered r 先在测试集内部按被试去均值：

```text
ŷ_i_centered = ŷ_i - mean(ŷ_j | s_j = s_i, j in test)
y_i_centered = y_i - mean(y_j | s_j = s_i, j in test)
```

然后把所有 centered 后的测试样本 pooled 到一起计算 Pearson r：

```text
within_subject_centered_r =
  corr({ŷ_i_centered}, {y_i_centered}), i in test
```

这个指标和 `per_subject_r_mean` 不同。`per_subject_r_mean` 是每个被试单独算 r 后再对被试取平均；centered r 是每个被试内先去均值，再把全部样本 pooled 起来算一个总体相关。它回答的问题是：**去掉每个被试自己的平均 fatigue 水平后，模型还能不能追踪同一被试内部的相对升降。**

| 指标 | 主要含义 | 适合支撑的结论 |
| --- | --- | --- |
| RMSE / MAE | fatigue 数值预测误差 | 模型校准和绝对分数可用性 |
| Raw r | 全部测试窗口混在一起时，预测值与真实值的总体高低顺序是否一致 | 总体 fatigue 排序、跨被试/跨天整体水平区分 |
| Within-subject centered r | 去掉被试均值后，同一被试内部高低波动是否一致 | 个体内疲劳动态追踪、是否超越 subject baseline |
| Per-subject r mean/std | 每个被试单独相关后的均值和稳定性 | 被试间表现是否均衡 |

**本轮结果的读法。** Raw r 明显高于 centered r，说明模型捕捉到的主要信号包含被试间或天级整体 fatigue 差异；当这些 subject-level baseline 被扣除后，个体内部波动仍能被部分追踪，但强度较弱。因此主性能结论应使用 RMSE、MAE、raw r；centered r 应作为更严格的动态追踪诊断指标。

## Cross-day 和 within-subject-day 的 centered r 结果

`cross_day` 中，`A1_Wphysio_no_audio` 同时取得最低 RMSE、最高 raw r 和最高 centered r，是当前最均衡的跨天候选。

| Experiment | RMSE | Raw r | Within-subject centered r | Per-subject r mean |
| --- | ---: | ---: | ---: | ---: |
| `A1_Wphysio_no_audio` | 0.9298 | 0.2594 | **0.1107** | 0.0687 |
| `B0_Wdeep_bio_only` | 0.9640 | 0.1521 | 0.1022 | 0.0833 |
| `B0_Wphysio_bio_only` | 0.9751 | -0.0199 | 0.0997 | 0.0933 |
| `B0_Wphysio_no_video` | 0.9603 | 0.1651 | 0.0900 | 0.0560 |
| `B0_Wdeep_no_video` | 0.9722 | 0.1362 | 0.0781 | 0.0431 |

`within_subject_day` 中，centered r 最优是 `B0_Wdeep_no_audio`，而 RMSE 最优是 `A1_Wphysio_no_audio`，raw r 最优是 `A2_Wdeep_full`。这说明被试内跨天场景存在一个清晰取舍：A1/Wphysio/no-audio 更适合低误差主候选，B0/Wdeep/no-audio 更适合个体内波动追踪诊断，A2/Wdeep/full 更适合总体排序对照。

| Experiment | RMSE | Raw r | Within-subject centered r | Per-subject r mean |
| --- | ---: | ---: | ---: | ---: |
| `B0_Wdeep_no_audio` | 0.9577 | 0.2838 | **0.1026** | 0.1038 |
| `A1_Wphysio_full` | 0.9755 | 0.1489 | 0.1004 | 0.0898 |
| `A2_Wdeep_full` | 0.9382 | 0.3161 | 0.0794 | 0.0679 |
| `A1_Wphysio_no_audio` | 0.9348 | 0.3032 | 0.0778 | 0.0555 |
| `A1_Wdeep_no_audio` | 0.9481 | 0.3067 | 0.0765 | 0.0844 |

## 数据范围与指标定义

### 分窗方式

本轮实验使用 EEG 对齐后的 canonical window index，而不是重新从原始多模态文件中临时切窗。索引文件包含 `28819` 行，对应 `1253` 个评分事件；每个事件固定展开为 `23` 个窗口，`event_window_id` 为 `0..22`。每个窗口长度为 `10s`，相邻窗口起点间隔 `5s`，因此每个事件形成一个 5 秒步长的重叠窗口序列。

每一行窗口都有稳定的 `sample_id` 和 `eeg_sample_index`：`sample_id` 从 `eeg_000000` 到 `eeg_028818`，`eeg_sample_index` 严格等于 `0..28818`。所有模态 embedding 都按这 28,819 行的同一顺序保存，后续训练直接按行读取 embedding 与 mask，不再改变窗口数量或顺序。

| 分窗字段 | 本轮取值 |
| --- | --- |
| 事件数 | `1253` |
| 每事件窗口数 | `23` |
| 总窗口数 | `28819` |
| 单窗口长度 | `10s` |
| 窗口 stride | `5s` |
| 窗口编号 | 每个事件内 `event_window_id = 0..22` |
| 样本编号 | `sample_id = eeg_{eeg_sample_index:06d}` |

### 数据集划分方式

训练、验证和测试划分完全使用 `/vePFS-0x0d/DailyEEG/splits_new/` 下的三个 EEG 协议：`cross_subject`、`cross_day`、`within_subject_day`。每个协议目录都包含 `pretrain.json`、`finetune.json`、`val.json`、`test.json` 和 `split_info.json`。监督训练时只把 `pretrain + finetune` 合并为 train，`val` 用于早停/模型选择，`test` 只用于最终报告指标。

这三个协议不是重新抽样得到的实验内随机划分，而是固定的 EEG 官方划分；本轮训练不重写、不过滤、不重排这些 split index。远端复核确认每个协议的 train、val、test 互不重叠，所有 index 都在 `0..28818` 范围内，并且 `train = pretrain + finetune`。

| 协议 | 划分含义 | Pretrain | Finetune | Train 合计 | Val | Test |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `cross_subject` | 测试集考察跨被试泛化 | 6122 | 11519 | 17641 | 5106 | 6072 |
| `cross_day` | 测试集考察跨日期泛化 | 6122 | 10691 | 16813 | 6187 | 5819 |
| `within_subject_day` | 测试集考察同一被试内跨天泛化 | 6122 | 11121 | 17243 | 5708 | 5868 |

### 实验缩写和模态分支含义

本报告中的实验名采用 `视频路线_Wear分支_模态组合` 的形式。例如 `A1_Wphysio_no_audio` 表示：视频使用 A1 路线，wear 使用 Wphysio 分支，启用 EEG + wear + video，不启用 audio。

视频路线 B0/A1/A2 都基于同一套 EEG 对齐 2x face ROI 视频窗口，并输出 256 维 `video_emb`。区别在于视频 embedding 生成时的视觉扰动策略：

| 缩写 | 本轮含义 | 产物路径 |
| --- | --- | --- |
| `B0` | baseline video route；无额外视觉增强的 2x face ROI DINOv2 embedding | `video/video_B0_2xroi_eeg23win_embeddings.npz` |
| `A1` | 在 2x face ROI 视频帧上加入 mild color / brightness 类增强后生成的 DINOv2 embedding | `video/video_A1_2xroi_eeg23win_embeddings.npz` |
| `A2` | 在 A1 的 color / brightness 增强基础上加入 grayscale 扰动后生成的 DINOv2 embedding | `video/video_A2_2xroi_eeg23win_embeddings.npz` |

Wear 的 `physio` 和 `deep` 是两条不同的 PPG/GSR/ACC wearable 表征路线，二者都输出 256 维 `wear_emb`，并且都已经对齐到同一批 28,819 个 EEG 窗口：

| 缩写 | 本轮含义 | 输入信号和处理 | 产物路径 |
| --- | --- | --- | --- |
| `Wphysio` / `wear_physio` | 可解释生理特征分支 | 对 PPG、GSR、ACC 先做固定预处理，再提取 HR/HRV、GSR slope/SCR、ACC motion/stationary 等统计/生理特征，并投影到 256 维 | `wear/wear_physio_preprocessed_eeg23win_embeddings.npz` |
| `Wdeep` / `wear_deep` | 序列型 wearable 表征分支 | 对预处理后的 PPG/GSR/ACC 重采样序列做固定序列 encoder / TCN-like 池化，再投影到 256 维 | `wear/wear_deep_sequence_preprocessed_eeg23win_embeddings.npz` |

模态组合后缀表示启用哪些 token：`full` = EEG + wear + video + audio；`no_audio` = EEG + wear + video；`no_video` = EEG + wear + audio；`bio_only` = EEG + wear。A1/A2 只出现在实际使用视频的组合中；`no_video` 和 `bio_only` 是路线无关控制项，因此只在 B0 名下报告一次。

| 项目 | 定义 |
| --- | --- |
| Canonical row count | `28819` 个 EEG 对齐 10 秒窗口 |
| 窗口定义 | 每个事件 23 个窗口，窗口长 10 秒，stride 5 秒 |
| 目标标签 | `fatigue` |
| Split 策略 | train = `pretrain + finetune`；validation = `val`；test = `test`；不重写 `splits_new` |
| 模型 | modality token cross-attention；每个模态为 256 维 token；train-only 特征归一化；train-standardized target MSE |
| 测试指标 | RMSE、MAE、pooled raw Pearson r、within-subject centered r、per-subject r mean/std；其中 centered r 只用于 `cross_day` 和 `within_subject_day` 的个体内波动诊断 |
| 完成矩阵 | 3 个协议 x 16 个实验 = 48 个 metrics 文件 |

## 分支覆盖率

| 分支 | 行数 | Embedding shape | Mask sum | 覆盖率 |
| --- | ---: | --- | ---: | ---: |
| `eeg` | 28819 | `(28819, 256)` | 28819 | 100.0% |
| `video_B0` | 28819 | `(28819, 256)` | 18017 | 62.5% |
| `video_A1` | 28819 | `(28819, 256)` | 18021 | 62.5% |
| `video_A2` | 28819 | `(28819, 256)` | 17992 | 62.4% |
| `audio` | 28819 | `(28819, 256)` | 17924 | 62.2% |
| `wear_physio` | 28819 | `(28819, 256)` | 24127 | 83.7% |
| `wear_deep` | 28819 | `(28819, 256)` | 24127 | 83.7% |

**校验说明。** 远端 preflight 返回 `ok=True`，已经检查 sample order、embedding shape、mask、canonical labels、EEG sample index 和 split range。EEGPT EEG 分支 100% 覆盖，所以所有 28,819 行至少有一个有效 token。

## 视频路线比较

| 协议 | 路线 | 使用视频的实验数 | 平均 RMSE | 平均 raw r | 该路线 RMSE 最优行 | 最优 RMSE | 对应 raw r |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| `cross_subject` | `B0` | 4 | 0.9337 | 0.0222 | `B0_Wdeep_no_audio` | 0.9154 | 0.0660 |
| `cross_subject` | `A1` | 4 | 0.9395 | -0.0205 | `A1_Wdeep_no_audio` | 0.9198 | 0.0092 |
| `cross_subject` | `A2` | 4 | 0.9373 | 0.0080 | `A2_Wphysio_no_audio` | 0.9163 | 0.0526 |
| `cross_day` | `B0` | 4 | 0.9581 | 0.2097 | `B0_Wphysio_no_audio` | 0.9410 | 0.1967 |
| `cross_day` | `A1` | 4 | 0.9550 | 0.2101 | `A1_Wphysio_no_audio` | 0.9298 | 0.2594 |
| `cross_day` | `A2` | 4 | 0.9526 | 0.2047 | `A2_Wphysio_no_audio` | 0.9339 | 0.2359 |
| `within_subject_day` | `B0` | 4 | 0.9531 | 0.2622 | `B0_Wphysio_no_audio` | 0.9434 | 0.2879 |
| `within_subject_day` | `A1` | 4 | 0.9538 | 0.2529 | `A1_Wphysio_no_audio` | 0.9348 | 0.3032 |
| `within_subject_day` | `A2` | 4 | 0.9465 | 0.2888 | `A2_Wdeep_full` | 0.9382 | 0.3161 |

**解读。** `cross_subject` 中三条视频路线的均值接近，最低整体 RMSE 仍由 bio-only 控制项给出。`cross_day` 中 A2 的视频使用行平均 RMSE 最低，但单行最佳仍是 `A1_Wphysio_no_audio`；两条路线都优于 B0 均值。`within_subject_day` 中 A2 的路线均值和最高 raw r 最突出，A1 仍保持最低单行 RMSE。整体上，A1 更适合做低误差主候选，A2 更适合保留为排序信号和路线鲁棒性的比较候选。

## 消融解读

**音频。** 18 个 full-vs-no-audio 配对中，full 相对 no-audio 的平均 ΔRMSE 为 `+0.0151`，平均 Δraw r 为 `-0.0271`。这说明当前 openSMILE 音频 token 在这个 fusion 设置下整体拖累前排候选，尤其影响 `cross_day` 和 `within_subject_day`。下一轮主候选建议采用 `no_audio`，同时单独重审音频 encoder、音频质量门控和缺失模态建模。

**视频。** 以共享 `no_video` / `bio_only` 作为控制项时，视频收益主要出现在 `cross_day` 和 `within_subject_day` 的无音频设置。A1/A2 都能在若干配对中低于 bio-only 控制项；带音频设置的收益更不稳定。`cross_subject` 中视频常常提高 RMSE，所以视频路线优势应按协议分别陈述。

**Wear。** Wphysio 是当前更稳的低 RMSE 选择，特别是 `cross_day` 和 `within_subject_day` 的主候选 `A1_Wphysio_no_audio`。Wdeep 在部分 within-subject 排序指标上有价值，例如 A2 相关行 raw r 更高。下一轮保留两条 wear 分支更合适：Wphysio 支撑主 RMSE 候选，Wdeep 检查序列特征对排序的贡献。

## 全部 48 个测试结果

| 协议 | 实验 | 启用模态 | RMSE | MAE | Raw r | Centered r | Per-subject r mean | Per-subject r std |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cross_day` | `A1_Wphysio_no_audio` | `eeg+wear+video` | 0.9298 | 0.7055 | 0.2594 | 0.1107 | 0.0687 | 0.2290 |
| `cross_day` | `A2_Wphysio_no_audio` | `eeg+wear+video` | 0.9339 | 0.7205 | 0.2359 | 0.0652 | 0.0402 | 0.1999 |
| `cross_day` | `B0_Wphysio_no_audio` | `eeg+wear+video` | 0.9410 | 0.7127 | 0.1967 | 0.0245 | 0.0089 | 0.1908 |
| `cross_day` | `A2_Wdeep_no_audio` | `eeg+wear+video` | 0.9428 | 0.7125 | 0.1619 | 0.0608 | 0.0469 | 0.2073 |
| `cross_day` | `B0_Wdeep_no_audio` | `eeg+wear+video` | 0.9508 | 0.7470 | 0.2366 | 0.0681 | 0.0600 | 0.2060 |
| `cross_day` | `A1_Wphysio_full` | `eeg+wear+video+audio` | 0.9551 | 0.7333 | 0.1886 | 0.0567 | 0.0310 | 0.2069 |
| `cross_day` | `A1_Wdeep_full` | `eeg+wear+video+audio` | 0.9558 | 0.7558 | 0.2216 | 0.0437 | 0.0224 | 0.1747 |
| `cross_day` | `A2_Wphysio_full` | `eeg+wear+video+audio` | 0.9600 | 0.7307 | 0.1964 | 0.0668 | 0.0308 | 0.2137 |
| `cross_day` | `B0_Wphysio_no_video` | `eeg+wear+audio` | 0.9603 | 0.7389 | 0.1651 | 0.0900 | 0.0560 | 0.2852 |
| `cross_day` | `B0_Wdeep_bio_only` | `eeg+wear` | 0.9640 | 0.7306 | 0.1521 | 0.1022 | 0.0833 | 0.1654 |
| `cross_day` | `B0_Wphysio_full` | `eeg+wear+video+audio` | 0.9687 | 0.7536 | 0.1975 | 0.0445 | 0.0287 | 0.1861 |
| `cross_day` | `B0_Wdeep_full` | `eeg+wear+video+audio` | 0.9719 | 0.7695 | 0.2082 | 0.0448 | -0.0013 | 0.2015 |
| `cross_day` | `B0_Wdeep_no_video` | `eeg+wear+audio` | 0.9722 | 0.7384 | 0.1362 | 0.0781 | 0.0431 | 0.1899 |
| `cross_day` | `A2_Wdeep_full` | `eeg+wear+video+audio` | 0.9736 | 0.7698 | 0.2245 | 0.0524 | 0.0326 | 0.1712 |
| `cross_day` | `B0_Wphysio_bio_only` | `eeg+wear` | 0.9751 | 0.7240 | -0.0199 | 0.0997 | 0.0933 | 0.1969 |
| `cross_day` | `A1_Wdeep_no_audio` | `eeg+wear+video` | 0.9793 | 0.7641 | 0.1708 | 0.0367 | 0.0287 | 0.2162 |
| `cross_subject` | `B0_Wphysio_bio_only` | `eeg+wear` | 0.9012 | 0.6861 | -0.0207 | -0.0415 | -0.0336 | 0.0417 |
| `cross_subject` | `B0_Wdeep_bio_only` | `eeg+wear` | 0.9060 | 0.7257 | 0.0835 | -0.0065 | -0.0277 | 0.1843 |
| `cross_subject` | `B0_Wdeep_no_audio` | `eeg+wear+video` | 0.9154 | 0.7401 | 0.0660 | 0.0681 | 0.0451 | 0.1793 |
| `cross_subject` | `A2_Wphysio_no_audio` | `eeg+wear+video` | 0.9163 | 0.7361 | 0.0526 | 0.0515 | 0.0537 | 0.0031 |
| `cross_subject` | `A1_Wdeep_no_audio` | `eeg+wear+video` | 0.9198 | 0.7317 | 0.0092 | 0.0064 | -0.0368 | 0.1495 |
| `cross_subject` | `A2_Wdeep_no_audio` | `eeg+wear+video` | 0.9259 | 0.7453 | 0.0171 | 0.0167 | 0.0190 | 0.0890 |
| `cross_subject` | `B0_Wdeep_no_video` | `eeg+wear+audio` | 0.9277 | 0.7489 | 0.0315 | 0.0155 | -0.0241 | 0.0958 |
| `cross_subject` | `B0_Wphysio_no_video` | `eeg+wear+audio` | 0.9338 | 0.7401 | -0.0479 | -0.0012 | -0.0144 | 0.0737 |
| `cross_subject` | `A1_Wphysio_full` | `eeg+wear+video+audio` | 0.9369 | 0.7458 | -0.0598 | -0.0093 | -0.0096 | 0.0116 |
| `cross_subject` | `B0_Wphysio_no_audio` | `eeg+wear+video` | 0.9370 | 0.7428 | 0.0282 | 0.0269 | 0.0351 | 0.0555 |
| `cross_subject` | `B0_Wphysio_full` | `eeg+wear+video+audio` | 0.9385 | 0.7463 | -0.0722 | -0.0056 | -0.0092 | 0.0312 |
| `cross_subject` | `A1_Wdeep_full` | `eeg+wear+video+audio` | 0.9417 | 0.7613 | 0.0574 | 0.0387 | 0.0168 | 0.1002 |
| `cross_subject` | `B0_Wdeep_full` | `eeg+wear+video+audio` | 0.9437 | 0.7624 | 0.0668 | 0.0880 | 0.0764 | 0.0972 |
| `cross_subject` | `A2_Wdeep_full` | `eeg+wear+video+audio` | 0.9465 | 0.7679 | 0.0226 | 0.0269 | 0.0108 | 0.0947 |
| `cross_subject` | `A1_Wphysio_no_audio` | `eeg+wear+video` | 0.9596 | 0.7770 | -0.0887 | -0.0416 | -0.0463 | 0.0558 |
| `cross_subject` | `A2_Wphysio_full` | `eeg+wear+video+audio` | 0.9604 | 0.7735 | -0.0603 | -0.0125 | -0.0159 | 0.0312 |
| `within_subject_day` | `A1_Wphysio_no_audio` | `eeg+wear+video` | 0.9348 | 0.7255 | 0.3032 | 0.0778 | 0.0555 | 0.2090 |
| `within_subject_day` | `A2_Wdeep_full` | `eeg+wear+video+audio` | 0.9382 | 0.7421 | 0.3161 | 0.0794 | 0.0679 | 0.2180 |
| `within_subject_day` | `A2_Wdeep_no_audio` | `eeg+wear+video` | 0.9392 | 0.7495 | 0.3071 | 0.0721 | 0.0680 | 0.2298 |
| `within_subject_day` | `B0_Wphysio_no_audio` | `eeg+wear+video` | 0.9434 | 0.7235 | 0.2879 | 0.0664 | 0.0847 | 0.2155 |
| `within_subject_day` | `A1_Wdeep_no_audio` | `eeg+wear+video` | 0.9481 | 0.7365 | 0.3067 | 0.0765 | 0.0844 | 0.1909 |
| `within_subject_day` | `A2_Wphysio_no_audio` | `eeg+wear+video` | 0.9488 | 0.7374 | 0.2852 | 0.0629 | 0.0543 | 0.2203 |
| `within_subject_day` | `B0_Wdeep_full` | `eeg+wear+video+audio` | 0.9546 | 0.7374 | 0.2317 | 0.0643 | 0.0547 | 0.1951 |
| `within_subject_day` | `B0_Wphysio_full` | `eeg+wear+video+audio` | 0.9568 | 0.7337 | 0.2453 | 0.0704 | 0.0575 | 0.2092 |
| `within_subject_day` | `A1_Wdeep_full` | `eeg+wear+video+audio` | 0.9568 | 0.7472 | 0.2527 | 0.0620 | 0.0594 | 0.2006 |
| `within_subject_day` | `B0_Wdeep_no_video` | `eeg+wear+audio` | 0.9575 | 0.7322 | 0.2493 | 0.0367 | 0.0216 | 0.1870 |
| `within_subject_day` | `B0_Wdeep_no_audio` | `eeg+wear+video` | 0.9577 | 0.7596 | 0.2838 | 0.1026 | 0.1038 | 0.1777 |
| `within_subject_day` | `A2_Wphysio_full` | `eeg+wear+video+audio` | 0.9599 | 0.7334 | 0.2468 | 0.0462 | 0.0247 | 0.2077 |
| `within_subject_day` | `B0_Wdeep_bio_only` | `eeg+wear` | 0.9602 | 0.7236 | 0.2263 | -0.0349 | -0.0393 | 0.1627 |
| `within_subject_day` | `B0_Wphysio_no_video` | `eeg+wear+audio` | 0.9619 | 0.7375 | 0.2361 | 0.0701 | 0.0524 | 0.1909 |
| `within_subject_day` | `A1_Wphysio_full` | `eeg+wear+video+audio` | 0.9755 | 0.7357 | 0.1489 | 0.1004 | 0.0898 | 0.1801 |
| `within_subject_day` | `B0_Wphysio_bio_only` | `eeg+wear` | 0.9983 | 0.7239 | -0.0181 | -0.0281 | -0.0183 | 0.1610 |

## 方法与验证细节

- **模型结构。** 每个启用模态堆叠为一个 256 维 token；模型包含 `Linear(256 -> hidden_dim)`、learnable modality embedding、1-head `MultiheadAttention`、learnable query pooling、`LayerNorm + Linear + ReLU + Dropout + Linear` 回归头。
- **训练设置。** AdamW；hidden dim 128；dropout 0.1；learning rate 1e-3；weight decay 1e-4；batch size 256；最多 80 epochs；patience 15；seed 240729；远端 NVIDIA H20 上使用 CUDA。
- **归一化。** 特征归一化只在 train token 上拟合；target 只用 train 统计量标准化后进入 MSE loss，报告指标再反变换到原始尺度。
- **centered r 计算。** 对 `cross_day` 和 `within_subject_day`，报告使用 test split 内的 subject-wise mean centering：先分别对 prediction 和 target 按测试被试去均值，再 pooled 计算 Pearson r。该指标用于诊断个体内相对波动，不作为单独的冠军选择标准。
- **验证结果。** 完整性脚本检查了 3 个 protocol summary、48 个 metrics 文件、`fatigue` target、测试指标字段、train/val/test mask coverage、per-subject r 字段，以及每个 metrics JSON 中的 EEGPT EEG branch path；结果 `errors=0`。
- **路线矩阵定义。** A1/A2 只加入真实使用视频的组合；`no_video` 和 `bio_only` 作为共享路线无关控制项保留在 B0 下。

## 限制与不确定性

- **跨被试排序信号偏弱。** `cross_subject` 最低 RMSE 行的 raw r 接近 0；因此当前结果更支持跨被试误差基线，而对 fatigue ranking 的跨被试推广仍需多 seed 和更强个体差异建模验证。
- **centered r 显示个体内动态仍有改进空间。** `cross_day` 和 `within_subject_day` 的 centered r 最高值分别为 `0.1107` 和 `0.1026`，明显低于对应 raw r 前排结果；论文表述应区分“总体 fatigue 水平排序”和“个体内相对波动追踪”。
- **配对消融包含训练随机性。** 为避免重复无视频行，A1/A2 没有各自复制 `no_video` / `bio_only` 控制项；视频贡献表使用共享控制项，因此适合判断方向，不适合当作严格同 seed 因果消融。
- **音频结论依赖当前 openSMILE 分支。** 当前结果说明这个音频 token 在该 fusion 设置下整体帮助有限；更强音频 encoder、音频质量门控或更细的缺失模态建模可能改变结论。
- **本报告是模型比较证据。** 它适合用于选择下一轮候选路线和论文叙事重点，不直接证明某一模态对 fatigue 的因果作用。

## 建议下一步

1. 将 `A1_Wphysio_no_audio` 作为 `cross_day` 和 `within_subject_day` 的主候选，围绕它做多 seed 稳定性验证。
2. 保留 `A2_Wdeep_full`、`A2_Wdeep_no_audio` 作为排序指标对照，因为 A2 在 `within_subject_day` 中给出最高 raw r。
3. 将 `B0_Wphysio_bio_only` 作为当前 `cross_subject` RMSE 基线，同时持续报告 raw r，避免把低 RMSE 误读成强排序能力。
4. 在 `cross_day` 和 `within_subject_day` 的后续报告中保留 centered r；将它作为个体内动态追踪诊断，与 RMSE/raw r 分开解释。
5. 单独重审音频分支：优先做 audio encoder 替换、audio mask sensitivity，或更强 fusion regularization，再决定 full multimodal 是否进入主结论。
6. 论文叙事建议围绕协议特异性组织：A1/A2 视频路线在跨天和被试内跨天设置下更有价值；跨被试设置当前以 EEG+wear 的 bio-only 校准基线为主。

## 来源产物

- 远端 merged Markdown：`/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/reports/eegpt_allvideo_fusion_matrix_all_protocols_summary.md`
- 远端 merged JSON：`/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/reports/eegpt_allvideo_fusion_matrix_all_protocols_summary.json`
- 远端 preflight：`/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/reports/eegpt_allvideo_alignment_preflight.json`
- 本地同步 JSON：`outputs/server_sync/eegpt_allvideo_fusion/eegpt_allvideo_fusion_matrix_all_protocols_summary.json`
- 本地同步 preflight：`outputs/server_sync/eegpt_allvideo_fusion/eegpt_allvideo_alignment_preflight.json`
