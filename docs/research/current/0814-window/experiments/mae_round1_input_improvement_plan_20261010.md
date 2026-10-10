# MAE 第一轮输入改进实验计划

日期：2026-10-10，Asia/Shanghai。状态：执行中。cross_day EEG 已完成正式 SSL、canonical token 导出和三个 seed 的冻结下游并通过一致性验收；within_subject_day EEG 严格对照按第 4.3 节停止。视频同时间格双路 cache 正在构建，后续阶段由独立控制器按验收门槛接续。

本轮分别验证两个输入改进：VideoMAE 使用已有 2.0× face ROI 视频；EEG-MAE 使用 canonical 训练窗口与过滤后 raw pool 的联合 SSL 训练池。两条实验独立进行，固定模型、遮挡、重建目标、优化参数、监督边界和情绪下游结构。核心证据是新输入相对匹配旧输入的配对收益，再报告其与 A1+MT11 强基线的距离。

历史实验及输入来源见[MAE 交接手册](mae_chat_handout_20261010.md)，尤其第 2.6、3、5 节。本文是后续执行合同，历史结果保持原来源与版本标记。

## 1 本轮要回答的问题

下一轮合同见[第二轮下游适配计划](mae_round2_downstream_adaptation_plan_20261010.md)，分别研究EEG单模态情绪适配、保守尾部微调与event损失。第二轮保留本轮来源、联合门槛与协议停止记录，其规划不改变本轮队列。

| 实验 | 唯一主动改进 | 直接对照 | 要回答的问题 |
| --- | --- | --- | --- |
| V：ROI VideoMAE | 将全幅视频输入换为已有 2.0× face ROI clip | 相同窗口、抽帧时间格和模型配置的全幅 VideoMAE | 聚焦面部的现有输入流程，能否改善 VideoMAE 的情绪下游表现？ |
| E：扩大池 EEG-MAE | 在原 canonical SSL train 中加入过滤后的 raw EEG 窗口 | 最终 v4 canonical-only EEG-MAE | 增加可用无标签 EEG 覆盖，能否改善相同 EEG-MAE 的情绪下游表现？ |

第一轮采用冻结 encoder 的单模态替换：V 实验保留 MT11 EEGPT 与 Wphysio；E 实验保留 Wphysio 与 DINO A1。Wear 表征与处理流程沿用现有参考。降低微调学习率、改变 event loss、情绪辅助重建、蒸馏、Wear 重建改造，以及两项输入改进的联合模型，留到后续单独验证。

## 2 固定的共同合同

### 2.1 身份、协议与监督

| 项目 | 固定值 |
| --- | --- |
| Canonical 导出宇宙 | 28,819 个窗口，`sample_id`、行顺序、窗口身份保持一致 |
| 原始窗口 | EEG 200 Hz、10 秒、59 通道，shape `(28819,2000,59)` |
| 协议 | `cross_day`、`within_subject_day`，分别训练上游模型 |
| Canonical split 根 | H20：`/vePFS-0x0d/DailyEEG/splits_new`；ncc 使用与之逐文件核对一致的视频 split 副本 |
| SSL 上游 seed | `240800`，先设置随机种子再初始化模型 |
| SSL 监督 | 仅信号/图像重建；情绪标签不进入训练、过滤或 checkpoint 选择 |
| SSL 验证 | 原 canonical val；独立固定验证掩码，seed 为 `240800+100003` |
| SSL checkpoint 选择 | 最小验证重建损失，沿用对应 v4 EEG / v2 Video 实现 |
| 下游 seed | `240800,240801,240802`，按相同 seed 配对 |
| 评价单位 | EMA event，每个 event 保留原 23 个窗口及其成员关系 |
| 下游 split | 复制当前 B0 bag 的 train/val/test event 索引，不重分配 event |
| 下游监督 | train 的 11 情绪标签；val 选 checkpoint；test 在配置冻结后报告 |

当前 bag 计数为 cross_day `731/269/253`、within_subject_day `749/246/258` 个 train/val/test event，两两 event 索引交集为 0。`within_subject_day` 仍共享被试和日期，按同日窗口留出诊断解释。

### 2.2 下游结构和优化设置

正式参考 route：

```text
A1_Wphysio_no_audio__eeg_eegpt_partial_ft_multitask_11label_v1
```

| 项目 | 固定值 |
| --- | --- |
| condition | `window_attention_regression_full_mean` |
| model / temporal policy | `window_replicated` / `uniform` |
| head | `H1_shared2_11xhead2`，11 标签及顺序沿用 `LABEL_NAMES` |
| token normalization / adapter | `per_modality` / `per_modality` |
| hidden dim | `128` |
| 优化器 / LR / weight decay | AdamW / `1e-3` / `1e-4` |
| epoch 上限 / patience / batch | `80` / `15` / `64` 个 event |
| dropout / modality dropout | `0.1` / `0.1` |
| 训练目标 | 现有逐窗口复制 event 标签的 MSE，保持实现一致 |
| checkpoint 选择 | `val_macro_standardized_rmse_min` |
| Audio | 关闭 |
| Encoder | 冻结，仅训练现有 adapter、fusion 与 head |

下游 token normalization 按各路线的 canonical train token 重新拟合，算法和拟合范围保持一致。EEG 原始信号的固定归一化参数按第 4.4 节复用；这两类归一化分别记录。

## 3 V 实验：基于已有 2.0× face ROI 的 VideoMAE

### 3.1 输入文件与已确认事实

视频工作主机为 `ncc_serve_4090`，Python 为 `/home/lzs/miniconda3/envs/eeg3dim/bin/python`。

| 输入 | ncc 文件路径 |
| --- | --- |
| 当前 aligned export 根 | `/mnt/dataset4/sitian/wzw/DailyEEG_multimodal_eeg_aligned_export` |
| 既有 2.0× ROI 视频 | `/mnt/dataset4/sitian/wzw/DailyEEG_multimodal_eeg_aligned_export/tmp/video_2xroi_openface_cache_full/openface/<sample_id>/openface_temporal_v1/window.mp4` |
| ROI 来源 sidecar | 同一 `openface_temporal_v1` 目录中的 `openface_target.json` |
| ROI manifest | `/mnt/dataset4/sitian/wzw/DailyEEG_multimodal_eeg_aligned_export/reports/video_2xroi_cache_manifest.jsonl` |
| ROI 生成脚本 | `/mnt/dataset4/sitian/wzw/DailyEEG_multimodal_eeg_aligned_export/scripts/10_prepare_2xroi_openface_window_cache.py` |
| DINO A1 输入来源及 mask | `/mnt/dataset4/sitian/wzw/DailyEEG_multimodal_eeg_aligned_export/embeddings/video/video_A1_2xroi_eeg23win_embeddings.npz` |
| 原全幅 VideoMAE metadata | `/home/lzs/DailyVideoMAE_20260927/input/video_metadata.npz` |
| 原全幅 8×112 cache | `/home/lzs/DailyVideoMAE_20260927/cache/video_8x112.uint8.mmap`，及同名 `.json` |
| 原始全幅录像 | metadata / sidecar 中的 `/mnt/dataset1/sitian/video/.../DJI_*.MP4` |
| 最终 v2 VideoMAE | `/tmp/wangzw_mae_repair_20261008/outputs/stage_a_v2/formal/<protocol>/video_seed_240800` |

既有 ROI clip 使用 2.0 倍扩展人脸框、224×224 图像、最多 16 帧、写入 fps=2；完整 16 帧文件播放时长约 8 秒，对应 sidecar 中的原始 10 秒窗口。ROI 脚本在原始区间上使用 `linspace(start,end-1e-3,16)` 抽帧，解码失败帧会跳过。后续读取必须使用 clip 的本地帧号，原始时间只用于身份、抽帧映射和 split 审计。

本次制定计划时重新核对了原 v2 VideoMAE 与 DINO A1 的 canonical mask：原 MAE 有效 18,012 行，ROI DINO 有效 18,021 行，交集为 18,012 行，原 MAE 独有为 0 行。ROI 独有的 9 行为 `eeg_003368,eeg_003381,eeg_003385,eeg_003388,eeg_005975,eeg_009702,eeg_019479,eeg_022622,eeg_022623`。新的 MAE ROI cache 仍需逐 clip 解码验收，18,021 是已有 ROI 表征可用数，不能直接当作新 cache 的完成数。

### 3.2 建立同时间格的两路 cache

1. 以 canonical `sample_id` 关联 ROI manifest、sidecar 和全幅 metadata。逐行核对原始录像、`clip_start_seconds`、`clip_end_seconds` 以及 `region=2x_face_roi`。新 metadata 仅保存身份与输入来源字段。
2. 对完整 16 帧 ROI 文件，按 8 个时间段的中心抽样，固定选择本地帧号 `1,3,5,7,9,11,13,15`，统一转 RGB，再用 `INTER_AREA` resize 到 112×112，转为 `[0,1]` 输入。保持现有模型的 8 帧输入长度。
3. 对不足 16 帧但可解码的文件，先建立成功写入帧到原始时间的映射；有至少 8 个不同有效帧时按同一分段中心规则采样。映射无法确认、少于 8 个不同帧或侧文件不一致的行，记录原因并从共同 mask 排除。完整 16 帧文件也抽查其时间映射。
4. 全幅对照从同一原始录像读取 ROI 所选帧对应的原始帧号，保留全幅，再用现有 RGB / resize / 数值转换构建 8×112 cache。原始帧号按 ROI 生成脚本中的源 fps 与舍入规则确定；对照无需再次检测或裁剪人脸。
5. 两路 cache 分开保存，均为 canonical 28,819 行、`uint8 (28819,3,8,112,112)`；记录输入 view、样本顺序指纹、帧号映射、源 manifest 指纹和失败原因。

原全幅 cache 使用原始 10 秒区间的 8 个分段中心，时间点与上述 ROI 缓存时间格不同。因此正式 ROI 输入对照使用新建的 `V_FULL_MATCH`；旧 V1 作为历史参照保留。相同时间格可以控制抽帧差异，ROI clip 的裁剪、224 像素预处理和既有编码仍共同构成此次“使用既有 ROI 输入流程”的改动。

### 3.3 固定共同可用 mask

定义：

```text
M_video_common = M_old_full_mae ∩ M_A1_roi ∩ M_roi_decode ∩ M_full_matched_decode
```

Stage A 两路均使用此 mask 过滤相同 canonical train / val；Stage B 两路的 video slot 也使用此 mask。保留全部原 event，不因视频缺失删 event；其 EEG/Wear 输入继续依原 mask 参与模型。

导出仍为 `(28819,256)`，共同 mask 外置零。ROI 原生新增可用的 9 行记入覆盖报告，本轮主对照使用共同 mask。若执行前两路出现新的解码失败，先冻结更新后的共同 mask，再训练两路，避免在训练途中变更样本集。

与强基线比较时另建 `B0_VIDEO_COMMON`：保留原 MT11 EEGPT、Wphysio、DINO A1 token，仅将 video mask 限制为 `M_video_common`，重跑同一冻结下游。原 `B0_NATIVE` 与已完成指标保留。`V_ROI` 与 `B0_VIDEO_COMMON` 的差值用于回答是否超出同可用性基线。

### 3.4 VideoMAE 保持不变的配置

| 项目 | 固定值 |
| --- | --- |
| 输入 | 8 帧，112×112，RGB `[0,1]` |
| Tubelet | `2×16×16`，共 196 个 token |
| Encoder / decoder / dim / heads | `6` / `2` / `256` / `8` |
| 初始化 | 当前项目 VideoMAE-style 随机初始化，seed `240800` |
| 遮挡 | `0.9`，固定遮挡数量，visible-token encoding |
| 重建目标 | 现有每 tubelet 标准化像素目标，masked MSE |
| 优化器 / LR / weight decay | AdamW / `1e-4` / `1e-4` |
| epoch 上限 / patience / batch | `40` / `8` / `8` |
| health probe | 原固定 val 取样规则，256 行，relative variation 下限 `1e-3` |
| token 导出 | 无遮挡 encoder 输出按原规则 mean pooling 为 256D |

两路使用同一代码快照、相同初始参数与训练/验证行顺序，保存初始化 state 指纹。当前修复版的 seed-before-init、固定验证掩码和 health gate 全部沿用。

## 4 E 实验：扩大并过滤 EEG SSL 训练池

### 4.1 输入文件与索引空间

EEG 和情绪下游主机为 `huoshan_TriDim`。Python 为 `/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/runtime/envs/eegpt-gpu-min/bin/python`。

| 输入 | H20 文件路径 |
| --- | --- |
| Canonical EEG | `/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new/X.npy`，`(28819,2000,59)` |
| Canonical 被试/日期/时间 | 同目录 `sub.npy,d.npy,ts.npy,meta.json` |
| Canonical 身份 index | `/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/index/eeg_aligned_window_index.jsonl` |
| Canonical splits | `/vePFS-0x0d/DailyEEG/splits_new/<protocol>/{pretrain,finetune,val,test}.json` |
| Raw 无标签 EEG 库 | `/vePFS-0x0d/DailyEEG/processed_cadt_raw_unlabeled/X.npy`，`(295641,2000,59)` |
| Raw 被试/日期/时间 | 同目录 `sub.npy,d.npy,ts.npy,meta.json` |
| 待过滤 raw 候选池 | `/vePFS-0x0d/DailyEEG/splits_new/shared_unlabeled_pretrain.json`，44,181 个唯一 raw-row ID |
| v4 canonical-only checkpoint / config | `/home/wangzw/mae_norm_channel_20261009/outputs/stage_a_train_channel/formal/<protocol>/eeg_seed_240800/{checkpoint.pt,config.json}` |
| v4 canonical-only token | 同一目录的 `window_embeddings.npz` |

44,181 是候选 raw 行数。扩大后的训练池大小为“原 canonical train + 过滤、去重后保留的 raw 行”，以最终 manifest 为准。Raw 与 canonical 的行号属于不同数组，训练身份使用 `source_kind + source_row`；任何 raw 行号都不能直接用于 canonical split 或导出顺序。

### 4.2 分协议构建 SSL train

| 协议 | 保留的 canonical SSL train | Raw pool 已知边界 | 执行规则 |
| --- | --- | --- | --- |
| cross_day | pretrain ∪ finetune，共 16,813 行 | raw 候选的 54 个 subject-day 与 pretrain 日期集合一致；与 val/test 日期交集为 0 | 重新审计日期与区间，并做质量过滤和重复去除后加入 |
| within_subject_day | pretrain ∪ finetune，共 17,243 行 | raw 候选与 val/test 均有 54 个 subject-day 重叠 | 对实际 val/test 时间区间执行排除，再加入剩余 raw 行 |

两个协议各自输出一份 manifest，不共用一份未经协议过滤的训练列表。原 canonical train 行与 train/val/test split 保持原样；SSL val 仍为原 canonical val，cross_day 6,187 行、within_subject_day 5,708 行。

### 4.3 过滤、去重与覆盖审计

过滤只依据信号可用性、来源、时间和协议边界。按以下顺序执行，每一步保存保留数与排除原因：

1. **索引与信号合同。** 确认 44,181 个索引唯一且属于 raw X 的行范围；核对两库的 200 Hz、59 通道顺序、单位、CADT 预处理版本及时间原点。未能确认的来源不进入新增池。
2. **还原物理区间。** 将两库映射为同一 `(subject_id,date,recording_id,start,end)` 表示，10 秒窗口使用半开区间 `[start,end)`；日期域整数通过 metadata 映射，不能仅凭相同整数判为相同日期。保存 raw/canonical 网格约 2 秒偏移的核验结果。
3. **协议隔离。** cross_day 排除所有不属于既定训练 subject-day 的 raw 行。两协议都对实际 val/test 输入的区间并集排除重叠行；within_subject_day 必须逐区间判定。重叠条件为同一物理时间轴上 `max(start_a,start_b) < min(end_a,end_b)`。时间来自信号实际支持范围；预处理如明确使用窗外上下文，则一并计入支持范围并记录 guard。原 canonical train 与 val/test 的实际信号交集也单独审计，发现既有交叠时停止该协议的严格对照，单列原合同问题。
4. **不可用信号过滤。** 排除形状错误、NaN/Inf、截断窗口和全 59 通道均无时间变化的 raw 窗口。高幅值或低幅值有限窗口保存质量统计，继续沿用 v4 稳健尺度与能量平衡处理；本轮不增加幅值裁剪、频域变换或新的去伪迹算法。
5. **重复去除。** 去掉 raw 候选内完全重复、与 canonical train 相同物理区间的重复记录；跨来源相同信号内容也核查。只保留一个重复输入，canonical 行优先。对训练池内部合法的部分时间重叠，保持原滑窗规则，并报告其数量和比例。
6. **覆盖与分布。** 按 subject-day 报告 canonical/raw 行数、有效记录小时、区间并集覆盖时长、新增独有覆盖、幅值分位数和排除比例。44,181 行与 canonical 行数的算术和不作为独立样本量。

若 within_subject_day 过滤后没有新增窗口，记录 `no_eligible_extra_windows`，该协议的扩大池实验为空对照，不放宽隔离规则。若有少量新增窗口，按实际规模执行并报告覆盖，不能沿用“增加 44,181 窗”的表述。

### 4.4 归一化、训练预算与模型保持不变

**原始信号尺度固定。** 按协议直接复用最终 v4 canonical-only 的 `raw_normalization`：训练窗口通道均值的中位数作为固定中心，关于该中心的窗口 RMS 中位数作为固定尺度。将同一组参数应用到 canonical、raw train 和 canonical val/export。本轮不在扩大池上重新拟合中心或尺度，以便把信号尺度变化排除出主要干预。保存归一化参数来源与指纹，分别记录 `normalization_fit_manifest` 和 `ssl_training_manifest`。

| 项目 | 固定值 |
| --- | --- |
| 输入 / patch | 10 秒、59 通道、200 Hz；10 个 1 秒 patch，每 patch `59×200` |
| Encoder / decoder / dim / heads | `6` / `2` / `256` / `8` |
| 初始化 | 与 v4 同架构随机初始化，seed `240800`；从头做 SSL |
| 遮挡 | `mask_ratio=0.7`，每窗遮挡 41/59 通道，贯穿 10 秒 |
| 重建目标 | `window_energy_balanced_mse_v1`，按窗口均衡，目标能量分母下限 1 |
| 原始信号 normalization | `train_channel_robust_zscore_v1`，复用对应协议的 canonical-only 参数 |
| 优化器 / LR / weight decay | AdamW / `1e-4` / `1e-4` |
| epoch 上限 / patience / batch | `100` / `15` / `128` |
| 采样 | 沿用按训练行均匀 shuffle 的规则；canonical 与保留 raw 合并成带来源身份的行表 |
| health | probe 256；global 与典型窗口 relative variation 下限均 `1e-3` |
| token 导出 | 仅原 canonical X，原 mean pooling，`(28819,256)`，有效 mask 全 true |

正式主实验保持 epoch 上限与早停规则。扩大池会增加每 epoch 的 optimizer steps，也会改变 54 个 raw 来源日期在训练采样中的占比；报告须记录总 steps、已见窗口数、best checkpoint steps、GPU 时间及按日期的输入构成。本轮直接回答“按相同 epoch 配置扩池后的整体收益”。

预设归因检查：若扩大池通过第 7 节的验证推进条件，在后续归因阶段增加 canonical-only 的相同步数对照，保持 batch、LR、初始化、归一化和 val 不变，并在扩大池相同的验证检查步数上选择 checkpoint。该对照单独命名 `E_CANON_STEP_MATCH`；它用于分离额外训练计算量与新增输入的作用。本轮计划不自动启动此扩展。扩大池上的归一化重拟合和 subject-day 均衡采样同样留作独立因素。

## 5 独立对照矩阵与工作量

### 5.1 正式路线

| 计划 ID | EEG | Wear | Video | Video mask | 用途 |
| --- | --- | --- | --- | --- | --- |
| `B0_NATIVE` | MT11 EEGPT | Wphysio | DINO A1 | 原生 | 原强基线，复用 |
| `E_CANON_V4` | canonical-only EEG-MAE v4 | Wphysio | DINO A1 | 原生 | E 实验对照，复用原 E1 |
| `E_POOL` | 扩大过滤池 EEG-MAE | Wphysio | DINO A1 | 原生 | E 实验候选 |
| `V_FULL_MATCH` | MT11 EEGPT | Wphysio | 同 ROI 时间格的全幅 VideoMAE | common | V 实验直接对照 |
| `V_ROI` | MT11 EEGPT | Wphysio | 已有 2.0× ROI VideoMAE | common | V 实验候选 |
| `B0_VIDEO_COMMON` | MT11 EEGPT | Wphysio | DINO A1 | common | V 实验的强基线匹配对照 |

主要配对差：`E_POOL − E_CANON_V4`、`V_ROI − V_FULL_MATCH`。强参考差：`E_POOL − B0_NATIVE`、`V_ROI − B0_VIDEO_COMMON`。旧全幅 V1 作为单列历史参照，不混入同时间格主要配对差。

E 与 V 各自以原 B0 bag 为起点，只替换自己的模态。`E_POOL` 的 Video 继续使用 DINO A1；`V_ROI` 的 EEG 继续使用 MT11 EEGPT。两项改进的收益分别计算。

### 5.2 默认预算

- 新正式 SSL：`E_POOL,V_FULL_MATCH,V_ROI × 2 protocols × 1 seed = 6` 个 cell。
- 核心输入对照：`E_CANON_V4,E_POOL,V_FULL_MATCH,V_ROI × 2 protocols × 3 seeds = 24` 条；其中 E_CANON_V4 的 6 条可复用，新增 18 条。
- 基线：B0_NATIVE 的 6 条复用；B0_VIDEO_COMMON 的 6 条在共同 mask 有变化时重跑。当前已知 9 行可用性差异，因此默认计入这 6 条。
- 预计正式情绪下游新增 24 条，汇总表共 36 条（含 12 条复用）；smoke 独立保存，不计入正式结果。

复用前核对对应 checkpoint、token、bag、模型实现和配置指纹。若旧对照无法满足同版本合同，在独立目录补跑对应控制，不将历史值视为精确配对。相同步数归因检查与 ROI 原生全覆盖实验单列为后续预算。

## 6 执行阶段与验收

### Stage 0：冻结输入与配置

生成两协议 EEG 扩大池 manifest、视频双路时间映射/common mask 和 `fixed_config.json`。记录原始输入路径、split 指纹、归一化指纹、代码 SHA 与 working-tree 源码快照。检查服务器可写 staging、环境和已有队列，实际启动时重新确认运行条件。

通过条件：canonical 身份与 split 一致；新增 raw 与 val/test 实际支持区间交集为 0；两路视频 train/val 行、采样时间格和 common mask 完全一致；改进因素之外的配置差异为 0。未通过时修复输入或映射并重验，正式训练在通过后开始。

### Stage A-smoke：输入、数值与表征检查

- EEG：两协议各 35 epoch，train/val 各按固定代表性规则最多取 1,024 行；训练子集覆盖 canonical 与新增 raw 来源，验证仍取 canonical val。
- Video：两路×两协议各 15 epoch，train/val 各最多 1,024 个共同有效窗口。
- smoke 使用独立目录，patience 设置为 smoke 长度，跳过全量 token 导出；正式模型重新初始化，完整正式 train 不继承 smoke 权重。

验收输入 shape、固定掩码、初始化可复现、loss/梯度有限、每轮 health 以及失败来源。EEG 沿用 v4 的典型窗口 health 和选中 checkpoint 的 `val/zero < 1` 门槛；Video 沿用 v2 health，并补记零预测重建诊断。两种视觉输入各自报告重建损失，不按跨视野重建损失排序情绪性能。

### Stage A-formal：正式 SSL 和 canonical 导出

6 个 cell 按固定配置训练、canonical val 选择 checkpoint，随后对全 canonical 宇宙做冻结导出。每份 token 检查 `(28819,256)`、有限值、唯一且相同顺序的 `sample_id`、预期 mask、无效行严格为零；EEG 原始 normalization 来源与冻结参数一致。

视频仅复制 checkpoint、config、token、审计及小型报告到 H20，保留 ncc clip 与 decoded cache。复制前后 SHA256 相同。跨主机 token 身份以 canonical index 为准。

### Stage B：冻结下游三 seed 对照

先对每条新增路线、每个协议跑 seed240800 的 3 epoch 接口 smoke；通过后完成全部预定正式 seed。接口 smoke 检查 bag 的 event/split/label 成员不变、非目标模态 token 不变、目标 slot 与 mask 替换正确、指标与 loss 有限，不用于效果筛选。

正式阶段沿用第 2.2 节配置，保存逐 event 的 val/test prediction、目标、subject/day/event 身份以及完整训练历史。最终表必须覆盖预期的 protocol×route×seed，存在失败或缺失时按 cell 标记状态，完成标记在完整验收后写入。

## 7 指标、判定与下一轮选择

### 7.1 固定报告口径

主预测读数为 11 情绪等权宏平均 event-level raw Pearson r；共同报告 within-subject centered r、RMSE、standardized RMSE 和逐情绪指标。每个 seed 先对 11 标签求均值，再对 3 seeds 求均值及样本 SD（`ddof=1`）；差值先按相同 seed 相减后汇总。

保存配对逐 event 预测，按相同 subject-day block 在每个 seed 内做 2,000 次 paired bootstrap，报告 delta 区间。三 seed SD 与 bootstrap 区间分别标记。该轮仍沿用已多次查看过的既有数据划分，结果归为预先固定方案下的探索性输入验证；上游单 seed 的不确定性单列。

结果报告顺序：

1. 输入事实：实际 raw 保留数、有效新增覆盖、两协议排除比例、视频共同有效行与失败数。
2. 技术读数：重建/零预测比、表征变化、effective rank、best epoch/steps、运行时间。
3. 独立改进：两项主要配对差及三 seed 方向，逐情绪变化、centered r 与误差。
4. 强基线距离：与各自匹配 B0 的配对差，区分“改善旧 MAE”和“超出强基线”。

### 7.2 推进条件

Checkpoint 始终按既定验证损失选择。候选进入下一轮的筛选只看完成的三 seed val：主要对照的 macro raw r 平均 delta 大于 0、至少 2/3 seed 为正，同时 macro centered r 平均 delta 不小于 `−1e-6`、macro standardized RMSE 平均 delta 不大于 `1e-6`。这是预设的推进规则，不作为显著性证明。

两协议分别判定。cross_day 作为跨日期输入收益的主要读数，within_subject_day 作为不同留出方式的并列诊断；保留完整两协议结果。只在一个协议通过时将候选限定为对应场景，不合并两协议均值掩盖差异。

Test 在路线、配置和全部 seed 完成后按预定口径报告，不据 test 重调 mask、过滤、epoch、seed 或损失。3/3 seed 同向且 centered/误差一致支持稳定方向；seed 分歧与局部标签收益保留为探索性发现。超过 B0 的判断仅使用本轮对应的可用性匹配参考。

### 7.3 历史定位值

| 已完成路线 | cross_day 宏 raw r | within_subject_day 宏 raw r |
| --- | --- | --- |
| B0_NATIVE | `0.3733 ± 0.0117` | `0.4373 ± 0.0045` |
| E_CANON_V4 / 原 E1 | `0.2682 ± 0.0085` | `0.3276 ± 0.0047` |
| 原全幅 V1 | `0.3582 ± 0.0155` | `0.3925 ± 0.0099` |

这些值来自最终 handout / v4 汇总，供理解原始差距；`V_FULL_MATCH` 和 `B0_VIDEO_COMMON` 的值由本轮匹配运行产生。

## 8 实现落点与产物组织

### 8.1 所需实现

| 当前入口 | 本轮需要的最小扩展 |
| --- | --- |
| [112_run_modality_mae.py](../../../../../scripts/multilabel/112_run_modality_mae.py) | EEG 训练支持带来源身份的 canonical/raw 联合行表；复用指定 normalization；canonical val 与导出数据保持独立 |
| [modality_mae.py](../../../../../src/daily_multimodal/training/modality_mae.py) | 训练取批接受联合来源；模型、遮挡、loss 与 pooling 沿用 v4；审计增加来源/steps |
| [114_run_video_mae.py](../../../../../scripts/multilabel/114_run_video_mae.py) | 构建/读取 ROI 本地帧与全幅匹配时间格 cache；记录 view、frame mapping 与 common mask 来源 |
| [118_run_mae_mt11_event_ablation.py](../../../../../scripts/multilabel/118_run_mae_mt11_event_ablation.py) | 继续用 E1/V1 的单 slot 替换；为本轮 variant 独立命名，并支持原 DINO A1 共同 mask 的匹配参考 |

计划制定时（执行改动前），112 强制 EEG X 与 canonical 28,819 行一致，114 按原始录像时间解码，118 的 route 命名区分 E1/V1 等条件。因此本轮需要联合池、ROI 读取和 variant 来源的代码扩展；已执行实现和门槛状态见第10节。

实现时验证 raw/canonical 行空间隔离、时间区间过滤、normalization 原样复用、ROI 帧映射及 cache 来源、共同 mask、非目标模态保持原样和导出身份。现有 MAE 修复回归继续覆盖位置编码、mask、数值与固定验证行为。

### 8.2 计划输出位置

以下目录为计划位置，执行时创建独立目录并保留全部旧输入与结果：

| 主机 | 计划根 |
| --- | --- |
| ncc 视频 staging / cache / SSL | `/home/lzs/mae_round1_input_20261010` |
| H20 EEG / 汇总 staging | `/home/wangzw/mae_round1_input_20261010` |
| 本地轻量同步 | `outputs/server_sync/mae_round1_input_20261010` |

```text
inputs/
  fixed_config.json
  eeg/<protocol>/ssl_train_manifest.jsonl
  eeg/<protocol>/filter_report.json
  video/frame_mapping.jsonl
  video/common_mask.npz
  video/cache_report.json
stage_a/<variant>/<smoke|formal>/<protocol>/<modality>_seed_240800/
  checkpoint.pt
  config.json
  window_embeddings.npz             # formal
downstream/<variant>/<smoke|formal>/<protocol>/seed_<seed>/
  metrics.json
  event_predictions.npz
  best_checkpoint.pt
reports/
  input_audit.json
  paired_metrics.csv
  per_label_metrics.csv
  results.json
  round1_input_report.md
```

源指纹、实际配置、输入/过滤计数、运行 steps 和 completed/failed 状态写入产物；`eeg` 与 `video` 的完成状态独立，汇总完成标记以预期 cell 全部有效为准。程序错误按既有授权修复并恢复失败阶段，保留已完成有效 cell 和协议合同。

## 9 本轮交付标准

第一轮完成时应有：两协议经过边界过滤的 EEG 扩大池清单；可追溯的 ROI/全幅同时间格输入与共同 mask；6 个预定正式 SSL cell 的模型及 canonical token；两项独立输入对照与匹配 B0 的完整三 seed 下游表；能够分别回答 ROI 输入和扩池 EEG 是否改善原 MAE、是否接近或超出各自强参考的配对报告。

## 10 执行记录（2026-10-10）

- 已实际登录 H20 与 ncc，创建本计划声明的独立 staging。执行源码快照、固定合同及轻量同步根为 `outputs/server_sync/mae_round1_input_20261010/`。
- `133` 从两库 metadata 独立还原 subject/session；在 raw 来源的54个 session 上核验实际 FIF 的200 Hz、59通道顺序、SI电压及 canonical/raw 切片逐值相等。44,181候选的数值、全通道常量和内容重复检查完成。
- cross_day 保留 raw 44,181行，加 canonical 16,813行，总训练表60,994行；新增独有时间覆盖62.0278小时；新增与原 canonical train 均无 val/test 信号区间交叠。smoke 的35轮、冻结v4归一化、典型窗口health及选中 checkpoint 的 val/zero 门槛均通过。正式 SSL 已完成100轮并选中第98轮；canonical token和三个seed冻结下游验收通过，详见本节完成记录。
- within_subject_day 的原 canonical train 有107行与 val/test 10秒信号区间相交。实际预处理对整段 session 做滤波和 ICA 拟合；进一步采用整段预处理上下文过滤后，该协议新增 raw 保留0行，原 canonical train 的17,243行均与 holdout 共享预处理 session。保留过滤前清单和上下文审计，状态为 `stopped_canonical_signal_overlap`，按第4.3节停止该协议的扩大池严格对照。cross_day 的训练与 holdout 预处理 session 无交叠，44,181行保留数不变。
- `134` 对完整16帧 ROI 使用本地 `1,3,5,7,9,11,13,15` 并按源脚本舍入映射到同一原始录像帧号；侧文件缺少短 clip 的成功写入帧映射，因此短 clip 按合同排除。当前使用8个 worker、只读 OpenCV 按已指定源帧号排序读取；逐 source 任务可恢复已验收映射。实际同一批40帧与原缓存逐像素相同，顺序读取90.78秒、逐帧seek194.66秒；新读取器的临近帧、重复帧和远距seek回归通过。源码迁移保存原builder指纹和 `decoder_revision.json`，其余输入来源指纹完全一致；采样、像素变换与输入合同保持一致。
- `112` 支持独立 canonical/raw 行表及指定v4 normalization；`114` 验证配对 view/cache 身份并保存初始化、零预测诊断、steps；`118` 增加 E_POOL、V_FULL_MATCH、V_ROI 和 B0_VIDEO_COMMON，检查共同 video mask，并复用完成 cell。
- `135` 按 smoke→formal→冻结三seed下游的门槛执行；`137` 是本地一次性跨主机控制器，在全部视频输入冻结后运行两协议两视野 smoke/formal，复制并校验小型模型/token产物，再启动 H20 视频下游；`136` 产生逐情绪、配对及每seed subject-day bootstrap报告。
- 两台服务器各48项 MAE 回归通过；补充的第49项覆盖配对 bootstrap 的 event 成员约束，H20通过。复用v4对照的下游源码哈希一致，EEG架构、patch转换、遮挡、重建及验证函数的 AST 与v4一致。

原合同预期6个SSL cell和36个下游cell继续保留。由于同日EEG触发输入停止条件，可执行范围为5个SSL cell和33个下游cell；另3个E_POOL下游cell标记stopped。最终将写入 `ROUND1_EXECUTION_FINISHED_WITH_PROTOCOL_STOP`，原六cell全部成功的完成标记不会用于本轮状态。阶段完成和效果结论分别验收。

执行恢复：cross_day EEG 首次正式训练在 epoch39、batch70 的梯度检查处停止，前38轮的日志保留；首次运行未保存中间权重，因此按原seed重放。`112 --recovery-checkpoint` 现在每轮原子保存模型、优化器、最佳状态、selector等待计数及全部随机状态，并在梯度错误时保存失败batch用于诊断。恢复参数不改变模型、数据、mask、损失、优化器或预算。包含模拟中断的6项 round1 测试通过，恢复后的训练历史和选中模型与不中断运行逐值一致。`137` 同时修复了SSH超时退出的问题，短暂连接错误按既有重试上限处理。

同位置重放已保存实际失败batch并定位：PyTorch 2.5.1 CUDA efficient-attention 在第一层 attention 反向传播出现NaN；原始输入及前向输出有限。对同一模型状态、窗口与mask，默认后端复现NaN，math SDPA后端梯度全部有限。`112 --retry-attention-math` 仅在梯度错误且优化器尚未更新时重算该batch，通过有限性检查后更新一次，再恢复被丢弃默认前向的CUDA随机流，后续mask/dropout流保持原seed序列。两主机53项相关回归通过（H20按运行分工跳过OpenCV测试），新增CUDA目标测试证明一次优化器更新、零跳过行与随机流逐值一致。正式训练从第38轮断点恢复，已越过原错误位置；每次恢复的loss差异、batch位置与更新次数写入 `attention_retry_records`，原失败batch和后端诊断保留。

计算台账 `inputs/execution_recovery/compute_recovery_ledger.json` 由两次实际epoch39/batch70 traceback、逐值相同的38轮日志及128行失败batch推导：首次完整丢弃尝试与第二次中断的部分epoch共丢弃18,266次优化器更新、2,335,948次训练窗口前向访问。第二次重放的完整38轮由断点复用。最终技术表分别记录成功路径的steps/seen_windows、数值重算访问和上述丢弃开销；训练前向访问总数排除smoke、验证、health probe及诊断profiling。未被首次计时或完整断点覆盖的失败耗时保留为不可用。

视频 I/O 并发调整（2026-10-10 17:05 +08:00）：实际资源检查显示CPU和内存仍有余量，视频CIFS共享目录同时有4个请求在途；将解码worker从4增至8，原父/子进程退出后续读已完成清单。切换时3,233条映射及其两路缓存的全部像素哈希在恢复后逐值一致，input_sources与builder SHA256保持原值。仅调整I/O并发，帧号、像素处理及训练合同保持固定；实际8个子进程已核验，Windows接续控制器恢复。当前日志为 `logs/video_inputs_sorted_w8.log`，审计为 `inputs/video/decoder_parallelism_20261010.json`，原日志保留。尚不对异质source组的速度变化作配对性能结论。

EEG 完成验收（2026-10-10 16:43 +08:00）：正式100轮历史及selected health通过，best epoch98的 val masked NMSE为0.5333835273，val/zero为0.6071299508，典型窗口相对变化为0.6916706945。canonical `(28819,256)` token的顺序、全有效mask和有限值通过，token SHA256为 `b618f9f9befc2c21d96206eb2d4a4edab316079a9753f6d437dced2986ef4437`。训练完成47,700次优化器更新、6,099,400次成功路径窗口访问；1,861次math重算增加238,208次前向访问，跳过行数为0。三个下游seed分别选中epoch3/24/19、运行18/39/34轮；11情绪五类val/test指标有限，配对事件/目标/subject/day身份及非EEG槽逐值一致。服务器 `EEG_STAGE_A_FORMAL_COMPLETE` 与 `EEG_STAGE_B_FORMAL_COMPLETE` 均存在，失败标记不存在。独立验收为 `inputs/eeg/completion_audit.json`，本地[验收副本](../../../../../outputs/server_sync/mae_round1_input_20261010/eeg_completion_audit.json)与正式config已核验传输SHA256。

EEG 验证推进判定：完整三seed val 的 E_POOL − E_CANON_V4 宏 raw r 为 `+0.01085586 ± 0.01792069`（2/3正向），centered r为 `+0.00408351 ± 0.00342480`；sRMSE为 `+0.0000231032 ± 0.00122359`，超过第7.2节的 `1e-6` 容差，联合门槛未通过。上述SD使用ddof=1；每seed的subject-day paired bootstrap均为2,000次，完整区间、逐情绪差及强参考距离保存在 `inputs/eeg/validation_comparison.json` 与本地[EEG验证记录](../../../../../outputs/server_sync/mae_round1_input_20261010/eeg_validation_report.md)。该单独判定读取val预测，test预测保持未读取；相同步数归因及其他下一轮扩展保持未启动。

Evidence status: cross_day EEG正式分支、三seed下游及val推进判定已完成；视频cache运行中。Video smoke/formal、视频冻结下游及完整配对报告尚待门槛验收，后续状态以服务器产物及完成标记为准。
