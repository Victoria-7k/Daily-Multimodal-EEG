# MAE 第二轮下游适配实验计划

日期：2026-10-10，Asia/Shanghai。状态：cross_day的核心A/B/C与D统一test报告均完成，36个唯一融合cell及4个EEG-only模型已验收；within_subject_day保留协议停止。A两路、B实用FT与C两表征新损失均未通过预声明val推进门槛；A的canonical适配test raw r +0.008106（3/3正向），B的层数收益在test反向，C整体收益未成立。保留原C冻结E1与window MSE；可选联合确认保持独立预算。

本轮分别检验 EEG 单模态情绪适配、保守微调和事件级损失。第一轮改变无标签预训练输入，第二轮研究如何把已有 MAE 表征转化为情绪预测能力。三个主实验保持输入来源、监督范围、结构和评价口径可追溯，先建立独立配对收益，最后再考虑联合配置。

前序合同见[第一轮输入改进计划](mae_round1_input_improvement_plan_20261010.md)；历史 v4、MT11 基线及部分微调设置见[MAE 交接手册](mae_chat_handout_20261010.md)。第一轮继续按自己的控制器与验收规则执行；本计划不改变其队列、推进门槛或完成状态。

## 1 研究问题与三项干预

| 子实验 | 改动发生在哪里 | 干预 | 核心对照 | 证明作用 |
| --- | --- | --- | --- | --- |
| A：EEG 单模态情绪适配 | SSL 与多模态融合之间 | 用 11 情绪标签单独更新 EEG encoder 尾部，导出新 token | 同一 SSL encoder 的冻结表征；另设 EEG-only 冻结读出头 | EEG 重建表征接受任务监督后，是否更适合情绪预测和融合？ |
| B：保守微调 | EEG 与原融合器联合训练时 | 降低 EEG 尾部学习率，再减少解冻层数 | 同一初始化的 EEG-only 联合微调，固定 head LR | 更小参数更新能否改善泛化并保留预训练表征？ |
| C：事件级损失 | 融合训练的误差计算 | 保留 23 窗均值聚合，改为 event MSE 或 event MSE 加弱窗口一致性 | 当前逐窗口复制标签 MSE；B0 同步改损失 | EMA 标签粒度与训练目标的对齐，是否改善预测？ |

三个主实验各自从固定基点出发。A 输出的情绪适配 encoder 暂不作为 B/C 的输入，B 的最佳微调模型暂不作为 C 的输入；这样可以分别解释三个因素。联合配置在第 9 节的独立确认阶段处理。

## 2 承接第一轮的产物与协议边界

### 2.1 两种固定 EEG 初始化

| 来源 ID | 含义 | H20 checkpoint / token 目录 |
| --- | --- | --- |
| `C` | 最终 v4 canonical-only EEG-MAE | `/home/wangzw/mae_norm_channel_20261009/outputs/stage_a_train_channel/formal/<protocol>/eeg_seed_240800` |
| `P` | 第一轮过滤扩大池 EEG-MAE | `/home/wangzw/mae_round1_input_20261010/stage_a/E_POOL/formal/<protocol>/eeg_seed_240800` |

每个目录的来源文件为 `checkpoint.pt,config.json,window_embeddings.npz`。两路均保持 6 层、8 heads、256D、10 个 1 秒 patch 的 EEG encoder，以及各自已验收的固定原始信号 normalization。C/P 在第一轮使用相同的 v4 通道尺度参数，该相同性在第二轮 Stage 0 再次核对。

按当前本地第一轮完成记录，cross_day P 的 SSL train 为 60,994 行，正式 SSL、canonical 导出和三 seed 冻结下游已验收。其相对 C 的 val 宏 raw r 为 `+0.01085586`，centered r 为 `+0.00408351`，standardized RMSE 为 `+0.0000231032`，原联合推进门槛未通过。记录见[第一轮 EEG 验证判定](../../../../../outputs/server_sync/mae_round1_input_20261010/eeg_validation_report.md)。

因此第二轮采用 **C 为 B/C 主基点，C 与 P 同时进入 A 的预定义来源对照**。P 的加入用于研究“扩大池表征是否需要情绪适配”的独立问题；保留其第一轮 `gate_passed=false`，不把它自动升级为主参考。具体数值仅说明承接条件，第二轮参数不从第一轮 test 结果选取。

### 2.2 可执行协议与停止状态

| 协议 | 第二轮合同 | 当前可执行状态 |
| --- | --- | --- |
| `cross_day` | 被试内跨日期泛化，完整执行 A/B/C | 以来源 checkpoint、split、预处理 session 隔离和完成验收均通过为前提 |
| `within_subject_day` | 保留完整 A/B/C 设计，作为条件执行协议 | 当前暂停严格新对照：原 canonical train 有 107 窗与 val/test 信号区间交叠，整段滤波/ICA 上下文也共享 session |

同日审计见[第一轮过滤报告](../../../../../outputs/server_sync/mae_round1_input_20261010/inputs/eeg/within_subject_day/filter_report.json)。当前 P 在该协议没有可用扩池模型；原 C/MT11 历史结果保留其原解释范围。

恢复同日严格对照需要一份经审计的独立 split/预处理合同，并在该合同上重建 C、P（如有合格新增输入）、MT11 参考及全部配对下游。该协议修订不在本轮实施范围内，不静默删除重叠 train 行或复用不同划分的旧指标。报告为计划 cell 保留 `stopped_protocol_overlap` / `pending_protocol_revision` 状态。

本计划制定时根据已同步产物确定这些边界；执行前重新核验服务器产物与来源一致性。视频 ROI 仍按第一轮执行，本轮主实验保持 DINO A1，因此 EEG 分支具备独立执行条件。

### 2.3 数据与强参考路径

| 项目 | H20 文件路径 |
| --- | --- |
| Canonical EEG | `/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new/X.npy`，`(28819,2000,59)` |
| 11 情绪标签 | 同目录 `y.npy`；通过 index/label-order 审计后按原 event 取标签 |
| Canonical index | `/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/index/eeg_aligned_window_index.jsonl` |
| Canonical split 根 | `/vePFS-0x0d/DailyEEG/splits_new` |
| 强参考根 | `/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/outputs/multiemotion_20260913/structure_matrix_A1` |
| 原参考 bag | 强参考根下 `bags/<protocol>/A1_Wphysio_no_audio__eeg_eegpt_partial_ft_multitask_11label_v1/seed_240800/ema_bags.npz` |
| B0 保存预测与指标 | 强参考根下 `runs/<protocol>/window_attention_regression_full_mean/seed_<seed>/{event_predictions.npz,metrics.json}` |
| C 冻结融合指标 | `/home/wangzw/mae_norm_channel_20261009/outputs/downstream_train_channel/single/formal/runs/<protocol>/E1/seed_<seed>`，复用时以实际 manifest 定位为准 |
| P 冻结融合指标 | `/home/wangzw/mae_round1_input_20261010/downstream/E_POOL/formal/runs/<protocol>/E_POOL/seed_<seed>`，复用时以实际 manifest 定位为准 |

监督阶段只读取原 canonical train event 对应的信号与标签。Raw pool 留在 SSL 阶段，不为它生成伪情绪标签，也不赋予相邻 EMA 标签。整个 canonical 宇宙的 token 导出属于冻结推理。

## 3 固定的共同实验合同

### 3.1 身份、种子与监督

| 项目 | 固定值 |
| --- | --- |
| Canonical 身份 | 28,819 行，同 `sample_id`、行顺序与窗口成员 |
| Event bag | 1,253 event，每 event 23 窗，沿用原 bag 的 event/split 成员 |
| cross_day event 数 | train/val/test = `731/269/253` |
| 同日原 event 数 | `749/246/258`，作为原合同记录，当前不据此启动严格对照 |
| 标签顺序 | inspired、alert、determined、attentive、active、hostile、nervous、upset、afraid、ashamed、fatigue |
| SSL seed | 原来源 `240800`，本轮不重做 SSL |
| A 的情绪适配/EEG-only probe seed | `240800`，每个来源各自训练；共同头初始化保持匹配 |
| 融合 / 联合 FT seed | `240800,240801,240802` |
| 监督单位 | event，23 窗属于同一标签；每个 event 的损失权重相同 |
| 标签标准化 | 仅由原 train event 的 11 标签均值/标准差拟合，各路线完全相同 |
| 模型选择 | event-level `val_macro_standardized_rmse_min` |
| Test | 参数、路线与种子完成冻结后报告，不用于选层数、学习率、lambda 或来源 |

A 只有一个监督适配 seed，三融合 seed 的 SD 不覆盖上游适配 seed 波动。B 的三个 seed 各自包含 EEG 尾部和融合器的联合训练波动。两者分别标记。

### 3.2 多模态融合配置

正式强参考为 `A1_Wphysio_no_audio__eeg_eegpt_partial_ft_multitask_11label_v1`。

| 项目 | 固定值 |
| --- | --- |
| condition / model / temporal policy | `window_attention_regression_full_mean` / `window_replicated` / `uniform` |
| head | `H1_shared2_11xhead2` |
| hidden dim / normalization / adapter | `128` / `per_modality` / `per_modality` |
| Wear / Video / Audio | Wphysio / DINO A1 / 关闭 |
| Wear / Video encoder | 使用原 token，冻结；不参加 A/B/C 的参数更新 |
| 优化器 / head LR / weight decay | AdamW / `1e-3` / `1e-4` |
| 最大 epoch / patience / event batch | `80` / `15` / `64` |
| dropout / modality dropout | `0.1` / `0.1` |
| 梯度裁剪 | global norm `1.0` |
| 默认损失 | 现有逐窗口 MSE；仅 C 改变该项 |

固定 token 的路线按各自 train token 拟合 normalization。实时更新 EEG encoder 的 A/B 路线在训练前用各自初始 SSL train token 拟合 feature normalization，并在整个该阶段固定；不得随 epoch 或 val/test 重新拟合。A 导出适配后 token 后，后续冻结融合按原 train-only 算法重新拟合下游 token normalization。

## 4 A 实验：EEG 单模态 11 情绪适配

### 4.1 路线与监督边界

```text
已验收 C 或 P SSL checkpoint
        ↓
EEG encoder → 256D token → EEG-only 11情绪头
        ↓
原 train event 标签更新头与指定 EEG 尾部；val 选模型
        ↓
移除任务头，冻结 encoder，导出 canonical 256D token
        ↓
替换原 B0 bag 的 EEG slot，训练原冻结融合器与11-head
```

MAE 的重建 decoder 不进入适配与融合。情绪适配 token 标记为 `11label_supervised_eeg_adaptation`，保留其 `ssl_source=C|P`、适配 split、seed、目标标签、可训练层和模型选择来源。它与未适配的 label-free SSL token 分开保存。

### 4.2 EEG-only 匹配配置

每个窗口先由原 MAE encoder 无遮挡编码、10 个时序 token mean pooling 为 256D；固定 train-only feature normalization 后，通过 `Linear(256,128)+GELU` 输入共同的 `H1_shared2_11xhead2`。A 的两种训练模式使用相同头架构、初始化、dropout、训练 event 顺序和标签标准化。

| EEG-only ID | 初始化 | EEG 可训练范围 | 用途 |
| --- | --- | --- | --- |
| `A_C_PROBE` | C | 完全冻结，仅训练任务头 | 测量原 canonical SSL 情绪可读性 |
| `A_C_ADAPT` | C | encoder 最后 2 个 Transformer blocks 与任务头 | 测量单模态情绪适配收益 |
| `A_P_PROBE` | P | 完全冻结，仅训练任务头 | 测量扩池 SSL 情绪可读性 |
| `A_P_ADAPT` | P | encoder 最后 2 个 Transformer blocks 与任务头 | 测量扩池表征的情绪适配收益 |

`PROBE` 使用可学习的同构情绪读出头，属于冻结 encoder 的非线性读出对照；它不称为线性 probe。PROBE 的训练不会改变原 token，ADAPT 才产生新的表征。

固定参数：encoder LR=`1e-5`（仅 ADAPT）、head LR=`1e-3`、AdamW weight decay=`1e-4`、head dropout=`0.1`、最大 80 epoch、patience15、batch64 event、gradient clip1.0。每批按 event 取 23 窗，窗口预测按 valid mask 计算当前 window MSE，再对 event 与11标签等权平均。EEG-only 阶段没有多模态 dropout。

Stem、position 和 encoder 前 4 层冻结。Encoder 全程使用确定性的 eval dropout 行为，指定末两层保留梯度；任务头使用 train 模式。保持已有 FP32 数值检查与 math attention 后端配置。此处采用与现有 MT11 上游 encoder LR/末两层范围对应的起点；模型架构、头结构和训练批次仍属于 EEG-MAE 的新实现，不把它称为复现 EEGPT。

### 4.3 导出与冻结融合对照

| 融合 ID | EEG slot | 配对作用 |
| --- | --- | --- |
| `F_C` | C 原 SSL token | 复用最终 v4 E1，亦为 B/C 公共基点 |
| `F_P` | P 原 SSL token | 复用第一轮 E_POOL |
| `F_A_C` | A_C_ADAPT token | 与 F_C 比较单模态适配 |
| `F_A_P` | A_P_ADAPT token | 与 F_P 比较单模态适配 |

四路均保留 Wphysio、DINO A1 和原生 modality mask。每路融合使用三个匹配 seed。导出仍为 `(28819,256)`、同 canonical 顺序、全 true EEG mask、有限值；投影和情绪头不替代 encoder 的256D输出。

主要差为 `F_A_C−F_C` 与 `F_A_P−F_P`；EEG-only 差为同来源 ADAPT−PROBE。输入与适配的交互差预先定义为：

\[
\Delta_{\mathrm{interaction}}=(F_{A,P}-F_P)-(F_{A,C}-F_C)
\]

上述差对每个 metric、每个匹配 seed 单独计算。F_A_P−F_A_C 可以回答适配后两种 SSL 初始化的差距；原始融合收益与该交互分别报告。

EEG-only 能力提高与融合能力提高分别验收。仅 EEG-only 变好时，结论限定为单模态任务适配，不据此宣布多模态收益。

## 5 B 实验：保守的 EEG-only 联合微调

### 5.1 固定 C 初始化，逐因素比较

这里“EEG-only”指多模态融合中只有 EEG encoder 接受梯度；Wphysio/DINO A1 仍作为冻结输入参加融合。融合器、adapter 和11-head照常训练。

| ID | EEG 可训练层 | EEG LR | 融合/head LR | 损失 | 要分离的因素 |
| --- | --- | --- | --- | --- | --- |
| `F_C` | 无 | — | `1e-3` | window MSE | 冻结基点，复用 |
| `B_T2_STD` | 最后 2 层 | `1e-4` | `1e-3` | window MSE | 对应历史尾部 LR 的新 EEG-only FT 控制 |
| `B_T2_LOW` | 最后 2 层 | `1e-5` | `1e-3` | window MSE | 仅降低 EEG LR |
| `B_T1_LOW` | 最后 1 层 | `1e-5` | `1e-3` | window MSE | 在低 LR 下仅减少解冻层数 |

主要差：`B_T2_LOW−B_T2_STD` 分离学习率作用；`B_T1_LOW−B_T2_LOW` 分离层数作用。每条 FT 同时与 F_C 比较，判断它是否优于冻结融合。

历史 M5-FT 同时更新 EEG/Wear/Video 三条 MAE encoder，初始化和输入来源也不同，不能作为 B_T2_STD 的精确控制。因此 B_T2_STD 在本轮独立训练。

### 5.2 微调行为与数值稳定

- 每个 seed 从同一 C SSL checkpoint 重新开始，三个 FT 候选的 head/adapter 初始参数与随机流匹配；不会从其他 FT 候选的 best checkpoint 接着训练。
- 构造或加载encoder尾部时保护head初始化之后的CPU/CUDA随机状态，或使用独立随机流，避免末1/末2层的模块构造消耗改变融合dropout。初始化与随机状态实际比对通过后，才认定同seed匹配；modality dropout与batch shuffle也保持独立、相同的规则。
- Stem、position、早期 blocks 冻结；mean pooling 与原 token 输出方式固定。末 1/2 层分别构建正确切点的 prefix，避免将一个已训练 block 缓存在冻结输入中。
- 使用初始 C train token 的固定 normalization，沿用原 batch64、window chunk128、80 epoch、patience15、head/modality dropout。核心 B 不增加 warmup、L2-SP、LoRA 或 head LR 扫描。
- Encoder 尾部保持 eval dropout、参数可训练；融合器保持 train 模式；沿用 FP32 math attention 和有限性检查。
- 训练前检查 prefix+tail 无更新推理是否还原初始 token；采用已有最大绝对误差 `2e-4` 门槛与对应批量大小审计。
- 首批保存可训练尾部的有限非零梯度与参数变化；末层之前参数、Wphysio/DINO输入及原始 normalization 必须保持不变。

若需要程序级恢复，保存模型、优化器、best selector、epoch/batch和全部随机状态；恢复不能跳过训练窗口。程序错误按既有恢复授权处理，科学参数保持原合同。

### 5.3 如何评价过拟合是否改善

记录每 epoch 的 train window MSE、train event sRMSE、val event sRMSE/raw r/centered r、best epoch、tail 参数相对初始值的 L2 偏移、典型 val token 相对变化及有效秩。Train 诊断采用固定无 dropout 的推理口径，避免与训练期随机 loss 混比。

判定以三 seed 的配对 val 泛化指标为准；训练误差下降、best epoch 推迟或参数偏移变小只提供解释。尾部几乎不更新时同时检查梯度、参数 delta 与预测，避免把“更新无效”解释为泛化改善。

Warmup、进一步降低 head LR 或权重偏移约束可作为后续单因素扩展；在核心 B 完成前不混入多个正则化策略。

## 6 C 实验：事件级损失与弱一致性

### 6.1 保持模型与聚合不变

对于一个 event 的一个情绪，设有效窗口数为 K、窗口预测为 p_t、标准化标签为 y。沿用现有有效窗 uniform 聚合：

\[
\bar p=\frac1K\sum_{t\in\mathrm{valid}}p_t
\]

当前损失及其分解为：

\[
L_{\mathrm{window}}=\frac1K\sum_t(p_t-y)^2
= (\bar p-y)^2+\frac1K\sum_t(p_t-\bar p)^2
=L_{\mathrm{event}}+L_{\mathrm{consistency}}
\]

定义本轮比较目标：

\[
L_{\lambda}=L_{\mathrm{event}}+\lambda L_{\mathrm{consistency}}
\]

最后在 event 和11标签上等权平均。固定 λ=`1,0,0.1`：1 为当前 window MSE，0 为纯事件 MSE，0.1 为事件 MSE 加弱窗口一致性。lambda 在查看第二轮结果前锁定，不追加扫描其他值。

λ=1 保留当前 window-loss 原代码路径，避免以等价公式的浮点舍入差异改变可复用控制。λ=0/0.1 使用模型已有 `prediction` 与 `window_prediction` 计算，按相同有效窗 mask 处理。输入验收要求每个 event 至少一个有效窗口；无有效窗时停止该 cell 并报告身份。

当前模型已在 event level 评价；C 改变的是训练误差。`window_replicated` 结构、uniform temporal weights、modality attention、normalization、adapter、head与评价过程全部固定。

### 6.2 B0 与 MAE 的成对目标对照

| 表征 | λ=1：当前 window MSE | λ=0：event MSE | λ=0.1：event + 弱一致性 |
| --- | --- | --- | --- |
| B0：MT11 EEGPT + Wphysio + DINO A1 | `B0_W`，复用 B0_NATIVE | `B0_EVENT` | `B0_EVENT_WEAK` |
| C：冻结 EEG-MAE + Wphysio + DINO A1 | `F_C`，复用 | `C_EVENT` | `C_EVENT_WEAK` |

六路都冻结 encoder，使用三个匹配 seed。B0 与 C 使用同 event、相同非EEG输入和 native mask；来源标签监督边界各自保留。该设计不需要等待新的 ROI cache。

分别报告每种表征下新损失相对旧损失的 delta，再报告“损失收益的差值”：

\[
\Delta_{\mathrm{loss\ interaction}}
=(C_{\mathrm{EVENT}}-F_C)-(B0_{\mathrm{EVENT}}-B0_W)
\]

λ=0.1 也计算相同交互。两路同向提升时解释为监督目标调整的普遍收益；MAE 取得更大收益时，才讨论该目标与 MAE 表征的适配关系。不会将 B0 原 window-loss 指标与 MAE 新 event-loss 指标作为唯一方法优劣证据。

### 6.3 解释边界与诊断

Event MSE 允许窗口预测在 event 内变化，也可能出现相反误差抵消。记录每个 event 的窗口预测方差、窗口极值、event预测方差与跨日期稳定性；这些诊断不替代 event 指标。

EMA 标签没有提供逐窗情绪真值。本轮的 p_t 仍是弱监督的窗口预测，不据其变化宣称已恢复真实连续情绪轨迹。λ=0.1 用于检验适量一致性是否保留较好的泛化；该约束与新增时间核、GRU 或 temporal Transformer 分开。

## 7 正式矩阵、预算与复用

### 7.1 每个合格协议的核心工作量

| 分支 | 正式 cell | 新增 | 可复用 |
| --- | ---: | ---: | ---: |
| A：EEG-only PROBE/ADAPT | C/P × 2种模式 × 1适配seed = 4 | 4 | 0 |
| A：冻结融合 | F_C/F_P/F_A_C/F_A_P × 3seed = 12 | 6 | F_C、F_P，共6 |
| B：冻结/联合FT | F_C、三个B候选 × 3seed = 12 | 9 | F_C，共3 |
| C：表征×loss | B0/C × 3种loss × 3seed = 18 | 12 | B0_W、F_C，共6 |

F_C 在三分支中的同一个 run 只汇总一次。每协议核心融合结果共 **36 个唯一 cell，其中 27 个新增、9 个复用**；另有4个新 EEG-only 正式 cell、2份情绪适配 canonical token及末1/末2层所需prefix。Smoke与诊断不计入正式结果。

当前 cross_day 按上述预算规划；within_subject_day 的同规模设计保持停止状态，不计入当前可执行新增预算。合格协议缺少 P 时，A 的 P 两种EEG-only模式和 F_P/F_A_P 共6个融合cell标记 unavailable，不用 C 替代并重复计数。

### 7.2 复用合同

复用 F_C、F_P、B0_W 前检查保存预测、bag、目标尺度、event身份、mask、模型配置、源码实现与初始优化参数。λ=1 的训练路径必须等同原代码，非目标模态 token 保持逐值一致。

若实现变化影响随机流、初始化、mask、loss或normalization，重跑受影响的控制，而不是把历史指标作为精确配对。补跑列入单独预算记录。历史 test 已被查看，整个第二轮按固定方案下的探索性机制与适配验证报告。

## 8 分阶段执行与验收

### Stage 0：冻结合同、来源与标签边界

建立 `fixed_config.json` 与 `source_registry.json`，固定 C/P checkpoint、normalization、原 B0 bag、split、label-order、种子、模块训练范围、λ、LR以及预计cell。保存源码快照与Git SHA/working-tree信息。

核对 EEG raw input、SSL token 与 bag 的 sample_id 映射；canonical train/val/test event 及信号/预处理支持区间维持协议隔离；A/B的训练信号只来自指定 train event，raw pool 和 holdout 不进入有标签更新。

通过条件：来源验收有效，非目标模态与原bag一致，目标标签一致，已声明改动之外的配置差异为0；跨日全部进入下一阶段，同日保留停止记录。服务器资源与运行状态在实际执行时重新检查，避免干扰第一轮控制器。

### Stage A-smoke：EEG-only 适配接口

四个A模式各跑10 epoch、seed240800，patience10，使用固定代表性 train/val 子集（各最多64个完整event），不访问test预测。各来源 PROBE 与 ADAPT 取相同 event；验证数值、label/mask、冻结参数不变及 ADAPT 梯度/参数更新。

Smoke 检查一组原 token 重建与导出形状，正式阶段从原 SSL checkpoint 与初始任务头重新开始。技术门槛包括有限loss/梯度、典型 val 表征非坍缩、PROBE encoder逐值不变和ADAPT末两层实际更新。

### Stage A-formal：适配与冻结融合

完成四个 EEG-only 模式的原train全量监督、val选择和预测审计。两个ADAPT导出canonical token；PROBE保留读出头及预测，无需重复导出原token。

新 F_A_C/F_A_P 先跑3 epoch融合接口smoke，再完成三个正式seed。验收非EEG槽、mask、event成员和target scale不变；选中checkpoint与表征来源正确；结果有限且覆盖全部预定seed。

### Stage B：保守微调

先完成末1/末2层prefix与原C token的等价性、层冻结清单和初始化匹配；三个B候选各跑10 epoch、seed240800的小型完整event子集smoke。保留冻结prefix参数、head参数及首步tail更新审计。

技术通过后从原C重新初始化，在原完整train上各跑三个正式seed。均按val event sRMSE选模型，保存每轮泛化读数、尾部偏移和选中模型预测。

### Stage C：事件损失

实现带有效窗mask的损失，并验证 λ=1 与当前损失/梯度等价、λ=0 的event聚合、λ=0.1的一致性权重及11标签等权。覆盖窗口不全有效、单有效窗口、恒定预测、相反误差抵消与batch内不同有效数的情况。

四个新C候选各跑3 epoch、seed240800的接口smoke，随后跑三个正式seed；冻结token、head初始化、模型输出/聚合、训练event顺序和modality dropout流与对应λ=1参考匹配。

### Stage D：统一配对报告

全部主实验配置和seed完成后汇总val，并按第9节生成选择记录；该记录冻结后统一报告test。A的EEG-only及融合结果、B/C的主差和交互差分别输出，不因先到达的单seed或test结果改变后续cell。

可以保存test预测供最终报告，但在候选选择阶段仅读取val叶；报告工具与选择工具分开。所有completed/stopped/failed状态都进入cell清单，完成标记以实际合格范围的全部有效产物为准。

## 9 判定、统计与联合确认

### 9.1 固定指标与统计单位

主预测读数为11情绪等权event-level raw r，共同报告within-subject centered r、RMSE、standardized RMSE、MAE及逐情绪指标。每seed先平均11标签，再汇总三个seed的均值与样本SD（ddof=1）；配对差先按相同seed相减。

保存val/test逐event预测、目标、subject/day/event身份。每seed按相同subject-day block做2,000次paired bootstrap，报告95% delta区间；交互差对四条路线共同重采样。不要只对seed平均预测做一次bootstrap。EEG-only一个适配seed的结果单列，不附三seed稳健性结论。

三个主问题各有预声明主要contrast：A的 F_A_C−F_C（P分支及其交互单列）、B的 B_T2_LOW−B_T2_STD、C的 C_EVENT−F_C（B0收益与交互共同展示）。层数及弱一致性对照提供机制证据。SD、bootstrap区间与seed方向不混称显著性；如后续增加正式多重检验，另固定检验族、方法和确认数据。

### 9.2 预设推进与选择规则

沿用第一轮的三seed val联合规则：相对指定控制的宏raw r平均delta>0、至少2/3seed为正，宏centered r平均delta≥−1e-6、宏sRMSE平均delta≤1e-6。模型的每个checkpoint仍由val sRMSE选择，联合规则只决定候选进入后续确认。

- A：分别判断 F_A_C 对 F_C、F_A_P 对 F_P；P收益保留其来源标签。EEG-only提高作为解释，融合三seed门槛决定融合候选。
- B：候选除通过其主要机制对照外，还须相对F_C通过联合规则，才能用作下一轮实用FT配置。通过者按三seed平均val sRMSE最小选择；相差≤1e-6时优先更少解冻层，再选较低encoder LR。
- C：λ=0/0.1分别对本表征λ=1判断；B0与MAE分别保存门槛，不以MAE的单独改善推断基线收益。通过者按val sRMSE最小选择，容差内优先λ=0.1以保留弱一致性。

所有未通过或方向分歧的cell仍报告。既定1e-6误差容差不随第一轮P或第二轮结果改变；若要另设具有实际意义的误差容差，应成为下一份独立合同。

与强参考比较采用同损失B0、同mask、同event、同seed。改善旧MAE、改善泛化、改善相对B0差距和超过B0分别陈述。

### 9.3 可选的联合确认阶段

核心A/B/C验收后，按冻结val选择记录定义至多一条联合候选：选定SSL来源→EEG-only情绪适配→选定EEG尾部FT范围/LR→选定event损失。分别保存去除A、去除B、恢复window损失的匹配控制，验证单因素收益在组合中是否保留；共4种配置×3seed=12个新的融合cell。

联合阶段不自动启动，另计训练预算；涉及已有适配后encoder的prefix需重新导出并验收。若只有部分因素通过，就组合通过的部分，并缩减相应消融。强参考如需新的loss而核心C尚无可复用结果，另补同损失B0的三个seed。

ROI VideoMAE接入属于后续输入×适配的扩展，待第一轮视频验收后独立建立共同mask与同损失B0参考；不将它加入本轮主A/B/C。预训练seed、情绪适配seed、独立确认划分或跨被试扩展也分别规划。

## 10 实现落点、产物与完成标准

### 10.1 最小实现需求

| 当前代码 | 第二轮需要的扩展 |
| --- | --- |
| [modality_mae.py](../../../../../src/daily_multimodal/training/modality_mae.py) | 加载C/P MAE encoder；保留原patch、normalization与embed；提供训练层清单及确定性表征检查 |
| [eeg_multitask.py](../../../../../src/daily_multimodal/training/eeg_multitask.py) | 复用label-order、event等权/聚合与评价语义；新建EEG-MAE适配入口，避免现有EEGPT专用backend误载MAE |
| [123_export_mae_frozen_prefix.py](../../../../../scripts/multilabel/123_export_mae_frozen_prefix.py) | prefix切点由固定末2层扩展为显式末1/末2层；记录checkpoint/normalization/layer-count来源及token等价性 |
| [124_run_mae_partial_ft.py](../../../../../scripts/multilabel/124_run_mae_partial_ft.py) | 支持仅EEG实时更新，Wear/Video保留固定slot；encoder与head LR独立；tail层数和variant manifest显式配置 |
| [structure_emotion.py](../../../../../src/daily_multimodal/training/structure_emotion.py) | 保留λ=1旧路径；新增event MSE/weak consistency选择与loss metadata，checkpoint/评价语义不变 |
| [118_run_mae_mt11_event_ablation.py](../../../../../scripts/multilabel/118_run_mae_mt11_event_ablation.py) | 适配token来源审计、新variant命名和同loss B0训练；原native参考只在匹配λ=1时复用 |
| [136_summarize_round1_inputs.py](../../../../../scripts/multilabel/136_summarize_round1_inputs.py) | 复用配对身份、逐情绪与每seedbootstrap逻辑；第二轮独立汇总A/B/C及交互，保留first-round gate来源 |

当前124要求三模态MAE prefix且固定末两层，当前eeg_multitask构造EEGPT backend，当前window训练路径没有上述lambda接口。表中内容是所需实现，不宣称现有参数可以直接运行第二轮。不预占脚本编号，避免与正在推进的第一轮入口冲突。

实现回归集中验证：层冻结与参数更新、EEG-only/fusion标签边界、prefix切点、normalization固定、lambda损失/梯度、原路径与随机流兼容，以及source/bag身份。程序恢复需覆盖连续与中断恢复的一致性；现有MAE修复检查继续运行。

### 10.2 计划产物目录

主计算为H20：`/home/wangzw/mae_round2_downstream_20261010`；Python沿用 `/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/runtime/envs/eegpt-gpu-min/bin/python`。本地轻量同步为 `outputs/server_sync/mae_round2_downstream_20261010`。以下为计划目录合同，实际 A 执行状态见第 11 节：

```text
inputs/
  fixed_config.json
  source_registry.json
  protocol_audit.json
  reuse_audit.json
prefixes/<source>/<protocol>/last<1|2>/
  prefix.npy
  tail.pt
  manifest.json
adaptation/<A_mode>/<smoke|formal>/<protocol>/seed_240800/
  best_checkpoint.pt
  config.json
  trainability_audit.json
  event_predictions.npz
  window_embeddings.npz              # ADAPT formal
fusion/<variant>/<smoke|formal>/<protocol>/seed_<seed>/
  best_checkpoint.pt
  metrics.json
  gradient_audit.json                 # FT
  event_predictions.npz
reports/
  cell_status.json
  selection_val_only.json
  macro_paired_metrics.csv
  per_label_seed_metrics.csv
  per_label_paired_deltas.csv
  per_seed_paired_bootstrap.csv
  interaction_metrics.csv
  overfit_diagnostics.csv
  results.json
  round2_downstream_report.md
```

最终融合checkpoint必须包含模型结构、可训练层、所有用于推理的encoder权重、SSL/适配初始化来源、固定normalization、label标准化和modality token定位，能够独立重建保存预测。冻结prefix本身可保持外部只读文件，但checkpoint/manifest必须明确其来源。

完成状态分A/B/C记录；cross_day完整验收和同日停止状态一起写入 `ROUND2_FINISHED_WITH_PROTOCOL_STOP`（如届时停止仍有效）。仅有checkpoint、GPU使用或部分seed，不构成阶段完成。

### 10.3 交付标准

第二轮完成时应具备：

1. C/P来源与协议审计，以及原强参考和可复用cell的匹配记录。
2. 四个EEG-only读出/适配模型、两个适配后canonical token，以及其冻结融合的三seed配对结果。
3. 三种EEG-only联合微调配置与冻结基点的三seed对照，层/LR因素和过拟合诊断清楚分开。
4. B0/MAE × 三种loss的完整表、每seed和每情绪差值及共同重采样交互区间。
5. 先冻结的val选择记录、随后生成的test报告和全部completed/stopped/failed状态；独立回答三个干预能否改善情绪预测，并给出其与匹配B0的距离。

Evidence status: 计划制定时，第二轮实现、目录、训练与结果均为 Planned；固定行为由 MAE、EEGPT 适配、融合、prefix 源码及第一轮验收记录支持。A/B/C/D的实现、运行与验收分别见第11/12/13/14节；核心计划已执行，可选联合确认未启动。

## 11 实验 A 执行记录（2026-10-10）

执行入口为 [139_run_round2_experiment_a.py](../../../../../scripts/multilabel/139_run_round2_experiment_a.py)，专用模型与 batch 恢复位于 [mae_emotion_adaptation.py](../../../../../src/daily_multimodal/training/mae_emotion_adaptation.py)。[140_report_round2_experiment_a.py](../../../../../scripts/multilabel/140_report_round2_experiment_a.py) 只读取 val 预测，冻结 A 的选择记录并生成逐 seed、逐情绪与四路线共同重采样的交互报告。原融合 trainer 保持与 v4/round1 字节一致，F_C/F_P 的六个 cell 继续精确复用。

H20 根为 `/home/wangzw/mae_round2_downstream_20261010`；当前使用 GPU 0。Stage 0 已核对原 X/y/index 的 28,819 行身份、11 标签顺序、731/269/253 event、canonical split、跨 session 预处理支持隔离、相同 C/P raw normalization，以及复用 bag/预测/源码。within_subject_day 记录 `stopped_protocol_overlap` 与 `pending_protocol_revision`。

数值接口检查曾在有标签更新前发现新入口关闭 eval fastpath 后，与原 token 的误差为 0.000930786，超过 2e-4。实际相同 batch 对照确认保留现有 eval/no-grad fastpath 后误差降至 9.54e-7；适配的 autograd 前向仍明确使用 FP32 math SDPA。原门槛、prefix、来源与科学参数保留，失败和后端诊断归档在 `inputs/attention_equivalence_{failure,diagnostic}.json`，恢复日志为 `logs/experiment_a_attention_repair.log`。

专项 4 项回归覆盖共同头与 dropout 随机流、PROBE 冻结、ADAPT 尾部更新、event 等权 mask，以及 optimizer/全部随机状态中断恢复；另有 2 项报告测试验证 val-only 读取与共同 bootstrap 交互，6 项新增检查在本地和 H20 均通过，实际执行源码 SHA256 一致。既有 MAE 修复、normalization 和 energy balance 的 37 项检查在本地通过。Smoke 使用均匀覆盖 train/val 身份的各 64 个完整 event，正式监督从原初始化重新开始；每个 batch 保存恢复状态，固定尺度由完整原 train event 拟合。

A 完成以 `EXPERIMENT_A_COMPLETE_WITH_PROTOCOL_STOP` 和全部 A 产物验收为准。Test 预测保留供完整第二轮 Stage D 统一披露；A 的独立报告只读 val。该完成标记限定 A，B/C 和完整第二轮完成标记按原合同另行推进。

实际完成：4 个正式 EEG-only 模型、2 份适配 canonical token、6 个新融合 cell、6 个精确复用 cell，以及四路适配 smoke/两路融合 smoke 均验收；`EXPERIMENT_A_COMPLETE_WITH_PROTOCOL_STOP` 与 `EXPERIMENT_A_REPLAY_VERIFIED` 已生成。原信号独立重建每份 token 的 1,152 窗，最大误差 C=1.91e-6、P=1.43e-6；六个融合 checkpoint 的 val/test 预测重建误差均为 0。87 份轻量产物同步到本地且 SHA256 逐文件一致，完整模型/token/prefix 保留在 H20。

Val 配对判定：C 适配的融合 raw r `0.191789→0.190567`，delta `−0.001222`，centered r delta `−0.000576`，sRMSE delta `+0.000629`；P 为 `0.202645→0.205664`，delta `+0.003019`，centered r delta `+0.007293`，sRMSE delta `−0.001475`。两路 raw r 均为 1/3 seed 正向，联合门槛均为 False。P 的 EEG-only raw r delta 为 `+0.016491`，其适配上游 seed 数为 1。后续 B/C 保持原 C 初始化。完整证据见[实验 A 结果解读](../../../../../outputs/server_sync/mae_round2_downstream_20261010/reports/experiment_a_findings.md)、[机器报告](../../../../../outputs/server_sync/mae_round2_downstream_20261010/reports/experiment_a_report.md)与[独立完成审计](../../../../../outputs/server_sync/mae_round2_downstream_20261010/reports/completion_audit.json)。

## 12 实验 B 执行记录（2026-10-10）

[141执行入口](../../../../../scripts/multilabel/141_run_round2_experiment_b.py)与[EEG-only联合微调模块](../../../../../src/daily_multimodal/training/mae_eeg_partial_ft.py)完成三项10-epoch smoke及九个正式cell，均从原C SSL初始化。末一层prefix由原C冻结block4推进原last2缓存得到，切点为前5层；全量canonical token按batch128通过2e-4等价门槛。构造tail时恢复head之后的CPU/CUDA随机状态，各seed三个配置的head、首批成员与dropout mask匹配；shuffle/modality dropout沿用原融合trainer共享NumPy流，与encoder构造独立。固定Wear/Video、native mask与原C train normalization逐值不变。

运行根为 `/home/wangzw/mae_round2_downstream_20261010`，日志为 `b_queue.log`，模型为 `fusion/B_T2_STD|B_T2_LOW|B_T1_LOW/formal/cross_day/seed_<seed>/best_checkpoint.pt`。[142报告入口](../../../../../scripts/multilabel/142_report_round2_experiment_b.py)仅读val，冻结双控制选择和两个机制差、三条FT−F_C；每seed对四路线共同subject-day bootstrap 2,000次。[143独立验收](../../../../../scripts/multilabel/143_verify_round2_experiment_b.py)从完整encoder权重重新构造tail与融合头，九个cell各抽1,152个原信号窗最大误差≤1.91e-6，val/test事件预测回放误差均0；冻结层、原始normalization和表征健康检查通过。5项新增本地/H20测试通过，其中batch中断恢复与连续训练权重完全相同。启动前split字典包含pretrain/finetune额外字段导致计数停止，已仅对合同的train/val/test三叶修复，归档错误后恢复。

Val raw r（三seed均值±SD）：F_C `0.191789±0.010105`、T2_STD `0.179695±0.003771`、T2_LOW `0.184801±0.012316`、T1_LOW `0.190796±0.010776`。低LR−标准LR的 Δraw r `+0.005107`（2/3正向）、Δcentered r `+0.006484`、ΔsRMSE `−0.000347`；末一层−末两层低LR分别为 `+0.005995`（3/3）、`+0.002502`、`−0.000175`，两项机制对照联合门槛通过。三条FT相对F_C门槛均False，`selected_practical_ft=null`。机制对照单seed 95%区间均跨0，推进规则与区间分别保留。

标记 `EXPERIMENT_B_COMPLETE_WITH_PROTOCOL_STOP` 与 `EXPERIMENT_B_REPLAY_VERIFIED` 均存在，within_subject_day九个新cell保持协议stop。97份机器轻量产物传输SHA256匹配，另同步[结果解读](../../../../../outputs/server_sync/mae_round2_downstream_20261010/reports/experiment_b/experiment_b_findings.md)；[机器报告](../../../../../outputs/server_sync/mae_round2_downstream_20261010/reports/experiment_b/experiment_b_report.md)、[选择记录](../../../../../outputs/server_sync/mae_round2_downstream_20261010/reports/experiment_b/selection_val_only.json)、[完成审计](../../../../../outputs/server_sync/mae_round2_downstream_20261010/reports/experiment_b/completion_audit.json)和逐情绪/seed/epoch/区间均已本地同步。完整模型与prefix保留在H20。C未启动，继续按原C冻结基点执行；test性能披露保持完整第二轮Stage D合同。


## 13 实验 C 执行记录（2026-10-10）

[144执行入口](../../../../../scripts/multilabel/144_run_round2_experiment_c.py)与独立[event损失模块](../../../../../src/daily_multimodal/training/event_loss_adaptation.py)在原C/B0冻结表征上完成4项3-epoch smoke和12个新正式cell。保留原融合源码；λ=1原运算顺序、masked loss/gradient和两个真实bag的3-epoch旧trainer训练权重/历史逐值一致。token、non-EEG槽、native mask、event/label/split、train normalization、初始head与首批shuffle/dropout均匹配。全部训练仅改变目标函数，支持batch/optimizer/selector/完整随机状态恢复。

[145 val选择入口](../../../../../scripts/multilabel/145_select_report_round2_experiment_c.py)分别对MAE/B0保存λ=0/0.1相对λ=1的三seed联合规则，并先冻结全部A/B/C共36个融合cell的选择与来源。MAE val raw r为λ=1/0/0.1：0.191789/0.141515/0.158590；B0为0.332861/0.289757/0.290710。四项loss contrast均0/3 raw r正向，门槛False，两个表征的loss候选均为空。[146独立验收](../../../../../scripts/multilabel/146_verify_round2_experiment_c.py)回放18个C/control checkpoint的val/test预测，36叶误差均0；λ=1控制也生成窗口方差/极值，保留全部六路线诊断。新增7项本地/H20测试通过，见[选择](../../../../../outputs/server_sync/mae_round2_downstream_20261010/reports/experiment_c/selection_val_only.json)、[val报告](../../../../../outputs/server_sync/mae_round2_downstream_20261010/reports/experiment_c/experiment_c_val_report.md)和[完成审计](../../../../../outputs/server_sync/mae_round2_downstream_20261010/reports/experiment_c/completion_audit.json)。

标记 `EXPERIMENT_C_COMPLETE_WITH_PROTOCOL_STOP` 与 `EXPERIMENT_C_REPLAY_VERIFIED` 已生成，within_subject_day保持既有协议stop。本节和第14节更新第11/12节记录之后的当前状态；早期A/B产物继续保留各自阶段快照。

## 14 Stage D：统一 test 性能与 ABC 有效性（2026-10-10）

[147统一报告入口](../../../../../scripts/multilabel/147_report_round2_stage_d.py)在val选择冻结后重算全部36个唯一融合cell的val/test 72叶，以及4个EEG-only模型8叶；指标与保存预测一致，成员/标签/目标尺度均匹配。每seed共同重采样12条融合路线2,000次subject-day blocks，保存逐情绪五指标、目标/预测SD、配对差与95%区间、A来源×适配交互、B层/LR对照、C loss交互和同损失B0差距。EEG-only一个适配seed单列；λ=0/0.1与原λ=1同时保存窗口变异与日期级诊断。可复查[统一机器报告](../../../../../outputs/server_sync/mae_round2_downstream_20261010/reports/stage_d/stage_d_report.md)、[冻结val选择](../../../../../outputs/server_sync/mae_round2_downstream_20261010/reports/stage_d/selection_val_only.json)与[完整有效性解读](../../../../../outputs/server_sync/mae_round2_downstream_20261010/reports/stage_d/abc_effectiveness_report.md)。

Test核心结果：原E1/F_C raw r 0.268234；A canonical适配为0.276340，配对+0.008106、3/3正向，三个seed的宏raw r配对95%区间均>0。P适配为0.308577，相对F_P 0.303645增加0.004932，centered r减少0.001489；与原E1的总差由扩池与适配分别贡献。A两路的val推进门槛保持False。

B标准LR/末两层低LR/末一层低LR test raw r为0.268280/0.270850/0.267267。低LR保留部分raw r与误差收益，末一层相对末两层低LR的test raw r减少0.003584（0/3正向），其val层数优势未保留到test。实用FT选择为空。

C的MAE纯event/弱一致性test raw r为0.220151/0.221826，相对原E1下降0.048083/0.046407；B0对应为0.350030/0.358388，相对B0_W 0.373318下降0.023288/0.014930。四项loss contrast均0/3正向；当前λ=1的一致性强度保留。Pure event的MAE centered r有局部+0.010321，整体raw r与误差优势仍由window目标保持。

本轮原C冻结E1与λ=1继续作为主基点，B0也保留原window目标；各阶段val实用候选为空，可选联合确认未启动。D完整源文件、输入cell与选择hash均核对，额外P控制的三seed预测回放补齐全部36个唯一融合checkpoint的val/test独立回放，误差均0。36个cell的原预测/指标以只读副本同步，模型/token/prefix保留H20。快照打包的route目录定位问题已修复并逐文件校验，原实验产物未改动。

完成标记 `ROUND2_STAGE_D_VAL_SELECTION_FROZEN`、`ROUND2_STAGE_D_TEST_REPORT_COMPLETE` 和 `ROUND2_CORE_ABC_COMPLETE_WITH_PROTOCOL_STOP` 已生成。36个唯一融合cell（27新增、9复用）与4个单seed EEG-only模型组成完成范围；within_subject_day严格协议stop保留，历史test已查看的探索性边界保留。
