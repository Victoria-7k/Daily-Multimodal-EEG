# Wang Ziwei — 日常情绪多模态分析项目 Academic CV 素材

> 用途：Nanyang Technological University GEM Trailblazer Exchange Programme、暑期科研申请、研究型硕士及博士申请  
> 整理日期：2026-09-12  
> 证据范围：当前仓库、代码、实验报告、结果文件、Git 历史及 Wang Ziwei 本人确认信息  
> 使用原则：只有存在直接证据或 Wang Ziwei 本人确认的内容才作为事实；其余信息标记为“待补充”。

## 1. 项目事实摘要

### 1.1 基本信息

| 项目 | 内容 | 证据状态 |
| --- | --- | --- |
| 项目中文名称 | **日常情绪多模态分析** | Wang Ziwei 本人确认 |
| 建议英文名称 | **EEG-Aligned Multimodal Analysis of Daily Affective States** | 建议译名，尚非正式项目名称 |
| 学校/机构 | **NCC Lab, Southern University of Science and Technology** | Wang Ziwei 本人确认 |
| 导师 | **沈新科助理教授** | Wang Ziwei 本人确认 |
| 导师英文写法 | **Assistant Professor Xinke Shen** | 建议拼写；正式英文姓名待核对学校主页 |
| 项目时间 | **June 2026 – Present** | 本人确认开始日期；Git 最早记录为 2026-06-26 |
| 当前状态 | **Ongoing**；正在整理成果、补充实验，并计划推进论文写作 | Wang Ziwei 本人确认 |
| Wang Ziwei 角色 | **Project Lead and Multimodal Researcher** | 本人确认其主导项目并独立负责多模态部分 |
| 团队分工 | Wang Ziwei 独立负责多模态研究；合作者提供已清洗、预处理的 EEG 数据；原始数据采集由其他成员完成 | Wang Ziwei 本人确认 |
| 研究领域 | Affective computing、biomedical artificial intelligence、multimodal learning、physiological signal processing、longitudinal health sensing | 由项目内容归纳 |

### 1.2 研究关键词

- Multimodal Learning
- Daily Affective State Analysis
- Fatigue Estimation
- Electroencephalography
- Ecological Momentary Assessment
- Wearable Sensing

### 1.3 核心研究问题

项目研究如何将连续采集的脑电、生理、面部视频和音频信号与稀疏的主观情绪及疲劳评分正确对齐，并建立能够适应跨日期、跨受试者和已见受试者保留日期等不同泛化场景的事件级预测模型。

当前证据最完整的具体任务是日常疲劳估计。其关键问题包括：

1. 如何解决密集传感器窗口与稀疏 Ecological Momentary Assessment（EMA，生态瞬时评估）标签之间的粒度错配；
2. 如何融合 Electroencephalography（EEG，脑电）、可穿戴生理、面部视频和音频信息；
3. 如何处理模态缺失、日期漂移和个体差异；
4. 如何在训练、验证和测试之间避免事件级及 subject-day 级数据泄漏；
5. 如何确定评分前不同时间范围的信号对预测的贡献。

### 1.4 科学、工程与应用价值

传统窗口级训练可能把同一条自评标签复制给多个窗口，使监督单位与真实标注过程不一致。本项目将一次 EMA 评分定义为一个监督事件，并把评分前约两分钟的多模态历史组织为一个事件包，从而使模型输入与真实标签生成过程一致。

项目进一步区分跨受试者、跨日期和已见受试者保留日期三种应用条件，为日常情绪与疲劳监测、个体化健康感知、多模态生物医学建模和纵向行为分析提供可复核的实验框架。

### 1.5 研究对象、数据和场景

- 15 名受试者；
- 150 个 subject-day；
- 原始 manifest 包含 1,272 个事件；
- 当前 canonical EEG-aligned 分析使用 1,253 个 EMA 事件；
- 共 28,819 个 10 秒窗口；
- 每个 EMA 事件包含 23 个重叠窗口，stride 为 5 秒，覆盖评分前约两分钟；
- 数据采集场景包括宿舍、课堂、自习室等日常环境；
- 受试者主要进行日常学习和生活任务；
- Wang Ziwei 未参与原始数据采集；
- EEG 数据由合作者完成清洗和预处理后提供给 Wang Ziwei。

数据来源在项目中被称为 `DailyEEG`。正式数据集名称、采集协议、数据所有权和公开条件待补充。

### 1.6 数据模态

| 模态 | 原始或上游输入 | 当前表示方法 |
| --- | --- | --- |
| EEG | 59 通道 EEG；canonical 输入为每窗 2,000 个采样点 | EEGPT 预训练编码器；主结果使用 fatigue-supervised partial fine-tuning 后的 256D token |
| Wearable physiology | Photoplethysmography（PPG）、Galvanic Skin Response（GSR）、三轴 accelerometry（ACC） | HR/HRV、皮肤电、运动与静止特征等 Wphysio 表示，并投影到 256D |
| Video | 2× 主脸 Region of Interest（ROI）窗口 | Frozen DINOv2 visual representation，输出 256D token |
| Audio | 视频音轨切片 | openSMILE eGeMAPS Functionals，投影为 256D token |

`EEGPT` 是仓库使用的模型名称，仓库没有给出可核实的正式全称，因此不自行扩写。eGeMAPS 指 extended Geneva Minimalistic Acoustic Parameter Set。

### 1.7 整体技术路线

```text
Collaborator-provided preprocessed EEG
+ PPG / GSR / ACC
+ facial video
+ audio
        ↓
EEG-aligned 10-second windows
        ↓
Modality-specific 256-dimensional representations
        ↓
23-window EMA event bags + missing-modality masks
        ↓
Within-window multimodal fusion
        ↓
Full-window / recent-window / state-guided temporal aggregation
        ↓
Event-level affect or fatigue prediction
        ↓
Cross-subject / cross-day / within-subject-day evaluation
```

统一事件输入合同为：

```text
tokens:        (23, 4, 256)
modality_mask: (23, 4)
label:         one EMA affect/fatigue score
```

### 1.8 关键模型与实验方法

- EEGPT partial fine-tuning；
- DINOv2 frozen visual representation；
- openSMILE eGeMAPS acoustic features；
- physiological feature engineering for PPG/GSR/ACC；
- masked modality-token attention；
- learnable-query pooling；
- Multilayer Perceptron（MLP）regression head；
- full-window mean、recent-30-second pooling 和 state-prior modality routing；
- train-only feature and target normalization；
- validation-based early stopping and checkpoint selection；
- matched multi-seed comparison；
- subject-day paired bootstrap；
- event-level label-permutation control；
- missing-modality、noise 和 shuffle robustness evaluation。

### 1.9 训练、验证和测试划分

- `pretrain + finetune` 合并为训练集；
- validation 用于早停、候选选择和 checkpoint 保存；
- test 仅用于锁定配置后的最终评价；
- normalization 统计量只在训练集拟合；
- event identity、23-window membership 和 subject-day overlap 均进行审计。

当前标量回归报告中的事件级划分如下：

| Protocol | Train events | Validation events | Test events |
| --- | ---: | ---: | ---: |
| Cross-day | 731 | 269 | 253 |
| Within-subject-day | 749 | 246 | 258 |
| Cross-subject | 767 | 222 | 264 |

### 1.10 评价指标、基线和统计分析

评价指标包括：

- Pearson correlation coefficient，报告 raw Pearson's `r`；
- Root Mean Squared Error（RMSE）；
- Mean Absolute Error（MAE）；
- within-subject centered Pearson's `r`；
- 序数预测分支使用 Quadratic Weighted Kappa（QWK）、Macro-F1 和 ordinal MAE。

主要匹配基线为 `window_attention_regression_full_mean`：模型分别预测一个 EMA 事件中的有效窗口，再对同一事件的窗口预测取等权平均。每个候选与同一 protocol、事件集合、split、mask、随机种子和训练预算下的窗口基线进行配对比较。

统计分析采用多随机种子均值与标准差、seed direction consistency，以及每个 seed 2,000 次 subject-day paired bootstrap。当前标题结果尚未形成可用于 CV 的正式统计显著性结论。

### 1.11 核心技术难点

1. 将稀疏 EMA 标签与连续多模态传感器流建立正确的监督关系；
2. 保证一个事件的全部窗口留在同一数据划分中；
3. 对齐 EEG、可穿戴、视频和音频的时间轴；
4. 处理不同模态的缺失和质量变化；
5. 区分跨人差异、跨日漂移和个体内部变化；
6. 在小规模纵向数据上进行可靠的模型选择；
7. 保持窗口级与事件级评价单位一致；
8. 使实验配置、预测、指标和统计结果可以追溯。

## 2. Wang Ziwei 个人贡献

### 2.1 已确认的个人角色

Wang Ziwei 是项目主导者，并独立负责多模态研究部分。推荐在 Academic CV 中使用：

> **Project Lead and Multimodal Researcher**

### 2.2 独立完成和主导的工作

1. **主导研究方向**：主导日常情绪多模态分析的研究问题细化、技术路线设计、实验设计和结果解释。
2. **独立完成多模态工作**：独立负责 EEG、可穿戴生理、面部视频和音频的多模态整合。
3. **实现数据对齐管线**：将合作者提供的预处理 EEG 与其他模态对齐到统一的 10 秒窗口和 EMA 事件。
4. **设计事件级监督合同**：将每条 EMA 及其 23 个窗口建模为一个监督样本，避免把同一标签视为 23 条独立观测。
5. **实现表示与融合模块**：实现或整合 256D modality token、缺失模态 mask、attention fusion、query pooling 和 temporal aggregation。
6. **实现训练与评价系统**：完成训练、early stopping、checkpoint、预测保存、指标计算和批量实验调度。
7. **设计泛化协议**：建立和审计 cross-subject、cross-day、within-subject-day 三种协议。
8. **执行实验与消融**：完成 encoder、fusion、normalization、temporal policy、缺失模态和标签置换等实验。
9. **完成统计分析**：使用 matched seeds、paired deltas、subject-day bootstrap 和方向一致性分析候选模型。
10. **整理研究成果**：撰写技术路线、实验计划、结果报告和投稿规划材料，并继续补充实验。

### 2.3 与团队工作的边界

| 工作 | Wang Ziwei 的参与程度 |
| --- | --- |
| 项目总体研究与多模态方向 | 主导 |
| 多模态数据处理、对齐、建模和评价 | 独立完成 |
| EEG 原始数据清洗和预处理 | 由其他合作者完成，Wang Ziwei 使用其输出 |
| 原始数据采集 | 未参与 |
| 数据采集协议和受试者管理 | 由其他成员完成；具体成员待补充 |
| 论文写作 | 尚在成果整理和追加实验阶段，计划推进 |

不建议使用以下表述：

- `Independently collected the multimodal dataset`；
- `Independently completed the entire project`；
- `Developed the EEG preprocessing pipeline`，除非后续能确认 Wang Ziwei 对该部分有直接贡献。

## 3. 可量化结果

> 以下模型性能均为 preliminary、non-peer-reviewed results。评价单位为 held-out EMA event。

| 内容 | 量化结果 | 证据位置 |
| --- | ---: | --- |
| 受试者 | 15 | `outputs/researchwrite/taffc_daily_multimodal/01_research_canon.md` |
| 纵向规模 | 150 subject-days | 同上 |
| 原始 manifest | 1,272 events | `outputs/reports/manifest_summary.json` |
| Canonical 分析数据 | 1,253 EMA events | `daily_affect_scalar_regression_best_route_technical_report_20260910.md` |
| EEG-aligned 窗口 | 28,819 | 同上 |
| 单事件时间结构 | 23 overlapping windows；10 seconds/window；5-second stride | 同上 |
| 模态表示 | 每模态 256 dimensions | `README.md` |
| EEG 输入 | `(28,819, 2,000, 59)` | `README.md` |
| 初始标量回归矩阵 | 171 runs = 3 protocols × 3 seeds × 19 conditions | 标量回归技术报告 |
| 完整标量回归证据 | 199 `metrics.json`；178 matched window pairs | 标量回归技术报告 |
| 标题级重复实验 | 7 downstream seeds | 标量回归技术报告 |
| Bootstrap | 每 seed 2,000 次 subject-day paired resampling | 标量回归技术报告 |
| EEG representation screen | 15 runs | `README.md` |
| Fusion route screen | 180 runs | `README.md`；属于 single-seed route screening |

### 3.1 Cross-day preliminary result

- 最佳路线：`bag_static_reg__temporal_last_30s`；
- event-level raw `r = 0.4091 ± 0.0314`；
- RMSE `= 0.8788 ± 0.0188`；
- 相对 matched full-window mean：`Δraw r = +0.0453 ± 0.0291`；
- 7/7 seeds 的 raw-r delta 为正；
- RMSE 相对基线降低 `0.0267`。

该结果支持：在跨日期泛化中，评分前最近约 30 秒的证据比完整两分钟等权平均更有效。

### 3.2 Within-subject-day preliminary result

- 最佳路线：`window_attention_regression_full_mean`；
- event-level raw `r = 0.4314 ± 0.0105`；
- RMSE `= 0.8952 ± 0.0065`；
- centered `r = 0.1964 ± 0.0113`。

该协议中，完整窗口平均保持为最强路线。

### 3.3 Cross-subject preliminary result

- 最佳路线：`prior_uniform_reg`；
- event-level raw `r = 0.1072 ± 0.0425`；
- RMSE `= 0.9405 ± 0.0429`；
- 相对 matched full-window mean：`Δraw r = +0.0631 ± 0.0596`；
- 6/7 seeds 的 raw-r delta 为正；
- `ΔRMSE = +0.0173`，位于预设 `+0.02` guardrail 内。

跨受试者绝对相关仍较低，因此适合写成协议差异和初步增益，不适合写成强跨人预测能力或领域领先结果。

### 3.4 当前不能使用的量化结论

现有证据不支持以下表述：

- statistically significant improvement；
- state-of-the-art performance；
- clinically validated system；
- real-time deployment；
- reduced annotation cost 或具体节省比例；
- 具体 GPU-hour、训练费用或运行速度提升；
- reliable within-person state tracking。

## 4. 研究产出

| 类型 | 标题或内容 | 作者/贡献 | 年份 | 正式状态 | 链接或位置 |
| --- | --- | --- | ---: | --- | --- |
| 技术报告 | *Daily-affect Scalar Regression: Best Raw-r Routes under Three Protocols* | 文件未列作者；Wang Ziwei 主导实验和结果整理 | 2026 | Completed internal technical report | `daily_affect_scalar_regression_best_route_technical_report_20260910.md` |
| 软件仓库 | *Daily-Multimodal-EEG* | Git 身份 `Victoria-7k` 已由 Wang Ziwei 确认为本人 | 2026 | Active research repository | `https://github.com/Victoria-7k/Daily-Multimodal-EEG` |
| 实验产物 | Config、checkpoint、prediction、metrics、CSV summary、bootstrap 和 figures | Wang Ziwei 独立负责多模态实验管线 | 2026 | 已生成；大体积运行产物主要保存在远端 | `outputs/` 及技术报告记录的远端路径 |
| 投稿规划 | *IEEE Transactions on Affective Computing 投稿蓝图* | 当前为论证与实验规划材料 | 2026 | Planning document | `outputs/researchwrite/taffc_daily_multimodal/exports/TAFFC_manuscript_blueprint_zh.md` |
| 数据集 | 内部 `DailyEEG` 数据 | Wang Ziwei 未参与采集；预处理 EEG 由合作者提供 | 待补充 | 未确认公开发布 | 待补充正式数据集信息 |

当前没有证据支持以下正式状态：

- paper published；
- preprint posted；
- submitted；
- under review；
- patent filed；
- public dataset release；
- poster or oral presentation delivered；
- competition award。

当前推荐项目状态写法：

> **Ongoing research; results consolidation and additional experiments in progress, with manuscript writing planned.**

在开始形成论文正文、作者列表和稳定稿件后，再使用 `manuscript in preparation`。

## 5. 技术与研究能力

### 5.1 研究能力

- Research problem formulation
- Research design
- Multimodal data alignment
- Data preprocessing and quality control
- EEG and physiological signal analysis
- Machine learning
- Deep learning
- Multimodal representation learning
- Affective computing
- Biomedical data analysis
- Longitudinal evaluation
- Missing-modality modeling
- Leakage-aware experimental design
- Regression and ordinal prediction
- Ablation studies
- Statistical analysis
- Multi-seed experiment management
- Scientific visualization
- Technical and academic writing
- Reproducible research

### 5.2 编程语言、框架和软件

| 类别 | 工具或技术 |
| --- | --- |
| Programming | Python；PowerShell/Bash 用于本地与远端调度 |
| Deep learning | PyTorch、AdamW、attention、MLP、partial fine-tuning |
| Numerical/data | NumPy、JSON/JSONL、CSV、NPZ |
| Visualization | Matplotlib、Pillow |
| Representations | EEGPT、DINOv2、openSMILE eGeMAPS、Wphysio；另评估 CBraMod、MOMENT 等候选路线 |
| Engineering | Git、GitHub、Linux、SSH、CUDA GPU server、automated scripts、unit tests |
| Experiment methods | Train-only normalization、early stopping、matched seeds、paired bootstrap、label permutation controls |

### 5.3 实验设备

已知数据源包括 EEG、PPG/GSR/ACC wearable、camera 和 microphone。设备品牌、型号、正式采样配置及采集软件待补充。

## 6. NTU 交换申请英文短版

**EEG-Aligned Multimodal Analysis of Daily Affective States**  
*Project Lead and Multimodal Researcher, NCC Lab, Southern University of Science and Technology; Advisor: Assistant Professor Xinke Shen | Jun 2026 – Present*

- Lead an independent multimodal research stream analyzing daily affect from preprocessed EEG, wearable physiology, facial video, and audio recorded during routine academic activities.

- Developed an event-level pipeline aligning 1,253 self-reports with 28,819 sensor windows from 15 participants, including representation integration, missing-modality masking, and temporal fusion.

- Evaluated 199 matched regression runs across three generalization protocols; recent 30-second aggregation improved cross-day Pearson correlation by 0.0453 across all seven seeds.

> 使用前请核对导师官方英文姓名。模型结果在论文或同行评审完成前应视为 preliminary results。

## 7. 暑研／硕博申请英文扩展版

**EEG-Aligned Multimodal Analysis of Daily Affective States**  
*Project Lead and Multimodal Researcher, NCC Lab, Southern University of Science and Technology; Advisor: Assistant Professor Xinke Shen | Jun 2026 – Present*

- Lead the project's multimodal research direction, independently designing and implementing data alignment, representation integration, temporal modeling, experimental evaluation, and result interpretation for daily affect analysis.

- Integrated collaborator-provided preprocessed EEG with independently developed wearable-physiology, facial-video, and audio pipelines, covering 1,253 ecological momentary assessment events and 28,819 aligned windows from 15 participants.

- Implemented a reproducible Python/PyTorch framework combining 256-dimensional EEGPT, physiological, DINOv2, and openSMILE representations with missing-modality masking, attention-based fusion, temporal aggregation, and automated artifact reporting.

- Designed leakage-aware cross-subject, cross-day, and within-subject-day evaluations using train-only normalization, validation-based checkpointing, seven matched seeds, held-out testing, and 2,000-iteration subject-day paired bootstraps.

- Evaluated 199 event-level regression runs; a recent-30-second policy achieved cross-day `r = 0.4091 ± 0.0314` and improved matched full-window averaging by 0.0453 across 7/7 seeds.

## 8. 证据表

| CV 陈述 | 支持证据 | 文件、结果或对话位置 | 可信度 |
| --- | --- | --- | --- |
| 项目名称为“日常情绪多模态分析” | Wang Ziwei 本人确认 | 2026-09-12 对话 | Confirmed |
| 项目隶属南方科技大学 NCC Lab | Wang Ziwei 本人确认 | 2026-09-12 对话 | Confirmed |
| 导师为沈新科助理教授 | Wang Ziwei 本人确认 | 2026-09-12 对话 | Confirmed |
| Wang Ziwei 是项目主导者 | Wang Ziwei 本人确认 | 2026-09-12 对话 | Confirmed |
| Wang Ziwei 独立负责多模态部分 | Wang Ziwei 本人确认 | 2026-09-12 对话 | Confirmed |
| 预处理 EEG 由合作者提供 | Wang Ziwei 本人确认 | 2026-09-12 对话 | Confirmed |
| Wang Ziwei 未参与数据采集 | Wang Ziwei 本人确认 | 2026-09-12 对话 | Confirmed |
| 场景包括宿舍、课堂和自习室 | Wang Ziwei 本人确认 | 2026-09-12 对话 | Confirmed |
| 项目开始于 2026 年 6 月 26 日 | 本人确认及 Git 记录 | Git commit `5d5c5d0` | Confirmed |
| 项目研究 EEG-aligned 多模态疲劳/情绪预测 | 根 README 和项目路线 | `README.md`；`technical_route_20260906.md` | Confirmed |
| 使用 EEG、Wear、Video、Audio 四模态 | 模态输入和 embedding 合同 | `README.md`；`repo-docs/modules/embedding-contract.md` | Confirmed |
| 数据含 15 名受试者、150 subject-days | Research canon 和 split audit | `outputs/researchwrite/taffc_daily_multimodal/01_research_canon.md` | Confirmed |
| 使用 1,253 EMA events 和 28,819 windows | 标量回归技术报告 | `daily_affect_scalar_regression_best_route_technical_report_20260910.md` | Confirmed |
| 每事件包含 23 个 10 秒重叠窗口 | EMA-bag 构建报告和实现 | 技术报告；`src/daily_multimodal/daily_affect/ema_bags.py` | Confirmed |
| 完成 199 次标量回归运行 | 技术报告记录 199 个 metrics | 标量回归技术报告 | Confirmed |
| Cross-day raw-r 提升 0.0453，7/7 seeds 正向 | Seven-seed matched result | 标量回归技术报告 cross-day section | Confirmed |
| 当前正在补实验和整理论文成果 | Wang Ziwei 本人确认 | 2026-09-12 对话 | Confirmed |
| 正式英文项目名 | 当前为建议译名 | 本文件 | Unverified |
| 导师英文姓名拼写为 Xinke Shen | 尚未核对学校官方页面 | 待核对 | Unverified |

## 9. 待补充信息表

| 缺失信息 | 为什么影响 CV | 建议向 Wang Ziwei 或导师确认的问题 |
| --- | --- | --- |
| NCC Lab 正式英文全称 | 机构行应使用官方名称 | NCC 是否为实验室正式名称或缩写？完整英文名称是什么？ |
| 导师官方英文姓名 | 防止姓名拼写错误 | 学校主页使用 `Xinke Shen`、`Shen Xinke` 还是其他拼写？ |
| 合作者姓名与分工 | 有助于准确区分团队贡献 | 谁负责 EEG 清洗预处理、数据采集和研究指导？ |
| 正式团队规模 | 有助于说明领导和协作范围 | 项目共有多少名教师、学生和数据采集成员？ |
| 数据采集时间与协议 | 论文和完整 CV 描述需要 | 数据在哪些月份采集？每名受试者参与几天？ |
| 受试者人口统计 | 影响数据集描述和论文方法 | 是否可以报告年龄、性别及其他人口统计摘要？ |
| 伦理审批 | 生物医学研究的重要合规信息 | 伦理委员会名称和审批编号是什么？ |
| 疲劳/情绪量表 | 决定标签解释 | EMA 问题、1–5 anchor 和量表理论依据是什么？ |
| 设备型号 | 能增强技术可信度 | EEG、PPG/GSR/ACC、camera 和 microphone 的具体型号是什么？ |
| GitHub 公开状态 | 决定能否写 open-source software | 仓库是否允许公开并长期保留？ |
| 数据共享状态 | 决定能否写 dataset release | 数据能否公开、申请访问或只保留内部使用？ |
| 论文实际启动状态 | 决定何时使用 manuscript in preparation | 是否已经建立正文文件、作者列表和目标期刊？ |
| 报告或展示经历 | 可以形成独立 CV 产出 | 是否已在组会、课程、论坛或会议中报告该项目？ |
| 计算成本 | 可量化实验管理能力 | 199 次实验使用何种 GPU、多少 GPU-hours？ |

## 10. 推荐保留的三条最强 CV 陈述

### 1. 研究主导性与贡献边界

> **Led and independently implemented the multimodal research pipeline for daily affect analysis, integrating collaborator-provided preprocessed EEG with wearable physiology, facial video, and audio.**

这条陈述最能体现研究主导性、独立技术贡献和清晰的合作边界。

### 2. 研究问题与数据规模

> **Constructed an event-level data framework aligning 1,253 ecological momentary assessments with 28,819 sensor windows from 15 participants across everyday academic environments.**

这条陈述突出事件级监督问题、数据规模和日常场景价值。

### 3. 实验严谨性与量化结果

> **Evaluated 199 matched regression runs under three leakage-aware protocols, identifying a recent-30-second temporal policy that improved cross-day correlation across all seven seeds.**

这条陈述提供可验证的实验规模、方法严谨性和最有说服力的初步结果。

## 主要证据索引

- `README.md`
- `technical_route_20260906.md`
- `daily_affect_scalar_regression_best_route_technical_report_20260910.md`
- `outputs/researchwrite/taffc_daily_multimodal/01_research_canon.md`
- `outputs/researchwrite/taffc_daily_multimodal/02_evidence_table.md`
- `outputs/server_sync/eegpt_centered_improvement/split_audit_subject_day.json`
- `outputs/reports/manifest_summary.json`
- `src/daily_multimodal/daily_affect/ema_bags.py`
- `src/daily_multimodal/daily_affect/regression.py`
- `src/daily_multimodal/daily_affect/regression_training.py`
- `scripts/daily_affect/90_run_daily_affect_scalar_regression.py`
- `scripts/daily_affect/91_summarize_daily_affect_scalar_regression.py`
- `scripts/daily_affect/92_plot_daily_affect_scalar_regression.py`
