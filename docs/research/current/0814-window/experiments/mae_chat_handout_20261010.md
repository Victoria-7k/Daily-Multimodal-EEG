# Daily Multimodal MAE实验交接手册

本聊天全部工作信息与结果 handout

封存日期：2026年10月10日，Asia/Shanghai。最终正式矩阵完成于2026年10月9日17:03:21；本次服务器结果复核于2026年10月10日11:28后进行。聊天：按顺序执行MAE主线实验；thread ID：01a0cfa0-c24a-70e2-a063-cea952c7035e。覆盖已回溯的87个回合，以及当前源码、训练配置、结果与产物。

本手册供继续研究、复现实验、撰写结果和接管队列使用。主文以最终有效结果为准，历史部分保留方案变更、原始结果、错误和修复。旧进度百分比、PID及预计完成时间均属于当时的运行快照，最终完成状态覆盖这些估计。

阅读顺序：先看第1、4、6、7节了解结论、矩阵和两轮效果；接管服务器时看第10至12节；准备新增实验时看第13节。Markdown保留完整横向结果表，Word把宽表拆成同一协议下的列组，数值与定义一致。

## 1 当前完成状态与主要结论

最终实验矩阵全部完成：2种协议 × 10条路线 × 3个下游seed，共60条有效结果。最后一轮新增48条正式结果，保留B0和V1的12条匹配结果。所有行status为ok，验证与测试各11情绪raw r均为有限值；NORMALIZATION_COMPLETE存在，当前NORMALIZATION_QUEUE_FAILED不存在。查询时未发现本队列的132、124或118训练进程。

三种模态的MAE预训练、canonical embedding导出、视频产物传回H20、冻结融合与部分微调下游验证均已完成。最终EEG/Wear使用第二轮v4，Video使用第一轮v2。结果支持以下判断。

- EEG/Wear遮挡时覆盖位置编码的问题已经修复。Video的位置编码原实现已保留，其修复重点为初始化种子、固定验证掩码和表征健康检查。
- 新产物通过近常量表征检查；第二轮还检查典型窗口，避免少量高能样本掩盖普通窗口的坍缩。
- 第一轮明显恢复三模态MAE的同日部分微调效果。第二轮进一步改善EEG和全MAE路线，Wear跨日结果回退。
- MAE表征已接受正式情绪下游验证。最终M5-FT相对匹配R0在跨日3/3个seed胜出，同日2/3胜出。
- B0仍是这套最终矩阵中两协议宏平均raw r最高的参考。保留B0作为基线，当前证据支持MAE的可预测性与局部收益。
- EEG、Wear、Video各自的单模态线性探针，以及独立分离尺度、损失和遮挡作用的因子消融，尚未在本聊天的正式矩阵中执行。

### 1.1 最终宏平均raw r

| 路线 | 跨日 raw r | 同日 raw r |
| --- | --- | --- |
| B0 | 0.3733 ± 0.0117 | 0.4373 ± 0.0045 |
| E1 | 0.2682 ± 0.0085 | 0.3276 ± 0.0047 |
| W1 | 0.3618 ± 0.0120 | 0.4345 ± 0.0096 |
| V1 | 0.3582 ± 0.0155 | 0.3925 ± 0.0099 |
| M2 | 0.2609 ± 0.0337 | 0.3616 ± 0.0119 |
| M3 | 0.2021 ± 0.0213 | 0.2929 ± 0.0034 |
| M4 | 0.3630 ± 0.0143 | 0.4036 ± 0.0072 |
| M5-F | 0.2256 ± 0.0434 | 0.3137 ± 0.0056 |
| M5-FT | 0.2412 ± 0.0351 | 0.3037 ± 0.0118 |
| R0 | 0.1565 ± 0.0245 | 0.2821 ± 0.0410 |

数值为11情绪等权宏平均test raw r的三seed均值 ± 样本标准差。三seed为240800、240801、240802。样本标准差使用ddof=1；它表示seed间波动，不能当作置信区间或统计显著性检验。平均行的标准差从每个seed的11情绪宏平均计算。

## 2 研究问题与数据边界

研究问题是：在保持现有强参考的评估协议、窗口身份、下游结构和种子不变时，无标签单模态MAE能否替换EEG、Wear或Video表征；全部模态MAE冻结或部分微调后能取得怎样的情绪预测效果。

流程由两个监督边界明确的阶段构成。Stage A在无标签输入上做遮挡重建，选择并冻结encoder；Stage B读取其canonical 256D window token，以情绪标签训练现有融合器和11-head回归。上游MAE重建能力、表征变化和下游情绪预测分别检验，三者不能互相代替。

### 2.1 Canonical数据契约

| 项目 | 当前合同 |
| --- | --- |
| Canonical窗口 | 28,819个，对齐sample_id及行顺序固定 |
| EEG输入 | X.npy，形状28819 × 2000 × 59，每窗10秒 |
| 情绪标签 | 28819 × 11 |
| EMA事件 | 1,253个，每事件23个窗口 |
| Event bag | tokens为1253 × 23 × 4 × 256；modality_mask为1253 × 23 × 4 |
| 目标模态 | EEG、Wear、Video；audio slot在本参考中关闭 |
| EEG token有效行 | 28,819 |
| Wear token有效行 | 24,127 |
| VideoMAE token有效行 | 18,012 |
| Token字段 | sample_id、embedding、valid_mask及来源metadata |
| 导出要求 | embedding为28819 × 256，顺序一致、数值有限，无效行置零并保留false mask |

| 模态 | 行数 | 维度 | 有效行 | 顺序与有限值 |
| --- | --- | --- | --- | --- |
| eeg | 28819 | 256 | 28819 | 两协议均通过 |
| wear | 28819 | 256 | 24127 | 两协议均通过 |
| video | 28819 | 256 | 18012 | 两协议均通过 |

有效性检查覆盖H20最终EEG/Wear及沿用的Video两协议全部六份token。ncc原始Video和v2 Video两协议也在本次复核中均通过形状、有限值及无效行零值检查。零向量的有效与无效状态由mask决定，不能把填零视为真实观测。

### 2.2 Index与split如何关联

Index定义数据宇宙及其身份、行号、被试、日期和时间窗。Split在这个数据宇宙内声明哪些行用于pretrain、finetune、val、test。整数索引需要与同一份index一一对应；改变canonical index的排序、范围或样本后，需要显式映射或重新生成split。

当前MAE主线继续使用原canonical 28,819窗口及splits_new。最终训练集是pretrain与finetune的并集，Stage A不读情绪标签。

| 协议 | pretrain窗口 | finetune窗口 | SSL train并集 | val窗口 | test窗口 |
| --- | --- | --- | --- | --- | --- |
| cross_day | 6122 | 10691 | 16813 | 6187 | 5819 |
| within_subject_day | 6122 | 11121 | 17243 | 5708 | 5868 |

窗口数来自本次读取的Stage-A config.json。Wear及Video还按原生valid mask过滤训练和验证行，因此各模态的实际有效训练数应继续以各自config的effective_train_count为准。

### 2.3 两协议的解释范围

cross_day用于被试内跨日期泛化比较；within_subject_day用于同被试、同日期的窗口留出诊断。当前正式表为了匹配参考，统一采用event级聚合评价。两种协议各自训练一个上游MAE，共享canonical行号，拥有不同训练、验证和测试归属。

| 协议 | 下游train事件 | val事件 | test事件 | 当前bag事件索引的两两交集 |
| --- | --- | --- | --- | --- |
| cross_day | 731 | 269 | 253 | train-val、train-test、val-test均为0 |
| within_subject_day | 749 | 246 | 258 | train-val、train-test、val-test均为0 |

本次直接读取现有B0 bag的train_index、val_index、test_index核验了上述交集。早期其他split投影或旧缓存的事件重叠描述不能套用到这两份现有bag。within_subject_day仍共享被试和日期，这个结果不能用于宣称独立整日或独立被试泛化；整日时间顺序问题需使用date_in_order等独立协议并另行审计。

### 2.4 shared_unlabeled_pretrain的决定

文件位置为/vePFS-0x0d/DailyEEG/splits_new/shared_unlabeled_pretrain.json。聊天中的历史审计确认，它是从raw EEG数据库选取SSL训练行的索引列表，未提供完整val/test合同。

历史审计范围：raw EEG corpus共295,641窗；该列表含44,181个唯一raw-row ID，范围317至294,533；37,991个ID超出canonical 28,819行的范围。它覆盖54个subject-day，与cross_day的pretrain日期集合一致，与该协议val/test日期交集为0。Raw网格与canonical窗口时间存在约2秒偏移，行号不能直接互换。这些raw pool数字来自本聊天当时的审计，本次封存未重新扫描raw pool。

决定：本聊天正式MAE矩阵使用canonical-only。额外raw无标签EEG可成为后续独立cross-day扩展预训练对照，下游canonical index及split保持固定。within_subject_day若使用raw pool，需要逐时间区间排除val/test重叠窗口，仅按日期过滤不足以建立隔离。

“只有一份索引为何能比较两协议”的答案是：索引统一定义样本，协议分别定义这些样本的归属。若仅扩充SSL训练输入，最终仍对同一canonical universe导出token，但需要重新审核无标签训练pool与val/test的时间和日期边界。

### 2.5 无标签训练的val与test

- SSL train输入用于更新encoder/decoder。
- SSL val输入用于固定独立掩码下的重建损失与已声明技术验收，不使用情绪标签。
- SSL test可在冻结后做一次诊断；本矩阵的Stage-A选择不使用test情绪标签。对全部canonical行导出token是冻结推理步骤。
- 下游train读取11情绪标签训练fusion/head；下游val选择checkpoint；下游test报告最终情绪指标。

Stage A固定的验证掩码使用独立seed+100003随机流，与训练同样采用固定遮挡数量，避免验证掩码改变训练随机序列或偶发整窗全部遮挡。

### 2.6 输入来源补充核验：canonical EEG、raw pool与视频ROI

2026年10月10日12:16后，按用户追问直接登录H20与ncc核验以下输入。这里补充实际文件来源，不改变已完成的实验及结果。

- Canonical EEG为H20的`/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new/X.npy`，当前shape为`(28819,2000,59)`。窗口身份由`/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/index/eeg_aligned_window_index.jsonl`定义。最终EEG-MAE的训练输入为`/vePFS-0x0d/DailyEEG/splits_new/<protocol>/pretrain.json`与`finetune.json`的并集：cross_day为`6122+10691=16813`，within_subject_day为`6122+11121=17243`。对应Stage-A config的effective_train_count一致；Stage A读取这些行的信号，不读取情绪标签。val/test行仅用于各自声明的验证及冻结推理。
- 更大的raw窗口库为H20的`/vePFS-0x0d/DailyEEG/processed_cadt_raw_unlabeled/X.npy`，当前shape为`(295641,2000,59)`；同目录`sub.npy`、`d.npy`、`ts.npy`和`meta.json`记录被试、日期域、窗口起点及采样规则。候选SSL pool由`/vePFS-0x0d/DailyEEG/splits_new/shared_unlabeled_pretrain.json`选择44,181个唯一raw-row ID，当前范围317至294,533。该列表对应raw X的行号，不能索引canonical X。当前重核验其54个subject-day与cross_day/pretrain日期集合相同，与cross_day val/test日期交集为0；与within_subject_day val/test各有54个subject-day重叠。该pool未被本轮EEG-MAE使用。
- 当前VideoMAE输入metadata为ncc的`/home/lzs/DailyVideoMAE_20260927/input/video_metadata.npz`。其759条唯一非空source_video_file均指向`/mnt/dataset1/sitian/video/.../DJI_*.MP4`；114从原始视频按时间抽取全幅帧，再直接resize到112×112。最终Video v2配置明确复用`/home/lzs/DailyVideoMAE_20260927/cache/video_8x112.uint8.mmap`。本轮VideoMAE没有接入既有2.0×face ROI clip，当前V1/M3/M4/M5的视频替换同时改变了视觉输入视野、表征来源及9行native mask差异。
- 与当前canonical sample_id对应的2.0×ROI缓存实际位于ncc的`/mnt/dataset4/sitian/wzw/DailyEEG_multimodal_eeg_aligned_export/tmp/video_2xroi_openface_cache_full/openface/<sample_id>/openface_temporal_v1/window.mp4`，对应sidecar为`openface_target.json`。清单为同一export根下`reports/video_2xroi_cache_manifest.jsonl`，当前18,021行且sample_id唯一。`scripts/10_prepare_2xroi_openface_window_cache.py`明确以2.0扩展人脸框；`scripts/run_video_embeddings_A1_A2_resume.sh`以此缓存生成DINO A1/A2。
- 当前aligned ROI clip使用224×224、最多16个抽样帧、写入fps为2；抽查完整clip为16帧、文件时长约8秒，sidecar仍标记其对应的原始10秒区间。后续复用这些文件时应按clip本地帧号/时间采样，保留sidecar原始区间用于身份与split核对。历史`DailyMultimodalEmbedding/outputs/cache/real_stage12_face_filter_full_v2_mainface`的640×640缓存属于另一批旧窗口，不应与当前aligned缓存混用。

用户要求后续VideoMAE基于已切好的2.0×face ROI视频。该输入路线尚未运行；现有全幅结果继续保留来源标记，ROI路线需要独立缓存、预训练及下游对照。

## 3 参考基线与监督来源

### 3.1 当前唯一正式参考

完整route ID如下。

~~~
A1_Wphysio_no_audio__eeg_eegpt_partial_ft_multitask_11label_v1
~~~

该参考的EEG是11情绪联合监督partial fine-tuning的EEGPT；Wear为Wphysio；Video为DINO A1；Audio关闭。EEG embedding seed固定240800，下游使用三个匹配seed。

下游condition为window_attention_regression_full_mean；当前metrics中的model_id为window_replicated，temporal_policy为uniform，head_variant为H1_shared2_11xhead2，normalization和adapter_mode均为per_modality。它沿用0814的逐窗口结构，并对一个EMA event的23个窗口按既有规则聚合，最终在event level评价。

B0从已完成reference artifacts读取，未用新MAE重新训练B0。B0的原有上游监督来源、DINO A1、bags、split、下游结构及seed一直保留。

### 3.2 两种MT需要分开记录

本矩阵所有路线均使用有标签的11-head多任务下游，简称“下游MT”。上游MT11 EEGPT是另一层监督，指EEG encoder已经接受11情绪共同监督。B0、W1、V1、M4保留该上游表征；E1、M2、M3、M5-F、M5-FT的EEG上游为无标签MAE。M5-FT在下游阶段再以标签微调最后两层。

R0没有SSL预训练，使用相同MAE架构随机初始化，在匹配的最后两层范围内联合监督训练。上游训练种子与下游训练种子的角色不同；三下游seed的波动不覆盖多组独立SSL预训练seed的波动。

聊天早期曾把MT11 EEG误称为fatigue-supervised。最终基线的正确来源是11-label supervised partial FT，研究引用和后续命名均使用这一来源。

### 3.3 早期B0为什么无法复刻11情绪表

早期116/117运行了18个窗口级B0/E1/W1结果。其B0使用label-free frozen EEGPT、Wphysio、DINO视频B0，直接以canonical窗口评估，下游模型为MultiLabelAttentionRegressor。cross-day宏平均分别为B0 0.1953、E1 0.1832、W1 0.1683。

这与用户指定的A1+MT11 reference在上游EEG监督、视频分支、评估单位、head结构和split投影方面均不同。用户随后要求把最好的完整A1+MT11 route固定为基线，入口改为118/119及后续路线。早期18个窗口级结果保留为历史诊断，不并入本手册的原始版60行匹配矩阵。

实验ID B0表示参考路线；video_B0表示一种视频表征版本。最终实验B0的Video使用video_A1。二者名称需要分辨。

## 4 完整路线矩阵

| 路线 | EEG | Wear | Video | 下游训练范围 | 最终完成量 |
| --- | --- | --- | --- | --- | --- |
| B0 | MT11 EEGPT | Wphysio | DINO A1 | 固定表征，原fusion与11-head | 2协议 × 3seed，复用 |
| E1 | EEG-MAE | Wphysio | DINO A1 | 固定表征，原fusion与11-head | 6条，新v4 |
| W1 | MT11 EEGPT | Wear-MAE | DINO A1 | 固定表征，原fusion与11-head | 6条，新v4 |
| V1 | MT11 EEGPT | Wphysio | VideoMAE | 固定表征，原fusion与11-head | 6条，复用v2 |
| M2 | EEG-MAE | Wear-MAE | DINO A1 | 固定表征，原fusion与11-head | 6条，新v4 |
| M3 | EEG-MAE | Wphysio | VideoMAE | 固定表征，原fusion与11-head | 6条，新v4 |
| M4 | MT11 EEGPT | Wear-MAE | VideoMAE | 固定表征，原fusion与11-head | 6条，新v4 |
| M5-F | EEG-MAE | Wear-MAE | VideoMAE | 三模态冻结，原fusion与11-head | 6条，新v4 |
| M5-FT | EEG-MAE | Wear-MAE | VideoMAE | 三encoder最后两层及下游联合训练 | 6条，新v4 |
| R0 | 随机初始化MAE | 同构随机MAE | 同构随机MAE | 与M5-FT同样的最后两层及下游 | 6条，新v4 |

最初M2/M3/M4暂缓，之后用户先明确授权M2并指定DINO A1，再授权M3、M4、M5-F、M5-FT、R0全部完成。这个授权更新覆盖了原计划的暂缓及基于M5-F成绩才进入FT的条件；数据、smoke、数值和梯度验收仍保留。

M2低于B0的关键比较背景是它同时替换了MT11 EEGPT和Wear，而E1仅替换EEG。原始匹配矩阵中E1整体也低于B0，W1仅有轻微宏平均优势。“E1和W1都更好”的判断需区分单情绪与11情绪平均。最终v4中W1也低于B0。

V1及含VideoMAE的路线保留各自native valid mask，与DINO A1记录有9个mask差异。现有对照因此包含表征及这部分可用性差异；严格共同有效mask重跑可进一步分离该影响，当前未完成。

### 4.1 R0的解释合同

R0随机初始化整个MAE架构，不加载SSL权重，但只训练与M5-FT相同的encoder末两层、投影、融合和预测头。其冻结prefix也来自随机初始化。R0检验的是“在匹配部分适配范围内，预训练初始化是否有收益”。全encoder监督从头训练属于另一条需要额外授权的实验。

第一轮v2保留原v3 R0：当时随机参数、unmasked输出及原始预处理已核对完全一致，最大差异0。第二轮改变EEG/Wear预处理，按协议重建随机prefix并重跑六条R0；Video随机初始化prefix继续匹配复用。

## 5 执行流程和训练设置

### 5.1 完整执行顺序

Canonical身份与split核对 → Stage-A smoke → 无标签正式预训练 → 全量window token导出 → Video小型产物复制及哈希验收 → 冻结E1/W1/V1 → M2/M3/M4 → M5-F → prefix等价性验收 → M5-FT/R0 smoke → 三seed正式微调 → 合并60行及最终有限值验收。

最终v4只重训EEG/Wear。Video token和prefix保留v2，冻结E1/W1、M2/M3/M4、M5-F共36条，M5-FT/R0共12条，合计48条新增正式结果。B0/V1另保留12条。旧输出和源数据保留，失败路线拥有独立staging。

### 5.2 上游模型设置

| 模态 | 输入与patch | Encoder | Decoder | 遮挡 | 正式训练 |
| --- | --- | --- | --- | --- | --- |
| EEG | 10个1秒patch，每patch 59 × 200值 | 6层，8 heads，256D | 2层重建 | v4固定遮挡41/59通道，贯穿10秒 | 最大100轮，batch128 |
| Wear | PPG每秒125，EDA每秒40，ACC每秒3 × 30 | 独立stem后合并，4层，4 heads，256D | 2层及三分支重建头 | 60%整秒遮挡 | 最大100轮，batch128 |
| Video | 8帧112 × 112，tubelet 2 × 16 × 16，共196token | 6层，8 heads，256D | 2层重建 | 90%tubelet遮挡 | 最大40轮，batch8 |

上游seed240800，learning rate 1e-4。EEG/Wear记录weight decay 1e-4、patience15；Video记录patience8。最终各cell的best epoch见第7节，逐轮训练长度保存在config与reviewed_evidence.json。Video实现为本项目的VideoMAE-style tubelet重建，不据名称推断加载了公开大型VideoMAE checkpoint。

### 5.3 Smoke覆盖

| 阶段 | 覆盖与结果 | 能确认的范围 |
| --- | --- | --- |
| 最初EEG/Wear | cross-day各2轮；EEG验证NMSE 1.296至1.252，Wear 1.136至1.020；形状及mask通过 | 输入、训练、导出闭环 |
| 最初Video | ncc原始MP4 2轮smoke后进入cache及正式队列 | 原始视频解码与训练闭环 |
| 原始partial FT v3 | 2协议 × M5-FT/R0，各10轮，4cell通过 | 无NaN，三尾部有梯度和参数更新 |
| Stage-A修复v2 | 3模态 × 2协议，各15轮，代表性train/val各最多1024行，共6cell通过 | 实现修复及早期表征检查 |
| 普通固定尺度尝试 | EEG/Wear两协议15轮smoke通过，但正式同日EEG第22轮触发健康guard | 短smoke不足以覆盖晚期风险 |
| 最终恢复v4 | EEG/Wear两协议各35轮，4cell通过；随后4项全量预训练完成 | 稳健尺度、平衡损失和通道遮挡组合通过门槛 |
| 最终partial FT | M5-FT/R0两协议各10轮，4cell通过，再完成12条正式结果 | 匹配新预处理后的尾部适配有效 |

Smoke成绩用于执行验收，不作正式性能结论。上游用代表性子集；partial FT smoke按真实完整下游数据验收。每轮表征检查覆盖256个代表性val窗口，不能当作整个数据集的逐行情绪可辨识性证明。

### 5.4 部分微调设置

M5-FT/R0使用float32冻结prefix，EEG/Wear每窗10个时序token，Video每窗196个tubelet token。最后两层前的stem、位置参数和早期encoder blocks冻结；重建decoder不参与情绪下游更新。视频prefix在ncc计算，传回H20后可以训练video tail，原始视频及解码cache留在ncc。

正式下游最大80轮，patience15，event batch64，window chunk128；encoder LR 1e-4，下游LR 1e-3。归一化拟合训练event的初始tokens并固定。尾部encoder在eval模式关闭dropout，同时参数保持requires_grad并接受更新；下游head和modality dropout维持训练行为。融合attention采用FP32 math后端，注意力方程和权重合同保留。

下游checkpoint统一由val macro standardized RMSE最小选择。Test用于报告。Repeated test screens已用于观察路线与修复，当前结果应按探索性对照呈现；若要宣称独立确认，需要另行预声明确认性数据与选择规则。

## 6 最终11情绪结果

原始指标来自最终results.json，独立重算三seed均值与样本SD后列入本表。每种协议每情绪的test event数量为跨日253、同日258。raw r与centered r越高越好，RMSE越低越好。

情绪标签顺序固定为inspired、alert、determined、attentive、active、hostile、nervous、upset、afraid、ashamed、fatigue。各表保留源产物的英文ID。

### 6.1 跨日raw r

| 情绪 | B0 | E1 | W1 | V1 | M2 | M3 | M4 | M5-F | M5-FT | R0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inspired | 0.3761 ± 0.0127 | 0.2953 ± 0.0320 | 0.3710 ± 0.0266 | 0.3805 ± 0.0380 | 0.2993 ± 0.0333 | 0.2853 ± 0.0457 | 0.3842 ± 0.0191 | 0.2785 ± 0.0392 | 0.2802 ± 0.0264 | 0.2469 ± 0.0711 |
| alert | 0.4046 ± 0.0580 | 0.3453 ± 0.0294 | 0.3931 ± 0.0186 | 0.3850 ± 0.0180 | 0.2421 ± 0.0737 | 0.3338 ± 0.0308 | 0.4256 ± 0.0621 | 0.2900 ± 0.0922 | 0.3161 ± 0.0958 | 0.1370 ± 0.0924 |
| determined | 0.4354 ± 0.0072 | 0.3472 ± 0.0076 | 0.4095 ± 0.0019 | 0.4350 ± 0.0367 | 0.3364 ± 0.0214 | 0.2835 ± 0.0555 | 0.4412 ± 0.0315 | 0.3053 ± 0.0374 | 0.2943 ± 0.0496 | 0.1925 ± 0.0166 |
| attentive | 0.5048 ± 0.0061 | 0.4461 ± 0.0286 | 0.4624 ± 0.0272 | 0.3332 ± 0.0586 | 0.4162 ± 0.0382 | 0.1627 ± 0.0958 | 0.3620 ± 0.0185 | 0.2489 ± 0.0188 | 0.2596 ± 0.0178 | 0.2628 ± 0.0268 |
| active | 0.3640 ± 0.0187 | 0.2523 ± 0.0185 | 0.2917 ± 0.0376 | 0.3201 ± 0.0117 | 0.2839 ± 0.0774 | 0.2102 ± 0.0162 | 0.3254 ± 0.0144 | 0.2294 ± 0.0373 | 0.2395 ± 0.0411 | 0.1334 ± 0.0029 |
| hostile | 0.2880 ± 0.0478 | 0.2826 ± 0.0323 | 0.2341 ± 0.0651 | 0.2564 ± 0.0270 | 0.1958 ± 0.0405 | 0.1397 ± 0.0349 | 0.2598 ± 0.0576 | 0.1399 ± 0.0282 | 0.1105 ± 0.0331 | -0.0224 ± 0.0452 |
| nervous | 0.3463 ± 0.0362 | 0.2119 ± 0.0078 | 0.3834 ± 0.0079 | 0.4605 ± 0.0081 | 0.2677 ± 0.0780 | 0.2750 ± 0.0577 | 0.4936 ± 0.0347 | 0.2762 ± 0.0617 | 0.2891 ± 0.0638 | 0.1826 ± 0.0880 |
| upset | 0.4030 ± 0.0495 | 0.2811 ± 0.0020 | 0.3952 ± 0.0185 | 0.3930 ± 0.0438 | 0.2742 ± 0.0655 | 0.1703 ± 0.0507 | 0.3500 ± 0.0070 | 0.1976 ± 0.1185 | 0.2658 ± 0.0245 | 0.1420 ± 0.0754 |
| afraid | 0.4188 ± 0.0258 | 0.1492 ± 0.0275 | 0.4417 ± 0.0265 | 0.4374 ± 0.0098 | 0.2614 ± 0.0599 | 0.1547 ± 0.0347 | 0.4583 ± 0.0164 | 0.1739 ± 0.0630 | 0.2137 ± 0.0201 | 0.2055 ± 0.0543 |
| ashamed | 0.2142 ± 0.0410 | 0.0893 ± 0.0284 | 0.2372 ± 0.0408 | 0.2131 ± 0.0118 | 0.0739 ± 0.0134 | 0.1262 ± 0.0748 | 0.2014 ± 0.0238 | 0.1549 ± 0.0693 | 0.1793 ± 0.0371 | 0.0872 ± 0.0293 |
| fatigue | 0.3513 ± 0.0234 | 0.2504 ± 0.0216 | 0.3602 ± 0.0298 | 0.3260 ± 0.0357 | 0.2187 ± 0.0507 | 0.0819 ± 0.0432 | 0.2914 ± 0.0156 | 0.1867 ± 0.0553 | 0.2055 ± 0.0500 | 0.1537 ± 0.0796 |
| 11情绪平均 | 0.3733 ± 0.0117 | 0.2682 ± 0.0085 | 0.3618 ± 0.0120 | 0.3582 ± 0.0155 | 0.2609 ± 0.0337 | 0.2021 ± 0.0213 | 0.3630 ± 0.0143 | 0.2256 ± 0.0434 | 0.2412 ± 0.0351 | 0.1565 ± 0.0245 |

### 6.2 同日窗口留出raw r

| 情绪 | B0 | E1 | W1 | V1 | M2 | M3 | M4 | M5-F | M5-FT | R0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inspired | 0.4433 ± 0.0081 | 0.3758 ± 0.0152 | 0.4148 ± 0.0336 | 0.3397 ± 0.0212 | 0.3954 ± 0.0035 | 0.2943 ± 0.0467 | 0.3616 ± 0.0115 | 0.3280 ± 0.0129 | 0.2996 ± 0.0479 | 0.2999 ± 0.0445 |
| alert | 0.4447 ± 0.0153 | 0.3441 ± 0.0167 | 0.4632 ± 0.0208 | 0.3879 ± 0.0119 | 0.3924 ± 0.0115 | 0.2937 ± 0.0260 | 0.4119 ± 0.0308 | 0.2995 ± 0.0346 | 0.2933 ± 0.0256 | 0.3107 ± 0.1191 |
| determined | 0.5359 ± 0.0090 | 0.4356 ± 0.0103 | 0.5251 ± 0.0159 | 0.4809 ± 0.0182 | 0.4210 ± 0.0018 | 0.4028 ± 0.0030 | 0.5134 ± 0.0082 | 0.4030 ± 0.0114 | 0.4158 ± 0.0053 | 0.3848 ± 0.0351 |
| attentive | 0.5704 ± 0.0025 | 0.4559 ± 0.0098 | 0.5695 ± 0.0014 | 0.5169 ± 0.0308 | 0.5138 ± 0.0057 | 0.4151 ± 0.0146 | 0.5503 ± 0.0203 | 0.4985 ± 0.0201 | 0.4722 ± 0.0264 | 0.4614 ± 0.0377 |
| active | 0.5179 ± 0.0238 | 0.4207 ± 0.0219 | 0.5120 ± 0.0309 | 0.4752 ± 0.0202 | 0.4386 ± 0.0246 | 0.3890 ± 0.0058 | 0.5154 ± 0.0287 | 0.4235 ± 0.0075 | 0.3911 ± 0.0377 | 0.3543 ± 0.0682 |
| hostile | 0.3236 ± 0.0164 | 0.2271 ± 0.0224 | 0.3480 ± 0.0095 | 0.3059 ± 0.0087 | 0.2650 ± 0.0365 | 0.2526 ± 0.0119 | 0.2968 ± 0.0126 | 0.2336 ± 0.0185 | 0.2311 ± 0.0151 | 0.2039 ± 0.0484 |
| nervous | 0.3949 ± 0.0151 | 0.3086 ± 0.0094 | 0.3904 ± 0.0270 | 0.3837 ± 0.0182 | 0.3068 ± 0.0041 | 0.2754 ± 0.0221 | 0.3912 ± 0.0182 | 0.2742 ± 0.0286 | 0.2877 ± 0.0267 | 0.2328 ± 0.0467 |
| upset | 0.3463 ± 0.0251 | 0.2220 ± 0.0226 | 0.3529 ± 0.0130 | 0.3524 ± 0.0115 | 0.3036 ± 0.0697 | 0.1965 ± 0.0191 | 0.3421 ± 0.0398 | 0.2456 ± 0.0173 | 0.2304 ± 0.0156 | 0.2047 ± 0.0250 |
| afraid | 0.4793 ± 0.0239 | 0.2773 ± 0.0297 | 0.4616 ± 0.0114 | 0.4483 ± 0.0170 | 0.3214 ± 0.0127 | 0.2595 ± 0.0066 | 0.4313 ± 0.0070 | 0.2674 ± 0.0302 | 0.2593 ± 0.0082 | 0.3028 ± 0.0129 |
| ashamed | 0.3224 ± 0.0255 | 0.1946 ± 0.0091 | 0.3421 ± 0.0348 | 0.2465 ± 0.0284 | 0.2713 ± 0.0054 | 0.2037 ± 0.0308 | 0.2310 ± 0.0151 | 0.2241 ± 0.0310 | 0.2102 ± 0.0192 | 0.2041 ± 0.0518 |
| fatigue | 0.4311 ± 0.0044 | 0.3421 ± 0.0037 | 0.3995 ± 0.0171 | 0.3797 ± 0.0107 | 0.3484 ± 0.0440 | 0.2391 ± 0.0085 | 0.3948 ± 0.0335 | 0.2529 ± 0.0171 | 0.2499 ± 0.0262 | 0.1432 ± 0.0199 |
| 11情绪平均 | 0.4373 ± 0.0045 | 0.3276 ± 0.0047 | 0.4345 ± 0.0096 | 0.3925 ± 0.0099 | 0.3616 ± 0.0119 | 0.2929 ± 0.0034 | 0.4036 ± 0.0072 | 0.3137 ± 0.0056 | 0.3037 ± 0.0118 | 0.2821 ± 0.0410 |

### 6.3 跨日centered r与RMSE

| 路线 | centered r | RMSE |
| --- | --- | --- |
| B0 | 0.1293 ± 0.0151 | 0.8390 ± 0.0028 |
| E1 | 0.1150 ± 0.0144 | 0.8696 ± 0.0034 |
| W1 | 0.1480 ± 0.0152 | 0.8473 ± 0.0051 |
| V1 | 0.0905 ± 0.0287 | 0.8447 ± 0.0080 |
| M2 | 0.1187 ± 0.0235 | 0.8731 ± 0.0076 |
| M3 | 0.0798 ± 0.0311 | 0.8870 ± 0.0044 |
| M4 | 0.1268 ± 0.0165 | 0.8473 ± 0.0072 |
| M5-F | 0.0893 ± 0.0112 | 0.8846 ± 0.0126 |
| M5-FT | 0.0958 ± 0.0023 | 0.8800 ± 0.0105 |
| R0 | 0.0653 ± 0.0023 | 0.8933 ± 0.0002 |

### 6.4 同日centered r与RMSE

| 路线 | centered r | RMSE |
| --- | --- | --- |
| B0 | 0.1908 ± 0.0096 | 0.8185 ± 0.0020 |
| E1 | 0.1394 ± 0.0067 | 0.8605 ± 0.0047 |
| W1 | 0.1990 ± 0.0095 | 0.8241 ± 0.0042 |
| V1 | 0.1685 ± 0.0093 | 0.8470 ± 0.0080 |
| M2 | 0.1570 ± 0.0046 | 0.8519 ± 0.0024 |
| M3 | 0.1409 ± 0.0061 | 0.8863 ± 0.0041 |
| M4 | 0.1921 ± 0.0009 | 0.8416 ± 0.0037 |
| M5-F | 0.1395 ± 0.0049 | 0.8743 ± 0.0033 |
| M5-FT | 0.1383 ± 0.0081 | 0.8785 ± 0.0091 |
| R0 | 0.0985 ± 0.0282 | 0.8841 ± 0.0134 |

centered r使用源metrics的within_subject_centered_r。它控制被试均值差异后的相关性，有助于区分raw r所包含的被试间稳定差异；这一个指标仍不足以排除全部身份或日期线索。三种指标均在每seed内对11标签等权平均，再汇总三seed。

### 6.5 预训练初始化对照

下面为配对M5-FT减R0的宏平均raw r。

| 协议 | seed240800 | seed240801 | seed240802 | 平均差 | 胜出 |
| --- | --- | --- | --- | --- | --- |
| 跨日 | 0.1249 | 0.0430 | 0.0864 | 0.0848 | 3/3 |
| 同日 | 0.0536 | -0.0040 | 0.0153 | 0.0216 | 2/3 |

跨日平均增益0.0848，三个seed均为正；同日平均增益0.0216，两个seed为正。这提供了匹配部分微调下的预训练收益证据。三个seed及两协议不构成正式统计显著性证明。M5-F的融合预测进一步表明冻结MAE表征可供当前下游利用；单模态情绪信息尚需独立探针量化。

## 7 两轮修复的问题和实际效果

这里的“两轮”指Stage-A v2确定性实现修复和最终v4组合恢复。更早的partial-FT v3数值稳定化，以及最终prefix/聚合脚本修复，属于执行可靠性修复，分别记录于第8节。

### 7.1 第一轮v2

| 原问题 | 修复 | 验证与产物 |
| --- | --- | --- |
| EEG/Wear先加position后整token替换，masked位置丢失位置编码 | mask内容后再加position | 固定seed/dropout关闭时，旧masked输出位置差为0，修复测试位置差0.9147 |
| 验证掩码随机且可能扰动训练流 | 固定数量、独立seed+100003的val mask | 重复验证一致，train RNG与val隔离 |
| Video模型构造后才设置seed | seed放在模型构造前 | 相同seed初始化一致 |
| 同日Video embedding近常数仍完成训练 | 每epoch检查relative variation、std及有效秩 | 低于0.001停止；新两协议Video均通过 |
| Wear整体loss掩盖PPG/EDA/ACC差异 | 分支loss及零预测基线 | 能看到不同分支重建难度 |
| 数值失效可能污染权重 | 更新前检查loss、梯度，导出检查embedding | 非有限值阻止optimizer更新及后续阶段 |

第一轮保持原逐秒预处理、架构尺寸、canonical、split及下游合同，重训三模态两协议共6个上游模型，并完成48条正式MAE候选。R0依据未改变的初始化与unmasked前向合同复用。

### 7.2 第二轮v4

普通train-channel mean/std版本原拟用于独立尺度对照，在同日EEG第22轮出现relative variation 0.000307，低于0.001门槛。进一步检查表明，最高能100个训练窗占跨日能量94.05%、同日90.59%；同日val对应占比99.94%。该损失对少数高能样本过度加权。

稳健尺度加整秒遮挡的35轮诊断通过非坍缩检查，但EEG重建未优于零预测，未释放正式训练。最终通过的恢复条件同时调整了固定尺度、重建损失与EEG遮挡。

| 问题 | 最终改动 | 保留的边界 |
| --- | --- | --- |
| 每秒z-score消除窗间幅度与慢变化 | 训练行拟合每通道median window center和median window RMS，随后固定应用 | val/test和prefix用相同统计；无逐秒重定标 |
| 高能窗口主导重建 | 每窗masked MSE除以完整目标的归一化能量，分母floor1，窗口等权 | 原值不clip，窗口不删除；Wear分支独立计算 |
| 少量高能样本掩盖多数窗口低变化 | 增加median-window relative variation门槛 | 全局及典型窗口变化都需超过0.001 |
| EEG整秒重建接近零预测 | 每窗遮挡41/59通道，从18个可见通道重建 | 遮挡raw内容在stem前置零，目标及梯度泄漏测试通过 |
| 只看健康度仍可能得到无用重建 | 最佳val reconstruction必须优于匹配zero predictor | Stage A只看输入及重建，不读情绪标签 |

最终preprocessing_version为train_channel_robust_zscore_v1，training_version为position_preserved_balanced_channel_eeg_v4。metadata里的std保存median RMS，需结合scale_estimator读取。这个条件是组合恢复，无法把最终增益归因到单独一个因素。Wear仍用整秒遮挡；Video、B0、V1保持v2对应内容；R0按新预处理重跑。

### 7.3 所选checkpoint的表征检查

| 版本 | 协议 | 模态 | 相对变化 | 典型窗口变化 | 有效秩 | 最佳轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| v2 | 跨日 | eeg | 0.0427 | 未纳入此版 | 2.06 | 75 |
| v2 | 跨日 | wear | 0.9005 | 未纳入此版 | 15.94 | 95 |
| v2 | 同日 | eeg | 0.0764 | 未纳入此版 | 4.43 | 54 |
| v2 | 同日 | wear | 0.9135 | 未纳入此版 | 15.96 | 96 |
| v4 | 跨日 | eeg | 0.7707 | 0.7620 | 63.75 | 100 |
| v4 | 跨日 | wear | 0.7484 | 0.7119 | 5.05 | 99 |
| v4 | 同日 | eeg | 0.7898 | 0.7732 | 48.04 | 99 |
| v4 | 同日 | wear | 0.7513 | 0.7311 | 5.27 | 98 |
| v2_video | 跨日 | video | 0.7724 | 未纳入此版 | 8.54 | 3 |
| v2_video | 同日 | video | 0.9238 | 未纳入此版 | 62.05 | 40 |

relative variation为中心化特征RMS除以特征RMS。有效秩是特征变化分布的诊断值，受probe、预处理和任务影响。通过0.001门槛排除了本次观察到的近常量输出，不能据此宣称全部256维独立有效。

本次读取的v2 EEG/Wear四条、v2 Video两条及v4 EEG/Wear四条，共10份config，其记录的每epoch均未标记collapsed。所选checkpoint见表。v2 EEG虽通过低门槛，有效秩仍约2至4；v4 EEG在当前probe上达到约48至64。v4 Wear有效秩约5，低于v2约16，这与其跨日成绩回退共同提示需要保留分支和下游诊断。

### 7.4 v4重建与零预测

| 协议 | 模态 | 验证loss | 零预测loss | loss比零预测 |
| --- | --- | --- | --- | --- |
| 跨日 | eeg | 0.5586 | 0.8785 | 0.6358 |
| 跨日 | wear | 0.0429 | 0.7206 | 0.0595 |
| 同日 | eeg | 0.5481 | 0.8475 | 0.6467 |
| 同日 | wear | 0.0434 | 0.7338 | 0.0592 |

Wear的零预测loss在此表对PPG、EDA、ACC三分支等权取均值，与聚合重建loss的合同一致。比值均小于1，表明所选模型对当前遮挡任务优于匹配零预测。v2与v4的目标尺度及重建任务发生变化，raw reconstruction loss的绝对值不能跨版本直接比较。

### 7.5 跨日成绩前后对照

| 路线 | 修改前 | 第一轮v2 | 第二轮v4 | v4减v2 |
| --- | --- | --- | --- | --- |
| B0 | 0.3733 | 0.3733 | 0.3733 | 0.0000 |
| E1 | 0.2461 | 0.2509 | 0.2682 | 0.0173 |
| W1 | 0.3767 | 0.3802 | 0.3618 | -0.0184 |
| V1 | 0.3640 | 0.3582 | 0.3582 | 0.0000 |
| M2 | 0.2499 | 0.2523 | 0.2609 | 0.0086 |
| M3 | 0.2107 | 0.1826 | 0.2021 | 0.0195 |
| M4 | 0.3630 | 0.3649 | 0.3630 | -0.0019 |
| M5-F | 0.1970 | 0.2031 | 0.2256 | 0.0225 |
| M5-FT | 0.2152 | 0.2245 | 0.2412 | 0.0168 |
| R0 | 0.1024 | 0.1024 | 0.1565 | 0.0540 |

### 7.6 同日成绩前后对照

| 路线 | 修改前 | 第一轮v2 | 第二轮v4 | v4减v2 |
| --- | --- | --- | --- | --- |
| B0 | 0.4373 | 0.4373 | 0.4373 | 0.0000 |
| E1 | 0.3197 | 0.3182 | 0.3276 | 0.0094 |
| W1 | 0.4440 | 0.4356 | 0.4345 | -0.0012 |
| V1 | 0.4261 | 0.3925 | 0.3925 | 0.0000 |
| M2 | 0.3405 | 0.3404 | 0.3616 | 0.0213 |
| M3 | 0.2251 | 0.2894 | 0.2929 | 0.0035 |
| M4 | 0.4311 | 0.3975 | 0.4036 | 0.0061 |
| M5-F | 0.2412 | 0.2997 | 0.3137 | 0.0139 |
| M5-FT | 0.1312 | 0.2806 | 0.3037 | 0.0231 |
| R0 | 0.2720 | 0.2720 | 0.2821 | 0.0101 |

“修改前”指完成原始A1+MT11匹配矩阵后的有效版本，其M5-FT/R0已经包含partial_v3数值修复。第一轮v2沿用原R0；第二轮v4沿用第一轮V1；B0在三个版本中始终保留。两张表统一重新计算宏平均均值，避免历史答复中总体SD与样本SD混用。

### 7.7 效果判断

第一轮最明确的恢复是同日M5-FT从0.1312到0.2806，增加0.1495，三个seed及11/11情绪均提高。M3、M5-F同日也提高；V1两个协议的宏平均下降。技术健康修复与原始V1的分数保持属于不同检验维度，原近常量Video无需因某个融合分数较高而保留为新有效encoder。

第二轮相对v2，E1跨日增加0.0173、同日增加0.0094；M5-F跨日增加0.0225、同日增加0.0139，两协议均3/3 seed提高；M5-FT跨日增加0.0168、同日增加0.0231，同日3/3 seed及10/11情绪提高。W1跨日下降0.0184，3/3 seed均下降。

第二轮跨日R0提高0.0540，超过M5-FT的绝对提升；M5-FT减R0的优势从v2 0.1220缩至v4 0.0848。同日优势从0.0087增至0.0216。因此，第二轮部分预测收益与输入处理改善一致，预训练净收益需参考匹配R0，不能把全部涨分归于SSL。

## 8 故障原因与恢复记录

| 故障 | 定位 | 修复和继续方式 | 最终状态 |
| --- | --- | --- | --- |
| 旧H20计算繁忙，只有1张可见GPU | 利用率100%，空闲显存不能代表可立即训练；部分占用进程位于不同namespace | 挂等待队列，先完成小型CPU/GPU smoke | 旧等待已由后续完成覆盖 |
| Video cache被坏MP4中断 | sub4/0409/DJI_0337.MP4，约1150.007秒帧无法解码，已完成2250clip | resume manifest保留已完成行，隔离失败并置video mask=false | 最终5条decode失败隔离，18012有效 |
| ncc cache进度慢 | CIFS读盘及CPU解码，进程出现D-state；GPU0高占用属于其他PID | 从本地decoded cache继续，分离cache与训练进度 | 原始及v2 Video均已完成 |
| H20旧端口失联 | TCP可达后SSH密钥交换被关闭 | 按用户指定迁移到wangzw@124.174.8.252:10022，核验输入与artifact | 当前实际SSH登录成功 |
| 共享实验根不可写 | canonical共享目录权限限制 | 使用用户可写staging，明确runner及output路径 | 后续M2、v2、v4独立输出 |
| 原始M5-FT同日NaN | Video近常量token，固定std约3.3e-5；train-mode dropout使标准化值从约2.6放大至8900，梯度非有限 | 关闭encoder dropout、FP32 math attention、每batch guard；统一v3重跑12条 | 全部通过smoke与正式矩阵 |
| v2本地复制控制器退出 | 出现129进程退出及0xC000013A控制事件；具体终止方未确认 | 一次性Windows任务仅恢复prefix复制，完成哈希handoff | 两protocol prefix及10文件哈希通过 |
| 普通固定尺度同日EEG第22轮近常量 | 高能窗主导MSE，15轮smoke漏掉晚期失效 | 隔离原输出，35轮smoke、稳健尺度与平衡损失、通道遮挡 | v4通过4项正式Stage A |
| Prefix与Stage-A token差异 | batch64与batch128 FP32路径差，最大0.0016806126，仅2/28819行越过2e-4；相同batch组合误差0 | 123新增match-stage-a-batch-size，132按checkpoint批大小导出；阈值不放宽 | 四份pretrained EEG/Wear prefix误差0，保留36条冻结结果 |
| 最终合并参数解包错误 | out,old接到两个路径加第三个preprocessing version | 只从sys.argv[1:3]取两路径，第三参数独立读；仅重跑聚合 | 60行通过，180训练artifact大小/mtime保持 |
| 用户自行检查脚本语法错误 | impot json及gep均为拼写错误 | 给出可复制的正确import/grep及SSH脚本 | 该错误不反映Video训练失败 |

最终prefix修复后45项回归通过；增加真实三参数merge回归后46项通过。代码检查、smoke、真实GPU前向等价、梯度/参数更新审计均已用于对应阶段验收。attention稳定后端前向差最大2.98e-8；initial token等价preflight约9.54e-7。各检查的对象和阈值不同，不能把这些数值拼成同一个误差指标。

用户最后明确授权：已授权队列因程序或执行错误停下时，直接诊断、范围内修复、验证并接续，保留有效run、数据、基线、协议、seed及验收门槛。新科学协议、破坏源数据、新费用或其他越界事项仍需另行方向。该规则已记录于AGENTS.md。

## 9 本聊天的重要决定与里程碑

| 时间或阶段 | 请求及决定 | 完成内容 |
| --- | --- | --- |
| 初始计划 | M2/M3/M4先暂缓，其他按顺序 | canonical EEG/Wear smoke及等待队列 |
| 数据讨论 | 解释shared_unlabeled_pretrain、index、split、SSL val/test | 选择canonical-only，扩展raw pool留作独立对照 |
| 视频发现 | ncc可能有视频，要求原服务器生成后传回 | 找到原MP4，启动Video smoke、cache、两协议队列 |
| 视频中断 | 修复并继续 | 缓存恢复、坏clip隔离、valid mask保留 |
| 基线纠正 | 用最好的A1_Wphysio_no_audio完整route | 从窗口级label-free B0切到A1+MT11匹配event表 |
| M2授权 | 试M2并注意DINO A1 | 完成两协议三seed，确认同时替换EEG/Wear |
| 2026年10月1日前后 | Video完成核验 | 原始VIDEO_STAGE_A_COMPLETE及两份token存在 |
| 2026年10月7日 | 更换H20为124.174.8.252:10022 | 登录新H20、核验可用输入，复制Video并完成V1 |
| 2026年10月7日 | GitHub同步 | main提交eb87ca3，共13文件，数据/checkpoint不入Git |
| 2026年10月7日 | 把剩余5路线都做完 | M3/M4/M5-F、M5-FT/R0，修复partial FT NaN |
| 2026年10月8日 | 排查三模态MAE，先计划再执行 | 实现v2位置编码、种子、固定mask、健康与数值guard |
| 2026年10月8日 | 退出检查与恢复 | 一次性任务恢复Video prefix传输，服务器队列继续 |
| 2026年10月8日 | v2完成 | 6个上游及48个下游候选，R0匹配复用 |
| 2026年10月8日至9日 | 执行第二阶段固定尺度 | 普通尺度失败，时间遮挡诊断未过零预测，进入组合恢复 |
| 2026年10月9日 | 修复prefix导出并继续 | 批大小匹配，保留36条冻结结果，完成12条FT/R0 |
| 2026年10月9日17:03 | 修复合并后完成 | 48新加12复用，60行最终矩阵及同步哈希 |
| 完成后的确认 | 两轮是否有效、问题是否修好、下游是否做过 | 分别交付技术修复证据、三阶段对照及M5-FT/R0配对结果 |
| 2026年10月10日 | 全聊天详尽handout | 封存当前结果、历史决定、源码入口及复现检查 |

时区可核对的最终完成时间使用Asia/Shanghai。ncc最初marker显示10月1日04:45的文件时间，早期日志未统一声明时区；本文仅据它记录10月1日完成，不把该文件时间强行转换为北京时间。

原聊天87回合的用户请求与已交付答复保留在本手册证据目录的chat_history.json。频繁状态查询和当时的ETA通过这份档案保留；主文使用最终完成状态，避免把已结束的等待误读为当前阻塞。

## 10 服务器与artifact地图

### 10.1 当前执行主机

| 主机 | 连接与角色 | 数据和输出策略 |
| --- | --- | --- |
| 新H20 | ssh -p 10022 wangzw@124.174.8.252 | EEG/Wear预训练与全部情绪下游；8张可见H20，本次确认本MAE队列已退出 |
| ncc_serve_4090 | 使用现有SSH alias，用户lzs | 原始MP4、decoded cache、VideoMAE及Video prefix导出 |
| 本地Windows | G:/Daily Multimodal | 源码、资料、轻量同步结果、relay传输、handout |
| 旧huoshan endpoint | 历史36083入口 | 当前不用它作为新H20结果可用性的依据 |

GPU空闲是时间快照。新H20本次查询可见8张H20，个别卡有其他占用；本聊天任务的完成状态由marker和结果矩阵确认。旧H20当时仅1张GPU的结论只适用于原容器可见资源。

### 10.2 Canonical路径

~~~
/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new/X.npy
/vePFS-0x0d/DailyEEG/splits_new/cross_day
/vePFS-0x0d/DailyEEG/splits_new/within_subject_day
/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned
/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/index/eeg_aligned_window_index.jsonl
/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/runtime/envs/eegpt-gpu-min/bin/python
~~~

Wphysio及原MT11/DINO A1的源文件由reference bag的来源metadata追溯。复现时核对source_npz_json、route_id、supervision_boundary和sample_id，route名字与shape本身不能单独证明来源。

### 10.3 训练版本输出

| 版本与作用 | 根目录 |
| --- | --- |
| 原始VideoMAE | /home/lzs/DailyVideoMAE_20260927/outputs/mae_20260927 |
| 原始M2 | /home/wangzw/outputs/mae_mt11_m2_20260930 |
| 原始V1 | /home/wangzw/outputs/mae_mt11_v1_20261007 |
| 原始剩余路线 | /home/wangzw/outputs/mae_remaining_20261007 |
| 第一轮v2 H20 | /home/wangzw/mae_repair_20261008 |
| 第一轮v2 ncc | /tmp/wangzw_mae_repair_20261008 |
| 普通固定尺度失败现场 | /home/wangzw/mae_norm_20261008 |
| 稳健尺度整秒遮挡诊断 | /home/wangzw/mae_norm_robust_20261009 |
| 最终v4 H20 | /home/wangzw/mae_norm_channel_20261009 |
| 最终本地结果 | G:/Daily Multimodal/outputs/server_sync/mae_norm_channel_20261009 |
| 本次handout证据包 | G:/Daily Multimodal/outputs/handout/mae_chat_handout_20261010 |

### 10.4 最终v4关键文件

以/home/wangzw/mae_norm_channel_20261009为根。

~~~
outputs/NORMALIZATION_COMPLETE
outputs/CPU_REGRESSION_COMPLETE
outputs/NORMALIZATION_PREFIX_COMPLETE
outputs/NORMALIZATION_PARTIAL_SMOKE_COMPLETE
outputs/stage_a_train_channel/formal/<protocol>/<modality>_seed_240800/
outputs/prefixes_train_channel/<pretrained|random>/<protocol>/<modality>/
outputs/downstream_train_channel/<single|pairs|all|partial>/formal/results.json
outputs/downstream_train_channel/results.json
outputs/downstream_train_channel/raw_r_tables.md
outputs/downstream_train_channel/logs/
~~~

每个上游cell包含checkpoint.pt、config.json、window_embeddings.npz。prefix manifest记录初始化、checkpoint/token/encoder来源指纹、归一化、requested/resolved batch size及等价性误差。随机prefix的权重种子保持匹配，原始信号归一化按protocol分别拟合。

Video v2在H20的路径为/home/wangzw/mae_repair_20261008/outputs/video_stage_a_v2/<protocol>/video_seed_240800。最终v4 Video prefix指向已验收的v2缓存，H20继续训练尾部时无需MP4或大型decoded cache。

### 10.5 同步与哈希

本次结果JSON与表格的服务器、本地SHA256一致。完整JSON大小1,423,993字节；原raw-r Markdown表4,910字节。

~~~
results.json
12b3bb3247bcdef11104376245114c3ecf8b80f1ad3a0322e1dbd5f7c0352091

raw_r_tables.md
016e0515b28cdcf09025ac45f9d7335c056f0c113872bb006b7eb745a73ff50d
~~~

最终JSON含reference_route、protocols、seeds、table_conditions、raw_normalization、video_reused_from、results、summary、macro_summary。原始raw-r表未含平均行；本handout从逐label/seed原值重算后补充。上游和Video transfer历史记录使用复制及哈希验证，源文件继续保留。

### 10.6 Git状态

本聊天已授权并完成一次GitHub同步，提交为eb87ca3c8b2fa40996ba4f4be93344720f589e3e，仓库Victoria-7k/Daily-Multimodal-EEG的main。该提交含当时VideoMAE、E1/W1/M2/V1入口和文档，共13个文件。

后续v2/v4源码、122至132新增入口、测试和文档处于当前working tree，不能据旧commit宣称这些后续修复已推送。本次handout仅整理与验证，没有新增commit或push。实验数据、embedding、prefix及checkpoint保留在各artifact目录，未纳入Git提交。

## 11 可复制的状态检查与复现指令

以下命令默认在Windows PowerShell执行。复杂Python或Bash用单引号here-string通过stdin传送，以免PowerShell提前解析$、重定向或命令替换。全部状态检查为只读，不会启动新训练。

### 11.1 实际登录与GPU

~~~powershell
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ConnectionAttempts=1 -p 10022 wangzw@124.174.8.252 'id -un; pwd'
ssh -p 10022 wangzw@124.174.8.252 'nvidia-smi'
ssh -p 10022 wangzw@124.174.8.252 'pgrep -af "[1]32_queue_mae_train_channel_normalization|[1]24_run_mae_partial_ft|[1]18_run_mae_mt11_event_ablation"'
~~~

登录输出wangzw才确认该endpoint可执行。pgrep无匹配时exit code 1属于“未发现进程”；它需要与完成marker、完整结果联合判断。

### 11.2 60行最终矩阵验收

~~~powershell
@'
import json, math
from pathlib import Path
base = Path("/home/wangzw/mae_norm_channel_20261009/outputs")
j = json.loads((base / "downstream_train_channel/results.json").read_text())
routes = ["B0","E1","W1","V1","M2","M3","M4","M5-F","M5-FT","R0"]
protocols = ["cross_day","within_subject_day"]
seeds = [240800,240801,240802]
expected = {(p,c,s) for p in protocols for c in routes for s in seeds}
rows = j["results"]
keys = [(r["protocol"],r["condition"],r["seed"]) for r in rows]
assert len(rows) == 60 and len(set(keys)) == 60 and set(keys) == expected
assert (base / "NORMALIZATION_COMPLETE").exists()
assert not (base / "NORMALIZATION_QUEUE_FAILED").exists()
for r in rows:
    assert r["metrics"]["status"] == "ok"
    for split in ("val","test"):
        m = r["metrics"][split]["per_label"]
        assert len(m) == 11
        assert all(math.isfinite(v["raw_r"]) for v in m.values())
print("PASS: 60 unique cells, both protocols, 3 seeds, finite 11-label val/test")
'@ | ssh -p 10022 wangzw@124.174.8.252 '/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/runtime/envs/eegpt-gpu-min/bin/python -'
~~~

成功标准为PASS文本且SSH返回0。只有marker而缺少预期cell，或有duplicate/non-finite，都不算完成。

### 11.3 独立重算宏平均和样本SD

~~~powershell
@'
import json, numpy as np
from pathlib import Path
f = Path("/home/wangzw/mae_norm_channel_20261009/outputs/downstream_train_channel/results.json")
j = json.loads(f.read_text())
for p in j["protocols"]:
    print(p)
    for c in j["table_conditions"]:
        rows = sorted(
            [r for r in j["results"] if r["protocol"] == p and r["condition"] == c],
            key=lambda r: r["seed"])
        values = np.array([
            np.mean([z["raw_r"] for z in r["metrics"]["test"]["per_label"].values()])
            for r in rows])
        print(c, f"{values.mean():.4f} +/- {values.std(ddof=1):.4f}")
'@ | ssh -p 10022 wangzw@124.174.8.252 '/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/runtime/envs/eegpt-gpu-min/bin/python -'
~~~

### 11.4 ncc Video完成检查

~~~powershell
@'
from pathlib import Path
import numpy as np
root = Path("/home/lzs/DailyVideoMAE_20260927/outputs/mae_20260927")
assert (root / "VIDEO_STAGE_A_COMPLETE").exists()
for p in ("cross_day","within_subject_day"):
    f = root / p / "video_seed_240800/window_embeddings.npz"
    with np.load(f, allow_pickle=True) as z:
        e = z["embedding"]
        v = z["valid_mask"].astype(bool)
        assert e.shape == (28819,256)
        assert int(v.sum()) == 18012
        assert np.isfinite(e).all() and (e[~v] == 0).all()
        print(p, "PASS", e.shape, "valid", int(v.sum()))
'@ | ssh ncc_serve_4090 '/home/lzs/miniconda3/envs/eeg3dim/bin/python -'
~~~

这是原始VideoMAE检查。v2两协议文件把root改为/tmp/wangzw_mae_repair_20261008/outputs/stage_a_v2/formal后检查token；原始VIDEO_STAGE_A_COMPLETE位于原始root，不应在v2 formal目录中错误寻找同名marker。Canonical sample_id顺序可另外对齐H20索引，现有两份v2在H20已验收。

### 11.5 哈希与复制

~~~powershell
ssh -p 10022 wangzw@124.174.8.252 'sha256sum /home/wangzw/mae_norm_channel_20261009/outputs/downstream_train_channel/results.json /home/wangzw/mae_norm_channel_20261009/outputs/downstream_train_channel/raw_r_tables.md'
Get-FileHash -Algorithm SHA256 'G:/Daily Multimodal/outputs/server_sync/mae_norm_channel_20261009/results.json'
Get-FileHash -Algorithm SHA256 'G:/Daily Multimodal/outputs/server_sync/mae_norm_channel_20261009/raw_r_tables.md'
~~~

若需要重新复制，先确认目标目录及现有文件，使用scp复制到明确输出目录，保留服务器源文件，再核对双方SHA256。无需搬运raw MP4或decoded mmap。

### 11.6 源码回归测试

~~~powershell
@'
cd /home/wangzw/mae_norm_channel_20261009 || exit 1
export PYTHONPATH=src
/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/runtime/envs/eegpt-gpu-min/bin/python -m unittest discover -s tests -p 'test_mae*.py' -v
/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/runtime/envs/eegpt-gpu-min/bin/python -m unittest discover -s tests -p 'test_modality_mae.py' -v
'@ | ssh -p 10022 wangzw@124.174.8.252 'tr -d "\r" | bash -s'
~~~

最终修复时合计46项通过。应运行该staging的当前源码与tests，避免在旧canonical源码目录中误验证另一个版本。该命令是回归检查，耗用CPU，不启动正式训练。

### 11.7 受控续跑入口

当前队列已完成，无需重启。将来若同一已授权队列出现程序错误，先按错误恢复合同定位与修复，再核验已完成组及CPU regression marker。最终恢复必须同时指定MAE_NORM_STAGING=/home/wangzw/mae_norm_channel_20261009和MAE_NORM_MODE=train_channel_robust，否则132默认进入普通train_channel旧目录。

入口132使用flock避免重复队列；已有冻结组被保留并重新验收，旧failed marker被归档。Prefix和partial阶段是否重跑需要根据已完成artifact判定，不能把“运行入口”当作纯状态查询。新科学设置应建新目录并重新建立匹配R0及选择规则。

## 12 代码和测试交接地图

| 入口或源文件 | 职责 |
| --- | --- |
| 112_run_modality_mae.py | EEG/Wear无标签训练、原始信号归一化、验证选择与canonical token |
| 113_queue_modality_mae_stage_a.sh | 早期H20四项上游队列 |
| 114_run_video_mae.py | ncc Video tubelet重建、解码cache恢复及token |
| 115_queue_video_mae_stage_a.sh | 原始Video smoke/cache/两协议训练 |
| 116_run_mae_window_fusion.py与117队列 | 早期未匹配MT11/A1的窗口级B0/E1/W1，历史保留 |
| 118_run_mae_mt11_event_ablation.py | 当前A1+MT11匹配冻结替换及结果表 |
| 119_queue_mae_mt11_event_ablation.sh | 原始匹配E1/W1 |
| 120_queue_m2_mt11_event_ablation.sh | M2且保持DINO A1 |
| 121_queue_v1_mt11_event_ablation.sh | 新H20单Video替换V1 |
| 122_queue_remaining_frozen_mae.sh | M3/M4/M5-F两协议三seed |
| 123_export_mae_frozen_prefix.py | 最后两层前prefix、初始化及等价检查、批大小匹配 |
| 124_run_mae_partial_ft.py | M5-FT/R0尾部训练、数值/梯度/更新guard |
| 125_prepare_mae_prefixes.sh | 原始H20/ncc三种初始化prefix准备 |
| 126_queue_mae_partial_ft.sh | 原始partial_v3 smoke及正式矩阵 |
| 127_transfer_video_prefixes.ps1 | Windows relay复制及哈希gate |
| 128_queue_mae_repair_stage_a.sh | v2六项smoke和上游正式队列 |
| 129_continue_mae_repair_pipeline.ps1 | 一次性v2本地传输与handoff控制器 |
| 130_queue_mae_repair_downstream.sh | v2冻结与FT 48候选 |
| 131_prepare_mae_repair_prefixes.sh | v2两协议prefix及来源指纹 |
| 132_queue_mae_train_channel_normalization.sh | 最终v4组合恢复、48候选及60行合并 |
| src/daily_multimodal/training/modality_mae.py | MAE模型、position、raw尺度、平衡loss与health |
| tests/test_mae_repairs.py | 位置编码、固定mask、初始化及健康等修复 |
| tests/test_mae_normalization.py | 训练行统计、固定尺度及prefix一致性 |
| tests/test_mae_energy_balance.py | 能量平衡、典型窗口、通道遮挡及泄漏 |
| tests/test_mae_remaining_routes.py | 尾部FT、随机控制、prefix及三参数合并 |
| tests/test_modality_mae.py | 基础MAE形状与功能合同 |

脚本均位于G:/Daily Multimodal/scripts/multilabel；源码和tests按表内相对路径位于仓库根。111_queue_st11_within_subject_day_splits_new.sh属于历史ST11矩阵上下文，其任务与当前MAE队列分开。

主要阅读材料为当前MAE路线表、repo-docs中的commands-and-artifacts、scripts/multilabel/README.md，以及本文对应的reviewed_evidence.json。它们分别承担研究合同、运维查找、入口清单和数值证据角色。

## 13 已完成与后续可选工作

后续下游适配已文档化为[第二轮计划](mae_round2_downstream_adaptation_plan_20261010.md)：EEG单模态11情绪适配、保守EEG尾部联合微调和同损失B0/MAE事件目标对照。该计划承接第一轮当前产物与协议边界；第二轮尚未实现或启动，本文历史矩阵保持原状态。

| 工作 | 状态 | 后续决策意义 |
| --- | --- | --- |
| 原始EEG/Wear/Video Stage A及embedding | 已完成、保留 | 原始匹配对照 |
| A1+MT11 B0及9条候选正式矩阵 | 已完成 | 60cell原始版本 |
| 第一轮v2上游与下游 | 已完成 | 实现修复效果，60cell含复用参考 |
| 第二轮最终v4上游与下游 | 已完成 | 48新加12复用，最终60cell |
| 位置编码、近常量、目标泄漏检查 | 已完成 | 修复已落实到源码和产物 |
| 三模态融合情绪下游验证 | 已完成 | M5-F及M5-FT/R0回答表征能否被现有head利用 |
| 每模态单独线性probe | 本聊天未做 | 分别量化EEG/Wear/Video情绪信息 |
| 尺度、loss、EEG遮挡独立因子消融 | 未做 | 分离v4组合恢复的机制 |
| 更多独立SSL seeds与确认性paired统计 | 未做 | 当前只有固定上游seed和3个下游seed |
| Video共同有效mask对照 | 未做 | 分离9个native-mask差异 |
| raw shared pool扩展EEG预训练 | 已讨论，未执行 | 回答额外无标签量的作用，先做时间边界审计 |
| 新修复代码GitHub同步 | 本次未执行 | 当前仍为working-tree，需后续明确同步授权 |

本聊天提到的0906 EMA-bag、ST11/MT11结构矩阵、Wear-only FM/NormWear、EQL-CAF、校准和置换控制，是并列研究上下文。本文未把这些全部实验重新运行或宣布完成；当前交付范围是本聊天的MAE主线及其直接对照。

2026-10-10 后续输入改进已整理为[第一轮计划](mae_round1_input_improvement_plan_20261010.md)：分别验证既有2.0×face ROI VideoMAE与扩大过滤池EEG-MAE，其余配置固定，使用共同视频mask、匹配抽帧时间格和逐时间区间的raw隔离。该轮尚未实现或启动，完成状态与本手册封存结果分开。

若后续目标为选择生产或论文主基线，继续保留用户指定B0。若目标为研究MAE预训练净收益，最直接的新增证据是单模态探针和预先声明的匹配确认性比较。方案需遵守train/val选择、test报告、协议单位和多seed门槛，新增工作不因本handout自动启动。

## 14 证据来源与重算规则

证据优先级为当前原始artifact与源码、配置和检查输出，其次为已有文档，再次为历史聊天答复。发生冲突时以当前文件控制最终状态。历史pool数字、原cache进度及旧GPU快照在相应节中明确标为历史审计。

最终raw r计算：对每个protocol、route、seed，先对11个test raw r取等权算术均值；再对3个seed取算术均值和样本标准差。逐情绪行直接对该情绪的3个seed汇总。配对差先按相同seed计算M5-FT减R0，再汇总。centered r/RMSE同样先标签后seed。四位小数仅为展示，reviewed_evidence保留原精度。

### 14.1 各版本原始结果文件

| 版本 | 源文件 | SHA256 |
| --- | --- | --- |
| original | `/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/outputs/mae_mt11_window_20260928/formal/results.json` | `c9d5c18e4e732604d693d57fe0b3f3bfdc1044e12b57a955ea8ea47245eea530` |
| original | `/home/wangzw/outputs/mae_mt11_m2_20260930/formal/results.json` | `75e1f3167a2628ecc8a70547303389d0ccb0585ab33bff43ef0394017a9d0d27` |
| original | `/home/wangzw/outputs/mae_mt11_v1_20261007/formal/results.json` | `f5fa2b3dd8b71c548f81641c3e53efa22da05e507900f28dd573c1d0589a77b0` |
| original | `/home/wangzw/outputs/mae_remaining_20261007/frozen_formal/results.json` | `44216650dfeb80b6783a97b616e525832ca72661b51e66a814a01b6fc379613d` |
| original | `/home/wangzw/outputs/mae_remaining_20261007/partial_v3/formal/results.json` | `e0996d6e05f3a5ef19aa04ba9fd9be0c5cef381867c6adc8ee82fa30570fd3d7` |
| v2 | `/home/wangzw/mae_repair_20261008/outputs/downstream_v2/single/formal/results.json` | `7d912c4db244d44d082407a045c7f4aae3e1ece35547257a6bd70d2b5ba27853` |
| v2 | `/home/wangzw/mae_repair_20261008/outputs/downstream_v2/pairs/formal/results.json` | `c8807f8e9ca1d77aa4b50ce7c0821dac53193ac75a5947daed7a729c0e1bfd6c` |
| v2 | `/home/wangzw/mae_repair_20261008/outputs/downstream_v2/all/formal/results.json` | `4658040bd014dddbcb99cc82faaeb7125a5e598a06bc375b15b08fd647614b4f` |
| v2 | `/home/wangzw/mae_repair_20261008/outputs/downstream_v2/partial/formal/results.json` | `6986caafb27ac9d809a6faf964750b01214ad8483cfcdfc48979ea4727b8c09a` |
| v2 | `/home/wangzw/mae_repair_20261008/outputs/downstream_v2/R0_reference.json` | `47cd3de56d33d158bb4020fc1cc1f606f54286dd0f25e511767978284d9205b1` |
| v4 | `/home/wangzw/mae_norm_channel_20261009/outputs/downstream_train_channel/results.json` | `12b3bb3247bcdef11104376245114c3ecf8b80f1ad3a0322e1dbd5f7c0352091` |

第一轮v2的60cell通过四组results.json及R0_reference.json合成，并检查重复B0一致；原始版通过五个输出组合成。最终v4已经包含统一60行results.json。三个版本均直接读取服务器文件，按protocol、condition、seed去重，验证完整网格并重新聚合。

### 14.2 本次证据目录

| 文件 | 内容 |
| --- | --- |
| reviewed_evidence.json | 本次H20读取的3版本结果汇总、逐情绪raw r、seed值、10配置health及源hash |
| supplemental_checks.json | ncc Video、最终token顺序和mask、bag事件split交集等补充 |
| chat_history.json | 87回合用户请求及已交付答复，历史信息可追溯 |
| audit_evidence.py | 读取原始artifact并重算的只读Python程序 |
| reviewed_snapshot.json与report-app/src/data.json | 报告使用的审阅快照和来源合同 |
| report-app/dist | 同一内容的本地报告预览构建 |
| handout源稿与Word | 面向阅读和交接的交付文档 |

本手册没有复制原始EEG、视频、标签或大型prefix，没有修改外部数据源，没有发布至Sites，也没有新增监控或训练任务。

## 15 交接核对清单

继续工作前先确认SSH使用wangzw@124.174.8.252、端口10022，并选择对应版本的staging和Python。引用最终结果时保留完整reference route、DINO A1、上游监督来源、两个protocol、event聚合、3个下游seed及SD定义。

选择技术修复结论时，依据源码和health/tests说明问题已经修复；选择情绪预测结论时，依据第6节结果与匹配R0。新encoder、输入预处理或训练范围变化后，重新建立R0。视频仍在ncc计算并复制小型产物，源数据与大型cache保留。

最终矩阵已经结束。新增实验、Git同步、自动监控或更换科学协议均作为独立请求处理；已授权队列的程序错误恢复按既有规则直接修复、验证并接续。完成判断继续要求marker、唯一预期cell、有限指标、协议及来源一致，GPU占用或进程退出本身只提供运行线索。

证据状态：除特别标注为历史审计、推断或后续工作外，最终数值与当前产物状态已在本次封存中核验。
