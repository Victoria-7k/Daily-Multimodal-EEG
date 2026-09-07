# Wear 模态预训练模型选择方案（2026-08-20）

> 独立方案文档。背景口径见 [technical_route_20260814.md](technical_route_20260814.md)；本文只讨论 Wear 模态的预训练模型接入，不涉及 EEG/Video/Audio 路线。
>
> 结论先行：**可以接入，且存在多个已核验为开放获取的预训练模型**。推荐按「MOMENT-1（通用多变量时序，MIT，零格式改造）→ Pulse-PPG（PPG 领域专精，Zenodo 发布）→ PRIMUS（ACC/IMU 领域专精，BSD-3-Clause-Clear）」的顺序分阶段验证，全部以 `frozen / partial FT` 两档并入现有融合矩阵，与 EEG 侧 route 矩阵口径同构。

---

## 1. 背景与现状

### 1.1 任务与数据口径

- 任务：EEG 对齐后的多模态 10 秒窗口预测 `fatigue`，共 `28819` 个窗口。
- Wear 输入三路信号，提取时统一重采样到目标采样率（`src/daily_multimodal/embeddings/wear_real.py`）：

| 信号 | 目标采样率 | 10 秒窗口形状 |
| --- | ---: | --- |
| PPG | 64 Hz | `(640, 1)` |
| GSR | 32 Hz | `(320, 1)` |
| ACC（三轴） | 32 Hz | `(320, 3)` |

- 每个窗口的融合 token 固定为 `wear_emb (N,256)`，`modality_mask` 顺序 `[eeg, wear, video, audio]` 不变。

### 1.2 现有 Wear 路线（无预训练）

| 路线 | 方法 | 是否监督训练 |
| --- | --- | --- |
| `Wphysio`（`wear_physio_features_v2`） | PPG HR/HRV、GSR slope/SCR、ACC motion/stationary 等手工特征 | 否，固定随机投影（`_project_to_256`，seed 16016） |
| `Wdeep`（`wear_deep_sequence_v1`） | 320×5 对齐矩阵（PPG+GSR+ACC）经**固定随机** 1D conv（kernel 3/5/9/15）+ 池化统计 | 否，固定随机投影 |

关键现状：`Wdeep` 是固定随机卷积特征提取器，**不是预训练 encoder**；整个 Wear 分支的监督信号只发生在最终融合回归器（modality-token cross-attention）中。融合矩阵结果显示 Wear 是当前稳定贡献来源（如 `within_subject_day` 最佳 `B0_Wdeep_no_audio` raw r `0.4252`），因此提升 Wear 表征质量有直接收益空间。

### 1.3 接入目标

1. 用**开放获取**（许可证允许 + 权重公开可下载）的预训练模型替换 `Wdeep` 的随机特征提取器，提供更强的生理/运动表征。
2. 保持 256D token 契约、`modality_mask`、融合器（`scripts/window_fatigue/32_run_eegpt_centered_loss.py` 的 AttentionRegressor）完全不变。
3. 与 EEG 侧同构：每个预训练 route 提供 `frozen` 与 `partial FT` 两档，融入现有 fusion matrix（B0/A1/A2 × wear-route × full/no_audio），按相同协议（`cross_day` / `within_subject_day` 主，`cross_subject` 诊断）评估。

---

## 2. 接入约束（为什么可行）

| 约束 | 现状 | 影响 |
| --- | --- | --- |
| 256D token 契约 | 融合器只吃 `(N,256)` 预计算 token（`32_run_eegpt_centered_loss.py` 的 `BRANCHES` 从 npz 懒加载） | 预训练 encoder 输出任何维度，经 256D 投影后即可接入，融合器零改动 |
| 输入数据可得性 | 每窗口 `sequence.npz` 缓存已含重采样后的 PPG/GSR/ACC 序列 | 预训练 encoder 可**离线逐窗口推理**，无需重读原始 CSV |
| 监督边界 | EEG 侧已有成熟口径：frozen = encoder 冻结 + 可学习投影头（fatigue 监督，只训 train/val）；partial FT = 解冻后部 block + 投影头，低 LR | Wear 侧直接复用同一机制（参考 `34_run_eeg_encoder_matrix.py` 的 profile 输出 `{protocol}/{profile}/seed_{seed}.npz`） |
| 协议纪律 | train = pretrain+finetune，val 早停选模型，test 冻结评估 | 不变 |

---

## 3. 候选模型调研（开放获取核验，2026-08-20）

> 核验方式：官方仓库 / 论文 / Zenodo / HuggingFace 记录交叉确认。标注「待核」的条目是接入前必须补齐的验证项。

### 3.1 候选总表

| 候选 | 面向信号 | 开放获取状态 | 许可证 | 输入规格 | 与当前管线适配 |
| --- | --- | --- | --- | --- | --- |
| **MOMENT-1**（首选） | 通用多变量时间序列 | ✅ HuggingFace `AutonLab/MOMENT-1-small` / `MOMENT-1-large`，权重直接下载 | ✅ **MIT** | 多变量，≤512 步 | **极佳**：现有 320×5 对齐矩阵直接可用，覆盖 PPG+GSR+ACC 三路，零格式改造 |
| **Pulse-PPG** | 接触式 PPG（实验室+野外） | ✅ Zenodo 正式发布 `10.5281/zenodo.17270931`（v1.0.0，2025-09），arXiv 2502.01108 | ⚠️ 许可证条目待核（发布物本身可下载） | 待核（需按模型卡片对齐窗口长度/采样率/归一化） | 好：PPG 通道 10s@64Hz 已缓存；野外训练数据与真实穿戴场景更接近 |
| **PRIMUS** | IMU（acc+gyro，6 通道） | ✅ 官方 GitHub `Nokia-Bell-Labs/pretrained-imu-encoders`（ICASSP'25，arXiv 2411.15127） | ✅ **BSD-3-Clause-Clear** | 待核（官方 README；预训练权重随仓库发布） | 中：ACC 需重采样到模型原生频率；**缺 gyro 通道**需零填充/复制适配 |
| **AnyPPG** | PPG（ECG 引导，10 万+ 小时） | ✅ arXiv 2511.01747；2026 综述（arXiv 2605.00973）明确标注 "AnyPPG (Open-Source Weights)" | ⚠️ 托管位置与许可证待核 | 待核 | 好：PPG 单通道路线备选 |
| **NormWear** | 多变量可穿戴生理信号 | ✅ 官方 GitHub `Mobile-Sensing-and-UbiComp-Laboratory/NormWear`（arXiv 2412.09758，ICML'25） | ⚠️ 许可证与权重形式待核 | 待核（输入模态清单是否含 GSR/ACC） | 待定：若同时覆盖 PPG/ACC 则接近「一体式」方案 |
| IMU2CLIP | IMU（acc+gyro） | ⚠️ 代码开源（`facebookresearch/imu2clip`，EMNLP'23 Findings）；权重仅社区镜像（非官方，需甄别） | ⚠️ Meta 系仓库常见 **CC-BY-NC**（仅限研究、不可商用） | 50 Hz、6 通道、2 秒窗口（100 样本） | 中：同样缺 gyro；若许可证为 NC 仅限研究用途 |
| SensorFM（Google Research） | 可穿戴多模态（万亿分钟预训练） | ❌ 仅有研究博客，**未开源** | — | — | 排除 |

> GSR/EDA 目前**没有**公开预训练模型：接受现状，GSR 继续走 `Wphysio` 手工特征，或作为 MOMENT 的一个通道。

### 3.2 分模型说明

- **MOMENT-1**：通用时间序列基础模型（MOMENT 论文，CC-BY-4.0 论文 + MIT 代码/权重），在数百万条跨领域时间序列上做 masked 预训练。最大支持 512 步、多通道输入。我们的 `_aligned_sequence_matrix`（320 步 × 5 通道，逐通道 z-norm）**直接满足其输入约束**，是风险最低、成本最低的首选。small 版本参数量小，适合先冒烟。
- **Pulse-PPG**：目前少见的「野外（field）训练」PPG 基础模型，针对可穿戴设备场景，比实验室干净 PPG 更接近我们的真实采集条件。Zenodo 发布物可直接下载。
- **PRIMUS**：Nokia Bell Labs 的 IMU 预训练 encoder，通过 IMU+视频+音频多模态自监督预训练。组织级许可证为 BSD-3-Clause-Clear（宽松许可）。注意其预训练输入为 acc+gyro 六通道，我们只有三轴 ACC，需要通道适配。
- **AnyPPG / NormWear**：更新、更大的候选（AnyPPG 为 10 万+ 小时 ECG 引导 PPG 预训练；NormWear 面向多变量可穿戴生理信号），但权重托管、许可证与输入规格尚未逐条核验，列为阶段 2 备选。
- **排除项**：SensorFM（未开源）；rPPG-MAE/PhySU-Net（面向**视频**远距离 PPG，不是接触式 PPG，且主要在视频域）。

---

## 4. 推荐路线（分阶段方案）

### 4.1 阶段 0：可行性冒烟（1–2 天）

- 接入 MOMENT-1-small frozen：读 `sequence.npz` → 320×5 矩阵 → MOMENT encoder → pooled 表征 → 256D 投影 → token npz。
- 先跑 3 个协议 × 少量 route（如 `B0_Wmoment_frozen_full/no_audio`），确认：token 形状/顺序与 canonical index 对齐、mask 覆盖、训练/评估链路跑通、单窗口推理耗时与显存。
- 对照组保留 `Wphysio`/`Wdeep` 同配置结果。

### 4.2 阶段 1：主选路线（MOMENT-1）

新增两组 wear route，与 EEG 命名风格一致：

| 新 route | 方法 |
| --- | --- |
| `wear_moment_frozen_v1` | MOMENT-1 encoder 冻结 + **可学习 256D 投影头**（fatigue 监督，train/val） |
| `wear_moment_partial_ft_v1` | 解冻 MOMENT 后部 block + 投影头，低 LR（如 `1e-5`），train/val 监督 |

- 先 small 后 large；若 small frozen 已优于 `Wdeep` 同配置，说明预训练表征本身有效。
- 融入现有 fusion matrix：`B0/A1/A2 × {Wmoment_frozen, Wmoment_partial_ft} × full/no_audio`，三协议。

### 4.3 阶段 2：领域专精路线（视阶段 1 结论决定）

| 新 route | 方法 | 适配动作 |
| --- | --- | --- |
| `wear_ppg_pulse_frozen_v1` / `wear_ppg_pulse_ft_v1` | Pulse-PPG 编码 PPG 通道 | 对齐模型原生窗口（10s 缓存序列中心裁剪/补齐、按模型卡片归一化） |
| `wear_imu_primus_frozen_v1` / `wear_imu_primus_ft_v1` | PRIMUS 编码 ACC | ACC 32→50 Hz 重采样；gyro 通道零填充（frozen 档对通道错配更敏感，优先 partial FT 或复制 ACC 适配） |
| GSR 处理 | 无预训练模型 | 保留 `Wphysio` 的 GSR 特征，在 256D 投影**之前**与预训练表征拼接；或把 GSR 作为 MOMENT 额外通道 |

> 注意：256D token 契约是「每模态一个 token」。PPG 预训练 + ACC 预训练 + GSR 特征需要**先拼接再投影**到 256D，而不是输出多个 wear token。若拼接后维度过高，可先各自降维再拼接。

### 4.4 阶段 3：结论与纳入主线

- 对通过阶段 1/2 门槛的 route 做 3-seed paired 验证（与同配置 `Wphysio`/`Wdeep` 配对），按 `cross_day` / `within_subject_day` 的 RMSE、raw r、centered r 汇总。
- 通过者写入 `technical_route_20260814.md` 的 Wear 表格，替换/并列现有 `Wdeep`。

---

## 5. 工程落点

| 环节 | 位置 | 改动 |
| --- | --- | --- |
| 预训练 encoder 推理 | `src/daily_multimodal/embeddings/wear_real.py` 新增 encoder profile（或新模块 `wear_pretrained.py`） | 消费 `sequence.npz`（PPG/GSR/ACC 重采样序列已缓存），产出预训练表征 |
| 离线 token 生成 | 新脚本 `16_extract_wear_pretrained_tokens.py`（或扩展 `15_extract_wear_embeddings.py`） | 逐窗口推理 → `wear_emb (N,256)` npz，对齐 `sample_id` 顺序 |
| 投影头/微调训练 | 仿 `daily_multimodal.training.eeg_encoder_matrix` 的 profile 机制 | 输出 `{embeddings_root}/wear_tokens/{protocol}/{profile}/seed_{seed}.npz`（对应 EEG 的 `eeg_encoder_256d_tokens`） |
| 融合矩阵挂载 | `scripts/window_fatigue/32_run_eegpt_centered_loss.py` | `BRANCHES` 增加 `wear_moment_frozen_v1` 等分支（指向新 npz，`emb_key="wear_emb"`、`mask_key="wear_mask"`、`modality_index=1`）；`EXPERIMENT_BRANCHES` 增加 `B0_Wmoment_frozen_full` / `_no_audio` 等 route 元组；`--experiment-set video_only` 自动纳入 |
| 结果报告 | 现有 matrix 汇总 + 文档 | 同协议、同指标口径；明确记录监督边界 |

依赖安装：MOMENT 通过 `pip install momentfm`（或直接 HF `from_pretrained`）；Pulse-PPG/PRIMUS 按官方仓库 README。服务器环境为 `runtime/envs/eegpt-gpu-min`（参考 change-log 的既有验证方式）。

---

## 6. 评估与验收标准

- 协议：`cross_day`、`within_subject_day` 为主；`cross_subject` 保留为诊断。
- 指标：RMSE、MAE、raw r、centered r（`evaluate_regression_with_centered` 口径），test split 冻结评估。
- 对照：与**同 EEG route、同 video route、同 full/no_audio 配置**下的 `Wphysio`/`Wdeep` 结果配对比较。
- 验收门槛（建议）：
  1. 阶段 1 至少一个主协议上，`wear_moment_frozen_v1` 的 raw r 或 RMSE 优于同配置 `Wdeep`（证明预训练表征本身 > 随机卷积基线）；
  2. `wear_moment_partial_ft_v1` 相对 frozen 不显著倒退（证明格式对齐/监督训练稳定）；
  3. 阶段 2 的领域专精 route 相对阶段 1 有稳定增益才纳入主线，否则以 MOMENT 为准。
- 记录要求：模型版本与权重哈希、重采样/填充参数、每个 route 的监督边界（frozen 投影头与 partial FT 都是 fatigue-supervised representation，不能表述为 label-free embedding，与 EEG 侧口径一致）。

---

## 7. 风险与缓解

| 风险 | 说明 | 缓解 |
| --- | --- | --- |
| 格式错配 | 采样率（32/64 Hz vs 模型原生频率）、段长（10s vs 模型预训练窗口）、通道（无 gyro） | 重采样 + 中心裁剪/补齐 + 分段池化；partial FT 可吸收部分错配；frozen 档若差则优先做格式对齐 |
| 领域差距 | 预训练数据是公开可穿戴/野外数据，与我们的设备、佩戴位置、采集协议不同；**没有疲劳专用预训练模型** | 疲劳映射仍由融合回归器学习；收益是经验性的，必须按现有协议做 ablation，不能预设增益 |
| 许可证 | 各模型许可证差异大（MIT / BSD-3-Clause-Clear / Zenodo 条目 / 可能的 CC-BY-NC） | 接入前逐条核验并记录；NC 类仅限研究用途需在文档标注；**首选 MIT/BSD 类** |
| 监督边界泄漏 | partial FT 若误用 test 信息会污染结果 | 严格只训 train/val，val 早停，test 冻结；与 EEG partial FT 同一纪律 |
| 计算成本 | 28,819 窗口 × encoder 推理；large 模型显存/耗时 | 先 small；离线逐窗口推理可并行分块；token 落盘后融合训练成本不变 |
| 复现性 | 权重下载源可能变动 | 固定模型版本 + 记录权重哈希 + 本地缓存 checkpoints（`outputs/checkpoints/`） |

---

## 8. 决策建议

1. **第一步就上 `wear_moment_frozen_v1`（MOMENT-1-small）**：唯一同时满足「开放获取（MIT + HF 直下）」「零格式改造（320×5 直接可喂）」「覆盖全部三路信号（含无预训练模型的 GSR）」三个条件的候选，成本最低，先验证预训练表征相对 `Wdeep` 随机卷积是否带来增益。
2. 若 frozen 有效，再跑 partial FT 与 large 版本；同时核验 Pulse-PPG（PPG 专精）与 PRIMUS（ACC 专精），决定是否需要领域专精路线。
3. 若 frozen 无效（增益 < 0），先检查格式对齐与归一化口径，再降级为「预训练表征 + 更强可学习投影头」或回到 `Wphysio`/`Wdeep` 基线——**不预设结论，以同一协议矩阵的 paired 结果为准**。

---

## 9. 参考资料

- 现状：`technical_route_20260814.md`；`src/daily_multimodal/embeddings/wear_real.py`；`scripts/window_fatigue/32_run_eegpt_centered_loss.py`；`scripts/embeddings/34_run_eeg_encoder_matrix.py`
- MOMENT-1：<https://huggingface.co/AutonLab/MOMENT-1-large>（MIT），<https://arxiv.org/abs/2402.03885>
- Pulse-PPG：<https://zenodo.org/records/17270931>，<https://arxiv.org/abs/2502.01108>
- PRIMUS：<https://github.com/Nokia-Bell-Labs/pretrained-imu-encoders>，<https://arxiv.org/abs/2411.15127>
- AnyPPG：<https://arxiv.org/abs/2511.01747>
- NormWear：<https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear>，<https://arxiv.org/abs/2412.09758>
- IMU2CLIP：<https://github.com/facebookresearch/imu2clip>，<https://aclanthology.org/2023.findings-emnlp.883/>
- SensorFM（未开源，仅参考）：<https://www.marktechpost.com/2026/07/10/google-research-introduces-sensorfm-a-wearable-health-foundation-model-pretrained-on-one-trillion-minutes-of-sensor-data/>
