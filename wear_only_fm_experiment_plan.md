# Wear-only 表征实验矩阵与模型获取计划

**目标**：在完全不使用 EEG、Video、Audio 的条件下，公平比较四类 wearable 表征对 10 秒窗口 `fatigue` 回归的贡献：`Wphysio`、`Wdeep`、已完成主线验证的强 baseline `Wmoment_frozen`、以及新的三支路专属模型 `W3FM_frozen`。

**核心问题**：在 `Wmoment_frozen` 已经作为通用时序预训练强 baseline 的前提下，模态专属预训练（PPG、ACC、GSR 分别编码）再做 wear 内部融合，能否在 wear-only 条件下进一步稳定提升疲劳预测？

---

## 1. 固定数据与评估边界

### 1.1 数据口径

- 样本单位：与 EEG 对齐的 **10 秒窗口**。
- 目标：`fatigue` 回归标签。
- 候选样本：`wear_mask=True` 的窗口；当前预计为 `24,127 / 28,819`。该数字只作为候选全集，不直接等同于主矩阵样本数。
- Wear 原始通道：PPG 1 通道、GSR/EDA 1 通道、ACC 三轴。
- Phase 0 必须从原始窗口和 frozen embedding smoke test 同时生成 `ppg_available_mask`、`gsr_available_mask`、`acc_available_mask`、`wear_mask`、`wear_complete_mask`，并报告每个 mask 及交集的窗口数。
- 主矩阵默认使用同一份 `wear_complete_mask = wear_mask & ppg_available_mask & gsr_available_mask & acc_available_mask & embedding_valid_mask`。任何一路 PPG、GSR、ACC 或 frozen embedding 无效时，该窗口从四条 route 中同时移除。
- `W3FM` 的 `submodality_mask` 只用于 smoke、诊断和未来 partial-submodality 实验；若主矩阵允许缺失子模态参与，就必须作为单独矩阵报告，不能与 complete-subset 主结果混为一个结论。

### 1.2 划分协议

| 角色 | protocol | 目的 |
|---|---|---|
| 主协议 1 | `cross_day` | 评估跨日期的疲劳预测能力 |
| 主协议 2 | `within_subject_day` | 评估同一被试内部、跨日期的疲劳变化预测 |
| 诊断协议 | `cross_subject` | 评估迁移到未见被试的能力 |

每一个 protocol 固定复用既有的 `pretrain / finetune / val / test` split。``val`` 只用于早停和选择 checkpoint，任何 test 指标均不得用于选模型、选 epoch 或调超参数。

### 1.3 监督边界

本实验把“冻结 encoder 缓存”和“疲劳监督训练”分开记录：

- **label-free 缓存层**：PaPaGei-S、HARNet10、NormWear 的原始 frozen embeddings 可以在候选 `wear_mask` 样本上一次性缓存；该步骤不读取 `fatigue` 标签。MOMENT 的原始 frozen embedding 或既有 `Wmoment_frozen` 中间缓存可复用，不把 MOMENT 当作未知新路线重新立项。
- **fatigue-supervised 表征层**：`Wmoment_frozen` 的 256D 投影头、`W3FM_frozen` 的三路投影层、gate 和最终回归头都使用对应 protocol 的 `pretrain + finetune` 训练、`val` 早停。名称中的 `frozen` 只表示外部 encoder 冻结，不表示最终 `wear_emb` 是 label-free embedding。
- **固定基线层**：`Wphysio` 与 `Wdeep` 复用现有固定 256D `wear_emb`；其中 `Wdeep` 是固定随机 1D Conv / TCN-like 特征器和固定随机投影，不是监督训练的 wearable encoder。
- 对原始数据进行的全局归一化、缺失值填补统计量、robust scaling 和任何依赖数据集统计量的预处理，只能在训练集拟合，再应用于 `val/test`。encoder 官方要求的确定性滤波或重采样可以缓存，但若其中有可拟合统计量，必须按 split 单独拟合并写入 manifest。
- 所有输出都必须显式记录 `train_supervision`、`encoder_profile`、`encoder_version`、`protocol`、`seed`、mask 口径和 checkpoint SHA256，避免把 frozen encoder、监督投影头和最终回归头混成一个不透明 route。

---

## 2. 主实验矩阵

### 2.1 参照与新方案

| route ID | 表征方法 | encoder 策略 | 统一输出 |
|---|---|---|---|
| `Wphysio` | 现有 HR/HRV、GSR slope/SCR、运动量、静息状态等可解释特征 | 固定手工特征 + 固定 256D 投影；训练统一 wear-only 回归头 | `wear_emb (256D)` |
| `Wdeep` | 现有 PPG/GSR/ACC 序列经固定随机 1D Conv / TCN-like 特征器 | 固定随机特征器 + 固定 256D 投影；训练统一 wear-only 回归头 | `wear_emb (256D)` |
| `Wmoment_frozen` | 已完成 3-seed 与方案 C 验证的 MOMENT-1-small 强 baseline | MOMENT encoder 冻结；256D 投影头为 fatigue-supervised；本实验中作为已知强 baseline 复用或按同口径重训 wear-only head | `wear_emb (256D)` |
| `W3FM_frozen` | PaPaGei-S（PPG）+ HARNet10（ACC）+ NormWear（GSR）+ gated fusion | 三个外部 encoder 冻结；三路投影、gate 与回归头为 fatigue-supervised | `wear_emb (256D)` |

### 2.2 运行数与配对 seeds

| 维度 | 设置 |
|---|---|
| protocol | `cross_day`、`within_subject_day`、`cross_subject` |
| route | 上表 4 条路线 |
| 下游训练 seed | `240729`、`240730`、`240731` |
| 总运行数 | `3 × 4 × 3 = 36` |

同一个 `protocol × seed` 下，四条 route 必须复用相同 split、batch 顺序、标签归一化参数和初始化 seed。`Wdeep` 的随机特征器另设并永久固定 `feature_seed=240800`，避免随机卷积本身成为重复实验的噪声来源。

---

## 3. 统一训练与比较配置

### 3.1 下游回归头

所有 route 的最终输入都只能是一个 256D `wear_emb`，统一接入：

```text
LayerNorm(256)
→ Linear(256, 128)
→ GELU
→ Dropout(0.1)
→ Linear(128, 1)
```

| 参数 | 固定值 |
|---|---:|
| optimizer | AdamW |
| learning rate | `1e-3` |
| weight decay | `1e-4` |
| batch size | `256` |
| 最大 epoch | `80` |
| early-stopping patience | `15` |
| checkpoint 选择指标 | validation RMSE 最小 |

每条路线必须记录：总参数量、可训练参数量、预计算 embedding 后的每 epoch 耗时、单窗口 embedding 推理耗时。`W3FM_frozen` 的可训练参数更多是方法本身的一部分；报告中透明列出即可。

### 3.2 报告指标与胜出规则

| 层级 | 指标 | 解释 |
|---|---|---|
| 主结果 | `cross_day` raw Pearson `r`、RMSE | 跨天泛化和绝对误差 |
| 主结果补充 | `within_subject_day` centered Pearson `r` | 同一被试内的疲劳相对变化 |
| 诊断 | `cross_subject` raw/centered `r`、RMSE | 未见被试泛化 |
| 稳定性 | 三 seed 的 mean ± std | 判断结论是否依赖偶然初始化 |
| 显著性 | seed 内 subject-day block paired bootstrap + seed 级 paired delta | 比较新方案与最强 baseline |

`W3FM_frozen` 达到以下条件时，视为第一轮胜出：

1. `cross_day raw r` 三 seed 平均值高于最强 baseline，且至少两个 seed 的 paired delta 为正；
2. `cross_day RMSE` 三 seed 平均值不高于该 baseline，且至少两个 seed 的 paired delta 不劣化；
3. `within_subject_day centered r` 持平或提升，且无明显 RMSE 回退；
4. 每个 seed 内对 `W3FM_frozen` 与最强 baseline 做 subject-day block paired bootstrap；最终报告三组 seed 的 bootstrap 区间、seed 级 mean ± std 和方向一致性；
5. 最强 baseline 必须从 `Wphysio`、`Wdeep`、`Wmoment_frozen` 中按同 protocol、同 seed、同 mask 口径选择，不能把 `Wmoment_frozen` 当作未知或弱 baseline。

禁止用某一个 route 的单次最优 test run 宣称模型胜出；也禁止只对 seed-averaged test prediction 做一次 bootstrap 后给出结论。seed-averaged prediction 可以作为补充可视化，不能作为唯一显著性证据。

---

## 4. 新方案：W3FM_frozen

### 4.1 模型结构

```text
PPG 10 s → PaPaGei-S frozen → 512D ┐
ACC 10 s → HARNet10 frozen  → d_acc ├→ 每路投影为 256D → masked gated pooling → wear_emb 256D
GSR 10 s → NormWear frozen  → 768D ┘
```

推荐实现：

```python
z_ppg = proj_ppg(e_ppg)   # [B, 256]
z_acc = proj_acc(e_acc)   # [B, 256]
z_eda = proj_eda(e_eda)   # [B, 256]

tokens = torch.stack([z_ppg, z_acc, z_eda], dim=1)  # [B, 3, 256]
logits = gate(tokens).squeeze(-1)                    # [B, 3]
logits = logits.masked_fill(~submodality_mask, -1e9)
weights = torch.softmax(logits, dim=1)
wear_emb = (weights.unsqueeze(-1) * tokens).sum(dim=1)
```

投影层统一采用 `LayerNorm(d_in) → Linear(d_in,256) → GELU → Dropout(0.1)`。`gate` 使用共享的 `Linear(256,1)`；它使不同样本能依据当前窗口自适应强调 PPG、运动或皮电信息。

`W3FM_frozen` 的命名只表示 PaPaGei-S、HARNet10、NormWear 三个外部 encoder 冻结。`proj_ppg/proj_acc/proj_eda`、`gate` 和 wear-only 回归头均属于 fatigue-supervised 部分，必须按 protocol 和 seed 单独训练并记录。

主实验只保留一个 `W3FM_frozen`，避免把“encoder 是否专属”“是否冻结”“内部融合是否有效”同时混在第一轮矩阵里。

### 4.2 模态预处理

| 模态 | 输入与目标格式 | 处理原则 |
|---|---|---|
| PPG | 单通道、10 s；重采样至 `125 Hz × 1250` 点 | 带通/质量控制后窗口 z-score；建议额外保存 SQI 供诊断 |
| ACC | 三轴、10 s；重采样至 `30 Hz × 300` 点，shape `[B,3,300]` | 统一单位和轴顺序；使用 train-set 按轴标准化；禁止每窗口 z-score |
| GSR | 单通道、完整 10 s，保留真实采样率 | 处理无效值、`log1p` 和 train-set robust scaling；禁止每窗口 z-score |

GSR 的绝对 tonic level、慢变趋势和 SCR 振幅可能与疲劳/唤醒相关。每窗口单独标准化会删除这些信息。ACC 每窗口 z-score 同样会削弱姿态、静息与活动强度差异。

---

## 5. 三个预训练模型的获取与冻结 embedding 缓存

建议将模型安装与 embedding 提取分开。PaPaGei、HARNet、NormWear 的依赖版本不同，分别使用独立环境提取原始 embedding，最后只把 `.npy` 特征矩阵交给主训练环境。

### 5.1 统一目录与可复现性清单

```text
wear_fm/
├── third_party/
│   ├── papagei-foundation-model/
│   ├── ssl-wearables/
│   └── NormWear/
├── weights/
│   ├── papagei_s.pt
│   ├── harnet10_ukb100k.pt
│   └── normwear_pretrain_ckpt.pth
├── staged_inputs/
│   ├── wear_window_index.csv
│   ├── ppg_10s.npz
│   ├── acc_10s.npz
│   └── gsr_10s.npz
├── embeddings/
│   ├── ppg_papagei_s_512d.npy
│   ├── acc_harnet10.npy
│   ├── gsr_normwear_768d.npy
│   └── embedding_manifest.json
└── logs/model_checksums.txt
```

`wear_window_index.csv` 是唯一的行号真相来源。三个 `.npy` 的第 `i` 行必须对应其中第 `i` 个 window；缓存完成后必须随机抽查至少 20 个 `subject/session/window_start` 与原始数据一致。每个 embedding 文件必须同时写出对应的有效性 mask，并在 Phase 0 汇总为 `wear_complete_mask`。

每次下载完成，记录以下信息到 `embedding_manifest.json`：模型名、来源 URL、repo commit/tag、权重文件名、SHA256、输入采样率、输入长度、预处理版本、输出 shape、有效窗口数、提取脚本 commit、窗口索引文件 SHA256。manifest 只描述 label-free frozen embedding 缓存；监督投影、gate、回归头的训练配置写入每个 run 的配置快照。

### 5.2 PPG：PaPaGei-S

PaPaGei 是专为 PPG 自监督预训练的开源模型；官方提供 **PaPaGei-S** 权重和 embedding 提取代码。主线选择 `papagei_s.pt`，其 encoder 输出为 512D。参考：[官方代码仓库](https://github.com/nokia-bell-labs/papagei-foundation-model)、[官方权重 Zenodo record](https://zenodo.org/records/13983110)。

```bash
mkdir -p wear_fm/third_party wear_fm/weights wear_fm/logs
cd wear_fm/third_party
git clone https://github.com/nokia-bell-labs/papagei-foundation-model.git

conda create -n wear-papagei python=3.10 -y
conda activate wear-papagei
cd papagei-foundation-model
pip install -r requirements.txt
pip install pyPPG==1.0.41

curl -L 'https://zenodo.org/records/13983110/files/papagei_s.pt?download=1' \
  -o ../../weights/papagei_s.pt
sha256sum ../../weights/papagei_s.pt | tee -a ../../logs/model_checksums.txt
```

提取要求：

1. 对每个 PPG 10 秒片段进行模型仓库推荐的滤波与质量检查；
2. 重采样至 125 Hz，得到 1250 点；
3. 使用仓库的 `feature_extraction.py` / `compute_signal_embeddings` 流程提取 512D；
4. 写出 `ppg_papagei_s_512d.npy`，shape 应为 `[N_wear, 512]`；
5. 先用 8 个窗口 smoke test，检查无 NaN/Inf、输出维度恒为 512，再批量缓存。

### 5.3 ACC：HARNet10

HARNet10 是使用大规模腕部加速度数据自监督训练的模型。官方 API 的输入定义正好是 **30 Hz、10 秒、三轴 300 点**，并公开了预训练 `feature_extractor` 与权重下载入口。参考：[官方仓库](https://github.com/OxWearables/ssl-wearables)。

```bash
cd wear_fm/third_party
git clone https://github.com/OxWearables/ssl-wearables.git

conda create -n wear-harnet python=3.10 -y
conda activate wear-harnet
cd ssl-wearables
pip install -r req.txt

python - <<'PY'
import torch

# 首次调用会按官方 torch.hub 配置取得预训练 harnet10 权重。
model = torch.hub.load(
    'OxWearables/ssl-wearables',
    'harnet10',
    class_num=5,
    pretrained=True,
    trust_repo=True,
)
torch.save(model.state_dict(), '../../weights/harnet10_ukb100k.pt')
print('saved:', '../../weights/harnet10_ukb100k.pt')
PY
sha256sum ../../weights/harnet10_ukb100k.pt | tee -a ../../logs/model_checksums.txt
```

提取要求：

1. 将原始 ACC 统一为 `[B, 3, 300]`；确认轴顺序与单位，记录是否做了方向校正或去重力；
2. 使用 `model.feature_extractor`，不使用随机初始化的 `classifier`；
3. 先打印一次输出 shape，记录 `d_acc`，再按 batch 提取并保存为 `acc_harnet10.npy`；
4. 在 manifest 中记录 Torch 和 PyTorch 版本，因为 `torch.hub` 可能随默认 revision 改变；如需严格复现，应同时记录本地 hub 缓存对应的 repo commit。

### 5.4 GSR / EDA：NormWear

NormWear 的预训练明确覆盖 GSR、PPG 和 IMU，官方 Release 提供主 backbone 的权重 `normwear_pretrain_ckpt.pth`。对于本实验，只将 GSR 作为单通道输入，不把 PPG 或 ACC 送进 NormWear，以保持三支路独立。参考：[官方仓库](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear)、[v1.0.0-alpha checkpoint release](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear/releases/tag/v1.0.0-alpha)。

```bash
cd wear_fm/third_party
git clone https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear.git
cd NormWear
git checkout v1.0.0-alpha

conda create -n wear-normwear python=3.10 -y
conda activate wear-normwear
pip install -r dependencies.txt

curl -L \
  'https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear/releases/download/v1.0.0-alpha/normwear_pretrain_ckpt.pth' \
  -o ../../weights/normwear_pretrain_ckpt.pth
sha256sum ../../weights/normwear_pretrain_ckpt.pth | tee -a ../../logs/model_checksums.txt
```

提取要求：

1. 输入为 `[B, 1, T]`，其中 `T = 10 × GSR 原始采样率`；`sampling_rate` 传真实采样率；
2. 调用 `NormWearModel.get_embedding(...)`；它返回 `[B, 1, P, 768]`；
3. 固定使用 patch mean pooling：`out.mean(dim=2).squeeze(1)`，得到 `[B,768]`；
4. 写出 `gsr_normwear_768d.npy`，并先完成 8 个窗口 smoke test；
5. 所有 10 秒窗口均以相同规则提取；不得在测试集上改变滤波、截断或 pooling 规则。

### 5.5 权重和输出的最小 smoke test

在正式提取前，每个模型都必须完成：

```text
加载 checkpoint
→ 输入 8 个真实的 10 秒窗口
→ eval() + torch.no_grad()
→ 断言 batch 维正确、无 NaN/Inf、embedding 方差非零
→ 保存 model/version/input/output 的日志
```

若 PaPaGei、HARNet 或 NormWear 在目标服务器无法正常安装，不得悄悄替换成另一个模型。先记录错误、环境版本和失败命令，再决定是否采用容器隔离或备用模型。

---

## 6. 运行顺序

### Phase 0：数据审计

1. 生成共享 `wear_window_index.csv`、`wear_mask.npy`、`ppg_available_mask.npy`、`gsr_available_mask.npy`、`acc_available_mask.npy`、`embedding_valid_mask.npy` 与 `wear_complete_mask.npy`。
2. 输出 PPG/GSR/ACC 的原始采样率、缺失率、常数片段率、每窗口有效长度分布，以及 `wear_mask` 到 `wear_complete_mask` 的覆盖率损失。
3. 确认四条 route 的训练/验证/测试窗口数在同一 `wear_complete_mask` 下完全一致。
4. 如果 `wear_complete_mask` 相比 `wear_mask` 覆盖率明显下降，主矩阵仍按 complete subset 执行，但必须额外报告覆盖率损失；是否再跑 partial-submodality 诊断矩阵另行决定。

**产物**：`wear_data_audit.md`、窗口索引、完整性 mask。

### Phase 1：外部 encoder 获取与 embedding 缓存

1. 下载三个官方 checkpoint，生成 checksum manifest。
2. 分别完成 PPG、ACC、GSR 的 smoke test。
3. 批量提取 label-free frozen embeddings，并保存 `row_id` 对齐验证结果与每路 embedding 有效性 mask。
4. 复用既有 `Wmoment_frozen` 证据作为强 baseline 背景；若为了 wear-only 矩阵重训其 projection/head，必须沿用同一 split、mask、seed 和监督边界。

**产物**：三份 embedding `.npy`、`embedding_manifest.json`、提取日志。

**2026-08-24 执行记录**：Phase 0-1 已在服务器 `/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned` 完成。空间检查显示 `/vePFS-0x0d` 可用约 `265G`，`/tmp` 可用约 `1.9G`；权重、源码和 embedding 均放在 `/vePFS-0x0d/.../outputs/wear_fm/`，未使用 `/tmp` 承载大文件。`wear_complete_mask` 为 `24127/28819`，与 PPG/ACC/GSR 三路 embedding valid mask 完全一致。已缓存 `ppg_papagei_s_512d.npy (28819,512)`、`acc_harnet10.npy (28819,1024)`、`gsr_normwear_768d.npy (28819,768)`，三路均 finite；本地摘要同步到 `outputs/server_sync/wear_fm_phase1_20260824/`，远端完整 manifest 为 `outputs/wear_fm/embeddings/embedding_manifest.json`。

### Phase 2：36-run 主矩阵

执行顺序：先完整跑 `cross_day`，再 `within_subject_day`，最后 `cross_subject`。

```text
for protocol in [cross_day, within_subject_day, cross_subject]:
  for seed in [240729, 240730, 240731]:
    for route in [Wphysio, Wdeep, Wmoment_frozen, W3FM_frozen]:
      train → val early stop → one-time test → save predictions and metrics
```

每个 run 需要保存：best checkpoint、`val_history.csv`、`test_predictions.parquet`（含 `row_id`、真实值、预测值）、`metrics.json`、配置快照。

**2026-08-25 执行记录**：Phase 2 已在服务器完成。先用既有 MOMENT frozen embedding cache 补齐 `Wmoment_frozen` 的 `240729/240730/240731` 三个下游 seed token，再运行 `cross_day`、`within_subject_day`、`cross_subject` × `Wphysio/Wdeep/Wmoment_frozen/W3FM_frozen` × 3 seeds，共 `36` 个 wear-only runs。每个 run 已写出 `metrics.json`、`best_checkpoint.pt`、`val_history.csv`、`predictions.npz` 和 `test_predictions.csv`；当前远端环境缺 `pyarrow/fastparquet`，`test_predictions.parquet` 未生成。服务器 `/vePFS-0x0d` 可用约 `265G`，`/tmp` 可用约 `705M`，运行时 `TMPDIR` 指向项目内 `outputs/tmp`。完整远端产物为 `outputs/wear_fm/phase2/`，本地同步副本为 `outputs/server_sync/wear_fm_phase2_20260825/`。Phase 2 均值表显示 `W3FM_frozen` 未在任一协议超过当 seed 最强 baseline；正式 stop/advance 判定仍留给 Phase 3 的 subject-day block paired bootstrap。

### Phase 3：汇总与判定

1. 按 `protocol × route` 汇总三 seed 的 mean ± std；
2. 在 `Wphysio`、`Wdeep`、`Wmoment_frozen` 中找到每个主协议、每个 seed 的最强同口径 baseline，并另报三 seed 平均意义下的最强 baseline；
3. 对每个 seed 内的 `W3FM_frozen` 与最强 baseline 做 subject-day block paired bootstrap；再汇总三 seed 的 paired delta mean ± std、方向一致性和 bootstrap 区间；
4. 制作主结果表、每个 protocol 的预测散点图和每被试相关系数分布图；
5. 仅按预注册胜出规则决定是否进入下一阶段。

**2026-08-25 执行记录**：Phase 3 已完成。`scripts/wear_only_fm/68_summarize_wear_only_fm_phase3.py` 对 Phase 2 冻结预测执行 `2000` 次 subject-day block paired bootstrap，并输出主结果表、seed-level W3FM vs best baseline bootstrap 表、每个 protocol 的 prediction scatter 和 per-subject r distribution。按 seed-level baseline 选择口径，`cross_day`/`cross_subject` 使用最高 raw r，`within_subject_day` 使用最高 centered r，RMSE 作为 tie-breaker。最终 gate 为 `stop_after_phase3_do_not_run_second_round_ablation`；`W3FM_frozen` 不进入第二轮消融。完整远端产物为 `outputs/wear_fm/phase3/`，本地同步副本为 `outputs/server_sync/wear_fm_phase3_20260825/`。

---

## 7. 第二轮消融：只在 W3FM_frozen 胜出后执行

| route | 回答的问题 |
|---|---|
| `W3FM_mean` | gate 是否有效；保留三路 frozen embeddings 和投影层，将 gated pooling 替换为 masked mean |
| `W3FM_no_pretrain` | 专属预训练 encoder 是否有效；保留同样三支路、投影、gate 和回归头，把 PaPaGei/HARNet/NormWear embeddings 替换为同模态非预训练特征或固定随机投影对照 |
| `W3FM_no_ppg` | PPG 支路的独立贡献 |
| `W3FM_no_acc` | 行为/运动支路的独立贡献 |
| `W3FM_no_eda` | 交感唤醒支路的独立贡献 |
| `W3FM_ppg_partial_ft` | PPG encoder 最后层局部微调能否进一步提升 |

第二轮优先只在 `cross_day` 和 `within_subject_day` 上进行，避免把计算量过早扩展到 `cross_subject`。`W3FM_mean` 只回答融合机制问题；预训练有效性必须由 `W3FM_no_pretrain` 或等价的非预训练三支路对照回答。

**2026-08-25 诊断执行记录**：虽然 Phase 3 formal gate 已判定 `stop_after_phase3_do_not_run_second_round_ablation`，但按后续问题定位需求，额外执行了 post-Phase3 diagnostic matrix。新增 `scripts/wear_only_fm/69_run_wear_only_fm_internal_ablation.py`，在 `cross_day` 与 `within_subject_day` 上运行 `W3FM_no_ppg`、`W3FM_no_acc`、`W3FM_no_gsr` 和 `W3FM_ppg_partial_ft` × seeds `240729/240730/240731`，共 `24` 个 run；每个 run 写出 `metrics.json`、`best_checkpoint.pt`、`val_history.csv`、`predictions.npz` 与 `test_predictions.csv`。服务器空间复查：`/vePFS-0x0d` 可用约 `264G`，`/tmp` 可用约 `703M`，运行时继续使用项目内 `outputs/tmp`；诊断矩阵目录约 `193M`。完整远端产物为 `outputs/wear_fm/internal_ablation/`，本地同步副本为 `outputs/server_sync/wear_fm_internal_ablation_20260825/`。

诊断结果：`cross_day` 上 `W3FM_no_acc` 明显优于完整 `W3FM_frozen`，三 seed 均值 RMSE `0.9841±0.0080`、raw r `0.0656±0.0610`、centered r `0.0484±0.0170`，相对完整 W3FM 的均值 delta 为 raw r `+0.0366`、RMSE `-0.0441`、centered r `+0.1130`，说明 ACC/HARNet10 支路在跨天协议下最可疑。`within_subject_day` 按主 centered-r 口径看，删除任何单一路径都没有改善完整 W3FM；`W3FM_no_acc` 虽降低 RMSE 到 `0.9976±0.0053` 且 raw r 为 `0.1152±0.0166`，centered r 降到 `-0.0949±0.0147`。`W3FM_ppg_partial_ft` 未稳定修复问题：`cross_day` raw r 均值 `-0.0313`、RMSE `1.0303`；`within_subject_day` centered r `0.0147`，仍低于完整 W3FM。该结果只作为失败来源诊断，不覆盖 Phase 3 的正式停止判定。

**2026-08-27 训练诊断记录**：针对 best epoch 普遍偏早的问题，新增 `scripts/wear_only_fm/70_summarize_wear_fm_training_diagnostics.py` 汇总 Phase 2、post-Phase3 internal ablation 与 training sweep 共 `93` 个 run 的 `metrics.json` 与 `val_history.csv`。当前结果显示 best epoch 偏早是 checkpoint-selection 现象，训练实际通常继续到 patience 窗口结束；W3FM-like route 的 train loss 持续快速下降，同时 val RMSE 在 best epoch 后明显漂高，符合 fast overfit 或优化过快，而非训练轮数不足。诊断指标：W3FM-like route `60/66` 触发 fast-overfit flag，普通 baseline `0/27`；W3FM-like train loss 平均下降 `64.71%`，普通 baseline 平均下降 `8.45%`。`66_run_wear_only_fm_phase2.py` 与 `69_run_wear_only_fm_internal_ablation.py` 已增加 richer history 字段和可选 `--selection-metric`，默认仍为 RMSE selection。

同日完成 W3FM_frozen 训练参数 sweep：远端 `outputs/wear_fm/training_sweep/` 共 `33` 个 run，本地同步到 `outputs/server_sync/wear_fm_training_sweep_20260827/`，由 `scripts/wear_only_fm/71_summarize_wear_fm_training_sweep.py` 生成 `wear_fm_training_sweep_summary.md`。`lr=1e-4` 可改善 RMSE 稳定性：`cross_day` RMSE 从原 W3FM `1.0282` 降到 `1.0055`，`within_subject_day` RMSE 从 `1.0293` 降到 `1.0117`；但 cross_day raw r 从 `0.0290` 降到 `-0.0004`，centered r 也下降。强正则小头 `--learning-rate 3e-4 --dropout 0.3 --weight-decay 1e-3 --hidden-dim 64` 只能部分降低 train loss drop，未恢复主指标。`within_subject_day` 使用 `--selection-metric centered_r` 会把 centered r 提到 `0.0574`，`lr=1e-4` 时到 `0.0594`，略高于 Wdeep centered r 均值，但 RMSE 明显升高到 `1.0793-1.0978`，raw r 仍低于 Wdeep。结论：当前问题主要是 W3FM 高维 frozen embedding 加 projection/gate/head 的快速过拟合或表示-融合错配；低学习率可作为 RMSE 稳定性诊断，centered-r selection 可作为口径冲突诊断，默认 Phase 2/3 训练口径暂不替换。

**2026-08-27 failure-mode 诊断记录**：新增 `scripts/wear_only_fm/72_diagnose_w3fm_failure_modes.py`，在 huoshan 上读取 canonical index、`wear_complete_mask`、三路 frozen embeddings、Phase 2 W3FM checkpoints/predictions 和 internal ablation summary，输出到远端 `outputs/wear_fm/failure_diagnostics/`，本地同步到 `outputs/server_sync/wear_fm_failure_diagnostics_20260827/`。该脚本不训练新 W3FM，不改变 Phase 3 stop 判定，只做解释性诊断：embedding split shift、top-feature correlation stability、轻量 ridge probe、checkpoint gate 权重和 gate/error association。

诊断结论：ACC/HARNet10 是当前最强可疑拖累分支。已有 leave-one-out 行为证据显示，`cross_day` 删除 ACC 后相对完整 W3FM raw r `+0.0366`、RMSE `-0.0441`、centered r `+0.1130`。ACC top train features 的训练相关性最高，但 cross_day test signed corr 为 `-0.0195`、sign agreement 仅 `0.2266`；within_subject_day test signed corr 也接近 `0`。轻量 ridge probe 中 PPG 明显强于 ACC：cross_day raw r `0.1107` vs `0.0599`，within_subject_day centered r `0.0315` vs `0.0028`，且 ACC 在两个协议里 train-test RMSE gap 最大。gate 进一步放大了这个问题：test 上 ACC 平均权重约 `0.50`，高误差样本的 ACC 权重比低误差样本高 `0.0985`（cross_day）和 `0.1047`（within_subject_day）。embedding 分布确有不匹配风险，但不是单一主因：GSR 的 train-standardized test shift 最大（cross_day `0.1480`、within_subject_day `0.1850`），而删除 GSR 没有带来同等行为改善；ACC 另有 `22-23%` 低方差 raw embedding 维度，后续救 W3FM 应优先做 ACC 分支过滤/降权/鲁棒融合，而不是继续单纯延长训练或只调学习率。

**2026-09-07 ACC 替换诊断记录**：按后续问题额外执行 `scripts/wear_only_fm/89_run_w3fm_acc_handcrafted_screen.py`。该脚本保留 PaPaGei-S PPG 与 NormWear GSR，直接从 `outputs/wear_fm/staged_inputs/acc_10s.npz` 生成 88 维 label-free ACC 统计/频域特征，用 `W3FM_acc_handcrafted` 替代 HARNet10 ACC embedding；同一报告读取已有 `W3FM_frozen` 与 `W3FM_no_acc` reference。运行协议为 `cross_day,within_subject_day` × seeds `240729/240730/240731`，新增 6 个 run。服务器空间复查：`/vePFS-0x0d` 可用约 `105G`，根分区可用约 `2.7G`；输出目录 `outputs/wear_fm/acc_replacement_screen/` 约 `13M`，本地同步副本为 `outputs/server_sync/wear_fm_acc_replacement_screen_20260907/`。

诊断结果：`cross_day` 上 handcrafted ACC 没有修复 W3FM，三 seed 均值 RMSE `1.0289+/-0.0120`、raw r `-0.0001+/-0.0168`、centered r `-0.0605+/-0.0173`，相对完整 `W3FM_frozen` raw r `-0.0291`、RMSE `+0.0007`，相对 `W3FM_no_acc` raw r `-0.0657`、RMSE `+0.0449`。因此跨日协议下最干净的诊断结论仍是删掉 ACC 比替换为简单稳定运动统计更好，问题不只是 HARNet10 高维 embedding 形式，也可能包括 ACC 在跨日 split 中携带的非稳定运动/佩戴状态信号。`within_subject_day` 上 handcrafted ACC 有可用信号：RMSE `1.0038+/-0.0091`、raw r `0.1580+/-0.0212`，raw r 高于完整 W3FM 和 no-ACC；但 centered r `0.0253+/-0.0169` 与完整 W3FM `0.0325+/-0.0131` 接近。结论只作为 post-Phase3 诊断，不改变 Phase 3 formal stop 判定；若继续救 W3FM，下一步应按 protocol 分开设计 ACC 分支过滤/降权或鲁棒 gate，而不是直接把 handcrafted ACC 升为主线。

**2026-09-11 NormWear-all 诊断记录**：按“三个模态都接 NormWear”的追问新增并执行 `scripts/wear_only_fm/90_run_w3fm_normwear_all_screen.py`。该脚本复用 staged 10 秒 PPG/ACC/GSR 输入和 `wear_complete_mask`，将三路都输入 frozen `NormWearModel`：PPG 与 GSR 做 patch mean pooling 得到 768D，ACC 先得到三轴 embedding 再按轴 mean pooling 成单个 768D ACC token，保持 W3FM 的三 token fusion 结构。下游投影、gate 与回归头仍按 protocol/seed 使用疲劳监督训练，并从 Phase 2、internal ablation 和 ACC replacement 结果读取 `Wphysio/Wdeep/Wmoment_frozen/W3FM_frozen/W3FM_no_acc/W3FM_acc_handcrafted` reference。服务器空间复查：运行前 `/vePFS-0x0d` 可用约 `79G`、根分区仅约 `88M`，因此命令显式设置 `TMPDIR/XDG_CACHE_HOME/MPLCONFIGDIR/HF_HOME` 到项目内 `outputs/tmp`；运行后 `/vePFS-0x0d` 可用约 `78G`。完整远端产物为 `outputs/wear_fm/normwear_all_screen/`，本地同步副本为 `outputs/server_sync/wear_fm_normwear_all_screen_20260911/`。

诊断结果：三路 NormWear 缓存均为 `(28819,768)`，有效行均为 `24127`，与 `wear_complete_mask` 一致。`cross_day` 上 `W3FM_normwear_all` 三 seed 均值 RMSE `0.9938+/-0.0198`、raw r `0.1515+/-0.0649`、centered r `0.0353+/-0.0463`；raw r 高于 Wdeep 与 no-ACC，但 RMSE 高于 `W3FM_no_acc`/Wdeep，centered r 低于 `W3FM_no_acc`。`within_subject_day` 上 RMSE `1.0363+/-0.0167`、raw r `0.1879+/-0.0035`、centered r `0.0379+/-0.0177`；raw r 接近 Wdeep，但 RMSE 明显差于 Wdeep `0.9813+/-0.0068`，centered r 也低于 Wdeep `0.0485+/-0.0054`。该分支说明统一 NormWear 表征能改善相关性读数，但仍没有把 W3FM 恢复为优于既有 wear-only baseline 的稳定主线；结论保持 post-Phase3 诊断，不改变 formal stop 判定。

---

## 8. 最终结果表模板

| protocol | route | seed | RMSE | MAE | raw r | centered r | per-subject r mean | best epoch | trainable params |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cross_day | Wphysio | 240729 |  |  |  |  |  |  |  |
| cross_day | Wdeep | 240729 |  |  |  |  |  |  |  |
| cross_day | Wmoment_frozen | 240729 |  |  |  |  |  |  |  |
| cross_day | W3FM_frozen | 240729 |  |  |  |  |  |  |  |

最终论文主表报告每个 route 在三 seed 下的 `mean ± std`；附录给出完整 36-run 表、参数量、耗时、bootstrap 置信区间和消融结果。
