# Wear × MOMENT 接入实验方案（2026-08-20）

> 目标：把 MOMENT-1 预训练时间序列模型接入 Wear 模态，产出与现有 `Wphysio`/`Wdeep` 可 paired 对比的融合矩阵结果。
> 背景：方案依据 [wear_model_selection_20260820.md](wear_model_selection_20260820.md)；技术口径见 [technical_route_20260814.md](../../current/0814-window/technical_route_20260814.md)。
> 一句话设计：**两阶段矩阵（1-seed screen → 3-seed paired confirm）+ 固定 EEG 主线（`eegpt_partial_ft_v1`）+ 视频固定 A1 + 音频保留 full + wear 基线同配对照 + 三协议全跑**，全部走现有 `32_run_eegpt_centered_loss.py` 融合链路，融合器零改动。

---

## 0. 目标与结论口径

| 问题 | 口径 |
| --- | --- |
| MOMENT 预训练表征是否优于随机卷积基线？ | `wear_moment_frozen_v1` vs `wear_deep`（同 EEG/video/audio/seed 配置） |
| 微调是否带来额外增益？ | `wear_moment_partial_ft_v1` vs `wear_moment_frozen_v1`（paired） |
| 换表征后融合整体是否变好？ | 新 route 的 test RMSE / raw r / centered r vs 现有矩阵同配置 top |
| 监督边界 | frozen 投影头与 partial FT 均为 fatigue-supervised representation（与 EEG 侧 `cbramod_frozen_v1` / `eegpt_partial_ft_v1` 同口径）；只有 train/val 参与训练，test 冻结评估 |
| 开放获取 | MOMENT-1-small（HF `AutonLab/MOMENT-1-small`，MIT），权重下载后落 `outputs/checkpoints/`，记录版本与哈希 |

---

## 1. 前置准备

1. **依赖**：服务器 `runtime/envs/eegpt-gpu-min` 安装 `pip install momentfm`；本地开发机仅需要代码冒烟（无 torch 时测试跳过，仿现有约定）。
   - **2026-08-20 实测**：momentfm 0.1.4 硬性钉死 `numpy==1.25.2`、`transformers==4.33.3`、`huggingface-hub==0.24.0`。已用 `pip install --no-deps momentfm` + `pip install "transformers==4.33.3"` 安装；**副作用：环境 numpy 2.4.6 → 1.25.2**（pandas 2.3.3 与 numpy 1.25.2 共存已验证可 import；需在变更记录中注明）。momentfm 源码经扫描仅 `np.Inf`（EarlyStopping 内，本路线不触发）一处 numpy-2 不兼容，运行路径不受影响。
2. **Checkpoint**：`huggingface-cli download AutonLab/MOMENT-1-small --local-dir outputs/checkpoints/moment-1-small`，记录 commit/哈希；本轮不下载 `MOMENT-1-large`（暂不试 large，见 §4 规模决策）。
   - **2026-08-20 实测**：服务器无法直连 huggingface.co（DNS 仅 IPv6）与 hf-mirror.com（443 间歇超时，最终可达）。已从 **hf-mirror.com** 下载 `config.json` / `model.safetensors` / `README.md` 到服务器 `outputs/checkpoints/moment-1-small/`（repo sha `411e288267f82cce86296dbe4d6c8bc533cc162f`；model.safetensors sha256 `785e6c6f57ffa7cac7e2a1fff6369618d49f2f441563ddea76c866231e5aa877`，145 MB）。MOMENT-1-small 实测 d_model=512，`task_name="embedding"` 输出 `(B, 512)` 序列级表征（momentfm 0.1.4 的 `forward` 为 keyword-only `x_enc`）。
3. **数据 preflight（2026-08-20 已完成）**：
   - aligned 28,819 窗口的 sample_id 为 `eeg_000000` 风格，**index 本身不含 wear 字段**；wear 元数据来自 `wear_physio_preprocessed_eeg23win_embeddings.npz`：`wear_mask`（**24,127/28,819，83.7%**）、`source_wear_file`（JSON，记录旧机器 `/mnt/dataset0/...` 路径）、`window_start_time/end_time`。
   - 源 CSV 实际位于 **`/vePFS-0x0d/DailyEEG_multimodal/raw/wear/out/<basename>`**（按 basename 重定位）；**372 个被引用文件（124 PPG + 124 GSR + 124 ACC）全部存在**，列结构与 `wear_real.TARGET_COLUMNS` 一致（PPG/GSR 双列、ACC 四列，时间格式 `YYYY-MM-DD HH:MM:SS`）。
   - `sample_id` 顺序与 canonical index **完全一致**（`np.array_equal` 通过）。
   - MOMENT-1-small 实测：模型加载 1.6 s；H20 上 64 窗口推理 0.67 s（**约 17 ms/窗口**，全量 24,127 窗口约 7 分钟）；输出 `(4, 512)` 全 finite。
4. **输入规格**：MOMENT 输入 = 现有 `_aligned_sequence_matrix` 产出（320 步 × 5 通道：PPG+GSR+ACC，逐通道 z-norm）**转置为通道在前 `(5, 320)`**（`_aligned_sequence_matrix` 返回 `(步, 通道)`），满足 MOMENT-1 的 ≤512 步、多通道约束；MOMENT 内部 RevIN 与现有归一化兼容。

---

## 2. 工程改动清单

| # | 改动 | 位置 | 内容 |
| --- | --- | --- | --- |
| T1 | 新训练模块 | `src/daily_multimodal/training/wear_moment_matrix.py`（新） | 仿 `eeg_encoder_matrix.py`：`WEAR_MOMENT_PROFILES = {"wear_moment_frozen_v1", "wear_moment_partial_ft_v1"}`；加载 aligned index + splits + wear 序列 → MOMENT encoder；frozen = encoder 冻结 + 可学习 256D 投影头（fatigue 监督，train/val）；partial FT = 解冻最后 2 个 block + norm + projection（encoder lr `1e-5`，head lr `1e-3`）；输出 `{embeddings_dir}/wear_tokens/{protocol}/{profile}/seed_{seed}.npz` |
| T2 | 新入口脚本 | `scripts/window_fatigue/16_run_wear_moment_matrix.py`（新） | 仿 `34_run_eeg_encoder_matrix.py`：`--profiles --protocols --seeds --embeddings-dir --out-json --out-md --preflight-only` 等 |
| T3 | 融合挂载 | `scripts/window_fatigue/32_run_eegpt_centered_loss.py` | `BRANCHES` 增加 `wear_moment_frozen_v1` / `wear_moment_partial_ft_v1`：filename = `wear_tokens/{protocol}/{profile}/seed_{eeg_seed}.npz`（复用 `--eeg-token-seed`，CLI 零新增参数），`emb_key="wear_emb"`、`mask_key="wear_mask"`、`modality_index=1`；`EXPERIMENT_BRANCHES` 增加 2 个 route：`A1_Wmoment_frozen_full` / `A1_Wmoment_ft_full`（见 §3.4）；另加可选小改 `--experiment-seed-fixed`：每个 experiment 用固定 seed（默认行为不变），保证 Phase 2 的 4 条 wear route 严格同 seed 配对；执行时用 `--experiment-set custom --experiments` 显式列出 12 项（**勿用 `video_only`**，否则会把 B0/A2 与 no_audio 全部带回来） |
| T4 | 单元测试 | `tests/test_wear_moment_matrix.py`（新） | 冒烟：合成 `sequence.npz` → token 形状 `(N,256)`、`sample_id` 顺序、NaN=0、mask 复用；无 torch 时跳过（仿 `test_eeg_encoder_matrix`） |
| T5 | token npz 契约 | 与 `write_eeg_embedding_npz` 对齐 | `sample_id / wear_emb (N,256) / wear_mask (N,) / modality_mask[:,1] / train_index / val_index / test_index / train_supervision / encoder_profile / encoder_version / protocol / seed` |
| T6 | 汇总与文档 | `scripts/README.md`、命令参考、`technical_route_20260814.md`（若通过） | 记录新 route 与监督边界 |

> 说明：`wear_moment_*` 的 token 训练是**独立于融合矩阵的中间产物**（每 protocol × profile × seed 一次），token 落盘后融合阶段与 Wphysio/Wdeep 完全同构。

---

## 3. 实验矩阵

### 3.1 Phase 0：技术冒烟（不计入矩阵）

单窗口级验证：`sequence.npz` → 320×5 → MOMENT-1-small → 表征池化 → 256D 投影。
验收：输出形状/顺序正确、NaN=0、单窗口 CPU/GPU 推理耗时与显存可接受、token 训练 1 epoch 可跑通。
产物：`outputs/reports/wear_moment_smoke_{timestamp}.json`。

### 3.2 Phase 1：screen 矩阵（1 seed）

| 维度 | 取值 | 数量 |
| --- | --- | --- |
| EEG branch | `eeg_eegpt_partial_ft_v1`（唯一） | 1 |
| Wear route | `wear_physio`、`wear_deep`（基线）、`wear_moment_frozen_v1`、`wear_moment_partial_ft_v1` | 4 |
| Video | `video_A1`（唯一） | 1 |
| Audio | full（唯一，默认保留） | 1 |
| Protocol | `cross_subject`（诊断）、`cross_day`、`within_subject_day` | 3 |
| Seed | 融合 seed 240729（`--experiment-seed-fixed`，组内 12 runs 严格同 seed）；token seed 240800 | 1 |

**runs = 1 × 4 × 1 × 1 × 3 = 12**。

**token 训练 runs（中间产物，不进入融合矩阵计数）**：`wear_moment_frozen_v1` / `wear_moment_partial_ft_v1` × 3 协议 × 3 seed（240800/240801/240802）一次算齐 = 18 次训练，Phase 1 只用 seed 240800 组；frozen 的 MOMENT 表征只提取一次缓存，各协议复用。

### 3.3 Phase 2：paired confirm 矩阵（3 seeds）

维度与 Phase 1 完全相同（EEG=1、Video=A1、Audio=full、Wear=4、Protocol=3），只把 seed 扩到 3 组：

| 维度 | 取值 | 数量 |
| --- | --- | --- |
| EEG branch | `eeg_eegpt_partial_ft_v1` | 1 |
| Wear route | 4（两个基线 + 两个 small 档，**同配置配对**） | 4 |
| Video | `video_A1` | 1 |
| Audio | full | 1 |
| Protocol | 3（`cross_subject` 为诊断口径） | 3 |
| Seed | 3 组配对：融合 240729/240730/240731 × token 240800/240801/240802（每组用 `--experiment-seed-fixed`，组内 12 runs 严格同 seed） | 3 |

**runs = 1 × 4 × 1 × 1 × 3 × 3 = 36**（3 次调用 × 12 项实验；Phase 1 的 12 runs 即融合 seed 240729 组，不重复计算）。
**token 训练 runs**：3 协议 × 2 profiles × 3 seeds = 18 次训练（Phase 1 复用其中 seed 240800 组，不重复计算）。

### 3.4 新增融合 route 清单（挂载到 `EXPERIMENT_BRANCHES`）

```
A1_Wmoment_frozen_full   = ("eeg", "wear_moment_frozen_v1", "video_A1", "audio")
A1_Wmoment_ft_full       = ("eeg", "wear_moment_partial_ft_v1", "video_A1", "audio")
```

基线 route 复用已有 `A1_Wphysio_full` / `A1_Wdeep_full`（已在 `EXPERIMENT_BRANCHES` 中），无需新增。

### 3.5 矩阵规模汇总

| 阶段 | 融合 runs | token 训练 runs | 预计 GPU 时间* |
| --- | ---: | ---: | ---: |
| Phase 0 冒烟 | 0 | 1（1 epoch） | < 30 min |
| Phase 1 screen | 12 | 18 次一次算齐（Phase 1 用 seed 240800 组） | 约 1–1.5 h |
| Phase 2 confirm | 36 | 复用 18 次 | 约 2–4 h |
| 合计 | 48 | 18 | 约 3–6 h |

*按 H20、单 fusion run 1–3 min、MOMENT-small 推理约 10–20 s/百窗口、token 训练每 run 5–15 min 估算，实际以冒烟为准。

---

## 4. 协议与超参

| 环节 | 取值（与现有矩阵一致，不新增自由度） |
| --- | --- |
| 划分 | train = pretrain + finetune；val 早停选模型；test 冻结评估；`splits_new` 三协议 |
| 融合训练 | AdamW lr `1e-3`、wd `1e-4`、batch 256、epochs ≤80、patience 15、dropout 0.1、hidden 128、AMP、loss `raw`（首轮不做 centered 变体，避免矩阵膨胀） |
| MOMENT token 训练 | frozen：投影头 lr `1e-3`；partial FT：encoder lr `1e-5`、head lr `1e-3`、解冻最后 2 个 block + norm + projection；epochs ≤80、patience 15、batch 256、dropout 0.1、AMP |
| MOMENT 表征池化 | 官方 `task_name="embedding"` 的 `sample_embeddings`（或对 patch token 做 mean/std 池化，二选一并在报告记录）；`MOMENT-1-small` 为主路线 |

**模型规模决策**：本方案**只跑 MOMENT-1-small**（约 12M 参数），暂不试 large。依据：① 本方案只取表征喂轻量融合器，输入仅 320×5×10 s，表征上限由 Wear 信号本身决定，不由模型容量决定；② 28,819 窗口的监督规模下，large（约 120M）partial FT 过拟合风险高，small 微调更稳；③ large 的收益是经验性的，不能预设。决策规则：Phase 1/2 全部用 small 判定 G1–G3；large（含 frozen 消融）本轮不做，若后续 small 结果接近门槛或需要更高表征上限，再按当时的基线重新评估。
| Mask 口径 | 新 token 的 `wear_mask` 直接复用现有 `wear_physio_preprocessed_eeg23win_embeddings.npz` 的 `wear_mask` |

---

## 5. 验收门槛（gate）

| 门槛 | 条件 | 动作 |
| --- | --- | --- |
| G0（冒烟） | token 100% 顺序对齐、NaN=0、推理耗时/显存可接受 | 不满足先修，不进矩阵 |
| G1（screen 通过） | 至少一个主协议（`cross_day` / `within_subject_day`；`cross_subject` 仅诊断）存在组合满足：同配置下 `wear_moment_frozen_v1` 的 raw r 或 RMSE 优于 `wear_deep`（paired，同 EEG/video/audio/seed） | 满足才进 Phase 2；否则记录负结果，本轮停 |
| G2（partial FT 增益） | `wear_moment_partial_ft_v1` 相对 frozen 在 G1 组合上 raw r 或 RMSE 不显著倒退（且任一项改善） | 决定 confirm 阶段是否保留 partial FT 档 |
| G3（confirm 稳定） | 3-seed 均值下增益方向一致（≥2/3 seeds 同号），且 test 上 raw r 或 RMSE 优于基线均值 | 通过 → 写入技术路线文档并考虑纳入主线；不通过 → 保留为诊断结论 |
| 附加记录 | 每个新 route 的 `train_supervision`（frozen 投影头 / partial FT 均为 fatigue-supervised）；模型版本与权重哈希 | 随报告输出 |

---

## 6. 产物

| 产物 | 路径 |
| --- | --- |
| MOMENT token（每 protocol/profile/seed） | 服务器 `{embeddings_root}/wear_tokens/{protocol}/{profile}/seed_{seed}.npz` |
| token 训练汇总 | `outputs/server_sync/wear_moment_20260820/` 下 JSON/Markdown（仿 `eeg_encoder_256d_5route_20260814` 布局） |
| 融合矩阵报告 | `{report_dir}/wear_moment_fusion_video_only_seed240800_raw.json` + `.md`（含 mask 覆盖率、paired deltas、三协议 top 表） |
| 覆盖率/顺序 preflight | `outputs/reports/wear_moment_preflight_{timestamp}.json` |

---

## 7. 风险与回退

| 风险 | 表现 | 缓解 |
| --- | --- | --- |
| wear 序列覆盖率不足 | preflight 覆盖率显著低于现有 wear_mask | 回退到 window index 源路径直读；若仍不足，用现有覆盖子集并在报告标注 |
| MOMENT 推理慢 | 28,819 窗口耗时长 | 分块并行 + 表征一次性缓存（各 protocol 复用）；small 优先 |
| partial FT 过拟合/不稳定 | val 早停 epoch 很小或震荡 | 降低 encoder lr 至 `5e-6`；只解冻 1–2 个 block；必要时 frozen 档单独进 confirm |
| 输入格式错配（MOMENT 对 320×5 的适配） | frozen 差但 partial FT 好 | 说明错配可被微调吸收；先按现有 `_aligned_sequence_matrix` 口径跑，若 frozen 全面劣于随机基线再检查归一化/通道顺序 |
| 融合 seed 与既有 report 不可比 | paired 分析失真 | 所有 paired 对比必须在**同一 fusion seed、同一配置**下进行；跨 report 只做参考不做判定 |

---

## 8. 执行顺序（任务清单）

1. T0 preflight：覆盖率/顺序/耗时冒烟（§1.3、§3.1）
2. T1–T4 代码：新模块 + 脚本 + 融合挂载 + 测试（§2）
3. 本地验证：`py_compile` + 无 torch 单测；服务器 smoke（1 protocol × 1 seed × 1 route，1 epoch）
4. Phase 1 screen：12 runs（§3.2，`--experiment-set custom` 显式列出 12 项）
5. 门槛判定 G1/G2（`cross_subject` 已在矩阵内，按诊断口径解读）
6. Phase 2 confirm：36 runs（§3.3，`--experiment-seed-fixed` 保证配对），paired 汇总
7. 门槛判定 G3；写报告；更新 `scripts/README.md`、命令参考；通过则更新 `technical_route_20260814.md` Wear 表格并登记 change-log

---

## 9. 追加：方案 C 全维度矩阵（2026-08-20 用户选定）

Phase 1/2 结论（frozen 主协议 3/3 稳定优于 Wdeep、partial FT 仅 cross_subject 稳定）之后，按用户选择执行**方案 C**：把两条 moment 路线铺满全部 video/audio 维度，EEG 只保留最相关两条：

| 维度 | 取值 | 数量 |
| --- | --- | --- |
| EEG branches | `eeg_eegpt_partial_ft_v1`（主线）+ `eeg_eegpt_frozen_v1`（低监督对照） | 2 |
| Fusion routes | B0/A1/A2 × Wphysio/Wdeep/Wmoment_frozen/Wmoment_ft × full/no_audio | 24 |
| Protocols | cross_subject / cross_day / within_subject_day | 3 |
| Seed | 融合 240729（`--experiment-seed-fixed`，严格配对）；EEG/wear token 均 240800 | 1 |

**runs = 2 × 24 × 3 = 144**（固定 seed 单轮；旧 180-run 矩阵是 run_number 递增 seed，口径不同不可混比）。

工程改动：
- `scripts/window_fatigue/32_run_eegpt_centered_loss.py` 的 `EXPERIMENT_BRANCHES` 新增 10 个 route：`A1/B0/A2_Wmoment_frozen_full/_no_audio`、`A1/B0/A2_Wmoment_ft_full/_no_audio`（其中 A1 的 full 两个已在 Phase 1/2 使用）。
- `scripts/window_fatigue/53_summarize_wear_moment_gates.py` 泛化为全维度：按 (protocol, eeg_branch, video, audio) 配置组配对，输出 36 个配置组的 G1/G2 delta 与门槛判定。

产物：`outputs/server_sync/wear_moment_20260820/planC_144/`（report JSON/MD + gates 汇总 + 完整结果表）。
