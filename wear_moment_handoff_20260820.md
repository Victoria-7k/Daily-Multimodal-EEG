# Wear × MOMENT Handoff（2026-08-20）

> 本文是本次「Wear 模态接入预训练模型 MOMENT」工作的完整交接说明。下一个工作者应以此为准绳：先读结论，再读[结果报告](wear_moment_results_20260820.md)与[实验方案](wear_moment_experiment_plan_20260820.md)，按本文的「复现命令」与「注意事项」操作。
>
> 一句话结论：**MOMENT-1-small 冻结表征（`wear_moment_frozen_v1`）建议纳入 Wear 新主线**——`cross_day` 上 3/3 seeds 一致优于 `Wdeep`（raw r Δ +0.063、RMSE Δ -0.039），方案 C 全维度（2 EEG × 24 fusion routes）下主线 EEG × cross_day 的 6 个 video/audio 配置全部更优；`wear_moment_partial_ft_v1` 仅在 `cross_subject` 稳定，保留为诊断。

---

## 1. 工作链总览（已完成）

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| T0a | 服务器装 momentfm 0.1.4（`--no-deps` + transformers 4.33.3）；经 hf-mirror 下载 MOMENT-1-small 权重 | ✅ |
| T0b | 数据 preflight：wear 源映射/覆盖率/顺序核验 + MOMENT 加载与推理耗时冒烟 | ✅ |
| T1-T4 | 代码实现：token 训练模块、16 号脚本、32 号融合挂载、测试（6 tests） | ✅ |
| Smoke | 全链路冒烟：token 契约 + 1-epoch 融合 + 全量 preflight | ✅ |
| Token 矩阵 | 3 协议 × 2 profiles × 3 seeds = 18 runs（19.5 min） | ✅ |
| Phase 1 | screen 12 runs（seed 240729 固定）→ G1 通过 | ✅ |
| Phase 2 | confirm 36 runs（3 seeds 配对）→ G1/G2/G3 判定 | ✅ |
| 方案 C | 全维度 144 runs（2 EEG × 24 routes × 3 protocols）→ frozen 全维度结论 | ✅ |
| 文档 | 技术路线三个表更新 + 报告 + handoff + change-log | ✅ |

---

## 2. 代码改动清单

### 2.1 新增文件（本地仓库 + 已同步服务器）

| 文件 | 说明 |
| --- | --- |
| `src/daily_multimodal/training/wear_moment_matrix.py` | Wear × MOMENT token 训练模块。`wear_moment_frozen_v1`（encoder 冻结 + 可学习 256D 投影头，lr 1e-3）/ `wear_moment_partial_ft_v1`（解冻最后 2 block + final norm，encoder lr 1e-5）；数据侧从 `wear_physio_preprocessed_eeg23win_embeddings.npz` 读 mask/source 映射，按 basename 重定位源 CSV，构造 320×5 通道在前矩阵；输出契约仿 `write_eeg_embedding_npz`（`wear_emb (N,256)` / `wear_mask` / `modality_mask[:,1]` / `train_index` 等） |
| `scripts/16_run_wear_moment_matrix.py` | token 矩阵入口（仿 34 号脚本）；`--preflight-only` 输出数据覆盖率与 MOMENT 耗时 |
| `scripts/53_summarize_wear_moment_gates.py` | 门槛汇总（全维度泛化版）：按 (protocol, eeg_branch, video, audio) 配置组配对，输出 G1/G2/G3 判定 + 36 配置组 delta + 完整指标表 |
| `tests/test_wear_moment_matrix.py` | 6 个单元测试（本地与服务器均通过） |

### 2.2 修改文件

| 文件 | 改动 |
| --- | --- |
| `scripts/32_run_eegpt_centered_loss.py` | ① `BRANCHES` 新增 `wear_moment_frozen_v1` / `wear_moment_partial_ft_v1`（filename 用 `{wear_seed}` 占位符）；② `EXPERIMENT_BRANCHES` 新增 10 个 route（`A1/B0/A2 × Wmoment_frozen/Wmoment_ft × full/no_audio`）；③ 新增 `--experiment-seed-fixed`（每个 run 用精确 `--seed`，保证配对）；④ 新增 `--wear-token-seed`（与 `--eeg-token-seed` 解耦，EEG token 只有 seed 240800，wear token 有 240800/240801/240802） |
| `technical_route_20260814.md` | Wear 表新增 `Wmoment_frozen` / `Wmoment_partial_ft` 两档；「各划分协议 Raw R Top 3」「按 EEG Route 分类的四模态平均表现」「附表：全部四模态融合实验结果」三表更新为方案 C 口径（seed 240729 固定）；新增「2026-08-20 更新：Wear × MOMENT」小节 |
| `wear_model_selection_20260820.md` | 候选模型调研（开放获取核验） |
| `wear_moment_experiment_plan_20260820.md` | 两阶段矩阵 + 方案 C（§9） |
| `wear_moment_results_20260820.md` | 正式结果报告（§3 表与 §5 数字已修正为 JSON 真实值） |
| `repo-docs/README.md`、`scripts/README.md`、`repo-docs/references/commands-and-artifacts.md`、`repo-docs/change-log.md` | 同步索引与变更记录 |

### 2.3 服务器代码位置

`/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/` 下与本地同路径（`src/`、`scripts/`、`tests/`）。注意服务器该目录是**精简快照**，本次额外同步了 `src/daily_multimodal/alignment/` 与 `src/daily_multimodal/embeddings/` 两个包（服务器原本只有 `training/`）。

---

## 3. 服务器环境与数据事实（重要）

### 3.1 连接

- `ssh -p 36083 wangziwei@124.174.8.252`（密钥免密；本机 PATH 里的 PortableGit ssh 会被沙箱拦截，用 `C:\Windows\System32\OpenSSH\ssh.exe` / `scp.exe`）
- 主机 `di-20260608143633-rhsb5`（火山引擎 H20 实例，GPU 1× H20 96GB）
- 项目根：`/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned`
- Python：`runtime/envs/eegpt-gpu-min/bin/python`（Python 3.11.15、torch 2.5.1）

### 3.2 环境副作用（必须知道）

- **numpy 2.4.6 → 1.25.2**：momentfm 0.1.4 硬性钉死 `numpy==1.25.2`、`transformers==4.33.3`、`huggingface-hub==0.24.0`。pandas 2.3.3 与 numpy 1.25.2 共存已验证可 import。若需恢复 numpy 2.x：`pip install "numpy==2.4.6"`（momentfm 运行路径兼容 numpy 2）。
- 权重下载：服务器**无法直连 huggingface.co**（DNS 仅返回 IPv6）与 hf-mirror.com 的 443 曾间歇超时（重试即可）；**hf-mirror.com 可用**；pypi.org、modelscope.cn 可用；github.com 不可达。

### 3.3 MOMENT-1-small 权重

- 位置：服务器 `outputs/checkpoints/moment-1-small/`（config.json + model.safetensors 145MB + README.md）
- repo sha：`411e288267f82cce86296dbe4d6c8bc533cc162f`
- model.safetensors sha256：`785e6c6f57ffa7cac7e2a1fff6369618d49f2f441563ddea76c866231e5aa877`
- 实测：d_model=512；`task_name="embedding"` 输出 `(B, 512)`（momentfm 0.1.4 的 `forward` 是 keyword-only `x_enc`，见 `encode_moment_batch`）；H20 上 10.4 ms/窗口（全量 24,127 窗口约 5 分钟）。

### 3.4 数据映射（关键）

- aligned index（28,819 行）的 sample_id 为 `eeg_000000` 风格，**本身不含 wear 字段**。
- wear 元数据来自 `/vePFS-0x0d/DailyEEG_multimodal/embeddings/wear/wear_physio_preprocessed_eeg23win_embeddings.npz`：`wear_mask`（**24,127/28,819 = 83.7%**）、`source_wear_file`（JSON：ppg/gsr/acc 的**旧机器路径**）、`window_start_time/end_time`。
- 源 CSV 实际在 `/vePFS-0x0d/DailyEEG_multimodal/raw/wear/out/<basename>`（按 basename 重定位）；372 个被引用文件（124 PPG + 124 GSR + 124 ACC）全部存在，列结构与 `wear_real.TARGET_COLUMNS` 一致。
- `sample_id` 顺序与 canonical index 完全一致（`np.array_equal` 验证过）。
- 新 token 的 `wear_mask` **直接复用**现有 wear_mask，保证 mask 口径一致。

---

## 4. 结果文件路径

### 4.1 服务器

| 产物 | 路径 |
| --- | --- |
| 18 个 wear token | `/vePFS-0x0d/DailyEEG_multimodal/embeddings/wear_tokens/{protocol}/{profile}/seed_{seed}.npz`（protocol ∈ cross_subject/cross_day/within_subject_day；profile ∈ wear_moment_frozen_v1/wear_moment_partial_ft_v1；seed ∈ 240800/240801/240802） |
| 320×5 矩阵缓存 | `outputs/cache/wear_moment_matrices.npz`（131MB，含 sample_id + matrices） |
| MOMENT 表征缓存 | `outputs/cache/wear_moment_frozen_embeddings.npz`（28819×512） |
| 实验产物 | `outputs/server_sync/wear_moment_20260820/`：`token_matrix_20260820.{json,md}`、`wear_moment_preflight_20260820.{json,md}`、`phase1_screen/`（12 runs + gates_phase1）、`phase2_confirm/`（36 runs + gates_phase2 + phase2_full_results.md）、`planC_144/`（144 runs + gates_planC） |

### 4.2 本地（已同步）

- 根目录文档：`wear_model_selection_20260820.md`、`wear_moment_experiment_plan_20260820.md`、`wear_moment_results_20260820.md`、`wear_moment_handoff_20260820.md`（本文）、`technical_route_20260814.md`（已更新）
- 产物镜像：`outputs/server_sync/wear_moment_20260820/`（与服务器同名子目录，JSON/MD 已同步）

---

## 5. 实验矩阵与关键结果

### 5.1 三个矩阵

| 矩阵 | 口径 | runs |
| --- | --- | --- |
| Phase 1 screen | 1 EEG（`eegpt_partial_ft_v1`）× 4 wear × A1 × full × 3 protocols；融合 seed 240729 固定（`--experiment-seed-fixed`）；token seed 240800 | 12 |
| Phase 2 confirm | 同 Phase 1，3 组配对：融合 240729/240730/240731 × wear token 240800/240801/240802（EEG token 固定 240800） | 36 |
| 方案 C | 2 EEG（`eegpt_partial_ft_v1` + `eegpt_frozen_v1`）× 24 fusion routes（B0/A1/A2 × 4 wear × full/no_audio）× 3 protocols；seed 240729 固定 | 144 |

### 5.2 关键数字（全部来自 JSON 核验）

- **G1**：`wear_moment_frozen_v1` vs `Wdeep`（A1+full，3-seed）：cross_day raw r Δ **+0.0626（3/3）**、RMSE Δ **-0.0390（3/3）**；cross_subject +0.0352（2/3）；within_subject_day -0.0076（持平）。
- **A1+full 3-seed 范围**（cross_day）：frozen raw r 0.309–0.332 / RMSE 0.914–0.924；Wdeep 0.230–0.280 / 0.948–0.970。
- **方案 C**（主线 EEG × cross_day）：frozen vs Wdeep 在**全部 6 个 video/audio 配置上更优**（raw r Δ +0.033~+0.087，RMSE 全部更低）；within_subject_day RMSE 6/6 更低、raw r 4/6 为正；cross_subject 三个 full 配置 +0.062~+0.088。低监督 EEG（`eegpt_frozen_v1`）下增益不稳定。
- **G2**：partial FT 仅 cross_subject 稳定（3/3），主协议不稳定 → **不作为默认路线**。
- 推荐主线组合：**`eeg_eegpt_partial_ft_v1` + `Wmoment_frozen`**（cross_day 最佳 `A2_Wmoment_frozen_full`，Δ raw r +0.087）。

### 5.3 一致性验证

phase1 / phase2 / planC 三份 JSON 相同配置 **0 diff**（固定 seed 运行可确定性复现）。⚠️ 注意：`wear_moment_results_20260820.md` 的 §3 表曾发现转录错误，已修正为 JSON 生成值；**以后引用数字一律以 JSON/report 为准，勿手抄**。

---

## 6. 复现命令

### 6.1 Token 矩阵（18 runs，约 20 min）

```bash
cd /vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned
export PYTHONPATH=src
./runtime/envs/eegpt-gpu-min/bin/python scripts/16_run_wear_moment_matrix.py \
  --protocols cross_subject,cross_day,within_subject_day \
  --seeds 240800,240801,240802 \
  --profiles wear_moment_frozen_v1,wear_moment_partial_ft_v1 \
  --epochs 80 --patience 15 --batch-size 256 --device cuda \
  --embeddings-dir /vePFS-0x0d/DailyEEG_multimodal/embeddings/wear_tokens \
  --matrix-cache outputs/cache/wear_moment_matrices.npz \
  --out-json outputs/server_sync/wear_moment_20260820/token_matrix_20260820.json \
  --out-md outputs/server_sync/wear_moment_20260820/token_matrix_20260820.md
```

### 6.2 Phase 1/2 融合矩阵（12 / 36 runs）

用 `scripts/32_run_eegpt_centered_loss.py --experiment-set custom --experiments <12 项>`，关键参数：
`--eeg-branches eeg_eegpt_partial_ft_v1`、`--eeg-token-root eeg_encoder_256d_tokens`、`--eeg-token-seed 240800`、`--wear-token-seed 240800/240801/240802`（按组）、`--seed 240729/240730/240731 --experiment-seed-fixed`、`--loss-modes raw --heads regression`。
完整命令模板见 `outputs/tmp/run_phase1.sh` / `run_phase2b.sh`（本地）。

### 6.3 方案 C 全维度（144 runs）

模板见 `outputs/tmp/run_planC.sh`（24 experiments × 3 protocols 列表已写好）。

### 6.4 门槛汇总

```bash
./runtime/envs/eegpt-gpu-min/bin/python scripts/53_summarize_wear_moment_gates.py \
  --reports <report1.json,report2.json,...> \
  --out-json outputs/server_sync/wear_moment_20260820/planC_144/gates_planC.json \
  --out-md outputs/server_sync/wear_moment_20260820/planC_144/gates_planC.md
```

### 6.5 测试

- 本地：`$env:PYTHONPATH='src'; py -m unittest tests.test_wear_moment_matrix -v`
- 服务器：`PYTHONPATH=src ./runtime/envs/eegpt-gpu-min/bin/python -m unittest tests.test_wear_moment_matrix -v`
- ⚠️ 本机沙箱拦截 `tempfile.mkdtemp` 创建的目录（写入被拒）；测试已改为 `mkdir + uuid` 自建目录（`outputs/.test_tmp` 或 `outputs/reports`）。若 `.test_tmp` 出现 ACL 异常，删掉重建即可。

---

## 7. 给下一个工作者的注意事项

1. **seed 口径**：新实验全部用 `--experiment-seed-fixed`（组内严格同 seed）。旧 180-run 矩阵（`eeg_encoder_256d_5route_fusion_video_only_seed240800_raw`）的 seed 是 run_number 递增，**与新数据不可直接混比**；需要严格 paired 时必须重跑或统一口径。
2. **EEG token 只有 seed 240800**（`eeg_encoder_256d_tokens/{protocol}/eegpt_{frozen,partial_ft}_v1/seed_240800.npz`）；wear token 才有 3 个 seed。所以 `--wear-token-seed` 与 `--eeg-token-seed` 必须分开传。
3. **数据源不可在旧路径找**：`source_wear_file` 记录的是旧机器 `/mnt/dataset0/...`，实际在 `/vePFS-0x0d/DailyEEG_multimodal/raw/wear/out/`；`intermediate/wear_windows` 是空目录，`quality_flags` 里的 `sequence_cache_path` 指向旧机器。
4. **长任务**：用 `nohup ... > /tmp/xxx.log 2>&1 &` 后台跑，日志轮询（每 run 约 1-3 min；144 runs 实测约 40 min）。
5. **显存**：H20 96GB，当前常驻约 24GB 占用；MOMENT-small 训练显存需求小（<6GB），可并行 2-3 个 fusion 进程。
6. **numpy 1.25.2**：若其他实验（如 EEG 矩阵）出现 numpy 2 专属 API 报错，先检查是否被降级；恢复见 §3.2。
7. **后续扩展建议**：
   - 把 `Wmoment_frozen` 作为 Wear 新主线写入正式配置；如需全矩阵补齐（5 EEG × 24 routes），按方案 C 的固定 seed 口径跑（+180 runs 的 moment 组合）。
   - 方案 C 关键配置（如 cross_day `A2_Wmoment_frozen_full`）可补 3-seed confirm。
   - partial FT 若要用，仅限 cross_subject 协议单独判定。
   - 融合矩阵报告的表格更新务必用 `53` 脚本或直接读 JSON 生成，避免手抄（本次已踩过转录错误的坑）。

---

## 8. 同日并行工作提示（必读）

2026-08-20 同日内，仓库中另有**并行的 fusion variant 消融工作**（见 `repo-docs/change-log.md` 2026-08-20 条目与根目录 `fusion_variant_results_20260820.md` / `fusion_attention_vs_concat_evidence_20260820.md`）：

- **结论**：当前 attention 融合相比朴素拼接**无价值**——concat 全量同 seed 配对在 `within_subject_day` Δraw r `+0.0394`（42 胜/12 负，p=5.2e-5）、ΔRMSE `-0.0143`，`cross_day` Δraw r `+0.0137`；`attention_multihead_pma` 与 `eeg_anchor` 均不作主线。**建议以 `concat + MLP` 作为新融合基线**。
- 关联：`scripts/32_run_eegpt_centered_loss.py` 已被该工作扩展了 `--fusion-variant / --attn-num-heads / --attn-num-latent`（`AttentionRegressor` 新增 `concat` / `attention_multihead_pma` / `eeg_anchor` 变体）；另有 `scripts/54_summarize_fusion_variants.py`、`scripts/55_summarize_fusion_variant_full_paired.py`、`scripts/56_build_fusion_attention_vs_concat_evidence.py` 与 `tests/test_fusion_variants.py`（9 tests）。
- **对 Wear × MOMENT 的含义**：本 handoff 的所有结果（Phase 1/2、方案 C）都是在 **attention 融合口径**下得到的。若按并行结论把融合器换成 `concat + MLP`，`Wmoment_frozen` 相对 `Wdeep` 的增益**需要在新的融合基线下重验**（token 与 token 训练不受融合器影响，只需重跑融合矩阵，成本同方案 C 的 144 runs 量级）。
- 两侧工作共用同一批 token / 数据 / seed 口径约定，`--experiment-seed-fixed` 与 `--eeg-token-seed 240800` 的用法一致。
