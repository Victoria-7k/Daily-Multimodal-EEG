# Fusion Variant 消融 Handoff 2026-08-20

> 本文档交接本轮「cross-attention 融合是否有价值」的完整工作：改了什么代码、用了哪些脚本、结果在哪里、以及下一个工作者接手需要注意的口径和坑。

## 0. 一句话结论

**在当前 28,819 窗口 EEG-aligned 四模态疲劳预测主线上，`AttentionRegressor`（单头 self-attention + learnable-query pooling，即 `technical_route_20260814.md` 里的 cross-attention 融合）相比朴素 `concat + MLP` 不提供任何预测价值，多数配置下反而更差。建议以 `concat + MLP` 替换当前 attention 融合作为新基线，raw r 主线资源转向提升单模态表征质量。**

完整证据：`fusion_attention_vs_concat_evidence_20260820.md`（根目录，含完整 180 行附表）。

## 1. 背景与目标

- 原主线融合器：`scripts/32_run_eegpt_centered_loss.py` 的 `AttentionRegressor`（`technical_route_20260814.md` §Cross-Attention 描述的那个）。
- 本轮要回答三个问题：
  1. attention 是否比朴素拼接好（实验 2，`concat`）；
  2. 把 attention 做厚是否有效（实验 3，`attention_multihead_pma`）；
  3. EEG 当锚点是否优于平权 self-attention（实验 4，`eeg_anchor`）。
- 实现方式：在**同一个** `AttentionRegressor` 内新增 `variant` 分支，只替换 `encode()`，head/损失/评估/归一化链路完全复用，保证对比干净。

## 2. 核心结论（证据摘要，全量矩阵已确认）

全量 180-run 配对（`concat`/`eeg_anchor` vs 归档 attention，**逐 run 同 seed**，paired 180/180）：

| protocol | concat Δraw r | concat ΔRMSE | concat 胜/负/平 | 符号检验 p |
| --- | ---: | ---: | ---: | ---: |
| `cross_subject` | −0.0013（持平） | +0.0060 | 22/33/5 | 0.18 |
| `cross_day` | +0.0137 | −0.0058 | 27/28/5 | 1.00 |
| `within_subject_day` | **+0.0394** | **−0.0143** | **42/12/6** | **5.2e-5** |

- 唯一 `attention` 略优的分支是 `eegpt_partial_ft`（Δraw r −0.008，接近噪声）；其余 4/5 条 EEG 分支 concat 全部更优（`cbramod_partial_ft` +0.033 最明显）。
- `attention_multihead_pma`（4 头 + 8 latent query）不通过：决策切片两协议 raw r 均低于 concat。
- `eeg_anchor` 部分信号（cross_subject raw r +0.012、cross_day RMSE −0.008），但 within_subject_day raw r −0.012，无一致优势，不作主线。
- 判定阈值：`|Δraw r| ≤ 0.005` 计为平、`> +0.01` 计为有意义提升。

## 3. 代码改动清单（详细）

### 3.1 `scripts/32_run_eegpt_centered_loss.py`（修改）

- `AttentionRegressor.__init__` 新增 `variant: str = "attention"`、`num_heads: int = 4`、`num_latent: int = 8`，带默认值（**向后兼容**，现有测试不受影响）；加校验：非法 variant 报错、pma 要求 `hidden_dim % num_heads == 0`、eeg_anchor 要求 `modality_count >= 1`。
- 按 variant 构建子模块：`concat` 加 `concat_projection = Linear(256*M + M, hidden_dim)`；`attention_multihead_pma` 加 `latent_queries (1, num_latent, hidden_dim)` + 多头 `cross_attention`；`eeg_anchor` 复用 `self_attention`。
- `encode()` 拆 4 分支，**默认 `attention` 路径逐字节保持原实现**（保证基线可复现）：
  - `concat`：`tokens * mask` 零掩码 → flatten → 拼接 `mask` 指示位 → 投影；
  - `attention_multihead_pma`：投影+modality embedding → 置零缺失 → latent query 对 token 做多头 cross-attention（`key_padding_mask=~mask`）→ mean-pool；
  - `eeg_anchor`：query = `x[:, :1]`（EEG），key/value = 全部 token → 取 EEG 输出；
  - `attention`：原逻辑不动。
- `_fit_model` 新增 `fusion_variant / attn_num_heads / attn_num_latent` 透传。
- `main()` 新增 `--fusion-variant`（choices）、`--attn-num-heads`（默认 4）、`--attn-num-latent`（默认 8）；`result` 字典与 `runtime` 块记录 `fusion_variant / attn_num_heads / attn_num_latent`；`_write_markdown` 表格加 `fusion` 列。

### 3.2 新增脚本

| 脚本 | 用途 |
| --- | --- |
| `tests/test_fusion_variants.py` | 9 个单测：四变体标量输出、全 head 兼容、变长 modality、mask 不变性、mask 生效、确定性、非法 variant 拒绝、pma 整除守卫、`_fit_model` 透传 |
| `scripts/54_summarize_fusion_variants.py` | 汇总决策切片多份 report JSON → 协议×变体均值表 + 逐 run 明细（可选归档参考列） |
| `scripts/55_summarize_fusion_variant_full_paired.py` | 全量矩阵 vs 归档 attention 按 `(protocol, experiment, eeg_branch, seed)` 逐 run 配对 → Δraw r/Δcentered r/ΔRMSE、胜/负/平、符号检验 p |
| `scripts/56_build_fusion_attention_vs_concat_evidence.py` | 从真实 JSON 生成证据文档 `fusion_attention_vs_concat_evidence_20260820.md`（零手工转录） |

### 3.3 文档改动

- `scripts/README.md`：`32` 行说明 `--fusion-variant`；新增 `54/55/56` 三行。
- `repo-docs/references/commands-and-artifacts.md`：新增「Fusion variant 决策切片」行（含汇总/全量配对命令）。
- `repo-docs/change-log.md`：新增两条（实现+执行、证据文档）。
- 根目录新增：`fusion_variant_results_20260820.md`（中间+最终结论）、`fusion_attention_vs_concat_evidence_20260820.md`（证据文档）。

## 4. 结果文件路径

### 4.1 本地（`outputs/server_sync/fusion_variant_20260820/`）

| 文件 | 内容 |
| --- | --- |
| `attention.json/.md`、`concat.json/.md`、`attention_multihead_pma.json/.md`、`eeg_anchor.json/.md` | 决策切片，每变体 8 runs（seed 240800 固定） |
| `summary.json` / `summary.md` | 决策切片汇总（协议×变体均值 + 逐 run 明细） |
| `fusion_variant_concat_full_seed240800_raw.json` | concat 全量 180 runs |
| `fusion_variant_eeg_anchor_full_seed240800_raw.json` | eeg_anchor 全量 180 runs |
| `fusion_variant_concat_full_paired.json/.md` | concat vs 归档 attention 配对结果 |
| `fusion_variant_eeg_anchor_full_paired.json/.md` | eeg_anchor vs 归档 attention 配对结果 |
| `cross_day_B0_Wphysio_full_seed240790.json`、`within_subject_day_B0_Wphysio_no_audio_seed240855.json`、`cross_subject_B0_Wphysio_no_audio_seed240735.json` | 3 条归档同 seed 定点复现（逐位命中） |
| `fusion_variant_decision.log`、`fusion_variant_full.log` | 服务器运行日志 |
| `sync_fv.tgz`、`sync_fv2.tgz` | 传输归档（可忽略） |

### 4.2 归档参照（本轮未改）

- `outputs/server_sync/eeg_encoder_256d_5route_20260814/reports/eeg_encoder_256d_5route_fusion_video_only_seed240800_raw.json`（0814 attention 180-run 矩阵）

### 4.3 根目录文档

- `fusion_attention_vs_concat_evidence_20260820.md`（证据文档，431 行）
- `fusion_variant_results_20260820.md`（结果速览）
- `outputs/reports/fusion_attention_vs_concat_evidence_20260820.json`（证据文档全表 JSON 副本）

### 4.4 服务器（`/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/`）

- `outputs/reports/fusion_variant_decision_seed240800/`（四变体 json/md + summary）
- `outputs/reports/fusion_variant_repro/`（3 条定点复现 json/md）
- `outputs/reports/fusion_variant_{concat,eeg_anchor}_full_seed240800_raw.{json,md}`
- `outputs/reports/fusion_variant_{concat,eeg_anchor}_full_paired.{json,md}`
- `outputs/logs/fusion_variant_{decision,full}.log`
- `/tmp/fusion_variant_{decision,repro,full}.sh`（临时脚本，可忽略）

## 5. 环境与访问

- **服务器 SSH**：`~/.ssh/config` 里的 `huoshan_TriDim`（`124.174.8.252:36083`，user `wangziwei`，key `id_ed25519`）。GPU：NVIDIA H20（单卡）。
- **服务器仓库**：`/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned`（无 git，靠 scp 同步文件）。
- **运行时**：`runtime/envs/eegpt-gpu-min/bin/python`，需 `PYTHONPATH=src`。
- **数据**：embedding `/vePFS-0x0d/DailyEEG_multimodal/embeddings`，split `/vePFS-0x0d/DailyEEG/splits_new`，root `--root /vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned`。
- **本地**：Windows；测试用 Python 3.12（`C:\Users\28303\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`），CPU torch 2.13.0 装在 `C:\torch`（避开 Windows 路径过长 WinError 206），跑测试时 `PYTHONPATH='src;C:\torch'`。**真实实验只能在服务器跑**（本地无 embedding 数据）。

## 6. 实验口径与关键 gotcha

- **固定口径**：`--eeg-token-root eeg_encoder_256d_tokens --eeg-token-seed 240800 --loss-modes raw --no-raw-baseline --subject-balanced-batches`，epochs 80 / patience 15 / hidden 128 / lr 1e-3 / wd 1e-4 / dropout 0.1 / batch 256。
- **种子**：默认 `--seed 240729` 且 `run_seed = 240729 + run_number`；`--experiment-seed-fixed` 使所有 run 用同一 seed（配对对比时用）；`--eeg-token-seed`（240800）是 EEG token 文件名里的 seed，与训练 seed 无关。
- ⚠️ **重要 gotcha**：当前脚本的 `EXPERIMENT_BRANCHES` 已含 wear_moment 分支，因此 `--experiment-set video_only` 现在会展开 **24 个**（不是 0814 时的 12 个）融合组合。要复现 0814 的 **180-run seed 序列**，必须用 `--experiment-set custom` 显式列出 36 个 `protocol:experiment`（12 组合 × 3 协议），顺序保持 `B0_Wphysio_full, B0_Wphysio_no_audio, B0_Wdeep_full, B0_Wdeep_no_audio, A1_Wphysio_full, A1_Wphysio_no_audio, A1_Wdeep_full, A1_Wdeep_no_audio, A2_Wphysio_full, A2_Wphysio_no_audio, A2_Wdeep_full, A2_Wdeep_no_audio`（见 §8 完整命令）。
- **确定性**：定点复现证明固定 seed 下结果可逐位复现（3/3 命中），所以同 seed 配对是可靠的。
- **指标**：test 的 `raw_r`、`within_subject_centered_r`、`rmse`（JSON 里 `test` 字段；`_metric_aliases` 有 `pearson_r`/`per_subject_r_mean` 别名）。

## 7. 已完成的实验（核对清单）

- [x] 实现 4 个 fusion variant + CLI（`scripts/32`）
- [x] 单测：本地 19 + 全量 151 tests，服务器 19 tests，compileall 全绿
- [x] 基线验证：3 条归档同 seed 定点复现逐位命中（cross_day `0.9481/0.3504/0.1176`、within_subject_day `0.9243/0.4049/0.2062`、cross_subject `0.9624/0.0841/0.0865`）
- [x] 决策切片：32 runs（4 变体 × 8），全部 rc=0
- [x] 全量配对矩阵：concat 180 runs + eeg_anchor 180 runs，全部 rc=0，paired 180/180
- [x] 证据文档 + 结果文档 + repo-docs 同步，validator 0 errors

## 8. 复现命令（服务器直接可跑）

**决策切片（单变体，把 `<variant>` 换成 attention/concat/attention_multihead_pma/eeg_anchor）：**

```bash
cd /vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned
PYTHONPATH=src runtime/envs/eegpt-gpu-min/bin/python scripts/32_run_eegpt_centered_loss.py \
  --root /vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned \
  --embeddings-root /vePFS-0x0d/DailyEEG_multimodal/embeddings \
  --splits-root /vePFS-0x0d/DailyEEG/splits_new \
  --experiment-set custom --protocols cross_day,within_subject_day \
  --experiments cross_day:B0_Wphysio_full,cross_day:B0_Wphysio_no_audio,cross_day:A1_Wdeep_full,cross_day:A1_Wdeep_no_audio,within_subject_day:B0_Wphysio_full,within_subject_day:B0_Wphysio_no_audio,within_subject_day:A1_Wdeep_full,within_subject_day:A1_Wdeep_no_audio \
  --eeg-branches eeg_eegpt_partial_ft_v1 \
  --eeg-token-root eeg_encoder_256d_tokens --eeg-token-seed 240800 \
  --loss-modes raw --no-raw-baseline --subject-balanced-batches \
  --experiment-seed-fixed --seed 240800 \
  --fusion-variant <variant> \
  --out-json outputs/reports/fusion_variant_decision_seed240800/<variant>.json \
  --out-md outputs/reports/fusion_variant_decision_seed240800/<variant>.md
```

**全量配对矩阵（concat，180 runs；与归档 attention 同 seed 序列）：**

```bash
cd /vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned
EXPS="cross_subject:B0_Wphysio_full,cross_subject:B0_Wphysio_no_audio,cross_subject:B0_Wdeep_full,cross_subject:B0_Wdeep_no_audio,cross_subject:A1_Wphysio_full,cross_subject:A1_Wphysio_no_audio,cross_subject:A1_Wdeep_full,cross_subject:A1_Wdeep_no_audio,cross_subject:A2_Wphysio_full,cross_subject:A2_Wphysio_no_audio,cross_subject:A2_Wdeep_full,cross_subject:A2_Wdeep_no_audio,cross_day:B0_Wphysio_full,cross_day:B0_Wphysio_no_audio,cross_day:B0_Wdeep_full,cross_day:B0_Wdeep_no_audio,cross_day:A1_Wphysio_full,cross_day:A1_Wphysio_no_audio,cross_day:A1_Wdeep_full,cross_day:A1_Wdeep_no_audio,cross_day:A2_Wphysio_full,cross_day:A2_Wphysio_no_audio,cross_day:A2_Wdeep_full,cross_day:A2_Wdeep_no_audio,within_subject_day:B0_Wphysio_full,within_subject_day:B0_Wphysio_no_audio,within_subject_day:B0_Wdeep_full,within_subject_day:B0_Wdeep_no_audio,within_subject_day:A1_Wphysio_full,within_subject_day:A1_Wphysio_no_audio,within_subject_day:A1_Wdeep_full,within_subject_day:A1_Wdeep_no_audio,within_subject_day:A2_Wphysio_full,within_subject_day:A2_Wphysio_no_audio,within_subject_day:A2_Wdeep_full,within_subject_day:A2_Wdeep_no_audio"
PYTHONPATH=src runtime/envs/eegpt-gpu-min/bin/python scripts/32_run_eegpt_centered_loss.py \
  --root /vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned \
  --embeddings-root /vePFS-0x0d/DailyEEG_multimodal/embeddings \
  --splits-root /vePFS-0x0d/DailyEEG/splits_new \
  --experiment-set custom --protocols cross_subject,cross_day,within_subject_day \
  --experiments "$EXPS" \
  --eeg-branches eeg_eegpt_frozen_v1,eeg_eegpt_partial_ft_v1,eeg_cbramod_frozen_v1,eeg_cbramod_partial_ft_v1,eeg_de_5band_1s_avg_v1 \
  --eeg-token-root eeg_encoder_256d_tokens --eeg-token-seed 240800 \
  --loss-modes raw --no-raw-baseline --subject-balanced-batches \
  --fusion-variant concat \
  --out-json outputs/reports/fusion_variant_concat_full_seed240800_raw.json \
  --out-md outputs/reports/fusion_variant_concat_full_seed240800_raw.md
```

**全量配对汇总：**

```bash
PYTHONPATH=src runtime/envs/eegpt-gpu-min/bin/python scripts/55_summarize_fusion_variant_full_paired.py \
  --archive-json outputs/reports/eeg_encoder_256d_5route_fusion_video_only_seed240800_raw.json \
  --variant-jsons outputs/reports/fusion_variant_concat_full_seed240800_raw.json
```

## 9. 下一步建议

1. **把默认融合切到 `concat`**：`--fusion-variant concat` 已可用，可先把新主线报告用 concat 重跑一遍（或直接以现有 `fusion_variant_concat_full_seed240800_raw.json` 作为新基线）。
2. **资源转向单模态表征**（这才是瓶颈）：wear（Wphysio/Wdeep 无监督、固定随机投影）、video（DINOv2 frozen）做轻量 fatigue 监督对齐，比继续折腾融合结构收益大。
3. 若要验证「强表征下 attention 是否回本」：在最强分支 `eegpt_partial_ft` 上继续观察（本轮唯一 attention 不亏的分支）。
4. 更新 `technical_route_20260814.md` 的 Cross-Attention 小节与根 README，把「modality-token cross-attention 融合」改为「concat + MLP 基线」，并引用证据文档。

## 10. 新对话第一条执行指令建议

```text
请读取 fusion_variant_ablation_handoff_20260820.md、fusion_attention_vs_concat_evidence_20260820.md 和 technical_route_20260814.md。当前结论是 modality-token attention 融合不优于朴素 concat（全量 180-run 同 seed 配对已确认），下一步请：1) 把默认融合基线切到 --fusion-variant concat（脚本 32 已支持）；2) 把工作重点转到提升单模态表征质量（尤其 wear/video 的疲劳监督对齐）；3) 同步更新 technical_route 与 README 的融合描述。实验在服务器 huoshan_TriDim 跑，runtime 用 runtime/envs/eegpt-gpu-min/bin/python，注意 --experiment-set video_only 现在会展开 24 个组合（含 wear_moment），要复现 0814 的 180-run seed 序列需用 --experiment-set custom 显式 36 项。
```
