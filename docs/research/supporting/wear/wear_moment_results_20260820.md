# Wear × MOMENT 实验结果（2026-08-20）

> 执行依据：[wear_moment_experiment_plan_20260820.md](wear_moment_experiment_plan_20260820.md)。
> 结论先行：**MOMENT-1-small 冻结表征（`wear_moment_frozen_v1`）在 `cross_day` 上 3/3 seeds 一致优于 `Wdeep`（raw r +0.063、RMSE -0.039），在 `cross_subject` 上稳定更优，在 `within_subject_day` 上持平；partial FT（`wear_moment_partial_ft_v1`）仅在 `cross_subject` 稳定增益，主协议上不稳定。方案 C 全维度扩展（144 runs）进一步确认：主线 EEG × cross_day 下 frozen 相对 Wdeep 在全部 6 个 video/audio 配置上都更优。建议将 `wear_moment_frozen_v1` 纳入融合矩阵作为 Wear 新主线候选。**

---

## 1. 实验口径

- 融合配置固定：EEG = `eeg_eegpt_partial_ft_v1`（256D token，seed 240800）、Video = A1、Audio = full、loss = raw、head = regression、hidden 128、batch 256、epochs ≤80、patience 15。
- 对比路线（同配置配对）：`A1_Wphysio_full`（基线）、`A1_Wdeep_full`（基线）、`A1_Wmoment_frozen_full`（新）、`A1_Wmoment_ft_full`（新）。
- 协议：`cross_subject`（诊断）、`cross_day`、`within_subject_day`；train = pretrain+finetune，val 早停，test 冻结评估。
- 3 组 seed 配对：融合 seed 240729/240730/240731（`--experiment-seed-fixed` 保证组内严格同 seed），wear token seed 240800/240801/240802，EEG token 固定 240800。
- MOMENT-1-small：hf-mirror 下载（repo sha `411e2882`，safetensors sha256 `785e6c6f...`，145MB），`task_name="embedding"` 输出 512D，320×5 通道在前输入；frozen 档投影头 lr 1e-3；partial FT 解冻最后 2 block + final norm（encoder lr 1e-5）。
- mask 复用现有 `wear_mask`（24,127/28,819，83.7%），token 为 fatigue-supervised representation。

## 2. Phase 1 screen（seed 240729 组）

G1 通过：`wear_moment_frozen_v1` 相对 `Wdeep` 同配置，cross_day raw r +0.074 / RMSE -0.032，cross_subject raw r +0.088 / RMSE -0.051；相对 `Wphysio` 三协议全部更优。进入 Phase 2。

﻿## 3. Phase 2 confirm（3 seeds）完整结果（修正版）

| protocol | seed | route | RMSE | MAE | raw r | centered r |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| cross_subject | 240729 | Wphysio | 0.9528 | 0.7528 | 0.0409 | 0.0800 |
| cross_subject | 240729 | Wdeep | 0.9829 | 0.7879 | -0.0182 | 0.0229 |
| cross_subject | 240729 | **Wmoment_frozen** | 0.9323 | 0.7355 | 0.0695 | 0.0922 |
| cross_subject | 240729 | Wmoment_ft | 0.9303 | 0.7394 | 0.0761 | 0.0722 |
| cross_subject | 240730 | Wphysio | 1.0089 | 0.8054 | 0.0108 | 0.0540 |
| cross_subject | 240730 | Wdeep | 0.9818 | 0.7882 | 0.0074 | 0.0113 |
| cross_subject | 240730 | **Wmoment_frozen** | 0.9771 | 0.7806 | 0.0278 | 0.0628 |
| cross_subject | 240730 | Wmoment_ft | 0.9645 | 0.7656 | 0.0374 | 0.0735 |
| cross_subject | 240731 | Wphysio | 0.9989 | 0.7912 | -0.0086 | 0.0331 |
| cross_subject | 240731 | Wdeep | 1.0088 | 0.8127 | -0.0161 | 0.0017 |
| cross_subject | 240731 | **Wmoment_frozen** | 1.0009 | 0.7977 | -0.0187 | 0.0256 |
| cross_subject | 240731 | Wmoment_ft | 0.9604 | 0.7663 | 0.0170 | 0.0617 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| cross_day | 240729 | Wphysio | 0.9579 | 0.7472 | 0.2995 | 0.1051 |
| cross_day | 240729 | Wdeep | 0.9479 | 0.7388 | 0.2579 | 0.0999 |
| cross_day | 240729 | **Wmoment_frozen** | 0.9159 | 0.7139 | 0.3319 | 0.1417 |
| cross_day | 240729 | Wmoment_ft | 0.9350 | 0.7274 | 0.2841 | 0.0981 |
| cross_day | 240730 | Wphysio | 0.9583 | 0.7460 | 0.3155 | 0.1159 |
| cross_day | 240730 | Wdeep | 0.9526 | 0.7405 | 0.2299 | 0.0763 |
| cross_day | 240730 | **Wmoment_frozen** | 0.9238 | 0.7081 | 0.3147 | 0.1277 |
| cross_day | 240730 | Wmoment_ft | 0.9163 | 0.7102 | 0.3400 | 0.1352 |
| cross_day | 240731 | Wphysio | 0.9297 | 0.7296 | 0.3232 | 0.1170 |
| cross_day | 240731 | Wdeep | 0.9702 | 0.7550 | 0.2804 | 0.0934 |
| cross_day | 240731 | **Wmoment_frozen** | 0.9141 | 0.7137 | 0.3094 | 0.1247 |
| cross_day | 240731 | Wmoment_ft | 0.9376 | 0.7243 | 0.2860 | 0.1187 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| within_subject_day | 240729 | Wphysio | 0.9452 | 0.7320 | 0.3611 | 0.1641 |
| within_subject_day | 240729 | Wdeep | 0.9270 | 0.7228 | 0.3871 | 0.1882 |
| within_subject_day | 240729 | **Wmoment_frozen** | 0.9255 | 0.7153 | 0.3774 | 0.1712 |
| within_subject_day | 240729 | Wmoment_ft | 0.9213 | 0.7193 | 0.3885 | 0.1764 |
| within_subject_day | 240730 | Wphysio | 0.9206 | 0.7306 | 0.3655 | 0.1527 |
| within_subject_day | 240730 | Wdeep | 0.9256 | 0.7164 | 0.3927 | 0.1875 |
| within_subject_day | 240730 | **Wmoment_frozen** | 0.9231 | 0.7180 | 0.3914 | 0.1855 |
| within_subject_day | 240730 | Wmoment_ft | 0.9400 | 0.7301 | 0.3774 | 0.1744 |
| within_subject_day | 240731 | Wphysio | 0.9327 | 0.7237 | 0.3684 | 0.1735 |
| within_subject_day | 240731 | Wdeep | 0.9301 | 0.7240 | 0.3959 | 0.1957 |
| within_subject_day | 240731 | **Wmoment_frozen** | 0.9345 | 0.7264 | 0.3839 | 0.1952 |
| within_subject_day | 240731 | Wmoment_ft | 0.9501 | 0.7329 | 0.3682 | 0.1757 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |

## 4. 门槛判定（3-seed paired）

| gate | 判定 | 依据 |
| --- | --- | --- |
| **G1**（frozen vs Wdeep） | **通过** | cross_day：raw r Δ **+0.0626（3/3 seeds）**、RMSE Δ **-0.0390（3/3）**；cross_subject：raw r Δ +0.0352（2/3）、RMSE Δ -0.0211（3/3）；within_subject_day：raw r Δ -0.0076、RMSE Δ +0.0002（持平） |
| **G2**（ft vs frozen） | 名义通过，**实质不稳定** | cross_subject 3/3 seeds 更优（raw r +0.0173、RMSE -0.0183）；cross_day 1/3（Δ -0.0153）、within_subject_day 1/3（Δ -0.0063）——主协议上无稳定增益 |
| **G3**（3-seed 稳定） | **通过** | 主协议 cross_day 上 frozen 相对 Wdeep 的增益方向 3/3 seeds 一致 |
| 附加 | — | frozen 相对 Wphysio：三协议 raw r 均值 +0.0059/+0.0118/+0.0193（within_subject_day 3/3 更优），RMSE 均值全部更低 |

## 5. 结论与建议

1. **`wear_moment_frozen_v1` 纳入主线候选**：在最重要的跨天泛化协议（`cross_day`）上 3-seed 一致优于 `Wdeep`（raw r 0.309–0.332 vs 0.230–0.280，RMSE 全部更低），`cross_subject` 稳定更优，`within_subject_day` 持平不倒退；相对 `Wphysio` 三协议均更优。**推荐作为 Wear 新主线 route**，与 EEGPT partial FT 配对使用。
2. **`wear_moment_partial_ft_v1` 保留为诊断/可选**：只在 `cross_subject` 稳定优于 frozen；两个主协议上不稳定（cross_day 上 1/3 seeds 反而差），不建议默认启用。若后续要启用，需按协议单独判定。
3. **解释方向**：MOMENT 冻结表征带来的增益集中在"跨天/跨被试泛化"（encoder 预训练特征更稳健），而对"同被试日内"场景与 Wdeep 相当——这与 EEG 侧"预训练表征主要提升泛化"的经验一致。
4. **代价**：frozen 档额外成本 ≈ 一次 MOMENT 推理（全量 ~5 分钟）+ 每 protocol/seed 一个轻量投影头训练（~2 分钟）；token 已落盘 `/vePFS-0x0d/DailyEEG_multimodal/embeddings/wear_tokens/`，融合训练成本与现有矩阵相同。

## 5b. 方案 C 全维度扩展（144 runs，seed 240729）

按用户选定方案 C 把 moment 两档铺满全部 video/audio 维度（2 EEG × 24 routes × 3 protocols，`--experiment-seed-fixed` 固定 seed，EEG/wear token 均 240800）。全量 144 runs 完成，汇总见 `outputs/server_sync/wear_moment_20260820/planC_144/gates_planC.md`（36 配置组 delta + 144 行完整指标表）：

- **主线 EEG（`eeg_eegpt_partial_ft_v1`）× `cross_day`：frozen 相对 Wdeep 在全部 6 个 video/audio 配置上都更优**（raw r Δ +0.033 ~ +0.087，RMSE 全部更低）——A1+full 的 Phase 1/2 结论推广到 B0/A2 与 no_audio 全维度成立；`within_subject_day` 上 RMSE 6/6 配置更低、raw r 4/6 为正（持平偏略好）；`cross_subject` 三个 full 配置大幅更优（+0.062~+0.088），no_audio 混杂。
- **低监督 EEG（`eeg_eegpt_frozen_v1`）下 moment 增益不稳定**（cross_day 4/6 raw r 为正、within_subject_day 混杂）——moment 表征的增益需要较强 EEG 主线配合。
- **partial FT 全维度无一致增益**（36 配置组中方向混杂），确认不作为默认路线。
- 推荐主线组合：**`eeg_eegpt_partial_ft_v1` + `Wmoment_frozen`**，全 video/audio 配置均可选；其中 `cross_day` 最佳为 `A2_Wmoment_frozen_full`（raw r Δ +0.087 / RMSE Δ +0.020）。

## 6. 产物

| 产物 | 位置 |
| --- | --- |
| 全部 18 个 wear token（3 协议 × 2 profiles × 3 seeds） | 服务器 `/vePFS-0x0d/DailyEEG_multimodal/embeddings/wear_tokens/{protocol}/{profile}/seed_{seed}.npz` |
| token 矩阵汇总 | `outputs/server_sync/wear_moment_20260820/token_matrix_20260820.{json,md}` |
| preflight | `outputs/server_sync/wear_moment_20260820/wear_moment_preflight_20260820.{json,md}`（28,819 行 / mask 24,127 / MOMENT 10.4ms 每窗口） |
| Phase 1 screen | `outputs/server_sync/wear_moment_20260820/phase1_screen/`（12 runs + gates） |
| Phase 2 confirm | `outputs/server_sync/wear_moment_20260820/phase2_confirm/`（36 runs + gates_phase2 + phase2_full_results.md） |
| 方案 C 全维度 | `outputs/server_sync/wear_moment_20260820/planC_144/`（144 runs + gates_planC + 完整指标表） |
| 代码 | `src/daily_multimodal/training/wear_moment_matrix.py`、`scripts/window_fatigue/16_run_wear_moment_matrix.py`、`scripts/window_fatigue/53_summarize_wear_moment_gates.py`（全维度泛化版）、`tests/test_wear_moment_matrix.py`（6 tests） |

## 7. 环境副作用记录

- 服务器 `eegpt-gpu-min` 环境：numpy 2.4.6 → **1.25.2**（momentfm 0.1.4 硬性钉死），pandas 2.3.3 共存已验证可 import；transformers 4.33.3、huggingface-hub 0.24.0 新增。若后续需要恢复 numpy 2.x，`pip install "numpy==2.4.6"` 即可（momentfm 运行时路径已确认兼容 numpy 2，除未触发的 `np.Inf`）。
- 服务器无法直连 huggingface.co / hf-mirror.com 的 443 曾间歇超时；权重经 hf-mirror.com 下载，已验证 sha256。
