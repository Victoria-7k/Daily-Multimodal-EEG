# Current Script Entry Points

当前 `scripts/` 按研究路线分组；`archive_legacy/` 保留历史工程线、旧口径和已归档诊断。新增或迁移脚本时优先放入所属路线目录，并同步本索引、对应子目录 README 与 `repo-docs/`。

## 分组索引

| 目录 | 边界 | 入口 |
| --- | --- | --- |
| `embeddings/` | 生成或评估窗口级 EEG/Wear/Video/Audio 256D token。 | `12`、`15`、`27`、`34` |
| `window_fatigue/` | 当前 EEG-aligned 10 秒窗口级连续 `fatigue` 回归主线、Wear x MOMENT 挂载和 fusion variant 证据。 | `16`、`32`、`53`-`56` |
| `calibration/` | 窗口级 fatigue prediction 的 Phase 1-4 calibration / loss / head / trend 诊断。 | `39`-`47`、`49`-`52` |
| `multilabel/` | fixed/frozen token 的 11-label supervised fusion baseline，独立于 fatigue calibration。 | `48` |
| `eql_caf/` | EQL-CAF 5 x 2s temporal-token 探索线；当前 `59`-`62` 是 repeat-token smoke，不作为真实时序证据。 | `57`-`64` |
| `wear_only_fm/` | Wear-only foundation-model 表征路线、W3FM formal gate、post-gate 诊断和 ACC 替换筛查。 | `65`-`72`、`89` |
| `daily_affect/` | daily-affect EMA-bag ordinal route、focused diagnostics、window-daily event bridge 和 expected-score Huber screen。 | `73`-`88` |

## 当前复现实验顺序

1. 如需重新生成非 EEG 模态 embedding，先运行 `scripts/embeddings/12_extract_audio_embeddings.py`、`scripts/embeddings/15_extract_wear_embeddings.py`、`scripts/embeddings/27_extract_dinov2_roi_embeddings.py`；EEG token 矩阵用 `scripts/embeddings/34_run_eeg_encoder_matrix.py`。
2. 窗口级融合主线用 `scripts/window_fatigue/32_run_eegpt_centered_loss.py --experiment-set video_only --eeg-token-root eeg_encoder_256d_tokens`；Wear x MOMENT token 用 `scripts/window_fatigue/16_run_wear_moment_matrix.py`，门槛汇总用 `scripts/window_fatigue/53_summarize_wear_moment_gates.py`。
3. Fusion variant 证据用 `scripts/window_fatigue/54_summarize_fusion_variants.py`、`scripts/window_fatigue/55_summarize_fusion_variant_full_paired.py`、`scripts/window_fatigue/56_build_fusion_attention_vs_concat_evidence.py`，均围绕 `32` 的同一路窗口级训练入口。
4. Calibration 诊断按 `scripts/calibration/39_run_prediction_calibration_phase1.py` -> `40` -> `41`；Phase 2 小矩阵按 `42 --execute` 后用 `43` 汇总，收窄候选再按 `44 --execute` 和 `45` 做 3-seed paired gate；Phase 3 head screen 用 `46 --execute` 和 `47`；总览/趋势图用 `49`、`50`；Phase 4 next-three 小筛查用 `51`、`52`。
5. 多标签 fixed-token 基线用 `scripts/multilabel/48_run_fixed_tokens_multilabel_fusion.py`；它不复用 `39`-`47` calibration 脚本，也不把 fatigue-supervised EEG token 当作 label-free baseline。
6. EQL-CAF 路线先用 `scripts/eql_caf/57_run_eql_caf_phase0_baselines.py` 复现当前四 token baseline，再用 `58` 生成 5 x 2s temporal index；`59`-`62` 当前只提供 `global_repeat_smoke_v1` 管线验证产物；随后用 `63` 合包，并用 `64` 跑 B1/B2/M1-M5 fatigue 矩阵。
7. Wear-only FM 路线先用 `scripts/wear_only_fm/65_run_wear_only_fm_phase1.py --mode stage` 生成 staged inputs，再用 `--mode smoke` 和 `--mode batch` 缓存外部 frozen encoder embeddings；Phase 2/3 分别用 `66`、`67`、`68`，post-gate 诊断用 `69`-`72`，ACC handcrafted 替换筛查用 `89`。
8. Daily-affect ordinal route 固定从 canonical 28,819 个 EEG-aligned 10 秒窗口构建每 event 23 窗口 EMA bags。主流程为 `scripts/daily_affect/73_build_daily_affect_bags.py`、`74`、`75`、`76`、`77`；focused line 用 `78`-`83`；window-daily event bridge 用 `84`-`87`；expected-score Huber 辅助损失筛查用 `88`。

## 维护规则

- 保留编号前缀，避免旧报告和远端运行记录失去可追踪性。
- 迁移或新增入口后，更新脚本内硬编码命令、根目录 README、`repo-docs/README.md`、`repo-docs/references/commands-and-artifacts.md` 和相关路线文档。
- `daily_affect/`、`eql_caf/`、`wear_only_fm/` 与窗口主线的输入、mask、协议和结论必须保持分开。
