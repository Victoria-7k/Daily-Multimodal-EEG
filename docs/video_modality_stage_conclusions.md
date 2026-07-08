# 视频模态阶段性实验结论

本文档整理当前视频模态阶段性结果，只保留原始指标和简短解读。

## 1. V4a Probe 诊断

| Probe / Split | Metric |
| --- | ---: |
| Subject Probe accuracy | `0.9777` |
| Session Probe accuracy | `0.9691` |
| Fatigue Ridge LOSO RMSE | `1.0333` |
| Fatigue Ridge LOSO Pearson r | `-0.0460` |
| Fatigue Ridge S1 RMSE | `0.9188` |
| Fatigue Ridge S1 Pearson r | `0.2746` |
| Fatigue Ridge S4 RMSE | `0.9620` |
| Fatigue Ridge S4 Pearson r | `0.1702` |
| Fatigue Ridge S2 RMSE | `1.0172` |
| Fatigue Ridge S2 Pearson r | `0.2254` |

简单解读：V4a embedding 中 subject/session 信息很强。S1 明显好于 LOSO，说明同被试场景下有可用 fatigue 信号，但跨被试泛化弱，存在明显 subject/session shortcut。

## 2. V4b 时序建模总体指标

`±` 表示 across folds 的标准差。S2 只有 1 个 chronological fold，因此 `±0.0000` 不代表稳定误差估计。

| Split | Model | RMSE | Pearson r | pred_std | truth_std |
| --- | --- | ---: | ---: | ---: | ---: |
| LOSO | V4a pooling | `1.0083 ± 0.1832` | `-0.0517 ± 0.1580` | `0.4255 ± 0.0361` | `0.7736 ± 0.1586` |
| LOSO | V4b-TCN | `0.9822 ± 0.1705` | `-0.0217 ± 0.1546` | `0.4143 ± 0.0417` | `0.7736 ± 0.1586` |
| LOSO | V4b-Transformer | `1.0063 ± 0.1727` | `-0.0427 ± 0.1385` | `0.4211 ± 0.0644` | `0.7736 ± 0.1586` |
| S1 | V4a pooling | `0.8801 ± 0.0448` | `0.3258 ± 0.0370` | `0.5701 ± 0.0407` | `0.8745 ± 0.0635` |
| S1 | V4b-TCN | `0.8703 ± 0.0335` | `0.3192 ± 0.0588` | `0.5375 ± 0.0097` | `0.8745 ± 0.0635` |
| S1 | V4b-Transformer | `0.8678 ± 0.0419` | `0.3275 ± 0.0348` | `0.5368 ± 0.0196` | `0.8745 ± 0.0635` |
| S4 | V4a pooling | `0.9477 ± 0.0555` | `0.1788 ± 0.0815` | `0.4972 ± 0.0460` | `0.8984 ± 0.0517` |
| S4 | V4b-TCN | `0.9536 ± 0.0677` | `0.1812 ± 0.1052` | `0.5105 ± 0.0446` | `0.8984 ± 0.0517` |
| S4 | V4b-Transformer | `0.9370 ± 0.0170` | `0.1962 ± 0.0724` | `0.4865 ± 0.0569` | `0.8984 ± 0.0517` |
| S2 | V4a pooling | `0.9703 ± 0.0000` | `0.2272 ± 0.0000` | `0.5573 ± 0.0000` | `0.9256 ± 0.0000` |
| S2 | V4b-TCN | `1.0197 ± 0.0000` | `0.0634 ± 0.0000` | `0.4791 ± 0.0000` | `0.9256 ± 0.0000` |
| S2 | V4b-Transformer | `0.9882 ± 0.0000` | `0.1706 ± 0.0000` | `0.5062 ± 0.0000` | `0.9256 ± 0.0000` |

简单解读：V4b 的收益有限且不稳定。TCN 在 LOSO RMSE 上略好，但 r 仍接近 0；Transformer 在 S1/S4 略好；两个 V4b 变体在 S2 都弱于 V4a。

## 3. LOSO Per-Subject Pearson r

| Subject | n | V4a r | V4b-TCN r | V4b-Transformer r |
| --- | ---: | ---: | ---: | ---: |
| sub-02 | 588 | `0.0589` | `0.0387` | `0.0403` |
| sub-03 | 396 | `0.0389` | `-0.2366` | `0.1115` |
| sub-04 | 372 | `-0.0353` | `0.1351` | `-0.0530` |
| sub-05 | 708 | `0.0632` | `0.1897` | `-0.0881` |
| sub-06 | 708 | `-0.1800` | `-0.2187` | `-0.1260` |
| sub-07 | 588 | `0.0697` | `-0.0435` | `-0.0696` |
| sub-08 | 756 | `-0.1798` | `0.0603` | `-0.1958` |
| sub-09 | 792 | `-0.1805` | `0.0790` | `0.0543` |
| sub-10 | 1080 | `0.1669` | `0.1425` | `0.1706` |
| sub-11 | 696 | `-0.2392` | `-0.2210` | `-0.1810` |
| sub-12 | 252 | `0.2621` | `0.0705` | `0.2018` |
| sub-13 | 288 | `-0.1051` | `0.1078` | `-0.0187` |
| sub-14 | 756 | `-0.2643` | `-0.2251` | `-0.2760` |
| sub-15 | 348 | `-0.1994` | `-0.1827` | `-0.1677` |

简单解读：per-subject r 波动很大，多个 subject 为负相关。V4b 没有稳定提升 per-subject r，也没有解决 LOSO 泛化。

## 4. ROI 输入区域结果

| Split | R1 2xROI RMSE / r | R2 upper-body RMSE / r | R3 full-frame RMSE / r |
| --- | ---: | ---: | ---: |
| LOSO | `1.0083 / -0.0517` | `1.0041 / -0.0174` | `1.0697 / -0.0222` |
| S1 | `0.8801 / 0.3258` | `0.8684 / 0.3255` | `0.8725 / 0.3951` |
| S4 | `0.9477 / 0.1788` | `0.9285 / 0.1842` | `0.9425 / 0.2560` |
| S2 | `0.9703 / 0.2272` | `1.0049 / 0.1435` | `1.0076 / 0.2132` |

简单解读：upper-body 在 LOSO/S1/S4 的 RMSE 略好，但 S2 明显弱于 2xROI，因此当前默认仍保留 R1 / 2x face ROI。full-frame 在 S1/S4 的 r 较高，但 LOSO/S2 RMSE 更差，不适合作为默认输入。

## 5. V4d Augmentation 结论更新

### 5.1 已作废的旧 A1/A2 判断

旧 A1/A2 全量 projected artifacts 是在 projection salt 修复前生成的。由于 A0/A1/A2 可能使用了不同随机投影矩阵，旧 paired embedding audit 中的结果：

- A0 vs A1 cosine mean 约 `-0.0189`
- A0 vs A2 cosine mean 约 `0.0196`
- L2 接近 `sqrt(2)`

不能解释为 augmentation 本身过强。旧 A0-A2 downstream/probe 结果也不应继续作为结论引用。

旧文件已归档到：

`outputs/archive/invalid_projection_salt_20260707_0125/`

### 5.2 Fixed-salt A1/A2 paired audit

projection salt 修复后，A1/A2 与 A0 共用 upper-body V4a projection salt：

`video_v4a_dinov2_upper_body_mean_std_max`

新的 paired audit 输出：

`outputs/reports/video_v4d_fixed_salt/paired_embedding_audit/`

| Pair | Cosine mean | Cosine median | L2 mean | Relative L2 |
| --- | ---: | ---: | ---: | ---: |
| A0 vs A1 | `0.9990` | `0.9993` | `0.0408` | `0.0408` |
| A0 vs A2 | `0.9938` | `0.9947` | `0.1076` | `0.1076` |

解读：A1/A2 在 fixed projection 下只是轻到中等扰动，不是正交级别的分布错位。因此，“A1/A2 把 fatigue 信号打掉”的旧结论需要撤回；下游判断必须使用 fixed-salt A1/A2 重跑结果。

### 5.3 Fixed-salt A0/A1/A2 train-only probe + downstream rerun

严格评估口径：validation/test 始终使用 deterministic A0 upper-body embedding；A1/A2 只作为 train-fold override。输出位置：

- downstream reports: `outputs/reports/video_variants/v4d_a0_a2_fixed_salt_train_only/`
- probe reports: `outputs/reports/video_probes/v4d_a0_a2_fixed_salt_train_only/`
- summary: `outputs/reports/video_v4d_fixed_salt/a0_a2_probe_eval_summary.{json,md}`

| Variant | Subject Probe | Session Probe | LOSO r | S1 r | S4 r | S2 r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | `0.9583` | `0.9525` | `-0.0150` | `0.3357` | `0.1509` | `0.1241` |
| A1 | `0.9583` | `0.9523` | `-0.0198` | `0.3254` | `0.1823` | `0.1387` |
| A2 | `0.9568` | `0.9520` | `-0.0356` | `0.3250` | `0.1634` | `0.1556` |

解读：A1/A2 并没有明显降低 subject/session probe，说明 train-only appearance augmentation 还没有真正削弱 shortcut。A1 对 S4 有提升，A2 对 S2 有提升，但二者都降低 S1，且 LOSO 更差。因此 fixed-salt A1/A2 不能作为 V4d 成功方案，只能作为后续 mixed original/augmented 或 embedding interpolation 的参考。

### 5.4 Adapter + GRL screening

第一轮筛选使用 PyTorch adapter/GRL 入口，30 epochs，`adapter_dim=64`，`hidden_dim=32`。B5 只使用 A1 train-fold override；validation/test 仍使用 deterministic A0。Subject/Session Probe 来自 LOSO out-of-fold representation。

输出位置：

- runner: `outputs/reports/video_grl_adapter/run_b0_b5_grl_adapter.sh`
- summary: `outputs/reports/video_grl_adapter/b0_b5_fixed_salt/b0_b5_grl_summary.{json,md}`
- split reports: `outputs/reports/video_grl_adapter/b0_b5_fixed_salt/{loso,s1,s4,s2}_metrics.{json,md}`

| Variant | lambda | Subject Probe | Session Probe | LOSO r | S1 r | S4 r | S2 r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | `0.0000` | `0.9739` | `0.9525` | `-0.0392` | `0.3316` | `0.1546` | `0.1824` |
| B1 | `0.0000` | `0.8707` | `0.8869` | `-0.0005` | `0.3897` | `0.1821` | `0.2026` |
| B2_lam0.001 | `0.0010` | `0.8570` | `0.8863` | `-0.0107` | `0.3783` | `0.1709` | `0.2018` |
| B2_lam0.005 | `0.0050` | `0.8666` | `0.8832` | `0.0143` | `0.3783` | `0.1680` | `0.2380` |
| B2_lam0.01 | `0.0100` | `0.8695` | `0.8855` | `-0.0010` | `0.3916` | `0.2328` | `0.1158` |
| B2_lam0.05 | `0.0500` | `0.8667` | `0.8883` | `0.0034` | `0.3463` | `0.1922` | `0.1184` |
| B3_lam0.001 | `0.0010` | `0.8732` | `0.8839` | `0.0030` | `0.3713` | `0.1503` | `0.1361` |
| B3_lam0.005 | `0.0050` | `0.8863` | `0.8874` | `-0.0022` | `0.3900` | `0.2017` | `0.1821` |
| B3_lam0.01 | `0.0100` | `0.8703` | `0.8824` | `-0.0192` | `0.3813` | `0.1584` | `0.1072` |
| B3_lam0.05 | `0.0500` | `0.8686` | `0.8873` | `0.0203` | `0.3873` | `0.1839` | `0.1431` |
| B4_lam0.001 | `0.0010` | `0.8738` | `0.8788` | `-0.0045` | `0.3624` | `0.1522` | `0.1386` |
| B4_lam0.005 | `0.0050` | `0.8562` | `0.8809` | `0.0253` | `0.3639` | `0.1622` | `0.1593` |
| B4_lam0.01 | `0.0100` | `0.8611` | `0.8859` | `0.0107` | `0.3706` | `0.1641` | `0.1313` |
| B4_lam0.05 | `0.0500` | `0.8700` | `0.8811` | `-0.0140` | `0.3829` | `0.1750` | `0.1664` |
| B5_A1_lam0.001 | `0.0010` | `0.8833` | `0.8914` | `-0.0420` | `0.3590` | `0.1760` | `0.1372` |
| B5_A1_lam0.005 | `0.0050` | `0.8504` | `0.8864` | `-0.0123` | `0.3600` | `0.1544` | `0.1685` |
| B5_A1_lam0.01 | `0.0100` | `0.8755` | `0.8952` | `0.0025` | `0.3863` | `0.1804` | `0.1481` |

解读：adapter 本身已经大幅降低 subject/session probe，并提升 S1/S4/S2；B1 是当前最稳的低复杂度候选。GRL 的额外收益不稳定，但 B2_lam0.005、B3_lam0.05、B4_lam0.005 在 LOSO r 上相对 B1 更好，适合二轮用更多 epochs/seed 复查。B5_A1 没有超过 B1，说明当前 A1 train-time augmentation + GRL 不是第一优先。

## 6. 当前简要结论

- V4a 是当前可用的视频深度视觉 baseline，但 subject/session shortcut 很强。
- V4b 时序建模收益有限，没有解决 LOSO/S2 泛化问题。
- ROI 对比后，默认输入仍保留 `2x face ROI`；`upper-body ROI` 保留为候选 ablation；`full-frame` 仅作为对照。
- V4d fixed-salt train-only A1/A2 已重跑；它们几乎没有降低 subject/session shortcut，且 LOSO/S1 不如 A0，只在 S4/S2 有局部提升。
- Adapter 是当前最值得推进的 V4d 方向；B1 已明显降低 subject/session probe 并提升 S1/S4/S2。
- 2026-07-07 B1/B2 repeat 后，B1 冻结为当前视频主模型；下一阶段暂停继续扫 GRL、appearance augmentation 和 10 秒 frame-level temporal encoder，转向事件时间尺度与个体化诊断。

## 7. V4d B1/B2 Repeat Stability And Adapter-Z Audit

New rerun compares `B1`, `B2_lam0.005`, and `B2_lam0.01` with deterministic A0 upper-body eval embeddings, 5 seeds, `epochs=30`, `adapter_dim=64`, and `hidden_dim=32`.

- repeat runner: `outputs/reports/video_grl_adapter/run_b1_b2_repeat_and_repr_audit.sh`
- repeat summary: `outputs/reports/video_grl_adapter/b1_b2_repeat_stability/repeat_stability_summary.{json,md}`
- per-seed reports: `outputs/reports/video_grl_adapter/b1_b2_repeat_stability/seed_{41..45}/{loso,s4,s2}_metrics.{json,md}`
- OOF adapter representation bundle: `outputs/reports/video_grl_adapter/representation_audit/b0_b1_b2_loso_representations.npz`
- adapter-z audit: `outputs/reports/video_grl_adapter/representation_audit/b0_b1_b2_representation_audit.{json,md}`

| Split | Variant | seeds | r mean | r std | r min | r max | RMSE mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LOSO | B1 | 5 | `-0.0010` | `0.0080` | `-0.0108` | `0.0133` | `0.9771` |
| LOSO | B2_lam0.005 | 5 | `0.0025` | `0.0088` | `-0.0104` | `0.0143` | `0.9741` |
| LOSO | B2_lam0.01 | 5 | `0.0026` | `0.0091` | `-0.0113` | `0.0143` | `0.9756` |
| S4 | B1 | 5 | `0.1815` | `0.0261` | `0.1549` | `0.2307` | `0.9223` |
| S4 | B2_lam0.005 | 5 | `0.1872` | `0.0226` | `0.1680` | `0.2302` | `0.9196` |
| S4 | B2_lam0.01 | 5 | `0.1800` | `0.0291` | `0.1482` | `0.2328` | `0.9226` |
| S2 | B1 | 5 | `0.1868` | `0.0404` | `0.1184` | `0.2391` | `0.9714` |
| S2 | B2_lam0.005 | 5 | `0.1784` | `0.0447` | `0.1177` | `0.2380` | `0.9763` |
| S2 | B2_lam0.01 | 5 | `0.1638` | `0.0447` | `0.1158` | `0.2363` | `0.9866` |

Repeat-stability conclusion: B2 small-lambda LOSO/S4 mean gains over B1 are tiny and within seed-level variance; S2 is more stable with B1. The first-pass B2_lam0.005 S2/LOSO advantage should be treated as a candidate signal, not a stable upgrade. S2 high-error subjects concentrate around `sub-03`, `sub-08`, `sub-09`, `sub-13`, and `sub-11`; S4 low-r subjects concentrate around `sub-05`, `sub-13`, `sub-14`, `sub-15`, and `sub-11`.

| Variant | dim | Subject Probe | Session Probe | Fatigue Ridge LOSO r | Fatigue Ridge S1 r | Fatigue Ridge S4 r | Fatigue Ridge S2 r | var mean | pred std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 256 | `0.9739` | `0.9525` | `-0.0279` | `0.2943` | `0.1536` | `0.0831` | `1.0523` | `0.3448` |
| B1 | 64 | `0.8707` | `0.8869` | `-0.0139` | `0.2214` | `0.0648` | `0.0542` | `1.0241` | `0.3848` |
| B2_lam0.005 | 64 | `0.8569` | `0.8849` | `-0.0129` | `0.2108` | `0.1273` | `0.0165` | `1.0169` | `0.4179` |
| B2_lam0.01 | 64 | `0.8664` | `0.8852` | `0.0196` | `0.2408` | `0.0984` | `0.1675` | `1.0268` | `0.4165` |

Adapter-z conclusion: adapter representations reduce subject/session probe accuracy, but Fatigue Ridge does not improve overall versus B0. B1's downstream gain is therefore more consistent with supervised adapter plus nonlinear fatigue head fitting than with a clearly better linear fatigue geometry in adapter z. Keep B1 as the main V4d candidate; treat B2 lambdas as conservative regularization candidates only.

## 8. B1-Frozen Event-Level Aggregation And LOSO Diagnostics

After the B1/B2 repeat, the next local implementation froze B1 and reused the synced LOSO OOF adapter representation bundle:

- input: `outputs/server_sync/video_v4d_results_2026-07-07/server_tree/outputs/reports/video_grl_adapter/representation_audit/b0_b1_b2_loso_representations.npz`
- event outputs: `outputs/embeddings/video_event_b1/E{1,2,3}_*.npz`
- event reports: `outputs/reports/video_event_b1/{loso,s1,s4,s2}_metrics.{json,md}`
- diagnostics: `outputs/reports/video_loso_diagnostics/`
- personalization: `outputs/reports/video_personalization/b1_loso_residual_calibration.{json,md}`

The OOF bundle has `8328` windows, `694` events, and `repr__B1.shape=(8328,64)` with no NaNs. Event-level pooling kept all `694` events; no event had fewer than the default `8` required windows.

| Model / diagnostic | LOSO r / RMSE | S1 r / RMSE | S4 r / RMSE | S2 r / RMSE | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| B1 window baseline, 5-seed mean | `-0.0010 / 0.9771` | `0.3897 / NA` | `0.1815 / 0.9223` | `0.1868 / 0.9714` | Frozen video backbone |
| E1 mean event pooling | `-0.0096 / 1.0807` | `0.3060 / 0.9242` | `0.1261 / 0.9865` | `0.0697 / 1.1847` | No promotion |
| E2 mean+std event pooling | `0.0245 / 1.0744` | `0.2724 / 0.9568` | `0.0696 / 1.0461` | `0.1186 / 1.1531` | No promotion |
| E3 mean+std+max event pooling | `0.0195 / 1.0730` | `0.2978 / 0.9480` | `0.1517 / 1.0074` | `0.0454 / 1.1875` | No promotion |
| E3 subject-centered target diagnostic | `0.1385 / 0.8894` | NA | NA | NA | Diagnostic only |
| B1 residual few-shot, best K/session | `0.0170 / 0.9996` | NA | NA | NA | No simple-bias personalization gain |

Interpretation:

- Simple 120 s pooling does not beat the B1 window-level baseline on S4/S2, so the 12-window TCN/Transformer branch should not start yet.
- Subject-centered event targets improve LOSO r for E3, which suggests subject baseline calibration is part of the failure; this is not a deployment method because it uses the held-out subject's true mean.
- LOSO per-subject signs are mixed: `9` positive-r subjects and `5` negative-r subjects in the B1 LOSO fold report, consistent with subject-dependent fatigue-behavior mapping differences.
- Residual few-shot calibration on B1 LOSO predictions worsened RMSE for K-event protocols and only reached r `0.0170` for 1-session, so simple bias correction is not enough; a future personalization branch should use a richer residual model or small subject-specific head.

## 9. Structure-Search Closure Diagnostics, 2026-07-08

The latest guidance closes the video-only structure-search phase: do not continue GRL sweeps, appearance augmentation, or 12-window temporal encoders until the baseline-vs-within-subject question is resolved. The next diagnostic priority is to separate subject baseline calibration from within-subject fatigue-behavior mapping.

Local artifact check for the requested `B1-R1` versus `B1-R2` ROI decision found that the original R1/R2 embedding files were not available in this workspace or synced server tree. A blocker report with the required server command template was written to `outputs/reports/video_roi_b1/roi_b1_artifact_check.{json,md}`. This means the final ROI version still needs a server rerun from the original R1 `2x face ROI` and R2 `upper-body` embeddings before the final video branch is locked.

Window-level B1 adapter representations were exported back into the existing `face_emb (N,256)` contract through deterministic projection:

- bundle: `outputs/embeddings/video_window_b1/B1_window_repr_embeddings.npz`
- centered bundle: `outputs/embeddings/video_window_b1/B1_window_repr_centered_embeddings.npz`
- reports: `outputs/reports/video_window_b1/`

| Diagnostic | Target | LOSO r | RMSE | pred_std | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| C0 B1 window representation | absolute fatigue | `-0.0448` | `0.9933` | `0.3517` | Absolute cross-subject calibration remains poor |
| C1 B1 window representation | subject-centered fatigue | `0.0157` | `0.8242` | `0.2636` | Centering greatly lowers RMSE, but r is still near zero |

The centered-label result supports baseline calibration as a large part of the RMSE failure, but the near-zero r after centering says the within-subject mapping is still not portable enough for deployment.

Affine few-shot calibration was then added to the personalization entrypoint. On original B1 LOSO predictions, regularized affine K=10 improved RMSE from `0.9251` to `0.8813`, but Pearson r stayed near zero (`0.0003`). On centered C1 predictions, 0-shot had the best RMSE (`0.7983`), and affine calibration did not improve it. Bias-only and affine few-shot calibration therefore do not restore predictive correlation; a future personalization branch should only proceed if it uses a stronger residual model or small subject-specific head.

Cross-subject centered transfer was added as a mapping-direction diagnostic:

- report: `outputs/reports/video_transfer_matrix/b1_centered_transfer_matrix.{json,md}`
- subjects: `14`
- within-subject diagonal r mean: `0.6981`
- off-diagonal transfer r mean: `0.0089`
- positive / negative off-diagonal pairs: `117 / 79`

This is the clearest signal so far: B1 contains strong within-subject learnable structure, but the learned centered mapping does not transfer across subjects. The next main direction should be subject calibration and subject-dependent mapping clusters, plus a final server-side `B1-R1` versus `B1-R2` rerun. Multimodal fusion should freeze the selected B1 video branch and compare no-video versus `+V4a` versus `+B1`; video-only structure search remains paused.
