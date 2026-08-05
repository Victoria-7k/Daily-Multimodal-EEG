# Within-Subject Fusion Complete Results

Generated from remote prediction shards after both production protocols completed.

## Scope

- Protocols: `event_grouped_5fold`, `session_held_out`
- Matrix: 20 experiments x 3 models x 2 protocols = 120 result rows
- Subjects: 14; paired windows: 7,368
- Models: `train_mean`, `concat_ridge_alpha10`, `learnable_cross_attention`
- Complete flat table: `outputs/server_sync/within_subject_fusion_120s10s/within_subject_fusion_complete_results.csv`
- Complete recomputed JSON: `outputs/server_sync/within_subject_fusion_120s10s/within_subject_fusion_complete_recomputed_metrics.json`
- V2 full metric long table: `outputs/server_sync/within_subject_fusion_120s10s/within_subject_fusion_complete_metrics_v2_long.csv`
- V2 full metric JSON: `outputs/server_sync/within_subject_fusion_120s10s/within_subject_fusion_complete_metrics_v2.json`

## Metric Coverage

The V2 metric table reports each model/experiment/protocol at three levels:
`window`, `as_run_composite_event`, and `strict_event_id_event`.

For every row it includes:

- `pooled_raw_r`
- `within_subject_centered_r`
- `per_subject_r_mean`
- `per_subject_r_std`
- `rmse`
- `mae`
- `centered_rmse`

`centered_rmse` is RMSE after centering prediction and target by subject OOF
mean at the evaluated level. `per_subject_r_std` is the population standard
deviation across eligible subjects.

Top strict-event cross-attention rows:

| Protocol | Rank | Experiment | Pooled raw r | Centered r | Per-subject r mean | Per-subject r std | RMSE | MAE | Centered RMSE |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `event_grouped_5fold` | 1 | `fusion_WdeepPre_FullSweepB3Lam005_full` | 0.5076 | 0.2673 | 0.1641 | 0.2882 | 0.7717 | 0.6037 | 0.7672 |
| `event_grouped_5fold` | 2 | `fusion_WphysioPre_FullSweepB3Lam005_full` | 0.5033 | 0.2536 | 0.1639 | 0.2862 | 0.7723 | 0.6037 | 0.7702 |
| `session_held_out` | 1 | `fusion_WphysioPre_FullSweepB3Lam005_full` | 0.4077 | 0.0625 | -0.0064 | 0.3306 | 0.8441 | 0.6617 | 0.8357 |
| `session_held_out` | 2 | `fusion_WdeepPre_FullSweepB3Lam005_full` | 0.3705 | 0.0096 | -0.0335 | 0.2863 | 0.8639 | 0.6743 | 0.8539 |

## Event Aggregation Audit

The run-time composite key `(subject_id, session_id, event_id)` produces `7,368` event rows, equal to the window count. Recomputing by `event_id` produces `614` events. The stored `event_id` values include subject/session row tokens in this dataset, while stored `session_id` varies at window granularity. Therefore, the original run summary is best read as an as-run window-granular event proxy; the analysis below uses `event_id` aggregation as the stricter event-level estimate.

## event_grouped_5fold Cross-Attention

- Best by strict event aggregation: `fusion_WdeepPre_FullSweepB3Lam005_full`; event subject-macro Pearson `0.1641`, event subject-macro RMSE `0.7741`, centered pooled Pearson `0.2673`.
- Best by as-run composite aggregation: `fusion_WdeepPre_FullSweepB3Lam005_full`; subject-macro Pearson `0.1380`.

| Wear | Video route | Ablation | Strict event Pearson | Strict event RMSE | Strict centered r | As-run Pearson | Window r | Window RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `WdeepPre` | `A1A2TrainOnlyA2` | `full` | 0.0651 | 0.7846 | 0.1744 | 0.0500 | 0.4333 | 0.8193 |
| `WdeepPre` | `A1A2TrainOnlyA2` | `no_audio` | 0.0290 | 0.8181 | 0.1375 | 0.0317 | 0.4039 | 0.8459 |
| `WdeepPre` | `B5A1Lam0001` | `full` | 0.1373 | 0.7873 | 0.2391 | 0.1203 | 0.4427 | 0.8285 |
| `WdeepPre` | `B5A1Lam0001` | `no_audio` | 0.1187 | 0.7990 | 0.2479 | 0.1011 | 0.4438 | 0.8290 |
| `WdeepPre` | `FullSweepB0` | `full` | 0.1031 | 0.7835 | 0.2035 | 0.0829 | 0.4289 | 0.8288 |
| `WdeepPre` | `FullSweepB0` | `no_audio` | 0.0482 | 0.8216 | 0.1519 | 0.0419 | 0.4085 | 0.8492 |
| `WdeepPre` | `FullSweepB3Lam005` | `full` | 0.1641 | 0.7741 | 0.2673 | 0.1380 | 0.4650 | 0.8114 |
| `WdeepPre` | `FullSweepB3Lam005` | `no_audio` | 0.1200 | 0.7995 | 0.2436 | 0.0959 | 0.4473 | 0.8327 |
| `WdeepPre` | `none` | `bio_only` | -0.0039 | 0.8157 | 0.1072 | 0.0054 | 0.3944 | 0.8473 |
| `WdeepPre` | `none` | `no_video` | 0.0969 | 0.7767 | 0.1976 | 0.0804 | 0.4450 | 0.8186 |
| `WphysioPre` | `A1A2TrainOnlyA2` | `full` | 0.1038 | 0.7771 | 0.2134 | 0.0763 | 0.4376 | 0.8220 |
| `WphysioPre` | `A1A2TrainOnlyA2` | `no_audio` | 0.0798 | 0.7918 | 0.1749 | 0.0644 | 0.4333 | 0.8211 |
| `WphysioPre` | `B5A1Lam0001` | `full` | 0.1062 | 0.7971 | 0.2019 | 0.0927 | 0.4302 | 0.8311 |
| `WphysioPre` | `B5A1Lam0001` | `no_audio` | 0.1099 | 0.7906 | 0.2324 | 0.0937 | 0.4428 | 0.8235 |
| `WphysioPre` | `FullSweepB0` | `full` | 0.1208 | 0.7776 | 0.1970 | 0.0887 | 0.4365 | 0.8275 |
| `WphysioPre` | `FullSweepB0` | `no_audio` | 0.1027 | 0.7862 | 0.2348 | 0.0847 | 0.4507 | 0.8136 |
| `WphysioPre` | `FullSweepB3Lam005` | `full` | 0.1639 | 0.7765 | 0.2536 | 0.1334 | 0.4581 | 0.8128 |
| `WphysioPre` | `FullSweepB3Lam005` | `no_audio` | 0.1340 | 0.7867 | 0.2386 | 0.1092 | 0.4464 | 0.8264 |
| `WphysioPre` | `none` | `bio_only` | 0.0022 | 0.7916 | 0.1028 | 0.0014 | 0.4267 | 0.8154 |
| `WphysioPre` | `none` | `no_video` | 0.1258 | 0.7725 | 0.1771 | 0.0917 | 0.4242 | 0.8257 |

## event_grouped_5fold Top Models By Strict Event Pearson

| Rank | Model | Experiment | Strict event Pearson | Strict event RMSE | As-run Pearson |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | `concat_ridge_alpha10` | `fusion_WphysioPre_B5A1Lam0001_no_audio` | 0.1943 | 0.9256 | 0.1436 |
| 2 | `concat_ridge_alpha10` | `fusion_WphysioPre_FullSweepB3Lam005_no_audio` | 0.1832 | 0.9573 | 0.1305 |
| 3 | `learnable_cross_attention` | `fusion_WdeepPre_FullSweepB3Lam005_full` | 0.1641 | 0.7741 | 0.1380 |
| 4 | `learnable_cross_attention` | `fusion_WphysioPre_FullSweepB3Lam005_full` | 0.1639 | 0.7765 | 0.1334 |
| 5 | `concat_ridge_alpha10` | `fusion_WdeepPre_FullSweepB3Lam005_no_audio` | 0.1539 | 0.9977 | 0.1118 |
| 6 | `learnable_cross_attention` | `fusion_WdeepPre_B5A1Lam0001_full` | 0.1373 | 0.7873 | 0.1203 |
| 7 | `learnable_cross_attention` | `fusion_WphysioPre_FullSweepB3Lam005_no_audio` | 0.1340 | 0.7867 | 0.1092 |
| 8 | `concat_ridge_alpha10` | `fusion_WdeepPre_B5A1Lam0001_no_audio` | 0.1332 | 1.3564 | 0.1036 |
| 9 | `concat_ridge_alpha10` | `fusion_WphysioPre_A1A2TrainOnlyA2_no_audio` | 0.1302 | 1.2487 | 0.0825 |
| 10 | `concat_ridge_alpha10` | `fusion_WdeepPre_FullSweepB0_no_audio` | 0.1258 | 1.6965 | 0.0813 |
| 11 | `learnable_cross_attention` | `fusion_WphysioPre_no_video` | 0.1258 | 0.7725 | 0.0917 |
| 12 | `concat_ridge_alpha10` | `fusion_WdeepPre_A1A2TrainOnlyA2_no_audio` | 0.1246 | 1.3721 | 0.0847 |

## event_grouped_5fold Best Strict Event Cross-Attention Subjects

Configuration: `fusion_WdeepPre_FullSweepB3Lam005_full`

| Subject | Event count | Pearson | RMSE |
| --- | ---: | ---: | ---: |
| `sub-02` | 29 | -0.2408 | 0.8821 |
| `sub-03` | 33 | 0.3888 | 1.0592 |
| `sub-04` | 24 | 0.4653 | 0.5116 |
| `sub-05` | 59 | 0.3699 | 0.6233 |
| `sub-06` | 58 | 0.4294 | 0.8324 |
| `sub-07` | 43 | 0.3363 | 0.6723 |
| `sub-08` | 55 | 0.1289 | 0.9738 |
| `sub-09` | 66 | 0.2669 | 0.8257 |
| `sub-10` | 80 | 0.2680 | 0.5730 |
| `sub-11` | 50 | 0.3282 | 0.6748 |
| `sub-12` | 21 | -0.4722 | 0.7297 |
| `sub-13` | 12 | -0.1929 | 0.9367 |
| `sub-14` | 61 | 0.3640 | 0.7601 |
| `sub-15` | 23 | -0.1419 | 0.7827 |

## session_held_out Cross-Attention

- Best by strict event aggregation: `fusion_WphysioPre_FullSweepB3Lam005_full`; event subject-macro Pearson `-0.0064`, event subject-macro RMSE `0.8426`, centered pooled Pearson `0.0625`.
- Best by as-run composite aggregation: `fusion_WphysioPre_FullSweepB3Lam005_full`; subject-macro Pearson `-0.0080`.

| Wear | Video route | Ablation | Strict event Pearson | Strict event RMSE | Strict centered r | As-run Pearson | Window r | Window RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `WdeepPre` | `A1A2TrainOnlyA2` | `full` | -0.1135 | 0.8560 | -0.0363 | -0.0983 | 0.3255 | 0.8898 |
| `WdeepPre` | `A1A2TrainOnlyA2` | `no_audio` | -0.1674 | 0.8898 | -0.1010 | -0.1380 | 0.2993 | 0.9122 |
| `WdeepPre` | `B5A1Lam0001` | `full` | -0.0612 | 0.8451 | 0.0355 | -0.0404 | 0.3622 | 0.8815 |
| `WdeepPre` | `B5A1Lam0001` | `no_audio` | -0.0380 | 0.8838 | 0.0275 | -0.0436 | 0.3512 | 0.9079 |
| `WdeepPre` | `FullSweepB0` | `full` | -0.0724 | 0.8518 | -0.0043 | -0.0661 | 0.3415 | 0.8789 |
| `WdeepPre` | `FullSweepB0` | `no_audio` | -0.1695 | 0.9009 | -0.1170 | -0.1436 | 0.2768 | 0.9240 |
| `WdeepPre` | `FullSweepB3Lam005` | `full` | -0.0335 | 0.8649 | 0.0096 | -0.0233 | 0.3408 | 0.9005 |
| `WdeepPre` | `FullSweepB3Lam005` | `no_audio` | -0.0640 | 0.8890 | -0.0090 | -0.0547 | 0.3253 | 0.9335 |
| `WdeepPre` | `none` | `bio_only` | -0.1967 | 0.8955 | -0.1308 | -0.1577 | 0.2810 | 0.9210 |
| `WdeepPre` | `none` | `no_video` | -0.1387 | 0.8569 | -0.0676 | -0.1069 | 0.3234 | 0.8916 |
| `WphysioPre` | `A1A2TrainOnlyA2` | `full` | -0.1333 | 0.8523 | -0.0585 | -0.0961 | 0.3228 | 0.8896 |
| `WphysioPre` | `A1A2TrainOnlyA2` | `no_audio` | -0.1493 | 0.8813 | -0.0721 | -0.1256 | 0.3299 | 0.9057 |
| `WphysioPre` | `B5A1Lam0001` | `full` | -0.0492 | 0.8723 | -0.0085 | -0.0455 | 0.3530 | 0.8964 |
| `WphysioPre` | `B5A1Lam0001` | `no_audio` | -0.1282 | 0.9094 | -0.0505 | -0.1052 | 0.3095 | 0.9306 |
| `WphysioPre` | `FullSweepB0` | `full` | -0.1051 | 0.8446 | -0.0077 | -0.0642 | 0.3426 | 0.8799 |
| `WphysioPre` | `FullSweepB0` | `no_audio` | -0.1001 | 0.8714 | -0.0149 | -0.0920 | 0.3551 | 0.8787 |
| `WphysioPre` | `FullSweepB3Lam005` | `full` | -0.0064 | 0.8426 | 0.0625 | -0.0080 | 0.3777 | 0.8793 |
| `WphysioPre` | `FullSweepB3Lam005` | `no_audio` | -0.0800 | 0.8789 | 0.0367 | -0.0657 | 0.3443 | 0.9088 |
| `WphysioPre` | `none` | `bio_only` | -0.2677 | 0.8560 | -0.1252 | -0.2080 | 0.3222 | 0.8734 |
| `WphysioPre` | `none` | `no_video` | -0.1176 | 0.8442 | -0.0518 | -0.0858 | 0.3192 | 0.8833 |

## session_held_out Top Models By Strict Event Pearson

| Rank | Model | Experiment | Strict event Pearson | Strict event RMSE | As-run Pearson |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | `concat_ridge_alpha10` | `fusion_WdeepPre_FullSweepB0_full` | 0.0486 | 248.1191 | 0.0243 |
| 2 | `concat_ridge_alpha10` | `fusion_WphysioPre_bio_only` | 0.0398 | 42.6051 | 0.0247 |
| 3 | `concat_ridge_alpha10` | `fusion_WphysioPre_no_video` | 0.0253 | 759.0537 | 0.0203 |
| 4 | `concat_ridge_alpha10` | `fusion_WdeepPre_no_video` | 0.0234 | 558.6657 | 0.0192 |
| 5 | `concat_ridge_alpha10` | `fusion_WphysioPre_B5A1Lam0001_full` | 0.0189 | 133.5527 | 0.0164 |
| 6 | `concat_ridge_alpha10` | `fusion_WphysioPre_FullSweepB0_full` | 0.0186 | 280.1762 | 0.0155 |
| 7 | `concat_ridge_alpha10` | `fusion_WdeepPre_A1A2TrainOnlyA2_full` | 0.0097 | 258.2785 | 0.0106 |
| 8 | `concat_ridge_alpha10` | `fusion_WphysioPre_A1A2TrainOnlyA2_full` | 0.0013 | 284.9465 | -0.0020 |
| 9 | `concat_ridge_alpha10` | `fusion_WphysioPre_FullSweepB3Lam005_no_audio` | -0.0006 | 5.0782 | 0.0129 |
| 10 | `learnable_cross_attention` | `fusion_WphysioPre_FullSweepB3Lam005_full` | -0.0064 | 0.8426 | -0.0080 |
| 11 | `learnable_cross_attention` | `fusion_WdeepPre_FullSweepB3Lam005_full` | -0.0335 | 0.8649 | -0.0233 |
| 12 | `learnable_cross_attention` | `fusion_WdeepPre_B5A1Lam0001_no_audio` | -0.0380 | 0.8838 | -0.0436 |

## session_held_out Best Strict Event Cross-Attention Subjects

Configuration: `fusion_WphysioPre_FullSweepB3Lam005_full`

| Subject | Event count | Pearson | RMSE |
| --- | ---: | ---: | ---: |
| `sub-02` | 29 | -0.0271 | 1.0372 |
| `sub-03` | 33 | -0.0260 | 1.2093 |
| `sub-04` | 24 | 0.5476 | 0.4962 |
| `sub-05` | 59 | -0.4461 | 0.7476 |
| `sub-06` | 58 | 0.0837 | 0.9533 |
| `sub-07` | 43 | 0.0278 | 0.7380 |
| `sub-08` | 55 | -0.0265 | 1.0802 |
| `sub-09` | 66 | 0.3593 | 0.8013 |
| `sub-10` | 80 | 0.1540 | 0.6219 |
| `sub-11` | 50 | 0.1560 | 0.7318 |
| `sub-12` | 21 | -0.6864 | 0.7366 |
| `sub-13` | 12 | -0.5584 | 1.1071 |
| `sub-14` | 61 | 0.1665 | 0.8172 |
| `sub-15` | 23 | 0.1855 | 0.7183 |

## Interpretation

B3 remains the most consistent video route. In the event-grouped protocol, the strict event-level best attention model is `WdeepPre + FullSweepB3Lam005 + full`, and the same configuration is also best under the as-run composite aggregation. In the harder session-held-out protocol, `WphysioPre + FullSweepB3Lam005 + full` is the best attention configuration, but its strict event Pearson is still slightly negative.

Audio is not carrying the result reliably. Several no-audio variants stay close to, or exceed, their full counterparts under strict event aggregation. The safer current story is that wear plus EEG plus video is doing most of the useful work, while audio should remain an ablation rather than a promoted requirement.

The strongest caution is the event-key audit. Before treating these as final paper-level event metrics, fix the event aggregation key so it produces 614 event rows directly in the runner, then rerun summary aggregation from saved predictions or rerun the matrix. The route ordering is still informative, but the pre-registered primary metric needs that correction.
