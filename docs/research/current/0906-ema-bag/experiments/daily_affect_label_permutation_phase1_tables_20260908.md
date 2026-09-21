# Daily-affect 标签置换 Phase 1 结果表

## 固定说明

- 置换模式：`global_shuffle`；在 train 与 validation split 内独立置换完整 EMA event 标签。
- 置换单位：一个 `(23,4,256)` EMA bag；token、模态 mask、event/subject/day 标识、split 和测试标签均保持不变。
- 每个条件使用一个 matched model seed `240729`，共 `5` 个 permutation ID。
- 所有条件的 `supervision_boundary` 均为 `label_free_or_fixed_embeddings_only`。

## Cross-day

`A1_Wphysio_full / dynamic_kernel_prior_uniform / per_modality normalization / per_modality adapter`

| 条件 | 标签变化比例 | best epoch | QWK | raw r | centered r | Macro-F1 | ordinal MAE | Expected RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean | -- | -- | 0.2411 | 0.2954 | 0.0723 | 0.2750 | 0.9328 | 1.0355 |
| Perm 0 | 0.679 | 27 | 0.0401 | 0.0507 | 0.0942 | 0.2050 | 1.0395 | 1.1540 |
| Perm 1 | 0.694 | 3 | -0.0578 | -0.0393 | -0.0332 | 0.1943 | 0.9960 | 1.0559 |
| Perm 2 | 0.682 | 5 | 0.1346 | 0.0683 | 0.0968 | 0.2016 | 1.0988 | 1.0817 |
| Perm 3 | 0.684 | 3 | 0.0370 | 0.0791 | 0.0560 | 0.1608 | 0.9921 | 0.9994 |
| Perm 4 | 0.647 | 27 | -0.0605 | -0.0722 | 0.0385 | 0.1920 | 1.0474 | 1.2196 |

| Phase 1 gate | 结果 |
| --- | --- |
| 标签、split、测试索引与测试标签审计 | 通过 |
| clean QWK 高于 null | 5/5 |
| clean raw r 高于 null | 5/5 |
| null centered r 为正 | 4/5 |
| 状态 | `stop_and_audit` |

## Cross-subject

`A2_Wdeep_full / bag_static / shared normalization / shared adapter`

| 条件 | 标签变化比例 | best epoch | QWK | raw r | centered r | Macro-F1 | ordinal MAE | Expected RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean | -- | -- | 0.2158 | 0.1146 | 0.0240 | 0.2248 | 0.8030 | 0.9945 |
| Perm 0 | 0.685 | 13 | 0.0257 | 0.0574 | 0.0783 | 0.0690 | 1.6894 | 1.2151 |
| Perm 1 | 0.671 | 4 | -0.1111 | -0.1457 | -0.0517 | 0.1103 | 0.9848 | 1.1029 |
| Perm 2 | 0.686 | 9 | -0.1318 | -0.0848 | 0.0283 | 0.1891 | 1.1742 | 1.1098 |
| Perm 3 | 0.674 | 19 | -0.0725 | -0.0230 | -0.0006 | 0.1253 | 1.1970 | 1.0443 |
| Perm 4 | 0.698 | 7 | -0.1664 | -0.1935 | -0.1642 | 0.1490 | 1.1515 | 1.1204 |

| Phase 1 gate | 结果 |
| --- | --- |
| 标签、split、测试索引与测试标签审计 | 通过 |
| clean QWK 高于 null | 5/5 |
| clean raw r 高于 null | 5/5 |
| null centered r 为正 | 2/5 |
| 状态 | `phase1_pass_expand_to_within_subject` |

## Within-subject-day

`B0_Wphysio_full / state_uniform / per_modality normalization / per_modality adapter`

| 条件 | 标签变化比例 | best epoch | QWK | raw r | centered r | Macro-F1 | ordinal MAE | Expected RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean | -- | -- | 0.3115 | 0.3015 | 0.0483 | 0.2996 | 0.7829 | 0.9438 |
| Perm 0 | 0.661 | 13 | -0.0076 | 0.0978 | 0.0011 | 0.1904 | 1.0504 | 1.0477 |
| Perm 1 | 0.682 | 44 | -0.0270 | -0.0515 | 0.0087 | 0.2226 | 1.1085 | 1.2656 |
| Perm 2 | 0.695 | 5 | 0.1255 | 0.0829 | 0.0360 | 0.2413 | 0.9341 | 1.0110 |
| Perm 3 | 0.678 | 35 | -0.0261 | -0.0259 | 0.0086 | 0.1688 | 0.9690 | 1.1864 |
| Perm 4 | 0.675 | 41 | 0.0565 | 0.0132 | -0.0235 | 0.1895 | 0.9109 | 1.1388 |

| Phase 1 gate | 结果 |
| --- | --- |
| 标签、split、测试索引与测试标签审计 | 通过 |
| clean QWK 高于 null | 5/5 |
| clean raw r 高于 null | 5/5 |
| null centered r 为正 | 4/5 |
| 状态 | `stop_and_audit` |
