# Token Normalization Scope Probe

本文记录 2026-08-24 对 `scripts/window_fatigue/32_run_eegpt_centered_loss.py` 新增 `--token-normalization` 后的快速效果检查。

## 改动

新增参数：

```text
--token-normalization shared
--token-normalization per_modality
```

`shared` 是历史默认行为：所有可用模态 token 共同拟合一套 train-only 256D mean/std，统计量形状为 `(1, 1, 256)`。

`per_modality` 是新增行为：每个模态 slot 分别拟合 train-only 256D mean/std，统计量形状为 `(1, M, 256)`。

两种模式都只使用 train split 拟合统计量，val/test 只复用 train 统计量。

## 实验口径

这是一个快速 screen，不是正式主线结论。

- 远端运行目录：`/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned`
- 临时运行脚本：`scripts/32_run_eegpt_centered_loss_token_norm_probe.py`
- 本地同步产物：`outputs/server_sync/token_norm_probe_20260824/`
- EEG branch：`eeg_eegpt_partial_ft_v1`
- EEG token root：`eeg_encoder_256d_tokens`
- EEG token seed：`240800`
- fusion variant：`attention`
- training seed：`240729`
- seed 口径：`--experiment-seed-fixed`
- head：`regression`
- scope 对照：`shared` vs `per_modality`
- route：
  - `cross_day:B0_Wphysio_no_audio`
  - `within_subject_day:B0_Wdeep_no_audio`

执行命令模板：

```bash
PYTHONPATH=src runtime/envs/eegpt-gpu-min/bin/python scripts/32_run_eegpt_centered_loss_token_norm_probe.py \
  --experiment-set custom \
  --experiments cross_day:B0_Wphysio_no_audio,within_subject_day:B0_Wdeep_no_audio \
  --eeg-branches eeg_eegpt_partial_ft_v1 \
  --eeg-token-root eeg_encoder_256d_tokens \
  --eeg-token-seed 240800 \
  --experiment-seed-fixed \
  --loss-modes raw_centered_corr \
  --lambdas 0.3 \
  --heads regression \
  --epochs 80 \
  --patience 15 \
  --device cuda \
  --token-normalization <shared|per_modality> \
  --out-json outputs/reports/token_norm_probe_20260824/<scope>.json \
  --out-md outputs/reports/token_norm_probe_20260824/<scope>.md
```

脚本默认会附带 raw baseline，因此每个 scope 实际输出 4 条 run。

## 结果

| protocol | experiment | loss | scope | RMSE | raw r | centered r | best epoch |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `cross_day` | `B0_Wphysio_no_audio` | `raw`, lambda 0 | `shared` | 0.9446 | 0.3092 | 0.1070 | 7 |
| `cross_day` | `B0_Wphysio_no_audio` | `raw`, lambda 0 | `per_modality` | 0.9239 | 0.3234 | 0.1040 | 1 |
| `cross_day` | `B0_Wphysio_no_audio` | `raw_centered_corr`, lambda 0.3 | `shared` | 0.9554 | 0.2918 | 0.1420 | 7 |
| `cross_day` | `B0_Wphysio_no_audio` | `raw_centered_corr`, lambda 0.3 | `per_modality` | 0.9506 | 0.3075 | 0.1057 | 2 |
| `within_subject_day` | `B0_Wdeep_no_audio` | `raw`, lambda 0 | `shared` | 0.9513 | 0.3837 | 0.1712 | 21 |
| `within_subject_day` | `B0_Wdeep_no_audio` | `raw`, lambda 0 | `per_modality` | 0.9673 | 0.3637 | 0.1422 | 6 |
| `within_subject_day` | `B0_Wdeep_no_audio` | `raw_centered_corr`, lambda 0.3 | `shared` | 0.9399 | 0.3743 | 0.1774 | 19 |
| `within_subject_day` | `B0_Wdeep_no_audio` | `raw_centered_corr`, lambda 0.3 | `per_modality` | 0.9257 | 0.3984 | 0.1773 | 2 |

## Paired Delta

Delta 定义为 `per_modality - shared`。RMSE 越低越好，raw r / centered r 越高越好。

| protocol | experiment | loss | delta RMSE | delta raw r | delta centered r |
| --- | --- | --- | ---: | ---: | ---: |
| `cross_day` | `B0_Wphysio_no_audio` | `raw`, lambda 0 | -0.0206 | +0.0142 | -0.0030 |
| `cross_day` | `B0_Wphysio_no_audio` | `raw_centered_corr`, lambda 0.3 | -0.0048 | +0.0157 | -0.0363 |
| `within_subject_day` | `B0_Wdeep_no_audio` | `raw`, lambda 0 | +0.0160 | -0.0199 | -0.0290 |
| `within_subject_day` | `B0_Wdeep_no_audio` | `raw_centered_corr`, lambda 0.3 | -0.0142 | +0.0240 | -0.0001 |

## 初步解读

`per_modality` 在这个 2-route screen 中不是单调更优。

有利信号：

- `cross_day:B0_Wphysio_no_audio` raw baseline 的 RMSE 和 raw r 都改善。
- 两条 `raw_centered_corr` run 的 raw r 都改善。
- `within_subject_day:B0_Wdeep_no_audio` 的 `raw_centered_corr` run 同时改善 RMSE 和 raw r，centered r 基本持平。

风险信号：

- `within_subject_day:B0_Wdeep_no_audio` raw baseline 明显变差。
- `cross_day:B0_Wphysio_no_audio` 的 `raw_centered_corr` centered r 从 0.1420 降到 0.1057。
- 两条 per-modality 改善项的 best epoch 都更早，提示训练动态和早停点发生变化，需要多 seed 验证。

## 结论

`per_modality` 值得继续作为 ablation 候选，但不能直接替换默认 `shared`。

建议下一步做 3-seed paired screen：

- 固定 `--experiment-seed-fixed`
- 至少覆盖 `cross_day` 与 `within_subject_day`
- 同时报告 RMSE、raw r、centered r 和 best epoch
- 对 `raw` 与当前候选 loss/head 分开判断

如果 3-seed 上 raw r/RMSE 改善稳定且 centered r 不系统性下降，再考虑扩展到完整 route matrix。

## 扩展 Route Screen

用户追问后，进一步把 raw-only screen 扩展到 `video_only` route set：

- 协议：`cross_day`、`within_subject_day`
- route 数：每个 protocol 24 条，共 48 条
- route 覆盖：B0/A1/A2 video、Wphysio/Wdeep/Wmoment frozen/Wmoment partial-ft、full/no_audio
- EEG branch：`eeg_eegpt_partial_ft_v1`
- seed：`240729`，`--experiment-seed-fixed`
- loss/head：`raw` + `regression`
- 本地同步产物：`outputs/server_sync/token_norm_route_screen_20260824/`

执行命令：

```bash
PYTHONPATH=src runtime/envs/eegpt-gpu-min/bin/python scripts/32_run_eegpt_centered_loss_token_norm_probe.py \
  --experiment-set video_only \
  --protocols cross_day,within_subject_day \
  --eeg-branches eeg_eegpt_partial_ft_v1 \
  --eeg-token-root eeg_encoder_256d_tokens \
  --eeg-token-seed 240800 \
  --experiment-seed-fixed \
  --loss-modes raw \
  --no-raw-baseline \
  --heads regression \
  --epochs 80 \
  --patience 15 \
  --device cuda \
  --token-normalization <shared|per_modality> \
  --out-json outputs/reports/token_norm_route_screen_20260824/<scope>_raw.json \
  --out-md outputs/reports/token_norm_route_screen_20260824/<scope>_raw.md
```

### Protocol Summary

Delta 定义为 `per_modality - shared`。

| protocol | route count | mean delta RMSE | mean delta raw r | mean delta centered r | RMSE wins | raw r wins | centered r wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cross_day` | 24 | -0.0126 | +0.0311 | +0.0167 | 17/24 | 20/24 | 19/24 |
| `within_subject_day` | 24 | +0.0331 | -0.0235 | -0.0321 | 2/24 | 3/24 | 0/24 |
| all | 48 | +0.0103 | +0.0038 | -0.0077 | 19/48 | 23/48 | 19/48 |

### Top Routes By Raw R

`cross_day` 中，`per_modality` 把 raw r 排名重排到更均匀的 0.32-0.33 区间：

| scope | route | RMSE | raw r | centered r |
| --- | --- | ---: | ---: | ---: |
| `shared` | `A1_Wphysio_no_audio` | 0.9208 | 0.3554 | 0.1289 |
| `shared` | `A1_Wmoment_frozen_full` | 0.9159 | 0.3319 | 0.1417 |
| `shared` | `A2_Wmoment_ft_full` | 0.9122 | 0.3231 | 0.1097 |
| `per_modality` | `B0_Wphysio_full` | 0.9187 | 0.3345 | 0.1446 |
| `per_modality` | `A1_Wphysio_full` | 0.9218 | 0.3331 | 0.1462 |
| `per_modality` | `A2_Wmoment_frozen_full` | 0.9253 | 0.3320 | 0.1394 |

`within_subject_day` 中，`per_modality` 的最佳 raw r 仍低于 shared 的最佳 raw r，且 centered r 整体下移：

| scope | route | RMSE | raw r | centered r |
| --- | --- | ---: | ---: | ---: |
| `shared` | `A1_Wphysio_no_audio` | 0.9141 | 0.3985 | 0.1859 |
| `shared` | `B0_Wphysio_no_audio` | 0.9316 | 0.3976 | 0.1981 |
| `shared` | `B0_Wmoment_frozen_no_audio` | 0.9209 | 0.3957 | 0.1796 |
| `per_modality` | `A2_Wdeep_no_audio` | 0.9316 | 0.3938 | 0.1678 |
| `per_modality` | `A1_Wdeep_no_audio` | 0.9324 | 0.3908 | 0.1643 |
| `per_modality` | `B0_Wmoment_frozen_no_audio` | 0.9295 | 0.3831 | 0.1589 |

### Largest Raw-R Gains And Losses

Largest gains were concentrated in `cross_day`, especially Wdeep routes:

| protocol | route | delta RMSE | delta raw r | delta centered r |
| --- | --- | ---: | ---: | ---: |
| `cross_day` | `A2_Wdeep_full` | -0.0380 | +0.1035 | +0.0336 |
| `cross_day` | `A1_Wdeep_no_audio` | -0.0289 | +0.0710 | +0.0341 |
| `cross_day` | `A2_Wdeep_no_audio` | -0.0334 | +0.0620 | +0.0233 |
| `cross_day` | `B0_Wdeep_full` | -0.0378 | +0.0581 | +0.0341 |
| `cross_day` | `B0_Wdeep_no_audio` | -0.0296 | +0.0545 | +0.0024 |

Largest losses were concentrated in `within_subject_day`, especially Wphysio no_audio and Wmoment full routes:

| protocol | route | delta RMSE | delta raw r | delta centered r |
| --- | --- | ---: | ---: | ---: |
| `within_subject_day` | `B0_Wphysio_no_audio` | +0.0549 | -0.0764 | -0.0775 |
| `within_subject_day` | `A1_Wphysio_no_audio` | +0.0708 | -0.0647 | -0.0592 |
| `within_subject_day` | `B0_Wmoment_frozen_full` | +0.0588 | -0.0529 | -0.0536 |
| `within_subject_day` | `A2_Wphysio_no_audio` | +0.0689 | -0.0468 | -0.0308 |
| `within_subject_day` | `A1_Wmoment_ft_no_audio` | +0.0528 | -0.0454 | -0.0304 |

### Updated Interpretation

The broader raw-only route screen makes the split-dependent behavior clear:

- `cross_day`: `per_modality` is a strong candidate. It improves mean raw r by `+0.0311`, improves mean RMSE by `-0.0126`, and wins raw r on 20/24 routes.
- `within_subject_day`: `per_modality` is not a candidate under raw-only training. It worsens mean RMSE by `+0.0331`, mean raw r by `-0.0235`, and centered r on all 24 routes.
- Overall: the global average hides this divergence. Treat `per_modality` as a protocol-specific ablation rather than a universal default.

Next gate should therefore be protocol-aware:

- For `cross_day`: run 3-seed paired confirmation on the 24-route video-only set.
- For `within_subject_day`: keep `shared` as the default unless a different loss/head changes the pattern.
- Do not replace the main default before a paired multi-seed result confirms the `cross_day` gain and checks whether the centered-r penalty reappears.
