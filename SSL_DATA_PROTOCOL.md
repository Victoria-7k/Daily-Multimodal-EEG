# SSL Data Protocol

所有 SSL 方法统一使用现有正式 `cross_day` split，不重新划分数据。

## Canonical index

所有 split JSON 中的整数均为以下 canonical index 的 0-based 行号：

```text
/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/index/eeg_aligned_window_index.jsonl
```

该 index 共包含 28,819 个窗口。

## Cross-day split

正式 split 位置：

```text
/vePFS-0x0d/DailyEEG/splits_new/cross_day
```

包含：

```text
pretrain.json
finetune.json
val.json
test.json
split_info.json
```

正式训练集定义为：

```text
train = pretrain.json ∪ finetune.json
```

因此所有 SSL 方法，包括：

```text
EEG-MAE
Wear-MAE
Video-MAE
V-JEPA
intra-modal JEPA
inter-modal JEPA
```

均使用：

```text
pretrain.json + finetune.json
```

作为允许参与自监督训练的数据范围。

`val.json` 用于验证和模型选择，`test.json` 仅用于最终测试，不参与任何 SSL 训练。
