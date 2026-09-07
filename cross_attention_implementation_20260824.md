# Cross-Attention 融合实现说明

本文解释 `technical_route_20260814.md` 中 “Cross-Attention 融合” 一节对应的代码实现。该路线对应当前 `scripts/32_run_eegpt_centered_loss.py` 中 `AttentionRegressor` 的默认 `variant="attention"` 分支。

## 一句话概括

每个样本先被表示成几个模态 token：

```text
EEG token     (256D)
Wear token    (256D)
Video token   (256D)
Audio token   (256D)
```

模型把这些 token 当成一个很短的“模态序列”，让它们互相做 attention。attention 后，再用一个可学习 query 对模态 token 做加权汇总，得到一个融合向量，最后进入 MLP 回归头预测疲劳分数。

## 输入如何堆成 token

代码先按实验配置加载各模态 embedding。比如：

```text
B0_Wphysio_full     -> EEG + Wphysio + B0 video + Audio
B0_Wphysio_no_audio -> EEG + Wphysio + B0 video
bio_only            -> EEG + Wear
```

每个模态文件都要满足：

```text
embedding shape == (N, 256)
mask shape      == (N,)
```

对应代码位置：

- `scripts/32_run_eegpt_centered_loss.py::_load_all_branches`
- `scripts/32_run_eegpt_centered_loss.py::_build_tokens`

`_build_tokens` 会把模态 embedding 堆成：

```text
tokens:     (N, M, 256)
token_mask: (N, M)
```

其中 `M` 是当前实验实际使用的模态数：

```text
full     -> M=4, [EEG, Wear, Video, Audio]
no_audio -> M=3, [EEG, Wear, Video]
bio_only -> M=2, [EEG, Wear]
```

`token_mask` 表示每个样本的每个模态是否可用。缺失模态不会被 attention 使用，也不会参与后续 pooling。

## 训练前归一化

模型训练前会在 train split 上拟合 token 的均值和标准差。默认口径是 `--token-normalization shared`，即所有可用模态 token 共同估计一套 256D 统计量：

```python
available = np.where(mask[indices, :, None], tokens[indices], np.nan)
mean = np.nanmean(available, axis=(0, 1), keepdims=True)
std = np.nanstd(available, axis=(0, 1), keepdims=True)
```

这里的 `indices` 是 train index。val/test 不参与 mean/std 拟合，因此不会引入 test 泄漏。

2026-08-24 起，脚本也支持 `--token-normalization per_modality`，保留模态维并为每个模态 slot 单独拟合：

```python
mean = np.nanmean(available, axis=0, keepdims=True)
std = np.nanstd(available, axis=0, keepdims=True)
```

两种模式的统计量形状分别是：

```text
shared:       (1, 1, 256)
per_modality: (1, M, 256)
```

归一化后的输入仍是：

```text
(B, M, 256)
```

## 模型主体

`AttentionRegressor` 中，0814 版 attention 路线的核心参数是：

```python
self.input_projection = torch.nn.Linear(256, hidden_dim)
self.modality_embedding = torch.nn.Parameter(torch.zeros(1, modality_count, hidden_dim))
torch.nn.init.normal_(self.modality_embedding, mean=0.0, std=0.02)
self.self_attention = torch.nn.MultiheadAttention(hidden_dim, 1, dropout=dropout, batch_first=True)
self.query = torch.nn.Parameter(torch.zeros(hidden_dim))
torch.nn.init.normal_(self.query, mean=0.0, std=0.02)
```

默认 `hidden_dim=128`。因此每个 256D 模态 token 先被投影到 128D：

```text
tokens: (B, M, 256)
Linear(256 -> 128)
hidden: (B, M, 128)
```

## Learnable Modality Embedding

代码中这一步是：

```python
x = self.input_projection(tokens) + self.modality_embedding
```

它的含义是：给每个模态位置学习一个“身份向量”。

```text
EEG hidden   = Linear(eeg_emb)   + learned_eeg_id
Wear hidden  = Linear(wear_emb)  + learned_wear_id
Video hidden = Linear(video_emb) + learned_video_id
Audio hidden = Linear(audio_emb) + learned_audio_id
```

`modality_embedding` 是 `torch.nn.Parameter`，会被 AdamW 更新。它不是手工 one-hot，也不是标签信息。它的作用是告诉 attention：当前 token 属于 EEG、Wear、Video 还是 Audio。

它和 `modality_mask` 的区别：

```text
modality_embedding -> 表示 token 是什么模态；可训练
modality_mask      -> 表示当前样本该模态是否可用；不可训练
```

## Attention 如何发生

0814 版 attention 使用：

```python
attended, _ = self.self_attention(
    x,
    x,
    x,
    key_padding_mask=~mask,
    need_weights=False,
)
```

严格说，这在 PyTorch 里是 self-attention，因为 `query=key=value=x`。但由于 `x` 的序列维度是模态序列，所以语义上是在做跨模态交互：

```text
EEG token   可以 attend 到 Wear / Video / Audio
Wear token  可以 attend 到 EEG / Video / Audio
Video token 可以 attend 到 EEG / Wear / Audio
Audio token 可以 attend 到 EEG / Wear / Video
```

`key_padding_mask=~mask` 会屏蔽缺失模态，使缺失 token 不能作为 key/value 被其他 token 使用。

输出仍然是：

```text
attended: (B, M, 128)
```

每个模态 token 现在都包含了它从其它模态读到的信息。

## Learnable Query Pooling

attention 后，模型还需要把 `M` 个模态 token 合成一个样本级向量。这里不用平均池化，而是学习一个 query 向量：

```python
scores = torch.matmul(attended, self.query).masked_fill(~mask, -1.0e9)
weights = torch.softmax(scores, dim=1) * mask.to(dtype=scores.dtype)
weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
pooled = torch.sum(attended * weights.unsqueeze(-1), dim=1)
```

可以理解为模型为每个样本动态分配模态权重：

```text
pooled = w_eeg   * attended_eeg
       + w_wear  * attended_wear
       + w_video * attended_video
       + w_audio * attended_audio
```

这些权重对每个样本都不同。缺失模态的 score 会被置为极小值，softmax 后权重接近 0，并且后续还会再次乘以 mask。

最终得到：

```text
pooled: (B, 128)
```

## 回归头如何输出疲劳分数

`pooled` 进入回归头：

```python
self.regression_head = torch.nn.Sequential(
    torch.nn.LayerNorm(hidden_dim),
    torch.nn.Linear(hidden_dim, hidden_dim),
    torch.nn.ReLU(),
    torch.nn.Dropout(dropout),
    torch.nn.Linear(hidden_dim, 1),
)
```

即：

```text
(B, 128)
-> LayerNorm
-> Linear(128 -> 128)
-> ReLU
-> Dropout
-> Linear(128 -> 1)
-> fatigue prediction
```

训练时，目标值会用 train split 的均值和标准差标准化。模型输出的是标准化空间中的预测，评估时再还原到原始 1-5 分数尺度。

## 哪些参数会训练

这一层融合模型中会被 AdamW 更新的参数包括：

```text
Linear(256 -> hidden_dim)
learnable modality embedding
MultiheadAttention 的 Q/K/V/O 参数
learnable query
LayerNorm + MLP regression head
```

不会在这一阶段被更新的部分包括：

```text
已经生成好的 EEG/Wear/Video/Audio 256D embedding 文件
固定的 sample_id 顺序
modality_mask
train/val/test split
```

## Cross-Attention 的完整数据流

```text
1. 读取每个模态的 .npz
   eeg_emb / wear_emb / video_emb / audio_emb

2. 对齐 sample_id
   确保所有模态顺序和 canonical index 一致

3. 堆叠 token
   tokens     = (N, M, 256)
   token_mask = (N, M)

4. train-only normalization
   用 train split 拟合 mean/std

5. 输入投影
   Linear(256 -> 128)

6. 加模态身份向量
   + learnable modality embedding

7. 模态间 self-attention
   MultiheadAttention(query=x, key=x, value=x)

8. query pooling
   learnable query 给各模态 token 分配动态权重

9. 回归头
   LayerNorm + MLP -> fatigue prediction

10. loss 与优化
    raw MSE 或 raw MSE + lambda * centered_loss
```

## 与后续变体的关系

`technical_route_20260814.md` 描述的是 0814 版默认 attention 路线，也就是：

```text
--fusion-variant attention
```

当前代码后来扩展了其它 fusion variant：

```text
concat
attention_multihead_pma
eeg_anchor
```

这些变体共用同一套加载、mask、head、loss 和 evaluation 链路，主要区别在 `AttentionRegressor.encode()`。因此阅读 `technical_route_20260814.md` 时，应把其中的 “cross-attention” 理解为默认单头 modality-token attention，而不是后续所有 fusion 变体的统称。

## 关键边界

这层 attention 只在“模态 token”之间交互，不建模 10 秒窗口内部的时间序列。EEG、Wear、Video、Audio 的时间结构已经在各自 encoder 中被压成 256D 表征。

因此它回答的问题是：

```text
给定 EEG/Wear/Video/Audio 四个窗口级表征，
怎样让模型按样本动态组合这些模态？
```

它不回答：

```text
EEG 10 秒波形内部如何建模？
Wear PPG/GSR/ACC 的秒级变化如何建模？
Video clip 内每一帧如何聚合？
Audio frame-level 特征如何聚合？
```

这些问题属于各单模态 encoder 的职责。
