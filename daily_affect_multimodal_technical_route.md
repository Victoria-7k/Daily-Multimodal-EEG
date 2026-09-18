# 日常情境四模态情绪回归：可执行技术路线

**版本：2026-08-24**  
**目标：在现有 EEG / Wear / Video / Audio 四模态 256D embedding 基础上，将“窗口级静态融合 + 标签复制”升级为“连续潜在情绪状态估计 + EMA 级动态时间聚合”。**

---

## 1. 研究问题与总体原则

当前实现中，每个时间窗口被表示为 4 个模态 token：

```text
EEG   : 256D
Wear  : 256D
Video : 256D
Audio : 256D
```

随后通过共享 `Linear(256 -> hidden_dim)`、modality embedding、模态间 self-attention、learnable query pooling 和 MLP 回归头预测情绪/疲劳分数。窗口级缺失模态通过 `modality_mask` 屏蔽。

该方案适合作为 baseline，但用于日常情境时有两个主要限制：

1. **四种模态的 embedding 来自不同 encoder，当前默认路线使用共享输入投影和共享 train-only 统计归一化；仓库也已经提供 per-modality normalization 对照入口，但尚未把 shared 或 per-modality 任一方确认为最终默认。**
2. **当前 EEG-aligned 主线中，同一个 EMA event 展开为 23 个 10 秒窗口、5 秒 stride；窗口继承同一事件标签。直接把这些窗口当作独立监督样本，会把一次真实监督扩展为多个高度相关的样本，并隐含评分前两分钟内每个重叠窗口都可独立承载同一标签的假设。**

本工作的主线不再强调“设计更复杂的 cross-attention”，而是将问题改写为：

> 从连续、异构、可能部分缺失的 EEG、Wear、Video 和 Audio 观测中估计随时间演化的潜在情绪状态，并学习稀疏 EMA 评分如何从评分前一段时间的潜在状态轨迹中形成。

本版本**不加入显式噪声检测、SQI 评分或 reliability estimator**。模型只保留普通的动态模态权重与 `modality_mask`；鲁棒性通过训练时的 modality dropout 和测试时的 missing-modality evaluation 检验。

---

# 2. 最终模型概览

整体数据流：

```text
                每次 EMA 前 2 min 连续多模态数据
                              │
                              ▼
            ┌───────────────────────────────────┐
            │  10 s × 23 个 EEG-aligned windows │
            └───────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
          EEG              Wear            Video / Audio
         256D              256D                256D
            │                 │                 │
            ▼                 ▼                 ▼
      modality-specific adapters（各自独立）
            │                 │                 │
            └──────────────┬──┴─────────────────┘
                           ▼
                 当前时刻 multimodal evidence
                           │
                    previous state z(t-1)
                           │
                           ▼
                   state prior p(t)
                           │
                           ▼
              prior-guided modality weighting
                           │
                           ▼
                  posterior state z(t)
                           │
                    递归遍历 23 个时间点
                           │
                           ▼
             z(1), z(2), ..., z(23)
                           │
                           ▼
               Dynamic EMA time kernel
                           │
                           ▼
                    EMA representation
                           │
                           ▼
                      Regression head
                           │
                           ▼
                         y_hat
```

核心模块只保留四项：

1. **Modality-specific adaptation**：将不同 encoder 的 embedding 映射到共同 affect space。
2. **Temporal latent affect state**：显式建模评分前 2 分钟内情绪状态的连续演化。
3. **Prior-guided multimodal state update**：上一时刻状态形成 prior，当前四模态 evidence 对 prior 进行更新。
4. **Dynamic EMA kernel**：将一次 EMA 建模为对过去一段潜在情绪轨迹的加权观测，而不是给每个窗口复制标签。

---

# 3. 数据组织方式

## 3.1 基本时间单位

第一版固定：

```text
LOOKBACK_SEC = 120
WINDOW_SEC   = 10
HOP_SEC      = 5
L            = 23 EEG-aligned temporal windows / EMA
M            = 4 modalities
D_IN         = 256
```

对于第 `i` 次 EMA，评分时间为 `T_i`，取：

```text
[T_i - 120 s, T_i)
```

并切成 23 个 10 s 窗口，相邻窗口 5 s stride，沿用当前 EEG-aligned 主线的事件窗口口径：

```text
window 01 : [-120, -110) s
window 02 : [-115, -105) s
window 03 : [-110, -100) s
...
window 22 : [ -15,   -5) s
window 23 : [ -10,    0) s
```

每个窗口包含：

```python
{
    "eeg":   (256,),
    "wear":  (256,),
    "video": (256,),
    "audio": (256,),
    "modality_mask": (4,),
}
```

一个 EMA event 的模型输入：

```text
tokens:        (L=23, M=4, 256)
modality_mask: (L=23, M=4)
label:  scalar 或 D 维情绪评分
```

DataLoader 后：

```text
tokens:        (B, 23, 4, 256)
modality_mask: (B, 23, 4)
y:             (B, D)
```

---

## 3.2 一个 EMA = 一个监督单元

**不要再把一个 EMA 的 23 个重叠窗口当作 23 个独立有标签样本。**

旧方式：

```text
x[-120,-110] -> y_i
x[-115,-105] -> y_i
...
x[-10,0]     -> y_i
```

新方式：

```text
{x_1, x_2, ..., x_23} -> y_i
```

其中 23 个窗口全部参与前向和反向传播，但只有一个真实的 bag-level EMA supervision。

这可以理解为：

```text
structured multiple-instance regression
```

而不是简单的 window-level regression。

---

## 3.3 数据划分

主实验沿用当前 EEG-aligned 工作中已经使用的三种协议：

```text
cross_subject
cross_day
within_subject_day
```

每个协议继续使用现有 split 文件中的：

```text
train = pretrain + finetune
validation = val
test = test
```

必须满足：

- 同一 EMA bag 的 23 个窗口不允许拆到不同 split。
- normalization 仅使用当前协议的 train 部分拟合。
- validation / test 不参与任何统计量拟合。
- 三种协议分别报告，不能把不同协议的窗口、EMA 或指标混成一个主结果。

后续同被试 held-out-day 实验统一使用 repaired `within_subject_day`；新的 subject-independent `8:1:1` split 应作为额外协议单独命名、单独报告，不覆盖当前三协议口径。

---

# 4. Step 1：Shared 与 Per-modality 双线路

## 4.1 为什么要并列尝试两条线路

现有实现将四种 256D embedding 统一送入：

```python
Linear(256, hidden_dim)
```

并在训练集上跨模态统计 mean / std。仓库当前也已经支持 `--token-normalization per_modality`，因此本计划不把 per-modality 预设为天然更优，而是把 shared 和 per-modality 作为地位相同的两条待验证线路。

第一条线路保留当前共享输入层：

```text
Shared Norm + Shared Linear(256 -> 128)
```

第二条线路使用模态独立归一化与 adapter：

```text
EEG   -> Norm_EEG   -> Adapter_EEG
Wear  -> Norm_Wear  -> Adapter_Wear
Video -> Norm_Video -> Adapter_Video
Audio -> Norm_Audio -> Adapter_Audio
```

每种模态单独在 train split 上计算：

```text
mean_m: (256,)
std_m : (256,)
```

只对该模态有效样本计算统计量。

---

## 4.2 Per-modality adapter 推荐实现

第一版保持轻量：

```python
class ModalityAdapter(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=128, dropout=0.1):
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
```

四个模态分别实例化，不共享参数：

```python
self.adapters = nn.ModuleDict({
    "eeg":   ModalityAdapter(256, 128),
    "wear":  ModalityAdapter(256, 128),
    "video": ModalityAdapter(256, 128),
    "audio": ModalityAdapter(256, 128),
})
```

per-modality 线路得到：

```text
h: (B, L, M, 128)
```

两条线路都可继续保留 learned modality embedding：

```text
h_t^m <- h_t^m + e_m
```

但它只表示 modality identity，不承担空间对齐功能。

---

# 5. Step 2：连续潜在情绪状态

## 5.1 状态变量

定义：

```text
z_t ∈ R^128
```

表示第 `t` 个 10 s 时间窗口之后的潜在 affect state。

对于一个 EMA bag：

```text
z_1, z_2, ..., z_23
```

共同描述评分前 2 分钟的连续情绪轨迹。

所有 bag 使用一个可学习的初始状态：

```python
self.z0 = nn.Parameter(torch.zeros(128))
```

第一篇暂时不加入 subject embedding、personal memory 或长期状态。

---

# 6. Step 3：Prior-guided multimodal state update

这是模型的核心模块。

每一个时间点执行三步：

```text
previous state
    ↓
predict prior
    ↓
read multimodal observations
    ↓
update state
```

---

## 6.1 State prior

对于 `t = 1`：

```text
p_1 = z_0
```

对于 `t > 1`：

```text
p_t = Transition(z_{t-1})
```

推荐：

```python
self.transition = nn.Sequential(
    nn.LayerNorm(128),
    nn.Linear(128, 128),
    nn.GELU(),
    nn.Linear(128, 128),
)

p_t = LayerNorm(z_prev + transition(z_prev))
```

不要第一版直接上 Transformer / Mamba。23 个时间点仍然较短，轻量 state transition 足够验证核心假设。

---

## 6.2 Prior-guided modality weighting

当前时间点有：

```text
h_t^EEG
h_t^Wear
h_t^Video
h_t^Audio
```

不再让四个 token 先完全自由 self-attention，而是由当前 affect prior `p_t` 查询各模态。

对每个模态计算：

```text
s_t^m = Score_m(p_t, h_t^m)
```

推荐 additive attention：

```python
score_m = v_m^T tanh(Wp * p_t + Wh_m * h_t_m)
```

其中 `Wp` 可共享，`Wh_m` 和 `v_m` 可按模态独立。

对缺失模态：

```python
score = score.masked_fill(~modality_mask, -1e9)
```

随后：

```text
alpha_t = softmax(score_t over modalities)
```

这里的 `alpha_t^m` 只解释为：

> 在当前 affect prior 下，该模态对状态更新的相对贡献。

**不要解释成噪声水平或信号质量。**

---

## 6.3 Multimodal evidence

每个模态增加独立 value projection：

```text
v_t^m = V_m(h_t^m)
```

融合得到：

```text
e_t = Σ_m alpha_t^m * v_t^m
```

其中：

```text
e_t ∈ R^128
```

---

## 6.4 Prior update

让模型学习当前 evidence 应该对 prior 修正多少。

推荐：

```python
update_input = concat([
    p_t,
    e_t,
    e_t - p_t,
])

delta_t = tanh(MLP_delta(update_input))
gate_t  = sigmoid(MLP_gate(concat([p_t, e_t])))

z_t = LayerNorm(p_t + gate_t * delta_t)
```

其中：

```text
delta_t: (128,)
gate_t : (128,)  # vector gate
```

直观解释：

```text
p_t     = 根据过去状态形成的当前 prior
 e_t    = 当前四模态提供的 evidence
delta_t = 当前 observation 建议的修正方向
gate_t  = 应该修正多少
 z_t    = 更新后的 posterior affect state
```

完整递归：

```text
z0
 ↓
p1 -> observations_1 -> z1
 ↓
p2 -> observations_2 -> z2
 ↓
...
 ↓
p23 -> observations_23 -> z23
```

---

# 7. Step 4：Dynamic EMA Time Kernel

## 7.1 核心动机

对于第 `i` 次 EMA，不能假设：

```text
z_1 = z_2 = ... = z_23 = y_i
```

而应假设 EMA 是过去一段连续 affect trajectory 的某种时间整合：

```text
y_i ≈ H(z_1, ..., z_23)
```

第一版使用受约束的 **Dynamic Mixture-of-Decay Kernel**，避免 unrestricted temporal attention 在小样本下过拟合。

---

## 7.2 三个时间尺度 basis

固定三个 decay basis：

```text
tau_1 = 15 s
tau_2 = 45 s
tau_3 = 120 s
```

对距离 EMA 的时间差 `Δt`：

```text
k_j(Δt) = exp(-Δt / tau_j)
```

对于 23 个 10 s window，可使用各 window 的中心点：

```text
Δt = [115, 110, 105, ..., 10, 5] s
```

得到固定 basis matrix：

```text
K_basis: (23, 3)
```

---

## 7.3 每次 EMA 动态预测 mixture weight

对于一个 EMA bag，从状态轨迹构造一个轻量 summary：

```text
u_i = concat([
    z_23,
    mean(z_1...z_23),
    z_23 - z_1,
])
```

维度：

```text
u_i: 384D
```

通过小 MLP：

```python
pi_i = softmax(kernel_head(u_i))
```

得到：

```text
pi_i = [pi_short, pi_medium, pi_long]
```

满足：

```text
pi_j >= 0
sum(pi_j) = 1
```

每次 EMA 的最终时间核：

```text
K_i(Δt) = Σ_j pi_i,j * exp(-Δt / tau_j)
```

再对 23 个时间点归一化：

```text
w_i,t = K_i(Δt_t) / Σ_t K_i(Δt_t)
```

最终 EMA representation：

```text
r_i = Σ_t w_i,t * z_i,t
```

---

## 7.4 为什么第一版固定 tau

第一版先固定：

```text
[15, 45, 120] s
```

只动态学习每个 EMA 的 mixture coefficient `pi`。

这样既是动态 kernel，又只有 3 个自由权重，比直接学习 23 个 arbitrary temporal attention weights 更适合稀疏 EMA supervision。

后续 ablation 再测试：

```text
fixed tau
vs.
learnable positive tau
```

不要把 learnable tau 作为主模型必要条件。

---

## 7.5 多维情绪评分

如果最终输出为多个 emotion dimensions：

```text
y_i ∈ R^D
```

推荐两阶段执行：

### 主实验

所有情绪维度共享一个 dynamic kernel：

```text
r_i -> multi-output regression head
```

优点：参数少，更稳健。

### 扩展实验

为每个情绪维度单独预测：

```text
pi_i^(d)
```

用于探索不同情绪维度是否具有不同的有效回溯时间尺度。

第一版不要一开始就使用 dimension-specific kernel。

---

# 8. Regression Head

EMA representation：

```text
r_i: (128,)
```

推荐回归头：

```python
self.regression_head = nn.Sequential(
    nn.LayerNorm(128),
    nn.Linear(128, 64),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(64, output_dim),
)
```

输出：

```text
y_hat: (B, D)
```

目标值仅使用 train split 统计 mean/std 进行标准化；val/test 只应用 train statistics。

---

# 9. Loss 设计

第一版保持简单。

## 9.1 主损失

优先使用 Huber loss：

```text
L_raw = Huber(y_hat, y)
```

相较纯 MSE，对 EMA 中少量异常评分更稳健。

如果需要严格继承当前代码，可同时保留 MSE 版本作为 baseline。

---

## 9.2 Within-subject centered loss

保留现有 centered-loss 思路，但它只作为辅助监督：

```text
L = L_raw + lambda_center * L_center
```

推荐初始：

```text
lambda_center = 0.2
```

`L_center` 用于鼓励模型捕捉同一个体内部随时间变化的情绪，而不是主要依赖 subject baseline。

实现时 batch sampler 应保证一个 batch 中尽量包含同一 subject 的多个 EMA event；如果某 subject 在当前 batch 只有一个 EMA，则该 subject 不计算 centered term。

**不要加入额外 future prediction、reconstruction、contrastive loss 或显式 noise loss。**

---

# 10. 训练时数据增强与鲁棒性

本版本不做显式噪声检测。

只保留一个简单、通用的增强：

## Modality dropout

训练时，以较小概率随机屏蔽某一整个模态：

```text
p_modality_drop = 0.10 ~ 0.20
```

要求：

- 至少保留一个有效模态。
- dropout 后同步修改 `modality_mask`。
- 不额外预测“缺失模态是什么”。

作用：

1. 防止模型过度依赖单一模态。
2. 提高现实缺失情况下的鲁棒性。
3. 为 missing-modality evaluation 提供合理训练基础。

第一版不加入人工噪声强度、corruption ranking 或 SQI supervision。

---

# 11. 推荐训练配置

```text
input_dim             = 256
hidden_dim            = 128
lookback_sec          = 120
window_sec            = 10
hop_sec               = 5
num_temporal_windows  = 23
num_modalities        = 4
kernel_basis_tau      = [15, 45, 120]
dropout               = 0.10
modality_dropout      = 0.10
lambda_center         = 0.20
optimizer             = AdamW
learning_rate         = 1e-3
weight_decay          = 1e-4
batch_size            = 16 or 32 EMA bags
max_epochs            = 100
early_stop_patience   = 10-15
```

建议：

- 主方法优先使用 label-free / frozen 单模态 token；若复用 `eegpt_partial_ft_v1`、`Wmoment_*` 等 fatigue-supervised token，必须在实验名和结论中标为监督控制线。
- 在 EMA-level 模型阶段只训练 adapter、state model、kernel 和 regression head，不端到端更新四个大 encoder。
- 使用 5 个随机种子报告 mean ± std。
- 以 validation CCC 或 MAE 做 early stopping，选一个并全程固定。

---

# 12. 代码模块建议

建议不要继续把所有逻辑堆到现有单个训练脚本中。

新增：

```text
src/
├── datasets/
│   └── ema_bag_dataset.py
│
├── models/
│   ├── modality_adapter.py
│   ├── affect_state_filter.py
│   ├── dynamic_ema_kernel.py
│   └── daily_affect_model.py
│
├── training/
│   ├── losses.py
│   └── samplers.py
│
└── evaluation/
    └── metrics.py

scripts/
├── 68_build_ema_event_bags.py
├── 69_train_daily_affect_state.py
├── 70_eval_daily_affect_state.py
└── 71_run_daily_affect_ablation.py
```

编号 `68-71` 顺延当前顶层脚本，避免与已有 calibration `39-47`、multi-label `48`、EQL-CAF `57-64` 和 Wear-only FM `65-67` 冲突。

---

# 13. 与 EQL-CAF 的边界

EQL-CAF 是当前正在尝试的另一种 temporal multimodal 方法，代码位于：

```text
src/daily_multimodal/temporal/
src/daily_multimodal/training/eql_caf.py
scripts/57-64
```

本路线与 EQL-CAF 必须保持独立：

- **数据输入不互相冒用**：本路线以 EMA event 为监督单元，输入为当前 EEG-aligned 主线的 23 个 10 秒重叠窗口；EQL-CAF 当前围绕每个 10 秒窗口内部的 5 个 2 秒 temporal token 契约展开。
- **实验命名不互相覆盖**：本路线使用 `daily_affect_state_*` 或 `ema_event_bag_*` 前缀；EQL-CAF 保留 `eql_caf_*` 前缀。
- **报告结论不互相借用**：EQL-CAF 的 `global_repeat_smoke_v1` 只验证 temporal token 管线、shape、mask 和训练入口；不能作为本路线的 EMA-level 动态时间聚合证据。本路线的 EMA bag 结果也不能作为 EQL-CAF 窗口内 2 秒 token 机制证据。
- **协议分别记录**：两条路线可以使用相同的 split 名称做对照，但必须分别保存配置、输入 manifest、模型名、seed 和输出报告。

---

# 14. 推荐类接口

## 14.1 Dataset

```python
class EMABagDataset(Dataset):
    def __getitem__(self, idx):
        return {
            "tokens": ...,        # (23, 4, 256)
            "modality_mask": ..., # (23, 4)
            "label": ...,        # (D,)
            "subject_id": ...,
            "ema_id": ...,
            "ema_time": ...,
        }
```

---

## 14.2 AffectStateFilter

```python
class AffectStateFilter(nn.Module):
    def forward(self, tokens, modality_mask):
        """
        tokens:        (B, L, M, 256)
        modality_mask: (B, L, M)

        returns
        -------
        states:          (B, L, 128)
        modality_weight: (B, L, M)
        """
```

---

## 14.3 DynamicEMAKernel

```python
class DynamicEMAKernel(nn.Module):
    def forward(self, states):
        """
        states: (B, L, 128)

        returns
        -------
        pooled:         (B, 128)
        temporal_weight:(B, L)
        mixture_weight: (B, 3)
        """
```

---

## 14.4 DailyAffectModel

```python
class DailyAffectModel(nn.Module):
    def forward(self, tokens, modality_mask):
        states, modality_weights = self.state_filter(tokens, modality_mask)
        pooled, temporal_weights, mixture_weights = self.ema_kernel(states)
        pred = self.regression_head(pooled)

        return {
            "pred": pred,
            "states": states,
            "modality_weights": modality_weights,
            "temporal_weights": temporal_weights,
            "kernel_mixture": mixture_weights,
        }
```

这些中间输出全部保存，后续用于可解释性分析。

---

# 15. Baseline 与消融实验

必须把“模型结构改进”和“EMA supervision 改进”分开验证。

## 15.1 Baseline 0：当前模型

保持原实现：

```text
每个窗口独立
4 modality tokens
-> shared projection
-> modality self-attention
-> query pooling
-> regression
```

EMA 前 2 min 内窗口复制同一标签。

该结果作为历史 baseline，不作为最终公平比较的唯一基线。

---

## 15.2 Baseline 1：Bag-level static fusion

保持当前窗口级 attention fusion，但改为：

```text
23 个窗口分别得到 fused representation
-> uniform mean pooling over time
-> EMA-level regression
```

此时一个 EMA 只计算一次 loss。

目的：单独验证“EMA 作为 bag-level supervision”是否有效。

---

## 15.3 Ablation A：Shared / Per-modality 双线路

```text
Bag-level static fusion
+ shared normalization + shared projection
vs.
Bag-level static fusion
+ per-modality normalization + per-modality adapter
```

两条线路地位相同，均需在相同协议、相同 seed、相同输入 bag 上尝试；根据 paired 结果决定是否进入后续 state/kernel 模块。

---

## 15.4 Ablation B：Latent affect state

加入连续：

```text
z_1 -> z_2 -> ... -> z_23
```

但 EMA 端仍使用 uniform temporal mean。

验证连续状态建模本身的价值。

---

## 15.5 Ablation C：Prior-guided update

比较：

```text
普通 temporal GRU
vs.
prior-guided multimodal update
```

验证“previous affect prior 主动查询当前 multimodal evidence”是否优于直接把融合特征输入 GRU。

---

## 15.6 Full model：Dynamic EMA kernel

```text
Modality-specific adapter
+ latent affect state
+ prior-guided update
+ dynamic EMA kernel
```

即本工作的最终主模型。

---

# 16. Dynamic EMA Kernel 专项消融

建议单独做一张表：

| Temporal supervision / pooling | 说明 |
|---|---|
| Window label replication | 当前做法 |
| Last-window only | 只使用 EMA 前最后 10 s |
| Uniform mean | 2 min 内等权平均 |
| Fixed exponential | 单一固定 decay |
| Global mixture-of-decay | 全部 EMA 共用一组 mixture weight |
| **Dynamic mixture-of-decay** | 每次 EMA 动态预测 mixture weight |

这一组实验直接回答：

> EMA 是否应该被视为一个时刻标签，以及动态时间积分是否有必要？

---

# 17. Look-back window 消融

主设置：

```text
120 s
```

建议额外测试：

```text
30 s
60 s
120 s
300 s
```

保持其他模型结构不变。

目的不是为了暴力寻找最佳超参数，而是回答一个有科学意义的问题：

> 日常 EMA 的有效多模态信息主要来自多长时间范围？

如果结果呈现：

```text
30 s < 60 s < 120 s ≈ 300 s
```

则可以较有说服力地支持 2 min history 的选择。

---

# 18. 模态消融与缺失模态实验

至少报告：

```text
EEG + Wear + Video + Audio
EEG + Wear + Video
EEG + Wear + Audio
EEG + Video + Audio
Wear + Video + Audio
EEG only
Wear only
Video only
Audio only
```

同时增加 test-time missing modality：

```text
随机缺失 1 个模态
随机缺失 2 个模态
```

注意：这里只评价 robustness，不声称模型显式估计噪声或可靠性。

---

# 19. 评价指标

对于连续情绪回归，建议主指标：

```text
MAE
RMSE
Pearson r
CCC
```

如果为多维输出，对每个情绪维度分别报告，同时报告宏平均。

另外建议增加：

```text
within-subject centered Pearson r / CCC
```

用于评估模型是否真正追踪一个体内部的日常情绪波动，而不只是学习稳定的个体差异。

统计上：

- 以 subject 为 bootstrap 单位计算 95% CI。
- 模型比较优先在 subject-level metric 上做 paired test。
- 不要把 23 个 window 当成 23 个独立统计样本。

---

# 20. 可解释性输出

模型训练时保存：

```text
modality_weights: (N_ema, 23, 4)
temporal_weights: (N_ema, 23)
kernel_mixture:   (N_ema, 3)
states:           (N_ema, 23, 128)
```

建议至少做三组图。

## 20.1 EMA 时间核

画：

```text
平均 temporal kernel
+ 个体 EMA kernel 示例
+ short / medium / long mixture coefficient 分布
```

回答：

> EMA 更偏向最近状态，还是整合较长时间历史？

---

## 20.2 模态权重随时间变化

对单个 EMA event：

```text
时间轴 × EEG/Wear/Video/Audio 权重
```

只解释为模型使用不同模态 evidence 的相对程度，不解释为生理“重要性”或“噪声大小”。

---

## 20.3 Latent affect trajectory

对 `z_1...z_23`：

- PCA / UMAP 仅用于展示；
- 比较高分与低分 EMA 前的 trajectory；
- 比较状态变化速度与最终 EMA 的关系。

该部分作为探索性结果，不作为第一版训练目标。

---

# 21. 推荐开发顺序

严格按以下顺序实现，避免一次性改动过多导致无法定位增益来源。

## Phase 0：复现当前 baseline

确认现有：

```text
attention variant
raw MSE / centered loss
4-modal / 3-modal / 2-modal
```

结果完全可复现。

---

## Phase 1：重构为 EMA bag

只修改 Dataset / DataLoader：

```text
(B, 4, 256)
```

改成：

```text
(B, 23, 4, 256)
```

模型仍使用现有 attention，在每个时间点独立融合；最后对 23 个时间点 uniform mean。

首先验证 pipeline 正确。

**验收条件：**

- sample_id / EMA id 对齐无误；
- 同一 EMA 的 23 个窗口完全在同一 split；
- train-only normalization 无泄漏；
- loss 数量等于 EMA 数，而不是窗口数。

---

## Phase 2：Shared / Per-modality 双线路筛选

在 Phase 1 的 EMA bag 静态融合基础上，同时运行两条线路：

```text
shared normalization + shared Linear
vs.
per-modality normalization + per-modality adapter
```

除输入归一化/adapter 外，其他结构不动。

**验收条件：**

- shared 与 per-modality 使用相同协议、相同 seed、相同 EMA bag 输入；
- per-modality 分支的四个 adapter 参数互不共享；
- 两条线路的缺失模态 `modality_mask` 行为与原实现一致；
- 与 Phase 1 做单变量对比，并按 paired 结果决定后续 Phase 3 采用哪条或两条都保留。

---

## Phase 3：实现 AffectStateFilter

加入：

```text
state transition
prior-guided modality weighting
evidence aggregation
state update gate
```

EMA 端先继续使用 uniform temporal mean。

**验收条件：**

- 输出 `states.shape == (B, 23, 128)`；
- 每个时间点有效 modality weights 之和为 1；
- 缺失模态权重为 0；
- 能输出和可视化连续 state trajectory。

---

## Phase 4：实现 Dynamic EMA Kernel

加入：

```text
3 fixed decay bases
sample-dependent mixture weights
normalized temporal weights
```

得到完整模型。

**验收条件：**

- `temporal_weights.sum(dim=1) == 1`；
- `kernel_mixture.sum(dim=1) == 1`；
- 不读取 label 生成 kernel；
- kernel 在不同 EMA 间确实可变化。

---

## Phase 5：完整实验

按以下顺序运行：

```text
1. baseline / ablation
2. kernel ablation
3. look-back ablation
4. modality ablation
5. test-time missing modality
6. interpretability
```

第一版不要并行加入新的自监督或个体化模块。

---

# 22. 第一篇建议明确不做的内容

为控制复杂度，本版本明确不加入：

```text
显式 noise / SQI estimator
reliability supervision
future latent prediction
world model objective
fast/slow 双状态模型
diffusion / flow matching
missing-modality reconstruction
subject ID embedding
long-term personal memory
端到端 fine-tuning 四个大 encoder
```

这些内容都可作为后续工作。

---

# 23. 最终方法定义

整项工作可以概括为：

> 日常情绪不是每个 10 秒窗口独立产生的静态标签，而是一个随时间连续演化的潜在状态；EEG、Wear、Video 和 Audio 是对该状态的异构观测，而稀疏 EMA 是对近期潜在状态轨迹的时间整合性测量。

对应模型：

```text
heterogeneous multimodal observations
        ↓
modality-specific affect space
        ↓
continuous latent affect filtering
        ↓
prior-guided multimodal state update
        ↓
dynamic EMA temporal integration
        ↓
daily affect regression
```

最核心的两个方法贡献：

1. **Prior-guided multimodal affect state filtering**：利用上一时刻的潜在情绪状态作为 prior，动态整合当前四模态 evidence，而不是对每个窗口做独立静态融合。
2. **Dynamic EMA observation kernel**：将一次稀疏 EMA 视为对近期潜在情绪轨迹的动态时间积分，避免将同一 EMA 标签机械复制给多个窗口。

该路线在保留当前预训练单模态 embedding 和整体训练框架的基础上，改动集中、可逐步消融、适合样本量有限的日常 EMA 数据，也为后续加入个体化、长程动力学和 world-model prediction 保留扩展空间。
