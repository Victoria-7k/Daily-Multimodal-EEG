# 日常情境四模态疲劳等级分类：可执行技术路线

**版本：2026-08-27**  
**目标：在现有 EEG / Wear / Video / Audio 四模态 256D embedding 基础上，将“窗口级静态融合 + 标签复制”升级为“连续潜在疲劳状态估计 + EMA 级动态时间聚合”，最终预测 1–5 级有序疲劳标签。**

---

## 1. 研究问题与总体原则

当前实现中，每个时间窗口被表示为 4 个模态 token：

```text
EEG   : 256D
Wear  : 256D
Video : 256D
Audio : 256D
```

随后通过共享 `Linear(256 -> hidden_dim)`、modality embedding、模态间 self-attention、learnable query pooling 和 MLP 分类头预测 1–5 级疲劳标签。窗口级缺失模态通过 `modality_mask` 屏蔽。

该方案适合作为 baseline，但用于日常情境时有两个主要限制：

1. **四种模态的 embedding 来自不同 encoder，当前默认路线使用共享输入投影和共享 train-only 统计归一化；仓库也已经提供 per-modality normalization 对照入口，但尚未把 shared 或 per-modality 任一方确认为最终默认。**
2. **当前 EEG-aligned 主线中，同一个 EMA event 展开为 23 个 10 秒窗口、5 秒 stride；窗口继承同一事件标签。直接把这些窗口当作独立监督样本，会把一次真实监督扩展为多个高度相关的样本，并隐含评分前两分钟内每个重叠窗口都可独立承载同一标签的假设。**

本工作的主线不再强调“设计更复杂的 cross-attention”，而是将问题改写为：

> 从连续、异构、可能部分缺失的 EEG、Wear、Video 和 Audio 观测中估计随时间演化的潜在疲劳状态，并学习稀疏 EMA 疲劳等级如何由评分前一段时间的潜在状态轨迹形成。

本版本**不加入显式噪声检测或 SQI 评分**。模型在普通动态模态权重基础上，增加由累计序数 Probe 的预测熵与等级方差共同形成的 ordinal-difficulty bias；该偏置只表示模型的序数预测不确定性，不解释为物理信号质量。鲁棒性通过训练时的 modality dropout、测试时的 missing-modality evaluation 和模态扰动实验共同检验。

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
   prior compatibility + ordinal-difficulty weighting
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
                 Ordinal classification head
                           │
                           ▼
                    1–5级分类概率
```

核心模块只保留四项：

1. **Modality-specific adaptation**：将不同 encoder 的 embedding 映射到共同 fatigue/affect space。
2. **Temporal latent fatigue state**：显式建模评分前 2 分钟内疲劳状态的连续演化。
3. **Ordinal-difficulty-regularized prior-guided multimodal state update**：上一时刻状态形成 prior；当前四模态 evidence 根据状态匹配度、预测熵和等级跨度共同更新 prior。
4. **Dynamic EMA kernel**：将一次 EMA 建模为对过去一段潜在疲劳轨迹的加权观测，而不是给每个窗口复制标签。

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
label:         1–5级整数疲劳标签
```

DataLoader 后：

```text
tokens:        (B, 23, 4, 256)
modality_mask: (B, 23, 4)
y:             (B,)  # int64，取值映射为 0...4 供 CrossEntropy 使用
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
structured multiple-instance ordinal classification
```

主方法最终只在 EMA bag 层面输出一次五分类结果，而不是对 23 个窗口分别计算分类损失。Phase 0 另保留 `window_replicated` 对照：它复用完全相同的窗口级 attention/query-pooling 和分类目标，对每个有效窗口复制 event 标签计算损失，再以窗口概率均值形成 event-level 测试预测；该对照专门检验 supervision unit 的作用。

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

- 同一 EMA bag 的 23 个窗口必须映射为一个 bag-level split；单一 leaf split 直接保留，`pretrain/finetune` 训练叶子可合并为 train，train/val/test 边界混合按窗口多数票投影并记录审计字段。
- normalization 仅使用当前协议的 train 部分拟合。
- validation / test 不参与任何统计量拟合。
- 三种协议分别报告，不能把不同协议的窗口、EMA 或指标混成一个主结果。

如果后续需要接入 `within_subject_day_strict` 或新的 subject-independent `8:1:1` split，应作为额外协议单独命名、单独报告，不覆盖当前三协议口径。

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

# 5. Step 2：连续潜在疲劳状态

## 5.1 状态变量

定义：

```text
z_t ∈ R^128
```

表示第 `t` 个 10 s 时间窗口之后的潜在 fatigue state。

对于一个 EMA bag：

```text
z_1, z_2, ..., z_23
```

共同描述评分前 2 分钟的连续疲劳轨迹。

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

不再让四个 token 先完全自由 self-attention，而是由当前 fatigue-state prior `p_t` 查询各模态。模态权重同时考虑：

1. 当前模态与历史状态先验是否匹配；
2. 该模态的辅助 Probe 对本次 EMA 疲劳等级预测是否确定。

### 6.2.1 EMA-level modality Probe

由于一次 EMA 只有一个真实标签，第一版不对 23 个窗口分别计算 Probe loss。对每个模态先在有效时间点上做 masked mean：

```text
g_i^m = Σ_t M_i,t^m h_i,t^m / (Σ_t M_i,t^m + eps)
```

其中 `g_i^m ∈ R^128`。每个模态使用独立的 cumulative ordinal Probe，而不是普通五维 softmax Probe。

先将 `g_i^m` 投影到一条共享的疲劳等级轴：

```text
a_i^m = w_m^T g_i^m + b_m
```

再预测四个累计概率：

```text
l_i,k^m = a_i^m - theta_m,k
q_i,k^m = sigmoid(l_i,k^m / T_m) = P(y_i > k | g_i^m)
k ∈ {1,2,3,4}
```

阈值必须满足：

```text
theta_m,1 < theta_m,2 < theta_m,3 < theta_m,4
```

实现时使用正增量保证单调性：

```text
theta_m,k = theta_m,k-1 + softplus(delta_m,k)
```

因此累计概率天然满足：

```text
q_i,1^m >= q_i,2^m >= q_i,3^m >= q_i,4^m
```

由累计概率恢复五类概率：

```text
p_i,1^m = 1 - q_i,1^m
p_i,2^m = q_i,1^m - q_i,2^m
p_i,3^m = q_i,2^m - q_i,3^m
p_i,4^m = q_i,3^m - q_i,4^m
p_i,5^m = q_i,4^m
```

从而保证 `p_i,c^m >= 0` 且 `Σ_c p_i,c^m = 1`。

### 6.2.2 Ordinal difficulty

普通预测熵只能衡量概率是否分散，不能区分“在 1/2 级之间犹豫”和“在 1/5 级之间犹豫”。因此将困难度定义为归一化熵与等级方差的凸组合。

归一化熵：

```text
D_H,i^m = -Σ_c p_i,c^m log(p_i,c^m+eps) / log(5)
```

等级期望与方差：

```text
mu_i^m = Σ_c c * p_i,c^m
V_i^m  = Σ_c (c-mu_i^m)^2 * p_i,c^m
D_V,i^m = V_i^m / 4
```

1–5 级分布的最大方差为 4，因此 `D_H`、`D_V` 均位于 `[0,1]`。最终序数困难度：

```text
D_ord,i^m = (1-beta_ord) * D_H,i^m + beta_ord * D_V,i^m
```

推荐第一版 `beta_ord=0.25`。例如 `[0.5,0.5,0,0,0]` 与 `[0.5,0,0,0,0.5]` 熵相同，但后者等级方差更大，因此得到更高的 `D_ord` 和更强的路由降权。

未校准版本固定 `T_m=1`；校准版本再按模态在 validation split 上拟合 temperature。第一版的 `D_ord,i^m` 表示该模态在整次 EMA bag 上的序数预测不确定性，并在该 EMA 的 23 个时间点共享。它不是 SQI，也不是物理噪声评分。

该设计面向“到达 EMA 时刻后，利用之前完整 2 min 数据进行离线等级判断”的场景。由于 `D_ord,i^m` 汇总了完整 bag，它不用于声称每个中间状态都是严格在线因果估计；若后续要求实时在线输出，应改为 prefix-only 或 time-varying MIL Probe。

若某模态在整次 EMA 内都无效，则该模态不计算 Probe loss，并继续由 `modality_mask` 在所有时间点屏蔽。

### 6.2.3 Prior compatibility score

对每个模态计算当前特征与状态先验的匹配分数：

```text
s_t^m = Score_m(p_t, h_t^m)
```

推荐 additive attention：

```python
score_m = v_m^T tanh(Wp * p_t + Wh_m * h_t_m)
```

其中 `Wp` 可共享，`Wh_m` 和 `v_m` 可按模态独立。

### 6.2.4 Ordinal-difficulty-regularized routing

将序数困难度作为匹配分数的弱惩罚项：

```text
routing_logit_i,t^m = s_i,t^m / tau_s - lambda_D * sg(D_ord,i^m)
```

其中 `sg()` 表示 stop-gradient：前向使用当前熵值，主分类损失不能通过路由权重反向操纵 Probe 熵；Probe 仍由自身的 EMA-level 辅助分类损失训练。

对缺失模态：

```python
routing_logit = routing_logit.masked_fill(~modality_mask, -1e9)
```

随后：

```text
alpha_t = softmax(routing_logit_t over modalities)
```

这里的 `alpha_t^m` 只解释为：

> 综合当前状态先验匹配程度与模型预测不确定性后，该模态对状态更新的相对贡献。

**不要将 `D_ord,i^m` 解释成信号质量，也不要将 `alpha_t^m` 解释成生理重要性。**单模态 Probe 置信度只是一项路由偏置；模态即使单独分类能力较弱，也可能包含有价值的互补信息，因此第一版使用较小的 `lambda_D`。

该机制仍不能识别测试时的“低熵、低方差但自信地预测到错误远端等级”。例如真实为 1 级、Probe 集中预测为 5 级时，`D_ord` 仍可能接近 0。训练阶段的序数 Probe loss 会强烈惩罚这种远距离错误；测试阶段没有真实标签，仍需依靠校准、模型集成、OOD 或额外质量证据处理这一边界。

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
 z_t    = 更新后的 posterior fatigue state
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

而应假设 EMA 疲劳等级是过去一段连续 fatigue trajectory 的某种时间整合：

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

## 7.5 有序疲劳等级

当前主任务固定为：

```text
y_i ∈ {1, 2, 3, 4, 5}
```

五个等级具有明确顺序，Dynamic EMA Kernel 先生成一个共享的 EMA representation：

```text
r_i -> 5-class ordinal classification head
```

第一版不为不同等级分别预测时间核，也不加入 dimension-specific kernel。等级顺序由第 9 节的 ordinal loss 显式编码。

---

# 8. Ordinal Classification Head

EMA representation：

```text
r_i: (128,)
```

推荐分类头：

```python
self.classification_head = nn.Sequential(
    nn.LayerNorm(128),
    nn.Linear(128, 64),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(64, 5),
)
```

输出：

```text
logits: (B, 5)
probs:  softmax(logits), shape (B, 5)
predicted_level: argmax(probs) + 1
```

标签在数据层保持 1–5 语义，进入 `CrossEntropyLoss` 前映射为 `0...4`。类别权重、先验类别频率和任何采样权重都只能由当前协议的 train split 估计。

---

# 9. Loss 设计

主任务明确为 1–5 级有序分类。普通交叉熵负责五类判别，ordinal term 负责区分“错相邻一级”和“跨多级错误”。

## 9.1 主分类损失

令：

```text
p_i = softmax(logits_i)
```

主分类项使用 class-weighted cross entropy；类别权重只根据 train split 计算：

```text
L_CE = -w_y log p_i,y
```

为显式利用等级顺序，增加基于累计概率的 ordinal loss。对于 `k=1...4`：

```text
F_i,k = Σ_{c<=k} p_i,c
G_i,k = 1[y_i <= k]
L_ord = (1/4) Σ_k (F_i,k - G_i,k)^2
```

主任务损失：

```text
L_main = L_CE + lambda_ord * L_ord
```

推荐初始：

```text
lambda_ord = 0.5
```

---

## 9.2 Within-subject ordinal ranking loss

原回归版本的 centered loss 不再直接使用。为了继续鼓励模型捕捉同一个体内部的疲劳升降，使用可选的 subject-level pairwise ranking 辅助项。

先由类别概率计算期望等级：

```text
mu_i = Σ_{c=1}^5 c * p_i,c
```

对同一 subject 且标签不同的 EMA 对 `(i,j)`：

```text
L_rank(i,j) = softplus(-sign(y_i-y_j) * (mu_i-mu_j))
```

如果一个 batch 中没有同被试、不同等级的有效样本对，则该 batch 的 `L_rank=0`。推荐初始：

```text
lambda_rank = 0.1
```

## 9.3 Modality Probe loss

每个模态、每次 EMA 只计算一次 cumulative ordinal Probe loss。对于 `k=1...4`，目标定义为：

```text
t_i,k = 1[y_i > k]
```

单模态序数损失：

```text
L_probe-ord^m = (1/4) Σ_k BCEWithLogits(l_i,k^m, t_i,k)
L_probe = mean_valid_modalities[L_probe-ord^m]
```

若累计阈值的正负样本明显不平衡，可为每个阈值使用仅由 train split 计算的 `pos_weight_k`。若真实标签为 1 级，序数目标是 `[0,0,0,0]`：预测为 2 级主要跨错一个阈值，预测为 5 级会跨错四个阈值，因此远距离等级错误受到更大惩罚。完全缺失的模态不参与平均。推荐：

```text
lambda_probe = 0.1
```

总损失：

```text
L = L_main + lambda_rank * L_rank + lambda_probe * L_probe
```

Probe 的 `D_ord` 进入路由时执行 `stop-gradient`；`L_probe` 仍可更新 Probe 和 adapter。第一版不加入 future prediction、reconstruction、contrastive loss、显式 noise loss 或 SQI supervision。

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

## Probe warm-up 与序数困难度路由启用

训练初期 Probe 尚未形成有意义的累计类别概率，不能立即让随机 `D_ord` 主导模态权重。采用分阶段策略：

```text
Stage A（前 5–10 epoch）:
lambda_D = 0
T_m = 1
训练原 Prior-guided weighting、主分类头和各模态 Probe

Stage B（未校准序数困难度主实验）:
lambda_D 从 0 线性增加到目标值
继续联合训练，T_m = 1，D_ord 进入 routing 时保持 detach
```

temperature scaling 作为校准消融单独执行：从最佳未校准 checkpoint 出发，在 validation split 上分别为四个 Probe 拟合正温度 `T_m`，以恢复后的五类概率 NLL 为目标；同一个 `T_m` 同时缩放四个阈值 logits，因此不会破坏累计概率单调性。随后冻结 encoder、adapter、Probe 和 `T_m`，只微调 state filter、Dynamic EMA Kernel 与主分类头 5–10 epoch。每个交叉验证 fold 独立拟合，test split 不参与校准。这样可以避免校准后 Probe/adapter 继续变化导致温度失效。

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
num_classes           = 5
num_ordinal_thresholds= 4
tau_state_score       = 1.0
lambda_D              = 0.10 or 0.25
lambda_ord            = 0.50
lambda_rank           = 0.10
lambda_probe          = 0.10
beta_ord              = 0.25
probe_warmup_epochs   = 5-10
optimizer             = AdamW
learning_rate         = 1e-3
weight_decay          = 1e-4
batch_size            = 16 or 32 EMA bags
max_epochs            = 100
early_stop_patience   = 10-15
```

建议：

- 主方法优先使用 label-free / frozen 单模态 token；若复用 `eegpt_partial_ft_v1`、`Wmoment_*` 等 fatigue-supervised token，必须在实验名和结论中标为监督控制线。
- 在 EMA-level 模型阶段只训练 adapter、state model、kernel、classification head 和 modality Probe，不端到端更新四个大 encoder。
- 使用 5 个随机种子报告 mean ± std。
- 以 validation Macro-F1 或 QWK 做 early stopping，选一个并全程固定；推荐 QWK 作为有序等级主选择指标。
- `lambda_D` 主实验从 `{0, 0.1, 0.25, 0.5, 1.0}` 中通过 validation 选择，避免熵惩罚压制单模态的互补信息。
- `beta_ord` 从 `{0, 0.25, 0.5}` 中消融；`beta_ord=0` 等价于只使用序数 Probe 恢复概率的预测熵。

---

# 12. 已落地代码模块

当前路线已经从单个训练脚本拆成独立 daily-affect 包和 `73-82` 入口。`73-77` 是首轮全矩阵，`78-80` 是 `A1_Wphysio_full / cross_day / per_modality` 的 focused follow-up，`81` 对真实 state-prior compatibility 做独立的配对汇总，`82` 检验事件级标签时间范围、跨日 token drift 和 video 稳定性。

当前 `scripts/` 顶层已经占用 `12`、`15`、`16`、`27`、`32`、`34`、`39-56`、`57-64`、`65-72`。因此本路线从 `73` 开始编号，避免覆盖 Wear-only FM 的 `65-72` 和 EQL-CAF 的 `57-64`。

新增模块位于现有包命名空间下：

```text
src/daily_multimodal/daily_affect/
├── __init__.py
├── ema_bags.py
├── adapters.py
├── modality_probe.py
├── affect_state_filter.py
├── dynamic_ema_kernel.py
├── model.py
├── losses.py
├── samplers.py
├── metrics.py
├── temperature_scaling.py
└── diagnostics.py

scripts/
├── 73_build_daily_affect_bags.py
├── 74_run_daily_affect_phase0_baselines.py
├── 75_run_daily_affect_state_matrix.py
├── 76_summarize_daily_affect_results.py
├── 77_plot_daily_affect_diagnostics.py
├── 78_report_daily_affect_focused_diagnostics.py
├── 79_run_daily_affect_focused_ablation.py
├── 80_run_daily_affect_focused_robustness.py
├── 81_report_daily_affect_prior_guidance.py
└── 82_run_daily_affect_bottleneck_audit.py
```

## 12.1 路径常量

所有脚本使用同一组默认路径，并允许 CLI 覆盖：

```text
RUN_TAG=daily_affect_ordinal_20260903
ALIGN_ROOT=/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned
EMB_ROOT=/vePFS-0x0d/DailyEEG_multimodal/embeddings
INDEX_PATH=$ALIGN_ROOT/index/eeg_aligned_window_index.jsonl
SPLITS_ROOT=/vePFS-0x0d/DailyEEG/splits_new
OUT_ROOT=$ALIGN_ROOT/outputs/$RUN_TAG
LOCAL_SYNC_ROOT=outputs/server_sync/$RUN_TAG
```

其中 `RUN_TAG` 使用实际开跑日期；`OUT_ROOT` 是远端主产物目录；`LOCAL_SYNC_ROOT` 只是同步回本地后的镜像，不作为远端训练脚本的默认写入根目录。

## 12.2 读取输入

`73_build_daily_affect_bags.py` 只读取当前 EEG-aligned 主线窗口级 embedding，不读取 EQL-CAF 的 packed temporal token：

```text
index:
  $INDEX_PATH

splits:
  $SPLITS_ROOT/{protocol}/...
  protocol in {cross_subject, cross_day, within_subject_day}

EEG:
  $EMB_ROOT/eeg/eeg_eegpt_eeg23win_embeddings.npz
  $EMB_ROOT/eeg_encoder_256d_tokens/{protocol}/{profile}/seed_{eeg_seed}.npz

Wear:
  $EMB_ROOT/wear/wear_physio_preprocessed_eeg23win_embeddings.npz
  $EMB_ROOT/wear/wear_deep_sequence_preprocessed_eeg23win_embeddings.npz
  $EMB_ROOT/wear_tokens/{protocol}/{profile}/seed_{wear_seed}.npz

Video:
  $EMB_ROOT/video/video_B0_2xroi_eeg23win_embeddings.npz
  $EMB_ROOT/video/video_A1_2xroi_eeg23win_embeddings.npz
  $EMB_ROOT/video/video_A2_2xroi_eeg23win_embeddings.npz

Audio:
  $EMB_ROOT/audio/audio_opensmile_eeg23win_embeddings.npz
```

各模态 NPZ 沿用当前窗口级契约：`sample_id` 必须与 `INDEX_PATH` 完全同序；embedding 字段为 `eeg_emb`、`wear_emb`、`video_emb` 或兼容旧名 `face_emb`、`audio_emb`；mask 优先读取单模态 mask 字段，缺省可从窗口级 `modality_mask[:, modality_index]` 取得。daily-affect 的窗口级模态顺序固定为 `[eeg, wear, video, audio]`。当前正式运行使用 `/vePFS-0x0d/DailyEEG/splits_new`，因为该目录同时提供 `cross_subject`、`cross_day` 和 `within_subject_day` 三种主协议。

## 12.3 脚本职责与产物

`73_build_daily_affect_bags.py`：

- 读取 `INDEX_PATH`、三种 protocol split 和选定四模态窗口级 NPZ。
- 按 EMA event 聚合为 `(N_ema, 23, 4, 256)`，强校验每个 event 恰好包含 `event_window_id=0..22` 的 23 个 10 秒窗口。
- 写出：

```text
$OUT_ROOT/bags/{protocol}/{route_id}/seed_{seed}/ema_bags.npz
$OUT_ROOT/bags/{protocol}/{route_id}/seed_{seed}/ema_bag_manifest.jsonl
$OUT_ROOT/bags/{protocol}/{route_id}/seed_{seed}/bag_build_report.json
```

`ema_bags.npz` 建议字段：

```text
tokens:              (N_ema, 23, 4, 256)
modality_mask:       (N_ema, 23, 4)
label:               (N_ema,)        # fatigue 原始 1...5
label_zero_based:    (N_ema,)        # 0...4，供 CE 使用
event_id:            (N_ema,)
subject_id:          (N_ema,)
day_id:              (N_ema,)
sample_id_matrix:    (N_ema, 23)
event_window_id:     (23,)
split:               (N_ema,)
window_leaf_split_counts_json: scalar JSON string
source_npz_json:     scalar JSON string
route_id:            scalar string
supervision_boundary: scalar string
```

bag-level split 策略固定为 `event_split_policy=single_leaf_or_train_leaf_merge_or_window_majority`：单一 leaf split 直接保留；同一 event 只在 `pretrain/finetune` 内混合时归入训练集合；若现有窗口级 split 在 train/val/test 边界拆分同一 event，则按该 event 的 23 个窗口多数票投影为一个 bag-level split，并在 `window_leaf_split_counts_json`、`mixed_train_leaf_event_count` 和 `projected_boundary_event_count` 中留痕。

`74_run_daily_affect_phase0_baselines.py`：

- 读取 `$OUT_ROOT/bags/{protocol}/{route_id}/seed_{seed}/ema_bags.npz`。
- 成对运行 `window_replicated` 与 `bag_static`。两者共享逐窗口 modality self-attention、learnable query pooling、adapter 和加权 CE + cumulative ordinal + ranking 目标；前者在所有有效窗口复制 event 标签计算损失，后者先对 23 个窗口 uniform mean 再以单个 EMA 标签计算损失。
- 对 `shared` 和 `per_modality` 两种 normalization 都保留同等地位的 paired 结果。
- 写出：

```text
$OUT_ROOT/phase0/{protocol}/{route_id}/{model_id}/norm_{normalization}__adapter_{adapter_mode}/seed_{seed}/config.json
$OUT_ROOT/phase0/{protocol}/{route_id}/{model_id}/norm_{normalization}__adapter_{adapter_mode}/seed_{seed}/best_checkpoint.pt
$OUT_ROOT/phase0/{protocol}/{route_id}/{model_id}/norm_{normalization}__adapter_{adapter_mode}/seed_{seed}/val_history.csv
$OUT_ROOT/phase0/{protocol}/{route_id}/{model_id}/norm_{normalization}__adapter_{adapter_mode}/seed_{seed}/metrics.json
$OUT_ROOT/phase0/{protocol}/{route_id}/{model_id}/norm_{normalization}__adapter_{adapter_mode}/seed_{seed}/predictions.npz
$OUT_ROOT/phase0/{protocol}/{route_id}/{model_id}/norm_{normalization}__adapter_{adapter_mode}/seed_{seed}/test_predictions.csv
```

`75_run_daily_affect_state_matrix.py`：

- 读取同一份 EMA bag，不重新构建输入。
- 执行 shared/per-modality 双线路、AffectStateFilter、prior-guided modality weighting、cumulative ordinal probe、dynamic EMA kernel、calibration 和缺失模态/扰动 ablation。
- 所有模型变体必须写入 `model_id`，避免把 prior、kernel、probe 或 normalization 的增益混在一起。
- 写出：

```text
$OUT_ROOT/runs/{protocol}/{route_id}/{model_id}/{normalization}/seed_{seed}/config.json
$OUT_ROOT/runs/{protocol}/{route_id}/{model_id}/{normalization}/seed_{seed}/best_checkpoint.pt
$OUT_ROOT/runs/{protocol}/{route_id}/{model_id}/{normalization}/seed_{seed}/val_history.csv
$OUT_ROOT/runs/{protocol}/{route_id}/{model_id}/{normalization}/seed_{seed}/metrics.json
$OUT_ROOT/runs/{protocol}/{route_id}/{model_id}/{normalization}/seed_{seed}/predictions.npz
$OUT_ROOT/runs/{protocol}/{route_id}/{model_id}/{normalization}/seed_{seed}/diagnostics.npz
$OUT_ROOT/runs/{protocol}/{route_id}/{model_id}/{normalization}/seed_{seed}/test_predictions.csv
```

`diagnostics.npz` 至少保存：

```text
modality_weights:     (N_ema, 23, 4)
probe_ordinal_logits: (N_ema, 4, 4)
probe_probs:          (N_ema, 4, 5)
probe_entropy:        (N_ema, 4)
probe_ordinal_var:    (N_ema, 4)
modality_difficulty:  (N_ema, 4)
temporal_weights:     (N_ema, 23)
kernel_mixture:       (N_ema, 3)
states:               (N_ema, 23, 128)
```

`76_summarize_daily_affect_results.py`：

- 读取 `$OUT_ROOT/phase0/**/metrics.json` 和 `$OUT_ROOT/runs/**/metrics.json`。
- 以 protocol、route_id、model_id、normalization、seed 为最小配对键，输出 paired delta 和 gate 结论。
- 写出：

```text
$OUT_ROOT/reports/daily_affect_ordinal_summary.json
$OUT_ROOT/reports/daily_affect_ordinal_summary.md
$OUT_ROOT/reports/protocol_route_summary.csv
$OUT_ROOT/reports/paired_delta_summary.csv
$OUT_ROOT/reports/gate_summary.csv
```

`77_plot_daily_affect_diagnostics.py`：

- 读取 `predictions.npz`、`diagnostics.npz`、`test_predictions.csv`。
- 将 `probe_probs (N_ema,4,5)` 与 `valid_modality_mask (N_ema,4)` 按 `test_index` 对齐，计算每模态 Probe 的 Accuracy、NLL、ECE 和 RPS；先在每个 seed 的 test event 上计算，再对 seed 等权汇总。
- 写出：

```text
$OUT_ROOT/figures/daily_affect_figure_manifest.json
$OUT_ROOT/figures/{protocol}/confusion_{route_id}_{model_id}_{normalization}.png|.svg
$OUT_ROOT/figures/{protocol}/temporal_kernel_{route_id}_{model_id}.png|.svg
$OUT_ROOT/figures/daily_affect_diagnostics_atlas_{protocol}.png
$OUT_ROOT/figures/daily_affect_probe_reliability_atlas_{protocol}.png
$OUT_ROOT/figures/{protocol}/probe_reliability_{route_id}_{model_id}.png|.svg
```

有 `matplotlib` 时输出 PNG；缺少 `matplotlib` 的最小环境中自动输出 SVG fallback，并在 manifest 里记录 `renderer`。paired delta 和 gate 表由 `76_summarize_daily_affect_results.py` 写入 `$OUT_ROOT/reports/paired_delta_summary.csv` 与 `$OUT_ROOT/reports/gate_summary.csv`。

Focused follow-up 入口独立写入新的 run tag，避免覆盖首轮三协议矩阵：

```text
FOCUSED_TAG=daily_affect_dynamic_a1_crossday_20260903
FOCUSED_ROOT=$ALIGN_ROOT/outputs/$FOCUSED_TAG
LOCAL_FOCUSED_SYNC_ROOT=outputs/server_sync/$FOCUSED_TAG
```

`78_report_daily_affect_focused_diagnostics.py`：

- 默认读取首轮 `$ALIGN_ROOT/outputs/daily_affect_ordinal_20260903`；
- 聚焦 `cross_day / A1_Wphysio_full / dynamic_kernel / per_modality` 对 `bag_static / per_modality`；
- 输出 seed-level delta、temporal recent-5 mass、kernel mixture、modality weight、modality difficulty 和相关 CSV：

```text
$FOCUSED_ROOT/focused_reports/cross_day_A1_Wphysio_full_dynamic_kernel_per_modality_focused_diagnostics.json
$FOCUSED_ROOT/focused_reports/cross_day_A1_Wphysio_full_dynamic_kernel_per_modality_focused_diagnostics.md
$FOCUSED_ROOT/focused_reports/*_temporal_weights.csv
$FOCUSED_ROOT/focused_reports/*_kernel_mixture.csv
$FOCUSED_ROOT/focused_reports/*_modality_weights.csv
```

`79_run_daily_affect_focused_ablation.py`：

- 只读已选定 focused bags；
- 在同一 `cross_day / A1_Wphysio_full / per_modality` 上比较 `bag_static`、`state_uniform`、`prior_uniform`、`prior_ordD_uniform`、`dynamic_fixed_short`、`dynamic_fixed_medium`、`dynamic_fixed_long`、`dynamic_kernel_no_prior`、`dynamic_kernel_prior_uniform` 与 `dynamic_kernel`；
- `bag_static` 写入 `$FOCUSED_ROOT/phase0/...`，其他候选写入 `$FOCUSED_ROOT/runs/...`，可继续由 `76_summarize_daily_affect_results.py` 汇总。

`81_report_daily_affect_prior_guidance.py`：

- 读取同一 focused run 的 metrics；
- 分别汇总 `state_uniform -> prior_uniform`、`prior_uniform -> prior_ordD_uniform`、`dynamic_kernel_no_prior -> dynamic_kernel_prior_uniform`、`dynamic_kernel_prior_uniform -> dynamic_kernel` 和完整路径对比；
- 写出 paired delta CSV、summary CSV 与 Markdown/JSON 报告，防止将 state prior 和 ordinal difficulty 当作同一个机制。

`82_run_daily_affect_bottleneck_audit.py`：

- 在同 seed、同 bag、同训练设置下，以静态 `bag_static` 比较 full 2-minute mean、last 10/30/60 s、first 30 s 与 short/medium/long fixed kernel；该比较只改变 EMA-level temporal aggregation，不引入 state/prior。
- 用 train event token 统计作为无标签 reference，按 test day 输出四模态 standardized centroid shift、geometric variance ratio 和 event availability。
- 读取 A1 video 的 `quality_flags`，按 split/day 输出 video window/event coverage、complete-event rate、usable-frame fraction；同时训练 video-only 和 non-video 对照。drift/quality 属于 input-only 诊断，不作为 causal claim。

`80_run_daily_affect_focused_robustness.py`：

- 读取 `$FOCUSED_ROOT/phase0` 与 `$FOCUSED_ROOT/runs` 的 frozen checkpoint；
- 对 `clean`、`missing`、`noise`、`shuffle` 场景分别扰动 EEG/Wear/Video/Audio；
- 输出：

```text
$FOCUSED_ROOT/robustness/focused_robustness_report.json
$FOCUSED_ROOT/robustness/focused_robustness_report.md
$FOCUSED_ROOT/robustness/focused_robustness_details.csv
$FOCUSED_ROOT/robustness/focused_robustness_summary.csv
```

## 12.4 推荐命令骨架

```bash
RUN_TAG=daily_affect_ordinal_20260903
ALIGN_ROOT=/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned
EMB_ROOT=/vePFS-0x0d/DailyEEG_multimodal/embeddings
INDEX_PATH=$ALIGN_ROOT/index/eeg_aligned_window_index.jsonl
SPLITS_ROOT=/vePFS-0x0d/DailyEEG/splits_new
OUT_ROOT=$ALIGN_ROOT/outputs/$RUN_TAG

PYTHONPATH=src runtime/envs/eegpt-gpu-min/bin/python scripts/73_build_daily_affect_bags.py \
  --root "$ALIGN_ROOT" \
  --index-path "$INDEX_PATH" \
  --embeddings-root "$EMB_ROOT" \
  --splits-root "$SPLITS_ROOT" \
  --protocols cross_subject,cross_day,within_subject_day \
  --route-ids B0_Wphysio_full,A1_Wphysio_full,A2_Wdeep_full \
  --seeds 240729,240730,240731 \
  --out-root "$OUT_ROOT/bags"

PYTHONPATH=src runtime/envs/eegpt-gpu-min/bin/python scripts/74_run_daily_affect_phase0_baselines.py \
  --bags-root "$OUT_ROOT/bags" \
  --out-root "$OUT_ROOT/phase0" \
  --model-ids window_replicated,bag_static \
  --normalizations shared,per_modality \
  --protocols cross_subject,cross_day,within_subject_day \
  --route-ids B0_Wphysio_full,A1_Wphysio_full,A2_Wdeep_full \
  --seeds 240729,240730,240731 \
  --device cuda \
  --skip-existing

PYTHONPATH=src runtime/envs/eegpt-gpu-min/bin/python scripts/75_run_daily_affect_state_matrix.py \
  --bags-root "$OUT_ROOT/bags" \
  --out-root "$OUT_ROOT/runs" \
  --normalizations shared,per_modality \
  --model-ids state_uniform,prior_uniform,prior_ordD_uniform,dynamic_kernel \
  --protocols cross_subject,cross_day,within_subject_day \
  --route-ids B0_Wphysio_full,A1_Wphysio_full,A2_Wdeep_full \
  --seeds 240729,240730,240731 \
  --device cuda \
  --skip-existing

PYTHONPATH=src runtime/envs/eegpt-gpu-min/bin/python scripts/76_summarize_daily_affect_results.py \
  --run-root "$OUT_ROOT" \
  --out-root "$OUT_ROOT/reports"

PYTHONPATH=src runtime/envs/eegpt-gpu-min/bin/python scripts/77_plot_daily_affect_diagnostics.py \
  --run-root "$OUT_ROOT" \
  --out-root "$OUT_ROOT/figures"
```

Focused follow-up 推荐命令：

```bash
FOCUSED_TAG=daily_affect_dynamic_a1_crossday_20260903
FOCUSED_ROOT=$ALIGN_ROOT/outputs/$FOCUSED_TAG

PYTHONPATH=src runtime/envs/eegpt-gpu-min/bin/python scripts/78_report_daily_affect_focused_diagnostics.py \
  --run-root "$OUT_ROOT" \
  --out-root "$FOCUSED_ROOT/focused_reports" \
  --protocol cross_day \
  --route-id A1_Wphysio_full \
  --normalization per_modality \
  --candidate-model dynamic_kernel

PYTHONPATH=src runtime/envs/eegpt-gpu-min/bin/python scripts/73_build_daily_affect_bags.py \
  --root "$ALIGN_ROOT" \
  --index-path "$INDEX_PATH" \
  --embeddings-root "$EMB_ROOT" \
  --splits-root "$SPLITS_ROOT" \
  --protocols cross_day \
  --route-ids A1_Wphysio_full \
  --seeds 240732,240733,240734,240735 \
  --out-root "$FOCUSED_ROOT/bags"

PYTHONPATH=src runtime/envs/eegpt-gpu-min/bin/python scripts/79_run_daily_affect_focused_ablation.py \
  --bags-root "$FOCUSED_ROOT/bags" \
  --out-root "$FOCUSED_ROOT" \
  --protocol cross_day \
  --route-id A1_Wphysio_full \
  --normalization per_modality \
  --seeds 240729,240730,240731 \
  --model-ids bag_static,state_uniform,prior_uniform,prior_ordD_uniform,dynamic_fixed_short,dynamic_fixed_medium,dynamic_fixed_long,dynamic_kernel_no_prior,dynamic_kernel_prior_uniform,dynamic_kernel \
  --device cuda \
  --skip-existing

PYTHONPATH=src runtime/envs/eegpt-gpu-min/bin/python scripts/79_run_daily_affect_focused_ablation.py \
  --bags-root "$FOCUSED_ROOT/bags" \
  --out-root "$FOCUSED_ROOT" \
  --protocol cross_day \
  --route-id A1_Wphysio_full \
  --normalization per_modality \
  --seeds 240732,240733,240734,240735 \
  --model-ids bag_static,dynamic_kernel \
  --device cuda \
  --skip-existing

PYTHONPATH=src runtime/envs/eegpt-gpu-min/bin/python scripts/76_summarize_daily_affect_results.py \
  --run-root "$FOCUSED_ROOT" \
  --out-root "$FOCUSED_ROOT/reports"

PYTHONPATH=src runtime/envs/eegpt-gpu-min/bin/python scripts/80_run_daily_affect_focused_robustness.py \
  --run-root "$FOCUSED_ROOT" \
  --out-root "$FOCUSED_ROOT/robustness" \
  --protocol cross_day \
  --route-id A1_Wphysio_full \
  --normalization per_modality \
  --models bag_static,dynamic_kernel \
  --seeds 240729,240730,240731,240732,240733,240734,240735 \
  --device cuda
```

真实 state-prior 修复后的独立复跑使用新的 root，不能与 2026-09-03 的 focused 输出混读：

```bash
PRIOR_V2_ROOT=$ALIGN_ROOT/outputs/daily_affect_prior_guidance_v2_20260905

PYTHONPATH=src runtime/envs/eegpt-gpu-min/bin/python scripts/79_run_daily_affect_focused_ablation.py \
  --bags-root "$ALIGN_ROOT/outputs/daily_affect_dynamic_a1_crossday_20260903/bags" \
  --out-root "$PRIOR_V2_ROOT" \
  --protocol cross_day \
  --route-id A1_Wphysio_full \
  --normalization per_modality \
  --model-ids bag_static,state_uniform,prior_uniform,prior_ordD_uniform,dynamic_kernel_no_prior,dynamic_kernel_prior_uniform,dynamic_kernel \
  --seeds 240729,240730,240731 \
  --device cuda

PYTHONPATH=src runtime/envs/eegpt-gpu-min/bin/python scripts/81_report_daily_affect_prior_guidance.py \
  --run-root "$PRIOR_V2_ROOT" \
  --out-root "$PRIOR_V2_ROOT/reports" \
  --protocol cross_day \
  --route-id A1_Wphysio_full \
  --normalization per_modality
```

## 12.5 2026-09-03 正式执行状态

已在远端 `/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned` 按上述路径执行 `RUN_TAG=daily_affect_ordinal_20260903`：

- `73_build_daily_affect_bags.py`：完成 `3 protocols x 3 routes x 3 seeds = 27` 个 bag；每个 bag 均为 `tokens (1253,23,4,256)`、`modality_mask (1253,23,4)`；报告在 `$OUT_ROOT/bags/daily_affect_bag_build_report.{json,md}`。
- split 投影记录：`cross_subject` 和 `cross_day` 各有 `42` 个 event 只在 `pretrain/finetune` 训练叶子内混合，归入 train；`within_subject_day` 有 `214` 个 event 跨 train/val/test 边界，按 23 窗口多数票投影为单一 bag-level split。
- `74_run_daily_affect_phase0_baselines.py`：完成 `54/54` 个旧版 `bag_static` baseline metrics，输出在 `$OUT_ROOT/phase0`。这些历史 run 使用旧的轻量窗口融合，不能与 2026-09-06 起共享 attention/query-pooling 的 `window_replicated` vs `bag_static` supervision 对照混用。
- `75_run_daily_affect_state_matrix.py`：完成 `216/216` 个 state/prior/kernel metrics，输出在 `$OUT_ROOT/runs`。
- `76_summarize_daily_affect_results.py`：汇总 `270` 个 run，输出 `$OUT_ROOT/reports/daily_affect_ordinal_summary.{json,md}`、`protocol_route_summary.csv`、`paired_delta_summary.csv` 和 `gate_summary.csv`。
- `77_plot_daily_affect_diagnostics.py`：manifest 记录 `918` 个 figure 条目，实际图像文件 `199` 个，renderer 为 `matplotlib`；manifest 在 `$OUT_ROOT/figures/daily_affect_figure_manifest.json`。
- 轻量本地同步副本：`G:\Daily Multimodal\outputs\server_sync\daily_affect_ordinal_20260903`，包含 `reports/`、`logs/` 和 `figures/`；大体积 `bags/phase0/runs` 保留在远端。

## 12.6 2026-09-03 focused follow-up 状态

已在同一远端工作目录执行 `FOCUSED_TAG=daily_affect_dynamic_a1_crossday_20260903`，目标是检查首轮最有发展空间的 `cross_day / A1_Wphysio_full / dynamic_kernel / per_modality`：

- `78_report_daily_affect_focused_diagnostics.py`：完成 kernel 和 modality 小报告。首轮 3-seed 配对中，`dynamic_kernel` 相对 `bag_static` 的 mean delta 为 QWK `+0.0387`、Macro-F1 `+0.0162`、ordinal MAE `-0.0501`；QWK wins `2/3`。
- kernel/modality 诊断显示候选主要使用近期窗口：三个 seed 的 recent-5 temporal mass 为 `0.6566`、`0.5519`、`0.5171`；kernel mixture 并非固定单一形状，短核与长核在不同 seed 中交替占优。
- `79_run_daily_affect_focused_ablation.py`：完成 `24` 个 3-seed focused ablation run。3-seed 均值中 `dynamic_kernel` 为 QWK `0.1921`、Macro-F1 `0.2453`、MAE `0.8208`；`dynamic_kernel_no_prior` 为 QWK `0.1902`、Macro-F1 `0.2485`、MAE `0.8195`；`dynamic_fixed_short` 为 QWK `0.1848`、Macro-F1 `0.2540`、MAE `0.8274`。这说明增益主要来自动态状态/时间核路径，ordinal difficulty prior 是可继续调的辅助因素。
- 候选与 baseline 已补齐到 `7` seeds。`dynamic_kernel` 对 `bag_static` 的 7-seed clean mean delta 为 QWK `+0.0133`、Macro-F1 `+0.0124`、ordinal MAE `-0.0378`，QWK wins `4/7`。该结果通过当前 preliminary gate，但优势幅度偏小，应标为 promising candidate，而不是替换主线。
- 同一组 event-level test prediction 的 regression bridge 为：`bag_static` 的 Expected RMSE/raw r/within-subject centered r 为 `1.0458 / 0.2119 / 0.0093`，`dynamic_kernel` 为 `1.0299 / 0.2318 / 0.0318`，配对 delta 为 `-0.0159 / +0.0199 / +0.0226`。这三列与窗口级主线使用相同定义，适合在同一 protocol 下共同阅读；EMA bag 与窗口级结果仍分别保留自己的样本单位和主 gate。
- `80_run_daily_affect_focused_robustness.py`：完成 `182` 条明细。clean 下 `dynamic_kernel` 优于 baseline；missing/noise/shuffle 中 EEG 和 Wear 扰动下多保持非负或接近持平，Video 是主要脆弱点：`missing video` 相对 baseline QWK `-0.0210`，`shuffle video` 相对 baseline QWK `-0.0306`。
- 轻量本地同步副本：`G:\Daily Multimodal\outputs\server_sync\daily_affect_dynamic_a1_crossday_20260903`，包含 `focused_reports/`、`reports/`、`robustness/` 和 focused ablation 顶层报告；大体积 bags、checkpoints 和 runs 保留在远端。

## 12.7 2026-09-05 state-prior 实现修复与复判

此前 focused 结果中的 `dynamic_kernel` 名称不足以证明使用了“上一时刻状态主动查询当前模态”的机制。现有实现已改为：对每个时间步先由上一 GRU state 构造 `p_t = LN(z_{t-1} + Transition(z_{t-1}))`，再用 `p_t` 与当前各模态 adapter token 的 additive compatibility 进入 routing score。由此拆出：`prior_uniform`（state compatibility）、`prior_ordD_uniform`（compatibility + ordinal difficulty）、`dynamic_kernel_prior_uniform`（dynamic kernel + compatibility）和 `dynamic_kernel`（dynamic kernel + compatibility + ordinal difficulty）。

- 修复后的 3-seed 复跑写入独立目录 `daily_affect_prior_guidance_v2_20260905`，只复用既有 bags，不覆盖 2026-09-03 的模型、checkpoint 或 robustness 结果。
- `dynamic_kernel_no_prior -> dynamic_kernel` 的 mean delta 为 QWK `-0.0461`、ordinal MAE `+0.0316`、Expected RMSE `+0.0445`、raw r `-0.0470`、centered r `-0.0634`，QWK wins `1/3`。因此真实完整 prior 路径在当前动态核候选上不进入 5-7 seed 扩展。
- 单独加入 state compatibility 的 `dynamic_kernel_no_prior -> dynamic_kernel_prior_uniform` 为 QWK `-0.0134`、raw r `-0.0215`，同样没有支持信号。`state_uniform -> prior_uniform` 的 QWK `+0.0007` 属于极弱诊断信号，不能外推为动态核收益。
- `prior_uniform -> prior_ordD_uniform` 的 QWK `+0.0168`、3/3 wins、MAE `-0.0158` 表明 ordinal difficulty 在没有动态核的 state route 上值得保留为诊断分支；它没有改变动态核当前以 `dynamic_kernel_no_prior` 继续作为候选的判断。
- 旧 root 的 7-seed `dynamic_kernel`/robustness 记录保留为历史结果；其 prior 语义与本节修复后的版本不同，不能作为真实 state-prior 的效果证据。

## 12.8 2026-09-05 bottleneck audit 状态

在独立 `daily_affect_bottleneck_audit_20260905` root 上，针对同一 `cross_day / A1_Wphysio_full / per_modality` bags 运行 `30` 个严格配对 static-screen（8 个 temporal policy 加 video-only/non-video，3 seeds），并输出 `360` 条 per-day modality drift 与 `882` 条 video quality rows：

- **事件级标签可辨识性**：完整两分钟 uniform baseline 的 QWK 为 `0.1534`。`kernel_short` 为 `0.2211`，相对 baseline mean delta QWK `+0.0677`、MAE `-0.0619`、raw r `+0.0622`，QWK wins `2/3`；`kernel_medium` 为 `0.2119`，delta QWK `+0.0585`、MAE `-0.0619`、raw r `+0.0488`，QWK wins `3/3`。last 10 s 的 QWK delta 仅 `+0.0010` 且 MAE `+0.0079`，first 30 s 的 QWK delta `-0.0392`。当前证据支持 EMA 标签更接近近期 30--60 s 的加权状态，而不是末 10 s 或完整两分钟的均值；下一步应给 `kernel_short`、`kernel_medium` 与 full-2min baseline 补 5--7 seeds，再与 `dynamic_kernel_no_prior` 配对。
- **跨日表征**：以 train token 为 reference 的 test-day centroid shift RMS 为 EEG `0.1738`、Wear `0.5954`、Video `0.8074`、Audio `0.5896`。Video 的 test event availability 为 `0.6319`，低于 EEG `1.0000` 和 Wear `0.9333`；跨日 video 不稳包含分布变化与覆盖变化两个来源。
- **Video 稳定性**：video-only full-2min 对照的 QWK `0.1845`，相对 all-modal full-2min 的 mean delta `+0.0310`、3/3 QWK wins，但 centered r `-0.0483`；non-video 对照 QWK delta `-0.0653`。这表明 video 含有 event-level 信号，却不能稳定承担跨日连续关系。test day 的 video window coverage 平均 `0.6111`、complete-video event rate `0.5739`，且部分日期为零覆盖。有效 video 的 `quality_flags` 全部为 4/4 usable frames，partial-frame fraction `0`，当前质量字段没有连续低质量样本可供 ROI-quality routing 学习；下一阶段应补面部检测/跟踪置信度、姿态、模糊和遮挡特征，再测试 quality-aware downweight 与 train-time video dropout。
- 远端完整产物在 `$ALIGN_ROOT/outputs/daily_affect_bottleneck_audit_20260905`；本地同步副本为 `G:\Daily Multimodal\outputs\server_sync\daily_affect_bottleneck_audit_20260905`。

---

# 13. 与 EQL-CAF 的边界

EQL-CAF 是当前正在尝试的另一种 temporal multimodal 方法，代码位于：

```text
src/daily_multimodal/temporal/
src/daily_multimodal/training/eql_caf.py
scripts/57-64
```

路径边界：

```text
daily-affect:
  $ALIGN_ROOT/outputs/$RUN_TAG/
  outputs/server_sync/$RUN_TAG/

EQL-CAF:
  $ALIGN_ROOT/outputs/eql_caf_20260823/
  outputs/server_sync/eql_caf_20260823/
```

本路线与 EQL-CAF 必须保持独立：

- **数据输入不互相冒用**：本路线以 EMA event 为监督单元，输入为当前 EEG-aligned 主线的 23 个 10 秒重叠窗口；EQL-CAF 当前围绕每个 10 秒窗口内部的 5 个 2 秒 temporal token 契约展开。
- **mask 命名不互相复用**：本路线只使用窗口级/EMA-bag 级 `modality_mask`，shape 为 `(N_ema, 23, 4)`；`token_mask` 只保留给 EQL-CAF 的窗口内 2 秒 token 契约，shape 为 `(N_window, 4, 5)`。
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
            "label": ...,         # scalar int64，原始语义1...5
            "subject_id": ...,
            "ema_id": ...,
            "ema_time": ...,
        }
```

---

## 14.2 AffectStateFilter

```python
class AffectStateFilter(nn.Module):
    def forward(self, adapted_tokens, modality_mask, modality_difficulty):
        """
        adapted_tokens:       (B, L, M, 128)
        modality_mask:        (B, L, M)
        modality_difficulty:  (B, M), detached normalized ordinal difficulty

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
        adapted_tokens = self.adapters(tokens)
        (
            probe_ordinal_logits,
            probe_probs,
            probe_entropy,
            probe_ordinal_var,
            modality_difficulty,
        ) = self.modality_probes(adapted_tokens, modality_mask)
        states, modality_weights = self.state_filter(
            adapted_tokens,
            modality_mask,
            modality_difficulty.detach(),
        )
        pooled, temporal_weights, mixture_weights = self.ema_kernel(states)
        logits = self.classification_head(pooled)

        return {
            "logits": logits,
            "probs": softmax(logits, dim=-1),
            "probe_ordinal_logits": probe_ordinal_logits,  # (B, 4, 4)
            "probe_probs": probe_probs,                    # (B, 4, 5)
            "probe_entropy": probe_entropy,                # (B, 4), D_H
            "probe_ordinal_var": probe_ordinal_var,        # (B, 4), D_V
            "modality_difficulty": modality_difficulty,
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
-> 5-class classification
```

EMA 前 2 min 内窗口复制同一标签。

该结果作为历史 baseline，不作为最终公平比较的唯一基线。

---

## 15.2 Baseline 1：Bag-level static fusion

保持当前窗口级 attention fusion，但改为：

```text
23 个窗口分别得到 fused representation
-> uniform mean pooling over time
-> EMA-level ordinal classification
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

## 15.4 Ablation B：Latent fatigue state

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

验证“previous fatigue-state prior 主动查询当前 multimodal evidence”是否优于直接把融合特征输入 GRU。

---

## 15.6 Ablation D：序数困难度修正动态权重

在相同 Prior-guided update 上比较：

| 配置 | Routing |
|---|---|
| P0 | `softmax(s)`，原始动态权重 |
| P1 | 普通五分类 Probe + softmax entropy，作为原论文式基线 |
| P2 | Cumulative ordinal Probe + entropy，`beta_ord=0` |
| P3 | Cumulative ordinal Probe + `D_ord`，`beta_ord>0` |
| P4 | P3 + temperature calibration |
| P5 | P3 去掉 `stop-gradient`，仅作为机制验证；P4 的 Probe 已冻结，不能用于检验 end-to-end difficulty gradient |

主实验扫描：

```text
lambda_D ∈ {0, 0.1, 0.25, 0.5, 1.0}
```

同时消融：

```text
beta_ord ∈ {0, 0.25, 0.5}
```

这一组实验分别回答：普通预测熵是否有效、序数 Probe 是否优于无序 Probe，以及显式等级跨度能否在不压制模态互补信息的前提下改善动态路由和缺失/扰动鲁棒性。

后续扩展可增加 time-varying MIL Probe：先对每个窗口输出 Probe logits，再通过受约束时间核聚合，只在 EMA 层面计算一次 Probe loss。该扩展不能在 23 个窗口上复制标签损失。

---

## 15.7 Full model：Dynamic EMA kernel

```text
Modality-specific adapter
+ latent fatigue state
+ ordinal-difficulty-regularized prior-guided update
+ dynamic EMA kernel
+ ordinal classification head
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

同时增加 controlled modality corruption：分别对 Video、Audio、Wear、EEG embedding 注入可控扰动或随机置换，检查对应 `D_ord,i^m` 是否上升、`alpha_i,t^m` 是否下降。这里只检验模型置信度路由的行为，不声称模型显式估计物理噪声或 SQI。

---

# 19. 评价指标

对于 1–5 级有序分类，建议主指标：

```text
Macro-F1
Balanced Accuracy
Quadratic Weighted Kappa (QWK)
Ordinal MAE = mean(|predicted_level - true_level|)
```

同时报告：

```text
Accuracy
每类 Precision / Recall / F1
5×5 confusion matrix
within-subject Macro-F1 / QWK 的均值与标准差
```

重点检查错误是否集中在相邻等级，并报告跨 2 级及以上的严重错误比例。

### 19.1 Regression bridge for cross-route reading

Daily-affect 的训练、early stopping 与 promotion gate 继续以 EMA-level ordinal 指标为准。为与窗口级 fatigue 主线共同阅读，额外把五分类概率的期望等级 `expected_score = sum_k p(k) * k` 投影到同一 `1..5` 疲劳分数轴，并在每个 protocol 的 event-level test split 上报告：

```text
Expected RMSE
Expected raw r
Expected within-subject centered r
```

`raw r` 表示 event-level 总体相关；`centered r` 先在当前 test split 内按 subject 对真实值和期望分数各自去均值，再计算 Pearson r。它们用于描述性横向比较，不改变 EMA bag 与窗口级模型的样本单位，也不替代 QWK / Macro-F1 / ordinal MAE 的 daily-affect gate。

### 19.2 Probe 校准与可靠性诊断

每个模态分别报告：

```text
Probe Accuracy / Macro-F1
四个累计阈值的 Accuracy / AUROC
NLL
Brier score
Ranked Probability Score (RPS)
ECE
预测熵四分位 Q1...Q4 的 Probe Accuracy
D_ord 四分位 Q1...Q4 的 Probe Accuracy / Ordinal MAE
```

额外将相同或相近熵值的预测按等级方差分组，检查“1/5 级分散”是否比“1/2 级分散”得到更高 `D_ord` 和更低路由权重。

若 `D_ord` 可以作为有效路由偏置，应至少观察到：低困难度组准确率更高、Ordinal MAE 更低；模态受扰动后 `D_ord` 上升；`D_ord` 上升时最终模态权重下降。另行统计“低 `D_ord` 但跨两级及以上预测错误”的比例，用于量化测试时自信远距离错误这一剩余局限。该诊断只建立统计关系，不声称每个低困难度预测都正确。

统计上：

- 在每个 seed 内以 subject-day 为 bootstrap 单位计算 paired delta 的 95% CI。
- 模型比较优先在 subject-level metric 上做 paired test。
- 不要把 23 个 window 当成 23 个独立统计样本。

---

# 20. 可解释性输出

模型训练时保存：

```text
modality_weights: (N_ema, 23, 4)
probe_ordinal_logits:(N_ema, 4, 4)
probe_probs:       (N_ema, 4, 5)
probe_entropy:     (N_ema, 4)
probe_ordinal_var: (N_ema, 4)
modality_difficulty:(N_ema, 4)  # D_ord
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

同时绘制每个模态的 bag-level normalized entropy `D_H,i^m`、等级方差 `D_V,i^m` 和综合困难度 `D_ord,i^m`。模态权重只解释为综合状态匹配和模型序数不确定性后的相对 evidence 使用程度，不解释为生理“重要性”或“噪声大小”。

---

## 20.3 Latent fatigue trajectory

对 `z_1...z_23`：

- PCA / UMAP 仅用于展示；
- 比较高分与低分 EMA 前的 trajectory；
- 比较状态变化速度与最终 EMA 的关系。

该部分作为探索性结果，不作为第一版训练目标。

---

# 21. 推荐开发顺序

严格按以下顺序实现，避免一次性改动过多导致无法定位增益来源。

脚本对应关系：

```text
Phase 0: scripts/74_run_daily_affect_phase0_baselines.py
Phase 1: scripts/73_build_daily_affect_bags.py + scripts/74_run_daily_affect_phase0_baselines.py
Phase 2: scripts/75_run_daily_affect_state_matrix.py --model-ids bag_static
Phase 3: scripts/75_run_daily_affect_state_matrix.py --model-ids state_uniform
Phase 4: scripts/75_run_daily_affect_state_matrix.py --model-ids prior_uniform,prior_ordD_uniform
Phase 5: scripts/75_run_daily_affect_state_matrix.py --model-ids dynamic_kernel
Phase 6: scripts/76_summarize_daily_affect_results.py + scripts/77_plot_daily_affect_diagnostics.py
Phase 7: scripts/78_report_daily_affect_focused_diagnostics.py + scripts/79_run_daily_affect_focused_ablation.py + scripts/80_run_daily_affect_focused_robustness.py
```

## Phase 0：复现当前 baseline

对应脚本为 `74_run_daily_affect_phase0_baselines.py`。该阶段从同一 EMA bag 成对训练 `window_replicated` 和 `bag_static`：两条线路使用相同窗口 attention/query-pooling、adapter、normalization 和序数目标，差别仅在窗口复制监督或 EMA bag 监督。输出固定在：

```text
$OUT_ROOT/phase0/{protocol}/{route_id}/{model_id}/norm_{normalization}__adapter_{adapter_mode}/seed_{seed}/
```

确认现有：

```text
attention variant
5-class CE / ordinal loss
4-modal / 3-modal / 2-modal
```

结果完全可复现。

---

## Phase 1：重构为 EMA bag

对应脚本为 `73_build_daily_affect_bags.py` 和 `74_run_daily_affect_phase0_baselines.py`。`73` 负责生成：

```text
$OUT_ROOT/bags/{protocol}/{route_id}/seed_{seed}/ema_bags.npz
```

`74` 读取该 bag，训练 `bag_uniform_static` baseline。

只修改 Dataset / DataLoader：

```text
(B, 4, 256)
```

改成：

```text
(B, 23, 4, 256)
```

`bag_static` 使用现有 attention 在每个时间点独立融合，最后对 23 个时间点 uniform mean；配对的 `window_replicated` 使用同一融合器，但在每个有效窗口计算复制的 event 标签损失，并在测试时做 event-level概率均值。

首先验证 pipeline 正确。

**验收条件：**

- sample_id / EMA id 对齐无误；
- 同一 EMA 的 23 个窗口映射为一个 bag-level split，并记录 leaf split 组成、训练叶子合并和边界多数票投影；
- train-only normalization 无泄漏；
- loss 数量等于 EMA 数，而不是窗口数。

---

## Phase 2：Shared / Per-modality 双线路筛选

对应脚本为 `75_run_daily_affect_state_matrix.py --model-ids bag_static --normalizations shared,per_modality`。两条线路均写入：

```text
$OUT_ROOT/runs/{protocol}/{route_id}/bag_static/{normalization}/seed_{seed}/
```

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

对应脚本为 `75_run_daily_affect_state_matrix.py --model-ids state_uniform`。该阶段继续读同一份 `ema_bags.npz`，只改变 state fusion 结构。

加入：

```text
state transition
prior-guided modality weighting
evidence aggregation
state update gate
```

EMA 端先继续使用 uniform temporal mean，并固定 `lambda_D=0`，先建立原始动态权重基线。

**验收条件：**

- 输出 `states.shape == (B, 23, 128)`；
- 每个时间点有效 modality weights 之和为 1；
- 缺失模态权重为 0；
- 能输出和可视化连续 state trajectory。

---

## Phase 4：实现 Cumulative Ordinal Probe 与序数困难度路由

对应脚本为 `75_run_daily_affect_state_matrix.py --model-ids prior_uniform,prior_ordD_uniform`。`prior_uniform` 只启用 Probe 恢复概率和困难度诊断；`prior_ordD_uniform` 加入 stop-gradient ordinal-difficulty bias。

加入：

```text
EMA-level masked-mean modality Probe
4-threshold cumulative ordinal logits
monotonic thresholds and 5-class probability recovery
normalized entropy + ordinal variance
stop-gradient ordinal-difficulty bias
Probe warm-up
```

先保持 EMA 端 uniform temporal mean，单独比较 P0–P5。

**验收条件：**

- `probe_ordinal_logits.shape == (B, 4, 4)`；
- `probe_probs.shape == (B, 4, 5)`；
- 四个累计概率单调不增，恢复后的五类概率非负且和为 1；
- `modality_difficulty.shape == (B, 4)` 且有效值位于 `[0,1]`；
- 每个 EMA、每个有效模态只计算一次 Probe loss；
- 相同熵下，远距离等级分散应比相邻等级分散产生更高 `D_ord`；
- 主损失沿 ordinal-difficulty-routing 路径不能更新 Probe，`L_probe` 可以更新 Probe；
- 人为扰动模态时记录 `D_H`、`D_V`、`D_ord` 和 modality weight 的变化。

---

## Phase 5：实现 Dynamic EMA Kernel

对应脚本为 `75_run_daily_affect_state_matrix.py --model-ids dynamic_kernel`。该阶段在 Phase 4 通过的 shared/per-modality 线路上加入可变时间核，仍写入 `$OUT_ROOT/runs/...`，并额外要求 `diagnostics.npz` 保存 `temporal_weights` 与 `kernel_mixture`。

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

## Phase 6：完整实验

对应脚本为 `75_run_daily_affect_state_matrix.py`、`76_summarize_daily_affect_results.py` 和 `77_plot_daily_affect_diagnostics.py`。完整矩阵的读取根目录固定为 `$OUT_ROOT/bags`，报告和图分别写到 `$OUT_ROOT/reports` 与 `$OUT_ROOT/figures`。

按以下顺序运行：

```text
1. baseline / ablation
2. kernel ablation
3. look-back ablation
4. modality ablation
5. ordinal-difficulty-routing / calibration ablation
6. test-time missing modality 与 controlled corruption
7. Probe reliability diagnostics
8. interpretability
```

第一版不要并行加入新的自监督或个体化模块。

---

## Phase 7：Focused follow-up

对应脚本为 `78_report_daily_affect_focused_diagnostics.py`、`79_run_daily_affect_focused_ablation.py` 和 `80_run_daily_affect_focused_robustness.py`。只在首轮矩阵已经筛出的单一路线、单一协议和单一 normalization 上执行，当前为：

```text
cross_day / A1_Wphysio_full / per_modality
baseline: bag_static
candidate: dynamic_kernel
```

执行顺序：

```text
1. 先用 78 出 kernel 和 modality 诊断小报告；
2. 再用 79 跑 3-seed focused ablation，拆开 state、prior、fixed kernel、no-prior dynamic kernel 和完整 dynamic kernel；
3. 只给候选 dynamic_kernel 与 baseline bag_static 补到 5-7 seeds；
4. 用 80 做 missing/corruption robustness。
```

继续发展门槛：

- 候选必须与同 seed、同 bag、同 normalization 的 baseline 配对；
- 7-seed mean delta QWK 和 Macro-F1 至少非负，MAE 不变差；
- seed-level 方向一致性至少达到 `4/7`；
- robustness 里不能只在 clean 场景成立；若某个核心模态扰动下显著劣于 baseline，应先做该脆弱点诊断，不扩展到更多协议。

---

## Routefix 执行附录（2026-09-05）

历史 `20260903` 的 EMA bags 固定为只读输入：

```text
outputs/daily_affect_ordinal_20260903/bags/{protocol}/{route_id}/seed_{seed}/ema_bags.npz
```

当前语义的训练与报告统一写入新的 `*_routefix_20260905` root，历史目标函数、旧 adapter 耦合和旧 routing 结果保持独立，不参与新汇总。

每个正式 condition 显式记录五个因子：

```text
normalization ∈ {shared, per_modality}
adapter_mode ∈ {shared, per_modality}
objective_id = class-weighted CE + cumulative ordinal loss + within-subject ranking
routing_id = Probe kind + difficulty source + beta_ord + detach flag
experiment_id = phase / routing profile / focused diagnostic
```

因此 `shared` 与 `per_modality` 是同等地位、同 seed 的待验证路线；routing-factorial 额外以 2x2 设计区分 token normalization 与 adapter sharing，不能把两者混称为同一个 `shared` 条件。

`75_run_daily_affect_state_matrix.py --routing-profiles` 将 P0--P5 显式映射为：

```text
P0: no-prior state/kernel baseline
P1: categorical Probe + entropy difficulty
P2: cumulative Probe + entropy difficulty
P3: cumulative Probe + entropy/ordinal-variance mix
P4: P3 + per-modality Probe temperature calibration, then freeze adapters/Probe and tune state/kernel/head
P5: P3 with difficulty gradient enabled (detach=false), mechanism-only
```

训练期 modality dropout 每次最多移除一个 bag-wide 有效模态，并保留至少一个有效模态。`80` 的 test-time robustness 覆盖 clean、单模态 missing、两模态成对 missing、noise 与 shuffle；结果同时报告预测指标、受保护的无剩余模态 event 数，以及四模态 routing/difficulty 相对 clean 的变化。

`76` 的 paired comparison 只匹配 `protocol / route / normalization / adapter_mode / objective_id / seed`。它对每 seed 的 static baseline 对候选 delta 输出 QWK、Macro-F1、ordinal MAE、Expected RMSE/raw r/centered r，随后报告 mean +/- std、方向一致性，并在 event-level prediction 可用时给 subject-day block bootstrap CI。

---

# 22. 第一篇建议明确不做的内容

为控制复杂度，本版本明确不加入：

```text
显式 noise / SQI estimator
外部 signal-quality / reliability 标签监督
window-level EMA 标签复制 Probe
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

> 日常疲劳等级不是每个 10 秒窗口独立产生的静态标签，而是对一段连续演化的潜在疲劳状态轨迹的有序观测；EEG、Wear、Video 和 Audio 是对该状态的异构证据，模态贡献由历史状态匹配程度、预测熵和不确定类别的等级跨度共同调节。

对应模型：

```text
heterogeneous multimodal observations
        ↓
modality-specific fatigue/affect space
        ↓
continuous latent fatigue filtering
        ↓
ordinal-difficulty-regularized prior-guided multimodal update
        ↓
dynamic EMA temporal integration
        ↓
1–5 level ordinal classification
```

最核心的三个方法贡献：

1. **Prior-guided multimodal fatigue-state filtering**：利用上一时刻的潜在疲劳状态作为 prior，动态整合当前四模态 evidence，而不是对每个窗口做独立静态融合。
2. **Ordinal-difficulty-regularized modality routing**：使用 EMA-level cumulative ordinal Probe，将归一化预测熵和等级方差组合为 `D_ord` 修正模态权重；远距离等级错误在序数 Probe 训练中跨越更多阈值，远距离分散也会在路由中受到比相邻等级分散更强的惩罚。stop-gradient 将路由信号与主任务操纵路径解耦，temperature calibration 作为独立增强消融。
3. **Dynamic EMA observation kernel**：将一次稀疏 EMA 视为对近期潜在疲劳轨迹的动态时间积分，避免将同一 EMA 标签机械复制给多个窗口。

该路线在保留当前预训练单模态 embedding 和整体训练框架的基础上，改动集中、可逐步消融、适合样本量有限的日常 EMA 数据，也为后续加入个体化、长程动力学和 world-model prediction 保留扩展空间。
