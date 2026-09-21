# Daily-affect 多模态 EMA-bag 技术路线

> Status: Current parallel route
> Route role: 0906 EMA-bag；与 0814 window 并列使用
> Research index: [研究文档索引](../../README.md)
> 文档版本：2026-09-08
> 结果数据截止：2026-09-08
> 范围：本文只说明 Daily-affect。窗口级连续回归和 EQL-CAF temporal-token 使用不同监督单位、模型与评价口径，不纳入本文结果排名。

## 1. 问题定义

Daily-affect 将一次 EMA 疲劳评分视为一条五级序数预测样本。模型读取评分前约两分钟内的 23 个重叠窗口，融合 EEG、Wear、Video、Audio 的历史证据，预测一个 event-level 疲劳等级 1--5。

路线关心两个问题：

1. EMA 评分由完整历史、近期状态，还是特定时间尺度的变化决定？
2. 同一窗口的四个模态应如何按可用性、当前证据和此前状态进行融合？

主指标为 quadratic weighted kappa（QWK）。它按照类别间二次距离加权，5 预测为 1 的代价高于 5 预测为 4。Macro-F1 和 ordinal MAE 分别描述类别均衡表现和平均等级误差。模型五类概率导出的 expected score 可报告 RMSE、raw Pearson r、within-subject centered r；它们是辅助连续读数，QWK 保持为主选型指标。

## 2. 数据与监督契约

### 2.1 Canonical EMA bag

| 项目 | 当前定义 |
| --- | --- |
| Canonical EEG-aligned 窗口数 | 28,819 |
| EMA event 数 | 1,253 |
| 每 event 窗口数 | 23 |
| 窗口长度与 stride | 10 秒，5 秒 |
| 时间索引 | event_window_id=0..22；0 最早，22 最接近评分时刻 |
| 模态顺序 | [EEG, Wear, Video, Audio] |
| 每模态 token | 256D |
| 模型输入 | tokens: (B,23,4,256)，modality_mask: (B,23,4) |
| 标签 | 每 event 一条 EMA fatigue，y in {1,2,3,4,5} |

modality_mask[t,m] 是窗口级模态可用性。缺失模态不参与该窗口融合；某窗口的四模态都缺失时，该窗口不参与时间汇总。23 个窗口是同一个监督样本的时间上下文，并非 23 条复制标签的独立样本。

scripts/daily_affect/73_build_daily_affect_bags.py 构建 bag，保存 sample_id_matrix、event 标识和 bag-level split 索引，使每个 event 预测可回溯到 23 个组成窗口。

### 2.2 三种主协议

| 协议 | 要回答的问题 | 执行原则 |
| --- | --- | --- |
| cross_subject | 新受试者泛化 | 按预定义 subject 边界构建 train、val、test bag |
| cross_day | 跨日期状态、设备与环境变化后的泛化 | 按预定义日期边界，重点审计 token drift 与跨日缺失 |
| within_subject_day | 已见个体在保留日期或时间段的泛化 | 按预定义 subject-day 划分，仍以 bag 为训练和评估单位 |

三种协议独立构建、训练、选型和报告。normalization、类别权重、模型参数和 probe temperature 只在 train 拟合；validation 用于 early stopping 和 checkpoint 选择；test 只用于锁定配置的最终评估。

### 2.3 Token preprocessing

上游四模态 token 是 Daily-affect head 的冻结输入。Daily-affect 显式比较两种 normalization 和 adapter：

| 因素 | shared | per_modality |
| --- | --- | --- |
| Token normalization | 四模态共用 train-only 均值和标准差 | 每模态各自拟合 train-only 统计量 |
| Adapter | 四模态共用 256->128 线性投影 | 每模态独立 256->128 投影 |

两种 adapter 都会加入可学习的 128D modality embedding，保留 EEG、Wear、Video、Audio 的身份。训练时默认 0.1 概率 modality dropout，使融合器学会在局部模态缺失时重新分配权重。

## 3. 输出、损失与评价

### 3.1 五分类输出

最终 head 产生五类 logits，softmax 后得到：

    p = (p1, p2, p3, p4, p5),  sum(c=1..5) pc = 1

离散预测为 argmax(p)。连续 expected score 由同一概率分布计算：

    y_hat_exp = sum(c=1..5) c * pc

项目没有额外连续回归 head。QWK、Macro-F1、ordinal MAE 评价离散类别；RMSE、raw r、centered r 评价 \hat y_exp。

### 3.2 主训练目标

除 expected-score Huber screen 外，最终分类头使用：

    L_head = L_weighted_CE
           + 0.5 * L_cumulative_ordinal
           + 0.1 * L_within_subject_rank

| 分量 | 技术含义 |
| --- | --- |
| class-weighted CE | 五分类基础监督，并降低类别不均衡造成的少数等级忽视 |
| cumulative ordinal loss | 监督四个累计阈值概率。跨越更多等级阈值的错误代价更大 |
| within-subject ranking loss | 对同一 subject 内标签不同的 event，约束 expected score 的排序方向 |

有 modality probe 的变体还加入 0.1 L_probe。probe 使用每模态在整个 bag 内的可用 token 聚合表示预测同一个 event 标签，用于路由辅助监督，并不是独立预测器。

模型按 validation QWK 保存 checkpoint。其定义为：

    w(i,j) = (i - j)^2 / (5 - 1)^2
    QWK    = 1 - sum(i,j)[w(i,j) * O(i,j)] / sum(i,j)[w(i,j) * E(i,j)]

其中 O 是观察混淆矩阵，E 是在相同行、列边际分布下的随机期望混淆矩阵。

## 4. 起点：window_replicated 与 bag_static

### 4.1 window_replicated

window_replicated 对每个有效窗口直接产生五类概率，再将一个 event 的 23 个窗口概率按时间权重汇总。它是“每个局部窗口都独立承载 EMA 标签”的对照，不是正式 event-level 监督定义。

### 4.2 bag_static：正式静态 baseline

bag_static 是后续候选的主要 event-level baseline。它先在每个 10 秒窗口融合四个模态，再以固定时间规则汇总 23 个窗口。

#### 窗口内：attention 模态融合

令 x_(t,m) 为第 t 个窗口、第 m 个模态的 256D token。adapter 与 modality embedding 给出：

    a(t,m) = A_m * x(t,m) + e_m,    a(t,m) is 128-dimensional

同一窗口的四个表示经过单头 self-attention：

    h(t,1:4) = MHA(a(t,1:4), a(t,1:4), a(t,1:4))

每个模态的更新表示可参考同一 10 秒内的其他可用模态。模型再以可学习 query q 做 pooling：

    alpha(t,m) = softmax over available m of dot(h(t,m), q)
    e(t)       = sum over m of alpha(t,m) * h(t,m)

保存于 diagnostics 的 modality_weights 即 \alpha_(t,m)。它描述本窗口最终融合时各可用模态的软路由比例：缺失模态为零，其余模态重新归一化。它不是因果贡献，也不是 self-attention 内部完整 4x4 attention map，因为内部矩阵未被保存。

#### 窗口间：固定时间汇总

默认 temporal_policy=uniform。令 v_t 表示窗口 t 至少有一个有效模态：

    beta(t) = v(t) / sum(j) v(j)
    s       = sum(t=0..22) beta(t) * e(t)

如果 23 个窗口都有效，每个权重是 1/23。uniform 仅指窗口之间等权，不代表窗口内四模态等权。s 再经 LayerNorm、MLP 与五类 head 得到 event 预测。

bag_static 没有跨窗口状态记忆，也没有依 event 内容改变时间权重。它给出最清晰的问题基线：局部四模态融合后，对完整两分钟历史等权平均是否足够？

## 5. 从 bag_static 向后加入的模块

以下变体共享输入契约、五类 head、主损失、validation-QWK checkpoint 选择与训练期 modality dropout。

### 5.1 state_uniform：跨窗口 affect state

state_uniform 在当前窗口中先以线性 evidence score 路由四模态：

    alpha(t,m) = softmax over m of baseScore(a(t,m))
    e(t)       = sum over m of alpha(t,m) * a(t,m)

随后由 GRUCell 写入状态：

    s(t) = GRUCell(e(t), s(t-1)),    s(-1) = 0

最后均匀汇总 s_0..s_22。较晚状态携带此前窗口记忆。

实现边界很重要：state_uniform 的历史状态不直接参与当前 \alpha_(t,m)；它先融合当前窗口，再更新 GRU。它同时把 bag_static 的窗口内 self-attention 加 query pooling 换成线性 evidence scorer。因此 bag_static 到 state_uniform 同时改变了窗口融合和时间状态两部分，差异不能简单解释为“只加了 GRU”。

### 5.2 prior_uniform：状态先验引导路由

prior_uniform 从上一状态构造当前先验：

    p(t) = LayerNorm(s(t-1) + transition(s(t-1)))

模型计算当前 token 与 prior 的相容性：

    c(t,m) = g(tanh(Wq * p(t) + Wm * a(t,m)))

并加入当前路由：

    alpha(t,m) = softmax over m of [baseScore(a(t,m)) + c(t,m)]

这就是 prior-guided modality weighting。它让模型在分配当前模态权重时判断 token 是否与此前累积的 latent state 连续。prior 是模型学习出的历史状态先验，不是人工赋予某个模态的固定重要性。

prior_uniform 仍以均匀规则汇总时间状态。它训练 cumulative modality probe，但不使用 ordinal-difficulty penalty。

### 5.3 prior_ordD_uniform：序数难度抑制

每模态 probe 输出五级分布，并计算两种 0--1 难度：

    d_entropy = -sum(c) pc * log(pc) / log(5)
    d_ordvar  =  sum(c) pc * (c - E[c])^2 / 4

默认混合为：

    d(m) = (1 - beta_ord) * d_entropy(m) + beta_ord * d_ordvar(m)
    beta_ord = 0.25

prior_ordD_uniform 在路由分数中扣除难度：

    score(t,m) = baseScore(a(t,m)) + c(t,m) - lambda_D * d(m)
    lambda_D = 0.25

probe 先 warm-up，再逐步提高 \lambda_D。默认 detach_difficulty=true，使分类主损失不通过 routing penalty 改变 difficulty 的定义。其目标是避免对五级标签高度不确定的模态占据过高权重。

### 5.4 固定时间规则：检验标签真正看哪段历史

固定规则将 \beta_t 预设为窗口位置函数，对所有 event 相同：

| policy | 规则 |
| --- | --- |
| uniform 或 full_2min | 每个有效窗口等权 |
| last_10s | 仅 index 22 |
| last_30s | index 18..22 |
| last_60s | index 12..22 |
| first_30s | index 0..4 |
| kernel_short / kernel_medium / kernel_long | 最近窗口更大，固定指数衰减，时间常数为 15、45、120 秒 |

它们用于检验 EMA 标签的 look-back。近期规则优于完整历史，表示评分更依赖近期状态；这些规则本身不随 event 自适应。

### 5.5 global_kernel_no_prior：全局 learned time kernel

模型预定义 short、medium、long 三个指数基核，并学习所有 event 共用的混合：

    pi   = softmax(u)
    beta = pi_short * b_short + pi_medium * b_medium + pi_long * b_long

它让数据选择一个总体时间尺度，但每个 event 使用同一条时间曲线。no_prior 表示没有 state-prior compatibility；ordinal-difficulty 也不实际进入路由分数。

### 5.6 dynamic_kernel_no_prior：event 自适应时间核

该变体保留 GRU states，但不使用 state-prior compatibility。它根据每个 event 的平均有效状态产生核混合：

    s_bar(n) = sum(t)[v(t) * s(t)] / sum(t) v(t)
    pi(n)    = softmax(W * LayerNorm(s_bar(n)))
    beta(n)  = sum(k in {short, medium, long}) pi(n,k) * b(k)

每个 event 可在短、中、长时程之间选择不同权重。它隔离了动态时间核在无 prior 路由时的作用。

### 5.7 dynamic_kernel_prior_uniform：当前 cross-day 优先候选

该模型在 dynamic_kernel_no_prior 上加入 prior-guided modality routing，但不加入 ordinal-difficulty penalty。它同时学习：

1. 窗口内，依据此前状态重分配 EEG、Wear、Video、Audio 权重。
2. event 间，依据各自状态轨迹混合 short、medium、long 时间核。

名称尾部 prior_uniform 是历史命名，表示 prior 路由没有额外 ordD penalty，配置入口仍要求 temporal_policy=uniform。实际 forward 的 temporal_weights 来自 dynamic kernel，并非 23 窗口等权。

### 5.8 dynamic_kernel 和 dynamic_fixed_*

dynamic_kernel 是完整版本：state、prior-guided routing、ordinal-difficulty penalty、event 自适应时间核均启用。

dynamic_fixed_short、dynamic_fixed_medium、dynamic_fixed_long 保留完整 state/prior/ordD 路由，但强制选择三个基核之一。它们检验动态核收益能否由一条固定近期规则完全解释。

### 5.9 P0--P5 routing profiles

scripts/daily_affect/75_run_daily_affect_state_matrix.py 将 probe 与 difficulty 设计拆开：

| profile | 含义 |
| --- | --- |
| P0 no-prior | 无 prior 对照，difficulty 不进入路由 |
| P1 categorical entropy | 五分类 probe，以预测熵作难度 |
| P2 cumulative entropy | 累计序数 probe，以预测熵作难度 |
| P3 ordinal mix | 累计序数 probe，以 entropy 与 ordinal variance 混合作 penalty |
| P4 probe calibrated | 在 P3 后用 validation 校准每模态 probe temperature，再短程 fine-tune routing |
| P5 end-to-end | 在 P3 基础上允许 routing penalty 向 probe 难度端到端回传 |

对 no_prior model id，即使配置记录含 difficulty_ordinal，模型内部仍解析为 none；只有 prior-enabled 模型会实际使用 difficulty。

## 6. 已完成证据

### 6.1 Cross-day focused 7-seed 阶梯消融

设置固定为 cross_day / A1_Wphysio_full / per_modality normalization / per_modality adapter / weighted CE + 0.5 ordinal + 0.1 rank。bag_static 是逐 seed 配对基线。三 seed 行用于机制筛选，七 seed 行构成当前更强证据。

| 模型 | seeds | QWK | 相对 bag_static Delta QWK | QWK 胜出 | 解读 |
| --- | ---: | ---: | ---: | ---: | --- |
| bag_static | 7 | 0.1864 +/- 0.0432 | - | - | 正式静态 event baseline |
| state_uniform | 3 | 0.1672 +/- 0.0295 | -0.0221 +/- 0.0744 | 1/3 | 单独 state 组合未显现稳定收益 |
| prior_ordD_uniform | 3 | 0.1690 +/- 0.0243 | -0.0203 +/- 0.0342 | 1/3 | 静态时间汇总下 prior/ordD 组合未形成收益 |
| dynamic_fixed_long | 3 | 0.2029 +/- 0.0292 | +0.0136 +/- 0.0535 | 2/3 | 长时程固定核有小幅信号 |
| dynamic_fixed_medium | 3 | 0.2109 +/- 0.0328 | +0.0216 +/- 0.0674 | 1/3 | 中期核尚不稳定 |
| dynamic_fixed_short | 3 | 0.2497 +/- 0.0568 | +0.0605 +/- 0.0616 | 2/3 | 近期核是强的方向性信号 |
| global_kernel_no_prior | 7 | 0.2230 +/- 0.0512 | +0.0366 +/- 0.0716 | 6/7 | 全局 learned look-back 有稳定正向差异 |
| dynamic_kernel_no_prior | 3 | 0.2007 +/- 0.0270 | +0.0114 +/- 0.0718 | 1/3 | 动态核独立证据仍需补齐 |
| dynamic_kernel_prior_uniform | 7 | 0.2515 +/- 0.0402 | +0.0651 +/- 0.0656 | 6/7 | 当前跨日优先候选 |
| dynamic_kernel | 3 | 0.2292 +/- 0.0247 | +0.0400 +/- 0.0520 | 2/3 | 完整 ordD 路径有方向性证据 |

当前可确认结论：dynamic_kernel_prior_uniform 相对匹配 bag_static 有 7-seed、6/7 胜出的 QWK 提升，同时 ordinal MAE 从 0.9571 降至 0.9221。它是 Daily-affect 的 cross-day 优先候选。dynamic_kernel_no_prior 仍须补齐同一七 seed，才能对 state-prior compatibility 的独立效应作正式归因。

### 6.2 Cross-subject 与 within-subject-day focused 3-seed 阶梯消融

为避免将 cross-day 的结论外推到其余协议，本轮在两个协议各自固定一个 route、normalization 与 adapter 组合，在相同三个 seed 上从 `bag_static` 逐层加入 state、动态时间核和 prior。训练目标固定为 weighted CE + 0.5 ordinal + 0.1 rank；每一行与同 seed 的 `bag_static` 配对。

**Cross-subject：A2_Wdeep_full / shared normalization / shared adapter**

| 模型 | QWK | 相对 bag_static 的 QWK | QWK 胜出 | Macro-F1 | ordinal MAE | 决策 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| bag_static | 0.1707 +/- 0.0654 | 基线 | -- | 0.2159 | 0.8295 | 当前受控对照最高 |
| state_uniform | 0.1587 +/- 0.0217 | -0.0120 +/- 0.0459 | 1/3 | 0.1695 | 0.8131 | 不推进 |
| dynamic_kernel_no_prior | 0.1125 +/- 0.0219 | -0.0582 +/- 0.0460 | 1/3 | 0.1535 | 0.8965 | 不推进 |
| prior_ordD_uniform | 0.0573 +/- 0.0819 | -0.1134 +/- 0.1180 | 1/3 | 0.1619 | 0.9141 | 不推进 |
| dynamic_fixed_short | 0.0392 +/- 0.0470 | -0.1315 +/- 0.0920 | 0/3 | 0.1536 | 0.9646 | 不推进 |
| dynamic_fixed_medium | 0.0605 +/- 0.0213 | -0.1102 +/- 0.0673 | 0/3 | 0.1375 | 0.9760 | 不推进 |
| dynamic_fixed_long | 0.0476 +/- 0.0059 | -0.1231 +/- 0.0698 | 0/3 | 0.1479 | 1.0013 | 不推进 |
| global_kernel_no_prior | 0.0067 +/- 0.0733 | -0.1640 +/- 0.0285 | 0/3 | 0.1375 | 0.9508 | 不推进 |
| dynamic_kernel | 0.0458 +/- 0.0393 | -0.1249 +/- 0.0321 | 0/3 | 0.1520 | 0.9949 | 不推进 |
| dynamic_kernel_prior_uniform | 0.0444 +/- 0.0388 | -0.1263 +/- 0.0342 | 0/3 | 0.1409 | 1.0808 | 不推进 |

该受控设置中，所有后续模块均未超过 `bag_static`。因此 cross-subject 当前保留静态 bag 聚合，不把 state、prior 或 kernel 扩展至 5--7 seed。该 `bag_static` 三 seed 均值高于此前广筛快照，但标准差较大，仍属于协议内描述性结果。

**Within-subject-day：B0_Wphysio_full / per_modality normalization / per_modality adapter**

| 模型 | QWK | 相对 bag_static 的 QWK | QWK 胜出 | Macro-F1 | ordinal MAE | 主要连续读数 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| bag_static | 0.1952 +/- 0.0315 | 基线 | -- | 0.2317 | 0.9251 | raw r 0.2363；RMSE 1.0391 |
| state_uniform | 0.2666 +/- 0.0441 | +0.0714 +/- 0.0367 | 3/3 | 0.2737 | 0.8850 | raw r 0.2877；RMSE 0.9928 |
| dynamic_kernel_no_prior | 0.2637 +/- 0.0230 | +0.0685 +/- 0.0087 | 3/3 | 0.2570 | 0.8889 | raw r 0.2687；RMSE 0.9752 |
| dynamic_fixed_long | 0.2596 +/- 0.0156 | +0.0644 +/- 0.0252 | 3/3 | 0.2658 | 0.8992 | raw r 0.2882；RMSE 0.9972 |
| prior_ordD_uniform | 0.2309 +/- 0.0276 | +0.0357 +/- 0.0242 | 3/3 | 0.2526 | 0.8889 | -- |
| global_kernel_no_prior | 0.2129 +/- 0.0341 | +0.0177 +/- 0.0076 | 3/3 | 0.2508 | 0.8721 | -- |
| dynamic_fixed_medium | 0.2145 +/- 0.0342 | +0.0193 +/- 0.0645 | 2/3 | 0.2561 | 0.8824 | -- |
| dynamic_fixed_short | 0.2115 +/- 0.0740 | +0.0163 +/- 0.0741 | 2/3 | 0.2372 | 0.9806 | -- |
| dynamic_kernel | 0.2049 +/- 0.0312 | +0.0097 +/- 0.0565 | 2/3 | 0.2422 | 0.9005 | -- |
| dynamic_kernel_prior_uniform | 0.1965 +/- 0.0382 | +0.0013 +/- 0.0663 | 1/3 | 0.2450 | 0.8915 | -- |

`state_uniform` 是 within-subject-day 当前 QWK 第一候选；`dynamic_kernel_no_prior` 的 QWK 方差较小、expected RMSE 最低，作为并列的机制候选保留。二者均需在相同配置下补到 5--7 matched seeds；`dynamic_kernel_prior_uniform` 没有获得增益，不进入扩展。这个协议中 state 或动态时间建模有价值，而显式 state-prior compatibility 没有带来附加收益。

### 6.3 EMA 标签的时间可辨识性

在 cross_day / A1_Wphysio_full / per_modality normalization / per_modality adapter 的三 seed bag_static 时间审计中：

| 时间规则 | QWK | 相对 full_2min |
| --- | ---: | ---: |
| full_2min | 0.1534 | 基线 |
| first_30s | 0.1143 | -0.0391 |
| last_10s | 0.1544 | +0.0010 |
| last_30s | 0.1800 | +0.0266 |
| last_60s | 0.1740 | +0.0206 |
| kernel_long | 0.1737 | +0.0203 |
| kernel_medium | 0.2119 | +0.0585 |
| kernel_short | 0.2211 | +0.0677 |

EMA 标签的可辨识信息集中在评分前的近期窗口，为 global/dynamic kernel 提供了动机。下一步需将 short、medium 与 static baseline 扩展到 5--7 matched seeds。

### 6.4 Missing 和 corruption 稳健性

在同一 cross-day 七 seed 设置，对 bag_static、dynamic_kernel_prior_uniform、global_kernel_no_prior 施加 missing、noise、shuffle。关键切片：

| 条件 | bag_static QWK | dynamic_kernel_prior_uniform QWK | 解读 |
| --- | ---: | ---: | --- |
| clean | 0.1864 | 0.2515 | 候选的 clean 优势 |
| missing video | 0.0990 | 0.2144 | 候选在 Video 缺失时更平稳退回其他模态 |
| shuffle video | 0.0778 | 0.0769 | Video 错误身份或时间对齐是共同脆弱点 |
| shuffle wear | 0.1890 | 0.1509 | 候选对错误 Wear token 更敏感 |
| noise EEG | 0.0007 | 0.1286 | EEG 强噪声显著损伤两种模型 |

cross-day test-event availability 为 EEG 1.0000、Wear 0.9333、Video 0.6319、Audio 0.5986；centroid-shift RMS 为 EEG 0.1738、Wear 0.5954、Video 0.8074、Audio 0.5896。下一步应把 Video ROI/追踪质量、缺失率和 day-shift 作为 quality-aware routing 特征，并将 Video/Wear dropout 纳入训练期实验。

### 6.5 Expected-score Huber 连续辅助损失

scripts/daily_affect/88_run_daily_affect_expected_score_huber.py 只在 matched bag_static bridge bag 上探索：

    L = L_head + lambda * L_Huber(y_hat_exp, y)

lambda 在 {0, 0.025, 0.05, 0.1, 0.2} 中仅按 validation 选择。lambda=0.2 的 validation raw-r 为 +0.0142、3/3 胜出，validation QWK 为 +0.0148。锁定后读取三 seed test：QWK -0.0029 +/- 0.0368、expected raw r +0.0031 +/- 0.0058、centered r +0.0093 +/- 0.0075、RMSE -0.0106 +/- 0.0170。

这是轻量连续读数改善候选，尚只有三 seed，且未扩展到 dynamic kernel；它不改变 QWK 主选型或当前候选结论。

## 7. 三种协议下当前最佳结果

下表按 Daily-affect 主指标 QWK 选取每个协议所有已完成结果中的最高值。三个赢家来自不同 token route 和训练配置，不能理解为同一模型已在三协议都完成 promotion。

| 协议 | 最佳 route 与模型 | 配置记录 | seeds | QWK | Macro-F1 | ordinal MAE | 证据状态 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| cross_subject | A2_Wdeep_full / bag_static | shared normalization + shared adapter | 3 | 0.1707 +/- 0.0654 | 0.2159 | 0.8295 | 最新受控阶梯中的协议内最高值；所有增量模块未过三 seed gate |
| cross_day | A1_Wphysio_full / dynamic_kernel_prior_uniform | per_modality normalization + per_modality adapter | 7 | 0.2515 +/- 0.0402 | 0.2732 | 0.9221 | 当前最强，已有 matched 7-seed static 对照 |
| within_subject_day | B0_Wphysio_full / state_uniform | per_modality normalization + per_modality adapter | 3 | 0.2666 +/- 0.0441 | 0.2737 | 0.8850 | 最新受控阶梯中相对 static +0.0714 QWK、3/3 胜出；待 5--7 seed promotion |

cross-day 最佳模型的 expected raw r 为 0.2833 +/- 0.0280，expected RMSE 为 1.0706 +/- 0.0550。它们是同一 1--5 量尺上的辅助连续读数，不改变按 QWK 选取最佳模型的规则。

## 8. 当前决策与下一步

1. 补齐 dynamic_kernel_no_prior 的 240729..240735 七个 matched seed，直接检验 state-prior compatibility 的因果贡献。
2. 将 dynamic_fixed_short、dynamic_fixed_medium 与 bag_static 扩展到 5--7 matched seeds，确定 EMA 标签的实际 look-back。
3. 在 dynamic_kernel_prior_uniform 中加入训练期 Video/Wear dropout，并检验 Video ROI 质量、缺失率和 day-shift 的 quality-aware routing 价值。
4. 将 within_subject_day 的 `state_uniform` 与 `dynamic_kernel_no_prior` 扩展至同一 5--7 个 matched seeds；前者按 QWK 主指标优先，后者检验其较低 RMSE 是否可复现。
5. cross_subject 保留 `bag_static`。当前 shared/shared 受控阶梯中，state、prior 和 kernel 没有正向 paired 证据；后续若重启该协议，应先重新筛选 token route 或 normalization/adapter 组合，而非继续扩展本轮增量模块。
6. cross-day 的 per_modality 结果不会自动成为其他协议默认；只有候选在同一 protocol、route、objective、normalization、adapter 与 seed 集合下持续优于 bag_static，且 subject-day paired bootstrap 支持方向一致性时，才提升为该协议默认模型。

## 9. 脚本与产物导航

| 目标 | 脚本 | 关键产物 |
| --- | --- | --- |
| 构建 EMA bags | scripts/daily_affect/73_build_daily_affect_bags.py | bags/{protocol}/{route}/seed_*/ema_bags.npz |
| Phase-0 静态对照 | scripts/daily_affect/74_run_daily_affect_phase0_baselines.py | window_replicated 与 bag_static paired runs |
| state/prior/kernel 矩阵 | scripts/daily_affect/75_run_daily_affect_state_matrix.py | P0--P5 routing profiles 与基础模型矩阵 |
| 多 seed 汇总和 atlas | scripts/daily_affect/76_summarize_daily_affect_results.py；77_plot_daily_affect_diagnostics.py | paired delta、bootstrap、diagnostics atlas |
| focused 机制消融 | scripts/daily_affect/78_report_daily_affect_focused_diagnostics.py；79_run_daily_affect_focused_ablation.py | fixed/global/dynamic kernel 与 prior 诊断 |
| 缺失与 corruption | scripts/daily_affect/80_run_daily_affect_focused_robustness.py | missing/noise/shuffle robustness report |
| prior 与 bottleneck | scripts/daily_affect/81_report_daily_affect_prior_guidance.py；82_run_daily_affect_bottleneck_audit.py | state-prior report、时间可辨识性与 drift |
| routing factorial | scripts/daily_affect/83_run_daily_affect_routing_factorial.py | normalization x adapter 解释性诊断 |
| expected-score 辅助损失 | scripts/daily_affect/88_run_daily_affect_expected_score_huber.py | validation-locked Huber screen |

当前关键同步产物：

- outputs/server_sync/daily_affect_dynamic_a1_crossday_routefix_20260906/norm_per_modality__adapter_per_modality/reports_7seed/daily_affect_ordinal_summary.md
- outputs/server_sync/daily_affect_dynamic_a1_crossday_routefix_20260906/norm_per_modality__adapter_per_modality/robustness_7seed/focused_robustness_report.md
- outputs/server_sync/daily_affect_bottleneck_audit_20260905/daily_affect_bottleneck_audit.md
- outputs/server_sync/daily_affect_expected_score_huber_20260907/expected_score_huber_screen.md
- outputs/server_sync/daily_affect_ordinal_20260903/reports/protocol_route_summary.csv

当前远端结果（尚未同步到本地 `outputs/server_sync/`）：

- /vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/outputs/daily_affect_focused_other_protocols_20260907/cross_subject/reports/daily_affect_ordinal_summary.md
- /vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/outputs/daily_affect_focused_other_protocols_20260907/within_subject_day/reports/daily_affect_ordinal_summary.md
