# 一条事件如何变成 smoke embedding

这条导览跟着一条评分事件走到最终输出。读完以后，你应该能回答三件事：事件从哪里来，窗口如何定界，后续模型为什么能稳定读取结果。精确命令集中放在 [运行命令和产物](../references/commands-and-artifacts.md)，字段名集中放在 [字段契约](../references/data-contracts.md)。

## Step 1: 只读元数据生成事件记录

一次真实运行从 `scripts/01_build_manifest.py` 开始。入口只负责把 `src/` 放进 `sys.path`，实际工作交给 [manifest 构建器](../../src/daily_multimodal/manifest/build_manifest.py)。构建器遍历 EEG 数据集里的 `sub-*/ses-*/beh/*.tsv`，用每一行评分记录里的 `absolute_onset_time` 作为事件时间，再去匹配 EEG sidecar、wear CSV 时间段和视频日期目录。

这个阶段不读取大型 BDF、MP4 或原始音频内容。它只收集路径、时间、标签和可用性布尔值，所以适合先在服务器上做只读覆盖率检查。当前本地同步的 [汇总报告](../../outputs/reports/manifest_summary.json) 确认有 `1272` 个事件，其中 `995` 个满足日期级完整多模态候选。

## Step 2: 视频音频候选变成精确片段

日期级视频候选只能说明某一天目录里有 MP4，不能保证某个评分窗口被哪段视频覆盖。`scripts/02_align_video_audio.py` 会先从 manifest 收集候选 MP4，再让 [视频音频对齐模块](../../src/daily_multimodal/alignment/video_audio_alignment.py) 调用 `ffprobe`，读取 `creation_time`、`duration` 和音频流元数据。长任务复跑时，脚本会复用 ffprobe cache；`--retry-failed-ffprobe` 会重新探测失败记录，`--ffprobe-timeout 0` 或负数会把超时传给 subprocess 时改成不限时。

对齐模块把 MP4 的 UTC `creation_time` 转到 `Asia/Shanghai`，再和事件窗口相交。相交结果写入 `video_candidates`，包含 `clip_start_seconds`、`clip_end_seconds`、`overlap_seconds`、`covers_window` 和音频流信息。这个步骤让后面的 `basic` embedding 优先使用精确 `video_candidates`，而不是只拿日期目录里的第一个候选。当前本地 `outputs/` 尚未同步精确对齐产物，所以读本地副本时仍要区分日期级候选和代码已支持的精确候选。

## Step 3: 事件切成可复用窗口

窗口索引把事件记录变成训练或探针可以复用的样本记录。`scripts/03_build_window_index.py` 调用 [窗口构建函数](../../src/daily_multimodal/alignment/event_windows.py)，默认取事件发生前 `-10` 到 `0` 秒，窗口长度 `10` 秒，步长 `5` 秒。这个默认配置会为一个事件生成一个基础窗口；如果范围更长，则按步长滑动生成多个窗口。

窗口记录的核心变化是 `event_id` 变成稳定的 `sample_id`，例如测试里确认的 `sub-02_ses-03_00_row-0012_win-0000`。同一条记录还携带 EEG 起点、wear 路径、视频候选、标签和 `has_eeg`、`has_wear`、`has_face`、`has_audio` 等可用性标记。想理解这层概念，可以继续读 [事件窗口的白话模型](../modules/event-window.md)。

## Step 4: 单事件探针先检查形状和非空数据

在真正写 embedding 前，`scripts/03_probe_one_event.py` 会选一个窗口并生成探针报告。探针不训练模型，也不做复杂特征；它检查 EEG 预期重采样形状、PPG/GSR/ACC 在窗口内的行数，以及视频和音频候选数量。当前本地 [探针形状报告](../../outputs/reports/probe_one_event_shapes.txt) 显示一个同步样本有 `2500` 个 EEG 目标采样点、`200` 行 PPG、`400` 行 GSR、`200` 行 ACC 和 `5` 个视频候选。

这个探针的价值是把时间对齐错误尽早暴露出来。窗口开始结束时间一旦错了，wear 行数、EEG offset 或视频候选数量会先变得异常，而不是等到训练脚本里才发现样本不可用。

## Step 5: basic encoder 写出统一 embedding 契约

`scripts/04_extract_one_event_embeddings.py` 和 `scripts/05_extract_smoke_embeddings.py` 最终都会进入 [embedding 批处理](../../src/daily_multimodal/embeddings/pipeline.py)。批处理逐个窗口调用 [basic encoder](../../src/daily_multimodal/embeddings/basic.py)，再把结果保存成压缩 `.npz` 和 JSON 报告。

当前 `basic` encoder 的定位是 smoke 流水线。`wear_emb` 会读取窗口内 PPG/GSR/ACC 数值并计算统计特征；`eeg_emb`、`face_emb`、`audio_emb` 目前用 metadata-derived placeholder 检查文件存在性、路径、统一维度和 mask。输出固定包含 `eeg_emb`、`wear_emb`、`face_emb`、`audio_emb` 四组 `(N, 256)` 数组，以及顺序为 `[eeg, wear, face, audio]` 的 `modality_mask`。更细的输出约束在 [统一 embedding 契约](../modules/embedding-contract.md)。

## Step 6: subject 级入口扩大同一条路径

`scripts/06_extract_subject_embeddings.py` 没有引入新的 embedding 形状，它只是先用 [subject 选择器](../../src/daily_multimodal/embeddings/subject.py) 过滤某个 `subject_id` 的窗口，再复用同一个批处理和保存逻辑。`--require-all-modalities` 打开后，窗口必须同时具备 `has_eeg`、`has_ppg`、`has_gsr`、`has_acc`、`has_face` 和 `has_audio`。

这说明阶段 7 的风险主要不在输出契约，而在样本选择、长时间 CSV 读取、精确视频候选和服务器资源。阶段 4-6 的服务器验证报告已经记录过一次性能问题：重复读取大型 wear CSV 会拖慢 smoke test；当前实现通过缓存 CSV 解析和二分定位窗口范围解决。

## Step 7: 完整候选集和 baseline 验收

`scripts/07_extract_all_embeddings.py` 现在负责阶段 8：读取已有 `window_index` 或从 manifest 构建基础窗口，默认保留 `has_eeg`、`has_ppg`、`has_gsr`、`has_acc`、`has_face` 和 `has_audio` 都为真的完整候选窗口，再复用同一个 basic embedding 批处理器写出 `all_complete_basic_embeddings.npz`、`all_complete_multimodal_manifest.jsonl`、全量报告和失败清单。

`scripts/08_train_baseline_mlp.py` 负责阶段 9：读取全量 `.npz`，先做 128 窗口以内的过拟合检查，再按 subject split 训练 `eeg_only`、`wear_only`、`audio_only`、`face_only`、`eeg_wear`、`eeg_audio`、`eeg_face` 和 `full` 八组轻量 MLP 回归 baseline。当前实现是 numpy 版小模型，不依赖 PyTorch，目标是验收 embedding 是否可被稳定读取和学习；真实模型升级应继续与这个 baseline 在同一 split 下对照。

## Step 8: baseline 固化后做第一个融合升级

`scripts/10_run_upgrade_ablation.py` 负责阶段 10。第一步用 `--snapshot-baseline` 固化阶段 9 的 `baseline_mlp_metrics.json` 和表格，写出 `baseline_reference_metrics.json`、`baseline_reference_table.md` 和 `baseline_reference_manifest.json`。这个快照是后续进阶模型的回退基准，避免新实验覆盖原始 baseline。

第一版真实升级是 `modality_token_attention`，它不重新抽 EEG、视频或音频原始特征，而是复用阶段 8 的四个 `(N, 256)` embedding。实现会把 EEG、wear、audio、face 堆成四个 modality token，用 `modality_mask` 排除缺失 token，以轻量 attention pooling 得到 `[N, 256]` 融合向量，再接阶段 9 同一套 MLP 回归头和 subject split。服务器全量对照中，baseline full test RMSE 为 `0.8756`，`modality_token_attention` test RMSE 为 `0.6968`，因此在 `model_upgrade_ablation_table.md` 中标记为 `accepted`。

## Step 9: 真实 encoder 前先固定契约和缓存

阶段 11 不直接接 WavLM、OpenFace 或 EEG 深度模型，而是先把真实 encoder 的边界收紧。[真实契约模块](../../src/daily_multimodal/embeddings/contracts.py) 要求每个真实单模态 embedding 是 `(256,)` 或 `(N, 256)` 浮点数组，不能包含 NaN 或无限值；[失败清单模块](../../src/daily_multimodal/embeddings/failures.py) 要求每条失败都带上 sample、modality、encoder profile、stage、error type 和 source path。空失败清单仍写为 `[]`，这样成功运行和“没有写失败文件”不会混在一起。

阶段 12 再运行 `scripts/11_prepare_real_embedding_cache.py`，从窗口索引准备真实缓存。Audio 会按 `video_candidates` 的精确秒数切出 mono 16 kHz wav；face 只生成 OpenFace CSV 目标路径；EEG 和 wear 写出窗口 JSON，记录 BDF 或 PPG/GSR/ACC 源路径、窗口时间和目标采样参数。readiness report 会列出 EEG、wear、face、audio 各自的 ready count、missing count 和失败清单路径。这个阶段的目标是先区分“数据切片不可用”和“深度模型不可用”，避免阶段 13 以后把 ffmpeg、OpenFace、checkpoint 和模型 shape 问题混成一类错误。

## Step 10: Audio 先接真实 frozen encoder

阶段 13 的入口是 `scripts/12_extract_audio_embeddings.py`。它读取阶段 12 生成的 `audio_clips` cache，而不是重新从 MP4 切片；每个窗口通过 `audio.json` 找到 16 kHz mono wav，交给 WavLM 或 wav2vec2 frozen backend 生成 frame-level hidden states，再做 mean pooling 和固定种子投影，写出 `audio_emb` 为 `(N, 256)` 的 `.npz`。输出还带 `sample_id`、`event_id`、`subject_id`、`modality_mask`、`quality_flags` 和 `encoder_version`，方便后面把 only-audio-replaced 对照接回阶段 9/10 的训练入口。

这一阶段不允许静默 fallback。checkpoint 路径不存在时写 `checkpoint_missing`；服务器缺 `torch`、`torchaudio` 或 `transformers` 时写 `dependency_missing`；cache 或 wav 缺失时写 `source_missing`。2026-06-29 的服务器验证使用 `lzs` 环境和 `facebook/wav2vec2-base-960h` safetensors fallback 跑通了 10 窗口和 `sub-12` 单被试；全量与 ablation 留到后续统一执行。

## 验证

本地最小验证命令是：

```powershell
python -m pytest tests -q
```

理解主线后，可以按 [运行命令和产物](../references/commands-and-artifacts.md) 在服务器或同步副本上复现阶段命令。没有真实数据路径时，单元测试仍能确认 manifest 匹配、窗口切分、视频音频对齐、embedding 输出、subject 选择、完整候选集提取、baseline subject split、升级对照判定、真实 embedding 契约、阶段 12 缓存失败语义和阶段 13 audio real 输出契约这些核心行为。

证据状态：除特别标注外，本页基于当前源码、测试、配置、本地输出副本和阶段验证记录已确认。
