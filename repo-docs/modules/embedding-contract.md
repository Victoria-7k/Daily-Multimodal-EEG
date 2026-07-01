# 统一 embedding 契约

## 白话模型

统一 embedding 契约让后续模型不用关心每个模态当前是真实编码器还是 smoke encoder。每个窗口都会尝试产出四个固定维度数组：EEG、wear、face、audio。缺失或不可用的模态用零向量和 `modality_mask` 表示，成功读取的模态用同样长度的向量表示。

这个契约先服务工程闭环。它证明路径、窗口、mask、质量报告和批量保存是通的；模型效果不是当前 `basic` profile 的目标。后续替换真实编码器时，最重要的是保持数组键、样本顺序、mask 顺序和报告字段可读。

## 代码模型

[basic encoder](../../src/daily_multimodal/embeddings/basic.py) 定义 `EMBED_DIM = 256` 和 `MODALITY_ORDER = ("eeg", "wear", "face", "audio")`。`extract_basic_embedding` 读取一个窗口，返回 `EmbeddingSample`。wear 分支会扫描 PPG/GSR/ACC CSV 的窗口内数值，计算均值、标准差、最小值、最大值等基础统计；EEG、face、audio 分支在当前阶段使用文件大小、窗口时长和路径 salt 生成 metadata-derived smoke 向量。

[真实 embedding 契约](../../src/daily_multimodal/embeddings/contracts.py) 是阶段 11 后新增的保护层。`RealEmbeddingResult` 记录一个真实单模态结果的 `sample_id`、`event_id`、`subject_id`、`modality`、`embedding`、`mask_value`、`quality_flags`、`encoder_version` 和 `source_paths`；`validate_embedding_shape` 只接受 `(256,)` 或 `(N, 256)` 的浮点数组，并拒绝 NaN、无限值和非浮点 dtype。真实 encoder 后续替换时，应先通过这层检查，再进入 `.npz` 打包。

[失败清单模块](../../src/daily_multimodal/embeddings/failures.py) 固化了真实 encoder 的可定位失败记录。`EmbeddingFailure` 必须包含 `modality`、`encoder_profile`、`stage`、`error_type` 和 `source_path` 等定位字段；`write_failure_list` 即使没有失败也会写出 JSON 空数组 `[]`，让阶段 12 以后每个失败窗口都能追到模态、文件、依赖或处理阶段。

[批处理保存器](../../src/daily_multimodal/embeddings/pipeline.py) 把多个 `EmbeddingSample` 堆叠成 `.npz`：

```text
eeg_emb, wear_emb, face_emb, audio_emb -> (N, 256)
modality_mask -> (N, 4), order [eeg, wear, face, audio]
sample_id, event_id, subject_id, session_id -> object arrays
labels, source_paths -> JSON strings
```

[embedding 测试](../../tests/test_embedding_pipeline.py) 确认一个只有 wear 可用的窗口会得到 `[0, 1, 0, 0]` mask、非零 `wear_emb` 和零 `eeg_emb`。同一测试也确认保存后的 `.npz` 维度是 `(2, 256)` 和 `(2, 4)`，并确认精确 `video_candidates` 会优先于日期级 `candidate_mp4_paths`。

阶段 12 的 [真实缓存准备模块](../../src/daily_multimodal/embeddings/cache.py) 尚不生成最终真实 embedding；它先把切片边界和目标缓存路径固定下来。cache key 使用 `{sample_id}/{modality}/{encoder_profile}`，audio 写 mono 16 kHz wav，face 写 OpenFace CSV 目标路径，EEG 和 wear 写窗口 JSON 描述。这样后续 WavLM、OpenFace、EEG 和 wear sequence encoder 失败时，可以先判断是缓存/切片问题还是模型问题。

[Audio 真实模块](../../src/daily_multimodal/embeddings/audio_real.py) 是第一个消费真实缓存的 encoder 接入点。它从 `audio_clips/<sample_id>/<encoder_profile>/audio.json` 读取 wav 路径，要求 frozen backend 返回 `[frames, hidden_dim]`，再 mean pooling 并投影到 256 维。后续的 EEG、face、wear 真实模块也沿用同一单模态 `.npz` 形状：只写本模态 embedding，并用 `modality_mask` 标记该模态是否可用。

v2 profile 仍遵守同一 `(N, 256)` 契约，但在 `quality_flags` 中暴露更多可审计信息。`audio_opensmile_egemaps_v1` 把 openSMILE eGeMAPS Functionals 当作单帧功能特征投影；`audio_emotion2vec_plus_v1` 对 frame 特征做 `mean_std_max` pooling 后投影；`wear_physio_features_v2` 用 PPG HR/HRV、GSR slope/SCR 和 ACC motion/stationary 特征投影，并把 `physio_feature_names`/`physio_feature_values` 写入每个样本的质量字段。缺依赖或缺 checkpoint 时，这些 profile 写结构化 failure，不静默退回旧 encoder。

[真实多模态打包器](../../src/daily_multimodal/embeddings/real_pipeline.py) 是阶段 17 的合并入口。它以 window index 为主表保留样本顺序、标签和 source paths，再按 `sample_id` 合并 EEG/Wear/Face/Audio 单模态真实 `.npz`。缺失或质量 mask 为 0 的模态会写零向量并保持 `modality_mask=0`；成功模态保留 `(N, 256)` embedding。输出仍兼容阶段 9/10 训练入口，同时额外写入每个样本的 `quality_flags` 和 `encoder_versions` JSON 字符串，便于定位真实 encoder 的质量和版本。

当前全量 all-real 产物保持同一契约，但可用性不再要求四模态全为 1。服务器 v2 全量打包的 `modality_mask` sum 是 `[738, 781, 207, 781]`，说明 EEG 有 43 行缺失，Face 有 207 行通过 true OpenFace 质量门槛、501 行因低成功率被 mask、73 行仍缺 CSV，Wear 和 Audio 全部可用。这个设计让阶段 18 可以比较 all-real、without-face、single real replacement 等实验，而不是因为某个模态缺失就丢掉整行样本。

## 接下去阅读

在主路径里，[Step 5: basic encoder 写出统一 embedding 契约](../walkthroughs/one-real-run.md#step-5-basic-encoder-写出统一-embedding-契约) 解释这层契约如何被脚本使用。需要查 `.npz`、报告和阶段产物时读 [运行命令和产物](../references/commands-and-artifacts.md)；需要查窗口字段时读 [字段契约](../references/data-contracts.md)。

证据状态：除特别标注外，本页基于当前源码和测试已确认。
