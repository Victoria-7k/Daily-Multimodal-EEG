# 真实多模态完整 Embedding 接入实施计划

## 项目整体目标和行为规范

本计划接续 [Daily Multimodal 多模态嵌入前半部分跑通计划](多模态嵌入前半部分跑通计划.md)。前半部分的目标是跑通皮肤电、PPG、加速度、面部录像、录音、脑电的多模态信号编码流程，完成从原始多模态数据到可复现、可替换、可训练的窗口级 embedding 工程；真实 embedding 接入阶段在这个闭环之上，把 smoke/basic encoder 逐步替换为真实窗口信号或真实特征抽取器。

**核心原则：** 先完成基础闭环，再尝试进阶模型；所有高级模型都是可替换插件，不是前置依赖。每次只升级一个模态或一个融合模块，并和基础版在同一数据切分、同一指标下对照。进阶模型效果不好、成本过高或稳定性差时，立即回退到基础版，不阻塞主流水线。

**总体思路：** 以每条情绪评分事件为中心，用 `absolute_onset_time` 做跨模态绝对时间对齐；EEG 使用 `RecordingStartTime + onset` 校验相对时间；PPG/GSR/ACC 使用 CSV 内绝对时间列切窗并合成为统一 `wear_emb`；视频与录音使用同日期 MP4 及其音频流切片。第一版已经完成 manifest、window index、单样本切窗、单样本基础 embedding、10 条样本 smoke test、单被试测试、完整可用样本集、baseline 和第一版融合升级；本计划继续逐步替换为 LaBraM/EEGPT、PatchTST/TS2Vec、OpenFace temporal Transformer、WavLM + openSMILE 和 modality Transformer。

**技术栈建议：** Python 3.10+、PyTorch、MNE、pandas/numpy/scipy、opencv/ffmpeg/ffprobe、torchaudio、transformers、OpenFace 或等价面部特征提取工具、openSMILE、pytest/unittest、YAML 配置、JSONL/NPZ/Markdown reports。

### 服务器工作目录与协作规范

必须使用个人服务器工作目录，不改动任何原始数据目录。当前服务器项目目录固定为：

```bash
/mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding
```

原因：`/mnt/dataset4` 空间较大，且 EEG 数据位于 `/mnt/dataset4/sitian/DailyEEG_dataset`，将个人工作目录固定在 `/mnt/dataset4/sitian/wzw` 下更便于和原始数据相邻管理。所有代码、配置、manifest、日志、临时输出、embedding 输出都应写入该 `wzw` 目录下。

后续工作规范 1：本地优先，在本地创建和修改文件，再同步至服务器端的 `wzw` 文件夹下。

本地工作目录：

```text
G:\Daily Multimodal
```

固定同步方向：

```powershell
scp -r .\configs .\src .\scripts .\tests .\pyproject.toml ncc_serve_4090:/mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding/
```

后续工作规范 2：大规模、长时间程序运行前都先做快速的小规模可行性测试。

强制执行顺序：

1. 先跑 1 条评分事件，确认所有模态可定位、可切窗、可输出 embedding。
2. 再跑 10 条评分事件，确认批处理、日志、异常跳过、输出格式稳定。
3. 再跑 1 个被试的所有可用事件，确认跨日期和跨 session 情况。
4. 最后才跑全部完整多模态事件。

每次扩大规模前都必须检查：

- manifest 或 window index 样本数是否符合预期。
- 每个模态的命中率是否符合预期。
- 输出 embedding shape 是否一致。
- 日志和失败清单中是否有静默失败。
- 输出目录是否写在 `/mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding` 下。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从当前阶段 8-10 的 smoke/basic 工程闭环，推进到 EEG、Wear、Face、Audio 四个模态都使用真实窗口信号或真实特征抽取器生成可训练、可复现、可回退的 256 维 embedding。

**Architecture:** 保持现有 `.npz` 契约不变：`eeg_emb`、`wear_emb`、`face_emb`、`audio_emb` 均为 `(N, 256)`，`modality_mask` 顺序仍为 `[eeg, wear, face, audio]`。新增真实 encoder profile、缓存层、失败清单 schema 和逐模态升级对照；每次只替换一个模态或一个融合模块，并和阶段 9 baseline reference、阶段 10 `modality_token_attention` 对照。

**Tech Stack:** Python 3.10+、numpy、MNE、ffmpeg/ffprobe、OpenFace、transformers/torchaudio 或 Hugging Face audio model runtime、PyTorch（真实深度模型阶段）、pytest/unittest、JSONL/NPZ/Markdown reports。

---

## 0. 当前状态基线

当前已经完成：

- 阶段 8：`all_complete_basic_embeddings.npz` 全量 995 条完整候选窗口，`failure_count=0`。
- 阶段 9：concat MLP baseline，`alert` 标签 subject split，full test RMSE `0.8756`。
- 阶段 10：`modality_token_attention` 融合升级，full test RMSE `0.6968`，`decision=accepted`。

当前仍是 smoke/basic embedding：

| 模态 | 当前实际 embedding | 完整目标 |
| --- | --- | --- |
| EEG | BDF 路径、文件大小、窗口时长生成 metadata smoke 向量 | 真实 EEG 窗口预处理 + LaBraM/EEGPT 或可回退 EEGNet/1D CNN frozen/projection |
| Wear | 真实读取 PPG/GSR/ACC 窗口数值，统计特征投影 | 统计特征 + raw sequence encoder，例如 TCN/PatchTST/TS2Vec |
| Face | MP4 路径、文件大小、窗口时长生成 metadata smoke 向量 | OpenFace CSV 窗口统计 + temporal GRU/Transformer；后续可加 VideoMAE |
| Audio | MP4/音频路径、文件大小、窗口时长生成 metadata smoke 向量 | MP4 音轨切片 + WavLM/wav2vec2 frozen embedding + openSMILE eGeMAPS |

## 1. 失败清单语义

`outputs/reports/model_upgrade_failures.json` 是升级实验的失败记录列表。

当前全量阶段 10 中：

```json
[]
```

这表示这次 `modality_token_attention` 升级没有记录到失败项，不是 bug，也不需要“修复”。它说明：

- subject split 完整；
- 升级实验已注册；
- 模型训练完成；
- metrics、model、ablation table、failure list 都成功写出；
- 没有 shape mismatch、NaN、缺文件、缺依赖或运行异常。

真正需要解决的是：在接入真实深度模型前，扩展失败清单 schema，让未来任何真实编码器失败都能被定位到具体模态、窗口、文件、依赖和阶段。

建议统一失败记录：

```json
{
  "sample_id": "sub-02_ses-03_00_row-0012_win-0000",
  "event_id": "sub-02_ses-03_00_row-0012",
  "subject_id": "sub-02",
  "modality": "audio",
  "encoder_profile": "wavlm_frozen_v1",
  "stage": "extract_audio_clip",
  "error_type": "extraction_failed",
  "error": "ffmpeg returned non-zero exit status 1",
  "source_path": "/mnt/dataset1/sitian/video/sub2/0228/example.MP4",
  "recoverable": true
}
```

必备 `error_type`：

```text
dependency_missing
checkpoint_missing
source_missing
extraction_failed
decode_failed
shape_mismatch
nan_embedding
quality_threshold_failed
oom
timeout
subject_split_incomplete
unsupported_upgrade
```

---

## 2. 文件结构规划

### 新增文件

```text
src/daily_multimodal/embeddings/contracts.py
src/daily_multimodal/embeddings/failures.py
src/daily_multimodal/embeddings/cache.py
src/daily_multimodal/embeddings/audio_real.py
src/daily_multimodal/embeddings/face_real.py
src/daily_multimodal/embeddings/eeg_real.py
src/daily_multimodal/embeddings/wear_real.py
src/daily_multimodal/embeddings/real_pipeline.py
scripts/11_prepare_real_embedding_cache.py
scripts/12_extract_audio_embeddings.py
scripts/13_extract_face_embeddings.py
scripts/14_extract_eeg_embeddings.py
scripts/15_extract_wear_embeddings.py
scripts/16_extract_all_real_embeddings.py
scripts/17_run_real_embedding_ablation.py
tests/test_embedding_failures.py
tests/test_real_embedding_contracts.py
tests/test_audio_real_embedding.py
tests/test_face_real_embedding.py
tests/test_eeg_real_embedding.py
tests/test_wear_real_embedding.py
tests/test_real_pipeline.py
真实多模态完整embedding执行报告.md
```

### 修改文件

```text
configs/encoders.yaml
configs/smoke.yaml
repo-docs/modules/embedding-contract.md
repo-docs/references/commands-and-artifacts.md
repo-docs/walkthroughs/one-real-run.md
repo-docs/change-log.md
多模态嵌入前半部分跑通计划.md
```

### 主要产物

```text
outputs/cache/audio_clips/
outputs/cache/openface/
outputs/cache/face_preprocessed/
outputs/cache/eeg_windows/
outputs/cache/wear_windows/
outputs/embeddings/audio_real_embeddings.npz
outputs/embeddings/face_real_embeddings.npz
outputs/embeddings/face_real_preprocessed_embeddings.npz
outputs/embeddings/eeg_real_embeddings.npz
outputs/embeddings/wear_real_embeddings.npz
outputs/embeddings/all_complete_real_embeddings.npz
outputs/reports/real_embedding_readiness_report.md
outputs/reports/real_embedding_failures.json
outputs/reports/face_quality_audit.md
outputs/reports/face_preprocessing_decision.json
outputs/reports/real_embedding_quality_summary.json
outputs/reports/real_embedding_ablation_table.md
```

---

## 3. 执行阶段

### 阶段 11：真实 encoder 接入前的契约和失败清单

**目标：** 在任何深度模型接入前，先固化真实 encoder 的输入输出契约、失败清单 schema、质量摘要和缓存目录规则。

**Files:**

- Create: `src/daily_multimodal/embeddings/contracts.py`
- Create: `src/daily_multimodal/embeddings/failures.py`
- Create: `tests/test_real_embedding_contracts.py`
- Create: `tests/test_embedding_failures.py`
- Modify: `configs/encoders.yaml`
- Create: `真实多模态完整embedding执行报告.md`

**Tasks:**

- [x] 定义 `RealEmbeddingResult` dataclass，字段包括 `sample_id`、`event_id`、`subject_id`、`modality`、`embedding`、`mask_value`、`quality_flags`、`encoder_version`、`source_paths`。
- [x] 定义 `EmbeddingFailure` dataclass，字段使用第 1 节 schema。
- [x] 添加 `validate_embedding_shape(name, array, expected_dim=256)`，检查 shape、dtype、NaN。
- [x] 添加 `write_failure_list(failures, path)`，空列表必须写成 `[]`。
- [x] 在 `configs/encoders.yaml` 增加 profile：

```yaml
profiles:
  basic:
    eeg: basic_smoke_metadata_v1
    wear: basic_wear_statistics_v1
    face: basic_smoke_metadata_v1
    audio: basic_smoke_metadata_v1
  real_v1:
    eeg: eeg_real_frozen_v1
    wear: wear_sequence_v1
    face: openface_temporal_v1
    audio: wavlm_frozen_v1
```

**Verification:**

```powershell
python -m pytest tests/test_real_embedding_contracts.py tests/test_embedding_failures.py -q
python -m compileall -q src scripts tests
```

**Acceptance:**

- 空失败清单写出 `[]`。
- 任何 embedding shape 非 `(256,)` 或 `(N, 256)` 时测试失败。
- 失败记录包含 `modality`、`encoder_profile`、`stage`、`error_type`、`source_path`。

### 阶段 12：真实缓存和切片层

**目标：** 先把真实数据切片和缓存跑通，避免深度模型失败时分不清是切片问题还是模型问题。

**Files:**

- Create: `src/daily_multimodal/embeddings/cache.py`
- Create: `scripts/11_prepare_real_embedding_cache.py`
- Create: `tests/test_real_pipeline.py`
- Modify: `repo-docs/references/commands-and-artifacts.md`

**Tasks:**

- [x] 为每个窗口生成稳定 cache key：`{sample_id}/{modality}/{encoder_profile}`。
- [x] Audio cache：从 `video_candidates` 中抽取窗口音频，保存 mono 16 kHz wav。
- [x] Face cache：为 MP4 生成 OpenFace CSV 目标路径，不在本阶段强制跑 OpenFace。
- [x] EEG cache：记录 BDF path、window start/end、采样率、目标重采样参数。
- [x] Wear cache：记录 PPG/GSR/ACC CSV path 和窗口时间范围。
- [x] 输出 `outputs/reports/real_embedding_readiness_report.md`，包含每个模态可切片数量、缺失数量、失败清单路径。

**Server small-run command:**

```bash
ssh ncc_serve_4090 "cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding && python - <<'PY'
import json
src = 'outputs/window_index/window_index_with_video_audio.jsonl'
out = 'outputs/window_index/real_cache_complete_10.jsonl'
selected = []
with open(src, encoding='utf-8-sig') as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        complete = all(bool(row.get(k)) for k in ['has_eeg', 'has_ppg', 'has_gsr', 'has_acc', 'has_face', 'has_audio'])
        if complete and row.get('video_candidates'):
            selected.append(row)
            if len(selected) == 10:
                break
with open(out, 'w', encoding='utf-8') as f:
    for row in selected:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
print('filtered_count=' + str(len(selected)))
print('filtered_path=' + out)
PY
PYTHONPATH=src python scripts/11_prepare_real_embedding_cache.py --window-index outputs/window_index/real_cache_complete_10.jsonl --max-windows 10 --cache-root outputs/cache/real_stage12_complete_10 --out-report outputs/reports/real_embedding_readiness_complete_10.md --failures-out outputs/reports/real_embedding_failures_complete_10.json"
```

注意：不要直接用默认 `outputs/window_index/window_index.jsonl` 的前 10 条判断阶段 12 是否成功；当前服务器该文件前 10 条是 `sub-01` 的不完整窗口，会主要验证 `EmbeddingFailure` 记录，而不是四模态真实缓存成功路径。

**Acceptance:**

- 10 窗口 cache 准备完成或每个失败窗口都有明确 `EmbeddingFailure`。
- 不写入原始数据目录。
- readiness report 明确列出 EEG/Wear/Face/Audio 的 ready count。

### 阶段 13：Audio 真实 embedding，WavLM/wav2vec2 frozen

**目标：** 第一个真实深度模型优先接 Audio，因为 MP4 音轨切片相对独立，便于验证深度模型接入链路。

**Files:**

- Create: `src/daily_multimodal/embeddings/audio_real.py`
- Create: `scripts/12_extract_audio_embeddings.py`
- Create: `tests/test_audio_real_embedding.py`
- Modify: `configs/encoders.yaml`

**Model choice:**

```text
Primary: WavLM frozen mean pooling + projection to 256
Fallback: wav2vec2 frozen mean pooling + projection to 256
No full fine-tuning in this phase
```

**Tasks:**

- [x] 检查服务器是否有 `torch`、`torchaudio`、`transformers`。默认 Python 缺少依赖，但 conda 环境 `lzs` 已确认具备 `torch`、`torchaudio`、`transformers` 和 `huggingface_hub`。
- [x] 如果 checkpoint 不存在，记录 `checkpoint_missing`，不静默 fallback。
- [x] 从 audio cache 读取 16 kHz mono wav。
- [x] WavLM/wav2vec2 输出 frame embedding 后做 mean pooling。生产路径已实现；服务器已用 `wav2vec2_frozen_v1` 真 checkpoint 跑通 10 窗口和 `sub-12` 单被试。
- [x] 用固定随机种子或保存 projection 权重，将输出投影到 256 维。
- [x] 写 `audio_real_embeddings.npz`，包含 `audio_emb`、`sample_id`、`modality_mask`、`quality_flags`。
- [ ] 接入 `scripts/17_run_real_embedding_ablation.py`，比较：
  - baseline full；
  - 当前 stage 10 fusion；
  - only audio replaced。

**Small-run command:**

```bash
ssh ncc_serve_4090 "cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding && source /home/lzs/miniconda3/etc/profile.d/conda.sh && conda activate lzs && PYTHONPATH=src python scripts/12_extract_audio_embeddings.py --window-index outputs/window_index/real_cache_complete_10.jsonl --max-windows 10 --cache-root outputs/cache/real_stage12_wav2vec2_10 --encoder-profile wav2vec2_frozen_v1 --checkpoint outputs/checkpoints/wav2vec2-base-960h --device cuda --out outputs/embeddings/audio_real_wav2vec2_10_embeddings.npz --failures-out outputs/reports/audio_real_wav2vec2_10_failures.json --summary-out outputs/reports/audio_real_wav2vec2_10_quality_summary.json"
```

注意：`microsoft/wavlm-base-plus` 已下载到 `outputs/checkpoints/wavlm-base-plus`，但该仓库当前可用权重为 `.bin`，在服务器 `torch 2.5.1` + 新版 `transformers` 下会触发 torch>=2.6 的安全限制；阶段 13 先采用带 `model.safetensors` 的 `facebook/wav2vec2-base-960h` 作为 frozen fallback。缺少依赖或 checkpoint 时，该命令仍应写出 `dependency_missing` 或 `checkpoint_missing`，不能生成伪 real embedding。

**Single-subject command:**

```bash
ssh ncc_serve_4090 "cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding && source /home/lzs/miniconda3/etc/profile.d/conda.sh && conda activate lzs && PYTHONPATH=src python scripts/12_extract_audio_embeddings.py --window-index outputs/window_index/audio_real_wav2vec2_sub-12.jsonl --cache-root outputs/cache/real_stage12_wav2vec2_sub-12 --encoder-profile wav2vec2_frozen_v1 --checkpoint outputs/checkpoints/wav2vec2-base-960h --device cuda --out outputs/embeddings/audio_real_wav2vec2_sub-12_embeddings.npz --failures-out outputs/reports/audio_real_wav2vec2_sub-12_failures.json --summary-out outputs/reports/audio_real_wav2vec2_sub-12_quality_summary.json"
```

**Full command:**

```bash
ssh ncc_serve_4090 "cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding && python scripts/12_extract_audio_embeddings.py --window-index outputs/window_index/window_index.jsonl --require-all-modalities --encoder-profile wavlm_frozen_v1 --out outputs/embeddings/audio_real_embeddings.npz --failures-out outputs/reports/audio_real_failures.json"
```

**Acceptance:**

- 10 窗口先通过，`audio_emb.shape == (10, 256)`。已通过：`success_count=10`、`failure_count=0`、`nan_count=0`。
- 全量前先跑单被试。已通过：`sub-12` 共 25 条完整窗口，Stage 12 cache 四模态 `ready=25, missing=0`，Stage 13 `audio_emb.shape == (25, 256)`、失败清单 `[]`、NaN 数量 0。
- 全量 audio 成功率、失败类型、平均音频时长写入质量摘要。
- NaN 数量为 0。
- 若失败率超过 5%，不进入下一个真实模态，先排查音频切片和 checkpoint。

### 阶段 14：Face 真实 embedding，OpenFace CSV + temporal encoder

**目标：** 用真实面部特征替代 face metadata smoke。

**Files:**

- Create: `src/daily_multimodal/embeddings/face_real.py`
- Create: `scripts/13_audit_face_quality.py`
- Create: `scripts/13_extract_face_embeddings.py`
- Create: `tests/test_face_real_embedding.py`
- Modify: `configs/encoders.yaml`

**Model choice:**

```text
First real version: OpenFace CSV statistics + projection to 256
Second version: OpenFace frame-level GRU/temporal Transformer + projection to 256
No VideoMAE until OpenFace path stable
```

**Video quality strategy:**

第一轮必须保留原始视频分支，不做人脸预处理，直接用 OpenFace 或等价工具抽取特征。这样可以先判断“真实但脏”的视频信号是否已经足够，而不是提前引入裁剪、增强、跟踪等处理带来的偏差。原始分支产物命名为 `face_raw_openface_stats_v1`。

同时必须做视频质量审计，但审计只记录指标，不改输入帧。质量审计至少输出：

- `face_detection_success_rate`：窗口内成功检测到主脸的帧比例。
- `mean_openface_confidence` 和 `low_confidence_ratio`：低置信阈值使用 `confidence < 0.80`。
- `pose_bad_ratio`：`abs(yaw) > 35`、`abs(pitch) > 25` 或 `abs(roll) > 25` 的帧比例。
- `dark_frame_ratio`：人脸区域亮度均值低于全量窗口第 10 百分位，或灰度均值 `< 40/255` 的帧比例。
- `blur_frame_ratio`：人脸区域 Laplacian variance 低于全量窗口第 10 百分位的帧比例。
- `multi_face_ratio`：同一帧检测到 2 张及以上人脸的比例。
- `main_face_ambiguity_ratio`：最大脸面积和第二大脸面积之比小于 `1.5`，且二者中心距离都接近画面中心时的比例。

**Raw-first decision rule:**

完成原始分支后，先用质量门槛和下游指标共同判断是否进入预处理分支。

质量门槛：

- 若全量窗口中 `face_detection_success_rate >= 0.80` 的窗口比例达到 80%；
- 且 `mean_openface_confidence >= 0.85` 的窗口比例达到 80%；
- 且 `multi_face_ratio > 0.10` 的窗口比例低于 5%；
- 且 `pose_bad_ratio > 0.30`、`dark_frame_ratio > 0.30`、`blur_frame_ratio > 0.30` 任一高风险窗口比例均低于 25%；
- 则原始分支通过质量门槛，不默认启用预处理。

下游指标门槛：

- 固定 subject split 和相同训练配置，至少跑 5 个 seed：`0,1,2,3,4`。
- 当前 `alert` 回归任务以 test RMSE 为主指标，MAE 和 Pearson 为辅指标；如果后续换成分类任务，则以 macro F1 为主、accuracy 为辅。
- 对比 `face_raw_openface_stats_v1`、`face_smoke_basic`、`all_real_without_face`、`all_real_with_raw_face`。
- 若 `all_real_with_raw_face` 相比 `all_real_without_face` 的 median test RMSE 没有下降至少 2%，或 Pearson 下降超过 `0.02`，并且高风险质量窗口上的 RMSE 明显更差，则判定原始 face 分支效果不足。
- 使用 subject-stratified paired bootstrap，按 test subject 分层重采样窗口 1000 次，报告 RMSE delta 的 95% CI；若 CI 大部分跨 0，则不能声称 face 原始分支有效，只能记为“不确定”。

进入预处理分支的触发条件：

- 质量门槛失败；或
- 原始 face 分支让全模态 median test RMSE 变差超过 2%；或
- 去掉 face 后 `all_real_without_face` 反而稳定优于 `all_real_with_raw_face`；或
- 多人脸/暗光/模糊/大角度窗口和高误差窗口有稳定相关性，例如高风险窗口 RMSE 比低风险窗口高 10% 以上。

**Preprocessing branch:**

预处理只作为第二分支，产物命名为 `face_preprocessed_openface_stats_v1`，不得覆盖原始分支缓存。

具体方法按从低风险到高风险逐步启用：

1. 主脸选择：使用 OpenFace、RetinaFace 或 MTCNN 检测所有脸。默认选择面积最大且最接近画面中心、并在时间上连续的 face track；相邻帧用 IoU、中心距离和面积变化保持同一 track。多人脸歧义高的窗口不强行选择，记录 `main_face_ambiguity` 并可置 `mask_value=0`。
2. 人脸裁剪和对齐：基于眼角/鼻尖/嘴角 landmarks 做 similarity transform，对齐后用 `1.25x` 人脸框 margin 裁剪，resize 到固定尺寸，例如 `224x224`。保留原始帧路径和 crop 参数，保证可复现。
3. 亮度增强：只在人脸 crop 的 luminance 通道上做 CLAHE 或有界 gamma correction，gamma 限制在 `[0.7, 1.5]`，禁止使用生成式修复或会改变表情语义的增强。
4. 模糊和大角度处理：默认不做去模糊。轻微模糊只记录 quality flag；严重模糊或 `abs(yaw) > 45`、`abs(pitch) > 35` 的窗口置低质量或 mask，不把坏窗口伪装成正常窗口。
5. 短缺帧插值：OpenFace 特征短缺失小于 `0.5s` 时可线性插值 AU/gaze/pose；连续缺失超过 `0.5s` 的片段记为无效，不插值。
6. 时序平滑：对 AU/gaze/pose 做窗口内 median filter 或 Savitzky-Golay 平滑，只用于减少检测抖动；平滑前后都要保留质量摘要。

预处理分支只有在同时满足以下条件时才替代默认 raw face：

- `face_detection_success_rate` 或有效窗口比例相对提升至少 20%；
- `multi_face_ratio` 或 `main_face_ambiguity_ratio` 明显下降；
- 5 个 seed 下 `all_real_with_preprocessed_face` 的 median test RMSE 比 `all_real_with_raw_face` 下降至少 2%，且 MAE 不变差超过 1%、Pearson 不下降超过 `0.02`；
- 至少 4/5 个 seed 的主指标方向一致；
- bootstrap 95% CI 不显示稳定负向影响。

如果预处理只改善质量指标但没有稳定改善下游指标，则保留预处理缓存和报告，但默认训练仍使用 raw face 分支。

**Tasks:**

- [x] 检查 OpenFace 可执行文件路径，缺失时记录 `dependency_missing`。服务器 `FeatureExtraction/OpenFaceOffline` 缺失；默认路径会写 `dependency_missing`，显式 `--allow-opencv-fallback` 才启用 raw OpenCV fallback。
- [x] 新增 `scripts/13_audit_face_quality.py`，先对原始 MP4 做质量审计，输出 `outputs/reports/face_quality_audit.md` 和 JSON 摘要。
- [x] 先对 1 个 MP4 跑 OpenFace，确认 CSV 字段包含 AU、gaze、pose、confidence、success。实际服务器缺 OpenFace，已改为等价 dirty raw fallback CSV：`success/confidence/face_count/face_area_ratio/gray_mean/laplacian_var/pose/gaze`，不做预处理。
- [x] 对每个窗口从 OpenFace/等价 CSV 切片，计算均值、标准差、成功率、低 confidence 比例。
- [x] 输出 `face_emb [256]` 和 `face_quality`。
- [x] 统计 `face_missing_ratio`，超过阈值时 `mask_value=0` 并记录 `quality_threshold_failed`。
- [x] 先完整跑通 `face_raw_openface_stats_v1`；只有触发 raw-first decision rule 时，才实现并运行 `face_preprocessed_openface_stats_v1`。当前 raw 质量门槛未通过，但下游 5-seed/bootstrap 尚未执行，因此不把预处理分支设为默认。
- [x] 输出 `outputs/reports/face_preprocessing_decision.json`，记录是否启用预处理、触发条件、质量指标变化、下游指标变化和最终默认分支。

**Small-run command:**

```bash
ssh ncc_serve_4090 "cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding && source /home/lzs/miniconda3/etc/profile.d/conda.sh && conda activate lzs && PYTHONPATH=src python scripts/13_extract_face_embeddings.py --window-index outputs/window_index/real_cache_complete_10.jsonl --cache-root outputs/cache/real_stage12_face_raw_10 --encoder-profile face_raw_openface_stats_v1 --allow-opencv-fallback --min-success-rate 0.10 --out outputs/embeddings/face_raw_openface_10_embeddings.npz --failures-out outputs/reports/face_raw_openface_10_failures.json --summary-out outputs/reports/face_raw_openface_10_quality_summary.json --decision-out outputs/reports/face_preprocessing_decision_10.json"
```

**Acceptance:**

- OpenFace/等价 CSV 缓存可复用，重复运行不重复处理同一个窗口 CSV。已生成 10 窗口和 `sub-12` 单被试 raw CSV cache。
- `face_emb.shape == (N, 256)`。已通过：10 窗口 `(10, 256)`，`sub-12` 单被试 `(25, 256)`。
- 每个低质量窗口都有 quality flag，不把坏脸窗口当作正常信号。已通过：10 窗口中 6 个低质量窗口写 `quality_threshold_failed` 且 face mask 为 0；`sub-12` 单被试 25 个窗口均达到当前 dirty fallback 的 `min_success_rate=0.10`。
- 原始 face 分支和预处理 face 分支必须使用不同 encoder profile 和不同缓存路径。已在配置中区分 `face_raw_openface_stats_v1` 与 `face_preprocessed_openface_stats_v1`。
- 是否启用预处理必须由质量门槛、5 seed 下游指标和 bootstrap delta 共同决定，不能只看单次训练结果。当前 `face_preprocessing_decision_10.json` 和 `face_preprocessing_decision_sub-12.json` 均保留默认 raw 分支，记录 raw 质量门槛未通过与下游门槛未运行；不默认启用预处理。

### 阶段 15：EEG 真实 embedding，MNE 预处理 + frozen EEG encoder

**目标：** 用真实 EEG 窗口替代 EEG metadata smoke。

**Files:**

- Create: `src/daily_multimodal/embeddings/eeg_real.py`
- Create: `scripts/14_extract_eeg_embeddings.py`
- Create: `tests/test_eeg_real_embedding.py`
- Modify: `configs/encoders.yaml`

**Model choice:**

```text
Fallback real baseline: MNE window -> bandpower/statistics -> projection to 256
Primary deep version: LaBraM or EEGPT frozen encoder + projection head to 256
No full fine-tuning in this phase
```

**Tasks:**

- [x] MNE 读取 BDF，只裁剪窗口附近数据，不整段加载后重复处理。
- [x] 统一重采样到 250 Hz。
- [x] notch 50 Hz，bandpass 1-45 Hz。
- [x] 输出窗口形状 `[channels, 2500]`，不符合时记录 `shape_mismatch`。
- [x] 先实现 bandpower/statistics real baseline，确认真实 EEG 数据读取链路。
- [x] 再接 LaBraM/EEGPT frozen encoder，checkpoint 缺失时记录 `checkpoint_missing`。已接入 Braindecode EEGPT `braindecode/eegpt-pretrained`，checkpoint 位于服务器 `outputs/checkpoints/eegpt-pretrained`。
- [x] projection 到 256 维，写 `eeg_real_embeddings.npz`。

**Small-run command:**

```bash
ssh ncc_serve_4090 "cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding && source /home/lzs/miniconda3/etc/profile.d/conda.sh && conda activate lzs && PYTHONPATH=src python scripts/14_extract_eeg_embeddings.py --window-index outputs/window_index/real_cache_complete_10.jsonl --cache-root outputs/cache/real_stage12_wav2vec2_10 --encoder-profile eeg_bandpower_v1 --out outputs/embeddings/eeg_real_bandpower_10_embeddings.npz --failures-out outputs/reports/eeg_real_bandpower_10_failures.json --summary-out outputs/reports/eeg_real_bandpower_10_quality_summary.json"
```

**Deep checkpoint command:**

```bash
ssh ncc_serve_4090 "cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding && source /home/lzs/miniconda3/etc/profile.d/conda.sh && conda activate lzs && PYTHONPATH=src python scripts/14_extract_eeg_embeddings.py --window-index outputs/window_index/real_cache_complete_10.jsonl --cache-root outputs/cache/real_stage12_eegpt_10 --encoder-profile eeg_deep_frozen_v1 --checkpoint outputs/checkpoints/eegpt-pretrained --device cpu --out outputs/embeddings/eeg_real_eegpt_10_embeddings.npz --failures-out outputs/reports/eeg_real_eegpt_10_failures.json --summary-out outputs/reports/eeg_real_eegpt_10_quality_summary.json"
```

**Acceptance:**

- 10 窗口 EEG 读取和重采样通过。已通过：`eeg_emb.shape == (10, 256)`、`success_count=10`、`failure_count=0`、`nan_count=0`、平均 64 通道。
- 每个成功窗口 shape 为 `[channels, 2500]` 或报告中记录原始 channel count 和有效 channel count。已记录 `mean_channel_count=64.0` 和 `target_window_samples=2500`。
- 全量前必须先跑单被试，例如 `sub-10`。已用 `sub-12` 25 条完整窗口通过：`eeg_emb.shape == (25, 256)`、失败清单 `[]`、NaN 数量 0。
- EEG deep encoder 失败时可回退 `eeg_bandpower_v1`。已验证 deep 路径：10 窗口 `eeg_emb.shape == (10, 256)`，`sub-12` 单被试 `eeg_emb.shape == (25, 256)`，失败清单均为 `[]`，NaN 数量 0；CUDA 单被试因显存不足会记录 `oom`，当前验收使用 CPU。
- 全量 EEGPT CPU 路径已完成：`outputs/embeddings/eeg_real_eegpt_full_embeddings.npz`，`success_count=738`、`failure_count=43`、`nan_count=0`；失败为可记录样本级失败，不阻塞后续打包，后续通过 `modality_mask` 回退。

### 阶段 16：Wear 真实 sequence embedding

**目标：** 在当前 wear 统计特征可用的基础上，加入 PPG/GSR/ACC raw sequence encoder。

**Files:**

- Create: `src/daily_multimodal/embeddings/wear_real.py`
- Create: `scripts/15_extract_wear_embeddings.py`
- Create: `tests/test_wear_real_embedding.py`
- Modify: `configs/encoders.yaml`

**Model choice:**

```text
Fallback: current PPG/GSR/ACC statistics + projection
Primary: resampled raw sequence + TCN/PatchTST/TS2Vec style frozen or lightweight trained encoder
```

**Tasks:**

- [x] 从 CSV 按绝对时间切 PPG/GSR/ACC。
- [x] PPG 目标采样率 64 Hz，GSR/ACC 目标采样率 32 Hz。
- [x] 对缺点、重复时间戳、非单调时间戳写 quality flags。
- [x] 输出 raw sequence cache 和统计特征 cache。
- [x] 加入 `wear_sequence_v1` lightweight deterministic sequence encoder，输出 256 维。
- [x] 和当前 `basic_wear_statistics_v1` 对照，若 sequence encoder 不提升则回退统计特征。阶段 18 全量 ablation 中 `wear_real_only_replaced` 为 rollback。

**Acceptance:**

- raw sequence shape 稳定。
- 统计特征和 sequence embedding 都可写入质量报告。
- motion intensity、stationary ratio、有效采样率进入 `wear_quality`。
- 已完成 10 窗口和 `sub-12` 单被试验收：10 窗口 `wear_emb.shape == (10, 256)`，单被试 `wear_emb.shape == (25, 256)`，失败清单均为 `[]`，NaN 数量 0；raw sequence cache shape 为 PPG `[640, 1]`、GSR `[320, 1]`、ACC `[320, 3]`。
- 全量 Wear 已完成：6 个 source-group chunk 均 `exit=0`，合并为 `outputs/embeddings/wear_real_sequence_full_embeddings.npz`，`embedded_count=781`、`success_count=781`、`failure_count=0`、`nan_count=0`；旧的长跑 full 进程已停止，旧产物备份为 `outputs/embeddings/wear_real_sequence_full_embeddings.stale_from_killed_processes.npz`。

### 阶段 17：全量真实多模态 embedding 打包

**目标：** 将真实 EEG、Wear、Face、Audio embedding 合并成完整 `.npz`，保持阶段 8/9/10 训练入口兼容。

**Files:**

- Create: `src/daily_multimodal/embeddings/real_pipeline.py`
- Create: `scripts/16_extract_all_real_embeddings.py`
- Create: `tests/test_real_pipeline.py`

**Tasks:**

- [x] 读取单模态真实 embedding 产物。
- [x] 以 window index 为主表按 `sample_id` 对齐，保持样本顺序稳定。
- [x] 对缺失或 mask=0 的模态写零向量和 `modality_mask=0`。
- [x] 合并 `quality_flags`、`encoder_versions`、`source_paths`。
- [x] 输出 `outputs/embeddings/all_complete_real_embeddings.npz`；单被试验收产物为 `outputs/embeddings/all_complete_real_sub-12_embeddings.npz`。
- [x] 输出 `outputs/reports/all_complete_real_embedding_report.json`；单被试验收产物为 `outputs/reports/all_complete_real_sub-12_embedding_report.json`。
- [x] 输出 `outputs/reports/all_complete_real_embedding_failures.json`；单被试验收产物为 `outputs/reports/all_complete_real_sub-12_embedding_failures.json`。

**Acceptance:**

- `eeg_emb/wear_emb/face_emb/audio_emb` 均为 `(N, 256)`。
- `modality_mask` 为 `(N, 4)`。
- `sample_id` 与原 window index 可追溯。
- 报告中列出每个模态的成功数、失败数、mask 分布。
- 已完成服务器 `sub-12` 单被试打包验收：`N=25`，四个 embedding 均为 `(25, 256)`，`modality_mask.shape == (25, 4)`，mask sum 为 `[25, 25, 25, 25]`，失败清单 `[]`，NaN 数量 0。
- 已完成服务器全量打包验收：`outputs/embeddings/all_complete_real_embeddings.npz` 共 781 行，四个 embedding 均为 `(781, 256)`，NaN 数量均为 0，`modality_mask` sum 为 `[738, 781, 657, 781]`；打包报告记录 `failure_count=43`，来自 EEG 缺失行，Face 原始质量 mask 为 124。

### 阶段 18：真实 embedding 训练和 ablation 对照

**目标：** 验证真实多模态 embedding 是否优于 smoke/basic 和阶段 10 融合升级。

**Files:**

- Create: `scripts/17_run_real_embedding_ablation.py`
- Create: `src/daily_multimodal/training/real_embedding_ablation.py`
- Create: `tests/test_real_embedding_ablation.py`

**Comparisons:**

```text
baseline_reference full concat MLP
stage10 modality_token_attention on smoke/basic embeddings
audio_real_only_replaced
face_real_only_replaced
face_raw_openface_stats_v1
face_preprocessed_openface_stats_v1
eeg_real_only_replaced
wear_real_only_replaced
all_real_concat_mlp
all_real_modality_token_attention
all_real_without_face
all_real_with_raw_face
all_real_with_preprocessed_face
```

**Command:**

```bash
ssh ncc_serve_4090 "cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding && python scripts/17_run_real_embedding_ablation.py --basic-embeddings outputs/embeddings/all_complete_basic_real_aligned_embeddings.npz --real-embeddings outputs/embeddings/all_complete_real_embeddings.npz --baseline outputs/reports/baseline_reference_metrics.json --stage10-metrics outputs/reports/modality_token_fusion_metrics.json --target-label alert --out-table outputs/reports/real_embedding_ablation_table.md --metrics-out outputs/reports/real_embedding_ablation_metrics.json --failures-out outputs/reports/real_embedding_ablation_failures.json"
```

**Acceptance:**

- 仍使用 subject split：train `sub-02` 到 `sub-10`，val `sub-11` 到 `sub-12`，test `sub-13` 到 `sub-15`。
- 输出 MAE/RMSE/Pearson。
- Face raw/preprocessed 对照必须跑 5 个 seed，并输出 median、mean、std、best、worst。
- 对 Face 对照输出 subject-stratified paired bootstrap 95% CI，重采样次数为 1000。
- 每个实验输出 `accepted` 或 `rollback`。
- 若 all-real 不提升，需要能从单模态 ablation 看出问题模态。

**当前进展：**

- [x] 新增阶段 18 ablation 入口，支持 baseline reference、stage10 reference、单模态 real-only replaced、all-real concat、all-real modality-token attention、without-face 和 face raw/preprocessed 名义对照。
- [x] 输出 Markdown table、metrics JSON 和 failures JSON。
- [x] 本地单元测试覆盖完整 split 成功路径、单被试 split 不完整失败路径和 CLI 入口。
- [x] 服务器 `sub-12` smoke ablation 已验证：脚本能读取 basic/real 对齐产物并写出 `subject_split_incomplete` failures；由于只有 `sub-12`，按阶段 18 规则缺 train/test split，不产生训练指标。
- [x] 生成对齐 basic 对照产物 `outputs/embeddings/all_complete_basic_real_aligned_embeddings.npz`，按 real 的 781 个 `sample_id` 过滤并重排，解决原始 basic 995 行与 real 781 行不一致问题。
- [x] 全量 all-real 打包后运行完整 subject split ablation：`experiment_count=13`、`failure_count=0`，split 为 train 554、val 85、test 142。
- [x] 在全量数据上运行 Face raw/preprocessed 5 seed 和 bootstrap CI，并据此给出最终 accepted/rollback 结论：`face_real_only_replaced` accepted，`face_raw_openface_stats_v1`、`face_preprocessed_openface_stats_v1`、`audio_real_only_replaced`、`eeg_real_only_replaced`、`wear_real_only_replaced`、`all_real_concat_mlp`、`all_real_modality_token_attention`、`all_real_without_face`、`all_real_with_raw_face`、`all_real_with_preprocessed_face` 均 rollback；face seed RMSE median 0.9028、mean 0.9043、std 0.0341、best 0.8569、worst 0.9456，bootstrap 95% CI delta RMSE 为 `[-0.0011, 0.0586]`。

---

## 4. 强制验证顺序

每个真实模态必须按以下顺序扩大：

```text
1 条窗口
10 条窗口
1 个被试，例如 sub-10
995 条完整候选
训练/ablation 对照
```

每一级必须检查：

- embedding shape；
- NaN 数；
- mask 分布；
- quality flags；
- failure list；
- runtime；
- 输出是否写在 `/mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding`。

---

## 5. 完成定义

本计划完成时，应该得到：

```text
outputs/embeddings/all_complete_real_embeddings.npz
outputs/reports/all_complete_real_embedding_report.json
outputs/reports/all_complete_real_embedding_failures.json
outputs/reports/real_embedding_quality_summary.json
outputs/reports/real_embedding_ablation_table.md
outputs/reports/real_embedding_ablation_failures.json
真实多模态完整embedding执行报告.md
```

最终验收：

- 四个模态都不再是 metadata smoke embedding。
- 每个模态都有真实数据读取或真实特征抽取过程。
- 每个 embedding 保持 256 维统一契约。
- 任一真实模态失败时，可以回退 basic 或上一版 accepted 结果。
- 真实全模态结果和阶段 9/10 基准在同一 subject split 下可比较。

## 6. v2 full 与 fatigue 下游验收记录（2026-07-01）

- [x] 服务器 full cache `outputs/cache/real_stage12_v2_full` 已完成，readiness 为四模态 781/781 ready。
- [x] EEG full v2 已完成：`outputs/embeddings/eeg_real_eegpt_full_v2_embeddings.npz`，shape `(738, 256)`，NaN 0，失败 43 个，分类为 `eeg_window_before_recording=29`、`eeg_window_after_recording=14`。
- [x] Wear physio v2 full 已完成：`outputs/embeddings/wear_physio_features_v2_full_embeddings.npz`，成功 781，失败 0，NaN 0。
- [x] Audio v2 full 采用 openSMILE eGeMAPS：`outputs/embeddings/audio_opensmile_egemaps_full_embeddings.npz`，成功 781，失败 0，NaN 0；emotion2vec plus 已完成 10-window smoke，full 未作为本轮主线 profile。
- [x] Face true OpenFace full 已完成并修复 HAAR/MTCNN 与 codec 两类问题：`outputs/embeddings/face_openface_real_full_embeddings.npz`，true OpenFace wrapper 对 10 秒窗口 clip 运行，成功 207，质量 mask 501，extraction failed 73，NaN 0。
- [x] all-real v2 已完成：`outputs/embeddings/all_complete_real_v2_embeddings.npz`，selected windows 781，四模态 shape 均 `(781, 256)`，NaN 0，mask sum `[738, 781, 207, 781]`。
- [x] 下游验证已从 `alert` 切换为 `fatigue`，fair ablation 与 subject CV 均输出 RMSE 和 Pearson r。
- [x] 四模态 subject-CV 修复后仍因 `sub-13` 无四模态完整样本导致空 fold；最终稳健验证使用 `--modalities eeg,wear,audio`。fatigue 四模态 fair real RMSE `1.3647`、r `-0.1437`；fatigue EWA fair real RMSE `1.0160`、r `0.1205`；fatigue EWA LOSO subject-CV `fold_count=14`、`subject_leakage=False`、RMSE mean `0.9697`、r mean `0.0636`。
