# 真实多模态完整 Embedding 执行报告

## 阶段 11：真实 encoder 契约和失败清单

状态：完成。

- 新增 `RealEmbeddingResult` 和 `validate_embedding_shape`，用于约束真实单模态 embedding 必须是 `(256,)` 或 `(N, 256)` 的浮点数组，且不能包含 NaN 或无限值。
- 新增 `EmbeddingFailure` 和 `write_failure_list`，失败记录落盘时空列表写为 `[]`，非空记录包含 `modality`、`encoder_profile`、`stage`、`error_type` 和 `source_path`。
- `configs/encoders.yaml` 保留旧 smoke 配置，并新增 `basic` 与 `real_v1` profile。

验证：

```powershell
python -m pytest tests/test_real_embedding_contracts.py tests/test_embedding_failures.py -q
python -m compileall -q src scripts tests
```

## 阶段 12：真实缓存和切片层

状态：完成。

- 新增 `build_cache_key(sample_id, modality, encoder_profile)`，稳定生成 `{sample_id}/{modality}/{encoder_profile}`，并拒绝路径穿越片段。
- 新增 `prepare_real_embedding_cache`，为 audio、face、EEG、wear 四个模态生成 cache 记录。
- Audio 默认调用 `ffmpeg` 从 `video_candidates` 的精确窗口切出 mono 16 kHz wav；缺少 `ffmpeg`、源文件或切片失败时写入 `EmbeddingFailure`。
- Face 在本阶段只写 OpenFace CSV 目标路径，不强制运行 OpenFace。
- EEG 写 BDF 路径、窗口时间、原采样率、目标 250 Hz 和 2500 samples。
- Wear 写 PPG/GSR/ACC 路径、窗口时间和目标采样率。
- 新增 `scripts/11_prepare_real_embedding_cache.py`，输出 readiness report 和失败清单。
- `load_window_index` 兼容带 UTF-8 BOM 的 JSONL，避免 Windows PowerShell 生成的临时窗口索引导致脚本入口失败。

验证：

```powershell
python -m pytest tests/test_real_pipeline.py -q
python -m pytest tests/test_real_embedding_contracts.py tests/test_embedding_failures.py tests/test_real_pipeline.py -q
python -m pytest tests/test_event_window.py -q
python -m compileall -q src scripts tests
```

服务器验证：

```bash
cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding
PYTHONPATH=src python scripts/11_prepare_real_embedding_cache.py \
  --window-index outputs/window_index/real_cache_complete_10.jsonl \
  --max-windows 10 \
  --cache-root outputs/cache/real_stage12_complete_10 \
  --out-report outputs/reports/real_embedding_readiness_complete_10.md \
  --failures-out outputs/reports/real_embedding_failures_complete_10.json
```

结果：`outputs/window_index/window_index_with_video_audio.jsonl` 中筛出的 10 条完整且带 `video_candidates` 的窗口全部通过缓存准备；EEG、Wear、Face、Audio 均为 `ready=10, missing=0, failures=0`，失败清单为 `[]`。缓存计数为 `audio.wav=10`、`audio.json=10`、`openface_target.json=10`、`eeg window.json=10`、`wear window.json=10`。

## 阶段 13：Audio 真实 embedding，WavLM/wav2vec2 frozen

状态：阶段 13 音频真实 embedding 路径已跑通到单被试；全量和 ablation 留到后续统一执行。

- 新增 `src/daily_multimodal/embeddings/audio_real.py`，从阶段 12 的 `audio_clips` cache 读取 16 kHz mono wav，调用 frozen audio backend 输出 frame embedding，mean pooling 后用固定随机种子投影到 256 维。
- 新增 `scripts/12_extract_audio_embeddings.py`，写出 `audio_real_embeddings.npz`，包含 `audio_emb`、`sample_id`、`event_id`、`subject_id`、`modality_mask`、`quality_flags` 和 `encoder_version`。
- checkpoint 不存在时写入 `checkpoint_missing`，缺少 `torch`、`torchaudio` 或 `transformers` 时写入 `dependency_missing`，不静默回退到 metadata 或随机 embedding。
- 新增 `tests/test_audio_real_embedding.py`，用可注入 fake backend 验证成功路径，用 CLI smoke 验证缺 checkpoint 会写失败清单并返回失败码。
- `configs/encoders.yaml` 增加 `audio_real_profiles`，记录 `wavlm_frozen_v1` 和 `wav2vec2_frozen_v1` 的 256 维、16 kHz、mean pooling、checkpoint required 约束。

本地验证：

```powershell
python -m pytest tests/test_audio_real_embedding.py -q
python -m compileall -q src scripts tests
```

服务器依赖巡检：

```text
默认 Python: torch=False, torchaudio=False, transformers=False
conda env lzs: torch=True, torchaudio=True, transformers=True, huggingface_hub=True
```

缺 checkpoint 验证：

```text
failure_count=10
failure_types={"checkpoint_missing": 10}
audio_emb_shape=(0, 256)
modality_mask_shape=(0, 4)
nan_count=0
```

上述负向验证用于确认缺 checkpoint 或默认环境缺依赖时不会静默生成伪 real embedding；真模型验收已切到具备依赖的 `lzs` 环境，并使用 wav2vec2 safetensors fallback 完成。

真实 checkpoint 处理：

- `microsoft/wavlm-base-plus` 已下载到 `outputs/checkpoints/wavlm-base-plus`，但当前可用权重是 `.bin`，在服务器 `torch 2.5.1` + 新版 `transformers` 下会触发 torch>=2.6 的安全限制，未用于阶段 13 验收。
- `facebook/wav2vec2-base-960h` 已通过 HF mirror 下载到 `outputs/checkpoints/wav2vec2-base-960h`，使用 `model.safetensors` 作为 `wav2vec2_frozen_v1` fallback checkpoint。

服务器真模型 10 窗口验证：

```bash
cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding
source /home/lzs/miniconda3/etc/profile.d/conda.sh
conda activate lzs
PYTHONPATH=src python scripts/12_extract_audio_embeddings.py \
  --window-index outputs/window_index/real_cache_complete_10.jsonl \
  --cache-root outputs/cache/real_stage12_wav2vec2_10 \
  --encoder-profile wav2vec2_frozen_v1 \
  --checkpoint outputs/checkpoints/wav2vec2-base-960h \
  --device cuda \
  --out outputs/embeddings/audio_real_wav2vec2_10_embeddings.npz \
  --failures-out outputs/reports/audio_real_wav2vec2_10_failures.json \
  --summary-out outputs/reports/audio_real_wav2vec2_10_quality_summary.json
```

结果：

```text
audio_emb.shape=(10, 256)
sample_count=10
failure_count=0
nan_count=0
modality_mask[0]=[0, 0, 0, 1]
failures=[]
```

## 阶段 17：全量真实多模态 embedding 打包

状态：阶段17打包入口已完成，并用服务器 `sub-12` 单被试真实产物通过验证。当前实现以 window index 为主表，保留窗口顺序、标签、subject/session 和 source paths；四个单模态 `.npz` 按 `sample_id` 对齐，缺失或 mask=0 的模态写零向量并保持 `modality_mask=0`。

- 新增 `src/daily_multimodal/embeddings/real_pipeline.py`，读取 EEG/Wear/Face/Audio 单模态真实 embedding，输出训练入口兼容的统一 `.npz`。
- 新增 `scripts/16_extract_all_real_embeddings.py`，命令行参数显式接收 `--eeg`、`--wear`、`--face`、`--audio` 和 `--window-index`。
- 输出字段包括 `sample_id`、`event_id`、`subject_id`、`session_id`、`labels`、`eeg_emb`、`wear_emb`、`face_emb`、`audio_emb`、`modality_mask`、`quality_flags`、`encoder_versions`、`source_paths`。
- 输出报告 `all_complete_real_embedding_report.json` 汇总每个模态的 `success_count`、`missing_count`、`masked_count` 和 encoder profile；失败清单记录缺失单模态行的 `source_missing`。

本地验证：

```powershell
python -m pytest tests/test_real_pipeline.py -q
python -m pytest tests -q
python -m compileall -q src scripts tests
```

结果：

```text
2 passed
60 passed
compileall passed
```

服务器验证：

```bash
cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding
source /home/lzs/miniconda3/etc/profile.d/conda.sh
conda activate lzs
PYTHONPATH=src python -m unittest discover -s tests
python -m compileall -q src scripts tests
```

结果：

```text
Ran 60 tests
OK
compileall passed
```

服务器 `sub-12` all-real 打包验证：

```bash
PYTHONPATH=src python scripts/16_extract_all_real_embeddings.py \
  --window-index outputs/window_index/audio_real_wav2vec2_sub-12.jsonl \
  --eeg outputs/embeddings/eeg_real_eegpt_sub-12_embeddings.npz \
  --wear outputs/embeddings/wear_real_sequence_sub-12_embeddings.npz \
  --face outputs/embeddings/face_raw_openface_sub-12_embeddings.npz \
  --audio outputs/embeddings/audio_real_wav2vec2_sub-12_embeddings.npz \
  --out outputs/embeddings/all_complete_real_sub-12_embeddings.npz \
  --report-out outputs/reports/all_complete_real_sub-12_embedding_report.json \
  --failures-out outputs/reports/all_complete_real_sub-12_embedding_failures.json
```

结果：

```text
selected_windows=25
failure_count=0
eeg_success_count=25 eeg_missing_count=0 eeg_masked_count=0
wear_success_count=25 wear_missing_count=0 wear_masked_count=0
face_success_count=25 face_missing_count=0 face_masked_count=0
audio_success_count=25 audio_missing_count=0 audio_masked_count=0
eeg_emb.shape=(25, 256), nan_count=0
wear_emb.shape=(25, 256), nan_count=0
face_emb.shape=(25, 256), nan_count=0
audio_emb.shape=(25, 256), nan_count=0
modality_mask.shape=(25, 4)
modality_mask.sum(axis=0)=[25, 25, 25, 25]
failures=[]
encoder_versions={"eeg": "eeg_deep_frozen_v1", "wear": "wear_sequence_v1", "face": "face_raw_openface_stats_v1", "audio": "wav2vec2_frozen_v1"}
```

剩余项：阶段17的全量 995 窗口打包尚未执行；阶段18 需要基于全量 all-real 产物运行 baseline/stage10/real-only/all-real ablation。

## 阶段 18：真实 embedding 训练和 ablation 对照入口

状态：阶段18 ablation 入口已实现并完成本地测试；服务器已用 `sub-12` 对齐的 basic/real 产物做 smoke 验证。由于 `sub-12` 只有验证集被试，缺 train/test subject split，服务器 smoke 的正确结果是写出 `subject_split_incomplete` failures，不产生训练指标。全量 all-real 打包与最终 ablation 结论仍待后续执行。

- 新增 `src/daily_multimodal/training/real_embedding_ablation.py`，复用阶段9/10的 MLP、subject split、MAE/RMSE/Pearson 指标口径。
- 新增 `scripts/17_run_real_embedding_ablation.py`，读取 `--basic-embeddings`、`--real-embeddings`、baseline reference 和可选 stage10 metrics。
- 支持实验项：baseline reference、stage10 modality-token reference、`audio_real_only_replaced`、`face_real_only_replaced`、`eeg_real_only_replaced`、`wear_real_only_replaced`、`all_real_concat_mlp`、`all_real_modality_token_attention`、`all_real_without_face`、`all_real_with_raw_face`、`all_real_with_preprocessed_face`。
- 输出 `real_embedding_ablation_table.md`、`real_embedding_ablation_metrics.json` 和 `real_embedding_ablation_failures.json`。
- Face seed summary 输出 seed count、median、mean、std、best、worst 和 `bootstrap_ci95_delta_rmse` 字段；全量阶段需要使用 5 seeds 和 1000 bootstrap iterations 生成最终结论。

本地验证：

```powershell
python -m pytest tests/test_real_embedding_ablation.py -q
python -m pytest tests -q
python -m compileall -q src scripts tests
```

结果：

```text
3 passed
63 passed
compileall passed
```

服务器验证：

```bash
cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding
source /home/lzs/miniconda3/etc/profile.d/conda.sh
conda activate lzs
PYTHONPATH=src python -m unittest discover -s tests
python -m compileall -q src scripts tests
```

结果：

```text
Ran 63 tests
OK
compileall passed
```

服务器 `sub-12` smoke ablation：

```bash
PYTHONPATH=src python scripts/17_run_real_embedding_ablation.py \
  --basic-embeddings outputs/embeddings/all_complete_basic_sub-12_aligned_embeddings.npz \
  --real-embeddings outputs/embeddings/all_complete_real_sub-12_embeddings.npz \
  --baseline outputs/reports/baseline_reference_metrics.json \
  --stage10-metrics outputs/reports/modality_token_fusion_metrics.json \
  --target-label alert \
  --out-table outputs/reports/real_embedding_ablation_sub-12_table.md \
  --metrics-out outputs/reports/real_embedding_ablation_sub-12_metrics.json \
  --failures-out outputs/reports/real_embedding_ablation_sub-12_failures.json \
  --epochs 20 \
  --overfit-limit 4 \
  --seeds 5 \
  --bootstrap-iterations 10
```

结果：

```text
experiment_count=0
failure_count=2
failures=[
  {"source": "basic_embeddings", "error_type": "subject_split_incomplete", "missing_splits": ["train", "test"]},
  {"source": "real_embeddings", "error_type": "subject_split_incomplete", "missing_splits": ["train", "test"]}
]
```

解释：这说明阶段18入口和失败语义可用，但单被试不能证明训练效果。下一步需要全量四模态真实 embedding，再按阶段18计划跑完整 subject split、5 seed face 对照和 bootstrap CI。

服务器单被试验证：

```bash
cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding
source /home/lzs/miniconda3/etc/profile.d/conda.sh
conda activate lzs
PYTHONPATH=src python scripts/11_prepare_real_embedding_cache.py \
  --window-index outputs/window_index/audio_real_wav2vec2_sub-12.jsonl \
  --cache-root outputs/cache/real_stage12_wav2vec2_sub-12 \
  --audio-encoder-profile wav2vec2_frozen_v1 \
  --out-report outputs/reports/real_embedding_readiness_wav2vec2_sub-12.md \
  --failures-out outputs/reports/real_embedding_failures_wav2vec2_sub-12.json
PYTHONPATH=src python scripts/12_extract_audio_embeddings.py \
  --window-index outputs/window_index/audio_real_wav2vec2_sub-12.jsonl \
  --cache-root outputs/cache/real_stage12_wav2vec2_sub-12 \
  --encoder-profile wav2vec2_frozen_v1 \
  --checkpoint outputs/checkpoints/wav2vec2-base-960h \
  --device cuda \
  --out outputs/embeddings/audio_real_wav2vec2_sub-12_embeddings.npz \
  --failures-out outputs/reports/audio_real_wav2vec2_sub-12_failures.json \
  --summary-out outputs/reports/audio_real_wav2vec2_sub-12_quality_summary.json
```

结果：

```text
filtered_subject=sub-12
filtered_count=25
eeg_ready_count=25, eeg_missing_count=0
wear_ready_count=25, wear_missing_count=0
face_ready_count=25, face_missing_count=0
audio_ready_count=25, audio_missing_count=0
audio_emb.shape=(25, 256)
sample_count=25
failure_count=0
nan_count=0
failures=[]
```

全量 audio embedding 和 `scripts/17_run_real_embedding_ablation.py` 对照尚未执行，按当前安排留到后续与其它真实模态统一跑。

## 阶段 15：EEG 真实 embedding，MNE + bandpower/statistics baseline

状态：`eeg_bandpower_v1` 和 `eeg_deep_frozen_v1` 已跑通到单被试；全量 EEG 和后续 ablation 留到后续统一执行。

- 新增 `src/daily_multimodal/embeddings/eeg_real.py`，读取阶段 12 的 `eeg_windows` cache，使用 MNE 从 BDF 裁剪窗口附近数据，执行 50 Hz notch、1-45 Hz bandpass、250 Hz 重采样，并要求窗口为 `[channels, 2500]`。
- 新增 `scripts/14_extract_eeg_embeddings.py`，写出 `eeg_real_embeddings.npz`，包含 `eeg_emb`、`sample_id`、`event_id`、`subject_id`、`modality_mask`、`quality_flags` 和 `encoder_version`。
- `eeg_bandpower_v1` 使用每通道统计量和 1-4/4-8/8-13/13-30/30-45 Hz bandpower 聚合，再用固定随机种子投影到 256 维。
- `eeg_deep_frozen_v1` 使用 Braindecode EEGPT frozen backend，从 `braindecode/eegpt-pretrained` 的 `model.safetensors` 抽取 2048 维 deep feature，再投影到 256 维。
- 缺 cache 或 BDF 时写 `source_missing`；窗口维度不是 2500 samples、空通道或非有限值时写 `shape_mismatch`；缺 MNE/Braindecode 时写 `dependency_missing`；缺 deep checkpoint 时写 `checkpoint_missing`；CUDA/CPU OOM 时写 `oom`。
- 新增 `tests/test_eeg_real_embedding.py`，覆盖成功写 `.npz`、缺 cache、shape mismatch、deep checkpoint 缺失、deep feature shape mismatch、OOM 分类和 CLI 失败清单。
- `configs/encoders.yaml` 增加 `eeg_real_profiles`，记录 `eeg_bandpower_v1` 和 `eeg_deep_frozen_v1` 的 256 维、250 Hz、2500 samples、Braindecode EEGPT backend 和 checkpoint 约束。

本地验证：

```powershell
python -m pytest tests -q
python -m compileall -q src scripts tests
```

结果：

```text
47 passed
compileall passed
```

服务器验证：

```bash
cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding
source /home/lzs/miniconda3/etc/profile.d/conda.sh
conda activate lzs
PYTHONPATH=src python -m unittest discover -s tests
python -m compileall -q src scripts tests
```

结果：

```text
Ran 47 tests
OK
compileall passed
```

服务器 10 窗口验证：

```bash
PYTHONPATH=src python scripts/11_prepare_real_embedding_cache.py \
  --window-index outputs/window_index/real_cache_complete_10.jsonl \
  --cache-root outputs/cache/real_stage12_wav2vec2_10 \
  --eeg-encoder-profile eeg_bandpower_v1 \
  --audio-encoder-profile wav2vec2_frozen_v1 \
  --out-report outputs/reports/real_embedding_readiness_eeg_bandpower_10.md \
  --failures-out outputs/reports/real_embedding_failures_eeg_bandpower_10.json
PYTHONPATH=src python scripts/14_extract_eeg_embeddings.py \
  --window-index outputs/window_index/real_cache_complete_10.jsonl \
  --cache-root outputs/cache/real_stage12_wav2vec2_10 \
  --encoder-profile eeg_bandpower_v1 \
  --out outputs/embeddings/eeg_real_bandpower_10_embeddings.npz \
  --failures-out outputs/reports/eeg_real_bandpower_10_failures.json \
  --summary-out outputs/reports/eeg_real_bandpower_10_quality_summary.json
```

结果：

```text
eeg_ready_count=10, eeg_missing_count=0
eeg_emb.shape=(10, 256)
success_count=10
failure_count=0
nan_count=0
mean_channel_count=64.0
target_window_samples=2500
failures=[]
```

服务器单被试验证：

```bash
PYTHONPATH=src python scripts/11_prepare_real_embedding_cache.py \
  --window-index outputs/window_index/audio_real_wav2vec2_sub-12.jsonl \
  --cache-root outputs/cache/real_stage12_wav2vec2_sub-12 \
  --eeg-encoder-profile eeg_bandpower_v1 \
  --audio-encoder-profile wav2vec2_frozen_v1 \
  --out-report outputs/reports/real_embedding_readiness_eeg_bandpower_sub-12.md \
  --failures-out outputs/reports/real_embedding_failures_eeg_bandpower_sub-12.json
PYTHONPATH=src python scripts/14_extract_eeg_embeddings.py \
  --window-index outputs/window_index/audio_real_wav2vec2_sub-12.jsonl \
  --cache-root outputs/cache/real_stage12_wav2vec2_sub-12 \
  --encoder-profile eeg_bandpower_v1 \
  --out outputs/embeddings/eeg_real_bandpower_sub-12_embeddings.npz \
  --failures-out outputs/reports/eeg_real_bandpower_sub-12_failures.json \
  --summary-out outputs/reports/eeg_real_bandpower_sub-12_quality_summary.json
```

结果：

```text
eeg_ready_count=25, eeg_missing_count=0
eeg_emb.shape=(25, 256)
sample_count=25
success_count=25
failure_count=0
nan_count=0
mean_channel_count=64.0
target_window_samples=2500
failures=[]
```

EEGPT checkpoint 接入：

```bash
cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding
source /home/lzs/miniconda3/etc/profile.d/conda.sh
conda activate lzs
HF_ENDPOINT=https://hf-mirror.com python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="braindecode/eegpt-pretrained",
    local_dir="outputs/checkpoints/eegpt-pretrained",
    resume_download=True,
)
PY
```

服务器环境：

```text
braindecode=1.5.2
pandas=2.3.3
checkpoint=outputs/checkpoints/eegpt-pretrained
checkpoint files=config.json, model.safetensors, pytorch_model.bin
```

加载说明：Braindecode 1.5.2 的 `EEGPT.from_pretrained()` 对该 checkpoint 的 `chans_id` 会出现 62 vs 19 的 shape mismatch；当前 backend 改为按真实 BDF 窗口通道数懒初始化模型，并从 `model.safetensors` 过滤加载 shape 匹配权重。10 窗口和单被试结果中均为 `loaded_key_count=102`、`skipped_key_count=1`、`skipped_keys_preview=['chans_id']`。

服务器 EEGPT 10 窗口验证：

```bash
PYTHONPATH=src python scripts/11_prepare_real_embedding_cache.py \
  --window-index outputs/window_index/real_cache_complete_10.jsonl \
  --cache-root outputs/cache/real_stage12_eegpt_10 \
  --eeg-encoder-profile eeg_deep_frozen_v1 \
  --audio-encoder-profile wav2vec2_frozen_v1 \
  --out-report outputs/reports/real_embedding_readiness_eegpt_10.md \
  --failures-out outputs/reports/real_embedding_failures_eegpt_10.json
PYTHONPATH=src python scripts/14_extract_eeg_embeddings.py \
  --window-index outputs/window_index/real_cache_complete_10.jsonl \
  --cache-root outputs/cache/real_stage12_eegpt_10 \
  --encoder-profile eeg_deep_frozen_v1 \
  --checkpoint outputs/checkpoints/eegpt-pretrained \
  --device cpu \
  --out outputs/embeddings/eeg_real_eegpt_10_embeddings.npz \
  --failures-out outputs/reports/eeg_real_eegpt_10_failures.json \
  --summary-out outputs/reports/eeg_real_eegpt_10_quality_summary.json
```

结果：

```text
eeg_ready_count=10, eeg_missing_count=0
eeg_emb.shape=(10, 256)
sample_count=10
success_count=10
failure_count=0
nan_count=0
mean_channel_count=64.0
target_window_samples=2500
deep_backend=braindecode_eegpt
deep_feature_dim=2048
failures=[]
```

服务器 EEGPT 单被试验证：

```bash
PYTHONPATH=src python scripts/11_prepare_real_embedding_cache.py \
  --window-index outputs/window_index/audio_real_wav2vec2_sub-12.jsonl \
  --cache-root outputs/cache/real_stage12_eegpt_sub-12 \
  --eeg-encoder-profile eeg_deep_frozen_v1 \
  --audio-encoder-profile wav2vec2_frozen_v1 \
  --out-report outputs/reports/real_embedding_readiness_eegpt_sub-12.md \
  --failures-out outputs/reports/real_embedding_failures_eegpt_sub-12.json
PYTHONPATH=src python scripts/14_extract_eeg_embeddings.py \
  --window-index outputs/window_index/audio_real_wav2vec2_sub-12.jsonl \
  --cache-root outputs/cache/real_stage12_eegpt_sub-12 \
  --encoder-profile eeg_deep_frozen_v1 \
  --checkpoint outputs/checkpoints/eegpt-pretrained \
  --device cpu \
  --out outputs/embeddings/eeg_real_eegpt_sub-12_embeddings.npz \
  --failures-out outputs/reports/eeg_real_eegpt_sub-12_failures.json \
  --summary-out outputs/reports/eeg_real_eegpt_sub-12_quality_summary.json
```

结果：

```text
eeg_ready_count=25, eeg_missing_count=0
eeg_emb.shape=(25, 256)
sample_count=25
success_count=25
failure_count=0
nan_count=0
mean_channel_count=64.0
target_window_samples=2500
deep_backend=braindecode_eegpt
deep_feature_dim=2048
failures=[]
```

补充：单被试使用 `--device cuda` 时服务器显存不足，失败清单会写 `error_type=oom`；当前 deep checkpoint 验收使用 CPU 完成。剩余项为全量 EEG embedding 和后续 ablation。

## 阶段 14：Face 真实 embedding，raw dirty video quality first

状态：`face_raw_openface_stats_v1` 已跑通到 10 窗口和 `sub-12` 单被试；服务器缺 OpenFace 可执行文件，当前显式使用 OpenCV Haar dirty raw fallback，不做裁剪/增强/跟踪预处理。全量 face、5 seed 下游对照和 bootstrap 留到后续统一执行。

- 新增 `src/daily_multimodal/embeddings/face_real.py`，读取阶段 12 的 `openface` cache，复用已有 CSV；缺 CSV 时默认要求 OpenFace `FeatureExtraction/OpenFaceOffline`，缺失则写 `dependency_missing`。
- 新增显式 `--allow-opencv-fallback`：仅在用户/命令显式启用时，从原始 MP4 的窗口片段用 ffmpeg 抽样帧，再用 OpenCV Haar detector 生成等价 dirty CSV。该 fallback 不做 face crop、亮度增强、跟踪、去模糊或表情修复。
- 由于服务器当前走 OpenCV Haar fallback，下面报告中的 `mean_face_detection_success_rate`、`mean_openface_confidence`、`mean_low_confidence_ratio` 是 fallback 生成的 OpenFace-compatible 质量字段，不是 OpenFace 原生可执行文件输出。
- 新增 `scripts/13_extract_face_embeddings.py` 和 `scripts/13_audit_face_quality.py`，输出 `.npz`、失败清单、质量 summary、Markdown audit 和 `face_preprocessing_decision*.json`。
- 低质量窗口保留样本行但把 face mask 置 0，同时写 `quality_threshold_failed`；这样不丢样本对齐，也不把坏脸窗口当作有效 face 信号。
- `configs/encoders.yaml` 增加 `face_raw_openface_stats_v1` 和 `face_preprocessed_openface_stats_v1`，两者缓存路径和 profile 分离。

本地验证：

```powershell
python -m pytest tests -q
python -m compileall -q src scripts tests
```

结果：

```text
53 passed
compileall passed
```

服务器验证：

```bash
cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding
source /home/lzs/miniconda3/etc/profile.d/conda.sh
conda activate lzs
PYTHONPATH=src python -m unittest discover -s tests
python -m compileall -q src scripts tests
```

结果：

```text
Ran 53 tests
OK
compileall passed
```

服务器环境检查：

```text
FeatureExtraction=None
OpenFaceOffline=None
ffmpeg=/usr/bin/ffmpeg
cv2=True
```

服务器 10 窗口 raw face 验证：

```bash
PYTHONPATH=src python scripts/11_prepare_real_embedding_cache.py \
  --window-index outputs/window_index/real_cache_complete_10.jsonl \
  --cache-root outputs/cache/real_stage12_face_raw_10 \
  --face-encoder-profile face_raw_openface_stats_v1 \
  --audio-encoder-profile wav2vec2_frozen_v1 \
  --eeg-encoder-profile eeg_deep_frozen_v1 \
  --out-report outputs/reports/real_embedding_readiness_face_raw_10.md \
  --failures-out outputs/reports/real_embedding_failures_face_raw_10.json
PYTHONPATH=src python scripts/13_extract_face_embeddings.py \
  --window-index outputs/window_index/real_cache_complete_10.jsonl \
  --cache-root outputs/cache/real_stage12_face_raw_10 \
  --encoder-profile face_raw_openface_stats_v1 \
  --allow-opencv-fallback \
  --min-success-rate 0.10 \
  --out outputs/embeddings/face_raw_openface_10_embeddings.npz \
  --failures-out outputs/reports/face_raw_openface_10_failures.json \
  --summary-out outputs/reports/face_raw_openface_10_quality_summary.json \
  --decision-out outputs/reports/face_preprocessing_decision_10.json
```

结果：

```text
face_ready_count=10, face_missing_count=0
face_emb.shape=(10, 256)
nan_count=0
embedded_count=10
success_count=4
failure_count=6
masked_count=6
failure_types={"quality_threshold_failed": 6}
mean_face_detection_success_rate=0.10
mean_openface_confidence=0.0535
mean_low_confidence_ratio=1.0
```

服务器 `sub-12` 单被试 raw face 验证：

```bash
PYTHONPATH=src python scripts/11_prepare_real_embedding_cache.py \
  --window-index outputs/window_index/audio_real_wav2vec2_sub-12.jsonl \
  --cache-root outputs/cache/real_stage12_face_raw_sub-12 \
  --face-encoder-profile face_raw_openface_stats_v1 \
  --audio-encoder-profile wav2vec2_frozen_v1 \
  --eeg-encoder-profile eeg_deep_frozen_v1 \
  --out-report outputs/reports/real_embedding_readiness_face_raw_sub-12.md \
  --failures-out outputs/reports/real_embedding_failures_face_raw_sub-12.json
PYTHONPATH=src python scripts/13_extract_face_embeddings.py \
  --window-index outputs/window_index/audio_real_wav2vec2_sub-12.jsonl \
  --cache-root outputs/cache/real_stage12_face_raw_sub-12 \
  --encoder-profile face_raw_openface_stats_v1 \
  --allow-opencv-fallback \
  --min-success-rate 0.10 \
  --out outputs/embeddings/face_raw_openface_sub-12_embeddings.npz \
  --failures-out outputs/reports/face_raw_openface_sub-12_failures.json \
  --summary-out outputs/reports/face_raw_openface_sub-12_quality_summary.json \
  --decision-out outputs/reports/face_preprocessing_decision_sub-12.json
```

结果：

```text
face_ready_count=25, face_missing_count=0
face_emb.shape=(25, 256)
nan_count=0
embedded_count=25
success_count=25
failure_count=0
masked_count=0
mean_face_detection_success_rate=0.7760
mean_openface_confidence=0.5806
mean_low_confidence_ratio=0.6960
```

决策：

```text
enable_preprocessing=false
default_branch=face_raw_openface_stats_v1
raw_quality_gate_passed=false
triggered_conditions=["raw_quality_gate_incomplete_or_failed"]
```

解释：raw dirty face 分支已经能生成真实视频派生 embedding，但质量门槛未通过，尤其 OpenCV fallback 的 confidence 和 low-confidence ratio 明显偏弱。由于阶段计划要求“质量门槛、5 seed 下游指标和 bootstrap delta 共同决定”是否启用预处理，当前不把预处理设为默认；后续应在全量和下游对照阶段决定是否执行 `face_preprocessed_openface_stats_v1`。

## 阶段 16：Wear 真实 sequence embedding

状态：`wear_sequence_v1` 已跑通到 10 窗口和 `sub-12` 单被试。当前版本不依赖外部 checkpoint，使用 PPG/GSR/ACC 真实窗口序列重采样、质量统计和固定投影生成 256 维 embedding；和 `basic_wear_statistics_v1` 的下游指标对照留到阶段 18 统一 ablation。

- 新增 `src/daily_multimodal/embeddings/wear_real.py`，读取阶段 12 的 `wear_windows/<sample_id>/<encoder_profile>/window.json`，按绝对时间切 PPG/GSR/ACC CSV。
- PPG 重采样到 64 Hz，10 秒窗口输出 `[640, 1]`；GSR/ACC 重采样到 32 Hz，分别输出 `[320, 1]` 和 `[320, 3]`。
- 对缺行、重复时间戳、非单调时间戳、有效采样率、motion intensity、stationary ratio 写入 quality flags。
- 每个窗口写 raw sequence cache `sequence.npz` 和统计质量 cache `stats.json`。
- 新增 `scripts/15_extract_wear_embeddings.py`，输出 `wear_real_embeddings.npz`、失败清单和质量 summary。
- `configs/encoders.yaml` 增加 `wear_sequence_v1` 与预留的 `wear_deep_sequence_v1` profile。

本地验证：

```powershell
python -m pytest tests/test_wear_real_embedding.py -q
python -m pytest tests -q
python -m compileall -q src scripts tests
```

结果：

```text
5 passed
58 passed
compileall passed
```

服务器验证：

```bash
cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding
source /home/lzs/miniconda3/etc/profile.d/conda.sh
conda activate lzs
PYTHONPATH=src python -m unittest discover -s tests
python -m compileall -q src scripts tests
```

结果：

```text
Ran 58 tests
OK
compileall passed
```

服务器 10 窗口 Wear sequence 验证：

```bash
PYTHONPATH=src python scripts/11_prepare_real_embedding_cache.py \
  --window-index outputs/window_index/real_cache_complete_10.jsonl \
  --cache-root outputs/cache/real_stage12_wear_sequence_10 \
  --wear-encoder-profile wear_sequence_v1 \
  --audio-encoder-profile wav2vec2_frozen_v1 \
  --eeg-encoder-profile eeg_deep_frozen_v1 \
  --face-encoder-profile face_raw_openface_stats_v1 \
  --out-report outputs/reports/real_embedding_readiness_wear_sequence_10.md \
  --failures-out outputs/reports/real_embedding_failures_wear_sequence_10.json
PYTHONPATH=src python scripts/15_extract_wear_embeddings.py \
  --window-index outputs/window_index/real_cache_complete_10.jsonl \
  --cache-root outputs/cache/real_stage12_wear_sequence_10 \
  --encoder-profile wear_sequence_v1 \
  --out outputs/embeddings/wear_real_sequence_10_embeddings.npz \
  --failures-out outputs/reports/wear_real_sequence_10_failures.json \
  --summary-out outputs/reports/wear_real_sequence_10_quality_summary.json
```

结果：

```text
wear_ready_count=10, wear_missing_count=0
wear_emb.shape=(10, 256)
sample_count=10
success_count=10
failure_count=0
masked_count=0
nan_count=0
mean_motion_intensity=9.840719890594482
mean_stationary_ratio=0.6489028213166145
mean_ppg_effective_sampling_rate_hz=1.0
mean_gsr_effective_sampling_rate_hz=1.0
mean_acc_effective_sampling_rate_hz=1.0
failures=[]
raw_sequence_shapes=PPG [640, 1], GSR [320, 1], ACC [320, 3]
```

服务器 `sub-12` 单被试 Wear sequence 验证：

```bash
PYTHONPATH=src python scripts/11_prepare_real_embedding_cache.py \
  --window-index outputs/window_index/audio_real_wav2vec2_sub-12.jsonl \
  --cache-root outputs/cache/real_stage12_wear_sequence_sub-12 \
  --wear-encoder-profile wear_sequence_v1 \
  --audio-encoder-profile wav2vec2_frozen_v1 \
  --eeg-encoder-profile eeg_deep_frozen_v1 \
  --face-encoder-profile face_raw_openface_stats_v1 \
  --out-report outputs/reports/real_embedding_readiness_wear_sequence_sub-12.md \
  --failures-out outputs/reports/real_embedding_failures_wear_sequence_sub-12.json
PYTHONPATH=src python scripts/15_extract_wear_embeddings.py \
  --window-index outputs/window_index/audio_real_wav2vec2_sub-12.jsonl \
  --cache-root outputs/cache/real_stage12_wear_sequence_sub-12 \
  --encoder-profile wear_sequence_v1 \
  --out outputs/embeddings/wear_real_sequence_sub-12_embeddings.npz \
  --failures-out outputs/reports/wear_real_sequence_sub-12_failures.json \
  --summary-out outputs/reports/wear_real_sequence_sub-12_quality_summary.json
```

结果：

```text
wear_ready_count=25, wear_missing_count=0
wear_emb.shape=(25, 256)
sample_count=25
success_count=25
failure_count=0
masked_count=0
nan_count=0
mean_motion_intensity=9.988039436340332
mean_stationary_ratio=0.8234482758620689
mean_ppg_effective_sampling_rate_hz=1.0
mean_gsr_effective_sampling_rate_hz=1.0
mean_acc_effective_sampling_rate_hz=1.0
failures=[]
```
