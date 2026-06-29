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
