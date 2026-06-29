# 运行命令和产物

> 这是查表材料。如果还不理解行为路径，先读 [一条事件如何变成 smoke embedding](../walkthroughs/one-real-run.md)。

## 阶段命令

| 阶段 | 命令 | 主要产物 | 源码入口 |
| --- | --- | --- | --- |
| 环境检查 | `python scripts/00_check_environment.py --config configs/paths.server.yaml` | 控制台环境报告 | [环境检查脚本](../../scripts/00_check_environment.py) |
| 构建 manifest | `python scripts/01_build_manifest.py --config configs/paths.server.yaml --out outputs/manifests/events_manifest.jsonl --coverage-out outputs/reports/manifest_coverage.json` | `events_manifest.jsonl`、`manifest_coverage.json` | [manifest 入口](../../scripts/01_build_manifest.py) |
| manifest 汇总验证 | `python scripts/02_validate_alignment.py --manifest outputs/manifests/events_manifest.jsonl --summary-out outputs/reports/manifest_summary.json` | `manifest_summary.json` | [manifest 汇总入口](../../scripts/02_validate_alignment.py) |
| 视频音频精确对齐 | `python scripts/02_align_video_audio.py --manifest outputs/manifests/events_manifest.jsonl --out outputs/manifests/events_manifest_with_video_audio.jsonl --report-out outputs/reports/video_audio_alignment_report.json` | enriched manifest、alignment report；默认还会写 ffprobe cache，但当前本地副本尚未同步该 cache | [视频音频对齐入口](../../scripts/02_align_video_audio.py) |
| 窗口索引 | `python scripts/03_build_window_index.py --manifest outputs/manifests/events_manifest.jsonl --out outputs/window_index/window_index.jsonl` | `window_index.jsonl` | [窗口索引入口](../../scripts/03_build_window_index.py) |
| 单事件探针 | `python scripts/03_probe_one_event.py --manifest outputs/manifests/events_manifest.jsonl --require-all-modalities` | `probe_one_event.json`、`probe_one_event_shapes.txt` | [探针入口](../../scripts/03_probe_one_event.py) |
| 单事件 embedding | `python scripts/04_extract_one_event_embeddings.py --window-index outputs/window_index/window_index.jsonl --require-all-modalities --encoder-profile basic` | `one_event_embeddings.npz`、`one_event_embedding_report.json` | [单事件 embedding 入口](../../scripts/04_extract_one_event_embeddings.py) |
| 10 条 smoke embedding | `python scripts/05_extract_smoke_embeddings.py --window-index outputs/window_index/window_index.jsonl --require-all-modalities --max-events 10 --encoder-profile basic` | `smoke_10_events_basic_embeddings.npz`、`smoke_10_events_basic_report.json` | [smoke embedding 入口](../../scripts/05_extract_smoke_embeddings.py) |
| 单被试 embedding | `python scripts/06_extract_subject_embeddings.py --window-index outputs/window_index/window_index.jsonl --subject-id sub-10 --require-all-modalities --encoder-profile basic` | `sub-10_basic_embeddings.npz`、`sub-10_basic_report.json` | [单被试入口](../../scripts/06_extract_subject_embeddings.py) |
| 全量 embedding | `python scripts/07_extract_all_embeddings.py --window-index outputs/window_index/window_index.jsonl --require-all-modalities` | `all_complete_basic_embeddings.npz`、`all_complete_multimodal_manifest.jsonl`、全量报告和失败清单 | [全量入口](../../scripts/07_extract_all_embeddings.py) |
| baseline MLP | `python scripts/08_train_baseline_mlp.py --embeddings outputs/embeddings/all_complete_basic_embeddings.npz --split subject --out-dir outputs/models` | `baseline_mlp.pt`、`baseline_mlp_metrics.json`、`baseline_mlp_table.md` | [baseline 入口](../../scripts/08_train_baseline_mlp.py) |
| baseline 基准快照 | `python scripts/10_run_upgrade_ablation.py --embeddings outputs/embeddings/all_complete_basic_embeddings.npz --baseline outputs/reports/baseline_mlp_metrics.json --snapshot-baseline --target-label alert` | `baseline_reference_metrics.json`、`baseline_reference_table.md`、`baseline_reference_manifest.json` | [升级对照入口](../../scripts/10_run_upgrade_ablation.py) |
| 融合升级对照 | `python scripts/10_run_upgrade_ablation.py --embeddings outputs/embeddings/all_complete_basic_embeddings.npz --baseline outputs/reports/baseline_reference_metrics.json --upgrade modality_token_attention --target-label alert` | `modality_token_fusion.pt`、`modality_token_fusion_metrics.json`、`model_upgrade_ablation_table.md`、`model_upgrade_failures.json` | [升级对照入口](../../scripts/10_run_upgrade_ablation.py) |
| 真实 embedding 缓存准备 | `python scripts/11_prepare_real_embedding_cache.py --window-index outputs/window_index/window_index.jsonl --max-windows 10 --out-report outputs/reports/real_embedding_readiness_10.md --failures-out outputs/reports/real_embedding_failures_10.json` | `outputs/cache/audio_clips/`、`outputs/cache/openface/`、`outputs/cache/eeg_windows/`、`outputs/cache/wear_windows/`、readiness report、真实 encoder 失败清单 | [缓存准备入口](../../scripts/11_prepare_real_embedding_cache.py) |
| Audio 真实 embedding | `python scripts/12_extract_audio_embeddings.py --window-index outputs/window_index/real_cache_complete_10.jsonl --max-windows 10 --cache-root outputs/cache/real_stage12_wav2vec2_10 --encoder-profile wav2vec2_frozen_v1 --checkpoint outputs/checkpoints/wav2vec2-base-960h --out outputs/embeddings/audio_real_wav2vec2_10_embeddings.npz --failures-out outputs/reports/audio_real_wav2vec2_10_failures.json --summary-out outputs/reports/audio_real_wav2vec2_10_quality_summary.json` | `audio_real_wav2vec2_10_embeddings.npz`、audio 失败清单、audio 质量摘要；缺 checkpoint 或依赖时应写 `checkpoint_missing` 或 `dependency_missing`，不生成伪 real embedding | [Audio 真实入口](../../scripts/12_extract_audio_embeddings.py) |

## 视频音频对齐重跑参数

| 参数 | 行为 |
| --- | --- |
| `--ffprobe-cache` | 默认写入 ffprobe cache；重复路径会复用已有探测结果。 |
| `--retry-failed-ffprobe` | 对 cache 中 `ok: false` 的记录重新探测，适合修复慢文件或临时失败。 |
| `--ffprobe-timeout 0` | 脚本会把 `0` 或负数转成 Python subprocess 的 `timeout=None`，表示不限时。 |
| `--ffprobe-workers` | 控制并发探测 MP4 的线程数，默认 `8`。 |

## 当前同步产物

| 产物 | 用途 | 当前证据 |
| --- | --- | --- |
| `outputs/manifests/events_manifest.jsonl` | 事件级 manifest，本地同步副本约 2.6 MB | [本地产物](../../outputs/manifests/events_manifest.jsonl) |
| `outputs/reports/manifest_summary.json` | manifest 汇总数字 | [汇总报告](../../outputs/reports/manifest_summary.json) |
| `outputs/window_index/window_index.jsonl` | 窗口级样本索引 | [本地窗口索引](../../outputs/window_index/window_index.jsonl) |
| `outputs/reports/probe_one_event_shapes.txt` | 单窗口形状和候选数量快照 | [形状报告](../../outputs/reports/probe_one_event_shapes.txt) |
| `outputs/embeddings/one_event_embeddings.npz` | 单事件 smoke embedding | [本地产物](../../outputs/embeddings/one_event_embeddings.npz) |
| `outputs/embeddings/smoke_10_events_basic_embeddings.npz` | 10 条事件 smoke embedding | [本地产物](../../outputs/embeddings/smoke_10_events_basic_embeddings.npz) |

阶段 8/9/10 的入口已在本地合成数据和服务器 `wzw` 工作目录中验证通过。服务器阶段 10 全量对照显示 `modality_token_attention` 相比 baseline full test RMSE 从 `0.8756` 降到 `0.6968`，判定为 `accepted`。阶段 11/12 的本地测试覆盖了真实 embedding 契约、失败清单、缓存 key、四模态 cache 记录和缺失源失败语义；服务器阶段 12 使用 `window_index_with_video_audio.jsonl` 中 10 条完整且带 `video_candidates` 的窗口验证通过，EEG、Wear、Face、Audio 均为 `ready=10, missing=0, failures=0`，失败清单为 `[]`。阶段 13 的 Audio 真实 embedding 已用服务器 `lzs` 环境和 `facebook/wav2vec2-base-960h` safetensors fallback 跑通 10 窗口与 `sub-12` 单被试；全量和 ablation 后续统一执行。

## 常用验证

| 目的 | 命令 |
| --- | --- |
| 跑全部单元测试 | `python -m pytest tests -q` |
| 跑阶段 11-12 单元测试 | `python -m pytest tests/test_real_embedding_contracts.py tests/test_embedding_failures.py tests/test_real_pipeline.py -q` |
| 跑阶段 13 Audio 单元测试 | `python -m pytest tests/test_audio_real_embedding.py -q` |
| 检查导入和语法 | `python -m compileall -q src scripts tests` |
| 验证 repo-docs 结构 | `python C:\Users\28303\.codex\skills\repo-docs\scripts\validate_repo_docs.py repo-docs --repo-root .` |
