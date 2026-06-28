# 运行命令和产物

> 这是查表材料。如果还不理解行为路径，先读 [一条事件如何变成 smoke embedding](../walkthroughs/one-real-run.md)。

## 阶段命令

| 阶段 | 命令 | 主要产物 | 源码入口 |
| --- | --- | --- | --- |
| 环境检查 | `python scripts/00_check_environment.py --config configs/paths.server.yaml` | 控制台环境报告 | [环境检查脚本](../../scripts/00_check_environment.py) |
| 构建 manifest | `python scripts/01_build_manifest.py --config configs/paths.server.yaml --out outputs/manifests/events_manifest.jsonl --coverage-out outputs/reports/manifest_coverage.json` | `events_manifest.jsonl`、`manifest_coverage.json` | [manifest 入口](../../scripts/01_build_manifest.py) |
| 视频音频精确对齐 | `python scripts/02_align_video_audio.py --manifest outputs/manifests/events_manifest.jsonl --out outputs/manifests/events_manifest_with_video_audio.jsonl --report-out outputs/reports/video_audio_alignment_report.json` | enriched manifest、alignment report；默认还会写 ffprobe cache，但当前本地副本尚未同步该 cache | [视频音频对齐入口](../../scripts/02_align_video_audio.py) |
| 窗口索引 | `python scripts/03_build_window_index.py --manifest outputs/manifests/events_manifest.jsonl --out outputs/window_index/window_index.jsonl` | `window_index.jsonl` | [窗口索引入口](../../scripts/03_build_window_index.py) |
| 单事件探针 | `python scripts/03_probe_one_event.py --manifest outputs/manifests/events_manifest.jsonl --require-all-modalities` | `probe_one_event.json`、`probe_one_event_shapes.txt` | [探针入口](../../scripts/03_probe_one_event.py) |
| 单事件 embedding | `python scripts/04_extract_one_event_embeddings.py --window-index outputs/window_index/window_index.jsonl --require-all-modalities --encoder-profile basic` | `one_event_embeddings.npz`、`one_event_embedding_report.json` | [单事件 embedding 入口](../../scripts/04_extract_one_event_embeddings.py) |
| 10 条 smoke embedding | `python scripts/05_extract_smoke_embeddings.py --window-index outputs/window_index/window_index.jsonl --require-all-modalities --max-events 10 --encoder-profile basic` | `smoke_10_events_basic_embeddings.npz`、`smoke_10_events_basic_report.json` | [smoke embedding 入口](../../scripts/05_extract_smoke_embeddings.py) |
| 单被试 embedding | `python scripts/06_extract_subject_embeddings.py --window-index outputs/window_index/window_index.jsonl --subject-id sub-10 --require-all-modalities --encoder-profile basic` | `sub-10_basic_embeddings.npz`、`sub-10_basic_report.json` | [单被试入口](../../scripts/06_extract_subject_embeddings.py) |
| 全量 embedding | `python scripts/07_extract_all_embeddings.py` | 当前为占位错误消息 | [全量占位入口](../../scripts/07_extract_all_embeddings.py) |

## 当前同步产物

| 产物 | 用途 | 当前证据 |
| --- | --- | --- |
| `outputs/manifests/events_manifest.jsonl` | 事件级 manifest，本地同步副本约 2.6 MB | [本地产物](../../outputs/manifests/events_manifest.jsonl) |
| `outputs/reports/manifest_summary.json` | manifest 汇总数字 | [汇总报告](../../outputs/reports/manifest_summary.json) |
| `outputs/window_index/window_index.jsonl` | 窗口级样本索引 | [本地窗口索引](../../outputs/window_index/window_index.jsonl) |
| `outputs/reports/probe_one_event_shapes.txt` | 单窗口形状和候选数量快照 | [形状报告](../../outputs/reports/probe_one_event_shapes.txt) |
| `outputs/embeddings/one_event_embeddings.npz` | 单事件 smoke embedding | [本地产物](../../outputs/embeddings/one_event_embeddings.npz) |
| `outputs/embeddings/smoke_10_events_basic_embeddings.npz` | 10 条事件 smoke embedding | [本地产物](../../outputs/embeddings/smoke_10_events_basic_embeddings.npz) |

## 常用验证

| 目的 | 命令 |
| --- | --- |
| 跑全部单元测试 | `python -m pytest tests -q` |
| 检查导入和语法 | `python -m compileall -q src scripts tests` |
| 验证 repo-docs 结构 | `python C:\Users\28303\.codex\skills\repo-docs\scripts\validate_repo_docs.py repo-docs --repo-root .` |
