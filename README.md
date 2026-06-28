# Daily Multimodal Embedding

本项目用于跑通 Daily Multimodal 数据的前半段工程闭环：从 EEG、PPG、GSR、ACC、面部录像和录音原始数据构建事件级 `manifest`，做视频音频时间对齐，生成窗口级样本，再输出基础版 smoke embedding。当前目标是先让数据路径、时间窗口、模态 mask、产物格式和服务器复跑流程可验证；最终模型效果不由当前 `basic` profile 代表。

更完整的仓库导览在 [repo-docs/README.md](repo-docs/README.md)。第一次理解代码时，建议从 [一条事件如何变成 smoke embedding](repo-docs/walkthroughs/one-real-run.md) 开始。

## 当前状态

| 层级 | 当前状态 | 证据 |
| --- | --- | --- |
| 阶段 0-2 | 服务器目录、项目脚手架、只读 manifest 构建和覆盖率报告已跑通 | [阶段 0-2 执行记录](阶段0-2执行记录.md) |
| 阶段 3 | 视频音频精确对齐入口已实现，支持 ffprobe cache、失败重试和不限时重跑 | [视频音频对齐脚本](scripts/02_align_video_audio.py) |
| 阶段 4-6 | window index、单事件探针、单事件和 10 条 smoke embedding 已在服务器验证 | [阶段 4-6 服务器验证报告](阶段4-6服务器验证报告.md) |
| 阶段 7 | 单被试 `basic` embedding 入口已实现，按 `subject_id` 选择窗口并复用统一 embedding 契约 | [单被试入口](scripts/06_extract_subject_embeddings.py) |
| 阶段 7+ | 全量提取入口仍是占位，需等待单被试验证后实现 | [全量占位入口](scripts/07_extract_all_embeddings.py) |

当前本地同步产物显示：

```text
events_total=1272
complete_wear_events=1127
video_day_events=1103
complete_multimodal_candidates=995
```

这些数字来自 `outputs/reports/manifest_summary.json`。本地 `outputs/` 目前包含阶段 0-6 的 manifest、window index、探针报告和 smoke embedding；尚未包含精确对齐后的 manifest、alignment report、ffprobe cache 或单被试 embedding 产物。

## 运行约定

1. 本地创建和修改代码，再同步到服务器 `wzw` 目录：`/mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding`。
2. 大规模或长时间任务前先跑小规模可行性测试，例如单事件、10 条 smoke、单被试。
3. 服务器端产生的 `outputs/manifests/`、`outputs/reports/`、`outputs/logs/`、`outputs/embeddings/` 需要同步回本地对应目录，保证本地保留可追踪副本。
4. 视频音频精确对齐会写 ffprobe cache；重跑慢文件时可用 `--retry-failed-ffprobe`，把 `--ffprobe-timeout` 设为 `0` 或负数表示不限时。

## 常用命令

```bash
python scripts/00_check_environment.py --config configs/paths.server.yaml

python scripts/01_build_manifest.py \
  --config configs/paths.server.yaml \
  --out outputs/manifests/events_manifest.jsonl \
  --coverage-out outputs/reports/manifest_coverage.json

python scripts/02_validate_alignment.py \
  --manifest outputs/manifests/events_manifest.jsonl \
  --summary-out outputs/reports/manifest_summary.json

python scripts/02_align_video_audio.py \
  --manifest outputs/manifests/events_manifest.jsonl \
  --out outputs/manifests/events_manifest_with_video_audio.jsonl \
  --report-out outputs/reports/video_audio_alignment_report.json

python scripts/03_build_window_index.py \
  --manifest outputs/manifests/events_manifest.jsonl \
  --out outputs/window_index/window_index.jsonl

python scripts/03_probe_one_event.py \
  --window-index outputs/window_index/window_index.jsonl \
  --require-all-modalities

python scripts/05_extract_smoke_embeddings.py \
  --window-index outputs/window_index/window_index.jsonl \
  --require-all-modalities \
  --max-events 10 \
  --encoder-profile basic

python scripts/06_extract_subject_embeddings.py \
  --window-index outputs/window_index/window_index.jsonl \
  --subject-id sub-10 \
  --require-all-modalities \
  --encoder-profile basic
```

更完整的命令、产物和字段契约见 [运行命令和产物](repo-docs/references/commands-and-artifacts.md) 与 [字段契约](repo-docs/references/data-contracts.md)。

## 本地验证

```bash
python -m pytest tests -q
python -m compileall -q src scripts tests
```

当前测试覆盖 manifest 构建、时间解析、窗口切分、单事件探针、视频音频对齐、embedding 保存和单被试选择。

