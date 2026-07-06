# Daily Multimodal Embedding

本项目把 Daily Multimodal 的 EEG、PPG、GSR、ACC、面部录像和录音素材，整理成窗口级 embedding、真实模态缓存、下游 ablation 和跨被试验证的一套可复现实验工程。

完整导览在 [repo-docs/README.md](repo-docs/README.md)。第一次理解代码时，建议先读 [一条事件如何变成 smoke embedding](repo-docs/walkthroughs/one-real-run.md)，再按 [运行命令和产物](repo-docs/references/commands-and-artifacts.md) 查脚本入口。

## 当前架构

| 层级 | 作用 | 主要入口 |
| --- | --- | --- |
| 数据发现 | 从 EEG/评分、wear、视频目录构建事件级 manifest 和覆盖率摘要 | `scripts/01_build_manifest.py`, `scripts/02_validate_alignment.py` |
| 时间对齐 | 用 ffprobe cache 把日期级视频/音频候选变成精确片段 | `scripts/02_align_video_audio.py`, `src/daily_multimodal/alignment/video_audio_alignment.py` |
| 窗口样本 | 默认把评分事件前 2 分钟切成 12 个 10 秒窗口，并生成稳定 `sample_id` | `scripts/03_build_window_index.py`, `src/daily_multimodal/alignment/event_windows.py` |
| smoke embedding | 用统一 `(N, 256)` 契约跑通基础 embedding、全量 basic、baseline 和融合升级 | `scripts/04_*` 到 `scripts/10_*`, `src/daily_multimodal/embeddings/basic.py` |
| 真实缓存 | 为 Audio、Face、EEG、Wear 准备窗口级 cache，并记录结构化失败 | `scripts/11_prepare_real_embedding_cache.py`, `src/daily_multimodal/embeddings/cache.py` |
| 单模态真实 embedding | 分别抽取 audio、face、eeg、wear 的真实 256 维表示和质量摘要 | `scripts/12_*` 到 `scripts/15_*` |
| all-real 打包 | 按 `sample_id` 合并四模态 `.npz`，缺失模态写零向量并保留 `modality_mask` | `scripts/16_extract_all_real_embeddings.py`, `src/daily_multimodal/embeddings/real_pipeline.py` |
| 下游验证 | 跑 real ablation、fair leakage controls、subject CV、wear/video 分支实验 | `scripts/17_*` 到 `scripts/28_*`, `src/daily_multimodal/training/` |
| 可视化和审计 | 画数据完整性/时间轴图，审计 EEG 覆盖、视频行为 flags、视频变体和 probe | `scripts/19_*`, `scripts/21_*`, `scripts/23_*` 到 `scripts/28_*` |

## 当前事实

本地和服务器记录显示，项目已经超过早期 smoke 阶段：

```text
events_total=1272
complete_multimodal_candidates=995
two-minute windows=8640
all_complete_real_v2 rows=781
real v2 mask sum=[738, 781, 207, 781]
```

当前稳健的 fatigue 验证使用 EEG/Wear/Audio 三模态，因为 true OpenFace 覆盖仍稀疏，四模态 LOSO 会出现空 fold。视频分支已经新增 V1/V2/V4、MediaPipe behavior flags、DINOv2 ROI、S1/S2/S3/S4 split 和 subject/session/fatigue probes。

## 运行约定

1. 本地创建和修改代码，再同步到服务器 `wzw` 目录：`/mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding`。
2. 大规模或长时间任务前先跑小规模可行性测试，例如单事件、10 条 smoke、单被试、20 窗口 smoke 或 chunk。
3. 服务器端产生的 `outputs/manifests/`、`outputs/reports/`、`outputs/logs/`、`outputs/embeddings/` 需要同步回本地对应目录，保证本地保留可追踪副本。
4. 视频音频精确对齐会写 ffprobe cache；重跑慢文件时可用 `--retry-failed-ffprobe`，把 `--ffprobe-timeout` 设为 `0` 或负数表示不限时。
5. 从 Windows PowerShell 调远程 Bash 时，复杂命令用单引号 here-string 管道到 `ssh ... 'bash -s'`，不要把 `$`、`$()`、heredoc、重定向或多行 Python 放进 PowerShell 双引号字符串。

## 常用验证

```bash
python -m pytest tests -q
python -m compileall -q src scripts tests
python C:\Users\28303\.codex\skills\repo-docs\scripts\validate_repo_docs.py repo-docs --repo-root .
```

