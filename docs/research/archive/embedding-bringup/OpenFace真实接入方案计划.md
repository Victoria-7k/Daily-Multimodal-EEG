# OpenFace 真实接入方案计划

## 目标

将当前 face 模态从 OpenCV Haar dirty fallback 升级为真正的 OpenFace `FeatureExtraction` 特征抽取，同时保持现有真实 embedding 契约不变：

- `face_emb` 输出仍为 `(N, 256)`。
- `modality_mask` 顺序仍为 `[eeg, wear, face, audio]`。
- 失败必须写入 failures JSON，不静默 fallback。
- 先跑 dirty/raw 视频质量观察，再决定是否启用预处理。

## 当前服务器结论

服务器环境：

```text
OS: Ubuntu 22.04
user: lzs
GPU: 5 x RTX 4090
workspace: /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding
```

已确认：

- `FeatureExtraction` / `OpenFaceOffline` 当前不在 PATH。
- Docker CLI 存在，但当前用户没有 Docker daemon 权限。
- `apptainer` / `singularity` 可用。
- `/mnt/dataset4` 空间充足。

因此最佳路线不是源码编译，也不是直接 Docker，而是：

```text
Apptainer OpenFace 镜像
+ wrapper 脚本
+ 现有 --openface-executable 参数
```

## 现有代码状态

现有 face pipeline 已经预留真实 OpenFace 接口：

```text
scripts/13_extract_face_embeddings.py
src/daily_multimodal/embeddings/face_real.py
```

支持：

- `--openface-executable`
- `OPENFACE_EXECUTABLE` 环境变量
- PATH 中的 `FeatureExtraction`
- PATH 中的 `OpenFaceOffline`
- `--allow-opencv-fallback`

当前重要问题：

```text
真 OpenFace 分支会对 source_path 原始 MP4 直接跑 FeatureExtraction，
而 OpenCV fallback 分支会按 clip_start_seconds/clip_end_seconds 抽窗口帧。
```

所以接入真 OpenFace 前必须修正为：

```text
先用 ffmpeg 切出 10 秒窗口级 MP4 clip，再对该 clip 跑 FeatureExtraction。
```

否则 OpenFace CSV 会来自整段视频，不是窗口级特征。

## 推荐实施路线

### 阶段 1：获取 OpenFace 镜像

在服务器执行：

```bash
mkdir -p /mnt/dataset4/sitian/wzw/tools/openface
cd /mnt/dataset4/sitian/wzw/tools/openface
apptainer pull openface.sif docker://algebr/openface:latest
```

如果镜像拉取失败，记录原因：

- 网络失败
- Docker Hub 访问失败
- 镜像不存在或权限问题
- Apptainer 转换失败

不要改用 `pip install openface`，那不是本项目需要的 OpenFace 2.2 `FeatureExtraction`。

### 阶段 2：确认容器内可执行文件

```bash
apptainer exec openface.sif find / -name FeatureExtraction 2>/dev/null | head
```

预期找到类似：

```text
/OpenFace/build/bin/FeatureExtraction
```

实际路径以服务器输出为准。

### 阶段 3：创建 wrapper

建议路径：

```text
/mnt/dataset4/sitian/wzw/tools/openface/feature_extraction.sh
```

示例：

```bash
#!/usr/bin/env bash
set -euo pipefail

OPENFACE_SIF=/mnt/dataset4/sitian/wzw/tools/openface/openface.sif
OPENFACE_BIN=/OpenFace/build/bin/FeatureExtraction

exec apptainer exec \
  --cleanenv \
  --bind /mnt/dataset1,/mnt/dataset4 \
  "$OPENFACE_SIF" \
  "$OPENFACE_BIN" "$@"
```

验证：

```bash
chmod +x /mnt/dataset4/sitian/wzw/tools/openface/feature_extraction.sh
/mnt/dataset4/sitian/wzw/tools/openface/feature_extraction.sh -help
```

## 必要代码修正

### 修正点

修改：

```text
src/daily_multimodal/embeddings/face_real.py
```

当前 `_run_openface(executable, source_path, csv_path)` 只接收原始 source MP4。

应增加窗口级 clip 逻辑：

1. 从 cache/window 读取 `clip_start_seconds` 和 `clip_end_seconds`。
2. 用 ffmpeg 生成窗口级 MP4：

```text
outputs/cache/.../openface/<sample_id>/<profile>/window.mp4
```

3. 对 `window.mp4` 调用 OpenFace：

```bash
FeatureExtraction -f window.mp4 -out_dir <cache_dir> -of openface
```

4. 输出仍读取：

```text
openface.csv
```

### 建议新增测试

新增或扩展：

```text
tests/test_face_real_embedding.py
```

覆盖：

- 真 OpenFace 分支调用前会先生成窗口级 clip。
- `_run_openface` 使用 clip，而不是整段 MP4。
- `--openface-executable` 存在时不走 OpenCV fallback。
- OpenFace 执行失败时写 `extraction_failed`。
- OpenFace 缺失且未允许 fallback 时写 `dependency_missing`。

## 验证顺序

### 1. 单命令 smoke

先找一个已存在的 raw MP4 和短输出目录：

```bash
/mnt/dataset4/sitian/wzw/tools/openface/feature_extraction.sh \
  -f <some-window-or-short-mp4> \
  -out_dir outputs/tmp/openface_smoke \
  -of openface_smoke
```

验收：

```text
outputs/tmp/openface_smoke/openface_smoke.csv exists
CSV 有 frame、confidence、success、pose/gaze/AU 等列
```

### 2. 10 窗口 OpenFace embedding

```bash
PYTHONPATH=src python scripts/13_extract_face_embeddings.py \
  --window-index outputs/window_index/real_cache_complete_10.jsonl \
  --cache-root outputs/cache/real_stage12_openface_real_10 \
  --encoder-profile face_raw_openface_stats_v1 \
  --openface-executable /mnt/dataset4/sitian/wzw/tools/openface/feature_extraction.sh \
  --out outputs/embeddings/face_openface_real_10_embeddings.npz \
  --failures-out outputs/reports/face_openface_real_10_failures.json \
  --summary-out outputs/reports/face_openface_real_10_quality_summary.json \
  --decision-out outputs/reports/face_preprocessing_decision_openface_real_10.json
```

验收：

- `face_emb.shape == (10, 256)`
- `nan_count == 0`
- failures 不应包含 `dependency_missing`
- 若质量低，可以出现 `quality_threshold_failed`

### 3. 单被试验证

```bash
PYTHONPATH=src python scripts/13_extract_face_embeddings.py \
  --window-index outputs/window_index/audio_real_wav2vec2_sub-12.jsonl \
  --cache-root outputs/cache/real_stage12_openface_real_sub-12 \
  --encoder-profile face_raw_openface_stats_v1 \
  --openface-executable /mnt/dataset4/sitian/wzw/tools/openface/feature_extraction.sh \
  --out outputs/embeddings/face_openface_real_sub-12_embeddings.npz \
  --failures-out outputs/reports/face_openface_real_sub-12_failures.json \
  --summary-out outputs/reports/face_openface_real_sub-12_quality_summary.json \
  --decision-out outputs/reports/face_preprocessing_decision_openface_real_sub-12.json
```

验收：

- `face_emb.shape == (25, 256)`
- `nan_count == 0`
- 报告记录 mean success/confidence/low-confidence ratio
- 和 OpenCV fallback 的 `face_raw_openface_sub-12_quality_summary.json` 对比

### 4. 全量真实 face

在 10 窗口和单被试通过后，再跑 full：

```bash
PYTHONPATH=src python scripts/13_extract_face_embeddings.py \
  --window-index outputs/window_index/real_cache_complete_full.jsonl \
  --cache-root outputs/cache/real_stage12_openface_real_full \
  --encoder-profile face_raw_openface_stats_v1 \
  --openface-executable /mnt/dataset4/sitian/wzw/tools/openface/feature_extraction.sh \
  --out outputs/embeddings/face_openface_real_full_embeddings.npz \
  --failures-out outputs/reports/face_openface_real_full_failures.json \
  --summary-out outputs/reports/face_openface_real_full_quality_summary.json \
  --decision-out outputs/reports/face_preprocessing_decision_openface_real_full.json
```

## 质量决策规则

当前代码里 raw quality gate：

```text
mean_face_detection_success_rate >= 0.80
mean_low_confidence_ratio <= 0.20
max(pose_bad, dark_frame, blur_frame, multi_face) <= 0.30
```

决策：

- 若 raw OpenFace quality gate 通过：保持 `face_raw_openface_stats_v1`。
- 若 raw OpenFace quality gate 不通过：进入预处理候选，但不能只凭质量摘要决定，仍需下游 5 seed + bootstrap。
- 若 OpenFace 比 OpenCV fallback 明显更稳定：后续全量 ablation 优先用真 OpenFace 产物。

## 风险和回退

| 风险 | 表现 | 回退 |
| --- | --- | --- |
| Apptainer 拉镜像失败 | `apptainer pull` 报网络或 registry 错 | 手动下载 sif 或换可访问镜像源 |
| 容器找不到 FeatureExtraction | `find / -name FeatureExtraction` 为空 | 换 OpenFace 镜像或源码构建 sif |
| 容器无法访问数据路径 | OpenFace 报 input not found | 修正 `--bind /mnt/dataset1,/mnt/dataset4` |
| OpenFace 处理整段视频太慢 | 单窗口耗时异常 | 必须先切 10 秒 clip |
| CSV 字段和当前 parser 不一致 | `OpenFace CSV does not contain...` | 扩展 `_openface_stats` 字段兼容 |
| 质量仍低 | `quality_threshold_failed` 多 | 保留 dirty raw 证据，进入预处理流程 |

## 最终接入判定

OpenFace 真接入完成的证据必须包括：

- `feature_extraction.sh -help` 成功。
- 10 窗口 OpenFace CSV 和 embedding 成功。
- 单被试 `face_emb.shape == (25, 256)`，NaN 为 0。
- full face OpenFace 产物生成。
- 和 OpenCV fallback 的质量报告对比完成。
- 若质量门槛不通过，明确记录是否进入预处理分支。
- all-real 打包和阶段 18 ablation 使用明确的 face 分支产物。

## 推荐结论

当前最佳方案：

```text
Apptainer OpenFace
-> wrapper 暴露为 --openface-executable
-> 代码修正为窗口级 clip 再跑 FeatureExtraction
-> 10 窗口
-> sub-12 单被试
-> full 781 窗口
-> 质量审计和阶段18下游对照
```

这条路线最符合当前服务器权限，最少污染环境，也能和现有 face pipeline 平滑衔接。
