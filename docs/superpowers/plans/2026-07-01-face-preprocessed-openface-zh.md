# Face 预处理 OpenFace 实施计划

> **给 agentic worker 的要求：** 实施本计划时必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，逐任务执行并用复选框跟踪进度。

**目标：** 新增独立的 `face_preprocessed_openface_stats_v1` 分支，在不覆盖 `face_raw_openface_stats_v1` 的前提下，通过人脸预检测、主脸 track、ROI clip 和 OpenFace retry 提高 Face 可用窗口数。

**架构：** raw 分支继续作为默认分支和对照基线。预处理分支单独采样帧、尝试 `0/90/180/270` 四个方向、选择稳定主脸、写出可复现 ROI clip，再复用现有 OpenFace CSV 统计和 `(N, 256)` embedding 契约。所有预处理决策必须写入 `quality_flags` 和报告，方便下游 ablation 决定是否把预处理分支设为默认。

**技术栈：** Python、NumPy、OpenCV、ffmpeg/ffprobe、现有 OpenFace Apptainer wrapper、现有 `.npz`/JSON 失败清单契约、`unittest`/pytest-compatible tests。

---

## 验收标准

### 必须满足的正确性

- raw 分支不变：运行 `face_raw_openface_stats_v1` 时不得写入预处理缓存、不得改变 raw cache 路径、不得改变 raw 质量语义。
- 预处理分支使用独立 profile/cache：`face_preprocessed_openface_stats_v1`。
- 禁止生成式人脸修复、强去模糊、任何可能改变表情语义的增强。
- 允许的增强只限 ROI luminance 上的 CLAHE 或有界 gamma correction。
- 每个样本保持原始 `sample_id`、`event_id`、`subject_id`。
- 输出 embedding shape 必须是 `(N, 256)`，NaN count 必须为 `0`。
- 低质量 ROI 结果保留样本行，写零向量并置 `modality_mask[:, 2] = 0`，不能静默丢弃。
- `Starting tracking` 后无 CSV 的窗口最多做一次预处理 ROI retry，并设置超时；retry 失败后仍写结构化失败。

### Full-run 质量门槛

以当前已修复 raw OpenFace full 结果为基线：

- raw full baseline：`selected_windows=781`、`face_success_count=207`、`extraction_failed=73`、`masked_count=501`。
- 预处理 full run 必须仍覆盖 `selected_windows=781`。
- 预处理 Face 有效窗口数至少 `249`，即相对 `207` 提升 20%。
- 预处理 `extraction_failed` 至多 `36`，即相对 `73` 降低 50%。
- `openface_abort_after_starting_tracking_no_csv` 至多 `10`，除非视觉抽样证明剩余多数窗口没有可恢复人脸。
- NaN count 必须为 `0`。
- 必须报告 `main_face_ambiguity_ratio`；超过歧义阈值的窗口必须 mask，不能强行选主脸。

### 默认分支晋升门槛

只有同时满足以下下游条件，预处理分支才允许替代 raw 成为默认 Face 分支：

- `all_real_with_preprocessed_face` 的 5 seed median test RMSE 至少比 `all_real_with_raw_face` 低 2%。
- MAE 不恶化超过 1%。
- Pearson r 不下降超过 `0.02`。
- 至少 4/5 个 seed 的主指标方向一致有利。
- bootstrap 95% CI 不显示稳定负向影响。

如果质量指标提升但下游门槛未通过，则保留预处理产物用于分析，默认训练仍使用 raw 分支。

---

## 文件职责

- 修改 `src/daily_multimodal/embeddings/face_real.py`
  - 接入预处理分支，但保持 raw 行为不变。
  - 对 `face_preprocessed_openface_stats_v1` 在 OpenFace 前准备 ROI clip。
  - 对 `Starting tracking` 后无 CSV 的失败加入一次有界 ROI retry。

- 新建 `src/daily_multimodal/embeddings/face_preprocessing.py`
  - 按可配置 FPS 抽样帧。
  - 尝试 `0/90/180/270` 四个方向。
  - 通过 detector backend protocol 跑人脸检测。
  - 选择主脸 track。
  - 写 ROI clip 和预处理 metadata。

- 修改 `scripts/13_extract_face_embeddings.py`
  - 增加显式预处理 CLI 参数，同时保留按 profile 自动启用。

- 新建 `scripts/22_compare_face_preprocessing.py`
  - 对比 raw 和 preprocessed 的质量摘要、失败清单、NPZ mask。
  - 输出 JSON 验收报告和 Markdown 表格。

- 修改 `tests/test_face_real_embedding.py`
  - 覆盖 raw 分支不变、预处理分支使用 ROI clip、无 CSV retry。

- 新建 `tests/test_face_preprocessing.py`
  - 覆盖抽帧、旋转选择、主脸 track、歧义、多边界 crop、metadata。

- 新建 `tests/test_face_preprocessing_comparison.py`
  - 覆盖质量验收阈值计算。

- 更新 `repo-docs/walkthroughs/one-real-run.md`
  - 解释 raw 与 preprocessed 分支行为。

- 更新 `repo-docs/references/commands-and-artifacts.md`
  - 记录新命令和新产物。

- 更新 `repo-docs/change-log.md`
  - 记录实现、验证、同步状态。

---

## Task 1：预处理数据模型与主脸 track 选择

**文件：**
- 新建 `src/daily_multimodal/embeddings/face_preprocessing.py`
- 新建 `tests/test_face_preprocessing.py`

- [ ] **Step 1：先写失败测试**

覆盖三类行为：

- 连续、居中、高置信度的人脸应成为主脸。
- 多个相似竞争人脸应标记 `ambiguous=True`。
- ROI crop 扩展后必须 clamp 到画面边界内。

测试命令：

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_face_preprocessing -v
```

预期：失败，因为 `daily_multimodal.embeddings.face_preprocessing` 尚不存在。

- [ ] **Step 2：实现最小数据模型**

实现：

- `FaceDetection`
- `FaceTrack`
- `expand_crop_box(...)`
- `choose_main_face_track(...)`

主脸排序规则：

1. 覆盖帧数最多。
2. 平均 confidence 更高。
3. 人脸面积更大。
4. 更靠近画面中心。

- [ ] **Step 3：验证**

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_face_preprocessing -v
```

预期：Task 1 的测试全部通过。

- [ ] **Step 4：提交**

```powershell
git add src/daily_multimodal/embeddings/face_preprocessing.py tests/test_face_preprocessing.py
git commit -m "Add face preprocessing track selection"
```

---

## Task 2：ROI clip metadata 与可复现记录

**文件：**
- 修改 `src/daily_multimodal/embeddings/face_preprocessing.py`
- 修改 `tests/test_face_preprocessing.py`

- [ ] **Step 1：写失败测试**

新增测试：`write_preprocessed_metadata(...)` 必须写出：

- `sample_id`
- `source_clip`
- `roi_clip`
- `detector_backend`
- `sample_fps`
- `roi_margin`
- `roi_size`
- `rotation`
- `crop_box`
- `detection_count`
- `sampled_frame_count`
- `mean_confidence`
- `main_face_ambiguity_ratio`
- `main_face_ambiguous`

运行：

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_face_preprocessing.FacePreprocessingMetadataTests -v
```

预期：失败，因为 `FacePreprocessingConfig` 和 metadata writer 尚不存在。

- [ ] **Step 2：实现配置和 metadata writer**

新增：

- `FacePreprocessingConfig(sample_fps=2.0, roi_margin=1.6, roi_size=384, min_track_detection_rate=0.30, ambiguity_threshold=0.50)`
- `write_preprocessed_metadata(...)`

- [ ] **Step 3：验证**

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_face_preprocessing -v
```

- [ ] **Step 4：提交**

```powershell
git add src/daily_multimodal/embeddings/face_preprocessing.py tests/test_face_preprocessing.py
git commit -m "Record face preprocessing metadata"
```

---

## Task 3：接入 `face_preprocessed_openface_stats_v1`

**文件：**
- 修改 `src/daily_multimodal/embeddings/face_real.py`
- 修改 `tests/test_face_real_embedding.py`

- [ ] **Step 1：写失败集成测试**

新增测试：当 `encoder_profile="face_preprocessed_openface_stats_v1"` 时：

- 先生成原始 `window.mp4`。
- 调用 `preprocess_clip(...)` 生成 `window_preprocessed.mp4`。
- OpenFace runner 必须读取 ROI clip，而不是原始 `window.mp4`。
- `quality_flags` 必须包含：
  - `face_preprocessing_used=True`
  - `face_preprocessing_profile`
  - `detector_backend`
  - `rotation`
  - `crop_box`
  - `main_face_ambiguity_ratio`

运行：

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_face_real_embedding.FaceRealEmbeddingTests.test_preprocessed_profile_runs_openface_on_roi_clip_and_records_quality_flags
```

预期：失败，因为 `extract_face_real_embeddings()` 尚无 `preprocess_clip` 参数。

- [ ] **Step 2：实现 profile hook**

在 `face_real.py` 增加：

- `PreprocessClip = Callable[[dict[str, Any], dict[str, Any], Path], tuple[Path, dict[str, Any]]]`
- `extract_face_real_embeddings(..., preprocess_clip: PreprocessClip | None = None, ...)`
- 当 profile 为 `face_preprocessed_openface_stats_v1` 时，调用 `_prepare_preprocessed_face_clip(...)`。
- 将预处理 metadata merge 到 `quality_flags`。

- [ ] **Step 3：验证**

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_face_real_embedding tests.test_face_preprocessing -v
```

- [ ] **Step 4：提交**

```powershell
git add src/daily_multimodal/embeddings/face_real.py tests/test_face_real_embedding.py
git commit -m "Add preprocessed OpenFace profile"
```

---

## Task 4：`Starting tracking` 后无 CSV 的 ROI retry

**文件：**
- 修改 `src/daily_multimodal/embeddings/face_real.py`
- 修改 `tests/test_face_real_embedding.py`

- [ ] **Step 1：写失败测试**

新增测试：OpenFace 第一次对 `window.mp4` 抛出：

```text
Device or file opened
Starting tracking
```

当 `retry_preprocessed_on_openface_abort=True` 时，系统应：

1. 调用 `preprocess_clip(...)` 生成 `window_preprocessed_retry.mp4`。
2. 对 ROI clip 再跑一次 OpenFace。
3. 第二次成功后写出 CSV 和 embedding。
4. 只 retry 一次。

运行：

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_face_real_embedding.FaceRealEmbeddingTests.test_openface_starting_tracking_no_csv_retries_preprocessed_roi_once
```

预期：失败，因为 retry flag 尚不存在。

- [ ] **Step 2：实现有界 retry**

新增参数：

```python
retry_preprocessed_on_openface_abort: bool = False
```

新增判断函数：

```python
def _is_openface_starting_tracking_abort(message: str) -> bool:
    return "Starting tracking" in message
```

行为：

- 只对 `Starting tracking` 后无 CSV 的 OpenFace 失败 retry。
- 只 retry 一次。
- retry 失败后写结构化 failure。
- retry 成功后在 `quality_flags` 写 `openface_preprocessed_retry_used=True`。

- [ ] **Step 3：验证**

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_face_real_embedding -v
```

- [ ] **Step 4：提交**

```powershell
git add src/daily_multimodal/embeddings/face_real.py tests/test_face_real_embedding.py
git commit -m "Retry OpenFace aborts with preprocessed ROI"
```

---

## Task 5：CLI 参数与可复现命令

**文件：**
- 修改 `scripts/13_extract_face_embeddings.py`
- 修改 `tests/test_face_real_embedding.py`
- 修改 `repo-docs/references/commands-and-artifacts.md`

- [ ] **Step 1：写失败 CLI 测试**

新增 subprocess 测试，调用：

```powershell
python scripts/13_extract_face_embeddings.py `
  --window-index <tmp>\window_index.jsonl `
  --cache-root <tmp>\cache `
  --encoder-profile face_preprocessed_openface_stats_v1 `
  --preprocess-face `
  --face-preprocess-fps 2.0 `
  --face-roi-margin 1.6 `
  --face-roi-size 384 `
  --out <tmp>\face_preprocessed.npz `
  --failures-out <tmp>\failures.json `
  --summary-out <tmp>\summary.json
```

预期：实现前 CLI 报 unknown args。

- [ ] **Step 2：新增 CLI 参数**

在 `scripts/13_extract_face_embeddings.py` 添加：

```python
parser.add_argument("--preprocess-face", action="store_true")
parser.add_argument("--face-detector-backend", default="opencv_haar")
parser.add_argument("--face-preprocess-fps", type=float, default=2.0)
parser.add_argument("--face-roi-margin", type=float, default=1.6)
parser.add_argument("--face-roi-size", type=int, default=384)
parser.add_argument("--retry-preprocessed-on-openface-abort", action="store_true")
```

自动启用规则：

```python
preprocess_face = args.preprocess_face or args.encoder_profile == "face_preprocessed_openface_stats_v1"
```

- [ ] **Step 3：记录 smoke 命令**

```bash
PYTHONPATH=src python scripts/13_extract_face_embeddings.py \
  --window-index outputs/window_index/real_cache_complete_10.jsonl \
  --cache-root outputs/cache/real_stage12_face_preprocessed_10 \
  --encoder-profile face_preprocessed_openface_stats_v1 \
  --openface-executable /mnt/dataset4/sitian/wzw/tools/openface/feature_extraction.sh \
  --preprocess-face \
  --face-detector-backend opencv_haar \
  --face-preprocess-fps 2.0 \
  --face-roi-margin 1.6 \
  --face-roi-size 384 \
  --retry-preprocessed-on-openface-abort \
  --out outputs/embeddings/face_preprocessed_openface_10_embeddings.npz \
  --failures-out outputs/reports/face_preprocessed_openface_10_failures.json \
  --summary-out outputs/reports/face_preprocessed_openface_10_quality_summary.json \
  --decision-out outputs/reports/face_preprocessing_decision_preprocessed_10.json
```

- [ ] **Step 4：验证**

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_face_real_embedding -v
```

- [ ] **Step 5：提交**

```powershell
git add scripts/13_extract_face_embeddings.py tests/test_face_real_embedding.py repo-docs/references/commands-and-artifacts.md
git commit -m "Expose face preprocessing CLI controls"
```

---

## Task 6：raw vs preprocessed 质量对比报告

**文件：**
- 新建 `scripts/22_compare_face_preprocessing.py`
- 新建 `tests/test_face_preprocessing_comparison.py`
- 修改 `repo-docs/references/commands-and-artifacts.md`

- [ ] **Step 1：写失败测试**

测试 `compare_quality(raw, preprocessed)`：

输入：

```python
raw = {"success_count": 207, "failure_types": {"extraction_failed": 73}, "nan_count": 0}
pre = {"success_count": 249, "failure_types": {"extraction_failed": 36}, "nan_count": 0}
```

期望：

- `quality_gate_passed=True`
- `minimum_success_count=249`
- `maximum_extraction_failed=36`

运行：

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_face_preprocessing_comparison -v
```

预期：失败，因为脚本尚不存在。

- [ ] **Step 2：实现 comparison utility**

核心规则：

- `minimum_success_count = ceil(raw_success_count * 1.20)`
- `maximum_extraction_failed = floor(raw_extraction_failed * 0.50)`
- `nan_count` 必须为 0。

CLI 参数：

```text
--raw-summary
--preprocessed-summary
--out-json
--out-md
```

- [ ] **Step 3：记录 comparison 命令**

```bash
PYTHONPATH=src python scripts/22_compare_face_preprocessing.py \
  --raw-summary outputs/reports/face_openface_real_full_quality_summary.json \
  --preprocessed-summary outputs/reports/face_preprocessed_openface_full_quality_summary.json \
  --out-json outputs/reports/face_preprocessing_acceptance.json \
  --out-md outputs/reports/face_preprocessing_acceptance.md
```

- [ ] **Step 4：验证**

```powershell
$env:PYTHONPATH='src'; python -m unittest tests.test_face_preprocessing_comparison -v
python -m compileall -q scripts/22_compare_face_preprocessing.py
```

- [ ] **Step 5：提交**

```powershell
git add scripts/22_compare_face_preprocessing.py tests/test_face_preprocessing_comparison.py repo-docs/references/commands-and-artifacts.md
git commit -m "Compare raw and preprocessed face quality"
```

---

## Task 7：服务器 smoke、full run 与下游 gate

**文件：**
- 修改 `repo-docs/change-log.md`
- 修改 `repo-docs/walkthroughs/one-real-run.md`

- [ ] **Step 1：服务器 10-window smoke**

```bash
ssh ncc_serve_4090 "cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding && source /home/lzs/miniconda3/etc/profile.d/conda.sh && conda activate lzs && PYTHONPATH=src python scripts/13_extract_face_embeddings.py --window-index outputs/window_index/real_cache_complete_10.jsonl --cache-root outputs/cache/real_stage12_face_preprocessed_10 --encoder-profile face_preprocessed_openface_stats_v1 --openface-executable /mnt/dataset4/sitian/wzw/tools/openface/feature_extraction.sh --preprocess-face --face-preprocess-fps 2.0 --face-roi-margin 1.6 --face-roi-size 384 --retry-preprocessed-on-openface-abort --out outputs/embeddings/face_preprocessed_openface_10_embeddings.npz --failures-out outputs/reports/face_preprocessed_openface_10_failures.json --summary-out outputs/reports/face_preprocessed_openface_10_quality_summary.json --decision-out outputs/reports/face_preprocessing_decision_preprocessed_10.json"
```

验收：

- `.npz` 存在。
- `nan_count=0`。
- 每个 processed row 的 `quality_flags` 含预处理字段。

- [ ] **Step 2：服务器 full preprocessed face extraction**

```bash
ssh ncc_serve_4090 "cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding && source /home/lzs/miniconda3/etc/profile.d/conda.sh && conda activate lzs && PYTHONPATH=src python scripts/13_extract_face_embeddings.py --window-index outputs/window_index/real_cache_complete_full.jsonl --cache-root outputs/cache/real_stage12_face_preprocessed_full --encoder-profile face_preprocessed_openface_stats_v1 --openface-executable /mnt/dataset4/sitian/wzw/tools/openface/feature_extraction.sh --preprocess-face --face-preprocess-fps 2.0 --face-roi-margin 1.6 --face-roi-size 384 --retry-preprocessed-on-openface-abort --out outputs/embeddings/face_preprocessed_openface_full_embeddings.npz --failures-out outputs/reports/face_preprocessed_openface_full_failures.json --summary-out outputs/reports/face_preprocessed_openface_full_quality_summary.json --decision-out outputs/reports/face_preprocessing_decision_preprocessed_full.json"
```

full-run 质量 gate：

- `success_count >= 249`
- `failure_types.extraction_failed <= 36`
- `nan_count == 0`

- [ ] **Step 3：比较 raw 和 preprocessed**

```bash
ssh ncc_serve_4090 "cd /mnt/dataset4/sitian/wzw/DailyMultimodalEmbedding && PYTHONPATH=src python scripts/22_compare_face_preprocessing.py --raw-summary outputs/reports/face_openface_real_full_quality_summary.json --preprocessed-summary outputs/reports/face_preprocessed_openface_full_quality_summary.json --out-json outputs/reports/face_preprocessing_acceptance.json --out-md outputs/reports/face_preprocessing_acceptance.md"
```

验收：

- `quality_gate_passed=true`
- 报告列出 raw/preprocessed 成功数、失败数和阈值。

- [ ] **Step 4：用 preprocessed face 重新打包 all-real**

只替换 Face 输入：

```bash
--face outputs/embeddings/face_preprocessed_openface_full_embeddings.npz
--out outputs/embeddings/all_complete_real_v2_preprocessed_face_embeddings.npz
--report-out outputs/reports/all_complete_real_v2_preprocessed_face_embedding_report.json
```

验收：

- row count `781`
- 四模态 shape 都是 `(781, 256)`
- NaN count `0`
- face mask sum 等于 preprocessed success count

- [ ] **Step 5：下游 5-seed 对照**

在相同 `sample_id` 和 `fatigue` target 上比较 raw/preprocessed face，记录：

- median RMSE
- MAE
- Pearson r
- 每个 seed 的方向
- bootstrap 95% CI

只有通过“默认分支晋升门槛”才允许把 preprocessed 设为默认。

- [ ] **Step 6：更新 repo-docs**

更新：

- `repo-docs/walkthroughs/one-real-run.md`
- `repo-docs/references/commands-and-artifacts.md`
- `repo-docs/change-log.md`

必须记录：

- 执行命令
- `success_count`
- `extraction_failed`
- mask sum
- 下游结果
- raw 是否仍为默认，或 preprocessed 是否晋升默认

- [ ] **Step 7：最终本地验证**

```powershell
$env:PYTHONPATH='src'; python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
python C:\Users\28303\.codex\skills\repo-docs\scripts\validate_repo_docs.py repo-docs --repo-root .
```

验收：

- unittest 全部通过。
- compileall exit code 为 `0`。
- repo-docs validator 为 `OK: 0 errors`。

- [ ] **Step 8：提交**

```powershell
git add src scripts tests repo-docs
git commit -m "Add preprocessed OpenFace face branch"
```

---

## 自审

- 需求覆盖：raw 分支保留、预检测、四方向、主脸 track、ROI crop、轻量增强边界、短缺帧策略、no-CSV retry、质量验收和下游晋升门槛均已覆盖。
- 占位检查：没有使用“后续实现”作为任务终点；每个任务都有文件、命令和预期结果。
- 命名一致性：预处理 profile 统一为 `face_preprocessed_openface_stats_v1`；raw profile 统一为 `face_raw_openface_stats_v1`；quality flags 使用 `face_preprocessing_*` 和 `openface_*_retry` 命名。

