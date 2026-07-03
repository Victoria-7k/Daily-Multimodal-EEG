# Wear Quality Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand wear full-window quality reporting so the 8328 face-filtered windows expose rows, rates, invalid/source counts, timestamp anomalies, flatline, PPG peaks/heart-rate plausibility, GSR anomalies, and ACC motion summaries.

**Architecture:** Keep the existing `wear_physio_features_v2` embedding path and add auditable fields to `quality_flags` plus a structured `quality_audit` block in the summary. The embedding shape, mask semantics, and failures contract stay unchanged.

**Tech Stack:** Python, NumPy, unittest/pytest, existing Daily Multimodal wear real embedding module.

---

### Task 1: Add Failing Quality Audit Tests

**Files:**
- Modify: `tests/test_wear_real_embedding.py`

- [x] **Step 1: Write the failing test**

Add a test that extracts two `wear_physio_features_v2` windows, one plausible and one flat/implausible, then asserts `summary["quality_audit"]` contains row/rate/invalid/source/timestamp/flatline/PPG/GSR/ACC rollups.

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_wear_real_embedding.py::WearRealEmbeddingTests::test_wear_quality_audit_summary_reports_requested_fields -q`

Expected: FAIL because `quality_audit` and flatline/plausibility fields do not exist yet.

### Task 2: Implement Per-Window Quality Flags

**Files:**
- Modify: `src/daily_multimodal/embeddings/wear_real.py`

- [x] **Step 1: Add flatline ratio and plausibility fields**

Compute per-modality flatline ratios from resampled sequences. For PPG v2, keep `peak_count` and `heart_rate`, and add `heart_rate_plausible` using 40-180 bpm with nonzero peak support.

- [x] **Step 2: Run focused tests**

Run: `python -m pytest tests/test_wear_real_embedding.py::WearRealEmbeddingTests::test_wear_quality_audit_summary_reports_requested_fields -q`

Expected: test progresses to summary aggregation assertions.

### Task 3: Implement Summary Aggregation

**Files:**
- Modify: `src/daily_multimodal/embeddings/wear_real.py`

- [x] **Step 1: Add `quality_audit` rollup**

Aggregate each requested field across emitted samples and include failure totals. Do not mark samples unusable based on quality warnings.

- [x] **Step 2: Run wear tests**

Run: `python -m pytest tests/test_wear_real_embedding.py -q`

Expected: all wear tests pass.

### Task 4: Sync Docs and Verify

**Files:**
- Modify: `repo-docs/references/data-contracts.md`
- Modify: `repo-docs/references/commands-and-artifacts.md`
- Modify: `repo-docs/change-log.md`

- [x] **Step 1: Document new fields and full-run command**

Record the expanded quality audit fields and the server full command against `real_cache_face_detected_full_v2_mainface.jsonl`.

- [x] **Step 2: Run local verification**

Run: `python -m pytest tests/test_wear_real_embedding.py -q`

Run: `python -m compileall -q src scripts tests`

- [x] **Step 3: Run server full verification**

Run `scripts/15_extract_wear_embeddings.py` on the 8328 face-filtered index with `wear_physio_features_v2`, then inspect the summary for `quality_audit.window_count=8328`.
