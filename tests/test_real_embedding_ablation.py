import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from daily_multimodal.training.real_embedding_ablation import run_real_embedding_ablation


class RealEmbeddingAblationTests(unittest.TestCase):
    def test_run_real_embedding_ablation_writes_comparisons_table_metrics_and_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            basic = root / "basic.npz"
            real = root / "real.npz"
            baseline = root / "baseline_reference_metrics.json"
            stage10 = root / "modality_token_fusion_metrics.json"
            out_table = root / "real_embedding_ablation_table.md"
            metrics_out = root / "real_embedding_ablation_metrics.json"
            failures_out = root / "real_embedding_ablation_failures.json"
            _write_split_embeddings(basic, offset=0.0)
            _write_split_embeddings(real, offset=0.5)
            _write_baseline_metrics(baseline, full_rmse=9.0)
            _write_stage10_metrics(stage10, rmse=8.0)

            result = run_real_embedding_ablation(
                basic_embeddings=basic,
                real_embeddings=real,
                baseline_metrics=baseline,
                stage10_metrics=stage10,
                target_label="alert",
                out_table=out_table,
                metrics_out=metrics_out,
                failures_out=failures_out,
                epochs=35,
                overfit_limit=4,
                seeds=[3],
                bootstrap_iterations=25,
            )

            table = out_table.read_text(encoding="utf-8")
            metrics = json.loads(metrics_out.read_text(encoding="utf-8"))
            failures = json.loads(failures_out.read_text(encoding="utf-8"))

        experiments = {row["experiment"] for row in result["experiments"]}
        self.assertIn("baseline_reference_full_concat_mlp", experiments)
        self.assertIn("stage10_modality_token_attention", experiments)
        self.assertIn("audio_real_only_replaced", experiments)
        self.assertIn("all_real_concat_mlp", experiments)
        self.assertIn("all_real_modality_token_attention", experiments)
        self.assertIn("all_real_without_face", experiments)
        self.assertIn("face_raw_openface_stats_v1", experiments)
        self.assertIn("face_preprocessed_openface_stats_v1", experiments)
        self.assertIn("| experiment | embedding_source | model | test_rmse | decision |", table)
        self.assertEqual(metrics["stage"], 18)
        self.assertEqual(metrics["target_label"], "alert")
        self.assertEqual(metrics["split"]["train_subjects"], ["sub-02"])
        self.assertEqual(metrics["face_seed_summary"]["seed_count"], 1)
        self.assertIn("bootstrap_ci95_delta_rmse", metrics["face_seed_summary"])
        self.assertEqual(failures, [])

    def test_incomplete_real_subject_split_records_failure_and_skips_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            basic = root / "basic.npz"
            real = root / "real.npz"
            baseline = root / "baseline_reference_metrics.json"
            out_table = root / "real_embedding_ablation_table.md"
            metrics_out = root / "real_embedding_ablation_metrics.json"
            failures_out = root / "real_embedding_ablation_failures.json"
            _write_split_embeddings(basic, offset=0.0)
            _write_minimal_embeddings(real)
            _write_baseline_metrics(baseline, full_rmse=9.0)

            result = run_real_embedding_ablation(
                basic_embeddings=basic,
                real_embeddings=real,
                baseline_metrics=baseline,
                stage10_metrics=None,
                target_label="alert",
                out_table=out_table,
                metrics_out=metrics_out,
                failures_out=failures_out,
                epochs=5,
                overfit_limit=2,
            )

            failures = json.loads(failures_out.read_text(encoding="utf-8"))

        self.assertEqual(result["experiments"], [])
        self.assertEqual(failures[0]["error_type"], "subject_split_incomplete")
        self.assertIn("real_embeddings", failures[0]["source"])

    def test_cli_runs_real_embedding_ablation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            basic = root / "basic.npz"
            real = root / "real.npz"
            baseline = root / "baseline_reference_metrics.json"
            out_table = root / "table.md"
            metrics_out = root / "metrics.json"
            failures_out = root / "failures.json"
            _write_split_embeddings(basic, offset=0.0)
            _write_split_embeddings(real, offset=0.25)
            _write_baseline_metrics(baseline, full_rmse=9.0)

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/archive_legacy/17_run_real_embedding_ablation.py",
                    "--basic-embeddings",
                    str(basic),
                    "--real-embeddings",
                    str(real),
                    "--baseline",
                    str(baseline),
                    "--target-label",
                    "alert",
                    "--out-table",
                    str(out_table),
                    "--metrics-out",
                    str(metrics_out),
                    "--failures-out",
                    str(failures_out),
                    "--epochs",
                    "20",
                    "--overfit-limit",
                    "4",
                    "--seeds",
                    "5",
                    "--bootstrap-iterations",
                    "10",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("table_path=", completed.stdout)
            self.assertTrue(out_table.is_file())
            self.assertTrue(metrics_out.is_file())
            self.assertTrue(failures_out.is_file())


def _write_baseline_metrics(path: Path, *, full_rmse: float) -> None:
    payload = {
        "stage": 9,
        "target_label": "alert",
        "split": {
            "train_subjects": ["sub-02"],
            "val_subjects": ["sub-11"],
            "test_subjects": ["sub-13"],
        },
        "overfit_check": {"passed": True},
        "runs": {
            "full": {
                "test": {"count": 2, "mae": 8.0, "rmse": full_rmse, "pearson": 0.0}
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_stage10_metrics(path: Path, *, rmse: float) -> None:
    path.write_text(
        json.dumps(
            {
                "stage": 10,
                "upgrade": "modality_token_attention",
                "test": {"count": 2, "mae": rmse - 1.0, "rmse": rmse, "pearson": 0.2},
            }
        ),
        encoding="utf-8",
    )


def _write_minimal_embeddings(path: Path) -> None:
    emb = np.zeros((1, 256), dtype=np.float32)
    np.savez_compressed(
        path,
        sample_id=np.array(["sample-0"], dtype=object),
        event_id=np.array(["event-0"], dtype=object),
        subject_id=np.array(["sub-02"], dtype=object),
        session_id=np.array(["ses-01"], dtype=object),
        eeg_emb=emb,
        wear_emb=emb,
        face_emb=emb,
        audio_emb=emb,
        modality_mask=np.ones((1, 4), dtype=np.int8),
        labels=np.array([json.dumps({"alert": 0.0})], dtype=object),
        source_paths=np.array(["{}"], dtype=object),
    )


def _write_split_embeddings(path: Path, *, offset: float) -> None:
    subjects = np.array(["sub-02"] * 4 + ["sub-11"] * 2 + ["sub-13"] * 2, dtype=object)
    targets = np.arange(len(subjects), dtype=np.float32)
    labels = np.array([json.dumps({"alert": float(value)}) for value in targets], dtype=object)
    base = np.zeros((len(subjects), 256), dtype=np.float32)
    base[:, 0] = targets + offset
    np.savez_compressed(
        path,
        sample_id=np.array([f"sample-{idx}" for idx in range(len(subjects))], dtype=object),
        event_id=np.array([f"event-{idx}" for idx in range(len(subjects))], dtype=object),
        subject_id=subjects,
        session_id=np.array(["ses-01"] * len(subjects), dtype=object),
        eeg_emb=base + 0.1,
        wear_emb=base + 0.2,
        face_emb=base + 0.3,
        audio_emb=base + 0.4,
        modality_mask=np.ones((len(subjects), 4), dtype=np.int8),
        labels=labels,
        source_paths=np.array(["{}"] * len(subjects), dtype=object),
    )


if __name__ == "__main__":
    unittest.main()
