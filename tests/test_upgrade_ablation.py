import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from daily_multimodal.training.upgrade_ablation import run_upgrade_ablation, snapshot_baseline_reference


class UpgradeAblationTests(unittest.TestCase):
    def test_snapshot_baseline_reference_copies_metrics_table_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embeddings = root / "all_complete_basic_embeddings.npz"
            metrics = root / "baseline_mlp_metrics.json"
            table = root / "baseline_mlp_table.md"
            stage8_report = root / "all_complete_basic_embedding_report.json"
            metrics_out = root / "baseline_reference_metrics.json"
            table_out = root / "baseline_reference_table.md"
            manifest_out = root / "baseline_reference_manifest.json"
            _write_minimal_embeddings(embeddings)
            _write_baseline_metrics(metrics)
            table.write_text("| run | test_rmse |\n| --- | ---: |\n| full | 0.8000 |\n", encoding="utf-8")
            stage8_report.write_text(
                json.dumps({"summary": {"success_count": 995, "failure_count": 0}}),
                encoding="utf-8",
            )

            manifest = snapshot_baseline_reference(
                embeddings_path=embeddings,
                baseline_metrics_path=metrics,
                baseline_table_path=table,
                stage8_report_path=stage8_report,
                metrics_out=metrics_out,
                table_out=table_out,
                manifest_out=manifest_out,
                created_at="2026-06-29T12:00:00+08:00",
            )

            copied_metrics = json.loads(metrics_out.read_text(encoding="utf-8"))
            copied_manifest = json.loads(manifest_out.read_text(encoding="utf-8"))

            self.assertEqual(copied_metrics["runs"]["full"]["test"]["rmse"], 0.8)
            self.assertEqual(table_out.read_text(encoding="utf-8"), table.read_text(encoding="utf-8"))
            self.assertEqual(manifest["baseline_overfit_passed"], True)
            self.assertEqual(copied_manifest["stage8_success_count"], 995)
            self.assertEqual(copied_manifest["stage8_failure_count"], 0)
            self.assertEqual(copied_manifest["embeddings_path"], str(embeddings))
            self.assertEqual(copied_manifest["target_label"], "alert")

    def test_cli_snapshot_baseline_writes_default_reference_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embeddings = root / "all_complete_basic_embeddings.npz"
            metrics = root / "baseline_mlp_metrics.json"
            table = root / "baseline_mlp_table.md"
            stage8_report = root / "all_complete_basic_embedding_report.json"
            _write_minimal_embeddings(embeddings)
            _write_baseline_metrics(metrics)
            table.write_text("| run | test_rmse |\n| --- | ---: |\n| full | 0.8000 |\n", encoding="utf-8")
            stage8_report.write_text(
                json.dumps({"summary": {"success_count": 995, "failure_count": 0}}),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/10_run_upgrade_ablation.py",
                    "--embeddings",
                    str(embeddings),
                    "--baseline",
                    str(metrics),
                    "--baseline-table",
                    str(table),
                    "--stage8-report",
                    str(stage8_report),
                    "--snapshot-baseline",
                    "--reference-metrics-out",
                    str(root / "baseline_reference_metrics.json"),
                    "--reference-table-out",
                    str(root / "baseline_reference_table.md"),
                    "--reference-manifest-out",
                    str(root / "baseline_reference_manifest.json"),
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("baseline reference snapshot saved", completed.stdout)
            self.assertTrue((root / "baseline_reference_metrics.json").exists())
            self.assertTrue((root / "baseline_reference_table.md").exists())
            self.assertTrue((root / "baseline_reference_manifest.json").exists())

    def test_registry_smoke_writes_ablation_table_and_empty_failures_for_complete_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embeddings = root / "embeddings.npz"
            baseline = root / "baseline_reference_metrics.json"
            out_table = root / "model_upgrade_ablation_table.md"
            failures_out = root / "model_upgrade_failures.json"
            _write_split_embeddings(embeddings)
            _write_baseline_metrics(baseline)

            result = run_upgrade_ablation(
                embeddings_path=embeddings,
                baseline_metrics_path=baseline,
                upgrade="registry_smoke",
                target_label="alert",
                out_table=out_table,
                failures_out=failures_out,
            )

            table = out_table.read_text(encoding="utf-8")
            failures = json.loads(failures_out.read_text(encoding="utf-8"))

            self.assertEqual(result["decision"], "rollback")
            self.assertEqual(result["baseline_metric"], 0.8)
            self.assertIn("| registry_smoke | framework | 0.8000 | 0.8000 | 0.0000 | rollback |", table)
            self.assertEqual(failures, [])

    def test_registry_smoke_records_failure_when_subject_split_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embeddings = root / "embeddings.npz"
            baseline = root / "baseline_reference_metrics.json"
            out_table = root / "model_upgrade_ablation_table.md"
            failures_out = root / "model_upgrade_failures.json"
            _write_minimal_embeddings(embeddings)
            _write_baseline_metrics(baseline)

            result = run_upgrade_ablation(
                embeddings_path=embeddings,
                baseline_metrics_path=baseline,
                upgrade="registry_smoke",
                target_label="alert",
                out_table=out_table,
                failures_out=failures_out,
            )

            failures = json.loads(failures_out.read_text(encoding="utf-8"))

            self.assertEqual(result["decision"], "rollback")
            self.assertEqual(failures[0]["error_type"], "subject_split_incomplete")
            self.assertIn("val", failures[0]["error"])
            self.assertIn("test", failures[0]["error"])

    def test_modality_token_attention_writes_metrics_model_and_ablation_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embeddings = root / "embeddings.npz"
            baseline = root / "baseline_reference_metrics.json"
            out_table = root / "model_upgrade_ablation_table.md"
            failures_out = root / "model_upgrade_failures.json"
            metrics_out = root / "modality_token_fusion_metrics.json"
            model_out = root / "modality_token_fusion.pt"
            _write_split_embeddings(embeddings)
            _write_baseline_metrics(baseline, full_rmse=999.0)

            result = run_upgrade_ablation(
                embeddings_path=embeddings,
                baseline_metrics_path=baseline,
                upgrade="modality_token_attention",
                target_label="alert",
                out_table=out_table,
                failures_out=failures_out,
                metrics_out=metrics_out,
                model_out=model_out,
                epochs=40,
                overfit_limit=4,
            )

            metrics = json.loads(metrics_out.read_text(encoding="utf-8"))
            table = out_table.read_text(encoding="utf-8")

            self.assertTrue(model_out.exists())
            self.assertEqual(metrics["upgrade"], "modality_token_attention")
            self.assertIn(result["decision"], {"accepted", "rollback"})
            self.assertIsInstance(result["upgrade_metric"], float)
            self.assertIn("| modality_token_attention | fusion |", table)


def _write_baseline_metrics(path: Path, *, full_rmse: float = 0.8) -> None:
    path.write_text(
        json.dumps(
            {
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
                        "modalities": ["eeg", "wear", "audio", "face"],
                        "test": {"count": 4, "mae": 0.7, "rmse": full_rmse, "pearson": 0.1},
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _write_minimal_embeddings(path: Path) -> None:
    np.savez_compressed(
        path,
        sample_id=np.array(["sample-0001"], dtype=object),
        event_id=np.array(["event-0001"], dtype=object),
        subject_id=np.array(["sub-02"], dtype=object),
        session_id=np.array(["ses-01"], dtype=object),
        eeg_emb=np.zeros((1, 256), dtype=np.float32),
        wear_emb=np.zeros((1, 256), dtype=np.float32),
        face_emb=np.zeros((1, 256), dtype=np.float32),
        audio_emb=np.zeros((1, 256), dtype=np.float32),
        modality_mask=np.ones((1, 4), dtype=np.int8),
        labels=np.array([json.dumps({"alert": 1.0})], dtype=object),
        source_paths=np.array(["{}"], dtype=object),
    )


def _write_split_embeddings(path: Path) -> None:
    subjects = np.array(["sub-02"] * 2 + ["sub-11"] * 2 + ["sub-13"] * 2, dtype=object)
    labels = np.array([json.dumps({"alert": float(idx)}) for idx in range(len(subjects))], dtype=object)
    emb = np.zeros((len(subjects), 256), dtype=np.float32)
    emb[:, 0] = np.arange(len(subjects), dtype=np.float32)
    np.savez_compressed(
        path,
        sample_id=np.array([f"sample-{idx:04d}" for idx in range(len(subjects))], dtype=object),
        event_id=np.array([f"event-{idx:04d}" for idx in range(len(subjects))], dtype=object),
        subject_id=subjects,
        session_id=np.array(["ses-01"] * len(subjects), dtype=object),
        eeg_emb=emb,
        wear_emb=emb,
        face_emb=emb,
        audio_emb=emb,
        modality_mask=np.ones((len(subjects), 4), dtype=np.int8),
        labels=labels,
        source_paths=np.array(["{}"] * len(subjects), dtype=object),
    )


if __name__ == "__main__":
    unittest.main()
