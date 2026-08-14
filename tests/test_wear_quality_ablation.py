import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from daily_multimodal.training.wear_quality_ablation import run_wear_quality_ablation


class WearQualityAblationTests(unittest.TestCase):
    def test_run_wear_quality_ablation_reports_wear_only_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            window_index = root / "windows.jsonl"
            physio = root / "physio.npz"
            deep = root / "deep.npz"
            out_json = root / "wear_ablation.json"
            out_table = root / "wear_ablation.md"
            _write_window_index(window_index)
            _write_wear_npz(physio, encoder="wear_physio_features_v2", offset=0.2)
            _write_wear_npz(deep, encoder="wear_deep_sequence_v1", offset=0.5)

            result = run_wear_quality_ablation(
                window_index=window_index,
                physio_embeddings=physio,
                deep_embeddings=deep,
                target_label="fatigue",
                out_json=out_json,
                out_table=out_table,
                epochs=10,
                hidden_dim=8,
            )

            metrics = json.loads(out_json.read_text(encoding="utf-8"))
            table = out_table.read_text(encoding="utf-8")

        self.assertEqual(
            list(result["experiments"]),
            [
                "W1_physio_full",
                "W2_deep_full",
                "W3_physio_high_quality",
                "W4_deep_high_quality",
                "W5a_deep_full",
                "W5b_deep_quality_flags_full",
                "W5c_deep_sample_weights_full",
                "W5d_deep_quality_flags_sample_weights_full",
                "W6_physio_ab_quality",
                "W7_deep_ab_quality",
            ],
        )
        self.assertEqual(result["target_label"], "fatigue")
        self.assertEqual(metrics["experiments"]["W1_physio_full"]["feature_set"], "wear_physio_features_v2")
        self.assertEqual(metrics["experiments"]["W3_physio_high_quality"]["quality_subset"], "A")
        self.assertEqual(metrics["experiments"]["W6_physio_ab_quality"]["quality_subset"], "A+B")
        self.assertEqual(metrics["experiments"]["W7_deep_ab_quality"]["quality_subset"], "A+B")
        self.assertFalse(metrics["experiments"]["W5a_deep_full"]["include_quality_flags"])
        self.assertFalse(metrics["experiments"]["W5a_deep_full"]["use_sample_weight"])
        self.assertTrue(metrics["experiments"]["W5b_deep_quality_flags_full"]["include_quality_flags"])
        self.assertFalse(metrics["experiments"]["W5b_deep_quality_flags_full"]["use_sample_weight"])
        self.assertFalse(metrics["experiments"]["W5c_deep_sample_weights_full"]["include_quality_flags"])
        self.assertTrue(metrics["experiments"]["W5c_deep_sample_weights_full"]["use_sample_weight"])
        self.assertTrue(metrics["experiments"]["W5d_deep_quality_flags_sample_weights_full"]["include_quality_flags"])
        self.assertTrue(metrics["experiments"]["W5d_deep_quality_flags_sample_weights_full"]["use_sample_weight"])
        for experiment in metrics["experiments"].values():
            self.assertIn("rmse_mean", experiment)
            self.assertIn("pearson_r_mean", experiment)
            self.assertIn("pred_std_mean", experiment)
            self.assertIn("truth_std_mean", experiment)
            self.assertIn("error_std_mean", experiment)
            self.assertIn("sample_weight_mean", experiment)
        self.assertIn("| experiment |", table)
        self.assertIn("pred_std", table)

    def test_cli_runs_wear_quality_ablation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            window_index = root / "windows.jsonl"
            physio = root / "physio.npz"
            deep = root / "deep.npz"
            out_json = root / "wear_ablation.json"
            out_table = root / "wear_ablation.md"
            _write_window_index(window_index)
            _write_wear_npz(physio, encoder="wear_physio_features_v2", offset=0.2)
            _write_wear_npz(deep, encoder="wear_deep_sequence_v1", offset=0.5)

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/archive_legacy/22_run_wear_quality_ablation.py",
                    "--window-index",
                    str(window_index),
                    "--physio-embeddings",
                    str(physio),
                    "--deep-embeddings",
                    str(deep),
                    "--target-label",
                    "fatigue",
                    "--out-json",
                    str(out_json),
                    "--out-table",
                    str(out_table),
                    "--epochs",
                    "10",
                    "--hidden-dim",
                    "8",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("experiment_count=10", completed.stdout)
            self.assertTrue(out_json.is_file())


def _write_window_index(path: Path) -> None:
    subjects = [f"sub-{idx:02d}" for idx in range(1, 7)]
    rows = []
    for subject_index, subject in enumerate(subjects):
        for sample_index in range(4):
            absolute_index = subject_index * 4 + sample_index
            rows.append(
                {
                    "sample_id": f"sample-{absolute_index:02d}",
                    "event_id": f"event-{absolute_index:02d}",
                    "subject_id": subject,
                    "label_columns": {"fatigue": float(absolute_index % 5) / 4.0},
                }
            )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_wear_npz(path: Path, *, encoder: str, offset: float) -> None:
    sample_ids = np.array([f"sample-{idx:02d}" for idx in range(24)], dtype=object)
    subject_ids = np.array([f"sub-{idx:02d}" for idx in range(1, 7) for _ in range(4)], dtype=object)
    base = np.zeros((24, 256), dtype=np.float32)
    base[:, 0] = np.linspace(0.0, 1.0, 24, dtype=np.float32) + offset
    quality_flags = []
    for idx in range(24):
        high = idx % 2 == 0
        quality_flags.append(
            json.dumps(
                {
                    "wear_quality_grade": "A" if high else "B",
                    "motion_intensity": 0.05 if high else 0.4,
                    "stationary_ratio": 0.8 if high else 0.1,
                    "heart_rate_plausible": high,
                    "ppg_hr_plausible": high,
                    "ppg_peak_insufficient": not high,
                    "gsr_slope_abnormal": False,
                    "gsr_scr_abnormal": not high,
                    "acc_motion_high": not high,
                    "wear_quality_risk_count": 0 if high else 2,
                }
            )
        )
    np.savez_compressed(
        path,
        sample_id=sample_ids,
        event_id=np.array([f"event-{idx:02d}" for idx in range(24)], dtype=object),
        subject_id=subject_ids,
        wear_emb=base,
        modality_mask=np.array([[0, 1, 0, 0]] * 24, dtype=np.int8),
        quality_flags=np.array(quality_flags, dtype=object),
        encoder_version=np.array([encoder] * 24, dtype=object),
    )


if __name__ == "__main__":
    unittest.main()
