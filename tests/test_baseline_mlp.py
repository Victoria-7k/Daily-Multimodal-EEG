import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from daily_multimodal.training.baseline_mlp import run_baseline_experiment


class BaselineMlpTests(unittest.TestCase):
    def test_run_baseline_experiment_uses_subject_split_and_writes_metrics_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embeddings = root / "embeddings.npz"
            model_out = root / "baseline_mlp.pt"
            metrics_out = root / "baseline_mlp_metrics.json"
            table_out = root / "baseline_mlp_table.md"
            _write_synthetic_embeddings(embeddings)

            result = run_baseline_experiment(
                embeddings_path=embeddings,
                model_out=model_out,
                metrics_out=metrics_out,
                table_out=table_out,
                target_label="alert",
                overfit_limit=8,
                epochs=40,
                seed=7,
            )

            metrics = json.loads(metrics_out.read_text(encoding="utf-8"))
            table = table_out.read_text(encoding="utf-8")

            self.assertTrue(model_out.exists())
            self.assertEqual(result["split"]["train_subjects"], ["sub-01", "sub-02"])
            self.assertEqual(result["split"]["val_subjects"], ["sub-11"])
            self.assertEqual(result["split"]["test_subjects"], ["sub-13"])
            self.assertLess(result["overfit_check"]["final_loss"], result["overfit_check"]["initial_loss"])
            self.assertIn("full", metrics["runs"])
            self.assertIn("test", metrics["runs"]["full"])
            self.assertIn("| full |", table)


def _write_synthetic_embeddings(path: Path) -> None:
    subjects = np.array(["sub-01"] * 4 + ["sub-02"] * 4 + ["sub-11"] * 4 + ["sub-13"] * 4, dtype=object)
    sample_id = np.array([f"sample-{idx:04d}" for idx in range(len(subjects))], dtype=object)
    event_id = np.array([f"event-{idx:04d}" for idx in range(len(subjects))], dtype=object)
    session_id = np.array(["ses-01"] * len(subjects), dtype=object)
    base = np.linspace(0.0, 1.0, len(subjects), dtype=np.float32)
    labels = np.array([json.dumps({"alert": float(value)}) for value in base], dtype=object)
    emb = np.zeros((len(subjects), 256), dtype=np.float32)
    emb[:, 0] = base
    emb[:, 1] = 1.0
    mask = np.ones((len(subjects), 4), dtype=np.int8)
    np.savez_compressed(
        path,
        sample_id=sample_id,
        event_id=event_id,
        subject_id=subjects,
        session_id=session_id,
        eeg_emb=emb,
        wear_emb=emb * 0.5,
        face_emb=emb * 0.25,
        audio_emb=emb * 0.75,
        modality_mask=mask,
        labels=labels,
        source_paths=np.array(["{}"] * len(subjects), dtype=object),
    )


if __name__ == "__main__":
    unittest.main()
