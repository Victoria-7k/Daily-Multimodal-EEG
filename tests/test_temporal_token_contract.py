import tempfile
import unittest
from pathlib import Path

import numpy as np

from daily_multimodal.temporal.token_contract import (
    MODALITY_ORDER,
    TOKEN_KEYS,
    validate_modality_temporal_npz,
    validate_packed_temporal_npz,
    write_packed_temporal_tokens,
)


class TemporalTokenContractTests(unittest.TestCase):
    def test_validate_modality_temporal_npz_checks_sample_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eeg.npz"
            np.savez_compressed(path, sample_id=np.asarray(["b", "a"]), eeg_tokens=np.zeros((2, 5, 256), dtype=np.float32))
            with self.assertRaises(ValueError):
                validate_modality_temporal_npz(path, modality="eeg", expected_sample_id=np.asarray(["a", "b"]))

    def test_write_and_validate_packed_temporal_tokens(self):
        rows = [
            _row("s1", {"fatigue": 2.0}),
            _row("s2", {"fatigue": 4.0}),
        ]
        arrays = {}
        for index, modality in enumerate(MODALITY_ORDER):
            tokens = np.full((2, 5, 256), fill_value=float(index + 1), dtype=np.float32)
            mask = np.ones((2, 5), dtype=np.int8)
            quality = np.ones((2, 5, 2), dtype=np.float32) * (index + 1)
            arrays[modality] = {
                TOKEN_KEYS[modality]: tokens,
                f"{modality}_token_mask": mask,
                f"{modality}_quality_features": quality,
            }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "packed.npz"
            report = write_packed_temporal_tokens(output_path=out, temporal_index_rows=rows, modality_arrays=arrays)
            self.assertEqual(report["row_count"], 2)
            validated = validate_packed_temporal_npz(out, expected_sample_id=np.asarray(["s1", "s2"]))
            self.assertEqual(validated["quality_feature_dim"], 2)
            with np.load(out, allow_pickle=True) as loaded:
                self.assertEqual(loaded["modality_mask"].tolist(), [[1, 1, 1, 1], [1, 1, 1, 1]])
                self.assertEqual(loaded["labels"].shape, (2, 1))

    def test_packed_validation_rejects_bad_token_shape(self):
        rows = [_row("s1", {"fatigue": 2.0})]
        arrays = {}
        for modality in MODALITY_ORDER:
            shape = (1, 4, 256) if modality == "audio" else (1, 5, 256)
            arrays[modality] = {TOKEN_KEYS[modality]: np.zeros(shape, dtype=np.float32)}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_packed_temporal_tokens(output_path=Path(tmp) / "bad.npz", temporal_index_rows=rows, modality_arrays=arrays)


def _row(sample_id, labels):
    return {
        "sample_id": sample_id,
        "event_id": sample_id.split("_")[0],
        "subject_id": "sub-01",
        "session_id": "2026-08-23",
        "token_start_seconds": [0, 2, 4, 6, 8],
        "token_end_seconds": [2, 4, 6, 8, 10],
        "labels": labels,
    }


if __name__ == "__main__":
    unittest.main()
