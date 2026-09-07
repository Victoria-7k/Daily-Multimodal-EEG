import tempfile
import unittest
from pathlib import Path

import numpy as np

from daily_multimodal.temporal.global_repeat_tokens import build_global_repeat_temporal_tokens


class GlobalRepeatTemporalTokenTests(unittest.TestCase):
    def test_builds_repeat_temporal_npz_from_window_embedding(self):
        rows = [
            _row("s1"),
            _row("s2"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "wear.npz"
            out = Path(tmp) / "wear_temporal.npz"
            emb = np.arange(2 * 256, dtype=np.float32).reshape(2, 256)
            np.savez_compressed(
                source,
                sample_id=np.asarray(["s1", "s2"]),
                wear_emb=emb,
                modality_mask=np.asarray([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=np.int8),
            )
            report = build_global_repeat_temporal_tokens(
                temporal_index_rows=rows,
                source_npz=source,
                output_npz=out,
                modality="wear",
            )
            self.assertEqual(report["token_shape"], [2, 5, 256])
            with np.load(out, allow_pickle=True) as loaded:
                self.assertEqual(loaded["wear_tokens"].shape, (2, 5, 256))
                np.testing.assert_allclose(loaded["wear_tokens"][0, 3], emb[0])
                self.assertEqual(loaded["wear_token_mask"].tolist(), [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])
                self.assertEqual(loaded["wear_quality_features"].shape, (2, 5, 1))

    def test_rejects_sample_id_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "audio.npz"
            np.savez_compressed(source, sample_id=np.asarray(["other"]), audio_emb=np.zeros((1, 256), dtype=np.float32))
            with self.assertRaises(ValueError):
                build_global_repeat_temporal_tokens(
                    temporal_index_rows=[_row("s1")],
                    source_npz=source,
                    output_npz=Path(tmp) / "audio_temporal.npz",
                    modality="audio",
                )


def _row(sample_id):
    return {
        "sample_id": sample_id,
        "event_id": sample_id,
        "subject_id": "sub-01",
        "token_start_seconds": [0, 2, 4, 6, 8],
        "token_end_seconds": [2, 4, 6, 8, 10],
    }


if __name__ == "__main__":
    unittest.main()
