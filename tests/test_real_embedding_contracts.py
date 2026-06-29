import unittest

import numpy as np

from daily_multimodal.embeddings.contracts import (
    RealEmbeddingResult,
    validate_embedding_shape,
)


class RealEmbeddingContractTests(unittest.TestCase):
    def test_validate_embedding_shape_accepts_single_and_batch_float_vectors(self):
        single = validate_embedding_shape("audio_emb", np.zeros(256, dtype=np.float64))
        batch = validate_embedding_shape("audio_emb", np.zeros((3, 256), dtype=np.float32))

        self.assertEqual(single.shape, (256,))
        self.assertEqual(single.dtype, np.float32)
        self.assertEqual(batch.shape, (3, 256))

    def test_validate_embedding_shape_rejects_wrong_shape_dtype_and_nan(self):
        with self.assertRaisesRegex(ValueError, "audio_emb.*expected"):
            validate_embedding_shape("audio_emb", np.zeros((2, 128), dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "audio_emb.*floating"):
            validate_embedding_shape("audio_emb", np.zeros(256, dtype=np.int64))

        bad = np.zeros(256, dtype=np.float32)
        bad[3] = np.nan
        with self.assertRaisesRegex(ValueError, "audio_emb.*NaN"):
            validate_embedding_shape("audio_emb", bad)

    def test_real_embedding_result_normalizes_embedding_and_validates_mask(self):
        result = RealEmbeddingResult(
            sample_id="sub-02_ses-01_row-0001_win-0000",
            event_id="sub-02_ses-01_row-0001",
            subject_id="sub-02",
            modality="audio",
            embedding=np.ones(256, dtype=np.float64),
            mask_value=True,
            quality_flags={"duration_seconds": 10.0},
            encoder_version="wavlm_frozen_v1",
            source_paths={"video": "/tmp/example.mp4"},
        )

        self.assertEqual(result.embedding.dtype, np.float32)
        self.assertEqual(result.mask_value, 1)
        self.assertEqual(result.modality, "audio")

        with self.assertRaisesRegex(ValueError, "mask_value"):
            RealEmbeddingResult(
                sample_id="sample",
                event_id="event",
                subject_id="sub-02",
                modality="audio",
                embedding=np.ones(256, dtype=np.float32),
                mask_value=3,
                quality_flags={},
                encoder_version="wavlm_frozen_v1",
                source_paths={},
            )


if __name__ == "__main__":
    unittest.main()
