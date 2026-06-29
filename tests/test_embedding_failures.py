import json
import tempfile
import unittest
from pathlib import Path

from daily_multimodal.embeddings.failures import (
    EmbeddingFailure,
    write_failure_list,
)


class EmbeddingFailureTests(unittest.TestCase):
    def test_write_failure_list_writes_empty_json_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "failures.json"

            write_failure_list([], out)

            self.assertEqual(out.read_text(encoding="utf-8").strip(), "[]")

    def test_embedding_failure_contains_required_locator_fields(self):
        failure = EmbeddingFailure(
            sample_id="sub-02_ses-01_row-0001_win-0000",
            event_id="sub-02_ses-01_row-0001",
            subject_id="sub-02",
            modality="audio",
            encoder_profile="wavlm_frozen_v1",
            stage="extract_audio_clip",
            error_type="extraction_failed",
            error="ffmpeg returned non-zero",
            source_path="/data/example.mp4",
            recoverable=True,
        )

        payload = failure.to_dict()

        for field in ["modality", "encoder_profile", "stage", "error_type", "source_path"]:
            self.assertIn(field, payload)
            self.assertTrue(payload[field])

    def test_write_failure_list_validates_error_type_and_serializes_dataclasses(self):
        with self.assertRaisesRegex(ValueError, "unsupported error_type"):
            EmbeddingFailure(
                sample_id="sample",
                event_id="event",
                subject_id="sub-02",
                modality="audio",
                encoder_profile="wavlm_frozen_v1",
                stage="extract_audio_clip",
                error_type="not_a_known_failure",
                error="bad",
                source_path="/data/example.mp4",
            )

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "failures.json"
            write_failure_list(
                [
                    EmbeddingFailure(
                        sample_id="sample",
                        event_id="event",
                        subject_id="sub-02",
                        modality="eeg",
                        encoder_profile="eeg_real_frozen_v1",
                        stage="prepare_eeg_window",
                        error_type="source_missing",
                        error="missing BDF",
                        source_path="/data/example.bdf",
                    )
                ],
                out,
            )

            payload = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(payload[0]["error_type"], "source_missing")
        self.assertTrue(payload[0]["recoverable"])


if __name__ == "__main__":
    unittest.main()
