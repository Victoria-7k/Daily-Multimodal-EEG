import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from daily_multimodal.embeddings.cache import (
    RealCacheProfiles,
    build_cache_key,
    prepare_real_embedding_cache,
)
from daily_multimodal.embeddings.real_pipeline import pack_real_embeddings


class RealPipelineCacheTests(unittest.TestCase):
    def test_build_cache_key_is_stable_and_rejects_path_traversal(self):
        self.assertEqual(
            build_cache_key("sub-02_row-1_win-0000", "audio", "wavlm_frozen_v1"),
            "sub-02_row-1_win-0000/audio/wavlm_frozen_v1",
        )

        with self.assertRaisesRegex(ValueError, "cache key"):
            build_cache_key("../escape", "audio", "wavlm_frozen_v1")

    def test_prepare_real_embedding_cache_writes_modality_cache_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "source"
            source_root.mkdir()
            video = _touch(source_root / "clip.mp4")
            eeg = _touch(source_root / "sample.bdf")
            ppg = _touch(source_root / "ppg.csv")
            gsr = _touch(source_root / "gsr.csv")
            acc = _touch(source_root / "acc.csv")
            cache_root = root / "cache"
            report_out = root / "reports" / "real_embedding_readiness_report.md"
            failures_out = root / "reports" / "real_embedding_failures.json"
            extracted_audio: list[Path] = []

            def fake_audio_extractor(source, start_seconds, end_seconds, output):
                self.assertEqual(source, video)
                self.assertEqual(start_seconds, 1.25)
                self.assertEqual(end_seconds, 4.75)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"wav")
                extracted_audio.append(output)

            summary = prepare_real_embedding_cache(
                [_cache_window(video, eeg, ppg, gsr, acc)],
                cache_root=cache_root,
                report_out=report_out,
                failures_out=failures_out,
                profiles=RealCacheProfiles(),
                audio_extractor=fake_audio_extractor,
            )

            failures = json.loads(failures_out.read_text(encoding="utf-8"))
            report = report_out.read_text(encoding="utf-8")

        self.assertEqual(summary["modalities"]["audio"]["ready_count"], 1)
        self.assertEqual(summary["modalities"]["face"]["ready_count"], 1)
        self.assertEqual(summary["modalities"]["eeg"]["ready_count"], 1)
        self.assertEqual(summary["modalities"]["wear"]["ready_count"], 1)
        self.assertEqual(failures, [])
        self.assertEqual(len(extracted_audio), 1)
        self.assertIn("Audio ready: 1", report)
        self.assertIn("Wear ready: 1", report)

    def test_prepare_real_embedding_cache_records_explicit_failures_for_missing_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = prepare_real_embedding_cache(
                [
                    _cache_window(
                        root / "missing.mp4",
                        root / "missing.bdf",
                        root / "ppg.csv",
                        root / "gsr.csv",
                        root / "acc.csv",
                    )
                ],
                cache_root=root / "cache",
                report_out=root / "readiness.md",
                failures_out=root / "failures.json",
                profiles=RealCacheProfiles(),
                audio_extractor=lambda source, start_seconds, end_seconds, output: None,
            )
            failures = json.loads((root / "failures.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["modalities"]["audio"]["ready_count"], 0)
        self.assertEqual(summary["modalities"]["audio"]["missing_count"], 1)
        self.assertGreaterEqual(len(failures), 4)
        self.assertTrue(all(failure["sample_id"] == "sample-1" for failure in failures))
        self.assertIn("source_missing", {failure["error_type"] for failure in failures})


class RealPipelineTests(unittest.TestCase):
    def test_pack_real_embeddings_aligns_by_window_index_and_masks_missing_modalities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            window_index = root / "window_index.jsonl"
            output_npz = root / "all_complete_real_embeddings.npz"
            report_out = root / "all_complete_real_embedding_report.json"
            failures_out = root / "all_complete_real_embedding_failures.json"
            window_index.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        _window("sample-a", subject_id="sub-11", alert="2"),
                        _window("sample-b", subject_id="sub-12", alert="4"),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            eeg_npz = _write_modality_npz(root / "eeg.npz", "eeg", ["sample-a", "sample-b"])
            wear_npz = _write_modality_npz(root / "wear.npz", "wear", ["sample-b", "sample-a"])
            face_npz = _write_modality_npz(
                root / "face.npz",
                "face",
                ["sample-a", "sample-b"],
                masked={"sample-b"},
            )
            audio_npz = _write_modality_npz(root / "audio.npz", "audio", ["sample-a"])

            summary = pack_real_embeddings(
                window_index=window_index,
                eeg_embeddings=eeg_npz,
                wear_embeddings=wear_npz,
                face_embeddings=face_npz,
                audio_embeddings=audio_npz,
                output_npz=output_npz,
                report_out=report_out,
                failures_out=failures_out,
            )

            with np.load(output_npz, allow_pickle=True) as loaded:
                sample_ids = loaded["sample_id"].astype(str).tolist()
                labels = [json.loads(value) for value in loaded["labels"].tolist()]
                eeg_emb = loaded["eeg_emb"]
                wear_emb = loaded["wear_emb"]
                face_emb = loaded["face_emb"]
                audio_emb = loaded["audio_emb"]
                modality_mask = loaded["modality_mask"]
                quality_flags = [json.loads(value) for value in loaded["quality_flags"].tolist()]
                encoder_versions = [
                    json.loads(value) for value in loaded["encoder_versions"].tolist()
                ]
            report = json.loads(report_out.read_text(encoding="utf-8"))
            failures = json.loads(failures_out.read_text(encoding="utf-8"))

        self.assertEqual(summary["selected_windows"], 2)
        self.assertEqual(sample_ids, ["sample-a", "sample-b"])
        self.assertEqual(labels, [{"alert": "2"}, {"alert": "4"}])
        self.assertEqual(eeg_emb.shape, (2, 256))
        self.assertEqual(wear_emb.shape, (2, 256))
        self.assertEqual(face_emb.shape, (2, 256))
        self.assertEqual(audio_emb.shape, (2, 256))
        self.assertEqual(modality_mask.tolist(), [[1, 1, 1, 1], [1, 1, 0, 0]])
        self.assertFalse(np.all(wear_emb[0] == 0.0))
        self.assertTrue(np.all(face_emb[1] == 0.0))
        self.assertTrue(np.all(audio_emb[1] == 0.0))
        self.assertEqual(quality_flags[0]["wear"]["sample_id"], "sample-a")
        self.assertEqual(quality_flags[1]["face"]["masked"], True)
        self.assertEqual(encoder_versions[0]["audio"], "audio_profile")
        self.assertEqual(report["stage"], 17)
        self.assertEqual(report["summary"]["success_count"], 2)
        self.assertEqual(report["modalities"]["audio"]["missing_count"], 1)
        self.assertEqual(report["modalities"]["face"]["masked_count"], 1)
        self.assertEqual(failures[0]["error_type"], "source_missing")
        self.assertEqual(failures[0]["modality"], "audio")
        self.assertEqual(failures[0]["sample_id"], "sample-b")

    def test_cli_writes_all_real_embedding_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            window_index = root / "window_index.jsonl"
            output_npz = root / "embeddings.npz"
            report_out = root / "report.json"
            failures_out = root / "failures.json"
            window_index.write_text(
                json.dumps(_window("sample-a", subject_id="sub-11", alert="3")) + "\n",
                encoding="utf-8",
            )
            paths = {
                modality: _write_modality_npz(root / f"{modality}.npz", modality, ["sample-a"])
                for modality in ("eeg", "wear", "face", "audio")
            }

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/16_extract_all_real_embeddings.py",
                    "--window-index",
                    str(window_index),
                    "--eeg",
                    str(paths["eeg"]),
                    "--wear",
                    str(paths["wear"]),
                    "--face",
                    str(paths["face"]),
                    "--audio",
                    str(paths["audio"]),
                    "--out",
                    str(output_npz),
                    "--report-out",
                    str(report_out),
                    "--failures-out",
                    str(failures_out),
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("embedding_path=", result.stdout)
            self.assertTrue(output_npz.is_file())
            self.assertTrue(report_out.is_file())
            self.assertTrue(failures_out.is_file())


def _window(sample_id: str, *, subject_id: str, alert: str) -> dict:
    return {
        "sample_id": sample_id,
        "event_id": sample_id.replace("sample", "event"),
        "subject_id": subject_id,
        "session_id": "ses-01",
        "label_columns": {"alert": alert},
        "window_start_time": "2025-02-28 14:13:00",
        "window_end_time": "2025-02-28 14:13:10",
        "eeg_bdf_path": f"/raw/{sample_id}.bdf",
        "wear_ppg_path": f"/raw/{sample_id}_ppg.csv",
        "wear_gsr_path": f"/raw/{sample_id}_gsr.csv",
        "wear_acc_path": f"/raw/{sample_id}_acc.csv",
        "candidate_mp4_paths": [f"/raw/{sample_id}.MP4"],
        "candidate_audio_paths": [f"/raw/{sample_id}.wav"],
    }


def _touch(path: Path) -> Path:
    path.write_text("x", encoding="utf-8")
    return path


def _cache_window(video: Path, eeg: Path, ppg: Path, gsr: Path, acc: Path) -> dict:
    return {
        "sample_id": "sample-1",
        "event_id": "event-1",
        "subject_id": "sub-02",
        "window_start_time": "2025-02-28 14:13:00",
        "window_end_time": "2025-02-28 14:13:10",
        "eeg_bdf_path": str(eeg),
        "wear_ppg_path": str(ppg),
        "wear_gsr_path": str(gsr),
        "wear_acc_path": str(acc),
        "video_candidates": [
            {
                "mp4_path": str(video),
                "clip_start_seconds": 1.25,
                "clip_end_seconds": 4.75,
                "covers_window": True,
            }
        ],
    }


def _write_modality_npz(
    path: Path,
    modality: str,
    sample_ids: list[str],
    *,
    masked: set[str] | None = None,
) -> Path:
    mask_index = {"eeg": 0, "wear": 1, "face": 2, "audio": 3}[modality]
    masked = masked or set()
    embeddings = []
    masks = []
    quality_flags = []
    for offset, sample_id in enumerate(sample_ids):
        value = (offset + 1) * (mask_index + 1)
        embedding = np.full(256, value, dtype=np.float32)
        mask = np.zeros(4, dtype=np.int8)
        if sample_id not in masked:
            mask[mask_index] = 1
        embeddings.append(embedding)
        masks.append(mask)
        quality_flags.append(
            json.dumps(
                {"sample_id": sample_id, "masked": sample_id in masked},
                ensure_ascii=False,
            )
        )

    np.savez_compressed(
        path,
        sample_id=np.array(sample_ids, dtype=object),
        event_id=np.array([sample_id.replace("sample", "event") for sample_id in sample_ids], dtype=object),
        subject_id=np.array(["sub-11"] * len(sample_ids), dtype=object),
        **{f"{modality}_emb": np.stack(embeddings).astype(np.float32)},
        modality_mask=np.stack(masks).astype(np.int8),
        quality_flags=np.array(quality_flags, dtype=object),
        encoder_version=np.array([f"{modality}_profile"] * len(sample_ids), dtype=object),
    )
    return path
