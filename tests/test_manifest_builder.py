import json
import tempfile
import unittest
from pathlib import Path

from daily_multimodal.manifest.build_manifest import build_manifest


class ManifestBuilderTests(unittest.TestCase):
    def test_build_manifest_matches_labels_to_modalities_by_absolute_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eeg_root = root / "DailyEEG_dataset"
            video_root = root / "video"
            wear_root = root / "wear"

            eeg_dir = eeg_root / "sub-02" / "ses-01" / "eeg"
            beh_dir = eeg_root / "sub-02" / "ses-01" / "beh"
            video_day = video_root / "sub2" / "0228"
            for path in [eeg_dir, beh_dir, video_day, wear_root]:
                path.mkdir(parents=True)

            (eeg_dir / "sub-02_ses-01_task-dailylife_eeg.json").write_text(
                json.dumps(
                    {
                        "RecordingStartTime": "2025-02-28T17:03:20",
                        "RecordingDuration": 11600.0,
                        "SamplingFrequency": 500.0,
                    }
                ),
                encoding="utf-8",
            )
            (eeg_dir / "sub-02_ses-01_task-dailylife_eeg.bdf").write_bytes(b"")
            (beh_dir / "sub-02_ses-01_task-dailylife_emotion_beh.tsv").write_text(
                "\t".join(
                    [
                        "onset",
                        "duration",
                        "absolute_onset_time",
                        "inspired",
                        "alert",
                        "determined",
                        "attentive",
                        "active",
                        "hostile",
                        "nervous",
                        "upset",
                        "afraid",
                        "ashamed",
                        "fatigue",
                        "activity_category",
                        "social_presence",
                        "activity_text",
                    ]
                )
                + "\n"
                + "1452.0\t10.0\t2025-02-28 17:27:32\t1\t2\t3\t4\t5\t1\t1\t1\t1\t1\t2\tIn-work\tY\twriting\n"
                + "2449.0\t8.0\t2025-02-28 17:44:09\t2\t2\t3\t3\t4\t1\t1\t2\t1\t1\t2\tOut-work\tN\twalking\n",
                encoding="utf-8",
            )
            for suffix in ["ACC", "GSR", "PPG"]:
                (wear_root / f"Study(Default)_UID3631()_ID3631_20250228170400_20250228201640_{suffix}.csv").write_text(
                    "value,time\n", encoding="utf-8"
                )
            (video_day / "DJI_0001.MP4").write_bytes(b"fake")

            manifest, coverage = build_manifest(
                eeg_dataset=eeg_root,
                video_root=video_root,
                wear_root=wear_root,
            )

        self.assertEqual(len(manifest), 2)
        self.assertEqual(coverage["events_total"], 2)
        self.assertEqual(coverage["complete_wear_events"], 2)
        self.assertEqual(coverage["video_day_events"], 2)
        self.assertEqual(coverage["complete_multimodal_candidates"], 2)

        first = manifest[0]
        self.assertEqual(first["subject_id"], "sub-02")
        self.assertEqual(first["session_id"], "ses-01")
        self.assertEqual(first["eeg_sampling_frequency"], 500.0)
        self.assertTrue(first["has_eeg"])
        self.assertTrue(first["has_ppg"])
        self.assertTrue(first["has_gsr"])
        self.assertTrue(first["has_acc"])
        self.assertTrue(first["has_video"])
        self.assertTrue(first["has_audio"])
        self.assertTrue(first["is_complete_multimodal_candidate"])
        self.assertEqual(first["activity_category"], "In-work")
        self.assertEqual(first["social_presence"], "Y")
        self.assertIn("DJI_0001.MP4", first["candidate_mp4_paths"][0])


if __name__ == "__main__":
    unittest.main()
