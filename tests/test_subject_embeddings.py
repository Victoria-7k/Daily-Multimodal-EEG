import unittest

from daily_multimodal.embeddings.subject import select_subject_windows, summarize_subject_selection


class SubjectEmbeddingSelectionTests(unittest.TestCase):
    def test_selects_subject_windows_and_requires_all_modalities_when_requested(self):
        windows = [
            {
                "sample_id": "sub-10_ses-01_row-0001_win-0000",
                "subject_id": "sub-10",
                "has_eeg": True,
                "has_ppg": True,
                "has_gsr": True,
                "has_acc": True,
                "has_face": True,
                "has_audio": True,
            },
            {
                "sample_id": "sub-10_ses-01_row-0002_win-0000",
                "subject_id": "sub-10",
                "has_eeg": True,
                "has_ppg": True,
                "has_gsr": False,
                "has_acc": True,
                "has_face": True,
                "has_audio": True,
            },
            {
                "sample_id": "sub-14_ses-01_row-0001_win-0000",
                "subject_id": "sub-14",
                "has_eeg": True,
                "has_ppg": True,
                "has_gsr": True,
                "has_acc": True,
                "has_face": True,
                "has_audio": True,
            },
        ]

        selected = select_subject_windows(
            windows,
            subject_id="sub-10",
            require_all_modalities=True,
        )
        summary = summarize_subject_selection(windows, selected, subject_id="sub-10")

        self.assertEqual([window["sample_id"] for window in selected], ["sub-10_ses-01_row-0001_win-0000"])
        self.assertEqual(summary["subject_id"], "sub-10")
        self.assertEqual(summary["all_windows"], 3)
        self.assertEqual(summary["subject_windows"], 2)
        self.assertEqual(summary["selected_windows"], 1)


if __name__ == "__main__":
    unittest.main()
