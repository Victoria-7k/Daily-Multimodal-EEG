import unittest
from datetime import datetime
from pathlib import Path

from daily_multimodal.io.wear import parse_wear_filename


class WearFilenameParseTests(unittest.TestCase):
    def test_parses_modality_file(self):
        parsed = parse_wear_filename(
            Path(
                "Study(Default)_UID3631()_ID3631_"
                "20250228170400_20250228201640_GSR.csv"
            )
        )

        self.assertEqual(parsed.uid, "3631")
        self.assertEqual(parsed.device_id, "3631")
        self.assertEqual(parsed.start_time, datetime(2025, 2, 28, 17, 4, 0))
        self.assertEqual(parsed.end_time, datetime(2025, 2, 28, 20, 16, 40))
        self.assertEqual(parsed.modality, "GSR")
        self.assertEqual(parsed.extension, "csv")

    def test_parses_summary_mat_file(self):
        parsed = parse_wear_filename(
            Path(
                "Study(Default)_UID3631()_ID3631_"
                "20250228170400_20250228201640.mat"
            )
        )

        self.assertEqual(parsed.modality, "SUMMARY_MAT")
        self.assertEqual(parsed.extension, "mat")


if __name__ == "__main__":
    unittest.main()
