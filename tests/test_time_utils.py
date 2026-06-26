import unittest
from datetime import datetime

from daily_multimodal.alignment.time_utils import (
    parse_absolute_time,
    subject_to_video_subject,
    time_to_video_day,
)


class TimeUtilsTests(unittest.TestCase):
    def test_parse_absolute_time_accepts_dataset_formats(self):
        self.assertEqual(
            parse_absolute_time("2025-02-28 17:27:32"),
            datetime(2025, 2, 28, 17, 27, 32),
        )
        self.assertEqual(
            parse_absolute_time("2025-02-28T17:03:20"),
            datetime(2025, 2, 28, 17, 3, 20),
        )
        self.assertEqual(
            parse_absolute_time("28-Feb-2025 17:27:32"),
            datetime(2025, 2, 28, 17, 27, 32),
        )

    def test_subject_and_day_match_video_layout(self):
        dt = datetime(2025, 2, 28, 17, 27, 32)

        self.assertEqual(subject_to_video_subject("sub-02"), "sub2")
        self.assertEqual(subject_to_video_subject("sub-10"), "sub10")
        self.assertEqual(time_to_video_day(dt), "0228")


if __name__ == "__main__":
    unittest.main()
