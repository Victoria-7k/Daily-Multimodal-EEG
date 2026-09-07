import tempfile
import unittest
from pathlib import Path

from daily_multimodal.temporal.window_slicing import (
    build_temporal_token_index,
    make_temporal_slices,
    read_jsonl,
    write_jsonl,
)


class TemporalWindowSlicingTests(unittest.TestCase):
    def test_make_default_2s_slices(self):
        slices = make_temporal_slices()
        self.assertEqual([(item.start_seconds, item.end_seconds) for item in slices], [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)])

    def test_reject_non_divisible_window(self):
        with self.assertRaises(ValueError):
            make_temporal_slices(window_seconds=10, token_seconds=3)

    def test_build_temporal_token_index_adds_boundaries(self):
        rows = [{"sample_id": "s1", "window_size_seconds": 10, "labels": {"fatigue": 3.0}}]
        temporal, audit = build_temporal_token_index(rows, token_seconds=2)
        self.assertEqual(temporal[0]["temporal_token_count"], 5)
        self.assertEqual(temporal[0]["token_start_seconds"], [0, 2, 4, 6, 8])
        self.assertEqual(audit["token_count_distribution"], {"5": 1})

    def test_jsonl_round_trip_uses_utf8_sig_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.jsonl"
            write_jsonl([{"sample_id": "s1"}], path)
            self.assertEqual(read_jsonl(path), [{"sample_id": "s1"}])


if __name__ == "__main__":
    unittest.main()
