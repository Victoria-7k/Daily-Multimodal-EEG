from pathlib import Path
import unittest

from daily_multimodal.split_paths import (
    CANONICAL_WITHIN_SUBJECT_DAY_ROOT,
    LEGACY_SPLITS_ROOT,
    resolve_protocol_split_root,
)


class SplitPathTests(unittest.TestCase):
    def test_legacy_within_subject_day_resolves_to_canonical_repaired_split(self) -> None:
        self.assertEqual(
            resolve_protocol_split_root(LEGACY_SPLITS_ROOT, "within_subject_day"),
            CANONICAL_WITHIN_SUBJECT_DAY_ROOT,
        )

    def test_other_protocol_keeps_requested_root(self) -> None:
        self.assertEqual(
            resolve_protocol_split_root(LEGACY_SPLITS_ROOT, "cross_day"),
            LEGACY_SPLITS_ROOT / "cross_day",
        )

    def test_custom_within_subject_day_root_is_preserved(self) -> None:
        custom_root = Path("custom/splits")
        self.assertEqual(
            resolve_protocol_split_root(custom_root, "within_subject_day"),
            custom_root / "within_subject_day",
        )


if __name__ == "__main__":
    unittest.main()
