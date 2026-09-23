from pathlib import Path
import unittest

from daily_multimodal.split_paths import (
    DATE_IN_ORDER_SPLIT_ROOT,
    LEGACY_SPLITS_ROOT,
    resolve_protocol_split_root,
)


class SplitPathTests(unittest.TestCase):
    def test_date_in_order_resolves_to_chronological_split(self) -> None:
        self.assertEqual(
            resolve_protocol_split_root(LEGACY_SPLITS_ROOT, "date_in_order"),
            DATE_IN_ORDER_SPLIT_ROOT,
        )

    def test_other_protocol_keeps_requested_root(self) -> None:
        self.assertEqual(
            resolve_protocol_split_root(LEGACY_SPLITS_ROOT, "cross_day"),
            LEGACY_SPLITS_ROOT / "cross_day",
        )

    def test_legacy_within_subject_day_keeps_legacy_root(self) -> None:
        self.assertEqual(
            resolve_protocol_split_root(LEGACY_SPLITS_ROOT, "within_subject_day"),
            LEGACY_SPLITS_ROOT / "within_subject_day",
        )

    def test_custom_date_in_order_root_is_preserved(self) -> None:
        custom_root = Path("custom/splits")
        self.assertEqual(
            resolve_protocol_split_root(custom_root, "date_in_order"),
            custom_root / "date_in_order",
        )


if __name__ == "__main__":
    unittest.main()
