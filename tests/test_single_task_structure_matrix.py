from __future__ import annotations

import unittest

from daily_multimodal.daily_affect.ema_bags import (
    BRANCHES, LABEL_NAMES, effective_route_id, resolve_route_branches, supervision_boundary,
)
from daily_multimodal.training.structure_emotion import conditions


class SingleTaskStructureMatrixTests(unittest.TestCase):
    def test_label_specific_eeg_branches_and_routes(self) -> None:
        self.assertEqual(len(conditions()), 19)
        for label in LABEL_NAMES:
            with self.subTest(label=label):
                branch = f"eeg_eegpt_partial_ft_single_{label}_v1"
                self.assertEqual(
                    resolve_route_branches("A1_Wphysio_no_audio", eeg_branch=branch),
                    (branch, "wear_physio", "video_A1"),
                )
                self.assertEqual(
                    BRANCHES[branch].filename.format(eeg_token_root="tokens", protocol="cross_day", eeg_seed=240800),
                    f"tokens/single_task/cross_day/{label}/seed_240800.npz",
                )
                self.assertEqual(
                    effective_route_id("A1_Wphysio_no_audio", eeg_branch=branch),
                    f"A1_Wphysio_no_audio__{branch}",
                )
                self.assertIn(label, supervision_boundary((branch, "wear_physio", "video_A1")))


if __name__ == "__main__":
    unittest.main()
