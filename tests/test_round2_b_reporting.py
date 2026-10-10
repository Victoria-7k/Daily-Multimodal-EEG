"""The practical gate requires both controls, and resolves the declared tie."""
import importlib.util
from pathlib import Path
import unittest

class BReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("report_b_tests", Path(__file__).resolve().parents[1] / "scripts/multilabel/142_report_round2_experiment_b.py")
        cls.report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.report)

    def gates(self):
        return {name: {"passed": False} for name in self.report.COEFFICIENTS}

    def test_both_mechanism_and_frozen_control_required(self):
        g = self.gates()
        g["B_T2_LOW-B_T2_STD"]["passed"] = True
        self.assertIsNone(self.report.select_candidate(g, {"B_T2_LOW": .9}))
        g["B_T2_LOW-F_C"]["passed"] = True
        self.assertEqual(self.report.select_candidate(g, {"B_T2_LOW": .9}), "B_T2_LOW")

    def test_tolerance_prefers_fewer_blocks_then_srmse(self):
        g = {name: {"passed": True} for name in self.report.COEFFICIENTS}
        self.assertEqual(self.report.select_candidate(g, {"B_T2_LOW": .9, "B_T1_LOW": .9000005}), "B_T1_LOW")
        self.assertEqual(self.report.select_candidate(g, {"B_T2_LOW": .9, "B_T1_LOW": .90001}), "B_T2_LOW")

if __name__ == "__main__":
    unittest.main()
