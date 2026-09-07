import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - runtime dependent
    torch = None

from daily_multimodal.daily_affect.ema_bags import build_daily_affect_bags
from daily_multimodal.daily_affect.metrics import classification_metrics, quadratic_weighted_kappa


class DailyAffectBagTests(unittest.TestCase):
    def test_build_daily_affect_bags_from_window_npzs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_path, splits_root, emb_root = _write_synthetic_inputs(root, event_count=4)

            result = build_daily_affect_bags(
                index_path=index_path,
                splits_root=splits_root,
                embeddings_root=emb_root,
                protocol="cross_day",
                route_id="B0_Wphysio_full",
                out_dir=root / "out/bags/cross_day/B0_Wphysio_full/seed_240800",
                eeg_seed=240800,
                wear_seed=240800,
            )

            self.assertEqual(result["row_count"], 4)
            self.assertEqual(result["tokens_shape"], [4, 23, 4, 256])
            with np.load(Path(result["bag_path"]), allow_pickle=True) as data:
                self.assertEqual(data["tokens"].shape, (4, 23, 4, 256))
                self.assertEqual(data["modality_mask"].shape, (4, 23, 4))
                self.assertEqual(data["train_index"].tolist(), [0, 1])
                self.assertEqual(data["val_index"].tolist(), [2])
                self.assertEqual(data["test_index"].tolist(), [3])

    def test_build_daily_affect_bags_allows_pretrain_finetune_mixed_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_path, splits_root, emb_root = _write_synthetic_inputs(root, event_count=4)
            protocol_root = splits_root / "cross_day"
            (protocol_root / "pretrain.json").write_text(json.dumps(list(range(0, 10))), encoding="utf-8")
            (protocol_root / "finetune.json").write_text(json.dumps(list(range(10, 46))), encoding="utf-8")

            result = build_daily_affect_bags(
                index_path=index_path,
                splits_root=splits_root,
                embeddings_root=emb_root,
                protocol="cross_day",
                route_id="B0_Wphysio_full",
                out_dir=root / "out/bags/cross_day/B0_Wphysio_full/seed_240800",
                eeg_seed=240800,
                wear_seed=240800,
            )

            self.assertEqual(result["mixed_train_leaf_event_count"], 1)
            with np.load(Path(result["bag_path"]), allow_pickle=True) as data:
                self.assertEqual(data["split"].astype(str).tolist(), ["train", "finetune", "val", "test"])
                self.assertEqual(data["train_index"].tolist(), [0, 1])

    def test_build_daily_affect_bags_projects_train_eval_boundary_event_by_majority(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_path, splits_root, emb_root = _write_synthetic_inputs(root, event_count=4)
            protocol_root = splits_root / "cross_day"
            (protocol_root / "pretrain.json").write_text(json.dumps(list(range(0, 4))), encoding="utf-8")
            (protocol_root / "finetune.json").write_text(json.dumps(list(range(23, 46))), encoding="utf-8")
            (protocol_root / "val.json").write_text(json.dumps(list(range(4, 23)) + list(range(46, 69))), encoding="utf-8")
            (protocol_root / "test.json").write_text(json.dumps(list(range(69, 92))), encoding="utf-8")

            result = build_daily_affect_bags(
                index_path=index_path,
                splits_root=splits_root,
                embeddings_root=emb_root,
                protocol="cross_day",
                route_id="B0_Wphysio_full",
                out_dir=root / "out/bags/cross_day/B0_Wphysio_full/seed_240800",
                eeg_seed=240800,
                wear_seed=240800,
            )

            self.assertEqual(result["projected_boundary_event_count"], 1)
            with np.load(Path(result["bag_path"]), allow_pickle=True) as data:
                self.assertEqual(data["split"].astype(str).tolist(), ["val", "finetune", "val", "test"])
                self.assertEqual(data["train_index"].tolist(), [1])
                self.assertEqual(data["val_index"].tolist(), [0, 2])

    def test_classification_metrics_include_qwk_and_macro_f1(self):
        metrics = classification_metrics(
            [0, 1, 2, 3, 4],
            [0, 2, 2, 3, 4],
            expected_score=[1.0, 2.2, 3.1, 3.8, 4.9],
            subject_ids=["s1", "s1", "s2", "s2", "s2"],
        )
        self.assertIn("macro_f1", metrics)
        self.assertIn("qwk", metrics)
        self.assertIn("expected_rmse", metrics)
        self.assertIn("balanced_accuracy", metrics)
        self.assertIn("severe_error_rate", metrics)
        self.assertEqual(len(metrics["per_class"]), 5)
        self.assertIsNotNone(metrics["expected_raw_r"])
        self.assertIsNotNone(metrics["expected_within_subject_centered_r"])
        self.assertIsNotNone(quadratic_weighted_kappa([0, 1, 2], [0, 1, 2]))

    def test_diagnostic_atlas_aggregation_is_seed_aware(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "daily_affect" / "77_plot_daily_affect_diagnostics.py"
        spec = importlib.util.spec_from_file_location("daily_affect_diagnostics_plot", script_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        records = [
            module.RunDiagnostic(
                protocol="cross_day",
                route_id="A1_Wphysio_full",
                model_id="dynamic_kernel_no_prior",
                experiment_id="state_matrix_native",
                normalization="per_modality",
                adapter_mode="per_modality",
                objective_id="weighted_ce__ord_0.5__rank_0.1",
                routing_id="probe_cumulative__difficulty_none__beta_0.25__detach_true",
                seed=seed,
                metrics_path=Path(f"seed_{seed}/metrics.json"),
                qwk=qwk,
                modality_weights=np.asarray(weights, dtype=np.float64),
                modality_difficulty=np.asarray([0.4, 0.5, 0.6, 0.7], dtype=np.float64),
                temporal_weights=np.full(23, 1.0 / 23.0, dtype=np.float64),
                confusion=np.eye(5, dtype=np.int64),
            )
            for seed, qwk, weights in (
                (1, 0.20, [0.2, 0.4, 0.3, 0.1]),
                (2, 0.40, [0.4, 0.2, 0.1, 0.3]),
            )
        ]
        group = module.aggregate_groups(records)[0]
        self.assertEqual(group["seed_count"], 2)
        self.assertAlmostEqual(group["qwk_mean"], 0.30)
        self.assertTrue(np.allclose(group["modality_weights"], [0.3, 0.3, 0.2, 0.2]))
        self.assertTrue(np.array_equal(group["confusion"], 2 * np.eye(5, dtype=np.int64)))

    def test_probe_reliability_metrics_respects_test_mask(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "daily_affect" / "77_plot_daily_affect_diagnostics.py"
        spec = importlib.util.spec_from_file_location("daily_affect_probe_reliability", script_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        probs = np.full((3, 4, 5), 0.2, dtype=np.float64)
        probs[:, 0] = np.asarray(
            [[0.9, 0.025, 0.025, 0.025, 0.025], [0.025, 0.9, 0.025, 0.025, 0.025], [0.025, 0.025, 0.9, 0.025, 0.025]]
        )
        valid = np.zeros((3, 4), dtype=bool)
        valid[:, 0] = True
        metrics = module.probe_reliability_metrics(probs, valid, np.asarray([0, 1]), np.asarray([0, 1]))
        self.assertIsNotNone(metrics)
        self.assertEqual(tuple(metrics.shape), (4, 4))
        self.assertAlmostEqual(float(metrics[0, 0]), 1.0)
        self.assertTrue(np.isnan(metrics[1]).all())

    def test_summary_pairs_across_routing_and_bootstraps_subject_days(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "daily_affect" / "76_summarize_daily_affect_results.py"
        spec = importlib.util.spec_from_file_location("daily_affect_summary", script_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            event_id = np.asarray(["e1", "e2", "e3", "e4"])
            subject_id = np.asarray(["s1", "s1", "s2", "s2"])
            day_id = np.asarray(["d1", "d1", "d2", "d2"])
            truth = np.asarray([0, 1, 3, 4], dtype=np.int64)
            test_index = np.arange(4, dtype=np.int64)
            common = {
                "event_id": event_id,
                "subject_id": subject_id,
                "day_id": day_id,
                "label_zero_based": truth,
                "test_index": test_index,
            }
            baseline_path = root / "baseline.npz"
            candidate_path = root / "candidate.npz"
            np.savez(baseline_path, **common, test_predicted_class=np.asarray([0, 0, 0, 0], dtype=np.int64))
            np.savez(candidate_path, **common, test_predicted_class=truth)
            baseline = {
                "protocol": "cross_day",
                "route_id": "A1_Wphysio_full",
                "model_id": "bag_static",
                "normalization": "per_modality",
                "adapter_mode": "per_modality",
                "objective_id": "weighted_ce__ord_0.5__rank_0.1",
                "routing_id": "probe_cumulative__difficulty_none__beta_0.25__detach_true",
                "seed": 7,
                "prediction_path": str(baseline_path),
                "test": {"qwk": 0.0, "ordinal_mae": 2.0},
            }
            candidate = {
                **baseline,
                "model_id": "prior_ordD_uniform",
                "experiment_id": "state_matrix_p3_ordinal_mix",
                "routing_id": "probe_cumulative__difficulty_ordinal__beta_0.25__detach_true",
                "prediction_path": str(candidate_path),
                "test": {"qwk": 1.0, "ordinal_mae": 0.0},
            }
            paired = module.paired_deltas([baseline, candidate], bootstrap_iters=50, bootstrap_seed=2)
            self.assertEqual(len(paired), 1)
            self.assertEqual(paired[0]["bootstrap_status"], "ok")
            self.assertEqual(paired[0]["bootstrap_unit"], "subject_day")
            self.assertEqual(paired[0]["bootstrap_block_count"], 2)

    @unittest.skipIf(torch is None, "torch is not installed in this runtime")
    def test_robustness_pair_missing_preserves_one_modality(self):
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "daily_affect" / "80_run_daily_affect_focused_robustness.py"
        spec = importlib.util.spec_from_file_location("daily_affect_robustness", script_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        tokens = np.ones((2, 23, 4, 256), dtype=np.float32)
        mask = np.ones((2, 23, 4), dtype=bool)
        changed, protected = module.remove_modalities_preserving_one(tokens, mask, np.asarray([0, 1]), (0, 1))
        self.assertEqual((changed, protected), (2, 0))
        self.assertTrue(mask[:, :, 2:].all())
        all_requested = (0, 1, 2, 3)
        changed, protected = module.remove_modalities_preserving_one(tokens, mask, np.asarray([0, 1]), all_requested)
        self.assertEqual((changed, protected), (0, 2))


@unittest.skipIf(torch is None, "torch is not installed in this runtime")
class DailyAffectTrainingTests(unittest.TestCase):
    def test_model_forward_shapes(self):
        from daily_multimodal.daily_affect.model import DailyAffectConfig, DailyAffectOrdinalModel

        model = DailyAffectOrdinalModel(DailyAffectConfig(model_id="dynamic_kernel", hidden_dim=16, adapter_mode="per_modality"))
        tokens = torch.randn(3, 23, 4, 256)
        mask = torch.ones(3, 23, 4, dtype=torch.bool)
        mask[:, :, 2] = False
        outputs = model(tokens, mask)
        self.assertEqual(tuple(outputs["class_logits"].shape), (3, 5))
        self.assertEqual(tuple(outputs["modality_weights"].shape), (3, 23, 4))
        self.assertEqual(tuple(outputs["temporal_weights"].shape), (3, 23))
        self.assertTrue(torch.all(outputs["modality_weights"][:, :, 2] == 0))

    def test_window_replicated_and_bag_static_share_attention_fusion(self):
        from daily_multimodal.daily_affect.model import DailyAffectConfig, DailyAffectOrdinalModel
        from daily_multimodal.daily_affect.training import supervision_tensors

        tokens = torch.randn(2, 23, 4, 256)
        mask = torch.ones(2, 23, 4, dtype=torch.bool)
        mask[0, 0] = False
        window_model = DailyAffectOrdinalModel(DailyAffectConfig(model_id="window_replicated", hidden_dim=16))
        bag_model = DailyAffectOrdinalModel(DailyAffectConfig(model_id="bag_static", hidden_dim=16))
        window_outputs = window_model(tokens, mask)
        bag_outputs = bag_model(tokens, mask)
        self.assertTrue(window_model.uses_window_attention)
        self.assertTrue(bag_model.uses_window_attention)
        self.assertEqual(set(window_model.state_dict()), set(bag_model.state_dict()))
        self.assertEqual(tuple(window_outputs["window_class_logits"].shape), (2, 23, 5))
        self.assertNotIn("window_class_logits", bag_outputs)
        labels = torch.as_tensor([1, 4], dtype=torch.long)
        raw_labels = labels.to(dtype=torch.float32) + 1.0
        subjects = torch.as_tensor([3, 8], dtype=torch.long)
        loss_logits, loss_labels, loss_raw, rank_scores, rank_subjects = supervision_tensors(
            window_outputs,
            labels,
            raw_labels,
            subjects,
        )
        self.assertEqual(tuple(loss_logits.shape), (45, 5))
        self.assertEqual(tuple(rank_scores.shape), (45,))
        self.assertEqual(loss_labels.tolist().count(1), 22)
        self.assertEqual(loss_labels.tolist().count(4), 23)
        self.assertEqual(loss_raw.tolist().count(2.0), 22)
        self.assertEqual(rank_subjects.tolist().count(3), 22)

    def test_focused_dynamic_ablation_model_ids(self):
        from daily_multimodal.daily_affect.model import DailyAffectConfig, DailyAffectOrdinalModel

        tokens = torch.randn(2, 23, 4, 256)
        mask = torch.ones(2, 23, 4, dtype=torch.bool)
        for model_id in ("dynamic_kernel_no_prior", "dynamic_kernel_prior_uniform", "dynamic_fixed_short", "dynamic_fixed_medium", "dynamic_fixed_long"):
            model = DailyAffectOrdinalModel(DailyAffectConfig(model_id=model_id, hidden_dim=16, adapter_mode="per_modality"))
            outputs = model(tokens, mask)
            self.assertEqual(tuple(outputs["class_logits"].shape), (2, 5))
            self.assertEqual(tuple(outputs["kernel_mixture"].shape), (2, 3))
            self.assertTrue(torch.allclose(outputs["temporal_weights"].sum(dim=1), torch.ones(2), atol=1e-5))
            if model_id.startswith("dynamic_fixed_"):
                self.assertTrue(torch.all(outputs["kernel_mixture"].sum(dim=1) == 1))

    def test_global_kernel_and_entropy_probe_route(self):
        from daily_multimodal.daily_affect.model import DailyAffectConfig, DailyAffectOrdinalModel

        tokens = torch.randn(2, 23, 4, 256)
        mask = torch.ones(2, 23, 4, dtype=torch.bool)
        global_kernel = DailyAffectOrdinalModel(DailyAffectConfig(model_id="global_kernel_no_prior", hidden_dim=16))
        global_outputs = global_kernel(tokens, mask)
        self.assertTrue(torch.allclose(global_outputs["temporal_weights"].sum(dim=1), torch.ones(2), atol=1e-5))
        entropy_route = DailyAffectOrdinalModel(
            DailyAffectConfig(
                model_id="prior_uniform",
                hidden_dim=16,
                probe_kind="categorical",
                difficulty_mode="entropy",
            )
        )
        entropy_outputs = entropy_route(tokens, mask)
        self.assertIn("probe_class_logits", entropy_outputs)
        self.assertTrue(entropy_route.uses_difficulty_penalty)

    def test_ordinal_loss_penalizes_distant_misallocation_more(self):
        from daily_multimodal.daily_affect.losses import cumulative_probability_ordinal_loss

        labels = torch.tensor([4], dtype=torch.long)
        adjacent_logits = torch.log(torch.tensor([[0.001, 0.001, 0.001, 0.996, 0.001]], dtype=torch.float32))
        distant_logits = torch.log(torch.tensor([[0.996, 0.001, 0.001, 0.001, 0.001]], dtype=torch.float32))
        self.assertGreater(
            float(cumulative_probability_ordinal_loss(distant_logits, labels)),
            float(cumulative_probability_ordinal_loss(adjacent_logits, labels)),
        )

    def test_expected_score_huber_loss_is_zero_at_target_and_increases_with_error(self):
        from daily_multimodal.daily_affect.losses import expected_score_huber_loss

        target = torch.tensor([2.0, 4.0])
        self.assertEqual(float(expected_score_huber_loss(target, target)), 0.0)
        near = expected_score_huber_loss(torch.tensor([2.2, 3.8]), target)
        far = expected_score_huber_loss(torch.tensor([3.8, 2.2]), target)
        self.assertGreater(float(far), float(near))

    def test_modality_dropout_and_routing_schedule(self):
        from daily_multimodal.daily_affect.training import apply_modality_dropout, difficulty_lambda_for_epoch

        tokens = torch.ones(3, 23, 4, 256)
        mask = torch.ones(3, 23, 4, dtype=torch.bool)
        _dropped_tokens, dropped_mask = apply_modality_dropout(
            tokens,
            mask,
            probability=1.0,
            rng=np.random.default_rng(12),
        )
        self.assertTrue(torch.all(dropped_mask.any(dim=1).sum(dim=1) == 3))
        self.assertEqual(difficulty_lambda_for_epoch(0, target=0.25, warmup_epochs=2, ramp_epochs=3), 0.0)
        self.assertAlmostEqual(difficulty_lambda_for_epoch(2, target=0.25, warmup_epochs=2, ramp_epochs=3), 0.25 / 3.0)
        self.assertEqual(difficulty_lambda_for_epoch(4, target=0.25, warmup_epochs=2, ramp_epochs=3), 0.25)

    def test_prior_guidance_and_difficulty_are_separate_switches(self):
        from daily_multimodal.daily_affect.model import DailyAffectConfig, DailyAffectOrdinalModel

        no_prior = DailyAffectOrdinalModel(DailyAffectConfig(model_id="dynamic_kernel_no_prior", hidden_dim=16))
        global_no_prior = DailyAffectOrdinalModel(
            DailyAffectConfig(model_id="global_kernel_no_prior", hidden_dim=16, difficulty_mode="ordinal")
        )
        prior_only = DailyAffectOrdinalModel(DailyAffectConfig(model_id="dynamic_kernel_prior_uniform", hidden_dim=16))
        full = DailyAffectOrdinalModel(DailyAffectConfig(model_id="dynamic_kernel", hidden_dim=16))
        self.assertFalse(no_prior.uses_prior_guidance)
        self.assertFalse(no_prior.uses_difficulty_penalty)
        self.assertFalse(global_no_prior.uses_prior_guidance)
        self.assertEqual(global_no_prior.resolved_difficulty_mode, "none")
        self.assertFalse(global_no_prior.uses_difficulty_penalty)
        self.assertTrue(prior_only.uses_prior_guidance)
        self.assertFalse(prior_only.uses_difficulty_penalty)
        self.assertTrue(full.uses_prior_guidance)
        self.assertTrue(full.uses_difficulty_penalty)

    def test_static_temporal_policies_select_expected_windows(self):
        from daily_multimodal.daily_affect.model import DailyAffectConfig, DailyAffectOrdinalModel

        tokens = torch.randn(2, 23, 4, 256)
        mask = torch.ones(2, 23, 4, dtype=torch.bool)
        last_window = DailyAffectOrdinalModel(DailyAffectConfig(model_id="bag_static", hidden_dim=16, temporal_policy="last_10s"))
        recent_windows = DailyAffectOrdinalModel(DailyAffectConfig(model_id="bag_static", hidden_dim=16, temporal_policy="last_30s"))
        short_kernel = DailyAffectOrdinalModel(DailyAffectConfig(model_id="bag_static", hidden_dim=16, temporal_policy="kernel_short"))
        self.assertTrue(torch.all(last_window(tokens, mask)["temporal_weights"][:, :-1] == 0))
        self.assertTrue(torch.all(recent_windows(tokens, mask)["temporal_weights"][:, :-5] == 0))
        self.assertGreater(float(short_kernel(tokens, mask)["temporal_weights"][:, -1].mean()), float(short_kernel(tokens, mask)["temporal_weights"][:, 0].mean()))

    def test_run_daily_affect_run_cpu_smoke(self):
        from daily_multimodal.daily_affect.training import load_bag_dataset, run_daily_affect_run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_path, splits_root, emb_root = _write_synthetic_inputs(root, event_count=6)
            build_daily_affect_bags(
                index_path=index_path,
                splits_root=splits_root,
                embeddings_root=emb_root,
                protocol="cross_day",
                route_id="B0_Wphysio_full",
                out_dir=root / "out/bags/cross_day/B0_Wphysio_full/seed_240800",
                eeg_seed=240800,
                wear_seed=240800,
            )
            dataset = load_bag_dataset(root / "out/bags/cross_day/B0_Wphysio_full/seed_240800/ema_bags.npz")
            result = run_daily_affect_run(
                dataset=dataset,
                protocol="cross_day",
                model_id="prior_ordD_uniform",
                normalization="shared",
                adapter_mode="per_modality",
                seed=7,
                run_dir=root / "run",
                epochs=1,
                batch_size=2,
                learning_rate=1e-3,
                weight_decay=0.0,
                dropout=0.0,
                hidden_dim=16,
                patience=2,
                lambda_d=0.25,
                probe_loss_weight=0.1,
                selection_metric="qwk",
                class_balanced_loss=False,
                calibrate_temperature=True,
                device="cpu",
                include_diagnostics=True,
            )
            self.assertEqual(result["model_id"], "prior_ordD_uniform")
            self.assertEqual(result["normalization"], "shared")
            self.assertEqual(result["adapter_mode"], "per_modality")
            self.assertTrue((root / "run/metrics.json").is_file())
            self.assertTrue((root / "run/predictions.npz").is_file())
            self.assertTrue((root / "run/diagnostics.npz").is_file())
            self.assertTrue((root / "run/test_predictions.csv").is_file())
            self.assertTrue(result["probe_calibration"]["temperatures"])

    def test_summary_script_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "runs"
            run_dir = run_root / "phase0/cross_day/B0_Wphysio_full/shared/seed_1"
            run_dir.mkdir(parents=True)
            metrics = {
                "protocol": "cross_day",
                "route_id": "B0_Wphysio_full",
                "model_id": "bag_static",
                "normalization": "shared",
                "seed": 1,
                "test": {
                    "qwk": 0.1,
                    "macro_f1": 0.2,
                    "accuracy": 0.3,
                    "ordinal_mae": 1.0,
                    "expected_rmse": 1.1,
                    "expected_raw_r": 0.2,
                    "expected_within_subject_centered_r": 0.3,
                },
            }
            (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            script = Path(__file__).resolve().parents[1] / "scripts" / "daily_affect" / "76_summarize_daily_affect_results.py"
            result = subprocess.run(
                [sys.executable, str(script), "--run-root", str(run_root), "--out-root", str(root / "reports")],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("run_count=1", result.stdout)
            self.assertTrue((root / "reports/daily_affect_ordinal_summary.json").is_file())
            summary = json.loads((root / "reports/daily_affect_ordinal_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["summary"][0]["test_expected_raw_r_mean"], 0.2)

    @unittest.skipIf(importlib.util.find_spec("matplotlib") is None, "matplotlib is not installed in this runtime")
    def test_diagnostic_plot_script_aggregates_seeds_into_atlases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "runs"
            for normalization, weights in (
                ("shared", [0.20, 0.40, 0.30, 0.10]),
                ("per_modality", [0.30, 0.20, 0.20, 0.30]),
            ):
                for seed in (1, 2):
                    run_dir = run_root / "runs/cross_day/A1_Wphysio_full/dynamic_kernel" / normalization / f"seed_{seed}"
                    run_dir.mkdir(parents=True)
                    metrics = {
                        "protocol": "cross_day",
                        "route_id": "A1_Wphysio_full",
                        "model_id": "dynamic_kernel",
                        "normalization": normalization,
                        "seed": seed,
                        "test": {"qwk": 0.10 + 0.05 * seed + (0.10 if normalization == "shared" else 0.0)},
                    }
                    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
                    np.savez(
                        run_dir / "diagnostics.npz",
                        modality_weights=np.tile(np.asarray(weights, dtype=np.float64), (3, 23, 1)),
                        modality_difficulty=np.tile(np.asarray([0.4, 0.5, 0.6, 0.7], dtype=np.float64), (3, 1)),
                        temporal_weights=np.full((3, 23), 1.0 / 23.0, dtype=np.float64),
                    )
                    np.savez(
                        run_dir / "predictions.npz",
                        test_index=np.asarray([0, 1, 2], dtype=np.int64),
                        label_zero_based=np.asarray([0, 1, 2], dtype=np.int64),
                        test_predicted_class=np.asarray([0, 1, seed], dtype=np.int64),
                    )
            out_root = root / "figures"
            legacy = out_root / "cross_day/modality_weights_legacy.png"
            legacy.parent.mkdir(parents=True)
            legacy.write_bytes(b"legacy")
            script = Path(__file__).resolve().parents[1] / "scripts" / "daily_affect" / "77_plot_daily_affect_diagnostics.py"
            result = subprocess.run(
                [sys.executable, str(script), "--run-root", str(run_root), "--out-root", str(out_root), "--clean"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("run_count=4", result.stdout)
            self.assertIn("group_count=2", result.stdout)
            self.assertIn("figure_count=2", result.stdout)
            self.assertFalse(legacy.exists())
            self.assertEqual(
                sorted(path.name for path in out_root.glob("*.png")),
                ["daily_affect_confusion_atlas.png", "daily_affect_diagnostics_atlas_cross_day.png"],
            )
            manifest = json.loads((out_root / "daily_affect_figure_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["figure_count"], 2)
            self.assertEqual(manifest["figures"][0]["groups"][0]["seed_count"], 2)

    def test_prior_guidance_report_script_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "runs"
            for model_id, qwk, mae in (("state_uniform", 0.1, 1.0), ("prior_uniform", 0.2, 0.9)):
                run_dir = run_root / "runs/cross_day/A1_Wphysio_full" / model_id / "per_modality/seed_1"
                run_dir.mkdir(parents=True)
                metrics = {
                    "protocol": "cross_day",
                    "route_id": "A1_Wphysio_full",
                        "model_id": model_id,
                        "normalization": "per_modality",
                        "adapter_mode": "per_modality",
                        "experiment_id": "state_matrix_native",
                    "seed": 1,
                    "test": {
                        "qwk": qwk,
                        "macro_f1": qwk,
                        "ordinal_mae": mae,
                        "expected_rmse": mae,
                        "expected_raw_r": qwk,
                        "expected_within_subject_centered_r": qwk,
                    },
                }
                (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            script = Path(__file__).resolve().parents[1] / "scripts" / "daily_affect" / "81_report_daily_affect_prior_guidance.py"
            result = subprocess.run(
                [sys.executable, str(script), "--run-root", str(run_root), "--out-root", str(root / "reports")],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("run_count=2", result.stdout)
            report = json.loads((root / "reports/prior_guidance_ablation_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["comparison_summary"][0]["comparison_id"], "state_prior_only")


def _write_synthetic_inputs(root: Path, *, event_count: int) -> tuple[Path, Path, Path]:
    index_path = root / "index/eeg_aligned_window_index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for event in range(event_count):
        for window in range(23):
            row_id = event * 23 + window
            rows.append(
                {
                    "sample_id": f"eeg_{row_id:06d}",
                    "event_id": f"event_{event:03d}",
                    "event_window_id": window,
                    "subject_id": f"sub-{event % 3 + 1:02d}",
                    "day_id": f"2026-01-{event + 1:02d}",
                    "labels": {"fatigue": float(event % 5 + 1)},
                }
            )
    index_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    splits_root = root / "splits"
    protocol_root = splits_root / "cross_day"
    protocol_root.mkdir(parents=True)
    leaf_by_event = ["pretrain", "finetune", "val", "test"]
    if event_count > 4:
        leaf_by_event.extend(["pretrain"] * (event_count - 4))
    for leaf in ("pretrain", "finetune", "val", "test"):
        indices = []
        for event, name in enumerate(leaf_by_event[:event_count]):
            if name == leaf:
                indices.extend(range(event * 23, event * 23 + 23))
        (protocol_root / f"{leaf}.json").write_text(json.dumps(indices), encoding="utf-8")
    emb_root = root / "embeddings"
    sample_id = np.asarray([row["sample_id"] for row in rows], dtype=object)
    rng = np.random.default_rng(5)
    _write_npz(emb_root / "eeg/eeg_eegpt_eeg23win_embeddings.npz", sample_id, "eeg_emb", 0, rng)
    _write_npz(emb_root / "wear/wear_physio_preprocessed_eeg23win_embeddings.npz", sample_id, "wear_emb", 1, rng)
    _write_npz(emb_root / "video/video_B0_2xroi_eeg23win_embeddings.npz", sample_id, "video_emb", 2, rng)
    _write_npz(emb_root / "audio/audio_opensmile_eeg23win_embeddings.npz", sample_id, "audio_emb", 3, rng)
    return index_path, splits_root, emb_root


def _write_npz(path: Path, sample_id: np.ndarray, key: str, modality_index: int, rng: np.random.Generator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((len(sample_id), 4), dtype=np.int8)
    mask[:, modality_index] = 1
    kwargs = {
        "sample_id": sample_id,
        key: rng.standard_normal((len(sample_id), 256)).astype(np.float32),
        "modality_mask": mask,
    }
    np.savez_compressed(path, **kwargs)


if __name__ == "__main__":
    unittest.main()
