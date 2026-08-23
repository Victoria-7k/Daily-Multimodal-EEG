import json
import shutil
import unittest
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

from daily_multimodal.training.wear_moment_matrix import (
    DEFAULT_PROTOCOLS,
    EMBEDDING_DIM,
    MATRIX_STEPS,
    WEAR_MOMENT_PROFILES,
    _aligned_sequence_matrix,
    _load_window_sequences,
    _strategy_for_profile,
    load_split_protocols,
    load_wear_moment_dataset,
    run_preflight,
    write_wear_token_npz,
)

_TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / "outputs" / ".test_tmp" / "wear_moment_tests"


def _tmp_dir(prefix: str) -> str:
    """沙箱只允许写工作区，且拦截 tempfile.mkdtemp 创建的目录；用 mkdir + uuid 自建。"""
    _TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    candidate = _TEST_TMP_ROOT / f"{prefix}{uuid.uuid4().hex[:10]}"
    candidate.mkdir()
    return str(candidate)


def _write_fake_wear_csvs(root: Path, prefix: str = "day1") -> dict[str, str]:
    """写出与 wear_real.TARGET_COLUMNS 一致的假 CSV，返回 {modality: path}。"""
    start = datetime(2025, 1, 10, 14, 22, 9)
    ppg_path = root / f"{prefix}_PPG.csv"
    gsr_path = root / f"{prefix}_GSR.csv"
    acc_path = root / f"{prefix}_ACC.csv"
    with ppg_path.open("w", encoding="utf-8") as handle:
        handle.write("PPG,csv_time_PPG\n")
        for step in range(300):
            handle.write(f"{0.5 + 0.1 * np.sin(step / 5)},{start.isoformat(' ')}\n")
    with gsr_path.open("w", encoding="utf-8") as handle:
        handle.write("GSR,csv_time_GSR\n")
        for step in range(300):
            handle.write(f"{0.2 + 0.01 * step},{start.isoformat(' ')}\n")
    with acc_path.open("w", encoding="utf-8") as handle:
        handle.write("Motion_dataX,Motion_dataY,Motion_dataZ,csv_time_motion\n")
        for step in range(300):
            handle.write(f"{step / 100.0},{-step / 200.0},{9.8 + step / 500.0},{start.isoformat(' ')}\n")
    return {"ppg": str(ppg_path), "gsr": str(gsr_path), "acc": str(acc_path)}


def _write_fake_index(root: Path, row_count: int, *, label_names=None) -> Path:
    names = label_names or ("inspired", "alert", "determined", "attentive", "active", "hostile", "nervous", "upset", "afraid", "ashamed", "fatigue")
    rows = []
    for index in range(row_count):
        labels = [1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, float(2 + (index % 3))]
        rows.append(
            {
                "sample_id": f"eeg_{index:06d}",
                "event_id": f"sub-01_day-000_event-{index:05d}",
                "subject_id": str(index % 5 + 1),
                "day_id": str(index // 20),
                "event_window_id": index % 23,
                "label_names": names,
                "labels": labels,
                "window_start_seconds": 1000.0 + index,
                "window_end_seconds": 1010.0 + index,
            }
        )
    path = root / "index.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    return path


def _write_fake_wear_meta(root: Path, index_rows: int, source_map: dict[str, str], *, mask_first: int = 3) -> Path:
    """source_map: {modality: path}；前 mask_first 行 masked。"""
    npz_path = root / "wear_meta.npz"
    sample_id = np.asarray([f"eeg_{i:06d}" for i in range(index_rows)], dtype=object)
    mask = np.ones(index_rows, dtype=np.int8)
    mask[:mask_first] = 0
    payload = []
    for index in range(index_rows):
        if index < mask_first:
            payload.append(json.dumps({"ppg": "", "gsr": "", "acc": ""}))
        else:
            payload.append(json.dumps({m: str(path) for m, path in source_map.items()}))
    np.savez_compressed(
        npz_path,
        sample_id=sample_id,
        wear_mask=mask,
        source_wear_file=np.asarray(payload, dtype=object),
        window_start_time=np.asarray(["2025-01-10 14:22:09"] * index_rows, dtype=object),
        window_end_time=np.asarray(["2025-01-10 14:22:19"] * index_rows, dtype=object),
    )
    return npz_path


def _write_splits(root: Path, row_count: int, protocols: tuple[str, ...] = DEFAULT_PROTOCOLS) -> Path:
    splits_root = root / "splits"
    for protocol in protocols:
        protocol_dir = splits_root / protocol
        protocol_dir.mkdir(parents=True, exist_ok=True)
        indices = np.arange(row_count)
        for name, values in (
            ("pretrain", indices[: row_count // 2]),
            ("finetune", indices[row_count // 2 : row_count * 3 // 4]),
            ("val", indices[row_count * 3 // 4 : row_count * 9 // 10]),
            ("test", indices[row_count * 9 // 10 :]),
        ):
            (protocol_dir / f"{name}.json").write_text(json.dumps({"indices": values.tolist()}), encoding="utf-8")
    return splits_root


class WearMomentMatrixTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(_TEST_TMP_ROOT, ignore_errors=True)

    def test_profiles_and_strategy_mapping(self):
        self.assertEqual(_strategy_for_profile("wear_moment_frozen_v1"), "frozen")
        self.assertEqual(_strategy_for_profile("wear_moment_partial_ft_v1"), "partial")
        with self.assertRaises(ValueError):
            _strategy_for_profile("wear_moment_large_frozen_v1")
        self.assertEqual(WEAR_MOMENT_PROFILES, {"wear_moment_frozen_v1", "wear_moment_partial_ft_v1"})

    def test_window_sequences_shapes_and_aligned_matrix(self):
        tmp = _tmp_dir("t")
        root = Path(tmp)
        csv_paths = _write_fake_wear_csvs(root)
        sequences = _load_window_sequences(
            csv_paths,
            wear_raw_root=root,
            window_start=datetime(2025, 1, 10, 14, 22, 9),
            window_end=datetime(2025, 1, 10, 14, 22, 19),
        )
        self.assertIsNotNone(sequences)
        self.assertEqual(sequences["ppg"].shape, (640, 1))
        self.assertEqual(sequences["gsr"].shape, (320, 1))
        self.assertEqual(sequences["acc"].shape, (320, 3))
        matrix = _aligned_sequence_matrix(sequences, target_steps=MATRIX_STEPS)
        self.assertEqual(matrix.shape, (MATRIX_STEPS, 5))
        self.assertTrue(np.isfinite(matrix).all())

    def test_dataset_order_validation_and_masking(self):
        tmp = _tmp_dir("t")
        root = Path(tmp)
        csv_paths = _write_fake_wear_csvs(root)
        index_path = _write_fake_index(root, row_count=30)
        meta_path = _write_fake_wear_meta(root, index_rows=30, source_map=csv_paths, mask_first=3)

        dataset = load_wear_moment_dataset(index_path, meta_path, wear_raw_root=root)
        self.assertEqual(dataset.row_count, 30)
        self.assertEqual(int(dataset.mask.sum()), 27)
        self.assertTrue(np.array_equal(dataset.matrices[:3], np.zeros((3, 5, MATRIX_STEPS))))
        self.assertTrue(np.isfinite(dataset.matrices[3:]).all())
        self.assertAlmostEqual(float(dataset.target[0]), 2.0)

        bad_meta = root / "bad_meta.npz"
        np.savez_compressed(
            bad_meta,
            sample_id=np.asarray([f"eeg_{i:06d}" for i in range(29, -1, -1)], dtype=object),
            wear_mask=np.ones(30, dtype=np.int8),
            source_wear_file=np.asarray([json.dumps(csv_paths)] * 30, dtype=object),
            window_start_time=np.asarray(["2025-01-10 14:22:09"] * 30, dtype=object),
            window_end_time=np.asarray(["2025-01-10 14:22:19"] * 30, dtype=object),
        )
        with self.assertRaises(ValueError):
            load_wear_moment_dataset(index_path, bad_meta, wear_raw_root=root)

    def test_matrix_cache_roundtrip(self):
        tmp = _tmp_dir("t")
        root = Path(tmp)
        csv_paths = _write_fake_wear_csvs(root)
        index_path = _write_fake_index(root, row_count=20)
        meta_path = _write_fake_wear_meta(root, index_rows=20, source_map=csv_paths, mask_first=2)
        cache = root / "matrix_cache.npz"
        first = load_wear_moment_dataset(index_path, meta_path, wear_raw_root=root, matrix_cache=cache)
        self.assertTrue(cache.is_file())
        second = load_wear_moment_dataset(index_path, meta_path, wear_raw_root=root, matrix_cache=cache)
        self.assertTrue(np.array_equal(first.matrices, second.matrices))

    def test_token_npz_contract(self):
        tmp = _tmp_dir("t")
        root = Path(tmp)
        dataset = load_wear_moment_dataset(
            _write_fake_index(root, row_count=12),
            _write_fake_wear_meta(root, index_rows=12, source_map=_write_fake_wear_csvs(root), mask_first=2),
            wear_raw_root=root,
        )
        splits = load_split_protocols(_write_splits(root, 12), ("cross_day",), dataset)
        tokens = np.random.default_rng(0).normal(size=(12, EMBEDDING_DIM)).astype(np.float32)
        out = root / "tokens" / "cross_day" / "wear_moment_frozen_v1" / "seed_240800.npz"
        write_wear_token_npz(
            tokens,
            out,
            dataset=dataset,
            profile="wear_moment_frozen_v1",
            protocol="cross_day",
            seed=240800,
            split=splits["cross_day"],
            train_supervision="frozen_encoder_trainable_projection_head",
            source_checkpoint="fake-checkpoint",
            checkpoint_sha256="deadbeef",
        )
        with np.load(out, allow_pickle=True) as loaded:
            self.assertEqual(loaded["wear_emb"].shape, (12, EMBEDDING_DIM))
            self.assertEqual(loaded["wear_mask"].tolist(), [0, 0] + [1] * 10)
            self.assertEqual(loaded["modality_mask"][:, 1].tolist(), loaded["wear_mask"].tolist())
            self.assertEqual(loaded["modality_mask"][:, 0].tolist(), [0] * 12)
            self.assertEqual(str(loaded["train_supervision"][0]), "frozen_encoder_trainable_projection_head")
            self.assertEqual(loaded["seed"].tolist(), [240800])

    def test_preflight_reports_coverage_without_torch(self):
        tmp = _tmp_dir("t")
        root = Path(tmp)
        csv_paths = _write_fake_wear_csvs(root)
        index_path = _write_fake_index(root, row_count=30)
        meta_path = _write_fake_wear_meta(root, index_rows=30, source_map=csv_paths, mask_first=3)
        out_json = root / "preflight.json"
        out_md = root / "preflight.md"
        result = run_preflight(
            index_path,
            meta_path,
            wear_raw_root=root,
            max_rows=30,
            out_json=out_json,
            out_md=out_md,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["row_count"], 30)
        self.assertEqual(result["mask_sum"], 27)
        self.assertTrue(out_json.is_file())
        self.assertTrue(out_md.is_file())


if __name__ == "__main__":
    unittest.main()
