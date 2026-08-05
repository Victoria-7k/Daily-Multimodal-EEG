from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
EEG_ROOT = Path("/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new")
SPLITS_ROOT = Path("/vePFS-0x0d/DailyEEG/splits_new")
EMB_ROOT = Path("/vePFS-0x0d/DailyEEG_multimodal/embeddings")


def main() -> None:
    _inspect_npy(EEG_ROOT / "X.npy")
    _inspect_npy(EEG_ROOT / "y.npy")
    _inspect_npy(EEG_ROOT / "sub.npy")
    _inspect_npy(EEG_ROOT / "d.npy")
    _inspect_npy(EEG_ROOT / "ts.npy")
    _inspect_index(ROOT / "index/eeg_aligned_window_index.jsonl")
    _inspect_npz(EMB_ROOT / "video/video_B0_2xroi_eeg23win_embeddings.npz")
    _inspect_npz(EMB_ROOT / "audio/audio_opensmile_eeg23win_embeddings.npz")
    _inspect_npz(EMB_ROOT / "wear/wear_physio_preprocessed_eeg23win_embeddings.npz")
    _inspect_npz(EMB_ROOT / "wear/wear_deep_sequence_preprocessed_eeg23win_embeddings.npz")
    _inspect_splits(SPLITS_ROOT)


def _inspect_npy(path: Path) -> None:
    arr = np.load(path, mmap_mode="r", allow_pickle=True)
    print(f"NPY {path}")
    print(f"  shape={arr.shape} dtype={arr.dtype}")
    try:
        print(f"  first={_short(arr[0])}")
    except Exception as exc:
        print(f"  first_error={exc}")


def _inspect_index(path: Path) -> None:
    print(f"INDEX {path}")
    first = []
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if count < 3:
                first.append(json.loads(line))
            count += 1
    print(f"  rows={count}")
    for idx, row in enumerate(first):
        print(f"  row{idx}_keys={sorted(row.keys())}")
        print(f"  row{idx}={json.dumps(row, ensure_ascii=False)[:1200]}")


def _inspect_npz(path: Path) -> None:
    print(f"NPZ {path}")
    with np.load(path, allow_pickle=True) as loaded:
        print(f"  keys={loaded.files}")
        for key in loaded.files:
            arr = loaded[key]
            print(f"  {key}: shape={arr.shape} dtype={arr.dtype} first={_short(arr[0]) if arr.shape else _short(arr)}")


def _inspect_splits(path: Path) -> None:
    print(f"SPLITS {path}")
    for file_path in sorted(path.rglob("*")):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(path)
        print(f"  FILE {rel} size={file_path.stat().st_size}")
        if file_path.suffix == ".npy":
            arr = np.load(file_path, allow_pickle=True)
            print(f"    npy shape={arr.shape} dtype={arr.dtype} first={_short(arr[0]) if arr.shape else _short(arr)}")
        else:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            print(f"    text={text[:1000].replace(chr(10), ' | ')}")


def _short(value: object) -> str:
    text = repr(value)
    return text[:500]


if __name__ == "__main__":
    main()
