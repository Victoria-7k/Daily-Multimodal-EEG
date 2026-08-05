# Final EEG-Aligned Multimodal Alignment Report

## Scope

This report covers the EEG-aligned 23-window multimodal artifacts copied to the new server.

Canonical EEG rows: `28819`. Canonical index: `/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/index/eeg_aligned_window_index.jsonl`.

## Artifact Summary

| Modality | New server path | Embedding shape | Mask sum | Missing | NaN | Aligned | SHA256 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `video_B0` | `/vePFS-0x0d/DailyEEG_multimodal/embeddings/video/video_B0_2xroi_eeg23win_embeddings.npz` | `(28819, 256)` | 18017 | 10802 | 0 | True | `133b36d6af45c2eed5b2945b223490b3b243eea2a404c9ffedc38fe2601581b3` |
| `video_A1` | `/vePFS-0x0d/DailyEEG_multimodal/embeddings/video/video_A1_2xroi_eeg23win_embeddings.npz` | `(28819, 256)` | 18021 | 10798 | 0 | True | `a82485929f5ae86bc18de394f8ea407e7a8d90a727e05aa898ff93d2379fee14` |
| `video_A2` | `/vePFS-0x0d/DailyEEG_multimodal/embeddings/video/video_A2_2xroi_eeg23win_embeddings.npz` | `(28819, 256)` | 17992 | 10827 | 0 | True | `d61bc37e808722bba35ec72573555dfce50803abaeab6e95bb616878ac2d0cea` |
| `audio` | `/vePFS-0x0d/DailyEEG_multimodal/embeddings/audio/audio_opensmile_eeg23win_embeddings.npz` | `(28819, 256)` | 17924 | 10895 | 0 | True | `110d01aa08b5691b4c6a1ed5ccf65c5e9dff2a3f2738d015b0175b3342080b73` |
| `wear_physio` | `/vePFS-0x0d/DailyEEG_multimodal/embeddings/wear/wear_physio_preprocessed_eeg23win_embeddings.npz` | `(28819, 256)` | 24127 | 4692 | 0 | True | `311c86ee4f4d8706d975b55d6719be1c7afe7a6e9bbeb073cdc1a211ae97a5f4` |
| `wear_deep` | `/vePFS-0x0d/DailyEEG_multimodal/embeddings/wear/wear_deep_sequence_preprocessed_eeg23win_embeddings.npz` | `(28819, 256)` | 24127 | 4692 | 0 | True | `b8783d15859e0dd1f7202e500d2faf610004bea0e68d0a40fb0e8aa5df8279fd` |

## Transfer Verification

Video B0/A1/A2 sha256 values were recomputed on the new server and match the old-server sha256 files exactly. Audio and wear sha256 values were verified in the earlier transfer step and are listed above.

## Missing Window Lists

Full missing-window lists are stored in `final_missing_windows_by_modality.json` in the same reports directory after transfer to the new server.

- `video_B0`: 10802 missing windows; first 10 sample ids: `['eeg_000000', 'eeg_000001', 'eeg_000002', 'eeg_000003', 'eeg_000004', 'eeg_000005', 'eeg_000006', 'eeg_000007', 'eeg_000008', 'eeg_000009']`
- `video_A1`: 10798 missing windows; first 10 sample ids: `['eeg_000000', 'eeg_000001', 'eeg_000002', 'eeg_000003', 'eeg_000004', 'eeg_000005', 'eeg_000006', 'eeg_000007', 'eeg_000008', 'eeg_000009']`
- `video_A2`: 10827 missing windows; first 10 sample ids: `['eeg_000000', 'eeg_000001', 'eeg_000002', 'eeg_000003', 'eeg_000004', 'eeg_000005', 'eeg_000006', 'eeg_000007', 'eeg_000008', 'eeg_000009']`
- `audio`: 10895 missing windows; first 10 sample ids: `['eeg_000000', 'eeg_000001', 'eeg_000002', 'eeg_000003', 'eeg_000004', 'eeg_000005', 'eeg_000006', 'eeg_000007', 'eeg_000008', 'eeg_000009']`
- `wear_physio`: 4692 missing windows; first 10 sample ids: `['eeg_000000', 'eeg_000001', 'eeg_000002', 'eeg_000003', 'eeg_000004', 'eeg_000005', 'eeg_000006', 'eeg_000007', 'eeg_000008', 'eeg_000009']`
- `wear_deep`: 4692 missing windows; first 10 sample ids: `['eeg_000000', 'eeg_000001', 'eeg_000002', 'eeg_000003', 'eeg_000004', 'eeg_000005', 'eeg_000006', 'eeg_000007', 'eeg_000008', 'eeg_000009']`

## Split Compatibility

All modality files retain `28819` rows and the canonical `eeg_sample_index = 0..28818` order, so `/vePFS-0x0d/DailyEEG/splits_new` can be reused without row deletion or split recomputation. Missing modality availability is represented by the modality mask only.

## No-Move Check

The transfer copied generated embeddings and reports. Original old-server video/audio sources and new-server wear raw data were not moved by this finalization step.

Evidence status: generated from old-server `.npz` artifacts and new-server sha256 checks on 2026-07-30.
