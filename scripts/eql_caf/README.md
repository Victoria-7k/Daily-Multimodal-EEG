# EQL-CAF Scripts

EQL-CAF temporal-token 探索线。当前 `59`-`62` 由窗口级 token 重复生成 5 个 2 秒 token，只验证接口与管线，不作为真实 lag-aware 证据。

| 脚本 | 用途 |
| --- | --- |
| `57_run_eql_caf_phase0_baselines.py` | 生成或执行 EQL-CAF Phase 0 baseline command matrix。 |
| `58_build_temporal_token_index.py` | 生成 5 x 2s temporal token 边界与审计 manifest。 |
| `59_extract_eeg_temporal_tokens.py` | EEG repeat-token smoke。 |
| `60_extract_wear_temporal_tokens.py` | Wear repeat-token smoke。 |
| `61_extract_video_temporal_tokens.py` | Video/Face repeat-token smoke。 |
| `62_extract_audio_temporal_tokens.py` | Audio repeat-token smoke。 |
| `63_pack_eql_caf_tokens.py` | 合并四模态 temporal token NPZ，并校验 sample_id、shape、mask 和 quality 字段。 |
| `64_run_eql_caf_fatigue_matrix.py` | 读取 packed temporal token，训练 B1/B2/M1-M5 fatigue 矩阵。 |
