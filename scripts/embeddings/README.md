# Embedding Scripts

生成或评估窗口级 256D token 的入口放在这里。

| 脚本 | 用途 |
| --- | --- |
| `12_extract_audio_embeddings.py` | 生成 openSMILE eGeMAPS audio 256D embedding。 |
| `15_extract_wear_embeddings.py` | 生成 Wphysio / Wdeep wearable 256D embedding。 |
| `27_extract_dinov2_roi_embeddings.py` | 生成 B0/A1/A2 2x ROI DINOv2 video 256D embedding。 |
| `34_run_eeg_encoder_matrix.py` | 生成并评估 EEG-only encoder 路线，可导出 256D EEG token。 |
