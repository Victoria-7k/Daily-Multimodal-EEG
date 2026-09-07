# Multi-Label Scripts

固定 token 的 11-label fusion baseline 入口放在这里；它与 scalar fatigue calibration 链路分开。

| 脚本 | 用途 |
| --- | --- |
| `48_run_fixed_tokens_multilabel_fusion.py` | 训练 fixed/frozen EEG/Wear/Video/Audio token 的 11-label fusion baseline，并逐标签报告指标。 |
