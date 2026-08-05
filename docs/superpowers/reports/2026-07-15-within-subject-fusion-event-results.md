# Within-Subject Fusion Results

## Event-Grouped Execution

- Protocol: `event_grouped_5fold`
- Target: `fatigue`
- Subjects: 14 (`sub-02` through `sub-15`)
- Paired cohort: 7,368 windows
- Matrix: 20 experiments x 3 models
- Completed jobs: 840/840
- Primary metric: event-level subject-macro Pearson
- Auxiliary metrics: event macro RMSE and within-subject-centered pooled Pearson

## Event-Grouped Best Attention Configurations

| Rank | Experiment | Primary Pearson | Event RMSE | Centered pooled Pearson |
| ---: | --- | ---: | ---: | ---: |
| 1 | `WdeepPre + FullSweepB3Lam005 + full` | 0.1380 | 0.8114 | 0.2213 |
| 2 | `WphysioPre + FullSweepB3Lam005 + full` | 0.1334 | 0.8128 | 0.2074 |
| 3 | `WdeepPre + B5A1Lam0001 + full` | 0.1203 | 0.8285 | 0.1977 |
| 4 | `WphysioPre + FullSweepB3Lam005 + no_audio` | 0.1092 | 0.8264 | 0.1956 |
| 5 | `WdeepPre + B5A1Lam0001 + no_audio` | 0.1011 | 0.8290 | 0.2055 |
| 6 | `WphysioPre + B5A1Lam0001 + no_audio` | 0.0937 | 0.8235 | 0.1913 |
| 7 | `WphysioPre + B5A1Lam0001 + full` | 0.0927 | 0.8311 | 0.1686 |
| 8 | `WphysioPre + FullSweepB0 + full` | 0.0887 | 0.8275 | 0.1527 |

## Event-Grouped Attention Route Comparison

| Wear | Video route | Full primary | No-audio primary |
| --- | --- | ---: | ---: |
| `WphysioPre` | `FullSweepB0` | 0.0887 | 0.0847 |
| `WphysioPre` | `FullSweepB3Lam005` | 0.1334 | 0.1092 |
| `WphysioPre` | `A1A2TrainOnlyA2` | 0.0763 | 0.0644 |
| `WphysioPre` | `B5A1Lam0001` | 0.0927 | 0.0937 |
| `WdeepPre` | `FullSweepB0` | 0.0829 | 0.0419 |
| `WdeepPre` | `FullSweepB3Lam005` | 0.1380 | 0.0959 |
| `WdeepPre` | `A1A2TrainOnlyA2` | 0.0500 | 0.0317 |
| `WdeepPre` | `B5A1Lam0001` | 0.1203 | 0.1011 |

## Event-Grouped Best Config By Subject

Configuration: `WdeepPre + FullSweepB3Lam005 + full`,
`learnable_cross_attention`.

| Subject | Event count | Pearson | RMSE |
| --- | ---: | ---: | ---: |
| `sub-02` | 348 | -0.2115 | 0.8955 |
| `sub-03` | 396 | 0.3244 | 1.1222 |
| `sub-04` | 288 | 0.3997 | 0.5401 |
| `sub-05` | 708 | 0.3094 | 0.6743 |
| `sub-06` | 696 | 0.3479 | 0.9096 |
| `sub-07` | 516 | 0.2571 | 0.7121 |
| `sub-08` | 660 | 0.1080 | 1.0203 |
| `sub-09` | 792 | 0.2332 | 0.8437 |
| `sub-10` | 960 | 0.2271 | 0.5995 |
| `sub-11` | 600 | 0.2694 | 0.7034 |
| `sub-12` | 252 | -0.3894 | 0.7466 |
| `sub-13` | 144 | -0.1439 | 0.9792 |
| `sub-14` | 732 | 0.3020 | 0.8052 |
| `sub-15` | 276 | -0.1008 | 0.8162 |
| **subject-macro mean** | **7,368** | **0.1380** | **0.8114** |

## Session-Held-Out Execution

- Protocol: `session_held_out`
- Target: `fatigue`
- Subjects: 14 (`sub-02` through `sub-15`)
- Paired cohort: 7,368 windows
- Matrix: 20 experiments x 3 models
- Completed jobs: 840/840
- Primary metric: event-level subject-macro Pearson

## Session-Held-Out Best Attention Configurations

| Rank | Experiment | Primary Pearson | Event RMSE | Centered pooled Pearson |
| ---: | --- | ---: | ---: | ---: |
| 1 | `WphysioPre + FullSweepB3Lam005 + full` | -0.0080 | 0.8760 | 0.0514 |
| 2 | `WdeepPre + FullSweepB3Lam005 + full` | -0.0233 | 0.8992 | 0.0079 |
| 3 | `WdeepPre + B5A1Lam0001 + full` | -0.0404 | 0.8790 | 0.0287 |
| 4 | `WdeepPre + B5A1Lam0001 + no_audio` | -0.0436 | 0.9209 | 0.0225 |
| 5 | `WphysioPre + B5A1Lam0001 + full` | -0.0455 | 0.9034 | -0.0069 |
| 6 | `WdeepPre + FullSweepB3Lam005 + no_audio` | -0.0547 | 0.9262 | -0.0075 |
| 7 | `WphysioPre + FullSweepB0 + full` | -0.0642 | 0.8768 | -0.0062 |
| 8 | `WphysioPre + FullSweepB3Lam005 + no_audio` | -0.0657 | 0.9133 | 0.0308 |

## Session-Held-Out Attention Route Comparison

| Wear | Video route | Full primary | No-audio primary |
| --- | --- | ---: | ---: |
| `WphysioPre` | `FullSweepB0` | -0.0642 | -0.0920 |
| `WphysioPre` | `FullSweepB3Lam005` | -0.0080 | -0.0657 |
| `WphysioPre` | `A1A2TrainOnlyA2` | -0.0961 | -0.1256 |
| `WphysioPre` | `B5A1Lam0001` | -0.0455 | -0.1052 |
| `WdeepPre` | `FullSweepB0` | -0.0661 | -0.1436 |
| `WdeepPre` | `FullSweepB3Lam005` | -0.0233 | -0.0547 |
| `WdeepPre` | `A1A2TrainOnlyA2` | -0.0983 | -0.1380 |
| `WdeepPre` | `B5A1Lam0001` | -0.0404 | -0.0436 |

## Session-Held-Out Best Config By Subject

Configuration: `WphysioPre + FullSweepB3Lam005 + full`,
`learnable_cross_attention`.

| Subject | Event count | Pearson | RMSE |
| --- | ---: | ---: | ---: |
| `sub-02` | 348 | -0.0264 | 1.0478 |
| `sub-03` | 396 | -0.0186 | 1.2768 |
| `sub-04` | 288 | 0.3981 | 0.5407 |
| `sub-05` | 708 | -0.3318 | 0.7635 |
| `sub-06` | 696 | 0.0616 | 1.0111 |
| `sub-07` | 516 | 0.0192 | 0.7691 |
| `sub-08` | 660 | -0.0231 | 1.1166 |
| `sub-09` | 792 | 0.2976 | 0.8465 |
| `sub-10` | 960 | 0.1343 | 0.6476 |
| `sub-11` | 600 | 0.1195 | 0.7609 |
| `sub-12` | 252 | -0.5374 | 0.7524 |
| `sub-13` | 144 | -0.4869 | 1.1292 |
| `sub-14` | 732 | 0.1325 | 0.8551 |
| `sub-15` | 276 | 0.1490 | 0.7460 |
| **subject-macro mean** | **7,368** | **-0.0080** | **0.8760** |

## Interpretation

The strongest event-grouped attention result uses the fold-fitted B3 route,
with `WdeepPre` slightly ahead of `WphysioPre`. The B3 route is also stronger
than B0 for both wear branches. Audio is not required for the strongest
configuration: removing audio reduces the primary Pearson from 0.1380 to
0.0959 for the `WdeepPre` B3 configuration.

The session-held-out protocol is substantially harder. Its best attention
configuration is also B3 full-modality, but the subject-macro Pearson drops to
`-0.0080`. This suggests the current fusion model can exploit within-session
event-grouped structure but does not yet give stable cross-session
within-subject generalization.

The Ridge baseline has very large RMSE in several full-modality settings due
to train-only feature standardization followed by held-out feature shifts. It
is retained as a diagnostic baseline and is excluded from route selection.

Raw sources:

- Event-grouped values were extracted from the completed
  `outputs/reports/fusion_matrix_within_subject_120s10s/within_subject_fusion_summary.json`
  before the session-held-out run overwrote the generic summary filename.
- Session-held-out values are preserved at
  `outputs/reports/fusion_matrix_within_subject_120s10s/within_subject_fusion_session_held_out_summary.json`.
