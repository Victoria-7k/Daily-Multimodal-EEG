#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned"
PY="$ROOT/runtime/envs/eegpt-gpu-min/bin/python"
STAGING="${MAE_REMAINING_STAGING:-/home/wangzw/mae_remaining_staging}"
BASE="${MAE_REMAINING_OUT:-/home/wangzw/outputs/mae_remaining_20261007}"
OUT="${MAE_PARTIAL_OUT:-$BASE/partial_v3}"
PREFIX_ROOT="${MAE_PREFIX_ROOT:-$BASE/prefixes}"
mkdir -p "$OUT/logs"
LOG="$OUT/logs/126_queue_$(date -u +%Y%m%dT%H%M%SZ).log"
trap 'touch "$OUT/PARTIAL_FT_QUEUE_FAILED"' ERR
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
echo "waiting_for_verified_prefixes" | tee -a "$LOG"
until [[ -f "$PREFIX_ROOT/PREFIX_H20_COMPLETE" && -f "$PREFIX_ROOT/VIDEO_PREFIX_TRANSFER_VERIFIED" && -f "$BASE/FROZEN_FORMAL_COMPLETE" ]]; do
  if [[ -f "$PREFIX_ROOT/PREFIX_PREPARATION_FAILED" || -f "$BASE/VIDEO_PREFIX_TRANSFER_FAILED" ]]; then
    echo "upstream_prefix_failed" | tee -a "$LOG"
    exit 1
  fi
  sleep 20
done
common=(--root "$ROOT" --reference-root "$ROOT/outputs/multiemotion_20260913/structure_matrix_A1" --prefix-root "$PREFIX_ROOT" --conditions M5-FT,R0 --device cuda --batch-size 64 --window-chunk-size 128)
run() { "$@" 2>&1 | tee -a "$LOG"; }
run "$PY" "$STAGING/124_run_mae_partial_ft.py" "${common[@]}" --out-root "$OUT/smoke" --protocols cross_day,within_subject_day --seeds 240800 --epochs 10 --patience 10
"$PY" - "$OUT/smoke/results.json" <<'PY'
import json, math, sys
value = json.load(open(sys.argv[1]))
rows = [r for r in value['results'] if r['condition'] != 'B0']
assert {(r['protocol'], r['condition'], r['seed']) for r in rows} == {
    (p, c, 240800) for p in ('cross_day', 'within_subject_day') for c in ('M5-FT', 'R0')}
assert len(rows) == 4
for row in rows:
    m = row['metrics']
    assert m['status'] == 'ok' and m['epochs_ran'] == 10
    assert m['training_version'] == 'last2_ft_encoder_eval_math_v3'
    assert m['fusion_attention_backend'] == 'math_fp32'
    assert len(m['history']) == 10 and all(math.isfinite(x['train_loss']) and math.isfinite(x['val_macro_standardized_rmse']) for x in m['history'])
    assert all(math.isfinite(v['raw_r']) for leaf in ('val', 'test') for v in m[leaf]['per_label'].values())
    assert all(a['first_backward_grad_abs_sum'] > 0 and a['first_step_max_abs_parameter_delta'] > 0 for a in m['gradient_audit'].values())
print('SMOKE_GATE_PASS: four cells, ten epochs each, gradients/updates/metrics finite')
PY
touch "$OUT/PARTIAL_FT_SMOKE_COMPLETE"
run "$PY" "$STAGING/124_run_mae_partial_ft.py" "${common[@]}" --out-root "$OUT/formal" --protocols cross_day,within_subject_day --seeds 240800,240801,240802 --epochs 80 --patience 15
"$PY" - "$OUT/formal/results.json" <<'PY'
import json, math, sys
value = json.load(open(sys.argv[1]))
rows = [r for r in value['results'] if r['condition'] != 'B0']
assert len(rows) == 12
assert {(r['protocol'], r['condition'], r['seed']) for r in rows} == {
    (p, c, s) for p in ('cross_day', 'within_subject_day') for c in ('M5-FT', 'R0') for s in (240800, 240801, 240802)}
for row in rows:
    m = row['metrics']
    assert m['status'] == 'ok' and m['training_version'] == 'last2_ft_encoder_eval_math_v3'
    assert all(math.isfinite(v['raw_r']) for leaf in ('val', 'test') for v in m[leaf]['per_label'].values())
print('FORMAL_GATE_PASS: twelve cells, all val/test raw-r finite')
PY
touch "$OUT/PARTIAL_FT_FORMAL_COMPLETE"
