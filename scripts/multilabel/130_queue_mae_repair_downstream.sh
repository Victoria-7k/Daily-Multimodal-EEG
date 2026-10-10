#!/usr/bin/env bash
# Fixed A1+MT11 reference; use only repaired, verified Stage-A artifacts.
set -Eeuo pipefail
ROOT=/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned
STAGING=/home/wangzw/mae_repair_20261008
BASE="$STAGING/outputs"
OUT="$BASE/downstream_v2"
PY="$ROOT/runtime/envs/eegpt-gpu-min/bin/python"
PREFIX="$BASE/prefixes_v2"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$STAGING/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$OUT/logs"
exec 9>"$OUT/.queue.lock"
if ! flock -n 9; then echo 'downstream_queue_already_running'; exit 0; fi
trap 'touch "$OUT/DOWNSTREAM_QUEUE_FAILED"' ERR
[[ -f "$BASE/stage_a_v2/STAGE_A_FORMAL_COMPLETE" && -f "$BASE/video_stage_a_v2/VIDEO_STAGE_A_TRANSFER_VERIFIED" ]]
[[ -f "$BASE/R0_UNMASKED_CONTRACT_VERIFIED" ]]

check_results() {
  "$PY" - "$1" "$2" "$3" <<'PY'
import json, math, sys
value=json.load(open(sys.argv[1])); conditions=sys.argv[2].split(','); seeds=[int(s) for s in sys.argv[3].split(',')]
rows=[r for r in value['results'] if r['condition']!='B0']
assert len(rows)==2*len(conditions)*len(seeds)
assert {(r['protocol'],r['condition'],r['seed']) for r in rows}=={(p,c,s) for p in ('cross_day','within_subject_day') for c in conditions for s in seeds}
for row in rows:
    metric=row['metrics']
    assert metric['status']=='ok'
    assert all(math.isfinite(v['raw_r']) for leaf in ('val','test') for v in metric[leaf]['per_label'].values())
    if row['condition']=='M5-FT':
        assert metric['training_version']=='last2_ft_encoder_eval_math_v3'
        assert all(a['first_backward_grad_abs_sum']>0 and a['first_step_max_abs_parameter_delta']>0 for a in metric['gradient_audit'].values())
print(f'GATE_PASS rows={len(rows)} conditions={conditions}')
PY
}

# The repair changes masked pretraining only; retain the completed matched R0 reference.
"$PY" - "$OUT/R0_reference.json" <<'PY'
import json, math, sys
from pathlib import Path
source=Path('/home/wangzw/outputs/mae_remaining_20261007/partial_v3/formal/results.json')
value=json.loads(source.read_text()); rows=[r for r in value['results'] if r['condition']=='R0']
assert len(rows)==6 and {(r['protocol'],r['seed']) for r in rows}=={(p,s) for p in ('cross_day','within_subject_day') for s in (240800,240801,240802)}
for row in rows:
    m=row['metrics']
    assert m['status']=='ok' and m['training_version']=='last2_ft_encoder_eval_math_v3'
    assert all(math.isfinite(v['raw_r']) for leaf in ('val','test') for v in m[leaf]['per_label'].values())
Path(sys.argv[1]).write_text(json.dumps({'source':str(source),'reuse_reason':'unmasked_encoder_architecture_preprocessing_and_downstream_contract_unchanged','results':rows},indent=2))
PY

common=(--root "$ROOT" --reference-root "$ROOT/outputs/multiemotion_20260913/structure_matrix_A1" \
        --mae-root "$BASE/stage_a_v2/formal" --video-mae-root "$BASE/video_stage_a_v2" --device cuda)
for group in single pairs all; do
  case "$group" in single) conditions=E1,W1,V1;; pairs) conditions=M2,M3,M4;; all) conditions=M5-F;; esac
  "$PY" "$STAGING/scripts/multilabel/118_run_mae_mt11_event_ablation.py" "${common[@]}" \
    --conditions "$conditions" --out-root "$OUT/$group/smoke" --protocols cross_day,within_subject_day \
    --seeds 240800 --epochs 3 --patience 3 2>&1 | tee "$OUT/logs/${group}_smoke.log"
  check_results "$OUT/$group/smoke/results.json" "$conditions" 240800
  "$PY" "$STAGING/scripts/multilabel/118_run_mae_mt11_event_ablation.py" "${common[@]}" \
    --conditions "$conditions" --out-root "$OUT/$group/formal" --protocols cross_day,within_subject_day \
    --seeds 240800,240801,240802 --epochs 80 --patience 15 2>&1 | tee "$OUT/logs/${group}_formal.log"
  check_results "$OUT/$group/formal/results.json" "$conditions" 240800,240801,240802
done
touch "$OUT/FROZEN_FORMAL_COMPLETE"

bash "$STAGING/scripts/multilabel/131_prepare_mae_repair_prefixes.sh" h20
until [[ -f "$PREFIX/VIDEO_PREFIX_TRANSFER_VERIFIED" ]]; do
  [[ ! -f "$BASE/VIDEO_PREFIX_TRANSFER_FAILED" ]]
  sleep 20
done
partial=(--root "$ROOT" --reference-root "$ROOT/outputs/multiemotion_20260913/structure_matrix_A1" \
         --prefix-root "$PREFIX" --conditions M5-FT --device cuda --batch-size 64 --window-chunk-size 128)
"$PY" "$STAGING/scripts/multilabel/124_run_mae_partial_ft.py" "${partial[@]}" \
  --out-root "$OUT/partial/smoke" --protocols cross_day,within_subject_day --seeds 240800 \
  --epochs 10 --patience 10 2>&1 | tee "$OUT/logs/partial_smoke.log"
check_results "$OUT/partial/smoke/results.json" M5-FT 240800
touch "$OUT/PARTIAL_FT_SMOKE_COMPLETE"
"$PY" "$STAGING/scripts/multilabel/124_run_mae_partial_ft.py" "${partial[@]}" \
  --out-root "$OUT/partial/formal" --protocols cross_day,within_subject_day --seeds 240800,240801,240802 \
  --epochs 80 --patience 15 2>&1 | tee "$OUT/logs/partial_formal.log"
check_results "$OUT/partial/formal/results.json" M5-FT 240800,240801,240802
touch "$OUT/PARTIAL_FT_FORMAL_COMPLETE"
touch "$OUT/REPAIR_DOWNSTREAM_COMPLETE"
