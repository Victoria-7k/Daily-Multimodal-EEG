#!/usr/bin/env bash
# Isolated raw-normalization ablation; all fitting uses train rows, old results stay intact.
set -Eeuo pipefail
ROOT=/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned
STAGING="${MAE_NORM_STAGING:-/home/wangzw/mae_norm_20261008}"
PREVIOUS=/home/wangzw/mae_repair_20261008/outputs
BASE="$STAGING/outputs"
STAGE="$BASE/stage_a_train_channel"
OUT="$BASE/downstream_train_channel"
PREFIX="$BASE/prefixes_train_channel"
PY="$ROOT/runtime/envs/eegpt-gpu-min/bin/python"
MODE="${MAE_NORM_MODE:-train_channel}"
SMOKE_EPOCHS=15
TRAINING_VERSION=position_preserved_fixed_validation_v2
PREPROCESSING_VERSION=train_channel_zscore_v1
if [[ "$MODE" == train_channel_robust ]]; then
  SMOKE_EPOCHS=35
  TRAINING_VERSION=position_preserved_balanced_channel_eeg_v4
  PREPROCESSING_VERSION=train_channel_robust_zscore_v1
elif [[ "$MODE" != train_channel ]]; then
  echo "unsupported normalization mode: $MODE" >&2; exit 1
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$STAGING/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$STAGING/tmp"
mkdir -p "$STAGE/logs" "$OUT/logs" "$PREFIX" "$STAGING/tmp"
exec 9>"$BASE/.normalization_queue.lock"
if ! flock -n 9; then echo normalization_queue_already_running; exit 0; fi
trap 'touch "$BASE/NORMALIZATION_QUEUE_FAILED"; printf "failed: %s\n" "$BASH_COMMAND" >&2' ERR
[[ -f "$BASE/CPU_REGRESSION_COMPLETE" && -f "$PREVIOUS/downstream_v2/REPAIR_DOWNSTREAM_COMPLETE" ]]
if [[ -f "$BASE/NORMALIZATION_QUEUE_FAILED" ]]; then
  mv -- "$BASE/NORMALIZATION_QUEUE_FAILED" "$BASE/logs/NORMALIZATION_QUEUE_FAILED.$(date -u +%Y%m%dT%H%M%SZ).$$"
fi
cd "$STAGING"

check_stage() {
  "$PY" - "$STAGING" "$1" "$MODE" "$SMOKE_EPOCHS" "$TRAINING_VERSION" "$PREPROCESSING_VERSION" <<'PY'
import hashlib, json, math, sys
from pathlib import Path
import numpy as np
from daily_multimodal.training.eeg_encoder_matrix import load_split_protocols
from daily_multimodal.training.modality_mae import representative_indices
base, phase = Path(sys.argv[1]), sys.argv[2]
mode, smoke_epochs, training_version, preprocessing_version = sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6]
canonical = Path('/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned')
ids = np.asarray([json.loads(line)['sample_id'] for line in (canonical/'index/eeg_aligned_window_index.jsonl').read_text().splitlines()])
for protocol in ('cross_day', 'within_subject_day'):
    split = load_split_protocols(Path('/vePFS-0x0d/DailyEEG/splits_new'), (protocol,), row_count=len(ids))[protocol]
    for modality in ('eeg', 'wear'):
        directory = base/'outputs/stage_a_train_channel'/phase/protocol/f'{modality}_seed_240800'
        config = json.loads((directory/'config.json').read_text())
        assert config['test_labels_read'] is False and config['training_version']==training_version
        assert config['preprocessing_version']==preprocessing_version
        norm=config['raw_normalization']; assert norm['modality']==modality and norm['mode']==mode
        valid=np.ones(len(ids),bool)
        if modality=='wear':
            for name in ('ppg_10s.npz','gsr_10s.npz','acc_10s.npz'):
                with np.load(canonical/'outputs/wear_fm/staged_inputs'/name,allow_pickle=True) as data:
                    assert np.array_equal(data['sample_id'].astype(str),ids)
                    valid &= data['valid_mask'].astype(bool)
        train=representative_indices(split.train[valid[split.train]],1024 if phase=='smoke' else 0)
        assert not np.intersect1d(train,split.val).size and not np.intersect1d(train,split.test).size
        assert norm['fit_row_count']==len(train)==config['effective_train_count']
        assert norm['fit_indices_sha256']==hashlib.sha256(np.sort(train).astype('<i8').tobytes()).hexdigest()
        for branch in norm['branches'].values():
            assert np.isfinite(branch['mean']).all() and np.isfinite(branch['scale']).all() and min(branch['scale'])>0
        audit=config['reconstruction_audit']
        assert audit['history'] and not audit['selected_embedding_health']['collapsed']
        assert all(math.isfinite(h['train_masked_nmse']) and math.isfinite(h['val_masked_nmse']) and not h['embedding_health']['collapsed'] for h in audit['history'])
        if mode=='train_channel_robust':
            assert config['runtime']['reconstruction_loss']=='window_energy_balanced_mse_v1'
            assert config['runtime']['eeg_masking']==('channel' if modality=='eeg' else 'temporal')
            assert all(h['embedding_health']['robust_health'] and h['embedding_health']['median_relative_variation']>=0.001 for h in audit['history'])
            selected=audit['history'][audit['best_epoch']-1]
            assert selected['validation']['relative_to_zero'] < 1.0, 'reconstruction must beat the matched zero predictor'
        if phase=='smoke':
            assert len(audit['history'])==smoke_epochs and config['token_export']=='skipped_for_smoke'
        else:
            with np.load(directory/'window_embeddings.npz',allow_pickle=False) as token:
                assert np.array_equal(token['sample_id'].astype(str),ids) and np.array_equal(token['valid_mask'].astype(bool),valid)
                assert token['embedding'].shape==(28819,256) and np.isfinite(token['embedding']).all()
                assert (token['embedding'][~valid]==0).all()
        print(f'NORMALIZATION_GATE_PASS phase={phase} protocol={protocol} modality={modality} fit_rows={len(train)}',flush=True)
PY
}

for phase in smoke formal; do
  for modality in eeg wear; do
    for protocol in cross_day within_subject_day; do
      directory="$STAGE/$phase/$protocol/${modality}_seed_240800"
      if [[ -f "$directory/config.json" ]]; then echo "retaining completed Stage-A cell: $directory"; continue; fi
      epochs=100; patience=15; count=0; skip=()
      if [[ "$phase" == smoke ]]; then epochs="$SMOKE_EPOCHS"; patience="$SMOKE_EPOCHS"; count=1024; skip=(--skip-full-export); fi
      printf '%s starting normalization phase=%s modality=%s protocol=%s\n' "$(date -Iseconds)" "$phase" "$modality" "$protocol"
      "$PY" scripts/multilabel/112_run_modality_mae.py \
        --aligned-root "$ROOT" --modality "$modality" --protocol "$protocol" \
        --out-root "$STAGE/$phase" --raw-normalization "$MODE" --seed 240800 --device cuda \
        --epochs "$epochs" --patience "$patience" --smoke-per-split "$count" "${skip[@]}" \
        --health-probe-count 256 --min-relative-variation 0.001 \
        2>&1 | tee "$STAGE/logs/${phase}_${modality}_$protocol.log"
    done
  done
  check_stage "$phase"
  if [[ "$phase" == smoke ]]; then
    [[ -f "$BASE/NORMALIZATION_SMOKE_COMPLETE" ]] || touch "$BASE/NORMALIZATION_SMOKE_COMPLETE"
  else
    [[ -f "$BASE/NORMALIZATION_STAGE_A_COMPLETE" ]] || touch "$BASE/NORMALIZATION_STAGE_A_COMPLETE"
  fi
done

check_results() {
  "$PY" - "$1" "$2" "$3" "$MODE" <<'PY'
import json, math, sys
value=json.load(open(sys.argv[1])); conditions=sys.argv[2].split(','); seeds=list(map(int,sys.argv[3].split(',')))
rows=[r for r in value['results'] if r['condition']!='B0']
assert len(rows)==2*len(conditions)*len(seeds)
assert {(r['protocol'],r['condition'],r['seed']) for r in rows}=={(p,c,s) for p in ('cross_day','within_subject_day') for c in conditions for s in seeds}
for row in rows:
    metric=row['metrics']; assert metric['status']=='ok'
    assert all(math.isfinite(v['raw_r']) for leaf in ('val','test') for v in metric[leaf]['per_label'].values())
    if row['condition'] in ('M5-FT','R0'):
        assert metric['training_version']=='last2_ft_encoder_eval_math_v3'
        assert all(a['first_backward_grad_abs_sum']>0 and a['first_step_max_abs_parameter_delta']>0 for a in metric['gradient_audit'].values())
        for modality in ('eeg','wear'):
            manifest=metric['prefix_manifests'][modality]
            assert manifest['raw_normalization']['mode']==sys.argv[4]
print(f'DOWNSTREAM_GATE_PASS conditions={conditions} rows={len(rows)}',flush=True)
PY
}

common=(--root "$ROOT" --reference-root "$ROOT/outputs/multiemotion_20260913/structure_matrix_A1" \
        --mae-root "$STAGE/formal" --video-mae-root "$PREVIOUS/video_stage_a_v2" --device cuda)
for group in single pairs all; do
  case "$group" in single) conditions=E1,W1;; pairs) conditions=M2,M3,M4;; all) conditions=M5-F;; esac
  if [[ -f "$OUT/$group/smoke/results.json" ]]; then
    echo "retaining completed frozen smoke: $group"
  else
    "$PY" scripts/multilabel/118_run_mae_mt11_event_ablation.py "${common[@]}" \
      --conditions "$conditions" --out-root "$OUT/$group/smoke" --protocols cross_day,within_subject_day \
      --seeds 240800 --epochs 3 --patience 3 2>&1 | tee "$OUT/logs/${group}_smoke.log"
  fi
  check_results "$OUT/$group/smoke/results.json" "$conditions" 240800
  if [[ -f "$OUT/$group/formal/results.json" ]]; then
    echo "retaining completed frozen formal: $group"
  else
    "$PY" scripts/multilabel/118_run_mae_mt11_event_ablation.py "${common[@]}" \
      --conditions "$conditions" --out-root "$OUT/$group/formal" --protocols cross_day,within_subject_day \
      --seeds 240800,240801,240802 --epochs 80 --patience 15 2>&1 | tee "$OUT/logs/${group}_formal.log"
  fi
  check_results "$OUT/$group/formal/results.json" "$conditions" 240800,240801,240802
done
[[ -f "$BASE/NORMALIZATION_FROZEN_COMPLETE" ]] || touch "$BASE/NORMALIZATION_FROZEN_COMPLETE"

# Video input/encoder are unchanged; reference existing read-only caches on the same host.
"$PY" - "$PREFIX" "$PREVIOUS" <<'PY'
import json,sys
from pathlib import Path
import numpy as np
root,old=map(Path,sys.argv[1:])
random=Path('/home/wangzw/outputs/mae_remaining_20261007/prefixes/random/shared/video')
for protocol in ('cross_day','within_subject_day'):
    pretrained=old/'prefixes_v2/pretrained'/protocol/'video'
    for init,source in (('pretrained',pretrained),('random',random)):
        assert (source/'PREFIX_COMPLETE').is_file()
        manifest=json.loads((source/'manifest.json').read_text())
        assert manifest['initialization']==init and manifest['initialization_seed']==240802
        assert manifest['valid_count']==18012 and manifest['prefix_shape']==[18012,196,256]
        with np.load(source/'window_embeddings.npz',allow_pickle=False) as cached, np.load(old/'video_stage_a_v2'/protocol/'video_seed_240800/window_embeddings.npz',allow_pickle=False) as token:
            assert np.array_equal(cached['sample_id'],token['sample_id'])
            assert np.array_equal(cached['valid_mask'],token['valid_mask'])
        destination=root/init/protocol/'video'; destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.is_symlink(): assert destination.resolve()==source.resolve()
        else:
            assert not destination.exists()
            destination.symlink_to(source,target_is_directory=True)
        print(f'VIDEO_PREFIX_REUSED init={init} protocol={protocol}',flush=True)
PY
for initialization in pretrained random; do
  for protocol in cross_day within_subject_day; do
    for modality in eeg wear; do
      source="$STAGE/formal/$protocol/${modality}_seed_240800"
      extra=()
      if [[ "$initialization" == pretrained ]]; then extra=(--checkpoint "$source/checkpoint.pt" --match-stage-a-batch-size)
      else extra=(--normalization-config "$source/config.json"); fi
      "$PY" scripts/multilabel/123_export_mae_frozen_prefix.py \
        --modality "$modality" --initialization "$initialization" "${extra[@]}" \
        --token-file "$source/window_embeddings.npz" --out-dir "$PREFIX/$initialization/$protocol/$modality" \
        --batch-size 128 --device cuda --eeg-source /vePFS-0x0d/DailyEEG/processed_cadt_addtime_new/X.npy \
        --wear-root "$ROOT/outputs/wear_fm/staged_inputs" \
        2>&1 | tee "$OUT/logs/prefix_${initialization}_${modality}_$protocol.log"
    done
  done
done
touch "$BASE/NORMALIZATION_PREFIX_COMPLETE"
partial=(--root "$ROOT" --reference-root "$ROOT/outputs/multiemotion_20260913/structure_matrix_A1" \
         --prefix-root "$PREFIX" --random-prefix-per-protocol --conditions M5-FT,R0 \
         --device cuda --batch-size 64 --window-chunk-size 128)
"$PY" scripts/multilabel/124_run_mae_partial_ft.py "${partial[@]}" \
  --out-root "$OUT/partial/smoke" --protocols cross_day,within_subject_day --seeds 240800 \
  --epochs 10 --patience 10 2>&1 | tee "$OUT/logs/partial_smoke.log"
check_results "$OUT/partial/smoke/results.json" M5-FT,R0 240800
touch "$BASE/NORMALIZATION_PARTIAL_SMOKE_COMPLETE"
"$PY" scripts/multilabel/124_run_mae_partial_ft.py "${partial[@]}" \
  --out-root "$OUT/partial/formal" --protocols cross_day,within_subject_day --seeds 240800,240801,240802 \
  --epochs 80 --patience 15 2>&1 | tee "$OUT/logs/partial_formal.log"
check_results "$OUT/partial/formal/results.json" M5-FT,R0 240800,240801,240802

"$PY" - "$OUT" "$PREVIOUS/downstream_v2" "$PREPROCESSING_VERSION" <<'PY'
import importlib.util,json,math,sys
from pathlib import Path
import numpy as np
out,old=map(Path,sys.argv[1:3])
rows={}
for group in ('single','pairs','all','partial'):
    for row in json.loads((out/group/'formal/results.json').read_text())['results']:
        key=(row['protocol'],row['condition'],row['seed'])
        if key in rows: assert rows[key]==row
        rows[key]=row
for row in json.loads((old/'single/formal/results.json').read_text())['results']:
    if row['condition']=='V1': rows[(row['protocol'],row['condition'],row['seed'])]=row
order=['B0','E1','W1','V1','M2','M3','M4','M5-F','M5-FT','R0']
protocols=['cross_day','within_subject_day']; seeds=[240800,240801,240802]
assert set(rows)=={(p,c,s) for p in protocols for c in order for s in seeds}
for row in rows.values():
    assert row['metrics']['status']=='ok'
    assert all(math.isfinite(v['raw_r']) for leaf in ('val','test') for v in row['metrics'][leaf]['per_label'].values())
spec=importlib.util.spec_from_file_location('norm_table',Path('scripts/multilabel/118_run_mae_mt11_event_ablation.py'))
runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
summary={}; macro={}
for p in protocols:
    summary[p]={}; macro[p]={}
    for c in order:
        metrics=[rows[(p,c,s)]['metrics'] for s in seeds]
        summary[p][c]={}
        for label in runner.LABEL_NAMES:
            values=np.array([m['test']['per_label'][label]['raw_r'] for m in metrics])
            summary[p][c][label]=f'{values.mean():.4f} ± {values.std(ddof=1):.4f}'
        values=np.array([np.mean([m['test']['per_label'][label]['raw_r'] for label in runner.LABEL_NAMES]) for m in metrics])
        macro[p][c]={'mean':float(values.mean()),'sample_sd':float(values.std(ddof=1)),'seed_values':values.tolist()}
value={'reference_route':runner.REFERENCE_ROUTE,'protocols':protocols,'seeds':seeds,'table_conditions':order,
       'raw_normalization':f'EEG_Wear_{sys.argv[3]}','video_reused_from':str(old),
       'results':list(rows.values()),'summary':summary,'macro_summary':macro}
(out/'results.json').write_text(json.dumps(value,ensure_ascii=False,indent=2))
runner._table(value,out/'raw_r_tables.md')
print('NORMALIZATION_FORMAL_GATE_PASS new_rows=48 reused_B0_V1_rows=12 total=60',flush=True)
PY
touch "$BASE/NORMALIZATION_COMPLETE"
