#!/usr/bin/env python3
"""Execute frozen round1 gates, preserving valid cells and inherited stop states."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock
import time
import numpy as np

ALIGNED=Path('/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned')
REFERENCE=ALIGNED/'outputs/multiemotion_20260913/structure_matrix_A1'
V4=Path('/home/wangzw/mae_norm_channel_20261009')
PROTOCOLS=('cross_day','within_subject_day')
VIEWS=('V_FULL_MATCH','V_ROI')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, log, gpu):
    log.parent.mkdir(parents=True,exist_ok=True)
    print(json.dumps({'starting':command,'log':str(log),'gpu':gpu}),flush=True)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONPATH':'src'}
    with log.open('a',encoding='utf-8') as handle:
        subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,env=env,check=True)


def gate(directory, ids, expected_mask, smoke, modality):
    c=json.loads((directory/'config.json').read_text());audit=c['reconstruction_audit'];history=audit['history']
    if not history or audit['selected_embedding_health']['collapsed']:raise ValueError('unhealthy_selected_checkpoint')
    if any(not np.isfinite(h['val_masked_nmse']) or not np.isfinite(h['train_masked_nmse']) or h['embedding_health']['collapsed'] for h in history):raise ValueError('unhealthy_history')
    selected=history[audit['best_epoch']-1]
    if modality=='eeg':
        for retry in audit.get('attention_retry_records',[]):
            if retry['rows_skipped']!=0 or retry['optimizer_updates']!=1 or retry['rng_after_retry']!='discarded_default_forward_stream' or not np.isfinite(retry['math_loss']):raise ValueError('invalid_numerical_attention_recovery')
        if selected['validation']['relative_to_zero']>=1:raise ValueError('EEG_selected_checkpoint_does_not_beat_zero')
        if c['raw_normalization']['version']!='train_channel_robust_zscore_v1':raise ValueError('wrong_normalization')
        old=json.loads((V4/f'outputs/stage_a_train_channel/formal/{c["protocol"]}/eeg_seed_240800/config.json').read_text())
        if c['raw_normalization']!=old['raw_normalization']:raise ValueError('frozen_normalization_changed')
        if c['runtime']['eeg_masking']!='channel' or c['runtime']['reconstruction_loss']!='window_energy_balanced_mse_v1':raise ValueError('EEG_method_changed')
        if any(not h['embedding_health']['robust_health'] or h['embedding_health']['median_relative_variation']<1e-3 for h in history):raise ValueError('EEG_typical_window_health_failed')
    if smoke:
        if len(history)!=(35 if modality=='eeg' else 15) or c['token_export']!='skipped_for_smoke':raise ValueError('incomplete_smoke')
        if c['effective_train_count']>1024 or c['effective_val_count']>1024:raise ValueError('smoke_too_large')
        if modality=='eeg' and min(c['training_source_counts'].values())<1:raise ValueError('smoke_missing_input_source')
    else:
        with np.load(directory/'window_embeddings.npz',allow_pickle=False) as z:
            emb=z['embedding'];mask=z['valid_mask'].astype(bool)
            if not np.array_equal(z['sample_id'].astype(str),ids) or emb.shape!=(28819,256) or len(set(ids))!=28819:raise ValueError('export_identity_failed')
            if not np.isfinite(emb).all() or not np.array_equal(mask,expected_mask) or np.any(emb[~mask]!=0):raise ValueError('export_values_or_mask_failed')
    print(json.dumps({'gate':'pass','directory':str(directory),'smoke':smoke,'modality':modality}),flush=True)
    (directory/'CELL_VALIDATED').write_text(json.dumps({'config_sha256':sha(directory/'config.json'),'checkpoint_sha256':sha(directory/'checkpoint.pt')}))
    return c


def stage_a(args):
    root=args.root;script=root/'scripts/multilabel';gpus=args.gpus.split(',');gpu_locks={gpu:Lock() for gpu in gpus}
    if args.kind=='eeg':
        cells=[('E_POOL','cross_day','eeg')]
        with (ALIGNED/'index/eeg_aligned_window_index.jsonl').open() as f:ids=np.asarray([json.loads(line)['sample_id'] for line in f])
        expected=np.ones(28819,dtype=bool)
        report=json.loads((root/'inputs/eeg/cross_day/filter_report.json').read_text())
        if report['status']!='pass':raise ValueError('EEG_input_gate_not_passed')
    else:
        cells=[(view,p,'video') for view in VIEWS for p in PROTOCOLS]
        if not (root/'inputs/video/VIDEO_INPUT_COMPLETE').exists():raise ValueError('video_input_gate_not_passed')
        with np.load(root/'inputs/video/common_mask.npz',allow_pickle=False) as z:ids=z['sample_id'].astype(str);expected=z['valid_mask'].astype(bool)
    for phase in ('smoke','formal'):
        def execute(item):
            i,(view,p,modality)=item;out=root/f'stage_a/{view}/{phase}';directory=out/p/f'{modality}_seed_240800';smoke=phase=='smoke'
            if not (directory/'config.json').exists():
                common=['--protocol',p,'--out-root',str(out),'--seed','240800','--device','cuda','--health-probe-count','256','--min-relative-variation','0.001']
                if modality=='eeg':
                    command=[sys.executable,str(script/'112_run_modality_mae.py'),'--modality','eeg','--aligned-root',str(ALIGNED),
                             '--ssl-training-manifest',str(root/f'inputs/eeg/{p}/ssl_train_manifest.jsonl'),
                             '--extra-eeg-root','/vePFS-0x0d/DailyEEG/processed_cadt_raw_unlabeled','--raw-normalization','train_channel_robust',
                             '--normalization-config',str(V4/f'outputs/stage_a_train_channel/formal/{p}/eeg_seed_240800/config.json'),
                             '--epochs','35' if smoke else '100','--patience','35' if smoke else '15','--batch-size','128','--recovery-checkpoint','--retry-attention-math']+common
                else:
                    command=[sys.executable,str(script/'114_run_video_mae.py'),'--video-metadata',str(root/f'inputs/video/{view}_metadata.npz'),
                             '--splits-root',str(root/'inputs/splits'),'--clip-cache',str(root/f'inputs/video/{view}.uint8.mmap'),
                             '--reuse-complete-cache','--input-view',view,'--epochs','15' if smoke else '40','--patience','15' if smoke else '8',
                             '--batch-size','8']+common
                if smoke:command+=['--smoke-per-split','1024','--skip-full-export']
                with gpu_locks[gpus[i%len(gpus)]]:
                    run(command,root/f'logs/{view}_{p}_{phase}.log',gpus[i%len(gpus)])
            config=gate(directory,ids,expected,smoke,modality)
            if not smoke:
                old=json.loads((root/f'stage_a/{view}/smoke/{p}/{modality}_seed_240800/config.json').read_text())
                if old['reconstruction_audit']['initial_state_sha256']!=config['reconstruction_audit']['initial_state_sha256']:raise ValueError('formal_initialization_differs_from_smoke')
            return (view,p,config)
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:results=list(pool.map(execute,enumerate(cells)))
        if args.kind=='video':
            for p in PROTOCOLS:
                configs=[c for view,protocol,c in results if protocol==p]
                for key in ('initial_state_sha256','train_count','val_count'):
                    if configs[0]['reconstruction_audit'][key]!=configs[1]['reconstruction_audit'][key]:raise ValueError('paired_video_training_contract_mismatch:'+key)
                if configs[0]['smoke_per_split'] and configs[0]['smoke_indices']!=configs[1]['smoke_indices']:raise ValueError('paired_video_row_order_changed')
        (root/f'{args.kind.upper()}_STAGE_A_{phase.upper()}_COMPLETE').touch()


def bag_gate(path, reference, condition, common_path):
    install=__import__('daily_multimodal.daily_affect.npz_compat',fromlist=['install_numpy_core_pickle_aliases'])
    install.install_numpy_core_pickle_aliases()
    with np.load(path,allow_pickle=True) as z,np.load(reference,allow_pickle=True) as old:
        for key in old.files:
            if key in ('tokens','modality_mask','source_npz_json','route_id','supervision_boundary'):continue
            if not np.array_equal(z[key],old[key]):raise ValueError('bag_membership_changed:'+key)
        target_slot=0 if condition=='E_POOL' else 2
        for slot in range(4):
            if slot==target_slot:continue
            if not np.array_equal(z['tokens'][:,:,slot],old['tokens'][:,:,slot]) or not np.array_equal(z['modality_mask'][:,:,slot],old['modality_mask'][:,:,slot]):raise ValueError('non_target_modality_changed')
        if condition!='E_POOL':
            with np.load(common_path) as common:
                mapping={sid:i for i,sid in enumerate(common['sample_id'].astype(str))}
                idx=np.asarray([mapping[sid] for sid in z['sample_id_matrix'].astype(str).ravel()]).reshape(z['sample_id_matrix'].shape)
                if not np.array_equal(z['modality_mask'][:,:,2].astype(bool),common['valid_mask'][idx]):raise ValueError('downstream_common_mask_changed')


def stage_b(args):
    root=args.root;gpu=args.gpus.split(',')[0]
    if args.kind=='eeg':cells=[('E_POOL','cross_day')]
    else:cells=[(view,p) for view in (*VIEWS,'B0_VIDEO_COMMON') for p in PROTOCOLS]
    for phase in ('smoke','formal'):
        for view,p in cells:
            out=root/f'downstream/{view}/{phase}';epochs=3 if phase=='smoke' else 80
            command=[sys.executable,str(root/'scripts/multilabel/118_run_mae_mt11_event_ablation.py'),'--root',str(ALIGNED),'--reference-root',str(REFERENCE),
                     '--mae-root',str(root/f'stage_a/{view}/formal'),'--out-root',str(out),'--conditions',view,'--protocols',p,
                     '--seeds','240800' if phase=='smoke' else '240800,240801,240802','--epochs',str(epochs),'--patience','3' if phase=='smoke' else '15','--device','cuda']
            if args.kind=='video':command+=['--video-common-mask',str(root/'inputs/video/common_mask.npz')]
            run(command,root/f'logs/{view}_{p}_downstream_{phase}.log',gpu)
            value=json.loads((out/'results.json').read_text());rows=[r for r in value['results'] if r['condition']==view]
            seeds={240800} if phase=='smoke' else {240800,240801,240802}
            if {(r['protocol'],r['seed']) for r in rows}!={(p,s) for s in seeds}:raise ValueError('downstream_cell_coverage_failed')
            for row in rows:
                m=row['metrics']
                if m['status']!='ok' or any(not np.isfinite(v['raw_r']) for leaf in ('val','test') for v in m[leaf]['per_label'].values()):raise ValueError('downstream_nonfinite_metric')
            from importlib.util import spec_from_file_location,module_from_spec
            spec=spec_from_file_location('round1_ref',root/'scripts/multilabel/118_run_mae_mt11_event_ablation.py');module=module_from_spec(spec);spec.loader.exec_module(module)
            bag_gate(module._replacement_bag(out,p,view),module._reference_bag(REFERENCE,p),view,root/'inputs/video/common_mask.npz')
        (root/f'{args.kind.upper()}_STAGE_B_{phase.upper()}_COMPLETE').touch()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--kind',choices=('eeg','video'),required=True);p.add_argument('--phase',choices=('a','b','all'),default='all')
    p.add_argument('--gpus',default='0');a=p.parse_args()
    try:
        if a.phase in ('a','all'):stage_a(a)
        if a.phase in ('b','all'):stage_b(a)
    except Exception as exc:
        (a.root/f'{a.kind.upper()}_QUEUE_FAILED.json').write_text(json.dumps({'time':time.time(),'error':repr(exc)},indent=2));raise
    (a.root/f'{a.kind.upper()}_QUEUE_FAILED.json').unlink(missing_ok=True)


if __name__=='__main__':main()
