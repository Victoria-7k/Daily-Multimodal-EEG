#!/usr/bin/env python3
"""Report fixed paired input contrasts and per-seed subject-day bootstrap."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from daily_multimodal.daily_affect.ema_bags import LABEL_NAMES
from daily_multimodal.daily_affect.npz_compat import install_numpy_core_pickle_aliases
install_numpy_core_pickle_aliases()
ALIGNED=Path('/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned')
REFERENCE=ALIGNED/'outputs/multiemotion_20260913/structure_matrix_A1'
V4=Path('/home/wangzw/mae_norm_channel_20261009/outputs')
PROTOCOLS=('cross_day','within_subject_day')
VARIANTS=('B0_NATIVE','E_CANON_V4','E_POOL','V_FULL_MATCH','V_ROI','B0_VIDEO_COMMON')
METRICS=('raw_r','within_subject_centered_r','rmse','standardized_rmse','mae')
PAIRS=(('E_POOL','E_CANON_V4','eeg_input'),('V_ROI','V_FULL_MATCH','video_input'),
       ('E_POOL','B0_NATIVE','eeg_strong_reference'),('V_ROI','B0_VIDEO_COMMON','video_strong_reference'))


def audit_reuse(root):
    """Check scientific implementation and actual reused carrier/token provenance."""
    import ast
    v4=V4.parent
    hashes={}
    for rel in ('src/daily_multimodal/training/structure_emotion.py','src/daily_multimodal/training/multihead_regression.py',
                'src/daily_multimodal/daily_affect/regression.py','src/daily_multimodal/daily_affect/training.py'):
        current=(root/rel).read_bytes();old=(v4/rel).read_bytes()
        if current!=old:raise ValueError('reused_downstream_implementation_changed:'+rel)
        hashes[rel]=hashlib.sha256(current).hexdigest()
    rel='src/daily_multimodal/training/modality_mae.py'
    def definitions(path):
        t=ast.parse(path.read_text());return {n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    current=definitions(root/rel);old=definitions(v4/rel)
    for name in ('TemporalMaskedAutoencoder','eeg_to_patches','eeg_reconstruction_mask','reconstruction_loss','_evaluate_eeg'):
        if current[name]!=old[name]:raise ValueError('reused_EEG_method_changed:'+name)
    route='A1_Wphysio_no_audio__eeg_eegpt_partial_ft_multitask_11label_v1'
    for p in PROTOCOLS:
        reference=REFERENCE/f'bags/{p}/{route}/seed_240800/ema_bags.npz'
        control=V4/f'downstream_train_channel/single/formal/bags/{p}/E1/seed_240800/ema_bags.npz'
        token=V4/f'stage_a_train_channel/formal/{p}/eeg_seed_240800/window_embeddings.npz'
        with np.load(reference,allow_pickle=True) as b,np.load(control,allow_pickle=True) as c,np.load(token,allow_pickle=False) as t:
            for key in b.files:
                if key in ('tokens','modality_mask','route_id','source_npz_json','supervision_boundary'):continue
                if not np.array_equal(b[key],c[key]):raise ValueError('reused_bag_membership_changed:'+key)
            for slot in (1,2,3):
                if not np.array_equal(b['tokens'][:,:,slot],c['tokens'][:,:,slot]) or not np.array_equal(b['modality_mask'][:,:,slot],c['modality_mask'][:,:,slot]):raise ValueError('reused_control_non_EEG_modality_changed')
            lookup={sid:i for i,sid in enumerate(t['sample_id'].astype(str))}
            idx=np.asarray([lookup[sid] for sid in c['sample_id_matrix'].astype(str).ravel()]).reshape(c['sample_id_matrix'].shape)
            if not np.array_equal(c['tokens'][:,:,0],t['embedding'][idx]) or not np.array_equal(c['modality_mask'][:,:,0].astype(bool),t['valid_mask'][idx].astype(bool)):raise ValueError('reused_E_CANON_V4_token_provenance_changed')
        hashes[str(token)]=hashlib.sha256(token.read_bytes()).hexdigest()
    (root/'inputs/reuse_final_audit.json').write_text(json.dumps({'status':'pass','source_sha256':hashes},indent=2))
    return hashes


def path_for(root,variant,protocol,seed):
    if variant=='B0_NATIVE':return REFERENCE/f'runs/{protocol}/window_attention_regression_full_mean/seed_{seed}'
    if variant=='E_CANON_V4':return V4/f'downstream_train_channel/single/formal/runs/{protocol}/E1/seed_{seed}'
    return root/f'downstream/{variant}/formal/runs/{protocol}/{variant}/seed_{seed}'


def corr(y,p):
    yc=y-y.mean(axis=0);pc=p-p.mean(axis=0)
    den=np.sqrt((yc*yc).sum(axis=0)*(pc*pc).sum(axis=0))
    return np.divide((yc*pc).sum(axis=0),den,out=np.zeros(y.shape[1]),where=den>1e-12)


def scores(y,p,subjects,scale):
    yc=y.copy();pc=p.copy()
    for subject in np.unique(subjects):
        mask=subjects==subject;yc[mask]-=yc[mask].mean(axis=0);pc[mask]-=pc[mask].mean(axis=0)
    error=p-y;rmse=np.sqrt((error*error).mean(axis=0))
    return np.asarray([corr(y,p).mean(),corr(yc,pc).mean(),rmse.mean(),(rmse/scale).mean(),np.abs(error).mean()])


def bootstrap(a,b,leaf,scale,iterations,seed):
    for field in ('event_id','target','subject_id','day_id'):
        if not np.array_equal(a[f'{leaf}_{field}'],b[f'{leaf}_{field}']):raise ValueError('paired_prediction_membership_changed:'+field)
    y=a[f'{leaf}_target'].astype(np.float64);pa=a[f'{leaf}_prediction'].astype(np.float64);pb=b[f'{leaf}_prediction'].astype(np.float64)
    subjects=a[f'{leaf}_subject_id'].astype(str);days=a[f'{leaf}_day_id'].astype(str)
    keys=np.char.add(np.char.add(subjects,'/'),days);blocks=[np.flatnonzero(keys==key) for key in np.unique(keys)]
    rng=np.random.default_rng(seed);values=np.zeros((iterations,len(METRICS)))
    for i in range(iterations):
        idx=np.concatenate([blocks[k] for k in rng.integers(0,len(blocks),len(blocks))])
        values[i]=scores(y[idx],pa[idx],subjects[idx],scale)-scores(y[idx],pb[idx],subjects[idx],scale)
    ci=np.quantile(values,[.025,.975],axis=0)
    return {metric:{'low':float(ci[0,i]),'high':float(ci[1,i])} for i,metric in enumerate(METRICS)}


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--bootstrap-iters',type=int,default=2000);a=p.parse_args();out=a.root/'reports';out.mkdir(exist_ok=True)
    reuse=audit_reuse(a.root)
    rows=[];lookup={};per_label=[];provenance={}
    for protocol in PROTOCOLS:
        for variant in VARIANTS:
            for seed in (240800,240801,240802):
                directory=path_for(a.root,variant,protocol,seed);path=directory/'metrics.json'
                row={'protocol':protocol,'variant':variant,'seed':seed,'status':'missing','metric_path':str(path)}
                if variant=='E_POOL' and protocol=='within_subject_day':row.update(status='stopped',reason='inherited_canonical_train_holdout_signal_overlap_107')
                elif path.exists():
                    m=json.loads(path.read_text());row.update(status=m['status'],metrics=m)
                    if m['status']!='ok':raise ValueError('invalid_metric:'+str(path))
                    if m['condition_id']!='window_attention_regression_full_mean' or m['head_variant']!='H1_shared2_11xhead2' or m['normalization']!='per_modality' or m['adapter_mode']!='per_modality':raise ValueError('downstream_contract_changed')
                    provenance[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                    for leaf in ('val','test'):
                        for label in LABEL_NAMES:
                            values=m[leaf]['per_label'][label]
                            if any(not np.isfinite(values[key]) for key in METRICS):raise ValueError('nonfinite_or_missing_label_metric')
                            per_label.append({'protocol':protocol,'variant':variant,'seed':seed,'leaf':leaf,'label':label,**{k:values[k] for k in METRICS}})
                    if not (directory/'event_predictions.npz').exists():raise FileNotFoundError('missing_predictions:'+str(directory))
                rows.append(row);lookup[protocol,variant,seed]=row
    summaries=[]
    for protocol in PROTOCOLS:
        for variant in VARIANTS:
            selected=[lookup[protocol,variant,seed] for seed in (240800,240801,240802)]
            if not all(row['status']=='ok' for row in selected):continue
            for leaf in ('val','test'):
                for key in METRICS:
                    values=np.asarray([np.mean([row['metrics'][leaf]['per_label'][label][key] for label in LABEL_NAMES]) for row in selected])
                    summaries.append({'protocol':protocol,'variant':variant,'leaf':leaf,'metric':key,'mean':float(values.mean()),'seed_sd':float(values.std(ddof=1)),'seed_count':3})
    paired=[];contrasts=[];gates=[];label_pairs=[];label_contrasts=[]
    for protocol in PROTOCOLS:
        for candidate,control,role in PAIRS:
            complete=all(lookup[protocol,v,s]['status']=='ok' for v in (candidate,control) for s in (240800,240801,240802))
            if not complete:continue
            for leaf in ('val','test'):
                deltas=[]
                for seed in (240800,240801,240802):
                    aa=lookup[protocol,candidate,seed];bb=lookup[protocol,control,seed]
                    ma=aa['metrics'];mb=bb['metrics'];scale=np.asarray(ma['train_target_std'],dtype=float)
                    if not np.array_equal(scale,np.asarray(mb['train_target_std'])):raise ValueError('paired_target_scale_changed')
                    # Reuse only within the exact canonical universe and fixed y/split contract.
                    with np.load(Path(aa['metric_path']).with_name('event_predictions.npz'),allow_pickle=False) as za,np.load(Path(bb['metric_path']).with_name('event_predictions.npz'),allow_pickle=False) as zb:
                        va={k:za[k] for k in za.files};vb={k:zb[k] for k in zb.files}
                    delta=np.asarray([np.mean([ma[leaf]['per_label'][label][key]-mb[leaf]['per_label'][label][key] for label in LABEL_NAMES]) for key in METRICS])
                    ci=bootstrap(va,vb,leaf,scale,a.bootstrap_iters,seed+90210)
                    deltas.append(delta)
                    for label in LABEL_NAMES:
                        for key in METRICS:
                            label_pairs.append({'protocol':protocol,'candidate':candidate,'control':control,'role':role,'leaf':leaf,
                                                'seed':seed,'label':label,'metric':key,'delta':float(ma[leaf]['per_label'][label][key]-mb[leaf]['per_label'][label][key])})
                    for i,key in enumerate(METRICS):paired.append({'protocol':protocol,'candidate':candidate,'control':control,'role':role,'leaf':leaf,'seed':seed,'metric':key,'delta':float(delta[i]),'bootstrap_low':ci[key]['low'],'bootstrap_high':ci[key]['high'],'bootstrap_iters':a.bootstrap_iters})
                deltas=np.asarray(deltas)
                contrasts.append({'protocol':protocol,'candidate':candidate,'control':control,'role':role,'leaf':leaf,
                                  'metrics':{key:{'delta_mean':float(deltas[:,i].mean()),'seed_sd':float(deltas[:,i].std(ddof=1)),
                                                  'positive_seed_count':int((deltas[:,i]>0).sum())} for i,key in enumerate(METRICS)}})
                for label in LABEL_NAMES:
                    for key in METRICS:
                        values=np.asarray([lookup[protocol,candidate,seed]['metrics'][leaf]['per_label'][label][key]-lookup[protocol,control,seed]['metrics'][leaf]['per_label'][label][key] for seed in (240800,240801,240802)])
                        label_contrasts.append({'protocol':protocol,'candidate':candidate,'control':control,'role':role,'leaf':leaf,'label':label,'metric':key,
                                                'delta_mean':float(values.mean()),'seed_sd':float(values.std(ddof=1)),'positive_seed_count':int((values>0).sum()),
                                                'seed_240800_delta':float(values[0]),'seed_240801_delta':float(values[1]),'seed_240802_delta':float(values[2])})
                if leaf=='val' and role.endswith('_input'):
                    passed=bool(deltas[:,0].mean()>0 and (deltas[:,0]>0).sum()>=2 and deltas[:,1].mean()>=-1e-6 and deltas[:,3].mean()<=1e-6)
                    gates.append({'protocol':protocol,'candidate':candidate,'control':control,'pass':passed,'source':'completed_three_seed_validation_only'})
                print(json.dumps({'pair_reported':candidate+'-'+control,'protocol':protocol,'leaf':leaf}),flush=True)
    inputs={}
    for protocol in PROTOCOLS:
        path=a.root/f'inputs/eeg/{protocol}/filter_report.json'
        if path.exists():inputs['eeg_'+protocol]=json.loads(path.read_text())
    video=a.root/'inputs/video/cache_report.json'
    if video.exists():inputs['video']=json.loads(video.read_text())
    recovery=a.root/'inputs/execution_recovery/recovery_notes.json'
    if recovery.exists():inputs['execution_recovery']=json.loads(recovery.read_text())
    compute_ledger=a.root/'inputs/execution_recovery/compute_recovery_ledger.json'
    if compute_ledger.exists():inputs['compute_recovery_ledger']=json.loads(compute_ledger.read_text())
    technical=[]
    for path in sorted((a.root/'stage_a').glob('*/*/*/*/config.json')):
        c=json.loads(path.read_text());audit=c['reconstruction_audit'];selected=audit['history'][audit['best_epoch']-1]
        retry_visits=sum(x['row_count'] for x in audit.get('attention_retry_records',[]))
        discarded_visits=inputs.get('compute_recovery_ledger',{}).get('total_discarded_reconstruction_forward_window_visits',0) if c.get('input_variant')=='E_POOL' and path.parents[2].name=='formal' and c['protocol']=='cross_day' else 0
        technical.append({'variant':c.get('input_variant'),'protocol':c['protocol'],'phase':path.parents[2].name,
                          'best_epoch':audit['best_epoch'],'best_checkpoint_steps':audit['best_checkpoint_steps'],'optimizer_steps':audit['optimizer_steps'],
                          'seen_windows':audit['seen_windows'],'gpu_wall_seconds':audit['gpu_wall_seconds'],'selected_validation':selected.get('validation'),
                          'selected_embedding_health':audit['selected_embedding_health'],'config_path':str(path),
                          'numerical_gradient_recovery':c.get('numerical_gradient_recovery'),'attention_retry_records':audit.get('attention_retry_records',[])})
        technical[-1].update({'numerical_retry_forward_window_visits':retry_visits,'discarded_attempt_forward_window_visits':discarded_visits,
                              'total_reconstruction_training_forward_window_visits':audit['seen_windows']+retry_visits+discarded_visits,
                              'compute_count_scope':'training forwards; excludes validation, probes, smoke for formal cells, and diagnostic profiling'})
    finished=all(r['status'] in ('ok','stopped') for r in rows)
    # A stopped protocol cannot satisfy the original all-six-SSL completion marker.
    result={'status':'finished_with_protocol_stop' if finished else 'running','expected_cells':36,'eligible_cells':33,
            'ok_cells':sum(r['status']=='ok' for r in rows),'stopped_cells':sum(r['status']=='stopped' for r in rows),
            'rows':rows,'summary_metrics':summaries,'input_audit':inputs,'technical':technical,'contrasts':contrasts,'per_label_contrasts':label_contrasts,'validation_gates':gates,'bootstrap_unit':'paired_subject_day_within_seed',
            'seed_sd_ddof':1,'upstream_seed_count':1,'classification':'predeclared_exploratory_input_validation','source_sha256':provenance,'reuse_audit_sha256':reuse}
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    (out/'input_audit.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2))
    write_csv(out/'paired_metrics.csv',paired);write_csv(out/'per_label_metrics.csv',per_label);write_csv(out/'summary_metrics.csv',summaries)
    write_csv(out/'per_label_paired_metrics.csv',label_pairs);write_csv(out/'per_label_contrasts.csv',label_contrasts)
    lines=['# MAE round1 输入验证','',f"状态：{result['status']}；有效 {result['ok_cells']}/33 个可执行下游 cell；另 3 个同日 EEG cell 按输入停止条件保留。",'',
           '主要配对为 E_POOL − E_CANON_V4 与 V_ROI − V_FULL_MATCH；强参考分别为 B0_NATIVE 与 B0_VIDEO_COMMON。上游 seed 固定 240800，下游三 seed 的 SD 使用 ddof=1；bootstrap 在每个 seed 内按相同 subject-day 配对重采样 2,000 次。','',
           '## 输入事实','']
    for protocol in PROTOCOLS:
        e=inputs.get('eeg_'+protocol)
        if e:lines.append(f"- EEG {protocol}：{e['status']}；canonical {e['canonical_train_count']} 行，raw 保留 {e['retained_raw_count']} 行；新增独有覆盖 {e['added_unique_hours']:.3f} 小时。")
    if 'video' in inputs:lines.append(f"- Video：共同有效 {inputs['video']['common_valid']} 行；逐 clip 排除 {len(inputs['video']['failures'])} 行；ROI 原生新增 9 行保留在覆盖记录。")
    lines+=['','## 完整三seed读数','','| 协议 | 路线 | split | raw r ± SD | centered r ± SD | RMSE ± SD | sRMSE ± SD |','| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for protocol in PROTOCOLS:
        for variant in VARIANTS:
            for leaf in ('val','test'):
                selected={row['metric']:row for row in summaries if row['protocol']==protocol and row['variant']==variant and row['leaf']==leaf}
                if not selected:continue
                cells=[f"{selected[key]['mean']:.4f} ± {selected[key]['seed_sd']:.4f}" for key in METRICS[:4]]
                lines.append(f"| {protocol} | {variant} | {leaf} | "+' | '.join(cells)+' |')
    lines+=['','## 主要配对与强参考距离','','| 协议 | 配对 | split | 宏 raw r delta ± seed SD | 正向 seeds | centered r delta | sRMSE delta |','| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for c in contrasts:
        m=c['metrics'];raw=m['raw_r'];lines.append(f"| {c['protocol']} | {c['candidate']} − {c['control']} | {c['leaf']} | {raw['delta_mean']:+.4f} ± {raw['seed_sd']:.4f} | {raw['positive_seed_count']}/3 | {m['within_subject_centered_r']['delta_mean']:+.4f} | {m['standardized_rmse']['delta_mean']:+.4f} |")
    lines+=['','## 逐情绪主要配对变化','','| 协议 | 配对 | split | 情绪 | raw r delta ± seed SD | 正向 seeds | centered r delta | sRMSE delta |','| --- | --- | --- | --- | ---: | ---: | ---: | ---: |']
    for protocol in PROTOCOLS:
        for candidate,control,role in PAIRS:
            if not role.endswith('_input'):continue
            for leaf in ('val','test'):
                for label in LABEL_NAMES:
                    selected={v['metric']:v for v in label_contrasts if v['protocol']==protocol and v['candidate']==candidate and v['control']==control and v['leaf']==leaf and v['label']==label}
                    if not selected:continue
                    raw=selected['raw_r'];lines.append(f"| {protocol} | {candidate} − {control} | {leaf} | {label} | {raw['delta_mean']:+.4f} ± {raw['seed_sd']:.4f} | {raw['positive_seed_count']}/3 | {selected['within_subject_centered_r']['delta_mean']:+.4f} | {selected['standardized_rmse']['delta_mean']:+.4f} |")
    lines+=['','## 验证推进判定','']
    for g in gates:lines.append(f"- {g['protocol']} / {g['candidate']}：{'通过' if g['pass'] else '停在本轮'}；依据完整三 seed val 的预设 raw-r、centered-r 和 sRMSE 联合门槛。")
    lines+=['- within_subject_day / E_POOL：输入审计发现 107 个原 canonical train 窗口与 val/test 实际信号区间相交，按计划第 4.3 节停止严格对照。','',
            '技术读数见 results.json 的 technical；逐情绪原始读数见 per_label_metrics.csv，配对均值/SD与每seed差值见 per_label_contrasts.csv 和 per_label_paired_metrics.csv；逐seed宏指标配对及bootstrap区间见 paired_metrics.csv。三seed方向与区间分别报告；本轮保留单上游seed和既有split的探索性解释范围。','']
    if 'execution_recovery' in inputs:lines+=['执行恢复记录见 input_audit.json：EEG在epoch39的efficient-attention backward复现NaN，按完整断点恢复；失败batch在更新前以同窗口、同mask的math SDPA重算，检查通过后更新一次，零跳过行，并恢复后续CUDA随机流。每次后端loss差异及batch位置写入technical。gpu_wall_seconds累计已保存的完整epoch段与当前成功恢复段，丢弃尝试及中断的部分epoch需要结合失败日志解释。','']
    (out/'round1_input_report.md').write_text('\n'.join(lines),encoding='utf-8')
    if finished:(a.root/'ROUND1_EXECUTION_FINISHED_WITH_PROTOCOL_STOP').touch()
    print(json.dumps({k:result[k] for k in ('status','ok_cells','eligible_cells','stopped_cells')}),flush=True)


if __name__=='__main__':main()
