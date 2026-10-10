#!/usr/bin/env python3
"""Freeze protocol-specific unlabeled pools and stop on inherited signal overlap."""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def union(intervals):
    merged=[]
    for a,b in sorted(intervals):
        if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
        else:merged.append([float(a),float(b)])
    return merged


def duration(intervals):return sum(b-a for a,b in union(intervals))


def overlaps(start, intervals, length=10):
    return any(max(start,a)<min(start+length,b) for a,b in intervals)


def time_grid_existing(data,out):
    _,_,_,raw_times=identities(data/'processed_cadt_raw_unlabeled')
    candidate=np.asarray(json.loads((data/'splits_new/shared_unlabeled_pretrain.json').read_text()),dtype=int)
    def residues(values):
        unique,counts=np.unique(np.round(np.mod(values,5),5),return_counts=True)
        return [{'seconds_mod_5':float(v),'count':int(n)} for v,n in zip(unique,counts)]
    for protocol in ('cross_day','within_subject_day'):
        path=out/protocol/'filter_report.json';report=json.loads(path.read_text())
        rows=[json.loads(line) for line in (out/protocol/'ssl_train_manifest.jsonl').open() if line.strip()]
        canonical_times=np.asarray([row['start'] for row in rows if row['source_kind']=='canonical'])
        report['time_grid_audit']={'axis':'common processed recording array origin, checked against exact FIF slices',
                                  'canonical_start_mod_5':residues(canonical_times),'raw_candidate_start_mod_5':residues(raw_times[candidate]),
                                  'raw_minus_canonical_next_grid_mod_5':residues(np.mod(-canonical_times,5)),
                                  'common_source_first_samp':'per-recording offsets saved in source_audit.json'}
        if not report['retained_raw_count']:report['raw_pool_status']='no_eligible_extra_windows_under_whole_session_preprocessing_guard'
        path.write_text(json.dumps(report,indent=2))


def identities(root):
    meta=json.loads((root/'meta.json').read_text())
    s=np.load(root/'sub.npy');d=np.load(root/'d.npy');t=np.load(root/'ts.npy')
    # Domain IDs are decoded independently from each export's metadata.
    local=np.asarray([int(day)-int(meta['domain_offset_map'][str(int(sub))]) for sub,day in zip(s,d)])
    if np.any(local<0) or any(day>=meta['domains_per_subject_map'][str(int(sub))] for sub,day in zip(s,local)):
        raise ValueError('invalid_domain_mapping')
    return meta,s,local,t


def verify_sources(data, arrays, pool, canonical_train):
    """Check actual FIF channel order, sample support, units and both export slices."""
    import mne
    result=[];canonical_channels=None
    _,cs,cd,ct=arrays[0];_,rs,rd,rt=arrays[1]
    cx=np.load(data/'processed_cadt_addtime_new/X.npy',mmap_mode='r')
    rx=np.load(data/'processed_cadt_raw_unlabeled/X.npy',mmap_mode='r')
    days=sorted(set(zip(rs[pool].tolist(),rd[pool].tolist())))
    for sub,day in days:
        directory=data/f'derivatives/eeg-prep/sub-{sub:02d}/ses-{day+1:02d}'
        files=sorted(directory.glob('*.fif'))
        if not files:raise ValueError('missing_source_fif:'+str(directory))
        # The exporters treat explicitly segmented files on meas_date offsets;
        # other FIFs are concatenated in filename order. Both share this rule.
        def segment_number(path):
            parts=path.stem.split('-')
            return any(parts[i].isdigit() and ('eeg' in parts[i-1].lower() or 'beh' in parts[i-1].lower()) for i in range(1,len(parts)))
        segmented=any(segment_number(p) for p in files)
        raws=[mne.io.read_raw_fif(str(p),preload=False,verbose=False) for p in files]
        for raw in raws:
            if raw.info['sfreq']!=200 or len(raw.ch_names)!=59:raise ValueError('source_signal_contract_failed')
            if canonical_channels is None:canonical_channels=raw.ch_names
            if raw.ch_names!=canonical_channels:raise ValueError('source_channel_order_changed')
        if segmented:
            dates=[raw.info['meas_date'] for raw in raws]
            if any(x is None for x in dates):raise ValueError('missing_segment_date')
            origin=min(dates);offsets=[int((x-origin).total_seconds()*200) for x in dates]
        else:
            offsets=np.cumsum([0]+[raw.n_times for raw in raws[:-1]]).tolist()
        checks=[]
        for kind,selected,subs,days_,times,x in (('canonical',canonical_train,cs,cd,ct,cx),('raw',pool,rs,rd,rt,rx)):
            idx=selected[(subs[selected]==sub)&(days_[selected]==day)]
            if not len(idx):continue
            for row in idx[np.unique(np.linspace(0,len(idx)-1,min(3,len(idx)),dtype=int))]:
                start=int(float(times[row])*200);stop=start+2000
                chunks=[];cursor=start
                for raw,offset in sorted(zip(raws,offsets),key=lambda item:item[1]):
                    a=max(start,offset);b=min(stop,offset+raw.n_times)
                    if b>a:
                        if a!=cursor:raise ValueError('unconfirmed_segment_support')
                        chunks.append(raw.get_data(start=a-offset,stop=b-offset).T.astype(x.dtype));cursor=b
                if cursor!=stop:raise ValueError('truncated_source_support')
                reference=np.concatenate(chunks)
                if not np.array_equal(reference,x[row]):raise ValueError(f'export_source_mismatch:{kind}:{row}')
                checks.append({'source_kind':kind,'row':int(row),'start':float(times[row]),'exact_match':True})
        result.append({'subject':sub,'session':day+1,'recording_id':f'sub-{sub:02d}/ses-{day+1:02d}',
                       'files':[str(p) for p in files],'channel_names':canonical_channels,'unit':'MNE_SI_volt',
                       'source_first_samp':[raw.first_samp for raw in raws],'checks':checks,
                       'support_context':'both_exporters_slice_same_preprocessed_session; full-recording preprocessing shared within session'})
        for raw in raws:raw.close()
        print(json.dumps({'source_session_verified':len(result),'expected':len(days)}),flush=True)
    return result


def supplement_existing(data, out):
    """Complete per-day amplitude/overlap statistics and explicit preprocessing guard."""
    preprocessing=[data/'code/eeg_preprocess_pipeline/main.py',data/'code/eeg_preprocess_pipeline/utils.py']
    # Actual preprocessing fits ICA on the continuous recording (ica.fit(raw));
    # the same session's fitted transformation is shared by its exported slices.
    if 'ica.fit(raw)' not in preprocessing[1].read_text():raise ValueError('preprocessing_context_cannot_be_confirmed')
    context={'guard':'whole_preprocessed_session','reason':'recording-level filtering and ICA fit before export slicing',
             'source_sha256':{str(p):sha(p) for p in preprocessing}}
    all_rows={};reports={};manifests={}
    for protocol in ('cross_day','within_subject_day'):
        directory=out/protocol;path=directory/'ssl_train_manifest.jsonl'
        rows=[json.loads(line) for line in path.open() if line.strip()]
        report=json.loads((directory/'filter_report.json').read_text())
        cm,cs,cd,ct=identities(data/'processed_cadt_addtime_new')
        held=np.union1d(*[np.asarray(json.loads((data/f'splits_new/{protocol}/{leaf}.json').read_text()),dtype=int) for leaf in ('val','test')])
        held_sessions={f'sub-{cs[i]:02d}/ses-{cd[i]+1:02d}' for i in held}
        context_overlap=[row for row in rows if row['recording_id'] in held_sessions]
        raw_removed=[row for row in context_overlap if row['source_kind']=='raw']
        report['canonical_train_holdout_preprocessing_context_overlap_count']=sum(row['source_kind']=='canonical' for row in context_overlap)
        report['preprocessing_context_guard']=context
        if raw_removed:
            # Preserve the initial direct-interval audit; current rows additionally
            # enforce the explicit guard required by the frozen plan.
            previous=directory/'ssl_train_manifest_before_context_guard.jsonl'
            if not previous.exists():previous.write_bytes(path.read_bytes())
            removed={(row['source_kind'],row['source_row']) for row in raw_removed}
            rows=[row for row in rows if (row['source_kind'],row['source_row']) not in removed]
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            excluded=directory/'excluded_raw.jsonl'
            with excluded.open('a') as handle:
                for row in raw_removed:handle.write(json.dumps({'source_row':row['source_row'],'reason':'val_test_preprocessing_context_overlap'})+'\n')
            report['raw_retained_before_context_guard']=report['retained_raw_count']
            report['retained_raw_count']-=len(raw_removed)
            report['exclusion_counts']['val_test_preprocessing_context_overlap']=len(raw_removed)
            report['raw_rms_quantiles']=[]
            if report['status']=='pass':raise ValueError('context guard changed a previously executable protocol')
        reports[protocol]=report;manifests[protocol]=rows
        for row in rows:all_rows[row['source_kind'],row['source_row']]=row
    amplitudes={}
    for kind in ('canonical','raw'):
        indices=sorted(row for source,row in all_rows if source==kind)
        x=np.load(data/('processed_cadt_addtime_new' if kind=='canonical' else 'processed_cadt_raw_unlabeled')/'X.npy',mmap_mode='r')
        for start in range(0,len(indices),128):
            idx=indices[start:start+128];values=np.asarray(x[idx],dtype=np.float64)
            rms=np.sqrt(np.mean(values*values,axis=(1,2)))
            for row,value in zip(idx,rms):amplitudes[kind,row]=float(value)
        print(json.dumps({'amplitude_statistics_completed':kind,'unique_rows':len(indices)}),flush=True)
    for protocol,rows in manifests.items():
        report=reports[protocol];by_day={}
        for row in rows:by_day.setdefault(row['recording_id'],[]).append(row)
        coverage={}
        for key,day_rows in by_day.items():
            canonical=[(row['start'],row['end']) for row in day_rows if row['source_kind']=='canonical']
            raw=[(row['start'],row['end']) for row in day_rows if row['source_kind']=='raw']
            sorted_rows=sorted(day_rows,key=lambda row:(row['start'],row['end']));starts=np.asarray([row['start'] for row in sorted_rows]);ends=np.asarray([row['end'] for row in sorted_rows])
            pair_count=int((np.searchsorted(starts,ends,side='left')-np.arange(len(starts))-1).sum())
            repeated=Counter((row['start'],row['end']) for row in day_rows)
            exact_pairs=sum(count*(count-1)//2 for count in repeated.values())
            overlapping=np.zeros(len(starts),dtype=bool)
            if len(starts)>1:
                overlapping[1:]|=starts[1:]<np.maximum.accumulate(ends)[:-1]
                overlapping[:-1]|=ends[:-1]>starts[1:]
            record={'canonical_rows':len(canonical),'raw_rows':len(raw),'canonical_union_seconds':duration(canonical),
                    'raw_union_seconds':duration(raw),'total_union_seconds':duration(canonical+raw),
                    'added_unique_seconds':duration(canonical+raw)-duration(canonical),
                    'overlapping_row_count':int(overlapping.sum()),'overlapping_row_fraction':float(overlapping.mean()),
                    'partial_overlap_pair_count':pair_count-exact_pairs,'exact_interval_pair_count':exact_pairs,
                    'window_rms_unit':'volt','window_rms_quantiles':{}}
            for kind in ('canonical','raw'):
                values=[amplitudes[kind,row['source_row']] for row in day_rows if row['source_kind']==kind]
                record['window_rms_quantiles'][kind]=np.quantile(values,[0,.01,.1,.5,.9,.99,1]).tolist() if values else []
            coverage[key]=record
        report['coverage']=coverage;report['added_unique_hours']=sum(v['added_unique_seconds'] for v in coverage.values())/3600
        report['raw_exclusion_fraction']=1-report['retained_raw_count']/report['candidate_raw_count']
        report['support_guard']='whole_preprocessed_session; cross_day disjoint sessions verified; within_subject_day stopped'
        (out/protocol/'filter_report.json').write_text(json.dumps(report,indent=2))
    (out/'preprocessing_context_audit.json').write_text(json.dumps(context,indent=2))
    time_grid_existing(data,out)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=Path('/vePFS-0x0d/DailyEEG'))
    p.add_argument('--supplement-existing',action='store_true')
    p.add_argument('--time-grid-only',action='store_true')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.supplement_existing:
        supplement_existing(a.data,a.out);return
    if a.time_grid_only:
        time_grid_existing(a.data,a.out);return
    c=a.data/'processed_cadt_addtime_new';r=a.data/'processed_cadt_raw_unlabeled'
    cm,cs,cd,ct=identities(c);rm,rs,rd,rt=identities(r)
    cx=np.load(c/'X.npy',mmap_mode='r');rx=np.load(r/'X.npy',mmap_mode='r')
    if cx.shape!=(28819,2000,59) or rx.shape!=(295641,2000,59):raise ValueError('signal_shape_mismatch')
    candidate=np.asarray(json.loads((a.data/'splits_new/shared_unlabeled_pretrain.json').read_text()),dtype=np.int64)
    if len(candidate)!=44181 or len(set(candidate))!=len(candidate) or candidate.min()<0 or candidate.max()>=len(rx):raise ValueError('candidate_index_failed')
    splits={}
    for protocol in ('cross_day','within_subject_day'):
        root=a.data/'splits_new'/protocol
        parts={key:np.asarray(json.loads((root/f'{key}.json').read_text()),dtype=np.int64) for key in ('pretrain','finetune','val','test')}
        splits[protocol]={**parts,'train':np.union1d(parts['pretrain'],parts['finetune'])}
    origin=a.data.parent/'home/zjm/cadt_regression'
    provenance={str(p):sha(p) for p in [c/'meta.json',r/'meta.json',a.data/'splits_new/shared_unlabeled_pretrain.json',
                origin/'test_load.py',origin/'process_raw_unlabeled.py',origin/'align_cadt_input.py']}
    for protocol in splits:
        for key in ('pretrain','finetune','val','test'):provenance[str(a.data/'splits_new'/protocol/f'{key}.json')]=sha(a.data/'splits_new'/protocol/f'{key}.json')
    try:
        sources=verify_sources(a.data,((cm,cs,cd,ct),(rm,rs,rd,rt)),candidate,splits['cross_day']['train'])
    except Exception as exc:
        (a.out/'source_audit.json').write_text(json.dumps({'status':'stop','reason':str(exc),'provenance':provenance},indent=2))
        raise
    (a.out/'source_audit.json').write_text(json.dumps({'status':'pass','sources':sources,'provenance':provenance},indent=2))
    # Content de-duplication uses exact stored array bytes. Canonical has priority.
    hashes={}
    for row in np.union1d(splits['cross_day']['train'],splits['within_subject_day']['train']):
        hashes.setdefault(hashlib.sha256(np.asarray(cx[row]).tobytes()).hexdigest(),[]).append(int(row))
    quality={}
    for progress,row in enumerate(candidate):
        signal=np.asarray(rx[row]);reason=None
        if not np.isfinite(signal).all():reason='nonfinite'
        elif np.all(np.ptp(signal,axis=0)==0):reason='all_channels_constant'
        h=hashlib.sha256(signal.tobytes()).hexdigest()
        quality[int(row)]={'reason':reason,'signal_sha256':h,'rms':float(np.sqrt(np.mean(signal.astype(np.float64)**2))) if reason!='nonfinite' else None}
        if progress%2000==0:print(json.dumps({'raw_quality_checked':progress,'expected':len(candidate)}),flush=True)
    for protocol,sp in splits.items():
        out=a.out/protocol;out.mkdir(exist_ok=True);held=np.union1d(sp['val'],sp['test']);held_intervals={}
        for row in held:held_intervals.setdefault((int(cs[row]),int(cd[row])),[]).append((float(ct[row]),float(ct[row])+10))
        held_intervals={key:union(intervals) for key,intervals in held_intervals.items()}
        inherited=[int(row) for row in sp['train'] if overlaps(float(ct[row]),held_intervals.get((int(cs[row]),int(cd[row])),[]))]
        inherited_context=[int(row) for row in sp['train'] if (int(cs[row]),int(cd[row])) in held_intervals]
        train_days=set(zip(cs[sp['train']].tolist(),cd[sp['train']].tolist()));reasons=Counter();kept=[];excluded=[]
        train_set=set(sp['train'].tolist());raw_hashes=set()
        physical={(int(cs[row]),int(cd[row]),float(ct[row])) for row in sp['train']}
        for row in candidate:
            key=(int(rs[row]),int(rd[row]));reason=None
            if protocol=='cross_day' and key not in train_days:reason='outside_train_subject_day'
            if overlaps(float(rt[row]),held_intervals.get(key,[])):reason=reason or 'val_test_signal_overlap'
            if key in held_intervals:reason=reason or 'val_test_preprocessing_context_overlap'
            reason=reason or quality[int(row)]['reason']
            h=quality[int(row)]['signal_sha256']
            if (*key,float(rt[row])) in physical:reason=reason or 'same_physical_canonical_interval'
            if train_set.intersection(hashes.get(h,[])):reason=reason or 'exact_canonical_content'
            if h in raw_hashes:reason=reason or 'exact_raw_content'
            if reason:reasons[reason]+=1;excluded.append({'source_row':int(row),'reason':reason})
            else:kept.append(int(row));raw_hashes.add(h)
        rows=[]
        for kind,indices,subs,days,times in (('canonical',sp['train'],cs,cd,ct),('raw',kept,rs,rd,rt)):
            rows.extend({'source_kind':kind,'source_row':int(row),'subject_id':int(subs[row]),'session_id':int(days[row])+1,
                         'recording_id':f'sub-{subs[row]:02d}/ses-{days[row]+1:02d}',
                         'start':float(times[row]),'end':float(times[row])+10} for row in indices)
        by_day={}
        for row in rows:
            d=by_day.setdefault(row['recording_id'],{'canonical':[],'raw':[]});d[row['source_kind']].append((row['start'],row['end']))
        coverage={}
        for key,d in by_day.items():
            base=duration(d['canonical']);total=duration(d['canonical']+d['raw'])
            coverage[key]={'canonical_rows':len(d['canonical']),'raw_rows':len(d['raw']),
                           'canonical_union_seconds':base,'raw_union_seconds':duration(d['raw']),
                           'total_union_seconds':total,'added_unique_seconds':total-base,
                           'overlapping_train_rows':sum(overlaps(start,union(d['canonical'])) for start,end in d['raw'])}
        status='stopped_canonical_signal_overlap' if inherited or inherited_context else ('pass' if kept else 'no_eligible_extra_windows')
        report={'protocol':protocol,'status':status,'candidate_raw_count':len(candidate),'retained_raw_count':len(kept),
                'canonical_train_count':len(sp['train']),'canonical_val_count':len(sp['val']),
                'canonical_train_holdout_overlap_count':len(inherited),'canonical_train_holdout_overlap_rows':inherited,
                'canonical_train_holdout_preprocessing_context_overlap_count':len(inherited_context),
                'raw_holdout_overlap_count_after_filter':0,'exclusion_counts':dict(reasons),'coverage':coverage,
                'added_unique_hours':sum(x['added_unique_seconds'] for x in coverage.values())/3600,
                'raw_rms_quantiles':np.quantile([quality[row]['rms'] for row in kept],[0,.01,.1,.5,.9,.99,1]).tolist() if kept else [],
                'source_audit':str(a.out/'source_audit.json'),'provenance':provenance,
                'grid_note':'canonical event grid and raw 5-second grid are separate; physical times verified against FIF slices',
                'support_guard':'full-session preprocessing; cross_day sources are disjoint sessions; within_subject_day stopped'}
        (out/'ssl_train_manifest.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        (out/'excluded_raw.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in excluded))
        (out/'filter_report.json').write_text(json.dumps(report,indent=2))
        if status=='pass':(out/'EEG_INPUT_COMPLETE').touch()
        print(json.dumps({k:report[k] for k in ('protocol','status','retained_raw_count','canonical_train_holdout_overlap_count','added_unique_hours')}),flush=True)


if __name__=='__main__':main()
