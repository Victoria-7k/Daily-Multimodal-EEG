#!/usr/bin/env python3
"""Build paired existing-ROI/full-frame caches without labels or face detection."""
from __future__ import annotations
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_frames(count):
    if count < 8:
        raise ValueError('fewer_than_8_frames')
    return np.floor((np.arange(8) + .5) * count / 8).astype(int)


def source_indices(row, fps, count):
    times=np.linspace(row['clip_start_seconds'],max(row['clip_start_seconds'],row['clip_end_seconds']-1e-3),16)
    return [max(0,min(count-1,int(round(float(times[k])*fps)))) if count else int(round(float(times[k])*fps)) for k in sample_frames(16)]


def read_sorted_frames(cap, indices, fps):
    """Read the same source frame IDs once, advancing through nearby GOPs."""
    import cv2
    result={};previous=None
    for fi in sorted(set(indices)):
        if previous is None or fi-previous>max(1,int(round(5*fps))):
            cap.set(cv2.CAP_PROP_POS_FRAMES,fi);previous=fi-1
        ok=True
        for _ in range(fi-previous):
            if not cap.grab():ok=False;break
        if ok:
            ok,img=cap.retrieve()
            if ok and img is not None:
                result[fi]=cv2.cvtColor(cv2.resize(img,(112,112),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
        previous=fi if ok else None
    return result


def process_group(job):
    import cv2
    cv2.setNumThreads(1)
    source, rows, root, shape, strategy = job
    full = cv2.VideoCapture(source)
    caches = {view: np.memmap(Path(root)/f'{view}.uint8.mmap', mode='r+', dtype=np.uint8, shape=shape)
              for view in ('V_ROI', 'V_FULL_MATCH')}
    results=[]
    try:
        if not full.isOpened():
            raise ValueError('source_video_open_failed')
        fps=float(full.get(cv2.CAP_PROP_FPS) or 30.0)
        n=int(full.get(cv2.CAP_PROP_FRAME_COUNT))
        selected=read_sorted_frames(full,[fi for row in rows for fi in source_indices(row,fps,n)],fps) if strategy=='sorted' else None
        for row in rows:
            cap=None
            try:
                i=row['index']; side=json.loads(Path(row['output_video_path']).with_name('openface_target.json').read_text())
                for key in ('sample_id','source_video_path','output_video_path','region','clip_start_seconds','clip_end_seconds'):
                    if side[key] != row[key]: raise ValueError('sidecar_manifest_mismatch:'+key)
                if row['region']!='2x_face_roi': raise ValueError('wrong_region')
                cap=cv2.VideoCapture(row['output_video_path']); images=[]
                while True:
                    ok,img=cap.read()
                    if not ok:break
                    images.append(img)
                # Existing sidecars do not record which source reads failed.
                # A partial clip therefore cannot establish local-to-source identity.
                if len(images)!=16: raise ValueError('unconfirmed_partial_frame_mapping:'+str(len(images)))
                local=sample_frames(16)
                times=np.linspace(row['clip_start_seconds'], max(row['clip_start_seconds'],row['clip_end_seconds']-1e-3),16)
                source_frames=[max(0,min(n-1,int(round(float(times[k])*fps)))) if n else int(round(float(times[k])*fps)) for k in local]
                if len(set(source_frames))!=8:raise ValueError('fewer_than_8_unique_source_frames')
                decoded=[]
                for fi in source_frames:
                    if selected is not None:
                        img=selected.get(fi);ok=img is not None
                    else:
                        full.set(cv2.CAP_PROP_POS_FRAMES,fi);ok,img=full.read()
                    if not ok or img is None:raise ValueError('full_matched_decode_failed:'+str(fi))
                    decoded.append(img)
                for view,frames in (('V_ROI',[images[k] for k in local]),('V_FULL_MATCH',decoded)):
                    value=np.stack(frames) if view=='V_FULL_MATCH' and selected is not None else np.stack([cv2.cvtColor(cv2.resize(x,(112,112),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB) for x in frames])
                    caches[view][i]=value.transpose(3,0,1,2)
                results.append({'index':i,'sample_id':row['sample_id'],'status':'ok','source_video_file':source,
                                'local_frame_indices':local.tolist(),'source_frame_indices':source_frames,
                                'source_fps':fps,'selected_source_times':times[local].tolist(),
                                'sidecar_sha256':digest(Path(row['output_video_path']).with_name('openface_target.json')),
                                'source_read_strategy':strategy,'decoder_builder_sha256':digest(__file__)})
            except Exception as exc:
                results.append({'index':row['index'],'sample_id':row['sample_id'],'status':'excluded','reason':str(exc)})
            finally:
                if cap is not None:cap.release()
    except Exception as exc:
        results=[{'index':r['index'],'sample_id':r['sample_id'],'status':'excluded','reason':str(exc)} for r in rows]
    finally:
        full.release()
        for c in caches.values():c.flush()
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export-root',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--old-token',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--source-read-strategy',choices=('seek','sorted'),default='seek')
    p.add_argument('--compatible-resume-builder',help='Explicit prior builder SHA; source identities must remain identical.')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    with np.load(args.metadata,allow_pickle=True) as z:meta={k:z[k] for k in z.files}
    ids=meta['sample_id'].astype(str);count=len(ids)
    if count!=28819 or len(set(ids))!=count:raise ValueError('canonical_identity_failed')
    roi_path=args.export_root/'embeddings/video/video_A1_2xroi_eeg23win_embeddings.npz'
    with np.load(roi_path,allow_pickle=True) as z:
        if not np.array_equal(z['sample_id'].astype(str),ids):raise ValueError('roi_order_mismatch')
        roi=z['valid_mask'].astype(bool) if 'valid_mask' in z else z['video_mask'].astype(bool)
    with np.load(args.old_token,allow_pickle=True) as z:
        if not np.array_equal(z['sample_id'].astype(str),ids):raise ValueError('old_order_mismatch')
        old=z['valid_mask'].astype(bool)
    manifest=args.export_root/'reports/video_2xroi_cache_manifest.jsonl'
    records=[json.loads(line) for line in manifest.open() if line.strip()]
    lookup={r['sample_id']:r for r in records}
    if len(lookup)!=len(records):raise ValueError('duplicate_roi_manifest_id')
    provenance={'metadata_sha256':digest(args.metadata),'old_token_sha256':digest(args.old_token),
                'roi_token_sha256':digest(roi_path),'manifest_sha256':digest(manifest),'builder_sha256':digest(__file__),
                'sample_ids_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest()}
    frozen=args.out/'input_sources.json'
    if frozen.exists() and json.loads(frozen.read_text())!=provenance:
        previous=json.loads(frozen.read_text())
        if previous.get('builder_sha256')!=args.compatible_resume_builder or {k:v for k,v in previous.items() if k!='builder_sha256'}!={k:v for k,v in provenance.items() if k!='builder_sha256'}:
            raise ValueError('input_changed_during_resume')
        archive=args.out/f'input_sources.previous.{previous["builder_sha256"]}.json'
        if not archive.exists():archive.write_text(json.dumps(previous,indent=2))
        (args.out/'decoder_revision.json').write_text(json.dumps({'previous_builder_sha256':previous['builder_sha256'],
            'current_builder_sha256':provenance['builder_sha256'],'source_read_strategy':args.source_read_strategy,
            'source_fingerprints_unchanged':True,'pixel_benchmark':'io_diagnostic_20261010/sequential_benchmark.json'},indent=2))
    frozen.write_text(json.dumps(provenance,indent=2))
    shape=(count,3,8,112,112)
    for view in ('V_ROI','V_FULL_MATCH'):
        path=args.out/f'{view}.uint8.mmap'
        if not path.exists():np.memmap(path,mode='w+',dtype=np.uint8,shape=shape).flush()
        elif path.stat().st_size!=int(np.prod(shape)):raise ValueError('cache_size_mismatch')
    completed={};mapping=args.out/'frame_mapping.jsonl'
    if mapping.exists():
        for line in mapping.open():
            r=json.loads(line);completed[r['index']]=r
    groups={};native=roi & old;initial_fail=[]
    for i in np.flatnonzero(native):
        if int(i) in completed:continue
        sid=ids[i];r=lookup.get(sid)
        try:
            if r is None:raise ValueError('missing_manifest_row')
            if r['source_video_path']!=str(meta['source_video_file'][i]):raise ValueError('source_path_mismatch')
            for key in ('clip_start_seconds','clip_end_seconds'):
                if abs(float(r[key])-float(meta[key][i]))>1e-3:raise ValueError('clip_time_mismatch')
            row={**r,'index':int(i)};groups.setdefault(r['source_video_path'],[]).append(row)
        except Exception as exc:initial_fail.append({'index':int(i),'sample_id':sid,'status':'excluded','reason':str(exc)})
    with mapping.open('a',encoding='utf-8') as handle:
        def save(rows):
            for r in rows:handle.write(json.dumps(r)+'\n');completed[r['index']]=r
            handle.flush()
        save(initial_fail)
        jobs=[(src,sorted(rows,key=lambda r:r['clip_start_seconds']),str(args.out),shape,args.source_read_strategy) for src,rows in groups.items()]
        with mp.Pool(args.workers) as pool:
            for rows in pool.imap_unordered(process_group,jobs):
                save(rows);print(json.dumps({'processed':len(completed),'expected':int(native.sum())}),flush=True)
    common=np.zeros(count,dtype=bool)
    for i,r in completed.items():common[i]=r['status']=='ok' and native[i]
    failures=[r for r in completed.values() if r['status']!='ok']
    report={'status':'pass','canonical_count':count,'old_valid':int(old.sum()),'roi_native_valid':int(roi.sum()),
            'roi_native_added_ids':ids[roi & ~old].tolist(),'common_valid':int(common.sum()),
            'failures':failures,'input_sources':provenance,'source_read_strategy':args.source_read_strategy,
            'decoder_revision':json.loads((args.out/'decoder_revision.json').read_text()) if (args.out/'decoder_revision.json').exists() else None}
    np.savez_compressed(args.out/'common_mask.npz',sample_id=ids,valid_mask=common)
    for view in ('V_ROI','V_FULL_MATCH'):
        for start in range(0,count,128):
            c=np.memmap(args.out/f'{view}.uint8.mmap',mode='r+',dtype=np.uint8,shape=shape)
            idx=np.arange(start,min(count,start+128));c[idx[~common[idx]]]=0;c.flush();del c
        payload={k:meta[k] for k in ('source_video_file','clip_start_seconds','clip_end_seconds')}
        np.savez_compressed(args.out/f'{view}_metadata.npz',sample_id=ids,video_mask=common,input_view=np.asarray(view),**payload)
        status={'complete':True,'shape':list(shape),'dtype':'uint8','format_version':3,'input_view':view,
                'completed_indices':np.flatnonzero(common).tolist(),'decode_failures':[],
                'sample_ids_sha256':provenance['sample_ids_sha256'],'common_mask_sha256':digest(args.out/'common_mask.npz'),
                'frame_mapping_sha256':digest(mapping),'input_sources':provenance}
        (args.out/f'{view}.uint8.mmap.json').write_text(json.dumps(status,indent=2))
    (args.out/'cache_report.json').write_text(json.dumps(report,indent=2))
    (args.out/'VIDEO_INPUT_COMPLETE').touch();print(json.dumps({k:v for k,v in report.items() if k!='failures'}),flush=True)


if __name__=='__main__':main()
