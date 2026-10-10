#!/usr/bin/env python3
"""One-shot Windows coordinator: wait for caches, transfer small products, finish reports."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

NCC='/home/lzs/mae_round1_input_20261010'
H20='/home/wangzw/mae_round1_input_20261010'
NPY='/home/lzs/miniconda3/envs/eeg3dim/bin/python'
HPY='/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/runtime/envs/eegpt-gpu-min/bin/python'
Q=shlex.quote


def remote(host,command):
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,command],text=True,capture_output=True,timeout=60)
    if r.returncode:raise RuntimeError(f'{host}: {r.stderr.strip()} {r.stdout.strip()}')
    return r.stdout.strip()


def python(host,runtime,program):return remote(host,Q(runtime)+' -c '+Q(program))


def wait_marker(host,root,marker,failed=None):
    failure_count=0
    while True:
        try:
            program=f"from pathlib import Path; r=Path({root!r}); print('complete' if (r/{marker!r}).exists() else 'failed' if {bool(failed)!r} and (r/{failed or ''!r}).exists() else 'waiting')"
            status=python(host,NPY if host=='ncc_serve_4090' else HPY,program);failure_count=0
            if status=='complete':print(f'{host}: {marker}',flush=True);return
            if status=='failed':raise ValueError(f'queue failed: {host}:{root}/{failed}')
            if marker=='inputs/video/VIDEO_INPUT_COMPLETE':
                alive=python(host,NPY,f"from pathlib import Path; r=Path({root!r}); p=r/'video_inputs.pid'; pid=p.read_text().strip() if p.exists() else ''; print(Path('/proc/'+pid+'/cmdline').exists())")
                if alive!='True':raise ValueError('video input builder exited without completion; inspect video_inputs.log')
        except (RuntimeError, subprocess.TimeoutExpired, OSError) as exc:
            failure_count+=1;print(f'transient SSH failure {failure_count}: {exc}',flush=True)
            if failure_count>=10:raise
        time.sleep(30)


def wait_gpus(host,candidates):
    while True:
        available=[]
        status=remote(host,'nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits')
        for line in status.splitlines():
            index,free,util=[float(value.strip()) for value in line.split(',')]
            if int(index) in candidates and free>=6144 and util<=20:available.append(str(int(index)))
        if available:
            print(json.dumps({'host':host,'available_gpus_at_launch':available,'gpu_preflight':status}),flush=True)
            return ','.join(available)
        time.sleep(30)


def launch(host,root,kind,phase,gpus,pid_name):
    runtime=NPY if host=='ncc_serve_4090' else HPY
    command=f"cd {Q(root)} && nohup env PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 {Q(runtime)} -u scripts/multilabel/135_run_round1_stages.py --root {Q(root)} --kind {kind} --phase {phase} --gpus {Q(gpus)} > logs/{pid_name}.log 2>&1 < /dev/null &"
    # A saved PID is checked against its command line before reusing the queue.
    body=f"cd {Q(root)} || exit 1\nif [ -f {pid_name}.pid ]; then pid=$(cat {pid_name}.pid); if [ -r /proc/$pid/cmdline ] && grep -aq 135_run_round1_stages /proc/$pid/cmdline; then echo retaining_pid=$pid; exit 0; fi; fi\n{command}\necho $! > {Q(root)}/{pid_name}.pid\ncat {Q(root)}/{pid_name}.pid\n"
    r=subprocess.run(['ssh','-o','BatchMode=yes',host,'bash -s'],input=body,text=True,capture_output=True,timeout=60)
    if r.returncode:raise RuntimeError(r.stderr+r.stdout)
    print(f'{host}: {r.stdout.strip()}',flush=True)


def transfer(host,source,local,destination_host=None,destination=None):
    digest=remote(host,'sha256sum '+Q(source)).split()[0]
    subprocess.run(['scp',f'{host}:{source}',str(local)],check=True,timeout=600)
    if hashlib.sha256(local.read_bytes()).hexdigest()!=digest:raise ValueError('download_sha256_mismatch')
    if destination_host:
        subprocess.run(['scp',str(local),f'{destination_host}:{destination}'],check=True,timeout=600)
        if remote(destination_host,'sha256sum '+Q(destination)).split()[0]!=digest:raise ValueError('upload_sha256_mismatch')
    return digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--local-out',type=Path,required=True);a=p.parse_args();a.local_out.mkdir(parents=True,exist_ok=True)
    wait_marker('ncc_serve_4090',NCC,'inputs/video/VIDEO_INPUT_COMPLETE')
    # Every formal video split must match the H20 canonical split byte for byte.
    for protocol in ('cross_day','within_subject_day'):
        for leaf in ('pretrain','finetune','val','test'):
            rel=f'{protocol}/{leaf}.json'
            aa=remote('ncc_serve_4090','sha256sum '+Q(NCC+'/inputs/splits/'+rel)).split()[0]
            bb=remote('huoshan_TriDim','sha256sum '+Q('/vePFS-0x0d/DailyEEG/splits_new/'+rel)).split()[0]
            if aa!=bb:raise ValueError('cross_host_split_mismatch:'+rel)
    launch('ncc_serve_4090',NCC,'video','a',wait_gpus('ncc_serve_4090',{0,2,3,4}),'video_queue')
    wait_marker('ncc_serve_4090',NCC,'VIDEO_STAGE_A_FORMAL_COMPLETE','VIDEO_QUEUE_FAILED.json')
    remote('ncc_serve_4090',f'cd {Q(NCC)} && tar -cf video_transfer.tar stage_a/V_FULL_MATCH stage_a/V_ROI inputs/video/common_mask.npz inputs/video/frame_mapping.jsonl inputs/video/*.json inputs/video/io_diagnostic_20261010 inputs/splits inputs/fixed_config.json')
    digest=transfer('ncc_serve_4090',NCC+'/video_transfer.tar',a.local_out/'video_transfer.tar','huoshan_TriDim',H20+'/video_transfer.tar')
    remote('huoshan_TriDim',f'cd {Q(H20)} && tar -xf video_transfer.tar')
    validation=f"""import json,numpy as np
from pathlib import Path
r=Path({H20!r});ids=np.asarray([json.loads(line)['sample_id'] for line in Path('/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned/index/eeg_aligned_window_index.jsonl').open()])
with np.load(r/'inputs/video/common_mask.npz') as z:
 assert np.array_equal(ids,z['sample_id'].astype(str));mask=z['valid_mask'].astype(bool)
for view in ('V_FULL_MATCH','V_ROI'):
 for p in ('cross_day','within_subject_day'):
  with np.load(r/f'stage_a/{{view}}/formal/{{p}}/video_seed_240800/window_embeddings.npz') as z:
   assert np.array_equal(z['sample_id'].astype(str),ids) and np.array_equal(z['valid_mask'].astype(bool),mask)
   assert z['embedding'].shape==(28819,256) and np.isfinite(z['embedding']).all() and (z['embedding'][~mask]==0).all()
(r/'video_transfer_verified.json').write_text(json.dumps({{'sha256':{digest!r},'canonical_identity':'pass','mask':'pass'}}))
print('video_transfer_verified')"""
    print(python('huoshan_TriDim',HPY,validation),flush=True)
    launch('huoshan_TriDim',H20,'video','b',wait_gpus('huoshan_TriDim',{2}),'video_downstream_queue')
    wait_marker('huoshan_TriDim',H20,'EEG_STAGE_B_FORMAL_COMPLETE','EEG_QUEUE_FAILED.json')
    wait_marker('huoshan_TriDim',H20,'VIDEO_STAGE_B_FORMAL_COMPLETE','VIDEO_QUEUE_FAILED.json')
    report_command=f'cd {Q(H20)} && env PYTHONPATH=src {Q(HPY)} scripts/multilabel/136_summarize_round1_inputs.py --root {Q(H20)} --bootstrap-iters 2000 > logs/report.log 2>&1 || touch {Q(H20)}/REPORT_FAILED'
    remote('huoshan_TriDim','nohup bash -c '+Q(report_command)+' > /dev/null 2>&1 < /dev/null &')
    wait_marker('huoshan_TriDim',H20,'ROUND1_EXECUTION_FINISHED_WITH_PROTOCOL_STOP','REPORT_FAILED')
    program=f"""import tarfile
from pathlib import Path
r=Path({H20!r})
with tarfile.open(r/'lightweight_results.tar','w') as t:
 for directory in ('reports','inputs','stage_a','logs'):
  for p in (r/directory).rglob('*'):
   if p.is_file() and p.suffix in ('.json','.jsonl','.md','.csv','.log'):t.add(p,arcname=str(p.relative_to(r)))
 for name in ('ROUND1_EXECUTION_FINISHED_WITH_PROTOCOL_STOP','video_transfer_verified.json'):t.add(r/name,arcname=name)
print('lightweight_ready')"""
    python('huoshan_TriDim',HPY,program)
    transfer('huoshan_TriDim',H20+'/lightweight_results.tar',a.local_out/'lightweight_results.tar')
    subprocess.run(['tar','-xf',str(a.local_out/'lightweight_results.tar'),'-C',str(a.local_out)],check=True)
    (a.local_out/'PIPELINE_COMPLETE').touch();print('ROUND1_PIPELINE_COMPLETE_WITH_PROTOCOL_STOP',flush=True)


if __name__=='__main__':main()
