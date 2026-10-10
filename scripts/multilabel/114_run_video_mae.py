#!/usr/bin/env python3
"""Train a label-free VideoMAE-style encoder from aligned raw video clips.

This runner reads only clip metadata, canonical sample IDs, and formal split
indices.  It never loads emotion targets.  It is intended to execute where the
original MP4 files are mounted; its small outputs can then be copied to the
aligned training host.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from daily_multimodal.training.modality_mae import (
    fixed_validation_mask, guarded_optimizer_step, probe_health,
    representative_indices, require_finite, require_healthy,
)

VIDEO_TRAINING_VERSION = "seed_before_init_fixed_validation_health_v2"


def _json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _indices(path: Path, count: int) -> np.ndarray:
    values = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=np.int64)
    if values.ndim != 1 or len(values) == 0 or values.min() < 0 or values.max() >= count:
        raise ValueError(f"invalid split index file: {path}")
    return values


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _decode_clip(path: str, start: float, end: float, *, frames: int, size: int) -> np.ndarray:
    import cv2

    capture = cv2.VideoCapture(path)
    if not capture.isOpened(): raise RuntimeError(f"cannot open video: {path}")
    try:
        duration = max(float(end) - float(start), 1e-3)
        positions = np.linspace(float(start), float(start) + duration, frames, endpoint=False) + duration / (2 * frames)
        decoded: list[np.ndarray] = []
        for position in positions:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(position) * 1000.0)
            ok, image = capture.read()
            if not ok or image is None: raise RuntimeError(f"decode failure at {position:.3f}s: {path}")
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
            decoded.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return np.stack(decoded, axis=0).transpose(3, 0, 1, 2).astype(np.float32) / 255.0
    finally:
        capture.release()


class VideoMaskedAutoencoder(torch.nn.Module):
    """Small VideoMAE-style tubelet encoder with visible-token encoding."""

    def __init__(self, *, frames: int, size: int, embedding_dim: int, encoder_layers: int, decoder_layers: int, heads: int) -> None:
        super().__init__()
        if frames % 2 or size % 16: raise ValueError("frames must divide by 2 and size by 16")
        self.frames, self.size = frames, size
        self.grid = (frames // 2, size // 16, size // 16)
        self.token_count = int(np.prod(self.grid))
        self.patch_dim = 3 * 2 * 16 * 16
        self.patch_embed = torch.nn.Conv3d(3, embedding_dim, kernel_size=(2, 16, 16), stride=(2, 16, 16))
        self.position = torch.nn.Parameter(torch.zeros(1, self.token_count, embedding_dim))
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, embedding_dim))
        layer = torch.nn.TransformerEncoderLayer(embedding_dim, heads, 4 * embedding_dim, batch_first=True, activation="gelu")
        dec_layer = torch.nn.TransformerEncoderLayer(embedding_dim, heads, 4 * embedding_dim, batch_first=True, activation="gelu")
        self.encoder = torch.nn.TransformerEncoder(layer, encoder_layers)
        self.decoder = torch.nn.TransformerEncoder(dec_layer, decoder_layers)
        self.output = torch.nn.Linear(embedding_dim, self.patch_dim)
        torch.nn.init.normal_(self.position, std=0.02); torch.nn.init.normal_(self.mask_token, std=0.02)

    def _tokens(self, video: torch.Tensor) -> torch.Tensor:
        token = self.patch_embed(video).flatten(2).transpose(1, 2)
        if token.shape[1] != self.token_count: raise ValueError(f"unexpected video shape: {tuple(video.shape)}")
        return token

    def _patchify(self, video: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = video.shape
        x = video.reshape(b, c, t // 2, 2, h // 16, 16, w // 16, 16)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(b, self.token_count, self.patch_dim)
        return (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True).clamp_min(1e-6)

    def forward(self, video: torch.Tensor, masked: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token = self._tokens(video); visible = ~masked; visible_count = int(visible[0].sum().item())
        if visible_count < 1 or not bool(torch.all(visible.sum(dim=1) == visible_count)): raise ValueError("invalid MAE mask")
        order = masked.to(torch.int64).argsort(dim=1)
        visible_idx, masked_idx = order[:, :visible_count], order[:, visible_count:]
        embedded = token + self.position
        encoded = self.encoder(torch.gather(embedded, 1, visible_idx[:, :, None].expand(-1, -1, embedded.shape[-1])))
        full = self.mask_token.expand(video.shape[0], self.token_count, -1).clone() + self.position
        full.scatter_(1, visible_idx[:, :, None].expand_as(encoded), encoded)
        return self.output(self.decoder(full)), self._patchify(video)

    def embed(self, video: torch.Tensor) -> torch.Tensor:
        return self.encoder(self._tokens(video) + self.position).mean(dim=1)


@dataclass(frozen=True)
class Runtime:
    epochs: int; batch_size: int; learning_rate: float; patience: int; mask_ratio: float; seed: int; device: str
    health_probe_count: int = 256
    min_relative_variation: float = 1e-3


def _mask(batch: int, tokens: int, ratio: float, device: torch.device) -> torch.Tensor:
    count = max(1, min(tokens - 1, int(round(tokens * ratio))))
    noise = torch.rand((batch, tokens), device=device)
    return noise.argsort(dim=1).argsort(dim=1) < count


def _cache_manifest(path: Path) -> Path:
    return path.with_suffix(path.suffix + '.json')


def _build_cache(meta: dict[str, np.ndarray], valid: np.ndarray, *, frames: int, size: int, path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Resume cache construction and quarantine clips that fail decode on this host."""
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (len(valid), 3, frames, size, size)
    expected = {'shape': list(shape), 'dtype': 'uint8', 'format_version': 2}
    manifest = _cache_manifest(path)
    completed = np.zeros((len(valid),), dtype=bool)
    failures: list[dict[str, Any]] = []
    resumed_from_partial = False
    if manifest.is_file():
        try:
            previous = json.loads(manifest.read_text(encoding='utf-8'))
            if all(previous.get(k) == v for k, v in expected.items()):
                completed[np.asarray(previous.get('completed_indices', []), dtype=np.int64)] = True
                failures = list(previous.get('decode_failures', []))
                for item in failures:
                    valid[int(item['index'])] = False
        except (json.JSONDecodeError, ValueError, KeyError, IndexError):
            pass
    expected_bytes = int(np.prod(shape))
    if path.is_file() and path.stat().st_size == expected_bytes:
        cache = np.memmap(path, mode='r+', dtype=np.uint8, shape=shape)
        if not completed.any():
            # Legacy queues wrote no manifest.  A nonzero uint8 row is a safe
            # recovery marker for a fully decoded clip; untouched rows are zero.
            completed = np.asarray(cache).reshape(len(valid), -1).any(axis=1)
            resumed_from_partial = bool(completed.any())
    else:
        cache = np.memmap(path, mode='w+', dtype=np.uint8, shape=shape)

    def write_manifest(*, complete: bool) -> None:
        status = {
            **expected,
            'complete': bool(complete),
            'completed_indices': np.flatnonzero(completed).astype(int).tolist(),
            'decode_failures': failures,
        }
        manifest.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')

    try:
        pending = np.flatnonzero(valid & ~completed)
        for done, index in enumerate(pending, start=1):
            try:
                cache[index] = np.rint(_decode_clip(str(meta['source_video_file'][index]), float(meta['clip_start_seconds'][index]), float(meta['clip_end_seconds'][index]), frames=frames, size=size) * 255.0).astype(np.uint8)
                completed[index] = True
            except RuntimeError as exc:
                valid[index] = False
                failures.append({
                    'index': int(index), 'sample_id': str(meta['sample_id'][index]),
                    'source_video_file': str(meta['source_video_file'][index]),
                    'error': str(exc),
                })
                print(f'quarantined_decode_failure index={index} sample_id={meta["sample_id"][index]}', flush=True)
            if done % 250 == 0:
                cache.flush(); write_manifest(complete=False)
                print(f'cached_clips={int(completed.sum())}/{int(valid.sum())}', flush=True)
        cache.flush(); write_manifest(complete=True)
    finally:
        del cache
    return valid, {
        'cache_path': str(path), 'resumed_from_partial': resumed_from_partial,
        'cached_clip_count': int(completed.sum()), 'video_valid_count_after_decode': int(valid.sum()),
        'decode_failure_count': len(failures), 'decode_failures': failures,
    }


def _open_cache(path: Path, *, count: int, frames: int, size: int) -> np.memmap:
    manifest = _cache_manifest(path)
    if not path.is_file() or not manifest.is_file(): raise FileNotFoundError(f'missing complete clip cache: {path}')
    status = json.loads(manifest.read_text(encoding='utf-8'))
    shape = (count, 3, frames, size, size)
    if not status.get('complete') or status.get('shape') != list(shape) or status.get('dtype') != 'uint8': raise ValueError(f'invalid clip cache manifest: {manifest}')
    if path.stat().st_size != int(np.prod(shape)): raise ValueError(f'clip cache size differs from manifest: {path}')
    return np.memmap(path, mode='r', dtype=np.uint8, shape=shape)


def _reuse_complete_cache(meta: dict[str, np.ndarray], valid: np.ndarray, *, frames: int, size: int, path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Check an existing cache without reopening it for writes."""
    cache = _open_cache(path, count=len(valid), frames=frames, size=size)
    del cache
    status = json.loads(_cache_manifest(path).read_text(encoding='utf-8'))
    valid = valid.copy()
    for failure in status.get('decode_failures', []):
        index = int(failure['index'])
        if str(meta['sample_id'][index]) != str(failure['sample_id']):
            raise ValueError('cache quarantine sample_id differs from metadata')
        valid[index] = False
    completed = np.zeros(len(valid), dtype=bool)
    completed[np.asarray(status['completed_indices'], dtype=np.int64)] = True
    if not completed[valid].all(): raise ValueError('complete cache contains an undecoded valid row')
    return valid, {'cache_path': str(path), 'cache_reused_read_only': True,
                   'cached_clip_count': int(completed.sum()), 'video_valid_count_after_decode': int(valid.sum()),
                   'decode_failure_count': len(status.get('decode_failures', [])), 'decode_failures': status.get('decode_failures', [])}


def _load_batch(meta: dict[str, np.ndarray], idx: np.ndarray, *, frames: int, size: int, cache: np.memmap | None) -> torch.Tensor:
    if cache is not None: return torch.from_numpy(np.asarray(cache[idx], dtype=np.float32) / 255.0)
    return torch.from_numpy(np.stack([_decode_clip(str(meta['source_video_file'][i]), float(meta['clip_start_seconds'][i]), float(meta['clip_end_seconds'][i]), frames=frames, size=size) for i in idx]))


@torch.no_grad()
def _loss(model: VideoMaskedAutoencoder, meta: dict[str, np.ndarray], idx: np.ndarray, runtime: Runtime, device: torch.device, *, frames: int, size: int, cache: np.memmap | None, details: bool = False):
    model.eval(); values=[]; weights=[]; zeros=[]; generator=np.random.default_rng(runtime.seed+100003)
    for start in range(0, len(idx), runtime.batch_size):
        batch=idx[start:start+runtime.batch_size]; video=_load_batch(meta,batch,frames=frames,size=size,cache=cache).to(device); mask=fixed_validation_mask(len(batch),model.token_count,runtime.mask_ratio,generator,device)
        pred,target=model(video,mask); loss=((pred-target).pow(2).mean(dim=-1)[mask]).mean()
        require_finite(loss,'video validation loss'); values.append(float(loss.cpu())); weights.append(len(batch))
        zeros.append(float(target.pow(2).mean(dim=-1)[mask].mean().cpu()))
    value=float(np.average(values,weights=weights));zero=float(np.average(zeros,weights=weights))
    return {'masked_nmse':value,'zero_masked_nmse':zero,'relative_to_zero':value/max(zero,1e-12)} if details else value


def _train(model: VideoMaskedAutoencoder, meta: dict[str, np.ndarray], train: np.ndarray, val: np.ndarray, runtime: Runtime, *, frames: int, size: int, cache: np.memmap | None) -> tuple[VideoMaskedAutoencoder, dict[str, Any]]:
    started=time.perf_counter()
    initial_sha=hashlib.sha256(b''.join(value.detach().cpu().numpy().tobytes() for value in model.state_dict().values())).hexdigest()
    _seed(runtime.seed); device=torch.device(runtime.device); model=model.to(device); optim=torch.optim.AdamW(model.parameters(),lr=runtime.learning_rate,weight_decay=1e-4); rng=np.random.default_rng(runtime.seed)
    best_state=None; best=float('inf'); best_epoch=0; wait=0; history=[]
    for epoch in range(1,runtime.epochs+1):
        shuffled=train.copy(); rng.shuffle(shuffled); losses=[]; model.train()
        for start in range(0,len(shuffled),runtime.batch_size):
            batch=shuffled[start:start+runtime.batch_size]; video=_load_batch(meta,batch,frames=frames,size=size,cache=cache).to(device); mask=_mask(len(batch),model.token_count,runtime.mask_ratio,device)
            pred,target=model(video,mask); loss=((pred-target).pow(2).mean(dim=-1)[mask]).mean()
            guarded_optimizer_step(model,optim,loss,f'video epoch={epoch} batch={start//runtime.batch_size}')
            losses.append(float(loss.detach().cpu()))
        val_detail=_loss(model,meta,val,runtime,device,frames=frames,size=size,cache=cache,details=True);val_loss=val_detail['masked_nmse']
        health=probe_health(model,val,runtime.batch_size,
            lambda idx: model.embed(_load_batch(meta,idx,frames=frames,size=size,cache=cache).to(device)),
            count=runtime.health_probe_count,minimum=runtime.min_relative_variation)
        record={'epoch':epoch,'train_masked_nmse':float(np.mean(losses)),'val_masked_nmse':val_loss,'embedding_health':health,'validation':val_detail,
                'optimizer_steps':epoch*((len(train)+runtime.batch_size-1)//runtime.batch_size),'seen_windows':epoch*len(train)}
        history.append(record); print(json.dumps({'modality':'video',**record}),flush=True)
        require_healthy(health,f'video epoch={epoch}')
        if val_loss<best: best,best_epoch,wait=val_loss,epoch,0; best_state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
        else:
            wait+=1
            if wait>=runtime.patience: break
    if best_state is None: raise RuntimeError('training failed before checkpoint selection')
    model.load_state_dict(best_state)
    return model,{'best_epoch':best_epoch,'best_val_masked_nmse':best,'history':history,'train_count':int(len(train)),'val_count':int(len(val)),'selected_embedding_health':history[best_epoch-1]['embedding_health'],
                 'initial_state_sha256':initial_sha,'optimizer_steps':history[-1]['optimizer_steps'],'best_checkpoint_steps':history[best_epoch-1]['optimizer_steps'],'seen_windows':history[-1]['seen_windows'],'gpu_wall_seconds':time.perf_counter()-started}


@torch.no_grad()
def _export(model: VideoMaskedAutoencoder, meta: dict[str, np.ndarray], valid: np.ndarray, runtime: Runtime, *, frames: int, size: int, cache: np.memmap | None) -> np.ndarray:
    device=torch.device(runtime.device); model.eval(); out=np.zeros((len(valid),model.position.shape[-1]),dtype=np.float32); idx=np.flatnonzero(valid)
    for start in range(0,len(idx),runtime.batch_size):
        batch=idx[start:start+runtime.batch_size]; value=model.embed(_load_batch(meta,batch,frames=frames,size=size,cache=cache).to(device))
        require_finite(value,f'video export start={start}'); out[batch]=value.cpu().numpy()
    return out


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-metadata',type=Path,required=True); parser.add_argument('--splits-root',type=Path,required=True); parser.add_argument('--protocol',choices=('cross_day','within_subject_day'),required=True); parser.add_argument('--out-root',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=8); parser.add_argument('--size',type=int,default=112); parser.add_argument('--embedding-dim',type=int,default=256); parser.add_argument('--encoder-layers',type=int,default=6); parser.add_argument('--decoder-layers',type=int,default=2); parser.add_argument('--heads',type=int,default=8)
    parser.add_argument('--epochs',type=int,default=40); parser.add_argument('--batch-size',type=int,default=8); parser.add_argument('--learning-rate',type=float,default=1e-4); parser.add_argument('--patience',type=int,default=8); parser.add_argument('--mask-ratio',type=float,default=0.9); parser.add_argument('--seed',type=int,default=240800); parser.add_argument('--device',default='cuda'); parser.add_argument('--smoke-per-split',type=int,default=0); parser.add_argument('--skip-full-export',action='store_true'); parser.add_argument('--clip-cache',type=Path); parser.add_argument('--build-cache-only',action='store_true')
    parser.add_argument('--reuse-complete-cache',action='store_true'); parser.add_argument('--health-probe-count',type=int,default=256); parser.add_argument('--min-relative-variation',type=float,default=1e-3)
    parser.add_argument('--input-view',choices=('V_ROI','V_FULL_MATCH'))
    args=parser.parse_args()
    if not torch.cuda.is_available() and args.device.startswith('cuda'): raise RuntimeError('CUDA required for VideoMAE')
    if not 0 < args.mask_ratio < 1: raise ValueError('mask ratio must be strictly between zero and one')
    if args.reuse_complete_cache and (not args.clip_cache or args.build_cache_only): raise ValueError('read-only cache reuse requires --clip-cache and forbids --build-cache-only')
    with np.load(args.video_metadata,allow_pickle=True) as data:
        required=('sample_id','source_video_file','clip_start_seconds','clip_end_seconds','video_mask')
        if any(key not in data for key in required): raise ValueError('video metadata lacks required fields')
        meta={key:data[key] for key in required}
        if args.input_view and ('input_view' not in data or str(data['input_view'].item())!=args.input_view):raise ValueError('metadata input view mismatch')
    count=len(meta['sample_id']); valid=meta['video_mask'].astype(bool)
    missing=[str(p) for p in meta['source_video_file'][valid] if not Path(str(p)).is_file()]
    if missing: raise FileNotFoundError(f'{len(missing)} video paths unavailable; first={missing[0]}')
    cache_audit: dict[str, Any] = {'cache_used': False, 'decode_failure_count': 0, 'decode_failures': []}
    if args.clip_cache:
        valid, cache_audit = (_reuse_complete_cache if args.reuse_complete_cache else _build_cache)(meta, valid, frames=args.frames, size=args.size, path=args.clip_cache)
        if args.build_cache_only:
            print(json.dumps({'gate':'pass','clip_cache':str(args.clip_cache),'video_valid_count':int(valid.sum()), **cache_audit})); return 0
        cache = _open_cache(args.clip_cache, count=count, frames=args.frames, size=args.size)
    else: cache = None
    if args.input_view:
        if not args.reuse_complete_cache or not args.clip_cache:raise ValueError('paired input view requires frozen cache')
        status=json.loads(_cache_manifest(args.clip_cache).read_text())
        identity=hashlib.sha256('\n'.join(meta['sample_id'].astype(str)).encode()).hexdigest()
        if status.get('input_view')!=args.input_view or status.get('sample_ids_sha256')!=identity:raise ValueError('cache view or order mismatch')
        expected=np.zeros(count,dtype=bool);expected[np.asarray(status['completed_indices'],dtype=int)]=True
        if not np.array_equal(expected,valid):raise ValueError('paired common mask changed')
        cache_audit.update(input_view=args.input_view,input_sources=status['input_sources'],common_mask_sha256=status['common_mask_sha256'],frame_mapping_sha256=status['frame_mapping_sha256'])
    split={name:_indices(args.splits_root/args.protocol/f'{name}.json',count) for name in ('pretrain','finetune','val','test')}; train=np.concatenate((split['pretrain'],split['finetune'])); train=train[valid[train]]; val=split['val'][valid[split['val']]]
    if args.smoke_per_split: train,val=representative_indices(train,args.smoke_per_split),representative_indices(val,args.smoke_per_split)
    if not len(train) or not len(val): raise ValueError('no video-valid train or val windows')
    runtime=Runtime(args.epochs,args.batch_size,args.learning_rate,args.patience,args.mask_ratio,args.seed,args.device,args.health_probe_count,args.min_relative_variation); out=args.out_root/args.protocol/f'video_seed_{args.seed}'; out.mkdir(parents=True,exist_ok=True)
    if (out/'checkpoint.pt').exists(): raise FileExistsError(f'preserve existing checkpoint; choose a new output root: {out}')
    _seed(runtime.seed)
    model=VideoMaskedAutoencoder(frames=args.frames,size=args.size,embedding_dim=args.embedding_dim,encoder_layers=args.encoder_layers,decoder_layers=args.decoder_layers,heads=args.heads); model,audit=_train(model,meta,train,val,runtime,frames=args.frames,size=args.size,cache=cache)
    config={'route':'video_mae_label_free_v1','modality':'video','protocol':args.protocol,'supervision_boundary':'unlabeled_raw_video_reconstruction_train_pretrain_plus_finetune__validation_reconstruction_selection','test_labels_read':False,'row_count':count,'video_valid_count':int(valid.sum()),'effective_train_count':int(len(train)),'effective_val_count':int(len(val)),'embedding_seed':args.seed,'runtime':asdict(runtime),'model':{'frames':args.frames,'spatial_size':args.size,'tubelet_size':[2,16,16],'embedding_dim':args.embedding_dim,'encoder_layers':args.encoder_layers,'decoder_layers':args.decoder_layers,'heads':args.heads},'cache_audit':cache_audit,'reconstruction_audit':audit}
    config.update(training_version=VIDEO_TRAINING_VERSION,preprocessing_version='raw_rgb_255__normalized_tubelet_target_v1',validation_mask='fixed_seed_plus_100003_fixed_count_independent_of_training_rng',smoke_per_split=args.smoke_per_split)
    config.update(input_variant=args.input_view or 'historical_full_frame',metadata_sha256=hashlib.sha256(args.video_metadata.read_bytes()).hexdigest(),split_root=str(args.splits_root/args.protocol))
    if args.smoke_per_split: config['smoke_indices']={'train':train.tolist(),'val':val.tolist()}
    checkpoint=out/'checkpoint.pt'; torch.save({'state_dict':model.state_dict(),'config':config},checkpoint); config['checkpoint']=str(checkpoint)
    if args.skip_full_export:
        config['token_export']='skipped_for_smoke'
    else:
        token=out/'window_embeddings.npz'; np.savez_compressed(token,sample_id=meta['sample_id'].astype(str),embedding=_export(model,meta,valid,runtime,frames=args.frames,size=args.size,cache=cache),valid_mask=valid); config['token_path']=str(token)
    _json(out/'config.json',config); print(json.dumps({'gate':'pass','checkpoint':str(checkpoint),'best_epoch':audit['best_epoch'],'best_val_masked_nmse':audit['best_val_masked_nmse'],'token_export':config.get('token_path','skipped_for_smoke')},ensure_ascii=False)); return 0


if __name__=='__main__': raise SystemExit(main())
