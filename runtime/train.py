"""DDP trainer with atomic checkpoints and stateless, resumable token sampling."""
import argparse
import contextlib
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .model import build_model
from .precision import configure as configure_precision, autocast


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path); temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        json.dump(data, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def log(run, rank, event, **fields):
    row = {'time': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'event':event, **fields}
    with (run / f'rank{rank}.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    if rank == 0:
        print(json.dumps(row), flush=True)
        atomic_json(run/'heartbeat.json', row)


def cross_entropy(logits, targets):
    return torch.nn.functional.cross_entropy(logits.flatten(0,1),targets.flatten())


class Tokens:
    def __init__(self, manifest, split, length, seed):
        self.seed, self.length = seed, length
        self.arrays, self.ends = [], []
        count = 0
        for row in manifest['shards']:
            p = Path(row[split]['path'])
            if p.stat().st_size != row[split]['tokens'] * 4:
                raise ValueError(f'Incomplete token shard {p}')
            arr = np.memmap(p, mode='r', dtype='<u4')
            blocks = max(0, (len(arr)-1)//length)
            if blocks:
                self.arrays.append(arr); count += blocks; self.ends.append(count)
        self.count = count
        if not count: raise ValueError('Dataset has no full sequences')

    def sample(self, index):
        epoch, pos = divmod(index, self.count)
        # Bijective block shuffle, independent of worker scheduling and restarts.
        rng = random.Random(self.seed + epoch)
        a = rng.randrange(1, self.count+1)
        while math.gcd(a, self.count) != 1: a = (a + 1) % self.count or 1
        pos = (a*pos + rng.randrange(self.count)) % self.count
        shard = int(np.searchsorted(self.ends, pos, side='right'))
        offset = (pos - (self.ends[shard-1] if shard else 0)) * self.length
        return torch.from_numpy(np.array(self.arrays[shard][offset:offset+self.length+1], dtype=np.int64))


def latest_checkpoint(run, fingerprint):
    # Incomplete .tmp files never become checkpoints. Verify content before load.
    entries = sorted(run.glob('checkpoint_*.json'), reverse=True)
    errors = []
    for entry in entries:
        try:
            meta = json.loads(entry.read_text())
            if meta['fingerprint'] != fingerprint: raise RuntimeError('Configuration mismatch: refusing to resume')
            path = run/meta['file']
            if sha256(path) != meta['sha256']: raise ValueError('Checkpoint SHA256 mismatch')
            return path
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f'{entry.name}: {exc}')
    if entries: raise RuntimeError('No valid checkpoint: ' + '; '.join(errors))
    return None


def save_checkpoint(run, model, optimizer, scaler, step, fingerprint, rank, world, keep_last=3):
    state = {'cpu':torch.get_rng_state(), 'cuda':torch.cuda.get_rng_state(),
             'python':random.getstate(), 'numpy':np.random.get_state()}
    rngs = [None]*world
    if world > 1: dist.all_gather_object(rngs, state)
    else: rngs[0] = state
    if rank == 0:
        name = f'checkpoint_{step:08d}.pt'; path=run/name; tmp=run/(name+'.tmp')
        with tmp.open('wb') as f:
            torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict(),
                        'scaler':scaler.state_dict(), 'step':step, 'fingerprint':fingerprint,
                        'rngs':rngs, 'world_size':world}, f)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
        meta={'file':name, 'step':step, 'sha256':sha256(path), 'fingerprint':fingerprint}
        atomic_json(path.with_suffix('.json'),meta)
        atomic_json(run/'latest.json',meta)
        # null means retain every scheduled checkpoint. Legacy recipes retain three.
        if keep_last is not None:
            if not isinstance(keep_last, int) or keep_last < 1: raise ValueError('Invalid checkpoint retention')
            for old in sorted(run.glob('checkpoint_*.json'))[:-keep_last]:
                old.with_suffix('.pt').unlink(missing_ok=True); old.unlink()
    if world > 1: dist.barrier()


def verify_fork(checkpoint, config, manifest, world):
    """Allow a documented arithmetic/retention change, never an implicit restart."""
    checkpoint = Path(checkpoint).resolve()
    meta = json.loads(checkpoint.with_suffix('.json').read_text())
    source = json.loads((checkpoint.parent/'resolved_config.json').read_text())
    identity = source['identity']
    expected = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if meta['file'] != checkpoint.name or meta['fingerprint'] != expected or sha256(checkpoint) != meta['sha256']:
        raise RuntimeError('Fork source checkpoint integrity mismatch')
    old = source['config']
    for key in ('model','tied_embeddings','seed','memory_fusion'):
        if old.get(key) != config.get(key): raise RuntimeError(f'Fork changes {key}')
    if source['synthetic_smoke'] or source['world_size'] != world or source['data'] != manifest:
        raise RuntimeError('Fork changes data, world size, or uses synthetic data')
    allowed = {'precision','checkpoint_every','keep_last_checkpoints'}
    previous = {k:v for k,v in old['training'].items() if k not in allowed}
    current = {k:v for k,v in config['training'].items() if k not in allowed}
    if previous != current: raise RuntimeError('Fork changes optimizer, schedule, batch, or validation')
    root = Path(__file__).parent
    for filename, expected_hash in identity['model_source_sha256'].items():
        if filename != 'train.py' and sha256(root/filename) != expected_hash:
            raise RuntimeError(f'Fork changes model source {filename}')
    return {'source_checkpoint':str(checkpoint), 'source_sha256':meta['sha256'],
            'source_fingerprint':meta['fingerprint'], 'source_step':meta['step'],
            'changes':{k:{'before':old['training'].get(k,3 if k=='keep_last_checkpoints' else None),
                          'after':config['training'].get(k,3 if k=='keep_last_checkpoints' else None)}
                       for k in sorted(allowed) if old['training'].get(k,3 if k=='keep_last_checkpoints' else None) != config['training'].get(k,3 if k=='keep_last_checkpoints' else None)},
            'preserved':['model','optimizer','RNG','data order','LR schedule','world size'],
            'scaler_policy':'restore for FP16; disabled for FP32'}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--run-dir',required=True)
    p.add_argument('--smoke',action='store_true'); p.add_argument('--stop-after',type=int)
    p.add_argument('--init-model',help='optional deterministic initialization checkpoint')
    p.add_argument('--resume-from',help='verified source checkpoint for a new arithmetic/retention fork')
    p.add_argument('--benchmark-updates',type=int,help='isolated throughput probe; no checkpoints or validation')
    a=p.parse_args(); c=json.loads(Path(a.config).read_text()); t=c['training']
    rank=int(os.environ.get('RANK',0)); world=int(os.environ.get('WORLD_SIZE',1)); local=int(os.environ.get('LOCAL_RANK',0))
    torch.set_num_threads(2); torch.cuda.set_device(local)
    precision=t.get('precision','fp16_mixed_fp32_params'); configure_precision(precision)
    execution=c.get('execution',{})
    if 'allow_tf32' in execution:
        torch.backends.cuda.matmul.allow_tf32=execution['allow_tf32']
        torch.backends.cudnn.allow_tf32=execution['allow_tf32']
    keep_last=t.get('keep_last_checkpoints',3)
    if keep_last is not None and (not isinstance(keep_last,int) or keep_last < 1): raise ValueError('Invalid checkpoint retention')
    if world>1: dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=180))
    if not a.smoke and world!=t['world_size']: raise RuntimeError('Formal training requires the configured world size')
    run=Path(a.run_dir); run.mkdir(parents=True,exist_ok=True)
    manifest=None
    if not a.smoke:
        if not c['data']['approved'] or not c['data']['manifest']: raise RuntimeError('Training data choice/preparation is not complete')
        manifest=json.loads(Path(c['data']['manifest']).read_text())
        if not manifest.get('ready'): raise RuntimeError('Dataset manifest is not ready')
    resolved={'config':c,'synthetic_smoke':a.smoke,'world_size':world,'data':manifest}
    identity={'model':c['model'],'tied_embeddings':c['tied_embeddings'],'seed':c['seed'],'training':t,'data':manifest,'synthetic':a.smoke,'world_size':world}
    if 'memory_fusion' in c:
        identity['memory_fusion'] = c['memory_fusion']
    if execution: identity['execution']=execution
    source_root=Path(__file__).parent
    identity['model_source_sha256']={str(p.relative_to(source_root)):sha256(p) for p in sorted(source_root.rglob('*.py'))}
    resolved['identity']=identity
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if rank==0: atomic_json(run/'resolved_config.json',resolved)
    random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
    model=build_model(c,checkpoint_activations=execution.get('checkpoint_activations',True)).cuda()
    init_path=Path(a.init_model) if a.init_model else None
    if init_path and init_path.is_file():
        init=torch.load(init_path,map_location='cpu',weights_only=False)
        if init['model_config'] != c['model'] or init['tied_embeddings'] != c['tied_embeddings']:
            raise RuntimeError('Initialization configuration mismatch')
        model.load_state_dict(init['model']); del init
    optimizer=torch.optim.AdamW(model.parameters(),lr=t['lr'],betas=tuple(t['betas']),eps=t['eps'],weight_decay=t['weight_decay'],fused=execution.get('fused_adamw',False))
    scaler=torch.amp.GradScaler('cuda',init_scale=128,enabled=precision=='fp16_mixed_fp32_params')
    checkpoint_path=latest_checkpoint(run,fingerprint) if rank==0 else None
    fork=None
    if rank==0 and not checkpoint_path and a.resume_from:
        if a.smoke or a.init_model: raise RuntimeError('A fork requires real data and no alternate initialization')
        fork=verify_fork(a.resume_from,c,manifest,world)
        checkpoint_path=fork['source_checkpoint']
        atomic_json(run/'fork-provenance.json',dict(fork,target_fingerprint=fingerprint,
            execution=execution,execution_note='Execution acceleration is recorded separately from the A/B training recipe.'))
    if world>1:
        paths=[str(checkpoint_path) if checkpoint_path else None]; dist.broadcast_object_list(paths,src=0); checkpoint_path=paths[0]
    step=0
    if checkpoint_path:
        saved=torch.load(checkpoint_path,map_location='cpu',weights_only=False,mmap=True)
        if saved['world_size']!=world: raise RuntimeError('Resume world size changed')
        if saved['fingerprint'] != fingerprint:
            provenance=json.loads((run/'fork-provenance.json').read_text())
            if saved['fingerprint'] != provenance['source_fingerprint'] or saved['step'] != provenance['source_step']:
                raise RuntimeError('Unverified source checkpoint identity')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        if scaler.is_enabled(): scaler.load_state_dict(saved['scaler'])
        step=saved['step']
        rng=saved['rngs'][rank]; torch.set_rng_state(rng['cpu']); torch.cuda.set_rng_state(rng['cuda']); random.setstate(rng['python']); np.random.set_state(rng['numpy']); del saved
    if execution.get('fused_adamw',False):
        # load_state_dict restores old param-group execution flags too.
        for group in optimizer.param_groups: group['fused']=True; group['foreach']=False
        for parameter,state in optimizer.state.items():
            if 'step' in state: state['step']=state['step'].to(parameter.device)
    compiled=torch.compile(model,dynamic=False) if execution.get('compile',False) else model
    loss_function=torch.compile(cross_entropy,dynamic=False) if execution.get('compile_loss',False) else cross_entropy
    train_model=DDP(compiled,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True) if world>1 else compiled
    log(run,rank,'started',step=step,world_size=world,parameters=sum(x.numel() for x in model.parameters()),synthetic=a.smoke,resumed=bool(checkpoint_path),precision=precision,execution=execution,checkpoint_every=t['checkpoint_every'],keep_last_checkpoints=keep_last)
    length=t['sequence_length']; micro=t['microbatch']; accum=t['gradient_accumulation']
    if manifest:
        train=Tokens(manifest,'train',length,c['seed']); valid=Tokens(manifest,'validation',length,c['seed']+1)
    def batch(index, validation=False):
        if a.smoke:
            generator=torch.Generator().manual_seed(c['seed']+index+(100000 if validation else 0))
            seq=torch.randint(4,min(100260,c['model']['vocab_size']),(micro,length+1),generator=generator)
        else:
            source=valid if validation else train
            seq=torch.stack([source.sample(index+j) for j in range(micro)])
        seq=seq.cuda(non_blocking=True); return seq[:,:-1],seq[:,1:]
    stop=min(a.stop_after or t['max_updates'],t['max_updates']); overflows=0
    if a.benchmark_updates:
        if a.benchmark_updates<6: raise ValueError('Benchmark needs at least six updates')
        stop=min(step+a.benchmark_updates,t['max_updates'])
    timings=[]
    while step<stop:
        started=time.monotonic(); train_model.train(); optimizer.zero_grad(set_to_none=True)
        update=step+1
        lr=t['lr']*update/t['warmup_updates'] if update<=t['warmup_updates'] else t['end_lr']+(t['lr']-t['end_lr'])*(t['max_updates']-update)/(t['max_updates']-t['warmup_updates'])
        for group in optimizer.param_groups: group['lr']=lr
        loss_sum=torch.zeros((),device='cuda')
        for j in range(accum):
            index=((step*accum+j)*world+rank)*micro
            x,y=batch(index)
            sync=train_model.no_sync() if world>1 and j<accum-1 else contextlib.nullcontext()
            with sync:
                with autocast(precision):
                    logits=train_model(x)[0]; loss=loss_function(logits,y)
                if not torch.isfinite(loss): raise RuntimeError('Non-finite loss; refusing unattended retry of divergent training')
                scaler.scale(loss/accum).backward(); loss_sum+=loss.detach()/accum
        scaler.unscale_(optimizer); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),t['grad_clip'])
        finite=torch.isfinite(norm).to(torch.int32)
        if world>1: dist.all_reduce(finite,op=dist.ReduceOp.MIN)
        if not finite.item():
            if not scaler.is_enabled(): raise RuntimeError('Non-finite FP32 gradient; stopping before optimizer update')
            scaler.update(new_scale=scaler.get_scale()/2); overflows+=1
            log(run,rank,'overflow',step=step,scale=scaler.get_scale(),consecutive=overflows)
            if overflows>=t['max_consecutive_overflows']: raise RuntimeError('Repeated gradient overflow')
            continue
        scaler.step(optimizer); scaler.update(); overflows=0; step=update
        if world>1: dist.all_reduce(loss_sum); loss_sum/=world
        torch.cuda.synchronize()
        elapsed=time.monotonic()-started; timings.append(elapsed)
        log(run,rank,'train',step=step,loss=loss_sum.item(),lr=lr,grad_norm=norm.item(),scale=scaler.get_scale(),tokens=step*world*micro*accum*length,seconds=elapsed,peak_memory_bytes=torch.cuda.max_memory_allocated())
        if not a.benchmark_updates and (step%t['validation_every']==0 or step==stop):
            model.eval(); vl=torch.zeros((),device='cuda')
            with torch.inference_mode(),autocast(precision):
                for j in range(1 if a.smoke else t['validation_batches']):
                    x,y=batch((j*world+rank)*micro,True); logits=model(x)[0]
                    vl+=torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten())
            vl/=1 if a.smoke else t['validation_batches']
            if world>1: dist.all_reduce(vl); vl/=world
            log(run,rank,'validation',step=step,loss=vl.item(),perplexity=math.exp(min(vl.item(),80)))
        if not a.benchmark_updates and (step%t['checkpoint_every']==0 or step==stop):
            save_checkpoint(run,model,optimizer,scaler,step,fingerprint,rank,world,keep_last)
            log(run,rank,'checkpoint',step=step)
    if rank==0:
        if a.benchmark_updates:
            atomic_json(run/'benchmark_summary.json',{'step':step,'execution':execution,'precision':precision,
                'measured_updates':len(timings)-5,'median_seconds':statistics.median(timings[5:]),
                'global_batch_tokens':world*micro*accum*length,'peak_memory_bytes':torch.cuda.max_memory_allocated(),
                'no_checkpoints_written':True,'not_a_formal_run':True})
        else: atomic_json(run/'stage_complete.json',{'step':step,'formal_complete':not a.smoke and step==t['max_updates'],'synthetic':a.smoke})
    if world>1: dist.destroy_process_group()

if __name__=='__main__': main()
