"""Read-only routing diagnostics on fixed validation prefixes in an allocated GPU job."""
import argparse
import contextlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from runtime.model import build_model
from runtime.train import Tokens, sha256, atomic_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--run-dir',required=True)
    p.add_argument('--initial',action='store_true');a=p.parse_args()
    c=json.loads(Path(a.config).read_text());run=Path(a.run_dir)
    if c.get('memory_fusion',{}).get('mode')!='token':
        return
    torch.set_num_threads(2);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    model=build_model(c).eval();step=0;checkpoint_sha=None
    if not a.initial:
        meta=json.loads((run/'latest.json').read_text());path=run/meta['file']
        checkpoint_sha=sha256(path);assert checkpoint_sha==meta['sha256']
        saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        model.load_state_dict(saved['model'],strict=True);step=saved['step'];del saved
    assert torch.cuda.is_available()
    model=model.cuda()
    source=Tokens(json.loads(Path(c['data']['manifest']).read_text()),'validation',2048,c['seed']+1)
    weights=[];norms=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        for i in range(8):
            ids=source.sample(i)[:256].unsqueeze(0).cuda()
            x=model.embed_scale*model.embed_tokens(ids)
            final,states=model.self_decoder(x,is_prefilling=True,return_layers=model.memory_fusion.config.source_layers)
            w=model.memory_fusion.routing_weights(final).float()
            weights.append(w.flatten(0,1).cpu())
            norms.append(torch.stack([s.float().square().mean(-1).sqrt().mean() for s in states]).cpu())
    w=torch.cat(weights);count=w.shape[-1]
    report=dict(step=step,checkpoint_sha256=checkpoint_sha,config=c['memory_fusion'],
                scope='CUDA FP16 autocast; validation sequence indices 0..7, first 256 tokens each; not full validation',
                mean_weights=w.mean(0).tolist(),std_weights=w.std(0).tolist(),
                argmax_fraction=torch.nn.functional.one_hot(w.argmax(-1),count).float().mean(0).tolist(),
                mean_entropy_nats=(-(w*w.clamp_min(1e-30).log()).sum(-1)).mean().item(),
                uniform_entropy_nats=math.log(count),fraction_max_above_0_95=(w.max(-1).values>.95).float().mean().item(),
                mean_source_rms=torch.stack(norms).mean(0).tolist())
    atomic_json(run/f'routing_step_{step:08d}.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
