"""Read-only CPU gradient diagnostic; same held-out prefix for each trained model.

This measures gradient magnitudes at intermediate states, not causal proof of
faster optimization. It never takes an optimizer step or rewrites a checkpoint.
"""
import argparse
import json
from pathlib import Path
import sys
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--release',required=True);p.add_argument('--run-dir',required=True)
    a=p.parse_args();release=Path(a.release);run=Path(a.run_dir)
    sys.path.insert(0,str(release))
    from runtime.model import build_model
    from runtime.train import Tokens, sha256, atomic_json
    torch.set_num_threads(2)
    resolved=json.loads((run/'resolved_config.json').read_text());c=resolved['config']
    for name,expected in resolved['identity']['model_source_sha256'].items():
        assert sha256(release/'runtime'/name)==expected,name
    meta=json.loads((run/'latest.json').read_text());path=run/meta['file']
    assert sha256(path)==meta['sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    model=build_model(c).eval();model.load_state_dict(saved['model'],strict=True);del saved
    states={};handles=[]
    def capture(index):
        def hook(module,args,out):
            out.retain_grad();states[index]=out
        return hook
    for i,layer in enumerate(model.self_decoder.layers,1):
        handles.append(layer.register_forward_hook(capture(i)))
    seq=Tokens(resolved['data'],'validation',2048,c['seed']+1).sample(0)[:257].unsqueeze(0)
    logits=model(seq[:,:-1])[0]
    loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),seq[:,1:].flatten())
    assert torch.isfinite(loss)
    loss.backward()
    activation={str(i):dict(gradient_rms=s.grad.float().square().mean().sqrt().item(),
                            gradient_norm=s.grad.float().norm().item(),
                            state_rms=s.detach().float().square().mean().sqrt().item()) for i,s in states.items()}
    parameter={}
    for i,layer in enumerate(model.self_decoder.layers,1):
        total=sum(p.grad.detach().double().square().sum().item() for p in layer.parameters() if p.grad is not None)
        parameter[str(i)]=total**.5
    result=dict(step=meta['step'],checkpoint_sha256=meta['sha256'],loss=loss.item(),
                scope='CPU FP32, validation sequence 0, first 256 next-token targets; no optimizer update',
                activation_gradients=activation,self_layer_parameter_gradient_norms=parameter,
                interpretation='Magnitudes alone do not establish better gradient direction, faster convergence, or a causal mechanism.')
    atomic_json(run/'gradient_flow_probe.json',result)
    print(json.dumps(result,indent=2),flush=True)
    for h in handles:h.remove()


if __name__=='__main__':main()
