"""Read-only replay of a saved failing microbatch; never update training state."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();sys.path.insert(0,str(args.release.resolve()))
    import torch
    from runtime.model import build_model
    from runtime.precision import configure,autocast
    torch.set_num_threads(2)
    saved=torch.load(args.snapshot,map_location='cpu',weights_only=False,mmap=True)
    assert saved['diagnostic_only']
    configure(saved['config']['training']['precision'])
    execution=saved['config'].get('execution',{})
    if 'allow_tf32' in execution:
        torch.backends.cuda.matmul.allow_tf32=execution['allow_tf32']
        torch.backends.cudnn.allow_tf32=execution['allow_tf32']
    model=build_model(saved['config'],checkpoint_activations=False).cuda().eval()
    model.load_state_dict(saved['model'])
    x=saved['input_tokens'].cuda();targets=saved['targets'].cuda()
    report={'snapshot':str(args.snapshot),'failure_update':saved['failure_update'],
            'training_weights_updated':False,'precision_changes_diagnostic_only':True,'cases':{}}
    def finite(value):
        if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
        if isinstance(value,(tuple,list)):return all(finite(v) for v in value)
        if isinstance(value,dict):return all(finite(v) for v in value.values())
        return True
    for label,precision in (('native',saved['config']['training']['precision']),('fp32_reference','fp32')):
        configure(precision)
        failures=[];hooks=[]
        def hook(name):
            def inspect(module,inputs,outputs):
                if not finite(outputs):failures.append({'module':name,'type':type(module).__name__,'inputs_finite':finite(inputs)})
            return inspect
        for name,module in model.named_modules():
            if not list(module.children()):hooks.append(module.register_forward_hook(hook(name)))
        try:
            with torch.inference_mode(),autocast(precision):
                logits=model(x)[0]
                loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),targets.flatten())
            value=float(loss)
            report['cases'][label]={'loss':value if bool(torch.isfinite(loss)) else str(value),'logits_finite':finite(logits),
                'first_nonfinite_leaf':failures[0] if failures else None,'nonfinite_leaves':failures}
        except Exception as error:report['cases'][label]={'error':str(error),'nonfinite_leaves':failures}
        finally:
            for handle in hooks:handle.remove()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
