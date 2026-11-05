"""Compare forward/backward numerics without updating any checkpoint weights."""
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from runtime.model import build_model
from runtime.train import Tokens
import runtime.yoco_official.gate_retention as gate_module
import runtime.yoco_official.cross_attention as attention_module
from runtime.yoco_official.rms_norm import RMSNorm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', default='checkpoint_00040000.pt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', choices=['validation', 'crash'], default='validation')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--cases', help='Comma-separated case names; fp32 reference is always included')
    parser.add_argument('--global-update', type=int, help='Recompute the actual 128-sequence update without optimizer steps')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    resolved = json.loads((args.run/'resolved_config.json').read_text())
    config = resolved['config']
    saved = torch.load(args.run/args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    model = build_model(config, checkpoint_activations=True).cuda().train()
    model.load_state_dict(saved['model'])
    split = 'train' if args.global_update else 'validation' if args.batch == 'validation' else 'train'
    tokens = Tokens(resolved['data'], split, 2048, config['seed']+(1 if split=='validation' else 0))
    first = (args.global_update-1)*128 if args.global_update else 0 if split == 'validation' else 21353*128
    count = 128 if args.global_update else 4
    batches = []
    for start in range(first,first+count,4):
        seq = torch.stack([tokens.sample(start+i) for i in range(4)]).cuda()
        batches.append((seq[:,:-1],seq[:,1:]))
    original_retention = gate_module.chunk_gate_retention
    original_head = model.output_projection.forward
    original_self = model.self_decoder.forward
    original_cross = model.cross_decoder.forward
    original_value_projection = model.cross_decoder.v_proj.forward
    original_key_projection = model.cross_decoder.k_proj.forward
    original_attention = attention_module.flash_attn_func
    original_mixers = [layer.mixer.forward for layer in model.cross_decoder.layers]
    original_ffns = [layer.ffn.forward for layer in model.cross_decoder.layers]
    original_norms = [(module,module.forward) for module in model.modules() if isinstance(module,RMSNorm)]

    def double_norm(module):
        def forward(value):
            output = module._norm(value)
            return output*module.weight if module.weight is not None else output
        return forward

    def precise_forward(function):
        def wrapped(*values, **kwargs):
            with torch.autocast('cuda', enabled=False):
                return function(*(v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v for v in values), **kwargs)
        return wrapped

    def retention_fp32(q, k, v, g, *args, **kwargs):
        with torch.autocast('cuda', enabled=False):
            out = original_retention(q.float(), k.float(), v.float(), g.float(), *args, **kwargs)
        return out.to(q.dtype)

    def attention_fp32(q, k, v, *args, **kwargs):
        with torch.autocast('cuda', enabled=False):
            out = original_attention(q.float(), k.float(), v.float(), *args, **kwargs)
        return out.to(v.dtype)

    def promote_value(*values, **kwargs):
        return original_value_projection(*values, **kwargs).float()

    report = dict(checkpoint_step=saved['step'], batch='global_train' if args.global_update else args.batch,
                  global_update=args.global_update, sequence_indices=list(range(first,first+count)),
                  torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(args.device),
                  no_optimizer_updates=True, cases=[])
    reference = {}
    cases = [('fp32',None,1.,False,False,True), ('bf16',torch.bfloat16,1.,False,False,True),
             ('fp16_scale1',torch.float16,1.,False,False,True), ('fp16_scale16',torch.float16,16.,False,False,True),
             ('fp16_scale4',torch.float16,4.,False,False,True), ('fp16_scale32',torch.float16,32.,False,False,True),
             ('fp16_scale128',torch.float16,128.,False,False,True),
             ('fp16_scale1024',torch.float16,1024.,False,False,True),
             ('fp16_retention_fp32',torch.float16,128.,True,False,True),
             ('fp16_exact_reduction',torch.float16,128.,False,False,False),
             ('fp16_sdpa_math',torch.float16,128.,False,True,True),
             ('fp16_no_checkpoint',torch.float16,128.,False,False,True),
             ('fp16_head_fp32_scale1',torch.float16,1.,False,False,True),
             ('fp16_head_fp32_scale128',torch.float16,128.,False,False,True),
             ('fp16_head_fp32_sdpa_math',torch.float16,128.,False,True,True),
             ('fp16_self_fp32',torch.float16,128.,False,False,True),
             ('fp16_cross_fp32',torch.float16,128.,False,False,True),
             ('fp16_attention_fp32',torch.float16,128.,False,False,True),
             ('fp16_shared_kv_fp32',torch.float16,128.,False,False,True),
             ('fp16_shared_v_accum_fp32_scale1',torch.float16,1.,False,False,True),
             ('fp16_shared_v_accum_fp32_scale128',torch.float16,128.,False,False,True),
             ('fp16_shared_v_and_attention_fp32',torch.float16,128.,False,False,True),
             ('fp16_cross_mixers_fp32',torch.float16,128.,False,False,True),
             ('fp16_cross_ffns_fp32',torch.float16,128.,False,False,True),
             ('fp32_tf32',None,1.,False,False,True),
             ('fp16_cross_fp32_head_fp32',torch.float16,128.,False,False,True),
             ('fp16_cross_fp32_head_fp32_retention_fp32',torch.float16,128.,True,False,True),
             ('fp16_sdpa_efficient',torch.float16,128.,False,False,True),
             ('fp16_sdpa_flash',torch.float16,128.,False,False,True),
             ('fp16_sdpa_cudnn',torch.float16,128.,False,False,True)]
    cases.append(('fp64',None,1.,False,False,True))
    if args.cases:
        selected = set(args.cases.split(',')) | {'fp32'}
        assert selected <= {case[0] for case in cases}, selected
        cases = [case for case in cases if case[0] in selected]
    for name, dtype, scale, fp32_retention, sdpa_math, reduced in cases:
        started = time.monotonic()
        model.zero_grad(set_to_none=True)
        model.double() if name == 'fp64' else model.float()
        for norm,original in original_norms:
            norm.forward = double_norm(norm) if name == 'fp64' else original
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = reduced
        torch.backends.cuda.matmul.allow_tf32 = name == 'fp32_tf32'
        gate_module.chunk_gate_retention = retention_fp32 if fp32_retention else original_retention
        model.output_projection.forward = precise_forward(original_head) if 'head_fp32' in name else original_head
        model.self_decoder.forward = precise_forward(original_self) if 'self_fp32' in name else original_self
        model.cross_decoder.forward = precise_forward(original_cross) if 'cross_fp32' in name else original_cross
        model.cross_decoder.v_proj.forward = original_value_projection
        model.cross_decoder.k_proj.forward = original_key_projection
        attention_module.flash_attn_func = original_attention
        if 'shared_kv_fp32' in name:
            model.cross_decoder.v_proj.forward = precise_forward(original_value_projection)
            model.cross_decoder.k_proj.forward = precise_forward(original_key_projection)
        if 'shared_v_' in name:
            model.cross_decoder.v_proj.forward = promote_value
        if name == 'fp16_attention_fp32' or 'shared_v_and_attention' in name:
            attention_module.flash_attn_func = attention_fp32
        for layer, mixer, ffn in zip(model.cross_decoder.layers, original_mixers, original_ffns):
            layer.mixer.forward = precise_forward(mixer) if 'cross_mixers_fp32' in name else mixer
            layer.ffn.forward = precise_forward(ffn) if 'cross_ffns_fp32' in name else ffn
        model.eval() if name == 'fp16_no_checkpoint' else model.train()
        mode = torch.autocast('cuda', dtype=dtype) if dtype else contextlib.nullcontext()
        if sdpa_math or name in ('fp16_sdpa_efficient','fp16_sdpa_flash','fp16_sdpa_cudnn'):
            from torch.nn.attention import sdpa_kernel, SDPBackend
            backend = SDPBackend.EFFICIENT_ATTENTION if name == 'fp16_sdpa_efficient' else SDPBackend.FLASH_ATTENTION if name == 'fp16_sdpa_flash' else SDPBackend.CUDNN_ATTENTION if name == 'fp16_sdpa_cudnn' else SDPBackend.MATH
            attention = sdpa_kernel(backend)
        else:
            attention = contextlib.nullcontext()
        row = dict(name=name, scale=scale)
        try:
            loss_sum = torch.zeros((),device='cuda')
            with attention:
                for x,y in batches:
                    with mode:
                        logits = model(x)[0]
                        loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), y.flatten())
                    loss_sum += loss.detach()/len(batches)
                    multiplier = scale/config['training']['gradient_accumulation'] if args.global_update else scale
                    (loss*multiplier).backward()
                    del logits
            norms, diffs, dots, reference_norms, param_norms = [], [], [], [], []
            bad = []
            for key, param in model.named_parameters():
                if param.grad is None:
                    continue
                divisor = scale*config['training']['world_size'] if args.global_update else scale
                grad = param.grad.detach()/divisor
                if not bool(torch.isfinite(grad).all()):
                    bad.append(key)
                    continue
                norm = grad.square().sum()
                norms.append(norm)
                param_norms.append((key, float(norm.sqrt())))
                if name == 'fp32':
                    reference[key] = grad.clone()
                else:
                    ref = reference[key]
                    diffs.append((grad-ref).square().sum())
                    dots.append((grad*ref).sum())
                    reference_norms.append(ref.square().sum())
            norm_squared = float(torch.stack(norms).sum())
            row.update(loss=float(loss_sum), gradient_norm=math.sqrt(norm_squared), nonfinite_parameters=bad,
                       largest_parameter_gradients=sorted(param_norms,key=lambda item:item[1],reverse=True)[:10])
            if diffs and not bad:
                ref_norm = float(torch.stack(reference_norms).sum())
                row.update(relative_gradient_l2=math.sqrt(float(torch.stack(diffs).sum())/ref_norm),
                           gradient_cosine=float(torch.stack(dots).sum())/math.sqrt(norm_squared*ref_norm))
        except Exception as error:
            row['error'] = repr(error)
        torch.cuda.synchronize()
        row['seconds'] = time.monotonic()-started
        report['cases'].append(row)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    gate_module.chunk_gate_retention = original_retention
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


if __name__ == '__main__':
    main()
