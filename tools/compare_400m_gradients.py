"""Compare identical full-update gradients at a saved checkpoint, without AdamW.

The 128 samples are accumulated on one GPU, preserving each backward's original
loss multiplier, then averaged over the original eight ranks. This diagnoses
arithmetic; it is not an eight-rank resumption or a formal training update.
"""
import argparse
import contextlib
import json
import math
from pathlib import Path
import sys
import time

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--update', type=int, default=11499)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    sys.path.insert(0, str(args.run.resolve() / 'release'))
    from runtime.model import build_model
    from runtime.train import Tokens

    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    # Leave room for the ongoing eight-rank diagnostic replay.
    torch.cuda.set_per_process_memory_fraction(0.5, args.device)
    resolved = json.loads((args.run / 'resolved_config.json').read_text())
    config = resolved['config']
    training = config['training']
    saved = torch.load(args.run / 'checkpoint_00010000.pt', map_location='cpu', mmap=True, weights_only=False)
    model = build_model(config, checkpoint_activations=False).cuda().train()
    model.load_state_dict(saved['model'])
    scale = saved['scaler']['scale']
    del saved
    originals = [(layer, layer.forward) for decoder in (model.self_decoder, model.cross_decoder)
                 for layer in decoder.layers]
    tokens = Tokens(resolved['data'], 'train', training['sequence_length'], config['seed'])
    count = training['world_size'] * training['microbatch'] * training['gradient_accumulation']
    first = (args.update - 1) * count
    report = dict(checkpoint_step=10000, update=args.update, global_samples=count,
                  loss_scale=scale, no_optimizer_updates=True, original_precision_unchanged=True,
                  limitation='Earlier saved weights; one-GPU accumulation differs from distributed reduction order.',
                  cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reference = {}
    for case in ('fp32_reference', 'eager_fp16', 'compiled_layers_fp16'):
        started = time.monotonic()
        model.zero_grad(set_to_none=True)
        torch.backends.cuda.matmul.allow_tf32 = case != 'fp32_reference' and config['execution']['allow_tf32']
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        if case == 'compiled_layers_fp16':
            for layer, forward in originals:
                layer.forward = torch.compile(forward, dynamic=False, options={'emulate_precision_casts': True})
        losses = []
        for index in range(first, first + count, training['microbatch']):
            seq = torch.stack([tokens.sample(index + n) for n in range(training['microbatch'])]).cuda()
            mode = (contextlib.nullcontext() if case == 'fp32_reference'
                    else torch.autocast('cuda', dtype=torch.float16))
            with mode:
                logits = model(seq[:, :-1])[0]
                loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), seq[:, 1:].flatten())
            losses.append(float(loss.detach()))
            (loss * (scale / training['gradient_accumulation'])).backward()
            del seq, logits, loss
        nonfinite, norms, differences, dots, ref_norms, largest = [], [], [], [], [], []
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            gradient = (parameter.grad.detach() / (scale * training['world_size'])).cpu()
            if not torch.isfinite(gradient).all():
                nonfinite.append(name)
                continue
            squared = float(gradient.square().sum())
            norms.append(squared)
            largest.append((name, math.sqrt(squared)))
            if case == 'fp32_reference':
                reference[name] = gradient
            else:
                ref = reference[name]
                differences.append(float((gradient - ref).square().sum()))
                dots.append(float((gradient * ref).sum()))
                ref_norms.append(float(ref.square().sum()))
        row = dict(name=case, loss=sum(losses) / len(losses), gradient_norm=math.sqrt(sum(norms)),
                   nonfinite_parameters=nonfinite, largest_gradients=sorted(largest, key=lambda x:x[1], reverse=True)[:12],
                   seconds=time.monotonic() - started)
        if differences and not nonfinite:
            row.update(relative_gradient_l2=math.sqrt(sum(differences) / sum(ref_norms)),
                       gradient_cosine=sum(dots) / math.sqrt(sum(norms) * sum(ref_norms)))
        report['cases'].append(row)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(row), flush=True)
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
