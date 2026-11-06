"""Locate non-finite module outputs in a captured failure, without weight updates."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_run.resolve() / 'release'))
    from runtime.model import build_model
    from runtime.train import Tokens
    import runtime.yoco_official.gate_retention as gate_module

    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    captured = torch.load(args.snapshot, map_location='cpu', weights_only=False, mmap=True)
    config = captured['config']
    torch.backends.cuda.matmul.allow_tf32 = config['execution']['allow_tf32']
    torch.backends.cudnn.allow_tf32 = config['execution']['allow_tf32']
    resolved = json.loads((args.source_run / 'resolved_config.json').read_text())
    training = config['training']
    update = captured.get('failure_update', 14311)
    microbatch = captured['microbatch'] if 'failure_update' in captured else 0
    rank = captured['rank'] if 'failure_update' in captured else 3
    tokens = Tokens(resolved['data'], 'train', training['sequence_length'], config['seed'])
    index = (((update - 1) * training['gradient_accumulation'] + microbatch)
             * training['world_size'] + rank) * training['microbatch']
    sequence = torch.stack([tokens.sample(index + n) for n in range(training['microbatch'])])
    if 'failure_update' in captured:
        assert torch.equal(sequence[:, :-1], captured['input_tokens']), 'Captured input differs from sampler'
    report = dict(snapshot=str(args.snapshot), saved_step=captured['step'], update=update,
                  rank=rank, microbatch=microbatch, no_optimizer_updates=True,
                  token_sha256=hashlib.sha256(sequence.numpy().tobytes()).hexdigest(), cases=[])
    report['nonfinite_weight_tensors'] = [name for name, value in captured['model'].items()
                                         if not torch.isfinite(value).all()]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def write():
        temp = args.output.with_suffix('.json.tmp')
        temp.write_text(json.dumps(report, indent=2) + '\n')
        temp.replace(args.output)

    def stats(value):
        value = value.detach()
        finite = torch.isfinite(value)
        result = dict(dtype=str(value.dtype), shape=list(value.shape),
                      nonfinite=int((~finite).sum()))
        if finite.any():
            result['maximum_finite_absolute'] = float(value[finite].abs().max())
        return result

    seq = sequence.cuda()
    original_retention = gate_module.chunk_gate_retention
    cases = ('eager_fp16', 'compiled_layers_fp16', 'eager_fp16_full_reduction',
             'compiled_layers_fp16_full_reduction', 'eager_fp32_tf32_reference')
    for case in cases:
        compiled_case = case.startswith('compiled_layers')
        # A read-only candidate: keep FP16 inputs/output but prevent truncated
        # intermediate GEMM accumulation. Formal execution remains untouched.
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = 'full_reduction' not in case
        model = build_model(config, checkpoint_activations=False).cuda().train()
        model.load_state_dict(captured['model'])
        case_record = dict(name=case)
        report['cases'].append(case_record)
        handles = []
        checked = []
        gate_module.chunk_gate_retention = original_retention

        def hook(name):
            def record(module, inputs, output):
                tensors = [output] if torch.is_tensor(output) else []
                if tensors:
                    item = dict(module=name, **stats(tensors[0]))
                    checked.append(item)
            return record

        if not compiled_case:
            def retention(*values, **kwargs):
                output = original_retention(*values, **kwargs)
                if not torch.isfinite(output).all():
                    contexts = case_record.setdefault('nonfinite_retention_contexts', [])
                    path = args.output.parent / f'retention-{case}-{len(contexts)}.pt'
                    torch.save(dict(inputs=[value.detach().cpu() for value in values],
                                    kwargs=kwargs, output=output.detach().cpu()), path)
                    contexts.append(str(path))
                return output

            gate_module.chunk_gate_retention = retention
            # The first non-finite leaf output narrows the failing operator.
            # Normalization inputs also expose failures in functional retention.
            def before_norm(name):
                def record(module, inputs):
                    checked.append(dict(module=name + ':input', **stats(inputs[0])))
                return record

            for name, module in model.named_modules():
                if not list(module.children()):
                    handles.append(module.register_forward_hook(hook(name)))
                    if name.endswith('subln'):
                        handles.append(module.register_forward_pre_hook(before_norm(name)))
        else:
            for decoder_name in ('self_decoder', 'cross_decoder'):
                for i, layer in enumerate(getattr(model, decoder_name).layers):
                    layer.forward = torch.compile(layer.forward, dynamic=False,
                                                  options={'emulate_precision_casts': True})
                    # Outside the compiled forward; leave its internal graph alone.
                    handles.append(layer.register_forward_hook(hook(f'{decoder_name}.layers.{i}')))
        mode = (torch.autocast('cuda', dtype=torch.float16) if case != 'eager_fp32_tf32_reference'
                else contextlib.nullcontext())
        try:
            with mode:
                logits = model(seq[:, :-1])[0]
                loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), seq[:, 1:].flatten())
            case_record.update(loss=str(float(loss.detach())), logits=stats(logits))
            case_record['module_outputs'] = checked
            case_record['first_nonfinite'] = next((row for row in checked if row['nonfinite']), None)
            del loss, logits
        except Exception as error:
            case_record['error'] = repr(error)
        for handle in handles:
            handle.remove()
        del model
        torch.cuda.empty_cache()
        write()
        print(json.dumps({key: value for key, value in case_record.items() if key != 'module_outputs'}), flush=True)
    report['complete'] = True
    gate_module.chunk_gate_retention = original_retention
    write()


if __name__ == '__main__':
    main()
