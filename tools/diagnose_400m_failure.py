"""Read-only forward checks of a failed update using a verified saved checkpoint.

No optimizer steps, checkpoint rewrites, or changes to the frozen release.
A check at an earlier checkpoint cannot reproduce the missing failure weights.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--checkpoint', default='checkpoint_00010000.pt')
    p.add_argument('--update', type=int, default=14311)
    p.add_argument('--rank', type=int, default=3)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    release = args.run / 'release'
    sys.path.insert(0, str(release))
    from runtime.model import build_model
    from runtime.train import Tokens

    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    resolved = json.loads((args.run / 'resolved_config.json').read_text())
    config = resolved['config']
    training = config['training']
    torch.backends.cuda.matmul.allow_tf32 = config['execution']['allow_tf32']
    torch.backends.cudnn.allow_tf32 = config['execution']['allow_tf32']
    checkpoint = args.run / args.checkpoint
    meta = json.loads(checkpoint.with_suffix('.json').read_text())
    with checkpoint.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    assert digest == meta['sha256'], 'Checkpoint checksum mismatch'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    report = dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  checkpoint=str(checkpoint), checkpoint_sha256=digest,
                  checkpoint_step=saved['step'], update=args.update, rank=args.rank,
                  no_optimizer_updates=True,
                  limitation='Uses the saved checkpoint, not unavailable step-14310 weights.',
                  cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def write():
        temporary = args.output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(args.output)

    bad_model = [name for name, value in saved['model'].items()
                 if value.is_floating_point() and not torch.isfinite(value).all()]
    bad_optimizer = [f'{name}:{key}' for name, state in saved['optimizer']['state'].items()
                     for key, value in state.items()
                     if torch.is_tensor(value) and not torch.isfinite(value).all()]
    report.update(nonfinite_model_tensors=bad_model, nonfinite_optimizer_tensors=bad_optimizer,
                  scaler=saved['scaler'], fusion_logits=saved['model']['memory_fusion.logits'].tolist())
    assert not bad_model and not bad_optimizer, 'Non-finite checkpoint state'
    tokens = Tokens(resolved['data'], 'train', training['sequence_length'], config['seed'])
    batches = []
    batch_records = []
    for j in range(training['gradient_accumulation']):
        index = (((args.update - 1) * training['gradient_accumulation'] + j)
                 * training['world_size'] + args.rank) * training['microbatch']
        seq = torch.stack([tokens.sample(index + n) for n in range(training['microbatch'])])
        record = dict(microbatch=j, first_sequence_index=index, minimum_token=int(seq.min()),
                      maximum_token=int(seq.max()),
                      token_sha256=hashlib.sha256(seq.numpy().tobytes()).hexdigest(),
                      unique_tokens=[int(row.unique().numel()) for row in seq])
        assert record['minimum_token'] >= 0 and record['maximum_token'] < config['model']['vocab_size']
        batch_records.append(record)
        batches.append(seq.cuda())
    report['batches'] = batch_records
    model = build_model(config, checkpoint_activations=False).cuda().train()
    model.load_state_dict(saved['model'])
    del saved
    originals = [(layer, layer.forward) for decoder in (model.self_decoder, model.cross_decoder)
                 for layer in decoder.layers]
    for case in ('eager', 'compiled_layers'):
        if case == 'compiled_layers':
            for layer, forward in originals:
                layer.forward = torch.compile(forward, dynamic=False,
                                              options={'emulate_precision_casts': True})
        case_record = dict(name=case, microbatches=[])
        report['cases'].append(case_record)
        for j, seq in enumerate(batches):
            started = time.monotonic()
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(seq[:, :-1])[0]
                loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), seq[:, 1:].flatten())
            record = dict(microbatch=j, loss=float(loss.detach()),
                          finite_logits=bool(torch.isfinite(logits).all()),
                          logits_maximum_absolute=float(logits.detach().abs().max()))
            del logits, loss
            torch.cuda.synchronize()
            record['seconds'] = time.monotonic() - started
            case_record['microbatches'].append(record)
            write()
            print(json.dumps(dict(case=case, **record)), flush=True)
    report['complete'] = True
    write()


if __name__ == '__main__':
    main()
