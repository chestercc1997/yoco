"""Replay a failed eight-GPU run in isolation and capture non-finite forwards.

Imports the immutable training release and keeps its arithmetic, data order,
optimizer/scaler/RNG restoration, validation and checkpoint cadence unchanged.
Failure snapshots are diagnostic artifacts, not formal training checkpoints.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--stop-after', type=int, default=14311)
    args = parser.parse_args()
    source = args.source_run.resolve()
    target = args.run_dir.resolve()
    assert target != source, 'Replay must never overwrite the formal run'
    target.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source / 'release'))
    import runtime.train as trainer

    manifest = json.loads((source / 'release-manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if name.startswith('runtime/'):
            assert trainer.sha256(source / 'release' / name) == digest
    config = source / 'release/configs/A.json'
    checkpoint = source / 'checkpoint_00010000.pt'
    rank = int(os.environ['RANK'])
    holder = {'step': 10000, 'microbatch': 0}
    original_build, original_loss, original_log = trainer.build_model, trainer.cross_entropy, trainer.log

    def before_forward(module, values):
        holder['tokens'] = values[0]

    def build(*values, **kwargs):
        model = original_build(*values, **kwargs)
        holder['model'] = model
        model.register_forward_pre_hook(before_forward)
        return model

    def capture(name, **extra):
        path = target / name
        temporary = path.with_suffix('.pt.tmp')
        snapshot = dict(model={key: value.detach().cpu() for key, value in holder['model'].state_dict().items()},
                        step=holder['step'], rank=rank, microbatch=holder['microbatch'],
                        config=json.loads(config.read_text()), source_checkpoint=str(checkpoint),
                        time=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
        if 'tokens' in holder:
            snapshot['input_tokens'] = holder['tokens'].detach().cpu()
        torch.save(snapshot, temporary)
        temporary.replace(path)
        trainer.atomic_json(path.with_suffix('.json'), {key: value for key, value in snapshot.items()
                            if key not in {'model', 'input_tokens', 'config'}})

    def loss(logits, targets):
        result = original_loss(logits, targets)
        if not torch.isfinite(result):
            capture(f'failure-rank{rank}.pt', failure_update=holder['step'] + 1,
                    loss=str(float(result.detach())), finite_logits=bool(torch.isfinite(logits).all()))
        holder['microbatch'] += 1
        return result

    def log(run, local_rank, event, **fields):
        original_log(run, local_rank, event, **fields)
        if event in {'started', 'train', 'overflow'}:
            holder['step'] = fields['step']
            holder['microbatch'] = 0
        if rank == 0 and event == 'train' and fields['step'] == 14310:
            capture('before-original-failure.pt', purpose='Exact replay state before update 14311')

    trainer.build_model, trainer.cross_entropy, trainer.log = build, loss, log
    if rank == 0:
        trainer.atomic_json(target / 'diagnostic-replay.json', dict(
            source_run=str(source), source_checkpoint=str(checkpoint), stop_after=args.stop_after,
            no_formal_run_changes=True, precision_unchanged=True, recipe_unchanged=True,
            purpose='Reproduce non-finite forward with frozen release; not a completed formal run'))
    sys.argv = ['runtime.train', '--config', str(config), '--run-dir', str(target),
                '--resume-from', str(checkpoint), '--stop-after', str(args.stop_after)]
    trainer.main()


if __name__ == '__main__':
    main()
