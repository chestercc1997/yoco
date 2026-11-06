"""Regression check against native GradScaler overflow semantics; CPU only."""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.train import backoff_loss_scaler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    def exercise(mode):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scaler = torch.amp.GradScaler('cpu', init_scale=8, growth_interval=3)

        def successful_update():
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(parameter.square()).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

        successful_update()
        successful_update()
        assert scaler.state_dict()['_growth_tracker'] == 2
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(parameter.square()).backward()
        if mode != 'remote_overflow':
            parameter.grad.fill_(float('inf'))
        scaler.unscale_(optimizer)
        previous_weight = parameter.detach().clone()
        if mode == 'native':
            scaler.step(optimizer)
            scaler.update()
        elif mode == 'legacy':
            scaler.update(new_scale=scaler.get_scale() / 2)
        else:
            backoff_loss_scaler(scaler)
        assert torch.equal(parameter, previous_weight), 'Overflow must not update weights'
        after_backoff = scaler.state_dict()
        successful_update()
        after_success = scaler.state_dict()
        replacement = torch.amp.GradScaler('cpu')
        replacement.load_state_dict(after_success)
        assert replacement.state_dict() == after_success
        return dict(after_backoff=after_backoff, after_one_success=after_success)

    native = exercise('native')
    fixed = exercise('fixed')
    legacy = exercise('legacy')
    remote = exercise('remote_overflow')
    assert fixed == native == remote
    assert fixed['after_backoff']['_growth_tracker'] == 0
    assert fixed['after_one_success']['scale'] == 4
    assert fixed['after_one_success']['_growth_tracker'] == 1
    assert legacy['after_one_success']['scale'] == 8
    result = dict(passed=True, device='cpu', native=native, fixed=fixed,
                  legacy_bug=legacy, synchronized_remote_overflow=remote,
                  precision_and_scaler_hyperparameters_unchanged=True,
                  checkpoint_state_roundtrip_passed=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
