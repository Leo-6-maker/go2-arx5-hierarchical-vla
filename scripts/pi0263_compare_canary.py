"""Compare complete continuous and restored PI0263 eight-unit canaries."""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0263')
R3_CODE = Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0260/R3/execution')
sys.path.insert(0, str(R3_CODE))
from pi0260_r3_compare_canary import model_delta, optimizer_delta


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def compare(arm):
    a = ROOT / arm / 'continuous/training'
    b = ROOT / arm / 'split/training'
    x, y = a / 'checkpoint-8', b / 'checkpoint-8'
    for path in (x, y):
        state = json.loads((path / 'trainer_state.json').read_text())
        assert (state['global_step'], state['max_steps']) == (8, 2000)
        assert all((path / name).is_file() for name in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth'))
    for name, fields in [('sample_audit.jsonl', ('unit', 'micro', 'task', 'family', 'slot', 'command', 'sequence', 'index', 'condition_sha256', 'original_unit', 'visit', 'color', 'template', 'language', 'nonlanguage_sha256', 'labels_masks_sha256')),
                         ('step_rng.jsonl', ('unit', 'micro', 'seed', 'condition_sha256', 'native_epsilon', 'native_tau', 'stock_initial_noise', 'mode_restored')),
                         ('S4_audit.jsonl', ('forward_index', 'native_epsilon_sha256', 'native_tau', 'stock_initial_noise')),
                         ('full_loss.jsonl', ('step',))]:
        left, right = rows(a / name), rows(b / name)
        assert len(left) == len(right) == 16, (name, len(left), len(right))
        for index, (u, v) in enumerate(zip(left, right), 1):
            assert all(u[key] == v[key] for key in fields), (name, index)
    for path in (a, b):
        assert json.loads((path / 'S4_parity.json').read_text())['status'] == 'PASS_PI0247_U_S4_PARITY'
        gradients = json.loads((path / 'S4_first_update_gradients.json').read_text())
        assert set(gradients) == {'1', '2', '3', '4'} and all(float(v) > 0 for v in gradients.values())
    model, model_tensors = model_delta(x, y)
    optimizer, optimizer_tensors = optimizer_delta(x, y)
    left = torch.load(x / 'scheduler.pt', map_location='cpu', weights_only=False)
    right = torch.load(y / 'scheduler.pt', map_location='cpu', weights_only=False)
    assert left['last_epoch'] == right['last_epoch'] == 8
    lr = max(abs(float(u) - float(v)) for u, v in zip(left['_last_lr'], right['_last_lr']))
    assert model <= 1e-6 and optimizer <= 1e-6 and lr <= 1e-12, (model, optimizer, lr)
    result = dict(status='PASS_PI0263_REAL_TWO_MICROBATCH_EXACT_RESUME_CANARY', arm=arm,
                  units=8, microbatches=16, model_max_abs=model, model_tensors=model_tensors,
                  optimizer_max_abs=optimizer, optimizer_tensors=optimizer_tensors,
                  scheduler_lr_abs=lr, packet_equal=True, mode_restored=True)
    target = ROOT / arm / 'canary_compare.json'
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('F', 'X'), required=True)
    compare(parser.parse_args().arm)
