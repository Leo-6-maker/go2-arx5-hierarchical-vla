"""Read-only final6000 schedule, RNG, S4 and Trainer integrity receipt."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0260/R3')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def seed(step):
    return int.from_bytes(hashlib.sha256(f'PI0260|26003|{step}'.encode()).digest()[:8], 'big') % (2**31)


def audit(arm, job):
    path = ROOT / arm / 'main' / 'training'
    expected = json.loads((ROOT / 'train_visits.json').read_text())['visits']
    assert len(expected) == 6000
    names = ('sample_audit.jsonl', 'step_rng.jsonl', 'S4_audit.jsonl', 'full_loss.jsonl')
    observed = list(zip(*(rows(path / name) for name in names), strict=True))
    assert len(observed) == 6000
    for step, (sample, rng, s4, loss) in enumerate(observed, 1):
        visit = expected[step - 1]
        assert (sample['update'], sample['task'], sample['command'], sample['sequence'], sample['index']) == (
            step, visit['task'], visit['command'], visit['sequence'], visit['index'])
        assert rng == dict(step=step, seed=seed(step), condition_sha256=sample['condition_sha256'], mode_restored=True)
        assert s4['forward_index'] == loss['step'] == step
        assert (s4['runtime_id'], s4['rollout_steps'], s4['rollout_head_evaluations']) == ('S4', 4, 4)
        assert (s4['lambda_position'], s4['lambda_delta']) == (.4, .1)
        assert s4['expert_action_in_condition'] is False and s4['active_action_modified'] is False
        assert all(math.isfinite(loss[key]) for key in ('total', 'rotation', 'speed_margin', 'peak_speed_mps'))
    checkpoint = path / 'checkpoint-6000'
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    assert (state['global_step'], state['max_steps']) == (6000, 6000)
    assert all((checkpoint / name).is_file() for name in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth',
                                                        'model.safetensors.index.json', 'model-00001-of-00002.safetensors',
                                                        'model-00002-of-00002.safetensors'))
    result = dict(schema='PI0260_R3_FINAL6000_READONLY_TRAINING_AUDIT_V1', status='PASS_FINAL6000_TRAINING_INTEGRITY',
                  arm=arm, pbs_job=job, completed_updates=6000, schedule_sha256=sha(ROOT / 'train_visits.json'),
                  audit_sha256={name: sha(path / name) for name in names}, source_sha256=sha(Path(__file__)), checkpoint_step=6000,
                  model_loads=0, model_queries=0, optimizer_updates=0)
    target = ROOT / 'evidence' / arm / 'final6000.training_audit.json'
    assert not target.exists()
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(status=result['status'], arm=arm, sha256=sha(target))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('Base', 'G'), required=True)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    audit(args.arm, args.job)
