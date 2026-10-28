"""Read-only paired-input, packet, budget and endpoint audit for both PI0261 arms."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0261')


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    plan = read(ROOT / 'PAIRING_PLAN.json')
    bank = {(row['task'], row['family']): row for row in plan['common_start_bank']}
    processed = read(ROOT / 'PROCESSED_PAIR_BANK_AUDIT.json')
    actual_bank = {(row['task'], row['family']): row for row in processed['rows']}
    assert len(plan['units']) == 2000 and len(bank) == len(actual_bank) == 60
    samples, packets = {}, {}
    for arm in ('C', 'P'):
        output = ROOT / arm / 'main/training'
        sample = rows(output / 'sample_audit.jsonl')
        rng = rows(output / 'step_rng.jsonl')
        s4 = rows(output / 'S4_audit.jsonl')
        loss = rows(output / 'full_loss.jsonl')
        assert len(sample) == len(rng) == len(s4) == len(loss) == 4000
        for micro_global, (x, y, z, w) in enumerate(zip(sample, rng, s4, loss), 1):
            unit = plan['units'][(micro_global - 1) // 2]
            micro = (micro_global - 1) % 2
            command = unit['order'][micro]
            selected = bank[unit['task'], unit['family']]['commands'] if arm == 'P' and unit['slot'] == 'PAIR_SLOT' else unit['ordinary']
            source = selected[command]
            assert (x['unit'], x['micro'], x['task'], x['family'], x['slot'], x['command'],
                    x['sequence'], x['index']) == (unit['unit'], micro, unit['task'], unit['family'],
                                                   unit['slot'], command, source['sequence'], source['index'])
            assert (y['unit'], y['micro'], y['condition_sha256'], y['command']) == (
                unit['unit'], micro, x['condition_sha256'], command)
            assert y['mode_restored'] and z['forward_index'] == w['step'] == micro_global
            assert y['native_epsilon'] == z['native_epsilon'] and y['native_tau'] == z['native_tau']
            assert y['stock_initial_noise'] == z['stock_initial_noise']
            assert (z['runtime_id'], z['rollout_steps'], z['rollout_head_evaluations']) == ('S4', 4, 4)
            assert z['expert_action_in_condition'] is False and z['active_action_modified'] is False
            assert (z['lambda_position'], z['lambda_delta']) == (.4, .1)
            assert all(math.isfinite(w[name]) for name in ('total', 'rotation', 'speed_margin'))
            if arm == 'P' and unit['slot'] == 'PAIR_SLOT':
                assert x['condition_sha256'] == actual_bank[unit['task'], unit['family']]['condition_digest'][command]
            if micro == 1:
                assert rng[micro_global - 2]['native_epsilon'] == y['native_epsilon']
                assert rng[micro_global - 2]['stock_initial_noise'] == y['stock_initial_noise']
                assert rng[micro_global - 2]['native_tau'] == y['native_tau']
        state = read(output / 'checkpoint-2000/trainer_state.json')
        assert (state['global_step'], state['max_steps']) == (2000, 2000)
        assert all((output / 'checkpoint-2000' / name).is_file() for name in
                   ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'model.safetensors.index.json'))
        assert read(ROOT / arm / 'canary_compare.json')['status'] == 'PASS_PI0261_REAL_TWO_MICROBATCH_EXACT_RESUME_CANARY'
        samples[arm], packets[arm] = sample, rng
    replay_count = 0
    for index, unit in enumerate(plan['units']):
        left, right = samples['C'][2 * index:2 * index + 2], samples['P'][2 * index:2 * index + 2]
        for a, b in zip(left, right):
            if unit['slot'] == 'REPLAY_SLOT':
                assert (a['sequence'], a['index'], a['condition_sha256']) == (b['sequence'], b['index'], b['condition_sha256'])
                replay_count += 1
        for micro in (0, 1):
            a, b = packets['C'][2 * index + micro], packets['P'][2 * index + micro]
            assert (a['seed'], a['native_epsilon'], a['stock_initial_noise'], a['native_tau']) == (
                b['seed'], b['native_epsilon'], b['stock_initial_noise'], b['native_tau'])
    assert replay_count == 2000
    result = dict(schema='PI0261_PAIRED_MAIN_TRAINING_AUDIT_V1', status='PASS_BOTH_2000_PAIRED_TRAINING_INTEGRITY',
                  arms=['C', 'P'], optimizer_updates={'C': 2000, 'P': 2000},
                  supervised_microbatches={'C': 4000, 'P': 4000},
                  replay_cross_arm_exact_examples=replay_count, pair_slot_processed_nolanguage_equal=60,
                  actual_random_packets_matched_within_unit_and_across_arms=True,
                  plan_sha256=sha(ROOT / 'PAIRING_PLAN.json'), selected_checkpoint='NONE',
                  deployment_selected_checkpoint='NONE', model_queries=0, optimizer_updates_by_audit=0)
    target = ROOT / 'MAIN_TRAINING_AUDIT.json'
    assert not target.exists()
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    run()
