"""Check predeclared, previously consumed controls in isolated IK clones.

This adds convergence evidence for sixteen accepted TRAIN queries. It reuses
the original rejected16 results, never changes a formal certificate and never
loads an SLA model or integrates a physical world.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np

from scripts.pi0292.eval_contract import matrix10
from scripts.pi0294 import task_path_model as original
from scripts.pi0294.task_state import ActualState, parse_state
from scripts.sla.diagnose_ik_precision_v71 import CONFIGS, ORIGINAL_SHA, evaluate
from go2arx5_vla.world_model.native_replay import forbid_integration


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(payload, output):
    import mujoco
    assert mujoco.__version__ == '3.2.7' and mujoco.get_mjcb_control() is None
    assert sha(inspect.getsourcefile(original.certify_chunk)) == ORIGINAL_SHA
    selection_path = Path(payload['selection']['path'])
    assert sha(selection_path) == payload['selection']['sha256']
    selection = read(selection_path)
    assert selection['settings'] == [list(v) for v in CONFIGS]
    assert selection['original_certifier_sha256'] == ORIGINAL_SHA
    assert len(selection['rows']) == len(payload['records']) == 16
    assert len({(r['native_run'], r['query']) for r in selection['rows']}) == 16
    monitor = Path(payload['monitor'])
    assert sha(monitor / 'RESULT.private.json') == selection['source_monitor_result_sha256']
    terminal = read(monitor / 'RESULT.private.json')
    plan = read(monitor / 'EXECUTION_SPEC.json')
    outcomes = {r['native_run']: r for r in terminal['panels']['TRAIN8']['outcomes']}
    groups = sorted(Path(plan['evaluation']['panels']['TRAIN8']).glob('*-group-*'))
    output = Path(output)
    output.mkdir(exist_ok=False)
    started, cpu = time.time(), time.process_time()
    models, rows = {}, []
    for i, (chosen, saved) in enumerate(zip(selection['rows'], payload['records'], strict=True)):
        assert chosen['native_run'] == saved['native_run'] and chosen['query'] == saved['query']
        assert saved['population'] == 'TRAIN8'
        assert chosen['original_classification'] == saved['classification'] == 'ACCEPTED_AND_CONSUMED'
        assert chosen['confirmed_commands'] == saved['confirmed_commands'] == 10
        roots = [g / 'runs' / chosen['native_run'] for g in groups
                 if (g / 'runs' / chosen['native_run']).exists()]
        assert len(roots) == 1
        root = roots[0]; native = root / 'native-01'
        for actor in (root, root.with_name(root.name + '-gpu')):
            assert read(actor / 'control/LEASE.json')['status'] == 'CLOSED'
        assert sha(native / 'RESULT.json') == outcomes[root.name]['result_sha256']
        spec = read(root / 'EXECUTION_SPEC.json')
        scene = Path(spec['model_path'])
        assert sha(scene) == read(native / 'RESULT.json')['scene_sha256']
        if str(scene) not in models:
            models[str(scene)] = mujoco.MjModel.from_binary_path(str(scene))
        model = models[str(scene)]
        capture = read(native / 'TASK_LEARNING_CAPTURES.private.json')['records'][chosen['query']]
        state = Path(capture['path']).parent / 'STATE.private.bin'
        assert sha(state) == chosen['state_sha256'] == capture['state_sha256']
        with forbid_integration(mujoco) as calls:
            sample = ActualState(model).update(parse_state(state.read_bytes()))
        assert not calls and sample['native_step'] == chosen['native_step']
        decision_path = native / f"TASK_LEARNING_DECISION_{chosen['query']:03d}.private.json"
        assert sha(decision_path) == chosen['decision_sha256']
        decision = read(decision_path)
        assert decision['index'] == chosen['query'] and decision['action_sha256'] == chosen['action_sha256']
        assert decision['execution_stage'] == saved['execution_stage']
        stage = {'CLOSE': 'close', 'LIFT': 'lift', 'HOLD': 'hold'}.get(decision['execution_stage'], 'approach')
        commands = read(native / 'TASK_LEARNING_CONFIRMED_COMMANDS.private.json')['records']
        assert sum(r['student_decision_sha256'] == chosen['decision_sha256'] for r in commands) == 10
        past = list(read(native / 'STABLE_TASK_CYCLES/-1/RESULT.private.json')['windows'])
        past += [dict(r, actual_installed_consumed_EEF10_BASE7=r['installed_consumed_EEF10']) for r in commands]
        past = [r for r in past if r['native_end_step'] <= sample['native_step']][-2:]
        assert [r['native_end_step'] for r in past] == [sample['native_step'] - 25, sample['native_step']]
        history = [dict(end_native_step=r['native_end_step'],
                        pose=matrix10(r['actual_installed_consumed_EEF10_BASE7'][:10])) for r in past]
        assert sha(native / 'COMMANDS.private.jsonl') == chosen['command_journal_sha256']
        events = [json.loads(line) for line in (native / 'COMMANDS.private.jsonl').read_text().splitlines()]
        saved_event = saved['original_native_certificate_event']
        matching = [r for r in events if r.get('native_step') == chosen['native_step']
                    and r.get('phase') == 'proposed' and r.get('wire_sha256') == saved_event['wire_sha256']]
        assert len(matching) == 1 and matching[0] == saved_event
        installed = np.asarray(saved_event['action_EEF10'])
        configs = []
        for j, settings in enumerate(CONFIGS):
            intent = dict(control=i, query=chosen['query'], settings=list(settings), state_sha256=sha(state))
            with (output / 'CLONE_CALLS.private.jsonl').open('a') as f:
                f.write(json.dumps(dict(intent, event='INTENT')) + '\n')
            value = evaluate(model, sample, installed, history, stage, settings,
                             output / f'{i:02d}-precision{j}.npz')
            configs.append(value)
            with (output / 'CLONE_CALLS.private.jsonl').open('a') as f:
                f.write(json.dumps(dict(intent, event='COMPLETED', result=value['result'])) + '\n')
            if j == 0:
                baseline = saved_event['native_reference_certificate']
                assert value['result'] == 'DIAGNOSTIC_PASS_ONLY'
                assert value['certificate']['points'] == baseline['points']
                for key in ('max_joint_speed_rad_s', 'max_joint_acceleration_rad_s2'):
                    assert np.isclose(value['certificate'][key], baseline[key], rtol=0, atol=1e-12)
        row = dict(**chosen, settings=configs, original_certificate_unchanged=True,
                   full_WBC_state_restored=False)
        rows.append(row)
        with (output / 'ROWS.private.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
    result = dict(schema='SLA_PREDECLARED_ACCEPTED16_IK_CONVERGENCE_V71', paths=len(rows),
        clone_certifier_calls=48, rows=rows, selection_sha256=payload['selection']['sha256'],
        original_certifier_sha256=ORIGINAL_SHA, original16_rejected_reused_not_rerun=True,
        new_model_generations=0, actual_SLA_model_loads=0, optimizer_updates=0, training_visits=0,
        physical_requests=0, physics_integrations=0, full_WBC_state_restored=False,
        formal_results_or_thresholds_changed=False, wall_s=time.time()-started, CPU_s=time.process_time()-cpu)
    (output / 'RESULT.private.json').write_text(json.dumps(result, allow_nan=False))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('payload', type=Path); p.add_argument('output', type=Path)
    a = p.parse_args(); result = run(read(a.payload), a.output)
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, allow_nan=False))
