"""Audit already sealed D4 episodes without dispatching a model or simulator."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from go2arx5_vla.world_model.multisource_d4_native import POPULATION_SHA
from go2arx5_vla.world_model.multitask_terminal_audit_v1 import replay
from go2arx5_vla.world_model.storage import digest


def read(path):
    return json.loads(path.read_bytes())


def query_records(directory):
    for folder in sorted(directory.iterdir()):
        result_file = folder/'QUERY_ACTUAL.private.json'
        if not result_file.is_file():
            continue
        actual = read(result_file)
        proposal = actual['action']
        if digest(proposal['path']) != proposal['sha256']:
            raise ValueError('D4_ACTUAL_QUERY_ACTION_BYTES_CHANGED')
        with np.load(proposal['path'], allow_pickle=False) as raw:
            width = .002+.040*raw['physical_H40'][:, 9].astype(np.float64)
        hits = np.flatnonzero(width <= .022)
        predicted = int(hits[0]) if hits.size else None
        intervals = actual.get('confirmed_intervals', [])
        if len(intervals) != actual['confirmed_atoms'] or any(
                row['native_end_step']-row['native_start_step'] != 25 for row in intervals):
            raise ValueError('D4_CONFIRMED_CONSUMPTION_LEDGER_DISAGREES')
        emitted = [r['emitted_gripper_width_m'] for r in intervals]
        measured = [r['measured_gripper_width_m'] for r in intervals]
        yield dict(query=actual['query'], native_start_step=actual['native_start_step'],
            generated_H40_first_below_22mm_k=predicted,
            projected_first_below_22mm_native_tick=(None if predicted is None else actual['native_start_step']+25*predicted),
            model_action_sha256=proposal['sha256'], full_H40_pass=actual['full_H40_compiled'],
            install_ACKs=len(actual['install_ACKs']), confirmed_atoms=len(intervals),
            partial_ticks=actual.get('partial_native_ticks'), rejection=actual.get('rejection'),
            actual_previous_width_m=(actual.get('previous_actual_gripper_command') or {}).get('width_m'),
            emitted_min_width_m=min(emitted, default=None), measured_min_width_m=min(measured, default=None),
            maximum_abs_width_tracking_error_m=max((abs(x-y) for x,y in zip(emitted, measured, strict=True)), default=None))


def case_result(case_file, actual):
    spec = read(case_file)
    directory = case_file.parent/'native-01'
    detail = actual.get('detail') or {}
    cached = detail.get('score')
    if cached is None and (directory/'MULTITASK_SCORE_PROGRESS.private.json').is_file():
        cached = read(directory/'MULTITASK_SCORE_PROGRESS.private.json')
    terminal = replay(spec, directory/'state.bin.journal', actual['actual_native_integrations'], cached)
    queries = list(query_records(directory/'g7-queries')) if (directory/'g7-queries').is_dir() else []
    if len({q['query'] for q in queries}) != len(queries):
        raise ValueError('D4_DUPLICATE_QUERY_NUMBER')
    minimum = lambda key:min((r[key] for r in queries if r[key] is not None), default=None)
    categories = Counter('no_crossing' if q['generated_H40_first_below_22mm_k'] is None
        else 'P10' if q['generated_H40_first_below_22mm_k'] < 10 else 'tail30' for q in queries)
    stops = actual.get('physics_stops') or []
    first_stop = min(stops, key=lambda event: event['steps']) if stops else None
    condition = spec['mobile_condition']
    return dict(case=case_file.parent.name, arm=spec['arm'], instruction=spec['task_instruction_id'],
        layout=condition['layout'], reset=condition['reset'], noise_seed=condition['seed'],
        scene_sha256=spec['frozen_files'][spec['model_path']],
        native_request=actual['physical_requests'], native_ticks=actual['actual_native_integrations'],
        source_sha=actual['source_sha'], native_result_sha256=digest(case_file.parent/'RESULT.private.json'),
        generations=actual['model_generations'], query_rows=len(queries), query_counts=dict(categories),
        confirmed_atoms=sum(r['confirmed_atoms'] for r in queries), emitted_min_width_m=minimum('emitted_min_width_m'),
        measured_min_width_m=minimum('measured_min_width_m'),
        first_failure=first_stop['reason'] if first_stop else detail.get('terminal_reason') or actual.get('failure'),
        first_stop=first_stop, technical_failure=actual['technical_failure'], STOP=actual['unsafe'],
        recorded_controller_mount=detail.get('initial_mount'),
        terminal_full_score=terminal, trace=(actual.get('trace_audit') or {}).get('status'),
        STOP_after_ticks=(actual.get('trace_audit') or {}).get('integrations_after_first_STOP'),
        wall_s=actual['wall_s'], CPU_s=actual['CPU_s'], queries=queries)


def paired_results(records):
    pairs = {}
    for row in records:
        key = row['layout'], row['instruction']
        pairs.setdefault(key, {})[row['arm']] = row

    def max_delta(a, b):
        return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))

    results = []
    for (layout, instruction), entries in pairs.items():
        if set(entries) != {'D4-A','D4-B'}:
            continue
        a, b = (entries[arm] for arm in ('D4-A','D4-B'))
        fields = ('reset', 'noise_seed', 'scene_sha256')
        if any(a[k] != b[k] for k in fields):
            raise ValueError('D4_PAIR_LOST_SEALED_SCENE_RESET_OR_NOISE')
        initial_a, initial_b = (r['terminal_full_score'].get('initial_state') for r in (a,b))
        deltas = None
        if initial_a is not None and initial_b is not None:
            deltas = {k:max_delta(initial_a[k], initial_b[k]) for k in ('qpos','qvel','ctrl','TCP','base')}
            deltas['objects'] = {color:max_delta(initial_a['objects'][color], initial_b['objects'][color])
                                for color in ('red','blue')}
            mounts = a['recorded_controller_mount'], b['recorded_controller_mount']
            deltas['controller_mount'] = max_delta(*mounts) if None not in mounts else None
        results.append(dict(layout=layout,instruction=instruction,initial_max_abs_difference=deltas,
                            stop_A=a['STOP'],stop_B=b['STOP'],
                            science_A=a['terminal_full_score'].get('score'),
                            science_B=b['terminal_full_score'].get('score')))
    return results


def audit(root, *, start=0, limit=12):
    root = Path(root)
    campaign = read(root/'EXECUTION_SPEC.json')
    if campaign['compiled_population']['sha256'] != POPULATION_SHA:
        raise ValueError('ORIGINAL_D4_COMPARISON_REQUIRED')
    records = []
    if not 0 <= start < start+limit <= len(campaign['cases']) or start % 2:
        raise ValueError('D4_AUDIT_REQUIRES_A_REGISTERED_PAIR_ALIGNED_WINDOW')
    for name in campaign['cases'][start:start+limit]:
        case = Path(name)
        result = case.parent/'RESULT.private.json'
        lease = case.parent/'control/LEASE.json'
        case_lease = read(lease)
        if not result.is_file() or case_lease['status'] != 'CLOSED':
            break
        if case_lease['spec_sha256'] != digest(case):
            raise ValueError('D4_ORIGINAL_CLOSED_CASE_SPEC_CHANGED')
        records.append(case_result(case, read(result)))
    return dict(cases=records, paired=paired_results(records), planned_first_wave=12,
                checked_case_window=[start,start+limit], new_generations=0, new_native_ticks=0, new_training_updates=0)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--limit', type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(audit(args.root,start=args.start,limit=args.limit), allow_nan=False))
