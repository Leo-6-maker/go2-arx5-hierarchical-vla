"""Audit already sealed D4 episodes without dispatching a model or simulator."""
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from go2arx5_vla.world_model.multisource_d4_native import POPULATION_SHA
from go2arx5_vla.world_model.multitask_terminal_audit_v1 import replay
from go2arx5_vla.world_model.storage import digest


def read(path):
    return json.loads(path.read_bytes())


def measured_query_times(directory):
    started, previous_end = None, None
    values, unmeasured = [], 0
    for folder in sorted(directory.iterdir()):
        artifacts = [folder/name for name in ('INPUT_META.private.json','MODEL_REQUEST.private.json','MODEL_RETURN.private.json')]
        if not all(path.is_file() for path in artifacts):
            unmeasured += 1
            continue
        meta, request, returned = map(read, artifacts)
        response = returned['response']
        clock = (meta['observation_available_monotonic'],meta['input_ready_monotonic'],
                 request['created_monotonic'],returned['wall_end_monotonic'])
        if len(response.get('branch_wall_s', ())) != 1 or 'worker_wall_s' not in response:
            unmeasured += 1
            continue
        provider = response['branch_wall_s'][0]
        if (not np.isfinite([*clock,provider,response['worker_wall_s']]).all()
                or tuple(sorted(clock)) != clock or request['request_id'] != response['request_id']
                or (previous_end is not None and previous_end > clock[0])):
            raise ValueError('ORIGINAL_QUERY_TIMING_LEDGER_NOT_CAUSAL')
        if started is None:
            started = clock[0]
        previous_end = clock[-1]
        values.append((clock[1]-clock[0],clock[2]-clock[1],clock[3]-clock[2],provider,response['worker_wall_s']))
    fields = ('observation_to_RGB_input','input_ready_to_request','request_to_reply',
              'provider_inside_worker','worker_inside_request_to_reply')
    summary = {}
    for label, bins in zip(fields,zip(*values),strict=False):
        summary[label] = dict(total_wall_s=float(sum(bins)),p50_s=float(np.quantile(bins,.5)),
                              p95_s=float(np.quantile(bins,.95)),first_s=float(bins[0]))
    return dict(observed_model_replies=len(values),queries_without_complete_timing=unmeasured,timing_s=summary,
                first_observation_to_last_reply_wall_s=previous_end-started if values else None,
                worker_and_provider_are_nested=True,compiler_and_storage_not_independently_timed=True)


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
        wall_s=actual['wall_s'], CPU_s=actual['CPU_s'],
        timing=measured_query_times(directory/'g7-queries') if (directory/'g7-queries').is_dir() else None,
        queries=queries)


def paired_results(records):
    pairs = {}
    for row in records:
        key = row['layout'], row['instruction']
        pairs.setdefault(key, {})[row['arm']] = row

    def max_delta(a, b):
        return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))

    def pose_delta(a,b):
        a,b = (np.asarray(x) for x in (a,b))
        delta = a[:3,:3].T@b[:3,:3]
        radians = float(np.arccos(np.clip((np.trace(delta)-1)/2,-1,1)))
        return dict(translation_m=float(np.linalg.norm(a[:3,3]-b[:3,3])),rotation_deg=float(np.degrees(radians)))

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
            positions = dict(TCP=pose_delta(initial_a['TCP'],initial_b['TCP']),
                             x5_mount_from_state=pose_delta(initial_a['base'],initial_b['base']))
            for color in ('red','blue'):
                positions[color] = pose_delta(initial_a['objects'][color],initial_b['objects'][color])
            if None not in mounts:
                positions['controller_mount'] = pose_delta(*mounts)
            qat, qbt = (np.asarray(state['qpos']) for state in (initial_a,initial_b))
            qa, qb = qat[3:7], qbt[3:7]
            if np.isclose(np.linalg.norm(qa),1.) and np.isclose(np.linalg.norm(qb),1.):
                angle = 2*np.arccos(np.clip(abs(qa@qb),0,1))
                positions['Go2_root_free_joint'] = dict(translation_m=float(np.linalg.norm(qat[:3]-qbt[:3])),
                                                       rotation_deg=float(np.degrees(angle)))
        else:
            positions = None
        results.append(dict(layout=layout,instruction=instruction,initial_max_abs_difference=deltas,
                            initial_pose_difference=positions,
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


def watch_first_wave(root, spec_sha, result, deadline):
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import unit_state
    from scripts.pi0292.durable import write

    spec_file = root/'EXECUTION_SPEC.json'
    if digest(spec_file) != spec_sha or result.exists():
        raise ValueError('D4_EXACT_EXISTING_CAMPAIGN_AND_NEW_REPORT_REQUIRED')
    campaign = read(spec_file)
    child = Path(campaign['cases'][11]).parent
    until = min(deadline,campaign['deadline_epoch']-120)
    status = 'AUDIT_TIME_BOUNDARY'
    while time.time() < until and not (result.parent/'STOP.json').exists():
        if (child/'RESULT.private.json').is_file() and read(child/'control/LEASE.json')['status'] == 'CLOSED':
            status = 'OBSERVED_FIRST_TWELVE_CLOSED'
            break
        unit = unit_state('wm02-'+root.name)
        lease = read(root/'control/LEASE.json')
        if lease['spec_sha256'] != spec_sha:
            raise ValueError('D4_OBSERVED_OWNER_LEASE_CHANGED')
        if lease['status'] == 'CLOSED' or not unit['pid']:
            status = 'ORIGINAL_CAMPAIGN_STOPPED_BEFORE_FIRST_WAVE'
            break
        write(result.parent/'WATCH_PROGRESS.private.json',dict(epoch=time.time(),phase='WAIT_FIRST_WAVE',
              watched_MainPID=unit['pid'],spec_sha256=spec_sha,native_requests=0))
        time.sleep(25)
    actual = audit(root)
    write(result,dict(actual,phase=status,source_spec_sha256=spec_sha,observed_epoch=time.time()))
    return actual


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--limit', type=int, default=12)
    parser.add_argument('--watch-result',type=Path)
    parser.add_argument('--spec-sha')
    parser.add_argument('--deadline',type=float)
    args = parser.parse_args()
    if args.watch_result:
        if not args.spec_sha or not args.deadline:
            parser.error('watch mode requires --spec-sha and --deadline')
        watch_first_wave(args.root,args.spec_sha,args.watch_result,args.deadline)
    else:
        print(json.dumps(audit(args.root,start=args.start,limit=args.limit), allow_nan=False))
