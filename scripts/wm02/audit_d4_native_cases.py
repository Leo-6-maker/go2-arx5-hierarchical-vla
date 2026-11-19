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


def audit(root, *, start=0, limit=12, preserve_errors=False):
    root = Path(root)
    campaign = read(root/'EXECUTION_SPEC.json')
    if campaign['compiled_population']['sha256'] != POPULATION_SHA:
        raise ValueError('ORIGINAL_D4_COMPARISON_REQUIRED')
    records = []
    errors = []
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
        archived = read(result)
        try:
            records.append(case_result(case, archived))
        except (KeyError, ValueError, OSError) as error:
            if not preserve_errors:
                raise
            errors.append(dict(case=case.parent.name,native_result_sha256=digest(result),
                native_requests=archived['physical_requests'],native_ticks=archived['actual_native_integrations'],
                generations=archived['model_generations'],audit_error=str(error)))
    return dict(cases=records, audit_errors=errors, paired=paired_results(records), planned_first_wave=12,
                checked_case_window=[start,start+limit], new_generations=0, new_native_ticks=0, new_training_updates=0)


def summarize_layouts(spec_file, spec_sha, artifacts):
    """Combine sealed evidence using the original case order and count raw effects."""
    if digest(spec_file) != spec_sha:
        raise ValueError('D4_SUMMARY_REQUIRES_THE_ORIGINAL_SPEC')
    spec = read(spec_file)
    if spec['compiled_population']['sha256'] != POPULATION_SHA or len(spec['cases']) != 48:
        raise ValueError('D4_SUMMARY_REQUIRES_ORIGINAL_48_CASE_POPULATION')

    planned = [Path(file).parent.name for file in spec['cases']]
    locked_cases, locked_errors, pair_counts, layouts = [], [], Counter(), []
    next_case = 0
    for artifact in artifacts:
        evidence = read(artifact)
        begin, end = evidence['checked_case_window']
        if (evidence['source_spec_sha256'] != spec_sha or begin != next_case
                or begin % 12 or not begin < end <= begin+12 <= len(planned)
                or any(evidence[key] for key in ('new_generations','new_native_ticks','new_training_updates'))):
            raise ValueError('D4_SUMMARY_REQUIRES_CONTIGUOUS_ORIGINAL_READ_ONLY_LAYOUTS')
        cases, errors = evidence['cases'], evidence.get('audit_errors', [])
        entries = [row['case'] for row in (*cases,*errors)]
        next_case = begin+len(entries)
        if (next_case > end or len(set(entries)) != len(entries)
                or set(entries) != set(planned[begin:next_case])
                or len({row['layout'] for row in cases}) > 1):
            raise ValueError('D4_SUMMARY_LAYOUT_EVIDENCE_DOES_NOT_MATCH_REGISTERED_CLOSED_PREFIX')

        matrix = Counter(both=0,A_only=0,B_only=0,neither=0,unscored=0)
        pairs = paired_results(cases)
        for pair in pairs:
            a,b = pair['science_A'],pair['science_B']
            key = 'unscored'
            if a and b:
                key = ('neither','A_only','B_only','both')[int(a['task_success'])+2*int(b['task_success'])]
            matrix[key] += 1
        pair_counts.update(matrix)
        layouts.append(dict(observed_case_window=[begin,next_case],planned_case_window=[begin,begin+12],
                            receipt_sha256=digest(artifact),paired_matrix=dict(matrix),
                            actual=summary_totals(cases,errors),pairs=pairs))
        locked_cases.extend(cases)
        locked_errors.extend(errors)

    by_arm_task = {}
    for arm in ('D4-A','D4-B'):
        for operation in ('HOLD','PLACE'):
            registered = [name for name in planned if f'-{operation}-' in name and name.endswith('-'+arm)]
            observed = [row for row in locked_cases if row['case'] in registered]
            errors = [row for row in locked_errors if row['case'] in registered]
            by_arm_task[arm+'/'+operation] = dict(planned=len(registered),**summary_totals(observed,errors))
    return dict(source_spec_sha256=spec_sha,compiled_population_sha256=POPULATION_SHA,
                original_case_budget=len(planned),without_closed_case_evidence=planned[next_case:],
                actual=summary_totals(locked_cases,locked_errors),paired_matrix=dict(pair_counts),
                scientific_result_by_arm_operation=by_arm_task,layouts=layouts,audit_errors=locked_errors,
                cases=[dict(case=row['case'],arm=row['arm'],instruction=row['instruction'],layout=row['layout'],
                            native_result_sha256=row['native_result_sha256'],first_failure=row['first_failure'],
                            trace=row['trace'],score=row['terminal_full_score']['score'],
                            actual=summary_totals([row],[])) for row in locked_cases],
                new_generations=0,new_native_ticks=0,new_training_updates=0)


def summary_totals(cases, errors):
    scores = [case['terminal_full_score']['score'] for case in cases if case['terminal_full_score']['score']]
    crossing = Counter(P10=0,tail30=0,no_crossing=0)
    for row in cases:
        crossing.update(row['query_counts'])
    return dict(terminal_cases=len(cases)+len(errors),verified_cases=len(cases),
        evidence_errors=len(errors),scored_cases=len(scores),
        native_requests=sum(row['native_request'] for row in cases)+sum(row['native_requests'] for row in errors),
        native_ticks=sum(row['native_ticks'] for row in (*cases,*errors)),
        model_generations=sum(row['generations'] for row in (*cases,*errors)),
        complete_success=sum(row['task_success'] for row in scores),
        correct_grasp=sum(row['ever_target_grasp'] for row in scores),
        lift20mm=sum(row['ever_target_lift20mm'] for row in scores),
        observed_safety_STOP=sum(row['STOP'] for row in cases),
        observed_illegal_contact_STOP=sum((row['first_stop'] or {}).get('reason') == 'ILLEGAL_CONTACT' for row in cases),
        observed_integrations_after_STOP=sum(row['STOP_after_ticks'] or 0 for row in cases),
        safety_STOP_with_unknown_post_stop_ticks=sum(row['STOP'] and row['STOP_after_ticks'] is None for row in cases),
        observed_technical_failures=sum(bool(row['technical_failure']) for row in cases),
        confirmed_atoms_observed=sum(row['confirmed_atoms'] for row in cases),
        H40_first_closure_by_region=dict(crossing),
        first_failures=dict(Counter(row['first_failure'] for row in cases if row['first_failure'])),
        observed_wall_s=sum(row['wall_s'] for row in cases),
        observed_CPU_s=sum(row['CPU_s'] for row in cases))


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


def watch_remaining_layouts(root, spec_sha, output, deadline):
    """Seal each remaining layout from the original, already running panel."""
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import unit_state
    from scripts.pi0292.durable import write

    wrapper = root/'EXECUTION_SPEC.json'
    if digest(wrapper) != spec_sha or (output/'RESULT.private.json').exists():
        raise ValueError('WATCH_REQUIRES_THE_ORIGINAL_D4_SPEC_AND_NEW_OUTPUT')
    spec = read(wrapper)
    if spec['compiled_population']['sha256'] != POPULATION_SHA or len(spec['cases']) != 48:
        raise ValueError('WATCH_REQUIRES_ORIGINAL_48_CASE_POPULATION')
    result = dict(source_spec_sha256=spec_sha,new_generations=0,new_native_ticks=0,new_training_updates=0,layouts=[])
    for begin in (12,24,36):
        stop = begin+12
        artifact = output/f'CASES-{begin+1:02d}-{stop:02d}.private.json'
        if artifact.exists():
            raise FileExistsError('DO_NOT_REPLACE_EARLIER_LAYOUT_RECEIPTS')
        last = Path(spec['cases'][stop-1]).parent
        status = 'OBSERVER_FINITE_EXIT'
        while time.time() < deadline and not (output/'STOP.json').exists():
            if ((last/'RESULT.private.json').is_file()
                    and read(last/'control/LEASE.json')['status'] == 'CLOSED'):
                status = 'ORIGINAL_LAYOUT_CLOSED'
                break
            lease = read(root/'control/LEASE.json')
            state = unit_state('wm02-'+root.name)
            if lease['spec_sha256'] != spec_sha:
                raise ValueError('ORIGINAL_D4_OWNER_CHANGED_DURING_OBSERVATION')
            if lease['status'] == 'CLOSED':
                status = 'ORIGINAL_CAPSULE_TERMINAL'
                break
            if not state['pid']:
                status = 'OWNER_EFFECT_REQUIRES_RECONCILIATION'
                break
            write(output/'WATCH_PROGRESS.private.json',dict(epoch=time.time(),phase='WAIT_D4_LAYOUT',
                  case_interval=[begin,stop],watched_MainPID=state['pid'],spec_sha256=spec_sha,native_requests=0))
            time.sleep(30)
        recorded = audit(root,start=begin,limit=12,preserve_errors=True)
        write(artifact,dict(recorded,phase=status,source_spec_sha256=spec_sha,observed_epoch=time.time()))
        result['layouts'].append(dict(case_interval=[begin,stop],phase=status,sha256=digest(artifact),
                                     verified_cases=len(recorded['cases']),audit_errors=recorded['audit_errors']))
        if status != 'ORIGINAL_LAYOUT_CLOSED':
            break
    write(output/'RESULT.private.json',dict(result,observed_epoch=time.time()))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--limit', type=int, default=12)
    parser.add_argument('--watch-result',type=Path)
    parser.add_argument('--watch-remaining',type=Path)
    parser.add_argument('--summarize-layouts',type=Path,nargs='+')
    parser.add_argument('--spec-sha')
    parser.add_argument('--deadline',type=float)
    args = parser.parse_args()
    if args.summarize_layouts:
        if not args.spec_sha or args.watch_result or args.watch_remaining:
            parser.error('summary mode requires --spec-sha and no watcher')
        summary = summarize_layouts(args.root/'EXECUTION_SPEC.json',args.spec_sha,args.summarize_layouts)
        print(json.dumps(summary,allow_nan=False))
    elif args.watch_result or args.watch_remaining:
        if not args.spec_sha or not args.deadline:
            parser.error('watch mode requires --spec-sha and --deadline')
        if args.watch_result and args.watch_remaining:
            parser.error('select one watch mode')
        if args.watch_result:
            watch_first_wave(args.root,args.spec_sha,args.watch_result,args.deadline)
        else:
            watch_remaining_layouts(args.root,args.spec_sha,args.watch_remaining,args.deadline)
    else:
        print(json.dumps(audit(args.root,start=args.start,limit=args.limit), allow_nan=False))
