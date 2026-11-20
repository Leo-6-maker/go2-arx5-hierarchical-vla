"""Release the untouched ADAPT0 DEV rows under the current Core allocation."""
import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import time

from go2arx5_vla.world_model import adapt5k_native as adapt
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import unit_state
from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0294.e1.journal import Journal
from scripts.wm02.multitask_panel_suffix import budgets, verified_prefix
from scripts.wm02.multitask_stage_cards_v6 import pin


NATIVE_SOURCE = 'df50e453c8b3f117d1af0c1a5036f06053126be4'
OWNER_KEYS = ('pid', 'proc_start', 'ControlGroup', 'InvocationID')
BOUNDARIES = (12, 18, 24)
CASE_LEAD_S = 1980
SITE_MARGIN_S = 180


def next_boundary(cursor):
    if not 6 < cursor < 24:
        raise ValueError('ADAPT0_TAIL_REQUIRES_THE_RECONCILED_SECOND_BLOCK')
    return next(end for end in BOUNDARIES if end > cursor)


def expected_cold(name, expected, *, deadline, stop):
    owner = unit_state(name)
    if owner['pid']:
        if any(owner[key] != expected[key] for key in OWNER_KEYS):
            raise PermissionError('ADAPT0_TAIL_PREDECESSOR_OWNER_CHANGED')
        try:
            fd = os.pidfd_open(owner['pid'])
        except ProcessLookupError:
            fd = None
        if fd is not None:
            try:
                current = unit_state(name)
                if current['pid'] and any(current[key] != expected[key] for key in OWNER_KEYS):
                    raise PermissionError('ADAPT0_TAIL_PREDECESSOR_PID_REUSED')
                watcher = select.poll()
                watcher.register(fd, select.POLLIN)
                while not watcher.poll(0):
                    stop()
                    wait = min(15, deadline - time.time())
                    if wait <= 0:
                        raise TimeoutError('ADAPT0_TAIL_PREDECESSOR_EXCEEDED_SITE_WINDOW')
                    watcher.poll(max(1, int(1000 * wait)))
            finally:
                os.close(fd)
    for _ in range(40):
        stop()
        current = unit_state(name)
        if not current['pid'] and not current['cgroup_pids'] and current['ActiveState'] in ('inactive', 'failed'):
            return current
        if current['pid'] and any(current[key] != expected[key] for key in OWNER_KEYS):
            raise PermissionError('ADAPT0_TAIL_PREDECESSOR_UNIT_WAS_REUSED')
        time.sleep(.25)
    raise PermissionError('ADAPT0_TAIL_PREDECESSOR_CGROUP_STILL_PRESENT')


def verified_hold_terminal(config, requested_stop, latest):
    hold = read_pin(config['hold_observer'])
    root = Path(hold['root'])
    expected_cold(hold['unit'], config['hold_owner'], deadline=latest, stop=requested_stop)
    result_card = pin(root/'RESULT.private.json')
    result = read_pin(result_card)
    if result['source_sha'] != hold['source_sha'] or (root/'STOP.json').exists():
        raise PermissionError('ADAPT0_TAIL_HOLD4_IDENTITY_OR_STOP')
    target = Path(hold['target'])
    if not result['collector_started']:
        capacity_refused = (result.get('first_failure') == 'PermissionError: HOLD4_ORIGINAL_COLLECTOR_RESOURCE_REJECTED'
                            and (root/'CAPACITY.private.json').is_file())
        if (target.exists() or (root/'NATIVE_INPUT.private.json').exists()
                or (result['status'] != 'HOLD4_STOP_OR_SITE_BOUNDARY' and not capacity_refused)):
            raise PermissionError('ADAPT0_TAIL_HOLD4_START_REQUIRES_RECONCILIATION')
        return result_card
    effect = result['collection']
    expected_cold('wm02-'+target.name+'.service', effect['unit'], deadline=latest, stop=requested_stop)
    receipt = result['collection_result']
    lease = read(target/'control/LEASE.json')
    outcome = read_pin(receipt)
    if (effect['spec'] != pin(target/'EXECUTION_SPEC.json') or lease['status'] != 'CLOSED'
            or lease['spec_sha256'] != effect['spec']['sha256'] or receipt != pin(target/'RESULT.private.json')
            or outcome['source_sha'] != read_pin(effect['spec'])['source_sha']
            or outcome['failure'] or outcome['unresolved_case_launches'] or (target/'STOP.json').exists()):
        raise PermissionError('ADAPT0_TAIL_HOLD4_NATIVE_RESULT_UNRECONCILED')
    return result_card


def wait_shared_eval(config, original, requested_stop, latest):
    observer = read_pin(config['mix_eval_observer'])
    gpu = read_pin(original['gpu_reference'])
    if observer['GPU_uuid'] != gpu['GPU_uuid']:
        return None
    expected_cold(observer['unit'], config['mix_eval_owner'], deadline=latest, stop=requested_stop)
    result_card = pin(Path(observer['root'])/'RESULT.private.json')
    result = read_pin(result_card)
    target = Path(observer['evaluation_root'])
    worker = 'wm02-'+target.name+'.service'
    if result['eval_started']:
        effect = result['evaluation']
        expected_cold(worker, effect['unit'], deadline=latest, stop=requested_stop)
        if (effect['spec'] != pin(target/'SCHEDULE.private.json') or not (target/'RESULT.private.json').is_file()
                or read(target/'control/LEASE.json')['status'] != 'CLOSED'):
            raise PermissionError('ADAPT0_TAIL_SHARED_GPU_EVAL_NOT_RECONCILED')
    elif (unit_state(worker)['pid'] or (target/'SCHEDULE.private.json').exists()):
        raise PermissionError('ADAPT0_TAIL_SHARED_GPU_EVAL_START_NOT_RECONCILED')
    return result_card


def reconcile(panel, segments, conditions, registration, target):
    card = registration['spec']
    if card != pin(target/'EXECUTION_SPEC.json') or read_pin(card)['source_sha'] != NATIVE_SOURCE:
        raise ValueError('ADAPT0_TAIL_COLD_SCOPE_IDENTITY_CHANGED')
    segments.append(dict(spec=card, result=pin(target/'RESULT.private.json')))
    return verified_prefix(dict(panel, previous_panel_segments=segments), conditions, allow_complete=True)


def schedule(config, root, native, original, environment, segments, conditions, cursor, deadline):
    end = next_boundary(cursor)
    target = Path(config['targets'][str(end)])
    unit = 'wm02-'+target.name
    if target.exists() or unit_state(unit+'.service')['pid']:
        raise PermissionError('ADAPT0_TAIL_TARGET_IDENTITY_ALREADY_USED')
    requests, ticks, generations = budgets(conditions[cursor:end])
    authority = root/f'AUTHORITY-{end}.private.json'
    write(authority, dict(source=adapt.AUTHORITY_SOURCE, plan_commit=adapt.PLAN, goal_plan_sha=adapt.GOAL,
        source_sha=NATIVE_SOURCE, stage=adapt.DEV, condition_stop=end, hardware_allowed=False,
        native_request_cap=requests, generation_cap=generations, tick_cap=ticks,
        compiled_population_sha256=adapt.POPULATION_SHA, deadline_epoch=deadline,
        allocation_deadline_epoch=deadline+SITE_MARGIN_S))
    path = root/f'INPUT-{end}.private.json'
    write(path, dict(original, root=str(target), unit=unit, authority=pin(authority), previous_panel_segments=segments))
    card = pin(path)
    log = root/f'launch-{end}.log'
    command = [native['python'], '-B', '-m', 'scripts.wm02.adapt5k_core_native_launch', str(path), card['sha256'], '--launch']

    def dispatch():
        result = subprocess.run(command, cwd=original['source_root'], env=environment, capture_output=True,
                                text=True, timeout=120)
        with log.open('x') as stream:
            stream.write(result.stdout+result.stderr)
        result.check_returncode()
        launch = json.loads(result.stdout)
        ids = [row['id'] for row, _ in conditions[cursor:end]]
        if (launch['conditions'] != ids or not launch['started'] or launch['deadline_epoch'] != deadline
                or launch['actual']['unit'] != unit or launch['spec']['path'] != str(target/'EXECUTION_SPEC.json')):
            raise RuntimeError('ADAPT0_TAIL_NATIVE_DISPATCH_REQUIRES_RECONCILIATION')
        return launch

    return target, Journal(root/'registration', root.name).once(f'DISPATCH_FIXED_TO_{end}', card, dispatch)


def run(config):
    root = Path(config['root']).resolve(strict=True)
    owner = unit_state(config['unit'])
    campaign = read_pin(config['campaign'])
    hold = read_pin(config['hold_observer'])
    block2 = read_pin(config['block2_observer'])
    mix_eval = read_pin(config['mix_eval_observer'])
    first = read_pin(config['first_scope'])
    original = read_pin(block2['first_source_card'])
    site_end = min(config['site_deadline_epoch'], campaign['deadline_epoch'])
    base = root.parent
    targets = [Path(config['targets'][str(end)]) for end in BOUNDARIES]
    if (os.getuid() != 2009 or root.stat().st_uid != os.getuid() or os.uname().nodename != 'corelab-4080s'
            or os.getpid() != owner['pid'] or owner['pid'] not in owner['cgroup_pids']
            or config['source_sha'] != (Path(__file__).resolve().parents[2]/'SOURCE_SHA.private').read_text().strip()
            or campaign['plan_sha'] != adapt.GOAL or campaign['deadline_epoch'] != campaign['t0_epoch']+14*86400
            or hold['campaign'] != config['campaign'] or block2['campaign'] != config['campaign']
            or hold['site_deadline_epoch'] != config['site_deadline_epoch']
            or block2['site_deadline_epoch'] != config['site_deadline_epoch']
            or mix_eval['campaign'] != config['campaign'] or mix_eval['site_deadline_epoch'] != config['site_deadline_epoch']
            or first['condition_stop'] != 6 or first['registered_root'] != block2['first_block']
            or first['source_sha'] != original['source_sha'] or original['source_sha'] != NATIVE_SOURCE
            or original['arm'] != 'ADAPT0' or first['compiled_population'] != original['population']
            or first['packages'] != {'ADAPT0':original['package']}
            or {p.parent for p in targets} != {base} or len(set(targets)) != len(BOUNDARIES)
            or any(p.exists() for p in targets)):
        raise PermissionError('ADAPT0_TAIL_REQUIRES_FIXED_SITE_AND_ORIGINAL_ADAPT0')
    deadline = site_end-SITE_MARGIN_S
    latest = deadline-CASE_LEAD_S
    source = Path(first['source_root']).resolve(strict=True)
    if (source != Path(original['source_root']).resolve(strict=True)
            or (source/'SOURCE_SHA.private').read_text().strip() != NATIVE_SOURCE):
        raise PermissionError('ADAPT0_TAIL_ORIGINAL_NATIVE_SOURCE_REQUIRED')
    conditions = adapt.conditions(read_pin(original['population']), adapt.DEV, adapt.arms_for(first))
    environment = dict(os.environ, **first['native_environment'])
    environment.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', PYTHONDONTWRITEBYTECODE='1',
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    stop_paths = (root/'STOP.json', Path(campaign['stop']))
    report = dict(status='WAITING_ORIGINAL_STUDENT_ROTATION_AND_HOLD4', owner=owner, results=[], source_sha=config['source_sha'],
        native_source=NATIVE_SOURCE, optimizer_updates=0, optimization_visits=0, native_requests=0, native_ticks=0, model_generations=0)
    lock = OwnerLock(root/'OWNER.lock')
    active = None

    def stopped():
        if any(p.exists() for p in stop_paths):
            if active is None:
                raise InterruptedError('ADAPT0_TAIL_OPERATOR_STOP')
            target, launch = active
            state = unit_state(launch['actual']['unit']+'.service')
            if state['pid'] and all(state[key] == launch['actual'][key] for key in OWNER_KEYS):
                stop_file = target/'STOP.json'
                if not stop_file.exists():
                    write(stop_file, dict(reason='ADAPT0_TAIL_STOP', epoch=time.time()))

    try:
        first_segment = dict(spec=config['first_scope'], result=pin(Path(first['registered_root'])/'RESULT.private.json'))
        segments = [first_segment]
        if len(verified_prefix(dict(first, previous_panel_segments=segments), conditions)) != 6:
            raise ValueError('ADAPT0_TAIL_REQUIRES_THE_SIX_ORIGINAL_TERMINALS')
        write(root/'READY.private.json', dict(report, verified_prefix=6, latest_launch_epoch=latest))
        report['hold_result'] = verified_hold_terminal(config, stopped, latest)
        report['shared_mix_eval_result'] = wait_shared_eval(config, original, stopped, latest)
        previous = read(Path(block2['root'])/'RESULT.private.json')
        if previous['status'] != 'ADAPT0_BLOCK2_DISPATCHED' or not previous['result']['started']:
            raise ValueError('ADAPT0_TAIL_REQUIRES_AN_ACTUAL_SECOND_BLOCK')
        second = Path(block2['next_root'])
        cursor = len(reconcile(first, segments, conditions, previous['result'], second))
        report['carried_native_requests'] = cursor
        while cursor < len(conditions):
            stopped()
            if time.time() >= latest:
                raise TimeoutError('ADAPT0_TAIL_REMAINING_SITE_WINDOW_BELOW_ORIGINAL_CASE_LEAD')
            active = schedule(config, root, first, original, environment, segments, conditions, cursor, deadline)
            target, launch = active
            report.update(status='ADAPT0_TAIL_NATIVE_DISPATCHED', active=launch)
            write(root/'PROGRESS.private.json', report)
            expected_cold(launch['actual']['unit']+'.service', launch['actual'], deadline=site_end, stop=stopped)
            active = None
            rows = reconcile(first, segments, conditions, launch, target)
            added = rows[cursor:]
            report['results'].append(dict(effect=segments[-1], start=cursor, stop=len(rows),
                task_successes=sum(bool(row['task_success']) for row in added)))
            report['native_requests'] += len(added)
            report['model_generations'] += sum(row['actual_model_generations'] for row in added)
            report['native_ticks'] += sum(row['native_ticks'] for row in added)
            del report['active']
            if len(rows) != next_boundary(cursor):
                raise TimeoutError('ADAPT0_TAIL_FINITE_PREFIX_PRESERVED')
            cursor = len(rows)
        report['status'] = 'ADAPT0_ORIGINAL_DEV24_SEALED'
        return 0
    except (InterruptedError, TimeoutError) as error:
        report.update(status='ADAPT0_TAIL_STOP_OR_SITE_BOUNDARY', first_failure=str(error))
        return 0
    except Exception as error:
        report.update(status='ADAPT0_TAIL_REQUIRES_RECONCILIATION', first_failure=f'{type(error).__name__}: {error}')
        return 1
    finally:
        write(root/'RESULT.private.json', report)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    args = parser.parse_args()
    os.umask(0o077)
    raise SystemExit(run(read_pin(dict(path=args.spec, sha256=args.sha256))))
