"""Registered CPU waiter and immutable, non-native GPU workers for D4."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def supervisor(spec):
    from scripts.pi0292.durable import read, write
    from scripts.pi0294.e1.journal import Journal
    from scripts.sla.close_goal_g1_v1 import wait_resource_owner
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
    from go2arx5_vla.sla_adapter.multisource_d4 import ARMS
    from go2arx5_vla.sla_adapter.runtime_v1 import read_pin

    root = Path(spec['registered_root'])
    journal = Journal(root/'control/journal', root.name)
    cards = [read_pin(pin) for pin in spec['children']]
    if tuple(card['arm'] for card in cards) != ARMS:
        raise ValueError('D4_ONLY_THE_FIXED_A_B_SEQUENCE_IS_ADMITTED')
    records, launched = [], set()
    lock, failure, code = None, None, 1
    started = time.monotonic()
    try:
        queue_deadline = spec['latest_pair_start_epoch']
        lock = wait_resource_owner(dict(spec, deadline_epoch=queue_deadline+300))
        for pin, card in zip(spec['children'], cards, strict=True):
            child = Path(card['registered_root'])
            if (root/'STOP.json').exists() or time.time() >= spec['deadline_epoch']-300:
                break
            result = {}
            for operation, output in (('train', 'RESULT.private.json'), ('restore', 'RESTORE.private.json')):
                if (root/'STOP.json').exists() or time.time() >= spec['deadline_epoch']-300:
                    break

                def execute():
                    command = [card['python'], '-B', __file__, operation, pin['path']]
                    with (child/'logs'/f'{operation}.log').open('xb') as log:
                        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                   env=dict(os.environ, **card['native_environment']))
                        launched.add(card['arm'])
                        write(root/'ACTIVE_CHILD.private.json', dict(arm=card['arm'], process=operation,
                            pid=process.pid, epoch=time.time(), spec_sha256=pin['sha256']))
                        exit_code = process.wait()
                    path = child/output
                    if not path.is_file():
                        raise RuntimeError('D4_CHILD_TERMINAL_MISSING_RESULT_NO_AUTO_REPLAY')
                    from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
                    actual = read(path)
                    return dict(result=dict(path=str(path),sha256=sha(path)), process_exit=exit_code,
                        result_exit=actual['exit_code'], new_updates=actual.get('new_updates'),
                        new_visits=actual.get('new_visits'), actual_generations=actual.get('actual_generations'))

                row = journal.once(f"{card['arm']}-{operation}", pin, execute)
                result[operation] = row
                if row['process_exit'] or row['result_exit']:
                    break
            records.append(dict(arm=card['arm'], **result))
            complete = len(result) == 2 and all(r['process_exit'] == r['result_exit'] == 0 for r in result.values())
            close(card, 0 if complete else 1, no_auto_replay=True)
            if not complete:
                break
        if len(records) == 2 and all('restore' in row and row['restore']['result_exit'] == 0
                                      and row['restore']['process_exit'] == 0 for row in records):
            code = 0
    except TimeoutError as error:
        failure = str(error)
        code = 75
    except BaseException:
        failure = traceback.format_exc()
    finally:
        if lock is not None:
            lock.close()
        for card in cards:
            folder = Path(card['registered_root'])
            if read(folder/'control/LEASE.json')['status'] == 'ACTIVE':
                close(card, code, not_started=card['arm'] not in launched, no_auto_replay=True)
    counted = [row['train'] for row in records if row.get('train',{}).get('new_updates') is not None]
    result = dict(exit_code=code, source_sha=spec['source_sha'], failure=failure, records=records,
        verified_updates=sum(row['new_updates'] for row in counted),
        verified_target_visits=sum(row['new_visits'] for row in counted),
        unresolved_update_denominator=bool(len(counted) != len(launched)),
        actual_arms_launched=sorted(launched), unstarted_arms=[arm for arm in ARMS if arm not in launched],
        new_source_visits=0, new_native_requests=0, new_integrations=0, elapsed_s=time.monotonic()-started)
    write(root/'RESULT.private.json', result)
    close(spec, code)
    return code


def main():
    operation, path = sys.argv[1:]
    spec = json.loads(Path(path).read_bytes())
    source = Path(spec['source_root'])
    if (operation not in ('supervise', 'train', 'restore')
            or spec['scope'] != 'SLA_D4_C20_G7_MATCHED_ONLY'
            or any(spec[key] for key in ('physics', 'new_physical_requests_budget', 'hardware_connections_allowed'))
            or Path(__file__).resolve() != source/'scripts/wm02/multisource_d4_bootstrap.py'):
        raise PermissionError('D4_REGISTERED_NON_NATIVE_SCOPE_REQUIRED')
    for member, expected in spec['source_files'].items():
        if hashlib.sha256((source/member).read_bytes()).hexdigest() != expected:
            raise ValueError('D4_FROZEN_SOURCE_CHANGED:'+member)
    source_seal = hashlib.sha256(json.dumps(spec['source_files'], sort_keys=True).encode()).hexdigest()
    if source_seal != spec['source_seal']:
        raise ValueError('D4_SOURCE_MEMBER_MANIFEST_CHANGED')
    roots = [str(source), *spec['dependency_roots']]
    sys.path[:0] = [prefix for root in roots for prefix in (root, root+'/src')]
    import go2arx5_vla.sla_adapter
    import go2arx5_vla.world_model
    import scripts.sla
    import scripts.wm02

    for package, directory in ((go2arx5_vla.sla_adapter,'src/go2arx5_vla/sla_adapter'),
                               (go2arx5_vla.world_model,'src/go2arx5_vla/world_model'),
                               (scripts.sla,'scripts/sla'), (scripts.wm02,'scripts/wm02')):
        package.__path__ = [root+'/'+directory for root in roots] + list(package.__path__)
    from scripts.wm02.mobile_boundary_runner_v1 import registered
    from go2arx5_vla.sla_adapter.runtime_v1 import read_pin

    registered(spec, path)
    plan = read_pin(spec['plan'])
    if any(plan['first_wave']['new_effect_caps'][key] != value for key,value in
           (('optimizer_updates',1000),('training_visits',32000),('native_requests',0))):
        raise ValueError('D4_EXECUTION_PLAN_BUDGET_MISMATCH')
    if operation == 'supervise':
        return supervisor(spec)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != spec['GPU_uuid']:
        raise PermissionError('D4_ONE_EXACT_BOUND_GPU_REQUIRED')
    from scripts.sla.train_multisource_d4 import run
    return run(spec, restore=operation == 'restore')


if __name__ == '__main__':
    raise SystemExit(main())
