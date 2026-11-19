"""Release the preflighted TRAIN scope only after the single world is cold."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import unit_state
from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
from go2arx5_vla.world_model.target_adaptation_teacher import AUTHORITY_SOURCE, CELLS, NATIVE_TICK_CAP, PLAN
from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0294.e1.journal import Journal
from scripts.wm02.multitask_stage_cards_v6 import pin


def previous_terminal(root, previous_pin, owner):
    lease = read(root/'control/LEASE.json')
    if lease['spec_sha256'] != previous_pin['sha256']:
        raise ValueError('Prior world lease identity changed')
    if owner['pid'] or owner['cgroup_pids']:
        return None
    if lease['status'] != 'CLOSED' or owner['ActiveState'] not in ('inactive', 'failed'):
        raise RuntimeError('Prior world ownership requires reconciliation')
    terminal = pin(root/'RESULT.private.json')
    result = read_pin(terminal)
    if (result['schema'] != 'SLA_D416_NATIVE_FOLLOWUP_ACTUAL' or result['failure']
            or result['unresolved_case_launches'] or any(row['technical_failure'] for row in result['records'])):
        raise RuntimeError('Prior world terminal evidence requires reconciliation')
    return terminal


def run(spec):
    root = Path(spec['root']).resolve(strict=True)
    previous = read_pin(spec['previous'])
    preflight = read_pin(spec['preflight'])
    config = read_pin(spec['cold_inputs'])
    authority = read_pin(config['authority'])
    old = Path(previous['registered_root'])
    target = Path(spec['new_root'])
    old_unit, new_unit = (f'wm02-{folder.name}.service' for folder in (old, target))
    latest_start = authority['deadline_epoch'] - preflight['case_lead_s']
    if (os.getuid() != preflight['uid'] or os.uname().nodename != preflight['host']
            or root.stat().st_uid != os.getuid() or unit_state(spec['unit'])['pid'] != os.getpid()
            or old.parent != target.parent or target.parent != root.parent
            or len({old, target, root, Path(config['root'])}) != 4
            or previous['evaluation_stage'] != 'J_SCREEN48'
            or previous['mobile_world_lock'] != preflight['mobile_world_lock']
            or preflight['source_sha'] != config['source_sha'] or preflight['registered_root'] != config['root']
            or preflight['compiled_population'] != config['population']
            or preflight['sim_only_project_authorization'] != config['authority']
            or authority['source'] != AUTHORITY_SOURCE or authority['plan_commit'] != PLAN
            or authority['hardware_allowed'] is not False or authority['native_request_cap'] != len(CELLS)
            or authority['native_tick_cap'] != NATIVE_TICK_CAP
            or authority['deadline_epoch'] >= authority['allocation_deadline_epoch']
            or latest_start != spec['latest_start']):
        raise PermissionError('Exact single-world binding and finite TRAIN scope required')

    result = dict(prior_unit=old_unit,next_unit=new_unit,latest_start=latest_start,
                  launch_calls=0,watcher_native_requests=0,watcher_integrations=0)
    lock = OwnerLock(root/'OWNER.lock')
    try:
        while time.time() < latest_start:
            if (root/'STOP.json').exists():
                result['status'] = 'STOPPED_WITHOUT_HANDOFF'
                return
            if target.exists():
                raise RuntimeError('Target collection already has a launch identity')
            owner = unit_state(old_unit)
            if owner['pid'] and (owner['pid'] != spec['prior_PID'] or owner['InvocationID'] != spec['prior_invocation']):
                raise RuntimeError('Prior world unit invocation changed')
            terminal = previous_terminal(old, spec['previous'], owner)
            if terminal is None:
                write(root/'PROGRESS.private.json',dict(status='WAIT_PREVIOUS_WORLD_OWNER',epoch=time.time(),
                      watcher_PID=os.getpid(),world_owner_PID=owner['pid'],native_requests=0))
                time.sleep(min(20,max(0,latest_start-time.time())))
                continue
            if time.time() >= latest_start or (root/'STOP.json').exists():
                break
            terminal_data = read_pin(terminal)
            if terminal_data['source_sha'] != previous['source_sha']:
                raise RuntimeError('Prior closed world source identity differs')
            draft = root/'LAUNCH_INPUT.private.json'
            inputs = dict(config,root=str(target),previous_world_owner=spec['previous'],previous_world_result=terminal)
            write(draft,inputs)
            immutable = pin(draft)

            def launch():
                env = dict(preflight['native_environment'],CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
                with (root/'launch.log').open('x') as log:
                    p = subprocess.run([preflight['python'],'-B','-m','scripts.wm02.target_adaptation_core_launch',
                        str(draft),immutable['sha256'],'--launch'],cwd=config['source_root'],env=dict(os.environ,**env),
                        stdout=log,stderr=subprocess.STDOUT,timeout=110)
                return dict(exit_code=p.returncode,actual_unit=unit_state(new_unit))

            result['launch_calls'] = 1
            actual = Journal(root/'handoff',root.name).once('START_FIRST6',dict(inputs=immutable),launch)
            if actual['exit_code'] != 0 or not actual['actual_unit']['pid']:
                raise RuntimeError('First-six managed launch requires reconciliation')
            result.update(status='FIRST6_MANAGED_UNIT_STARTED',unit_PID=actual['actual_unit']['pid'])
            return
        result['status'] = 'ORIGINAL_SAFE_START_WINDOW_ENDED'
    except Exception as error:
        result.update(status='HANDOFF_REQUIRES_RECONCILIATION',failure=f'{type(error).__name__}: {error}')
        raise
    finally:
        write(root/'RESULT.private.json',result)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    args = parser.parse_args()
    os.umask(0o077)
    run(read_pin(dict(path=args.spec,sha256=args.sha256)))
