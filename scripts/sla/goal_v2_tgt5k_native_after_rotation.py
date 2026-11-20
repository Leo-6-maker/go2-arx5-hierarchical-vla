"""Wait on Core owners once, then dispatch the original TGT5K evaluation block."""
import argparse
import json
import os
from pathlib import Path
import time

from go2arx5_vla.world_model import scale20k_native as scale
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import unit_state
from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0294.e1.journal import Journal
from scripts.sla.goal_v2_core_mix_after_native import await_cold
from scripts.sla.scale20k_scheduled_handoff import verify_seal
from scripts.wm02.multitask_panel_suffix import budgets
from scripts.wm02.multitask_stage_cards_v6 import pin
from scripts.wm02.scale20k_core_native_launch import GOAL, launch


def cold(name, expected, *, deadline, stopped):
    observed = unit_state(name)
    if observed['pid'] and any(observed[k]!=expected[k] for k in ('pid','proc_start','ControlGroup')):
        raise PermissionError('TGT5K_OBSERVED_UNEXPECTED_PREDECESSOR_OWNER')
    return await_cold(name,observed,deadline=deadline,stopped=stopped)


def run(spec):
    root = Path(spec['root']).resolve(strict=True)
    owner = unit_state(spec['unit'])
    campaign = read_pin(spec['campaign'])
    replay_spec,rotation_spec,native = map(read_pin,(spec['replay_observer_spec'],spec['rotation_spec'],spec['native_config']))
    if (owner['pid']!=os.getpid() or owner['pid'] not in owner['cgroup_pids']
            or os.uname().nodename!='corelab-4080s' or root.stat().st_uid!=2009 or os.getuid()!=2009
            or campaign['plan_sha']!=GOAL or native['campaign']!=spec['campaign']
            or replay_spec['campaign']!=spec['campaign'] or rotation_spec['campaign']!=spec['campaign']
            or spec['site_deadline_epoch']!=rotation_spec['site_deadline_epoch']
            or spec['site_deadline_epoch']!=native['site_deadline_epoch']
            or (root/'RESULT.private.json').exists() or (root/'READY.private.json').exists()):
        raise PermissionError('TGT5K_NATIVE_REQUIRES_ITS_EXACT_FINITE_CORE_OBSERVER')
    stop_files = (root/'STOP.json',Path(campaign['stop']),Path(native['root'])/'STOP.json')
    stopped = lambda: any(p.exists() for p in stop_files)
    deadline = min(spec['site_deadline_epoch'],campaign['deadline_epoch'])-120
    actual = dict(status='WAITING_STRICT_RGB16',source_sha=native['source_sha'],native_started=False,
                  native_requests=0,native_ticks=0,optimizer_updates=0,model_generations=0)
    lock = OwnerLock(root/'OBSERVER.lock')
    try:
        if stopped() or time.time()>=deadline:
            raise PermissionError('TGT5K_NATIVE_STOPPED_OR_WINDOW_CLOSED')
        write(root/'READY.private.json',dict(actual,owner=owner,deadline_epoch=deadline,
            rgb16_dependency=spec['replay_observer_unit'],rotation_dependency=spec['rotation_unit']))
        replay_root = Path(replay_spec['root'])
        actual['rgb16_observer_terminal'] = cold(spec['replay_observer_unit'],spec['replay_observer_owner'],
            deadline=min(deadline,replay_spec['site_deadline_epoch']+120),stopped=stopped)
        handoff = read(replay_root/'RESULT.private.json')
        worker = unit_state(replay_spec['worker'])
        if worker['pid'] or worker['cgroup_pids']:
            raise PermissionError('TGT5K_RGB16_WORKER_NOT_COLD')
        if handoff['status']!='TGT_RGB16_PASS':
            actual.update(status='TGT5K_STRICT_RGB16_UNQUALIFIED',rgb16_status=handoff['status'],
                          rgb16_failure=handoff.get('first_failure'),rgb16_attempts=handoff.get('generation_attempts',0))
            return 0
        replay = handoff['replay']
        replay_run = replay_root/'tgt5k-rgb16-worker'
        if (Path(replay['path'])!=replay_run/'RESULT.private.json'
                or Path(handoff['spec']['path'])!=replay_run/'EXECUTION.private.json'
                or handoff['generations']!=16 or handoff['generation_attempts']!=16
                or read(replay_run/'control/LEASE.json')['status']!='CLOSED'):
            raise ValueError('TGT5K_RGB16_SUCCESS_REQUIRES_FULL_COLD_EVIDENCE')
        scale.endpoint(native['arm'],native['package'],native['evaluation'],replay)
        actual.update(rgb16=pin(Path(replay['path'])),status='WAITING_ORIGINAL_MIX_ROTATION')
        write(root/'PROGRESS.private.json',actual)

        actual['mix_observer_terminal'] = cold(spec['rotation_unit'],spec['rotation_owner'],
            deadline=deadline,stopped=stopped)
        report = read(Path(rotation_spec['root'])/'RESULT.private.json')
        if report['status']=='MIX_SUCCESSOR_REGISTERED':
            registration = report['registration']
            successor = Path(rotation_spec['successor_root'])
            card = read_pin(registration['spec'])
            if (Path(registration['spec']['path'])!=successor/'SCHEDULE.private.json'
                    or card['unit']!=rotation_spec['successor_unit']
                    or registration['deadline_epoch']>spec['site_deadline_epoch']):
                raise PermissionError('TGT5K_REQUIRES_ORIGINAL_ROTATION_OWNER_CLOSURE')
            trainer_unit = card['unit']+'.service'
            actual['mix_worker_terminal'] = cold(trainer_unit,registration['unit'],deadline=deadline,stopped=stopped)
            result = read(successor/'RESULT.private.json')
            lease = read(successor/'control/LEASE.json')
            if lease['status']!='CLOSED' or lease['unit']!=trainer_unit:
                raise PermissionError('TGT5K_ROTATED_MIX_LEASE_NOT_COLD')
            actual['mix_seal'] = verify_seal(result['stages']['train'])
        elif report['status']!='ORIGINAL_MIX_MILESTONE_ALREADY_SEALED':
            raise PermissionError('TGT5K_MIX_ROTATION_REQUIRES_RECONCILIATION:'+report['status'])

        ordered = scale.conditions(read_pin(native['population']),scale.DEV,{native['arm']})
        selected = scale.next_block(dict(condition_stop=6),ordered,0)
        requests,ticks,generations = budgets(selected)
        end = min(time.time()+3.75*3600,deadline)
        if stopped() or end-time.time()<=1980:
            actual.update(status='TGT5K_QUALIFIED_AWAITING_FRESH_SITE_WINDOW')
            return 0
        authority = root/'AUTHORITY.private.json'
        write(authority,dict(source=scale.AUTHORITY_SOURCE,plan_commit=scale.PLAN,goal_plan_sha=GOAL,
            source_sha=native['source_sha'],compiled_population_sha256=scale.POPULATION_SHA,stage=scale.DEV,
            condition_stop=6,native_request_cap=requests,tick_cap=ticks,generation_cap=generations,
            hardware_allowed=False,deadline_epoch=end,allocation_deadline_epoch=end+120))
        execution = root/'NATIVE_INPUT.private.json'
        ready = dict(native,replay=replay,replay_worker_spec=handoff['spec'],authority=pin(authority))
        write(execution,ready)
        if stopped():
            raise PermissionError('TGT5K_STOP_BEFORE_NATIVE_DISPATCH')
        registration = Journal(root/'submission',root.name).once('LAUNCH_FIXED_TGT5K_FIRST_SIX',pin(execution),
            lambda:launch(ready,start=True))
        actual.update(status='TGT5K_FIRST6_DISPATCHED',native_started=True,registration=registration)
        return 0
    except TimeoutError as error:
        actual.update(status='TGT5K_QUALIFIED_AWAITING_FRESH_SITE_WINDOW' if 'rgb16' in actual
                      else 'TGT5K_RGB16_OWNER_EXCEEDED_SITE_WINDOW',first_failure=str(error))
        return 0
    except (Exception,KeyboardInterrupt) as error:
        actual.update(status='TGT5K_NATIVE_REQUIRES_RECONCILIATION',first_failure=f'{type(error).__name__}: {error}')
        return 1
    finally:
        try:
            write(root/'RESULT.private.json',actual)
        finally:
            lock.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    args = parser.parse_args()
    os.umask(0o077)
    raise SystemExit(run(read_pin(dict(path=args.spec,sha256=args.sha256))))
