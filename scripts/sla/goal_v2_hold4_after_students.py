"""Collect one TRAIN HOLD block after the owned student and MIX rotation closes."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.world_model import target_adaptation_teacher as train
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import unit_state
from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0294.e1.journal import Journal
from scripts.sla.goal_v2_tgt5k_native_after_rotation import cold
from scripts.sla.scale20k_scheduled_handoff import pin


GIB = 1024**3
HOLD_SOURCE = '55030f539c448728133d8ce7cf8dcc5afa4be725'


def require_unqualified_tgt(result):
    failure = result.get('rgb16_failure') or ''
    attempts = result.get('rgb16_attempts')
    replay = result.get('rgb16_status')
    numerical = (replay=='TGT_RGB16_FAIL' and type(attempts) is int and 0<attempts<=16
                 and 'SCALE20K_CORE_REPLAY_STRICT_OUTPUT_OR_GUARD_MISMATCH' in failure)
    unavailable = attempts==0 and 'TGT_RGB16_RESOURCE_REJECT:' in failure
    if (result['status']!='TGT5K_STRICT_RGB16_UNQUALIFIED' or result['native_started']
            or not (numerical or unavailable)):
        raise ValueError('HOLD4_TGT_PRIORITY_REQUIRES_RECONCILED_TERMINAL_OR_EXPLICIT_UNQUALIFIED_GATE')


def native_terminal(target, card, owner, *, deadline, stopped):
    info = read_pin(card)
    if Path(info['registered_root'])!=target or Path(card['path'])!=target/'EXECUTION_SPEC.json':
        raise ValueError('HOLD4_TERMINAL_WRONG_NATIVE_EXECUTION_IDENTITY')
    terminal = cold('wm02-'+target.name+'.service',owner,deadline=deadline,stopped=stopped)
    lease = read(target/'control/LEASE.json')
    effect = pin(target/'RESULT.private.json')
    outcome = read_pin(effect)
    if (lease['status']!='CLOSED' or lease['spec_sha256']!=card['sha256']
            or outcome['source_sha']!=info['source_sha'] or outcome['unresolved_case_launches']
            or outcome['failure']):
        raise ValueError('HOLD4_NATIVE_TERMINAL_NEEDS_RECONCILIATION')
    return terminal,effect,outcome


def run(spec):
    root = Path(spec['root']).resolve(strict=True)
    target = Path(spec['target'])
    owner = unit_state(spec['unit'])
    campaign = read_pin(spec['campaign'])
    rotation, tgt, prepared, old_input = map(read_pin,
        (spec['rotation'],spec['tgt_observer'],spec['preflight'],spec['prepared_inputs']))
    source = Path(old_input['source_root']).resolve(strict=True)
    base = read_pin(prepared['receipt']['spec'])
    end = min(spec['site_deadline_epoch'],campaign['deadline_epoch'])
    if (os.getuid()!=2009 or root.stat().st_uid!=os.getuid() or os.uname().nodename!='corelab-4080s'
            or owner['pid']!=os.getpid() or owner['pid'] not in owner['cgroup_pids']
            or spec['source_sha']!=(Path(__file__).resolve().parents[2]/'SOURCE_SHA.private').read_text().strip()
            or campaign['plan_sha']!=train.GOAL_HOLD_PLAN or campaign['deadline_epoch']!=campaign['t0_epoch']+14*86400
            or rotation['campaign']!=spec['campaign'] or tgt['campaign']!=spec['campaign']
            or tgt['rotation_spec']!=spec['rotation'] or rotation['site_deadline_epoch']!=spec['site_deadline_epoch']
            or base['compiled_population']!=old_input['population'] or base['scope']!=train.GOAL_HOLD_SCOPE
            or base['source_sha']!=old_input['source_sha'] or base['source_sha']!=HOLD_SOURCE
            or source.joinpath('SOURCE_SHA.private').read_text().strip()!=HOLD_SOURCE
            or prepared['registered_native_unit'] or target.exists() or (root/'RESULT.private.json').exists()):
        raise PermissionError('HOLD4_REQUIRES_OWN_FINITE_STUDENT_ROTATION_AND_COLD_POPULATION')
    stops = (root/'STOP.json',target/'STOP.json',Path(campaign['stop']))
    stopped = lambda: any(p.exists() for p in stops)
    result = dict(status='WAITING_ORIGINAL_STUDENT_ROTATION',source_sha=spec['source_sha'],
        collector_started=False,native_requests=0,native_ticks=0,optimizer_updates=0,optimization_visits=0,
        model_generations=0,admitted_episodes=0,admitted_windows=0)
    lock = OwnerLock(root/'OWNER.lock')
    try:
        latest = end-base['case_lead_s']-360
        write(root/'READY.private.json',dict(result,owner=owner,latest_start_epoch=latest,
                                            original_mix=spec['rotation_owner'],tgt=spec['tgt_owner']))
        result['mix_handoff_terminal'] = cold(rotation['unit'],spec['rotation_owner'],deadline=latest,stopped=stopped)
        mix = read(Path(rotation['root'])/'RESULT.private.json')
        if mix['status']=='MIX_SUCCESSOR_REGISTERED':
            reg = mix['registration']
            successor = read_pin(reg['spec'])
            if (successor['root']!=rotation['successor_root'] or successor['unit']!=rotation['successor_unit']
                    or successor['deadline_epoch']>end):
                raise ValueError('HOLD4_MIX_ROTATION_OWNER_CHANGED')
            result['mix_worker_terminal'] = cold(successor['unit']+'.service',reg['unit'],deadline=latest,stopped=stopped)
            folder = Path(successor['root'])
            mix_result = read(folder/'RESULT.private.json')
            if (read(folder/'control/LEASE.json')['status']!='CLOSED'
                    or read_pin(mix_result['stages']['train'])['exit_code'] not in (0,75)):
                raise ValueError('HOLD4_MIX_PREDECESSOR_REQUIRES_RECONCILIATION')
        elif mix['status']!='ORIGINAL_MIX_MILESTONE_ALREADY_SEALED':
            raise ValueError('HOLD4_MIX_ROTATION_REQUIRES_RECONCILIATION:'+mix['status'])
        result['prior_ADAPT0_block'] = mix['fixed_native_block']
        result['tgt_handoff_terminal'] = cold(tgt['unit'],spec['tgt_owner'],deadline=latest,stopped=stopped)
        tgt_report = read(Path(tgt['root'])/'RESULT.private.json')
        if tgt_report['status']=='TGT5K_FIRST6_DISPATCHED':
            registration = tgt_report['registration']
            t = Path(read_pin(tgt['native_config'])['root'])
            terminal,effect,r = native_terminal(t,registration['spec'],registration['actual'],
                                               deadline=latest,stopped=stopped)
            if not 1<=r['new_verified_native_requests']<=6 or any(
                    e['technical_failure'] for e in r['records'] if e['role']=='STUDENT'):
                raise ValueError('HOLD4_TGT_TERMINAL_UNRESOLVED')
            result['prior_TGT5K_block'] = dict(terminal=terminal,effect=effect,requests=r['new_verified_native_requests'])
        else:
            require_unqualified_tgt(tgt_report)
            result['prior_TGT5K_block'] = dict(status=tgt_report['status'],attempts=tgt_report['rgb16_attempts'],
                                             failure=tgt_report['rgb16_failure'])
        free = int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines()
                        if l.startswith('MemAvailable:')))*1024
        disk = shutil.disk_usage(root).free
        write(root/'CAPACITY.private.json',dict(free_host_bytes=free,free_disk_bytes=disk,
            minimum_host_bytes=35*GIB,minimum_disk_bytes=80*GIB,cpus=base['cpus']))
        if free<=35*GIB or disk<=80*GIB or not set(base['cpus']) <= os.sched_getaffinity(0):
            raise PermissionError('HOLD4_ORIGINAL_COLLECTOR_RESOURCE_REJECTED')
        world = OwnerLock(base['mobile_world_lock'])
        world.close()
        native_deadline = min(end-360,time.time()+5400)
        if stopped() or native_deadline-time.time()<=base['case_lead_s']:
            raise TimeoutError('HOLD4_OWNED_SITE_WINDOW_TOO_SHORT')
        authority = root/'AUTHORITY.private.json'
        write(authority,dict(read_pin(old_input['authority']),deadline_epoch=native_deadline,
                             allocation_deadline_epoch=native_deadline+120))
        card = root/'NATIVE_INPUT.private.json'
        write(card,dict(old_input,root=str(target),authority=pin(authority)))
        env = dict(os.environ,**base['native_environment'])
        env.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        python = base['python']

        def launch():
            with (root/'native_dispatch.log').open('x') as log:
                p = subprocess.run([python,'-B','-m','scripts.wm02.target_adaptation_core_launch',str(card),
                    pin(card)['sha256'],'--launch'],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=115)
            state = unit_state('wm02-'+target.name+'.service')
            if p.returncode or state['pid'] not in state['cgroup_pids']:
                raise RuntimeError('HOLD4_NATIVE_LAUNCH_EFFECT_REQUIRES_RECONCILIATION')
            return dict(unit=state,spec=pin(target/'EXECUTION_SPEC.json'))

        if stopped():
            raise InterruptedError('HOLD4_STOPPED_BEFORE_LAUNCH')
        activated = Journal(root/'handoff',root.name).once('START_ONE_FIXED_TRAIN_HOLD4',pin(card),launch)
        result.update(status='HOLD4_NATIVE_STARTED',collector_started=True,collection=activated)
        write(root/'PROGRESS.private.json',result)
        terminal,effect,r = native_terminal(target,activated['spec'],activated['unit'],deadline=end-120,stopped=stopped)
        inherited_ticks = sum(row['native_ticks'] for row in r['records'] if row['role']=='EXPERT')
        result.update(status='HOLD4_NATIVE_TERMINAL',collector_terminal=terminal,collection_result=effect,
                      native_requests=r['new_verified_native_requests'],native_ticks=inherited_ticks,
                      expert_successes=r['expert_task_success'])
        if r['new_verified_native_requests']==0 or stopped() or end-time.time()<120:
            return 0
        admit = root/'ADMISSION_INPUT.private.json'
        write(admit,dict(campaign_spec=activated['spec'],campaign_result=effect,
                         C20_statistics=spec['C20_statistics'],output=str(root/'admitted')))
        admission_env = dict(env,PYTHONPATH=spec['admission_PYTHONPATH'])
        with (root/'admission.log').open('x') as log:
            subprocess.run([spec['admission_python'],'-B','-m','scripts.sla.admit_target_adaptation',
                str(admit),pin(admit)['sha256']],cwd=source,env=admission_env,stdout=log,stderr=subprocess.STDOUT,
                check=True,timeout=min(240,end-time.time()))
        admitted = pin(root/'admitted/ACTUAL.private.json')
        evidence = read_pin(admitted)
        result.update(status='HOLD4_NATIVE_AND_LABEL_ADMISSION_SEALED',admission=admitted,
            admitted_episodes=evidence['admitted_episodes'],admitted_windows=evidence['admitted_windows'])
        return 0
    except (InterruptedError,TimeoutError) as error:
        result.update(status='HOLD4_STOP_OR_SITE_BOUNDARY',first_failure=str(error))
        return 0
    except Exception as error:
        result.update(status='HOLD4_REQUIRES_RECONCILIATION',first_failure=f'{type(error).__name__}: {error}')
        return 1
    finally:
        write(root/'RESULT.private.json',result)
        lock.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    args = parser.parse_args()
    os.umask(0o077)
    raise SystemExit(run(read_pin(dict(path=args.spec,sha256=args.sha256))))
