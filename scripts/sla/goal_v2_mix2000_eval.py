"""Release the original MIX2000 offline evaluation when the registered segment seals exactly."""
import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_contract import DOMAINS, PLAN as SCALE_PLAN
from go2arx5_vla.sla_adapter.scale20k_evaluation import PER_DOMAIN, RESTORE, data_sha
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import unit_state
from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0294.e1.journal import Journal
from scripts.sla.goal_v2_core_mix_after_native import PLAN, TRAINER, await_cold
from scripts.sla.scale20k_core_offline import GIB, offline_endpoint, pin, resources
from scripts.sla.scale20k_scheduled_handoff import verify_seal


ARM, STEP = 'SCALE-MIX20K', 2000


def exact_result(root, owner, result):
    effect = read_pin(result)
    if (owner['pid'] or owner['cgroup_pids'] or result['path'] != str(root/'train/RESULT.private.json')
            or effect['arm'] != ARM or effect['source_sha'] != TRAINER or effect['mode'] != 'train'
            or effect.get('failure') or effect['new_native_requests'] or effect['new_native_ticks']):
        raise ValueError('MIX2000_EVAL_REQUIRES_ORIGINAL_COLD_TRAINING_RECEIPT')
    step = effect['global_step']
    state = effect['status'],effect['exit_code']
    if step == STEP and state == ('FIXED_SEGMENT_SAVED',0):
        return True,step
    if 1000 < step < STEP and state == ('BOUNDED_SAVED_PREFIX',75):
        return False,step
    raise ValueError('MIX2000_EVAL_REQUIRES_EXACT_SUCCESS_OR_VERIFIED_EARLY_PREFIX')


def after_segment(root, name, expected, deadline, stopped):
    running = unit_state(name)
    if running['pid'] and any(running[k]!=expected[k] for k in ('pid','proc_start','ControlGroup','InvocationID')):
        raise PermissionError('MIX2000_EVAL_WAITED_ON_A_DIFFERENT_TRAINING_OWNER')
    terminal = await_cold(name,running,deadline=deadline,stopped=stopped)
    result = read(root/'RESULT.private.json')['stages']['train']
    exact,step = exact_result(root,terminal,result)
    return dict(step=step,exact=exact,terminal=terminal,training=result)


def resource_window(gpu, lock_path):
    cap = resources(gpu)
    reserve = math.ceil(cap['total_bytes']*.8)
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid',
        '--format=csv,noheader'],text=True,timeout=10).splitlines()
    try:
        lock = OwnerLock(lock_path)
    except BlockingIOError:
        lock = None
    else:
        lock.close()
    checks = dict(allocator=reserve+2*GIB<cap['total_bytes'],GPU=cap['free_bytes']>reserve+2*GIB,
                  host=cap['host_bytes']>36*GIB,exclusive=gpu not in map(str.strip,apps) and lock is not None)
    return dict(checks=checks,allocator_bytes=reserve,fraction=.8,GPU_headroom_bytes=2*GIB,host_headroom_bytes=36*GIB,**cap)


def run(spec):
    root, target, initial = (Path(spec[k]).resolve(strict=True) for k in ('root','evaluation_root','initial_training_root'))
    source = Path(__file__).resolve().parents[2]
    campaign = read_pin(spec['campaign'])
    initial_card, rotation = map(read_pin,(spec['initial_training_spec'],spec['rotation_spec']))
    end = min(campaign['deadline_epoch'],spec['site_deadline_epoch'])
    eval_unit = 'wm02-'+target.name+'.service'
    state = unit_state(spec['unit'])
    role_counts = dict(ordinary=PER_DOMAIN*len(DOMAINS),restore=len(RESTORE)*len(DOMAINS))
    if (root.stat().st_uid != os.getuid() or os.getuid()!=2009 or os.uname().nodename!='corelab-4080s'
            or state['pid']!=os.getpid() or state['ActiveState']!='active'
            or campaign['plan_sha']!=PLAN or campaign['deadline_epoch']!=campaign['t0_epoch']+14*86400
            or initial_card['source_sha']!=TRAINER or initial_card['global_stop_step']!=STEP
            or initial_card['root']!=str(initial) or rotation['campaign']!=spec['campaign']
            or rotation['previous_mix_schedule']!=spec['initial_training_spec']
            or rotation['site_deadline_epoch']!=spec['site_deadline_epoch']
            or not initial_card['deadline_epoch']<end
            or spec['source_sha']!=(source/'SOURCE_SHA.private').read_text().strip()
            or (root/'RESULT.private.json').exists() or (target/'SCHEDULE.private.json').exists()):
        raise PermissionError('MIX2000_EVAL_REQUIRES_EXACT_REGISTERED_ROTATION_AND_FINITE_SITE_SCOPE')
    panel = read_pin(spec['panel'])
    seed = read_pin(spec['baseline_training_spec'])
    if (panel['data']!=data_sha(seed['data']) or panel['old_TEST_access']
            or any(len(panel['rows'][d])!=PER_DOMAIN for d in DOMAINS)):
        raise ValueError('MIX2000_EVAL_REQUIRES_ORIGINAL_MATCHED_THREE_DOMAIN_POPULATION')
    stops = (root/'STOP.json',target/'STOP.json',Path(campaign['stop']))
    stopped = lambda: any(p.exists() for p in stops)
    result = dict(status='WAITING_EXACT_MIX2000',registered_eval_unit=eval_unit,formal_step=STEP,
        fixed_counts=role_counts,source_sha=spec['source_sha'],optimizer_updates=0,
        optimization_visits=0,model_generations=0,native_requests=0,native_ticks=0,eval_started=False)
    lock = OwnerLock(root/'OWNER.lock')
    try:
        worker_source, worker_sha = source, spec['source_sha']
        latest_start = end-1800
        write(root/'READY.private.json',dict(result,owner=state,latest_start_epoch=latest_start,
                                            initial=spec['initial_training_owner'],rotation=spec['rotation_owner']))
        a = after_segment(initial,spec['initial_training_unit'],spec['initial_training_owner'],latest_start,stopped)
        result['initial'] = a
        preceding_eval = spec.get('preceding_eval_observer')
        if preceding_eval:
            previous = read_pin(preceding_eval['spec'])
            name = previous['unit']
            owner = unit_state(name)
            expected = preceding_eval['owner']
            if (previous['campaign']!=spec['campaign'] or previous['evaluation_root']!=spec['evaluation_root']
                    or previous['panel']!=spec['panel'] or previous['site_deadline_epoch']!=spec['site_deadline_epoch']
                    or name!=expected['unit'] or name==spec['unit']
                    or owner['pid'] and any(owner[k]!=expected[k] for k in ('pid','proc_start','ControlGroup','InvocationID'))):
                raise PermissionError('MIX2000_EVAL_PRECEDING_OBSERVER_IDENTITY_OR_PANEL_CHANGED')
            await_cold(name,owner,deadline=latest_start,stopped=stopped)
            preceding_result = pin(Path(previous['root'])/'RESULT.private.json')
            preceding = read_pin(preceding_result)
            result['preceding_eval'] = preceding_result
            if preceding['eval_started'] or preceding['status']!='COLD_PREFIX_BELOW_2000_NO_EVAL':
                result.update(status='MIX2000_EVAL_PRECEDING_OBSERVER_TERMINAL_PRESERVED',
                              preceding_status=preceding['status'])
                return 0
            runtime = spec['evaluation_runtime']
            worker_source = Path(runtime['path']).resolve(strict=True).parents[2]
            worker_sha = previous['source_sha']
            if (pin(worker_source/'scripts/sla/scale20k_core_offline.py')!=runtime
                    or (worker_source/'SOURCE_SHA.private').read_text().strip()!=worker_sha):
                raise PermissionError('MIX2000_EVAL_ORIGINAL_OFFLINE_RUNTIME_CHANGED')
        exact = a
        if not a['exact']:
            rotation_owner = unit_state(spec['rotation_unit'])
            if rotation_owner['pid'] and any(rotation_owner[k]!=spec['rotation_owner'][k]
                    for k in ('pid','proc_start','ControlGroup','InvocationID')):
                raise PermissionError('MIX2000_EVAL_ROTATION_OWNER_CHANGED')
            await_cold(spec['rotation_unit'],rotation_owner,deadline=latest_start,stopped=stopped)
            handoff = read(Path(rotation['root'])/'RESULT.private.json')
            if handoff['status']!='MIX_SUCCESSOR_REGISTERED' or not handoff['successor_started']:
                raise ValueError('MIX2000_EVAL_EXACT_ENDPOINT_UNSEALED_ROTATION:'+handoff['status'])
            registered = handoff['registration']
            card = read_pin(registered['spec'])
            following = Path(rotation['successor_root']).resolve(strict=True)
            if (card['root']!=str(following) or card['unit']!=rotation['successor_unit']
                    or card['previous_train_result']!=a['training'] or card['global_stop_step']!=STEP
                    or card['source_sha']!=TRAINER or card['deadline_epoch']>end):
                raise ValueError('MIX2000_EVAL_ROTATION_SUCCESSOR_CARD_CHANGED')
            exact = after_segment(following,card['unit']+'.service',registered['unit'],latest_start,stopped)
            result['successor'] = exact
        if not exact['exact']:
            result['status'] = 'COLD_PREFIX_BELOW_2000_NO_EVAL'
            return 0
        train, step = exact['training'],exact['step']
        seal = verify_seal(train)
        training = pin(Path(train['path']).parent/'SPEC.private.json')
        if seal['step']!=STEP or panel['data']!=data_sha(read_pin(training)['data']):
            raise ValueError('MIX2000_EVAL_EXACT_SEAL_OR_PANEL_BINDING_CHANGED')
        result['sealed'] = seal
        cfg = dict(root=str(target),host=os.uname().nodename,uid=os.getuid(),plan=SCALE_PLAN,arm=ARM,step=step,
            training_source_sha=TRAINER,source_sha=worker_sha,panel=spec['panel'],
            previous_train_result=train,training_spec=training,GPU_uuid=spec['GPU_uuid'],
            JAX_memory_fraction=.8,neural_owner_lock=spec['neural_owner_lock'])
        offline_endpoint(cfg,target)
        headroom = resource_window(spec['GPU_uuid'],spec['neural_owner_lock'])
        write(root/'RESOURCE.private.json',headroom)
        if not all(headroom['checks'].values()) or shutil.disk_usage(target).free<=32*GIB:
            result['status'] = 'EXACT_MIX2000_EVAL_RESOURCE_REJECTED'
            return 0

        def launch():
            deadline = min(end,time.time()+5340)
            if stopped() or deadline-time.time()<=1800 or unit_state(eval_unit)['pid']:
                raise PermissionError('MIX2000_EVAL_EFFECTIVE_SITE_WINDOW_OR_OWNER_NOT_AVAILABLE')
            card = target/'SCHEDULE.private.json'
            if card.exists():
                raise FileExistsError(card)
            write(card,dict(cfg,deadline_epoch=deadline))
            env = dict(spec['eval_environment'],CUDA_VISIBLE_DEVICES=spec['GPU_uuid'],JAX_PLATFORMS='cuda',
                XLA_PYTHON_CLIENT_MEM_FRACTION='.80',XLA_PYTHON_CLIENT_PREALLOCATE='false')
            args = ['systemd-run','--user','--unit='+eval_unit,'--property=Restart=no',
                '--property=UMask=0077','--property=KillMode=control-group','--property=CPUAffinity=16 17',
                '--property=MemoryMax=22G','--property=TimeoutStopSec=90','--property=StandardInput=null',
                '--property=WorkingDirectory='+str(worker_source),
                '--property=RuntimeMaxSec='+str(math.ceil(deadline-time.time())+30),
                '--property=StandardOutput=append:'+str(target/'supervisor.log'),
                '--property=StandardError=append:'+str(target/'supervisor.log')]
            args.extend('--setenv='+k+'='+v for k,v in env.items())
            args.extend([spec['python'],'-B','-m','scripts.sla.scale20k_core_offline',str(card),pin(card)['sha256']])
            subprocess.run(args,check=True,capture_output=True,timeout=23)
            live = unit_state(eval_unit)
            if live['pid'] not in live['cgroup_pids'] or live['ActiveState']!='active':
                raise RuntimeError('MIX2000_EVAL_LAUNCH_EFFECT_REQUIRES_RECONCILIATION')
            return dict(unit=live,spec=pin(card),deadline_epoch=deadline,stop=str(target/'STOP.json'))

        result['evaluation'] = Journal(root/'handoff',root.name).once('RELEASE_ONE_ORIGINAL_MIX2000_OFFLINE',seal,launch)
        result.update(status='ORIGINAL_MIX2000_EVAL_REGISTERED',eval_started=True)
        return 0
    except (InterruptedError,TimeoutError) as error:
        result.update(status='MIX2000_EVAL_STOP_OR_SITE_BOUNDARY',first_failure=f'{type(error).__name__}: {error}')
        return 0
    except Exception as error:
        result.update(status='MIX2000_EVAL_REQUIRES_RECONCILIATION',first_failure=f'{type(error).__name__}: {error}')
        return 1
    finally:
        write(root/'RESULT.private.json',result)
        lock.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    args = parser.parse_args()
    raise SystemExit(run(read_pin(dict(path=args.spec,sha256=args.sha256))))
