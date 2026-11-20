"""Release one original MIX segment after the fixed ADAPT0 rows 7–12 settle."""
import argparse
import json
import math
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import time


PLAN = '1b2a25b7c1720051a2627d30831fb18130462374'
TRAINER = '31076fc7a64302b0705ae4d7e526c4341209a100'
NATIVE_SOURCE = 'df50e453c8b3f117d1af0c1a5036f06053126be4'
ADAPT_POPULATION = 'cdcd114a712739315d06ee923fe8221142d911ab6b60a64f30eac1b619eec2e0'
GIB = 1024**3


def verify_native_scope(effect, expected_root):
    from go2arx5_vla.world_model import adapt5k_native as adapt
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
    from scripts.wm02.multitask_panel_suffix import verified_prefix
    from scripts.wm02.multitask_stage_cards_v6 import pin

    root = Path(expected_root).resolve(strict=True)
    card = effect['spec']
    if card['path']!=str(root/'EXECUTION_SPEC.json'):
        raise ValueError('MIX_ROTATION_NATIVE_CARD_DOES_NOT_NAME_THE_FIXED_SECOND_BLOCK')
    panel = read_pin(card)
    result_pin = pin(root/'RESULT.private.json')
    result = read_pin(result_pin)
    parents = panel['previous_panel_segments']
    added = result['new_verified_native_requests']
    if (panel['scope']!=adapt.SCOPE or panel['evaluation_stage']!=adapt.DEV or panel['condition_stop']!=12
            or panel['compiled_population']['sha256']!=ADAPT_POPULATION or panel['source_sha']!=NATIVE_SOURCE
            or set(panel['packages'])!={'ADAPT0'} or len(parents)!=1 or result['exit_code'] not in (0,1)
            or not 1<=added<=6 or result['unresolved_case_launches'] or result['failure']):
        raise ValueError('MIX_ROTATION_REQUIRES_RECONCILED_FIXED_STUDENT_RESULTS')
    conditions = adapt.conditions(read_pin(panel['compiled_population']),adapt.DEV,adapt.arms_for(panel))
    rows = verified_prefix(dict(panel,previous_panel_segments=[*parents,dict(spec=card,result=result_pin)]),conditions)
    if len(rows)!=6+added:
        raise ValueError('MIX_ROTATION_ADAPT0_PREFIX_DOES_NOT_MATCH_ACTUAL_REQUESTS')
    return dict(panel=card,effect=result_pin,new_native_requests=added,new_native_ticks=sum(r['native_ticks'] for r in rows[6:]),
                new_model_generations=sum(r['actual_model_generations'] for r in rows[6:]),
                task_success=sum(bool(r['task_success']) for r in rows[6:]),integrations_after_STOP=0)


def await_cold(name, observed, *, deadline, stopped):
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import unit_state

    if observed['pid']:
        fd = os.pidfd_open(observed['pid'])
        try:
            current = unit_state(name)
            if any(current[k]!=observed[k] for k in ('pid','proc_start','ControlGroup')):
                raise PermissionError('MIX_ROTATION_WAITED_PID_OR_CGROUP_CHANGED')
            ready = select.poll()
            ready.register(fd,select.POLLIN)
            while True:
                if stopped():
                    raise InterruptedError('MIX_ROTATION_OPERATOR_STOP')
                left = deadline-time.time()
                if left<=0:
                    raise TimeoutError('MIX_ROTATION_PREDECESSOR_EXCEEDED_SITE_WINDOW')
                if ready.poll(max(1,int(1000*min(15,left)))):
                    break
        finally:
            os.close(fd)
    for _ in range(40):
        if stopped():
            raise InterruptedError('MIX_ROTATION_OPERATOR_STOP')
        current = unit_state(name)
        if not current['pid'] and not current['cgroup_pids'] and current['ActiveState'] in ('inactive','failed'):
            return current
        if current['pid'] and (not observed['pid'] or current['proc_start']!=observed.get('proc_start')):
            raise PermissionError('MIX_ROTATION_NEW_OWNER_REUSED_PREDECESSOR_UNIT')
        time.sleep(.25)
    raise PermissionError('MIX_ROTATION_PREDECESSOR_DID_NOT_CLOSE_ITS_PROCESS_GROUP')


def resources(root, draft, stopped, *, own_gpu_only=False):
    from scripts.pi0292.durable import write
    from scripts.sla.scale20k_core_prefix import resources as gpu_read

    # Native shutdown can leave transient memory pressure. This bounded read
    # records every sample and retains the previous launch thresholds.
    samples = []
    for _ in range(15):
        if stopped():
            raise InterruptedError('MIX_ROTATION_OPERATOR_STOP')
        pressure = {line.split()[0]:float(line.split()[1].split('=')[1])
                    for line in Path('/proc/pressure/memory').read_text().splitlines()}
        samples.append(dict(epoch=time.time(),**pressure))
        if max(pressure.values())<=.1:
            break
        time.sleep(6)
    roster = subprocess.check_output(['systemctl','--user','list-units','--state=running',
        '--no-legend','--plain','wm02-*.service'],text=True,timeout=15).splitlines()
    cuda = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader'],text=True,timeout=15).splitlines()
    conflicting_cuda = [row for row in cuda if row.split(',')[0].strip()==draft['GPU_uuid']] if own_gpu_only else cuda
    available = gpu_read(draft['GPU_uuid'])
    reserve = math.ceil(available['total_bytes']*draft['JAX_memory_fraction'])+2*GIB
    disk = shutil.disk_usage(root).free
    checks = dict(gpu=available['free_bytes']>reserve,host=available['host_bytes']>42*GIB,
        psi=max(pressure.values())<=.1,project_workers=not roster,compute_owners=not conflicting_cuda,disk=disk>32*GIB,
        stop_clear=not stopped())
    write(root/'RESOURCE.private.json',dict(GPU=available,checks=checks,pressure_samples=samples,
        running=roster,gpu_owners=cuda,conflicting_gpu_owners=conflicting_cuda,disk=disk,requirements=dict(GPU_bytes=reserve,host_bytes=42*GIB,
        disk_bytes=32*GIB,max_pressure_avg10=.1)))
    failures = [key for key,passed in checks.items() if not passed]
    if failures:
        raise PermissionError('MIX_ROTATION_RESOURCE_REJECT:'+','.join(failures))


def verify_unstarted_student_block(spec, block_scope, handoff, native_root, training_gpu):
    from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import unit_state

    failure = 'PermissionError: ADAPT_NATIVE_SITE_RESOURCE_OR_LOCK_UNAVAILABLE'
    witness = read_pin(spec['unstarted_student_resource_refusal'])
    expected_resource = native_root.with_name(native_root.name+'-launch')/'RESOURCE.private.json'
    has_later_owner = unit_state(block_scope['next_unit']+'.service')
    prior = read_pin(handoff['first_block']['result'])
    student_gpu = witness['GPU_uuid']
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader'],text=True,timeout=15).splitlines()
    occupants = [row for row in apps if row.split(',')[0].strip()==student_gpu]
    if (handoff['status']!='ADAPT_NEXT_BLOCK_REQUIRES_RECONCILIATION'
            or handoff['first_failure']!=failure or 'result' in handoff
            or any(handoff[field]!=0 for field in ('native_requests_started','native_ticks','model_generations','optimizer_updates'))
            or prior['new_verified_native_requests']!=6 or prior['unresolved_case_launches'] or prior['failure']
            or Path(spec['unstarted_student_resource_refusal']['path'])!=expected_resource
            or witness['native_requests']!=0 or witness['checks']['gpu_idle']
            or student_gpu==training_gpu or not occupants or native_root.exists()
            or has_later_owner['pid'] or has_later_owner['cgroup_pids']):
        raise PermissionError('MIX_ROTATION_REQUIRES_A_RECONCILED_UNSTARTED_STUDENT_BLOCK_AND_CURRENT_GPU_CONFLICT')
    return dict(status='ADAPT0_ROWS_7_TO_12_UNSTARTED_RESOURCE_REJECTION',requests=0,observed_gpu_owners=occupants,
                resource=spec['unstarted_student_resource_refusal'],prior=handoff['first_block'])


def watch(spec):
    from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import unit_state
    from scripts.pi0292.durable import OwnerLock, read, write
    from scripts.pi0294.e1.journal import Journal
    from scripts.sla.scale20k_core_prefix import closed_predecessor
    from scripts.sla.scale20k_scheduled_handoff import pin, verify_seal

    root = Path(spec['root']).resolve(strict=True)
    block_watch = Path(spec['block_observer_root']).resolve(strict=True)
    prior_root = Path(spec['previous_mix_root']).resolve(strict=True)
    native_root = Path(spec['native_root'])
    campaign = read_pin(spec['campaign'])
    previous = read_pin(spec['previous_mix_schedule'])
    block_scope = read_pin(spec['block_observer_spec'])
    state = unit_state(spec['unit'])
    if (os.uname().nodename!='corelab-4080s' or root.stat().st_uid!=os.getuid() or os.getuid()!=2009
            or state['pid']!=os.getpid() or campaign['plan_sha']!=PLAN or previous['source_sha']!=TRAINER
            or previous['root']!=str(prior_root) or block_scope['next_root']!=str(native_root)
            or block_scope['unit']!=spec['block_observer_unit'] or block_scope['campaign']!=spec['campaign']
            or block_scope['site_deadline_epoch']!=spec['site_deadline_epoch']):
        raise PermissionError('MIX_ROTATION_REQUIRES_ORIGINAL_OWNED_SITE_SCOPE')
    stops = (root/'STOP.json',prior_root/'STOP.json',native_root/'STOP.json',Path(campaign['stop']))
    stopped = lambda: any(path.exists() for path in stops)
    lock = OwnerLock(root/'OWNER.lock')
    report = dict(status='WAIT_BLOCK2',optimizer_updates=0,model_generations=0,native_requests=0,native_ticks=0,
                  successor_started=False)
    try:
        blocker = unit_state(spec['block_observer_unit'])
        expected = spec['block_observer_owner']
        if blocker['pid'] and any(blocker[k]!=expected[k] for k in ('pid','proc_start','ControlGroup')):
            raise PermissionError('MIX_ROTATION_WRONG_ADAPT_OBSERVER_OWNER')
        deadline = min(spec['site_deadline_epoch']-1860,campaign['deadline_epoch']-1860)
        write(root/'READY.private.json',dict(report,unit=state,depends_on=blocker,observe_until_epoch=deadline))
        report['block_observer_terminal'] = await_cold(spec['block_observer_unit'],blocker,deadline=deadline,stopped=stopped)
        handoff = read(block_watch/'RESULT.private.json')
        fallback = 'unstarted_student_resource_refusal' in spec
        if fallback:
            report['unstarted_student_block'] = verify_unstarted_student_block(spec,block_scope,handoff,native_root,previous['GPU_uuid'])
        else:
            if handoff['status']!='ADAPT0_BLOCK2_DISPATCHED':
                raise ValueError('MIX_ROTATION_ADAPT_BLOCK2_WAS_NOT_RECONCILED')
            native = handoff['result']
            native_unit = block_scope['next_unit']+'.service'
            observed = unit_state(native_unit)
            if (not native['started'] or native['deadline_epoch']>spec['site_deadline_epoch']
                    or observed['pid'] and any(observed[k]!=native['actual'][k] for k in ('pid','proc_start','ControlGroup'))):
                raise PermissionError('MIX_ROTATION_NATIVE_OWNER_DOES_NOT_MATCH_HANDOFF')
            write(root/'PROGRESS.private.json',dict(status='WAITING_FIXED_NATIVE_BLOCK2',native_owner=observed))
            report['native_terminal'] = await_cold(native_unit,observed,deadline=deadline,stopped=stopped)
            native_spec = read_pin(native['spec'])
            native_env = dict(os.environ,PYTHONPATH=native_spec['native_environment']['PYTHONPATH'],
                              OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
            native_path = Path(native_spec['source_root']).resolve(strict=True)
            if native_spec['source_sha']!=NATIVE_SOURCE or not native_path.is_relative_to(root.parent):
                raise PermissionError('MIX_ROTATION_ORIGINAL_NATIVE_RUNTIME_REQUIRED')
            command = [native_spec['python'],'-B',__file__,'--native',str(block_watch/'RESULT.private.json'),str(native_root)]
            checked = subprocess.check_output(command,cwd=native_path,env=native_env,text=True,timeout=90)
            report['fixed_native_block'] = json.loads(checked)

        mix_unit = previous['unit']+'.service'
        owner = unit_state(mix_unit)
        if owner['pid'] or owner['cgroup_pids']:
            raise PermissionError('MIX_ROTATION_ORIGINAL_TRAINER_IS_NOT_COLD')
        prior = read(prior_root/'RESULT.private.json')['stages']['train']
        successor = Path(spec['successor_root']).resolve(strict=True)
        draft = dict(previous,root=str(successor),unit=spec['successor_unit'],previous_train_result=prior)
        readback,original,step = closed_predecessor(draft,successor)
        if readback!=prior or str(original)!=spec['training_source'] or not 1000<step<=draft['global_stop_step']:
            raise ValueError('MIX_ROTATION_REQUIRES_ORIGINAL_FORWARD_TRAINING_PREFIX')
        report['seal'] = verify_seal(prior)
        if step==draft['global_stop_step']:
            report['status'] = 'ORIGINAL_MIX_MILESTONE_ALREADY_SEALED'
            return 0

        def release():
            resources(root,draft,stopped,own_gpu_only=fallback)
            if fallback:
                verify_unstarted_student_block(spec,block_scope,handoff,native_root,draft['GPU_uuid'])
            owner_check = OwnerLock(draft['neural_owner_lock'])
            owner_check.close()
            end = min(time.time()+14340,spec['site_deadline_epoch'],campaign['deadline_epoch'])
            if stopped() or end-time.time()<=1800:
                raise PermissionError('MIX_ROTATION_CANNOT_START_WITHIN_REMAINING_SITE_WINDOW')
            card = successor/'SCHEDULE.private.json'
            if card.exists() or unit_state(draft['unit']+'.service')['pid']:
                raise PermissionError('MIX_ROTATION_SUCCESSOR_ALREADY_CLAIMED')
            write(card,dict(draft,deadline_epoch=end))
            env = dict(spec['trainer_environment'],TMPDIR=str(successor/'tmp'),XDG_CACHE_HOME=str(successor/'cache'))
            command = ['systemd-run','--user','--unit='+draft['unit'],'--property=Restart=no','--property=UMask=0077',
                '--property=KillMode=control-group','--property=CPUAffinity=12 13 14 15','--property=MemoryMax=24G',
                '--property=StandardInput=null','--property=TimeoutStopSec=90',
                '--property=RuntimeMaxSec='+str(math.ceil(end-time.time())+30),'--property=WorkingDirectory='+str(original),
                '--property=StandardOutput=append:'+str(successor/'logs/queue.log'),
                '--property=StandardError=append:'+str(successor/'logs/queue.log')]
            command.extend('--setenv='+key+'='+value for key,value in env.items())
            command.extend([spec['python'],'-B','-m','scripts.sla.scale20k_core_prefix',str(card),pin(card)['sha256']])
            subprocess.run(command,check=True,capture_output=True,timeout=20)
            running = unit_state(draft['unit']+'.service')
            if running['pid'] not in running['cgroup_pids'] or running['ActiveState']!='active':
                raise RuntimeError('MIX_ROTATION_SUCCESSOR_START_REQUIRES_RECONCILIATION')
            return dict(unit=running,spec=pin(card),deadline_epoch=end,stop=str(successor/'STOP.json'))

        actual = Journal(root/'handoff',root.name).once('ONE_ORIGINAL_MIX_SUFFIX_AFTER_FIXED_ADAPT0_BLOCK2',report['seal'],release)
        report.update(status='MIX_SUCCESSOR_REGISTERED',successor_started=True,registration=actual)
        return 0
    except InterruptedError as error:
        report.update(status='STOPPED_BEFORE_SUCCESSOR',first_failure=str(error))
        return 0
    except Exception as error:
        report.update(status='MIX_ROTATION_REQUIRES_RECONCILIATION',first_failure=f'{type(error).__name__}: {error}')
        return 1
    finally:
        try:
            write(root/'RESULT.private.json',report)
        finally:
            lock.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha_or_native_root')
    parser.add_argument('--native',action='store_true')
    args = parser.parse_args()
    if args.native:
        handoff = json.loads(Path(args.spec).read_bytes())
        print(json.dumps(verify_native_scope(handoff['result'],args.sha_or_native_root),allow_nan=False))
    else:
        from go2arx5_vla.sla_adapter.runtime_v1 import read_pin

        raise SystemExit(watch(read_pin(dict(path=args.spec,sha256=args.sha_or_native_root))))
