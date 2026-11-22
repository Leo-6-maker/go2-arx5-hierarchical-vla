"""Single-world durable producer for the registered TRAIN240 and VAL24."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from go2arx5_vla.world_model.mfree_recovery_v2 import register
from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close, dirs
from go2arx5_vla.world_model.multitask_scope_v6 import read_pin, validate
from go2arx5_vla.world_model import multisource_d4_native as d4
from go2arx5_vla.world_model.storage import digest
from scripts.pi0292.durable import OwnerLock, write
from scripts.pi0294.e1.journal import Journal
from scripts.wm02.multitask_stage_cards_v6 import case_card, cold_native_owner, pin, require_unstarted_cases
from scripts.wm02.multitask_teacher_campaign_v6 import interface_records, teacher_row
from scripts.wm02.sim_only_campaign_v51 import fatal_control_violation
from scripts.sla.core_hopper_expert import order

def read(path):
    return json.loads(Path(path).read_bytes())


def capacity(root):
    disk = shutil.disk_usage(root)
    mem = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                   if x.startswith('MemAvailable:'))) * 1024
    return mem >= 13*1024**3 and disk.free >= max(50*1024**3,int(.15*disk.total))+8*1024**3


def reason(row,native):
    if row['hard_STOP']:
        events = native.get('physics_stops') or []
        stop = native.get('mobile_safety_stop') or {}
        if not isinstance(stop,dict):
            stop = {}
        # Preserve event details in the original result even if attribution is partial.
        event = events[0] if events and isinstance(events[0],dict) else {}
        return str(stop.get('reason') or event.get('reason') or 'NATIVE_SAFETY_STOP')
    detail = native.get('detail') or {}
    return detail.get('terminal_reason') or row['first_failure'] or 'UNCLASSIFIED_TERMINAL'


def snapshot(root, admitted, epoch, spec):
    value = dict(episodes=admitted, sealed_epoch=epoch, split='TRAIN', source=spec['scope'],
                 admitted_episodes=len(admitted),windows=sum(r['windows'] for r in admitted))
    if spec.get('multienv'):
        value.update(plan=spec['plan_commit'], producer=spec['source_sha'], population=spec['compiled_population'])
    path = root/'snapshots'/('TRAIN-%03d.private.json'%len(admitted))
    write(path,value)
    return pin(path)


def run(spec):
    root = Path(spec['registered_root'])
    authority = read_pin(spec['sim_only_project_authorization'])
    pop = read_pin(spec['compiled_population'])
    panel = {r['id']:r for r in read_pin(spec['registered_panel'])}
    interface = cold_native_owner(spec['interface_predecessor'])
    interface_records(interface,spec['mobile_world_lock'],identity=(d4.E0_SCOPE,d4.E0_PLAN))
    templates = {s['mobile_condition']['family']:s for s in map(read,interface['cases'])}
    journal = Journal(root/'control/journal',root.name)
    records = []
    admitted = []
    last_snapshot = 0
    last_time = time.time()
    streak, paused = {}, set()
    code, failure = 1, None
    pending = None
    not_started = {}
    multienv=spec.get('multienv')
    storage=spec.get('multienv_payload_storage')
    if storage and not multienv:
        raise PermissionError('Split payload storage is restricted to the PR425 collector')
    continuation=spec.get('multienv_cold_handoff')
    if continuation:
        from scripts.sla.core_multienv_wave_suffix import closed_handoff
        previous=closed_handoff(spec)
        already_paid={r['condition'] for r in previous['paid']}
        paused={tuple(cell) for cell in previous['retained_paused_cells']}
        not_started={r['condition']:r['reason'] for r in previous['paused']}
        for row in previous['retained_contact_streak']:
            signature=row['signature']
            if signature is not None:
                reason_name, phase, pairs=signature
                signature=(reason_name,phase,tuple(tuple(pair) for pair in pairs))
            streak[row['family'],row['instruction']]=signature,row['count']
    else:
        already_paid=set()
    if multienv:
        from scripts.sla import core_multienv_scope as me
    repair=spec.get('carry_repair')
    series=bool(repair and 'completed_pilot' in repair)
    original=Path(read_pin(repair['original_producer'])['registered_root']) if repair else root
    if multienv:
        original=Path(read_pin(multienv['handoff_owner'])['registered_root'])
    def state(stage, **extra):
        write(root/'PROGRESS.private.json', dict(stage=stage,epoch=time.time(),completed=len(records),
              actual_requests=sum(r['requests'] for r in records),closed_ticks=sum(r['ticks'] for r in records),
              admitted_train=len(admitted),current=pending,**extra))
    def stopped():
        return (root/'STOP.json').exists() or (original/'STOP.json').exists() or time.time()+1980 >= spec['deadline_epoch']
    def resources_ok():
        if not multienv:
            return capacity(root)
        mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                     if x.startswith('MemAvailable:')))*1024
        if storage:
            from scripts.sla.core_multienv_payload import capacity as payload_capacity
            volumes=payload_capacity(root,storage,multienv)
            write(root/'CAPACITY.private.json',dict(volumes=volumes,available_host_memory=mem,
                  min_host_memory=13*1024**3,epoch=time.time()))
            return volumes['passed'] and mem>=13*1024**3
        usage=shutil.disk_usage(root)
        reserve=(multienv['trajectory_reserve_bytes']+multienv['snapshot_reserve_bytes']
                 +multienv['transfer_reserve_bytes'])
        floor=max(50*1024**3,int(.15*usage.total))+8*1024**3
        write(root/'CAPACITY.private.json',dict(free_disk=usage.free,floor=floor,reserve=reserve,
            available_host_memory=mem,min_host_memory=13*1024**3,epoch=time.time()))
        return mem>=13*1024**3 and usage.free>=floor+reserve
    def run_one(row,world):
        item = panel[row['id']]
        family = row['family']
        folder,case = case_card(spec,templates[family],item,spec['registered_panel'])
        acquired=time.time()
        lease_end=min(spec['campaign_deadline_epoch'],acquired+2160)
        site_lease=root/'control'/('world-'+row['id']+'.json')
        inode=os.fstat(world.handle.fileno())
        write(site_lease,dict(kind='CORELAB_LIVE_SINGLE_WORLD_LEASE',owner_pid=os.getpid(),uid=os.getuid(),
              acquired_epoch=acquired,deadline_epoch=lease_end,lock=spec['mobile_world_lock'],
              lock_inode=inode.st_ino,lock_device=inode.st_dev,condition=row['id']))
        case.update(registry_scope=spec['scope'],collection_split=row['split'],compiled_population=spec['compiled_population'],
                    expert_training_eligible=row['split']=='TRAIN',site_world_lease=pin(site_lease),
                    allocation_deadline_epoch=lease_end,deadline_epoch=lease_end-60)
        if storage:
            case.update(multienv_payload_storage=storage,multienv=multienv)
        if repair:
            case['carry_repair']=repair
        require_unstarted_cases(case['registry'],{row['id']})
        validate(case)
        dirs(folder)
        register(spec,[(folder,case)])
        path = folder/'EXECUTION_SPEC.json'
        command=[case['python'],'-B',str(Path(spec['source_root'])/'scripts/wm02/v6_bootstrap.py'),
                 '--case',str(path),'--world-lock-fd',str(world.handle.fileno())]
        with (folder/'logs/case.log').open('xb') as stream:
            child=subprocess.Popen(command,cwd=spec['source_root'],env=dict(os.environ,**case['native_environment']),
                stdout=stream,stderr=subprocess.STDOUT,pass_fds=(world.handle.fileno(),))
        write(root/'CURRENT_CASE.private.json',dict(id=row['id'],pid=child.pid,epoch=time.time(),spec=pin(path)))
        child.wait()
        result=folder/'RESULT.private.json'
        if not result.is_file():
            raise RuntimeError('Child exited without terminal evidence: '+row['id'])
        return dict(spec=pin(path),result=pin(result),exit=child.returncode)

    try:
        write(root/'SUPERVISOR.private.json',dict(pid=os.getpid(),source_sha=spec['source_sha'],epoch=time.time()))
        if repair:
            from scripts.sla.core_hopper_expert import repair_rows
            rows=repair_rows(pop,repair)
            first48=[r['id'] for r in order(pop)[:48]]
            while not stopped():
                records0=read(original/'RECORDS.private.json')
                if [r['condition'] for r in records0[:48]]==first48:
                    break
                if (original/'RESULT.private.json').is_file():
                    raise ValueError('Original first 48 closed without the registered prefix')
                state('WAIT_ORIGINAL_FIRST48')
                time.sleep(30)
        else:
            rows=me.order(pop) if multienv else order(pop)
            if continuation:
                scheduled=set(continuation['conditions'])
                rows=[row for row in rows if row['id'] in scheduled]
                write(root/'PREDECESSOR_LEDGER.private.json',previous)
            write(root/'VAL_PRODUCER.private.json',dict(source=authority['frozen_producer'],source_seal=spec['source_seal'],
                                                        before_first_native=True))
        for row in rows:
            cell = row['family'],row['instruction']
            if row['split']=='TRAIN' and cell in paused:
                not_started[row['id']]='PAUSED_REPEAT_CONTACT_MECHANISM'
                continue
            while not stopped() and (not resources_ok() or (original/'STUDENT_TURN_READY.private.json').exists()):
                state('WAIT_CAPACITY_OR_STUDENT_HANDOFF')
                time.sleep(20)
            if stopped():
                break
            pending = row['id']
            while not stopped():
                try:
                    world=OwnerLock(spec['mobile_world_lock'])
                    if (series or multienv) and (stopped() or (original/'STUDENT_TURN_READY.private.json').exists()
                            or (multienv and not resources_ok())):
                        world.close()
                        state('WAIT_CAPACITY_OR_STUDENT_HANDOFF')
                        if not stopped():
                            time.sleep(20)
                        continue
                    break
                except BlockingIOError:
                    state('WAIT_WORLD'); time.sleep(20)
            else:
                break
            try:
                state('DISPATCH')
                receipt=journal.once(row['id'],dict(condition=row['id'],population=spec['compiled_population']['sha256'],
                                     source=spec['source_sha']),lambda:run_one(row,world))
            finally:
                world.close()
            case,native=read_pin(receipt['spec']),read_pin(receipt['result'])
            lease=read(Path(case['registered_root'])/'control/LEASE.json')
            if lease['status']!='CLOSED' or lease['spec_sha256']!=receipt['spec']['sha256'] or native['source_sha']!=spec['source_sha']:
                raise ValueError('Terminal identity or lease mismatch')
            outcome=teacher_row(case,native)
            rec=dict(condition=row['id'],split=row['split'],requests=native['physical_requests'],
                     ticks=native['actual_native_integrations'],result=receipt['result'],score=outcome['score'],
                     task_and_mobile_success=outcome['task_and_mobile_success'],technical_failure=outcome['technical_failure'],
                     hard_STOP=outcome['hard_STOP'],post_STOP_ticks=outcome['integrations_after_STOP'],
                     first_failure=reason(outcome,native),admitted=False)
            records.append(rec)
            write(root/'RECORDS.private.json',records)
            if (rec['requests'] != 1 or rec['ticks'] > row['native_max_ticks'] or fatal_control_violation(outcome)
                    or (outcome['hard_STOP'] and rec['post_STOP_ticks'] != 0) or outcome['technical_failure']):
                raise RuntimeError('Preserve terminal for technical or control review: '+row['id'])
            if outcome['task_and_mobile_success'] and outcome['completed'] and not outcome['hard_STOP']:
                request=Path(case['registered_root'])/'ADMISSION_INPUT.private.json'
                write(request,dict(case=receipt['spec'],native=receipt['result'],condition=row,statistics=spec['C20_statistics_path']))
                output=subprocess.check_output([spec['admission_python'],'-B','-m','scripts.sla.core_hopper_admission',
                        str(request),digest(request)],cwd=spec['source_root'],env=dict(os.environ,**spec['native_environment']),text=True)
                verdict=json.loads(output)
                if verdict['admitted']:
                    rec.update(admitted=True,episode=verdict['episode'])
                else:
                    rec['first_failure']='ADMISSION: '+verdict['first_failure']
            if row['split']=='TRAIN':
                if rec['admitted']:
                    admitted.append(rec['episode']);streak.pop(cell,None)
                else:
                    last,n=streak.get(cell,(None,0))
                    signature=me.failure_signature(case,rec) if multienv else rec['first_failure']
                    n=n+1 if signature is not None and last==signature else 1
                    streak[cell]=signature,n
                    if multienv:
                        rec['first_failure_signature']=signature
                    if n>=2:
                        paused.add(cell)
            if len(admitted)>last_snapshot and (len(admitted)-last_snapshot>=4 or time.time()-last_time>=21600):
                snapshot(root,admitted,time.time(),spec)
                last_snapshot=len(admitted);last_time=time.time()
            write(root/'RECORDS.private.json',records)
            pending=None
            state('CASE_SEALED',paused_train_cells=sorted(paused))
            if series and not rec['admitted']:
                paused.add(cell)
                state('REPAIRED_CELL_CLOSED',paused_train_cells=sorted(paused),
                      first_failure=rec['first_failure'])
                break
        code=0
    except Exception:
        failure=traceback.format_exc()
    finally:
        if len(admitted)>last_snapshot:
            snapshot(root,admitted,time.time(),spec)
        detail = {}
        if multienv:
            attempted = already_paid | {r['condition'] for r in records} | {pending}
            for row in me.order(pop):
                if row['id'] not in attempted:
                    not_started.setdefault(row['id'],'UNSTARTED_AFTER_TECHNICAL_TERMINAL' if failure else 'UNSTARTED_AT_STOP_OR_DEADLINE')
            detail.update(not_started=not_started,static_unbound=pop['unbound'])
        write(root/'RESULT.private.json',dict(exit_code=code,failure=failure,unreconciled=pending,records=records,
            paid_requests=sum(r['requests'] for r in records),verified_ticks=sum(r['ticks'] for r in records),
            admitted_train=len(admitted),paused_train_cells=sorted(paused),source_sha=spec['source_sha'],**detail))
        close(spec,code)
    return code
