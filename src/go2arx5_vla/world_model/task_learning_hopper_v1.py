"""Fresh single-H200 native Trainer qualification; PBS owns all allocation."""
import json
import os
import subprocess
import time
from pathlib import Path

from . import target_gr00t_profile_v1 as profile
from .storage import digest

PROJECT = 'CFP04-CF-007'
SCOPE = 'GR00T_TASK_LEARNING_CORE_HOPPER_V1'


def job(job_id, server=None):
    """Use the configured official server address, avoiding its broken alias."""
    server = server or os.environ.get('PBS_SERVER')
    profile.need(bool(server), 'PRIVATE_CONFIGURED_OFFICIAL_PBS_SERVER')
    query = subprocess.run(['/cm/shared/apps/pbspro/current/bin/qstat', '-f', '-F', 'json',
        job_id.split('@')[0] + '@' + server], capture_output=True, text=True, timeout=40, check=True)
    jobs = json.loads(query.stdout)['Jobs']
    matches = [v for k,v in jobs.items() if k.split('.')[0] == job_id.split('.')[0]]
    profile.need(len(matches) == 1, 'ONE_ACTUAL_HOPPER_JOB')
    return matches[0]


def allocated(record, job_id, *, uid, hostname, user):
    resources = record['Resource_List']
    hosts = {h.split('/')[0].split('.')[0] for h in record.get('exec_host','').split('+')}
    profile.need(uid == 22182 and user == 'e1192548' and record['Job_Owner'].split('@')[0] == user
        and record['job_state'] == 'R' and record['project'] == PROJECT
        and record['Job_Name'] == 'gr00t_task_hopper' and int(resources['ngpus']) == 1
        and int(resources['nodect']) == 1 and resources['select'].split(':')[0] == '1'
        and 'gpu_model=H200' in resources['select'] and hostname.split('.')[0] in hosts
        and bool(job_id), 'REAL_OWN_PROJECT_SINGLE_H200_ALLOCATION')


def site(spec):
    """Validate the actual Hopper allocation and immutable inputs before Torch."""
    recipe = profile.validate(spec)
    root = Path(spec['registered_root'])
    jid = os.environ.get('PBS_JOBID')
    profile.need(bool(jid) and root.resolve() == root
        and root.is_relative_to(Path('/scratch/e1192548/Projects/GR00T-WBC')), 'HOPPER_COMPUTE_SCRATCH')
    actual = job(jid, spec.get('pbs_server'))
    allocated(actual, jid, uid=os.getuid(), hostname=os.uname().nodename, user=os.environ.get('USER'))
    lease = json.loads((root / 'control/LEASE.json').read_text())
    profile.need(lease['status'] == 'ACTIVE' and lease['job_id'] == jid
        and lease['spec_sha256'] == digest(root / 'EXECUTION_SPEC.json'), 'NEW_HOPPER_CHILD_LEASE')
    registry = json.loads(Path(spec['registry']).read_text())
    profile.need(any(e.get('path') == str(root) and e.get('scope') == profile.PROFILE
        and e.get('source_sha') == spec['source_sha'] for e in registry['entries']), 'CANONICAL_HOPPER_REGISTRY')
    for name,pin in spec['source_files'].items():
        profile.need(digest(Path(spec['source_root']) / name) == pin, 'SEALED_HOPPER_SOURCE_' + name)
    profile.need(time.time() + spec['closeout_reserve_s'] < spec['deadline_epoch']
        and spec['closeout_reserve_s'] >= 1200, 'FRESH_FINITE_HOPPER_WINDOW')
    return recipe


def run(c):
    from contextlib import closing
    from scripts.pi0292.durable import OwnerLock, write
    from scripts.pi0294.e1.journal import Journal
    from scripts.pi0294.night10.native_relay import snapshot, once
    from .target_gr00t_allocation_v1 import compare_next_update, fit4_gate, child_environment
    need = profile.need
    root = Path(c['root'])
    jid = os.environ.get('PBS_JOBID')
    actual = job(jid, c.get('pbs_server'))
    allocated(actual, jid, uid=os.getuid(), hostname=os.uname().nodename, user=os.environ.get('USER'))
    lease = json.loads((root / 'control/LEASE.json').read_text())
    need(lease['status'] == 'PBS_SUBMITTED' and lease['job_id'] == jid
         and lease['config_sha256'] == digest(root / 'ALLOCATION_CONFIG.json'), 'NEW_SUBMITTED_HOPPER_SCOPE')
    from scripts.pi0294.multitask.shared_vanda import seconds
    started = time.monotonic()
    deadline = min(c['absolute_deadline_epoch'], time.time() + seconds(actual['Resource_List']['walltime'])
        - seconds(actual.get('resources_used',{}).get('walltime','00:00:00')) - 120)
    need(time.time() + 2400 < deadline, 'HOPPER_START_AND_CLOSEOUT_WINDOW')
    need(not (root / 'ALLOCATION_STARTED.private.json').exists(), 'NEVER_REPLAY_UNKNOWN_HOPPER_WORKER')
    for name,pin in c['source_files'].items():need(digest(Path(c['source_root']) / name) == pin, 'HOPPER_SOURCE_' + name)
    for name,pin in c['dataset_files'].items():need(digest(Path(c['bindings']['dataset']) / name) == pin, 'HOPPER_DATA_' + name)
    for name,pin in c['parent']['files'].items():need(digest(Path(c['parent']['path']) / name) == pin, 'HOPPER_PARENT_' + name)
    write(root / 'ALLOCATION_STARTED.private.json', dict(job_id=jid, PID=os.getpid(), epoch=time.time(),
        deadline_epoch=deadline, actual_job=actual))
    probe = subprocess.run([c['python'], '-B', '-c',
        'import torch,json;p=torch.cuda.get_device_properties(0);print(json.dumps(dict(count=torch.cuda.device_count(),name=p.name,uuid=str(p.uuid),major=p.major,total_memory=p.total_memory)))'],
        capture_output=True, text=True, timeout=90, check=True)
    gpu = json.loads(probe.stdout)
    need(gpu['count'] == 1 and 'H200' in gpu['name'] and gpu['major'] == 9
         and gpu['total_memory'] >= 130*1024**3, 'ACTUAL_SINGLE_H200_NOT_SITE_ALIAS')
    write(root / 'ACTUAL_GPU.private.json', gpu)
    journal = Journal(root / 'control/journal', root.name)
    children = []
    with closing(OwnerLock(root / 'NEURAL_OWNER.lock')):
        def child(name, mode, endpoint, *, start=0, resume=None, generation=None):
            need(time.time() + 2400 < deadline, 'HOPPER_CHILD_START_WINDOW')
            r = root / name
            def register():
                r.mkdir()
                for folder in ('control','tmp','cache'):(r / folder).mkdir()
                alias = r / 'model_inputs/nvidia/Cosmos-Reason2-2B'
                alias.parent.mkdir(parents=True)
                alias.symlink_to(c['bindings']['cosmos'], target_is_directory=True)
                recipe = profile.recipe(c['dataset_sha256'],c['statistics_sha256'], mode=mode,
                    variant='FULL_ACTION_SIDE', endpoint=endpoint)
                spec = dict(registered_root=str(r), source_root=c['source_root'], source_sha=c['source_sha'],
                    source_files=c['source_files'], bindings=c['bindings'], parent=c['parent'],
                    profile=profile.PROFILE, kind=profile.KIND, execution_site='HOPPER_SINGLE_GPU',pbs_server=c.get('pbs_server'),
                    recipe=recipe, allocation_request_id=root.name, generation_id=generation or root.name+'-'+name,
                    start_step=start, resume=resume, GPU_uuid=gpu['uuid'], GPU_name=gpu['name'],
                    GR00T_updates=endpoint-start, WM_updates=0, Actor_updates=0, TASK_ADAPTER_updates=0,
                    physics=False, WBC_changes=False, parent_manifest_sha256=profile.PARENT,
                    role='TASK-LEARNING-HOPPER-'+name, registry=c['registry'],
                    deadline_epoch=deadline, closeout_reserve_s=1200)
                profile.validate(spec)
                write(r / 'EXECUTION_SPEC.json', spec)
                with closing(OwnerLock(c['ledger_lock'])):
                    reg,pin = snapshot(c['registry'])
                    need(not any(e.get('path') == str(r) for e in reg['entries']), 'FRESH_HOPPER_CHILD')
                    reg['entries'].append(dict(path=str(r),role=spec['role'],scope=profile.PROFILE,
                        local_lease=str(r/'control/LEASE.json'),owner='EXISTING_LOCAL_PC',source_sha=c['source_sha']))
                    once(Path(c['registry']),reg,r.name+'-REGISTER',expected_before_sha256=pin)
                write(r / 'control/LEASE.json',dict(status='ACTIVE',job_id=jid,
                    spec_sha256=digest(r/'EXECUTION_SPEC.json'),deadline_epoch=deadline))
                return dict(spec_sha256=digest(r/'EXECUTION_SPEC.json'))
            registered = journal.once(name+'-REGISTER',dict(mode=mode,endpoint=endpoint,resume=resume),register)
            def execute():
                env = child_environment(c['source_root'],r)
                t = time.monotonic()
                with (r/'stdout.log').open('xb') as out,(r/'stderr.log').open('xb') as err:
                    process = subprocess.Popen([c['python'],'-B','-m','scripts.wm02.train_target_gr00t_v1',
                        '--spec',str(r/'EXECUTION_SPEC.json')],cwd=c['source_root'],env=env,stdout=out,stderr=err)
                    write(root/'CURRENT_CHILD.private.json',dict(PID=process.pid,name=name,epoch=time.time()))
                    try:status = process.wait(timeout=max(1,deadline-time.time()-1200))
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        try:process.wait(timeout=60)
                        except subprocess.TimeoutExpired:process.kill();process.wait()
                        status = 124
                result = json.loads((r/'RESULT.json').read_text())
                closed = json.loads((r/'control/LEASE.json').read_text())
                need(closed['status']=='CLOSED' and closed['worker_exit_code']==status,'HOPPER_CHILD_TERMINAL')
                return dict(name=name,exit_code=status,result=result,wall_s=time.monotonic()-t,
                    result_sha256=digest(r/'RESULT.json'))
            row = journal.once(name+'-EXECUTE',registered,execute)
            children.append(row)
            write(root/'CHILDREN_ACTUAL.private.json',dict(children=children))
            need(row['exit_code']==0 and row['result']['global_step']==endpoint,'HOPPER_CHILD_FAILED_'+name)
            return row['result']
        continuous = child('continuous4','RESTORE4',4,generation=root.name+'-restore4')
        split = child('split2','RESTORE4',2,generation=root.name+'-restore4')
        cp = Path(split['checkpoint'])
        resumed = child('resume4','RESTORE4',4,start=2,
            resume=dict(path=str(cp),sha256=digest(cp/'CHECKPOINT_COMPLETE.json')),generation=root.name+'-restore4')
        proof = compare_next_update(continuous['checkpoint'],resumed['checkpoint'])
        need(json.loads((Path(continuous['checkpoint'])/'SAMPLER_STATE.json').read_text())['cursor']==128,
             'HOPPER_FOUR_STEP_CURSOR128')
        write(root/'FOUR_VERSUS_TWO_PLUS_TWO_ACTUAL.json',dict(proof,comparison='CONTINUOUS4_VS_FRESH_PROCESS_2_PLUS_2'))
        mean_step = max(continuous['wall_s']/4, resumed['wall_s']/2)
        forecast = mean_step*200*1.3 + 600
        write(root/'FIT4_TIME_ADMISSION.json',dict(conservative_predicted_s=forecast,remaining_s=deadline-time.time()))
        if time.time()+forecast+1200>=deadline:
            return dict(status='HOPPER_ORIGINAL_TRAINER_RESTORE_READY_WAIT_FIT4_WINDOW',proof=proof,
                children=children,physics_integrations=0,formal_training=False,wall_s=time.monotonic()-started)
        fit = child('fit4-200','FIT4',200)
        gate = fit4_gate(root/'fit4-200')
        write(root/'FIT4_GATE_200.json',gate)
        return dict(status='HOPPER_ORIGINAL_TRAINER_FIT4_SEALED',proof=proof,fit4_gate=gate,
            children=children,physics_integrations=0,formal_training=False,wall_s=time.monotonic()-started)
