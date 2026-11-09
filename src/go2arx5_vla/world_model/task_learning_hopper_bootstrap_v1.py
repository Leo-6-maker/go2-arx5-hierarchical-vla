"""Activate one registered PBS SFT job, then exec the original finite Trainer."""
import json,math,os,subprocess,time,traceback
from pathlib import Path
from contextlib import closing
from . import task_learning_training_profile_v1 as task
from . import target_gr00t_profile_v1 as original
from .task_learning_hopper_v1 import job,allocated
from .storage import digest


def configuration(c):
    original.need(type(c.get('pbs_server'))is str and bool(c['pbs_server'].strip()),
        'EXPLICIT_OFFICIAL_PBS_SERVER_BEFORE_FORMAL_SUBMISSION')
    original.need(Path(c['root']).is_absolute()and Path(c['root']).is_relative_to(Path(task.PROJECT_ROOT))
        and type(c['predicted_work_s'])in(int,float)and math.isfinite(c['predicted_work_s'])
        and c['predicted_work_s']>0 and c['predicted_work_s']+1200+120<c['wall_s'],
        'OWN_PROJECT_ROOT_AND_MEASURED_FINITE_FORMAL_WINDOW')
    return execution_spec(c,job_id='NOT_ALLOCATED_PREFLIGHT',
        gpu=dict(uuid='NOT_ALLOCATED',name='NOT_ALLOCATED'),deadline=c['absolute_deadline_epoch'])


def preflight(c):
    from .task_learning_training_admission_v1 import validate
    return validate(configuration(c))


def execution_spec(c,*,job_id,gpu,deadline):
    p=c['recipe'];start=c['start_step']
    s=dict(registered_root=c['root'],source_root=c['source_root'],source_sha=c['source_sha'],
        source_files=c['source_files'],bindings=c['bindings'],parent=c['parent'],python=c['python'],
        profile=task.PROFILE,kind=task.KIND,execution_site='HOPPER_SINGLE_GPU',
        pbs_server=c['pbs_server'],pbs_job_name='gr00t_task_sft',job_id=job_id,recipe=p,
        allocation_request_id=Path(c['root']).name,generation_id=c['generation_id'],
        start_step=start,resume=c.get('resume'),GPU_uuid=gpu['uuid'],GPU_name=gpu['name'],
        GR00T_updates=p['endpoint']-start,WM_updates=0,Actor_updates=0,TASK_ADAPTER_updates=0,
        physics=False,WBC_changes=False,parent_manifest_sha256=original.PARENT,
        role=c['role'],registry=c['registry'],deadline_epoch=deadline,closeout_reserve_s=1200,
        task_learning_admission=c['task_learning_admission'],project_storage_admission=c['project_storage_admission'])
    for key in('endpoint_decision','first_seed_learning_and_native_gain'):
        if key in c:s[key]=c[key]
    task.validate(s)
    return s


def activate(c):
    """No submission, requeue, physics or model training occurs in this helper."""
    from scripts.pi0292.durable import OwnerLock,write
    from scripts.pi0294.e1.journal import Journal
    from scripts.pi0294.multitask.shared_vanda import seconds
    configuration(c)
    root=Path(c['root']);need=original.need;lease_path=root/'control/LEASE.json'
    jid=os.environ.get('PBS_JOBID')
    need(bool(jid)and bool(c.get('pbs_server'))and root.resolve()==root
        and root.is_relative_to(Path(task.PROJECT_ROOT)),'EXPLICIT_ACTUAL_PROJECT_JOB_AND_SERVER')
    actual=job(jid,c['pbs_server'])
    allocated(actual,jid,uid=os.getuid(),hostname=os.uname().nodename,user=os.environ.get('USER'),
        expected_job_name='gr00t_task_sft')
    lease=json.loads(lease_path.read_text())
    need(lease.get('status')=='PBS_SUBMITTED'and lease.get('job_id')==jid
        and lease.get('config_sha256')==digest(root/'ALLOCATION_CONFIG.json'),
        'NEW_OWN_SUBMITTED_FORMAL_ALLOCATION')
    need(not(root/'WORKER_STARTED.private.json').exists()
        and not(root/'ALLOCATION_STARTED.private.json').exists(),'UNKNOWN_BOOTSTRAP_NEVER_REPLAY')
    remaining=seconds(actual['Resource_List']['walltime'])-seconds(actual.get('resources_used',{}).get('walltime','00:00:00'))
    deadline=min(c['absolute_deadline_epoch'],time.time()+remaining-120)
    need(time.time()+c['predicted_work_s']+1200<deadline,'MEASURED_FINITE_FORMAL_WINDOW_AND_CLOSEOUT')
    for name,pin in c['source_files'].items():
        need(digest(Path(c['source_root'])/name)==pin,'SEALED_FORMAL_SOURCE_'+name)
    from .target_gr00t_allocation_v1 import child_environment
    env=child_environment(c['source_root'],root)
    q=subprocess.run([c['python'],'-B','-c',
        'import torch,json;p=torch.cuda.get_device_properties(0);print(json.dumps(dict(count=torch.cuda.device_count(),name=p.name,uuid=str(p.uuid),major=p.major,total_memory=p.total_memory)))'],
        env=env,capture_output=True,text=True,timeout=90,check=True)
    gpu=json.loads(q.stdout)
    need(gpu['count']==1 and gpu['name']=='NVIDIA H200'and gpu['major']==9
        and gpu['total_memory']>=130*1024**3,'ACTUAL_SINGLE_QUALIFIED_H200')
    s=execution_spec(c,job_id=jid,gpu=gpu,deadline=deadline)
    registry=json.loads(Path(c['registry']).read_text())
    need(any(e.get('path')==str(root)and e.get('scope')==task.PROFILE
        and e.get('source_sha')==c['source_sha']for e in registry['entries']),
        'ORIGINAL_REGISTERED_FORMAL_JOB_OWNER')
    journal=Journal(root/'control/journal',root.name)
    def effect():
        alias=root/'model_inputs/nvidia/Cosmos-Reason2-2B'
        alias.parent.mkdir(parents=True,exist_ok=True)
        need(not alias.exists()and not alias.is_symlink(),'FRESH_PRIVATE_MODEL_ALIAS')
        alias.symlink_to(c['bindings']['cosmos'],target_is_directory=True)
        write(root/'ACTUAL_GPU.private.json',gpu)
        write(root/'EXECUTION_SPEC.json',s)
        current=json.loads(lease_path.read_text());need(current==lease,'ALLOCATION_LEASE_UNCHANGED')
        current.update(status='ACTIVE',spec_sha256=digest(root/'EXECUTION_SPEC.json'),deadline_epoch=deadline)
        write(lease_path,current)
        write(root/'ALLOCATION_STARTED.private.json',dict(job_id=jid,epoch=time.time(),deadline_epoch=deadline,
            actual_job=actual,source_sha=c['source_sha'],automatic_retry=False))
        return dict(spec_sha256=digest(root/'EXECUTION_SPEC.json'),job_id=jid)
    with closing(OwnerLock(c['ledger_lock'])):
        journal.once('ACTUAL_ALLOCATION_ACTIVATE',dict(config_sha256=digest(root/'ALLOCATION_CONFIG.json'),job_id=jid),effect)
    return s,env


def run(c):
    from scripts.pi0292.durable import OwnerLock,write
    root=Path(c['root']);started=time.monotonic()
    with closing(OwnerLock(root/'NEURAL_OWNER.lock'))as owner:
        try:
            s,env=activate(c)
            os.set_inheritable(owner.handle.fileno(),True)
            argv=[c['python'],'-B','-m','scripts.wm02.train_target_gr00t_v1','--spec',str(root/'EXECUTION_SPEC.json')]
            os.chdir(c['source_root'])
            os.execve(c['python'],argv,env)
        except BaseException:
            failure=traceback.format_exc()
            if not(root/'WORKER_STARTED.private.json').exists():
                write(root/'BOOTSTRAP_FAILURE.private.json',dict(failure=failure,GR00T_updates=0,new_policy_models=0))
                path=root/'control/LEASE.json';lease=json.loads(path.read_text())
                if lease.get('job_id')==os.environ.get('PBS_JOBID')and lease.get('status')in('PBS_SUBMITTED','ACTIVE'):
                    write(root/'RESULT.json',dict(status='FORMAL_BOOTSTRAP_FAILED_BEFORE_MODEL',
                        failure=failure,GR00T_updates=0,actual_model_loads=0,native_physics=0))
                    write(root/'WORKER_EXIT.private.json',dict(exit_code=1,wall_s=time.monotonic()-started,
                        failure=failure,job_id=os.environ.get('PBS_JOBID'),phase='BOOTSTRAP_BEFORE_MODEL'))
                    lease.update(status='CLOSED',worker_exit_code=1,closed_epoch=time.time(),
                        result_sha256=digest(root/'RESULT.json'))
                    write(path,lease)
            raise
