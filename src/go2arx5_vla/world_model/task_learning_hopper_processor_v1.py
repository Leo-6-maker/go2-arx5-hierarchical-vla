"""One allocated single-H200 job; Processor tensors remain on CPU.

PBS remains the scheduler. This entry submits nothing, loads no policy model,
and grants no training admission without the actual cross-site batch identities.
"""
import json,math,os,signal,time,traceback
from pathlib import Path
from contextlib import closing
from .target_gr00t_profile_v1 import need
from .task_learning_hopper_v1 import PROJECT,job
from .storage import digest

PROFILE='TASK_LEARNING_CONTINUOUS_PROCESSOR_H200_ALLOCATION_V1'
JOB_NAME='gr00t_task_proc'
ROOT=Path('/scratch/e1192548/Projects/GR00T-WBC')


def configuration(c):
    need(c.get('profile')==PROFILE and c.get('physics')is False
        and c.get('policy_models')==c.get('optimizer_updates')==0
        and type(c.get('pbs_server'))is str and bool(c['pbs_server'].strip())
        and c.get('reserved_gpus')==1 and c.get('ncpus')==12
        and c.get('mem_resource')=='225gb'
        and Path(c['root']).is_absolute()and Path(c['root']).is_relative_to(ROOT)
        and type(c.get('predicted_work_s'))in(int,float)and math.isfinite(c['predicted_work_s'])
        and c['predicted_work_s']>0 and c['predicted_work_s']+300<c['wall_s'],
        'FINITE_SITE_SUPPORTED_H200_ALLOCATION_CPU_PROCESSOR_NO_POLICY_OR_PHYSICS')


def allocated(record,c,*,uid,user,hostname):
    configuration(c);r=record['Resource_List']
    hosts={v.split('/')[0].split('.')[0]for v in record.get('exec_host','').split('+')}
    need(uid==22182 and user=='e1192548'
        and record['Job_Owner'].split('@')[0]==user
        and record['job_state']=='R'and record['project']==PROJECT
        and record['Job_Name']==JOB_NAME and int(r.get('ngpus',0))==1
        and 'ngpus=1'in r['select'].split(':')and 'gpu_model=H200'in r['select'].split(':')
        and int(r['nodect'])==1 and int(r['ncpus'])==c['ncpus']
        and r['select'].split(':')[0]=='1'
        and str(r['mem']).lower()==c['mem_resource'].lower()
        and hostname.split('.')[0]in hosts and not hostname.split('.')[0].startswith('hopper-l-'),
        'ACTUAL_OWN_PROJECT_SINGLE_H200_ALLOCATION_WITH_DECLARED_CPU_AND_MEMORY')


def inputs(c):
    from .stable_task_scene_v1 import pinned
    from .task_learning_transfer_v1 import inventory
    root=Path(c['dataset'])
    need(root.is_relative_to(ROOT)and root.resolve()==root,'OWN_CANONICAL_PRIVATE_CONTINUOUS_DATA')
    transfer=pinned(c['data_transfer']);actual=inventory(root)
    need(transfer.get('status')=='PASS_PRIVATE_TASK_LEARNING_DATA_TRANSFER'
        and transfer.get('manifest_sha256')==actual['manifest_sha256']==c['dataset_sha256']
        and transfer.get('statistics_sha256')==actual['statistics_sha256']==c['statistics_sha256']
        and transfer.get('files')==actual['files'],
        'ACTUAL_CORE_EXPORT_PROCESSOR_AND_IDENTICAL_PRIVATE_TRANSFER')
    need(not(root/'PROCESSOR_HOPPER_RECONSTRUCTION.json').exists()
        and not(root/'target-processor-hopper-v1').exists(),
        'UNKNOWN_OR_EXISTING_PROCESSOR_ATTEMPT_NEVER_REPLAYED')
    return actual


def run(c):
    from scripts.pi0292.durable import OwnerLock,write
    from scripts.pi0294.e1.journal import Journal
    from scripts.pi0294.multitask.shared_vanda import seconds
    root=Path(c['root']);lp=root/'control/LEASE.json';cp=root/'ALLOCATION_CONFIG.json'
    jid=os.environ.get('PBS_JOBID');configuration(c)
    need(root.resolve()==root and root.stat().st_uid==os.getuid()==22182 and bool(jid),
        'OWN_ACTUAL_CPU_WORKER_ROOT_AND_PBS_ID')
    with closing(OwnerLock(root/'OWNER.lock')):
        lease=json.loads(lp.read_bytes())
        need(lease.get('status')=='PBS_SUBMITTED'and lease.get('job_id')==jid
            and lease.get('config_sha256')==digest(cp)
            and json.loads(cp.read_bytes())==c,'EXACT_FRESH_SUBMITTED_CPU_WORKER')
        need(not(root/'WORKER_STARTED.private.json').exists(),'NEVER_REPLAY_EXISTING_CPU_WORKER')
        start,cpu=time.monotonic(),time.process_time();code=1;failure=None;result={};handlers={}
        try:
            actual=job(jid,c['pbs_server'])
            allocated(actual,c,uid=os.getuid(),user=os.environ.get('USER'),hostname=os.uname().nodename)
            remaining=seconds(actual['Resource_List']['walltime'])-seconds(actual.get('resources_used',{}).get('walltime','00:00:00'))
            deadline=min(c['absolute_deadline_epoch'],time.time()+remaining-120)
            need(time.time()+c['predicted_work_s']+120<deadline,'FINITE_CPU_PROCESSOR_WORK_AND_CLOSEOUT_WINDOW')
            reg=json.loads(Path(c['registry']).read_bytes())
            need(any(e.get('path')==str(root)and e.get('scope')==PROFILE
                and e.get('source_sha')==c['source_sha']and e.get('local_lease')==str(lp)
                for e in reg['entries']),'ACTUAL_EXISTING_CPU_WORKER_REGISTRY')
            for name,pin in c['source_files'].items():
                need(digest(Path(c['source_root'])/name)==pin,'SEALED_CPU_PROCESSOR_SOURCE_'+name)
            original=inputs(c)
            def interrupted(sig,frame):raise RuntimeError('FINITE_CPU_PROCESSOR_STOP_'+str(sig))
            handlers={s:signal.signal(s,interrupted)for s in(signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
            signal.alarm(max(1,int(deadline-time.time())))
            # PBS owns the device mask, including for this CPU-only data phase.
            pbs_device_mask=os.environ.get('CUDA_VISIBLE_DEVICES')
            write(root/'WORKER_STARTED.private.json',dict(job_id=jid,epoch=time.time(),actual_job=actual,
                source_sha=c['source_sha'],dataset_sha256=original['manifest_sha256'],deadline_epoch=deadline))
            current=dict(lease,status='ACTIVE',deadline_epoch=deadline);write(lp,current)
            journal=Journal(root/'control/journal',root.name)
            def qualify():
                import torch
                torch.set_num_threads(c['ncpus'])
                need(not torch.cuda.is_initialized(),'PROCESSOR_CPU_ONLY_BEFORE_RECONSTRUCTION')
                from scripts.wm02.qualify_target_gr00t_processor_v1 import qualify as original_qualify
                value=original_qualify(c['dataset'],c['bindings']['source'],c['bindings']['cosmos'],
                    verify_transfer=True,transfer_site='HOPPER_SINGLE_GPU',dataset_family='TASK_LEARNING_CONTINUOUS_V1')
                need(not torch.cuda.is_initialized(),'PROCESSOR_CPU_ONLY_AFTER_RECONSTRUCTION')
                need(os.environ.get('CUDA_VISIBLE_DEVICES')==pbs_device_mask,'PBS_DEVICE_MASK_UNCHANGED')
                return {k:v for k,v in value.items()if k!='records'}
            checked=journal.once('ORIGINAL_PROCESSOR_RECONSTRUCTION',dict(config_sha256=digest(cp)),qualify)
            result.update(status='PASS_ACTUAL_HOPPER_CONTINUOUS_PROCESSOR_RECONSTRUCTION',
                Processor=checked,manifest_sha256=checked['manifest_sha256'],statistics_sha256=checked['statistics_sha256'],
                Processor_admission_sha256=digest(Path(c['dataset'])/'PROCESSOR_HOPPER_RECONSTRUCTION.json'),
                CUDA_initialized=False)
            code=0
        except BaseException:failure=traceback.format_exc()
        finally:
            if handlers:signal.alarm(0)
            result.update(exit_code=code,failure=failure,job_id=jid,source_sha=c['source_sha'],config_sha256=digest(cp),
                wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu,
                new_policy_models=0,new_optimizer_updates=0,new_native_integrations=0,GPU_allocations=1,
                reserved_GPU_wall_s=time.monotonic()-start)
            if code:result['status']='HOPPER_PROCESSOR_FAILED_PRESERVED'
            write(root/'RESULT.private.json',result)
            write(root/'WORKER_EXIT.private.json',dict(exit_code=code,failure=failure,job_id=jid,
                wall_s=result['wall_s'],CPU_s=result['CPU_s']))
            current=json.loads(lp.read_bytes())
            need(current.get('job_id')==jid and current.get('config_sha256')==digest(cp)
                and current.get('status')in('PBS_SUBMITTED','ACTIVE'),'OWN_UNCHANGED_CPU_WORKER_LEASE')
            current.update(status='CLOSED',worker_exit_code=code,closed_epoch=time.time(),
                result_sha256=digest(root/'RESULT.private.json'));write(lp,current)
            for sig,handler in handlers.items():signal.signal(sig,handler)
        return code
