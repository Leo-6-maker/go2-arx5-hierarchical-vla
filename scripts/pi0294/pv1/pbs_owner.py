"""One native PV1 submission chain across actually eligible Hopper projects."""
import argparse
import ast
import fcntl
from datetime import datetime,timedelta,timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import uuid

from scripts.pi0292.allocation_admission import PBS,owned_jobs,personal_quota
from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0294.e0_scope import pbs_epoch
from scripts.pi0294.e1.pbs_owner import native
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.mp1.pbs_owner import native_queue_limit
from .allocation import PROJECT_SOURCE,native_request_binding
from .auth_lifecycle import context as auth_context
from .decision import AUTHORITY
from .neural_ledger import Ledger
from .neural_scope import validate
from .resources import admit_resources,project_eligible
from .bank_reuse import preflight as bank_preflight
from .model_inputs import preflight as model_preflight

PAD=60
AMGR='/opt/am/python/bin/amgr'


def project(name,*,query=subprocess.run):
    if not re.fullmatch(r'CFP[0-9]{2}-(?:CF|SF)-[0-9]{3}',name):
        raise PermissionError('PV1_ONLY_EXACT_EXISTING_NATIVE_PROJECT_NAMES')
    def call(arguments):
        p=query([AMGR]+arguments,capture_output=True,text=True,timeout=30)
        if p.returncode or '15006' in p.stdout+p.stderr:
            raise PermissionError('PV1_PROJECT_AUTH_EXPIRED_15006' if '15006' in p.stdout+p.stderr else 'PV1_PROJECT_QUERY_FAILED')
        return p.stdout
    values=json.loads(call(['ls','project','--name',name,'--json-info']))
    balance=ast.literal_eval(call(['checkbalance','project','--name',name]))
    if len(values)!=1:raise PermissionError('PV1_ONE_NATIVE_PROJECT_RECORD')
    p=values[0];day=datetime.now(timezone(timedelta(hours=8))).date().isoformat();user=os.environ['USER']
    if (p['name']!=name or user not in p['users'] or 'hopper-m-02' not in p['cluster']
            or not project_eligible(dict(p,member=user in p['users'],balance_su=float(balance['su'])),day)):
        raise PermissionError('PV1_FRESH_PROJECT_ACTIVE_DATE_MEMBERSHIP_CLUSTER_AND_BALANCE')
    return dict(source=PROJECT_SOURCE,project=p,balance_su=float(balance['su']),observed_epoch=time.time())


def active_items(value):
    return [s for s in value['PBS_submissions'] if s['status'] not in ('NATIVE_TERMINAL','NATIVE_QSUB_REJECTED_NO_JOB')]


def dependency_for(value):
    active=active_items(value)
    if not active:return None
    if any(not s.get('job_id') for s in active):
        raise PermissionError('PV1_UNKNOWN_PRIOR_QSUB_INTENT_RECONCILE_BEFORE_NEW_SUBMISSION')
    return active[-1]['job_id']


def native_chain(value):
    registered={s['job_id']:s for s in value['PBS_submissions'] if s.get('job_id')}
    jobs={k:j for k,j in owned_jobs(os.environ['USER']).items() if j.get('Job_Name','').startswith('pi0294_pv1_')}
    if set(jobs)-set(registered) or sum(j['job_state']=='R' for j in jobs.values())>1:
        raise PermissionError('PV1_UNREGISTERED_OR_MULTIPLE_NATIVE_PV1_JOBS_NO_NEW_QSUB')
    for key,job in jobs.items():
        item=registered[key]
        if job['project']!=item['project'] or job['Job_Name']!=item['job_name']:
            raise PermissionError('PV1_REGISTERED_NATIVE_JOB_PROJECT_OR_NAME_CHANGED')
    return jobs


def budget_request(value,request):
    if dependency_for(value)!=request.get('dependency_afterok'):
        raise PermissionError('PV1_EXACT_LAST_OWN_JOB_DEPENDENCY_FOR_ONE_GPU_CHAIN')
    if value['unknown_effects']:raise PermissionError('PV1_UNKNOWN_NATIVE_EFFECT_NO_NEW_SUBMISSION')
    final=request['stage']=='P5_FINAL'
    admit_resources(value['used'],value['reserved'],dict(H200_s=request['walltime_s']+PAD,PBS_submissions=1),final=final)
    extra=request['additional_output_upper_bytes'];reserve=0 if final else 40*1024**3
    reserved=sum(s['additional_output_upper_bytes'] for s in active_items(value))
    peak=value['new_storage_peak_bytes'];current=value.get('new_storage_current_bytes',peak)
    if (type(peak) is not int or not 0<=peak<=250*1024**3 or type(current) is not int or not 0<=current<=peak
            or current+reserved+extra+reserve>250*1024**3):
        raise PermissionError('PV1_ALL_QUEUED_OUTPUT_UPPERS_AND_FINAL_STORAGE_RESERVE')
    if request['stage'] in ('P2_PRIMARY','P4_REPLICA'):
        block=request['training_block'];replica=request['stage']=='P4_REPLICA'
        if block.get('replica',False) is not replica:
            raise PermissionError('PV1_EXACT_NATIVE_PRIMARY_OR_REPLICA_RUN_IDENTITY')
        key=('REPLICA_' if replica else '')+block['arm']
        if any(s.get('training_run_key')==key for s in active_items(value)):
            raise PermissionError('PV1_NO_DUPLICATE_QUEUED_OR_RUNNING_TRAINING_BLOCK')
        progress=value.get('training_runs',{}).get(key,{}).get('last_native_step',0)
        if progress!=request['expected_native_start_step']:
            raise PermissionError('PV1_MATCH_ACTUAL_CHARGED_PROGRESS_BEFORE_QUEUEING_RESUME')
    elif any(s.get('evaluation_candidate')==request.get('evaluation_candidate')
            for s in active_items(value) if s.get('evaluation_candidate') is not None):
        raise PermissionError('PV1_NO_DUPLICATE_QUEUED_CHECKPOINT_EVALUATIONS')


def prepare(root,*,stage,project_name,walltime_s=14400,additional_output_upper_bytes=80*1024**3,memory_gb=128):
    if os.environ.get('PBS_JOBID'):raise PermissionError('PV1_QSUB_OWNER_ON_LOGIN_ONLY')
    root=Path(root);m,bindings,inputs,bound,a=validate(root)
    cold_bank=bank_preflight(root,m,inputs,bound)
    cold_model=model_preflight(root,bindings)
    ledger=Ledger(root,final=True) if stage=='P5_FINAL' else Ledger(root)
    if stage not in ('B0_DEV','P2_PRIMARY','P4_DEV','P4_REPLICA','P5_FINAL') or memory_gb not in (128,225):
        raise PermissionError('PV1_ONLY_IMPLEMENTED_PHASES_AND_MATURE_MEMORY_PROFILES')
    if not a['T0_epoch']<=time.time()<a['stop_new_jobs_epoch']:
        raise PermissionError('PV1_ORIGINAL_NEW_SUBMISSION_CUTOFF')
    proof=project(project_name);context=auth_context()
    proof['authentication_context_fingerprint']=context['fingerprint']
    q=personal_quota(os.environ['USER']);queue=native_queue_limit()
    queue['source']='NATIVE_QSTAT_EFFECTIVE_QUEUE_LIMITS'
    if (type(walltime_s) is not int or not 2760<walltime_s<=min(8*3600,queue['maximum_wall_s'])
            or type(additional_output_upper_bytes) is not int or not 0<additional_output_upper_bytes<=210*1024**3
            or q['hard_bytes']-q['used_bytes']<additional_output_upper_bytes+(0 if stage=='P5_FINAL' else 40*1024**3)
            or not 0<=time.time()-q['report_epoch']<=900):
        raise PermissionError('PV1_REAL_NATIVE_WALLTIME_PERSONAL_QUOTA_AND_OUTPUT_HEADROOM')
    value=read(ledger.path);native_chain(value);number=len(value['PBS_submissions'])+1
    future_storage=sum(s['additional_output_upper_bytes'] for s in active_items(value))+additional_output_upper_bytes
    if q['hard_bytes']-q['used_bytes']<future_storage+(0 if stage=='P5_FINAL' else 40*1024**3):
        raise PermissionError('PV1_PERSONAL_QUOTA_FOR_ALL_QUEUED_OUTPUT_UPPERS')
    if number>24:raise PermissionError('PV1_TOTAL_24_PBS_SUBMISSIONS_ACROSS_ALL_PROJECTS')
    directory=root/'pbs'/('alloc-'+str(number).zfill(2));directory.mkdir(mode=0o700)
    write(directory/'NATIVE_QUEUE_LIMIT.json',queue);(directory/'NATIVE_QUEUE_LIMIT.json').chmod(0o400)
    write(directory/'PREQUEUE_PROJECT.private.json',proof)
    names=set(m['execution']['sealed_inputs'])|{'NEURAL_BINDINGS.private.json'}
    queue_name=str((directory/'NATIVE_QUEUE_LIMIT.json').relative_to(root));names.add(queue_name)
    r=dict(schema='PI0294_PV1_NATIVE_ALLOCATION_REQUEST_V1',authority_release=AUTHORITY,
        run_uuid=root.name,campaign_run_uuid=a['run_uuid'],source_sha=m['code']['vla_commit'],
        run_manifest_sha256=sha(root/'RUN_MANIFEST.json'),stage=stage,submit_count=number,
        request_id=str(uuid.uuid4()),submission_nonce=str(uuid.uuid4()),project=project_name,
        authentication_context_fingerprint=context['fingerprint'],authentication_principal=context['principal'],
        job_name='pi0294_pv1_'+str(number).zfill(2),created_epoch=time.time(),walltime_s=walltime_s,ncpus=12,
        numerical_mode='DETERMINISTIC_CUDA_MATH_SDP_V1',data_bank_status='SEALED_FULL_SUCCESS_OFFICIAL_CODEC_ADMITTED',
        new_bank_manifest_sha256=bound['new_bank_manifest_sha256'],cold_bank_preflight=cold_bank,
        cold_model_input_preflight=cold_model,
        additional_output_upper_bytes=additional_output_upper_bytes,
        queue_limit_relative=queue_name,queue_limit_sha256=sha(root/queue_name),
        sealed_inputs={n:sha(root/n) for n in sorted(names)},python=bindings['python'],
        dependency_afterok=dependency_for(value))
    admission_version=m.get('authority',{}).get('training_admission_version')
    if admission_version is not None:r['scientific_admission_version']=admission_version
    if stage in ('P2_PRIMARY','P4_REPLICA'):
        block=read(root/'plans/PRIMARY_TRAIN_BLOCK.json');r['training_block']=block
        p=block.get('resume_checkpoint_relative')
        r['expected_native_start_step']=read(root/p/'CHECKPOINT_COMPLETE.json')['global_step'] if p else 0
    elif stage=='P5_FINAL':r['evaluation_candidate']=read(root/'plans/PV1_P5_LOCKBOX_PLAN.json')['candidate_freeze']['candidate']
    else:r['evaluation_candidate']=read(root/'plans/SOURCE_EVALUATION_PLAN.json')['candidate']
    from .neural import stage_work
    stage_work(root,r);budget_request(value,r)
    from .future_admission import prequeue
    future=None if stage=='P5_FINAL' else prequeue(root,r,'PREPARE')
    if future is not None:write(directory/'FUTURE_BUDGET_PREPARE.json',future)
    write(directory/'REQUEST.json',r);(directory/'REQUEST.json').chmod(0o400)
    code=root/m['code']['capsule_relative'];hours,remainder=divmod(walltime_s,3600);minutes,seconds=divmod(remainder,60)
    lines=['#!/bin/bash -l','#PBS -N '+r['job_name'],'#PBS -P '+project_name,'#PBS -q auto',
        '#PBS -r n','#PBS -v PV1_REQUEST_ID='+r['request_id']+',PV1_SUBMISSION_NONCE='+r['submission_nonce'],
        f'#PBS -l select=1:ncpus=12:mem={memory_gb}gb:ngpus=1:gpu_model=H200',
        f'#PBS -l walltime={hours:02d}:{minutes:02d}:{seconds:02d}','#PBS -m n','#PBS -j oe',
        '#PBS -o '+str(directory/'pbs.log'),'set -eu','export PYTHONDONTWRITEBYTECODE=1',
        'export PYTHONPATH='+shlex.quote(str(code)+':'+str(code/'src')),
        'export PV1_RUN_MANIFEST='+shlex.quote(str(code.parent/'WORKER_RUN_MANIFEST.json')),
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1',
        'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1',
        'cd '+shlex.quote(str(code)),
        'exec '+shlex.join([bindings['python'],'-m','scripts.pi0294.pv1.launcher','--root',str(root),'--request',str(directory/'REQUEST.json')])]
    (directory/'job.pbs').write_text('\n'.join(lines)+'\n');(directory/'job.pbs').chmod(0o400)
    write(directory/'JOB_SCRIPT_BINDING.json',dict(request_sha256=sha(directory/'REQUEST.json'),job_script_sha256=sha(directory/'job.pbs')))
    (directory/'JOB_SCRIPT_BINDING.json').chmod(0o400)
    return dict(status='PV1_PREPARED_NOT_SUBMITTED',submit_count=number,request_id=r['request_id'],
        walltime_s=walltime_s,dependency_afterok=r['dependency_afterok'],project=project_name)


def submit(root,number):
    root=Path(root);m,bindings,inputs,bound,_=validate(root)
    cold_bank=bank_preflight(root,m,inputs,bound)
    cold_model=model_preflight(root,bindings)
    lock=OwnerLock(root/'PV1_QSUB_OWNER.lock')
    directory=root/'pbs'/('alloc-'+str(number).zfill(2));r=read(directory/'REQUEST.json')
    ledger=Ledger(root,final=True) if r['stage']=='P5_FINAL' else Ledger(root)
    binding=read(directory/'JOB_SCRIPT_BINDING.json')
    try:
        if cold_bank!=r['cold_bank_preflight']:
            raise PermissionError('PV1_PREQUEUE_BANK_BINDING_CHANGED_AFTER_PREPARE')
        if cold_model!=r['cold_model_input_preflight']:
            raise PermissionError('PV1_PREQUEUE_MODEL_ALIAS_CHANGED_AFTER_PREPARE')
        if (sha(directory/'REQUEST.json')!=binding['request_sha256'] or sha(directory/'job.pbs')!=binding['job_script_sha256']
                or time.time()-r['created_epoch']>=120):
            raise PermissionError('PV1_FRESH_IMMUTABLE_PREPARED_REQUEST_AND_NATIVE_QUEUE_LIMIT')
        native_chain(read(ledger.path))
        with ledger.transaction() as value:
            if number!=len(value['PBS_submissions'])+1 or number!=r['submit_count']:
                raise PermissionError('PV1_ONE_NEXT_GLOBAL_SUBMISSION_INTENT_NO_RETRY')
            budget_request(value,r)
            from .future_admission import prequeue
            future=None if r['stage']=='P5_FINAL' else prequeue(root,r,'SUBMIT',request_bytes=(directory/'REQUEST.json').read_bytes())
            if future is not None:write(directory/'FUTURE_BUDGET_SUBMIT.json',future)
            item=dict(status='QSUB_INTENT_OUTCOME_UNKNOWN_NO_RETRY',request_id=r['request_id'],run_uuid=root.name,
                project=r['project'],job_name=r['job_name'],stage=r['stage'],request_sha256=sha(directory/'REQUEST.json'),
                reserved_H200_s=r['walltime_s'],accounting_pad_s=PAD,additional_output_upper_bytes=r['additional_output_upper_bytes'],
                dependency_afterok=r['dependency_afterok'],started_epoch=time.time())
            if r['stage'] in ('P2_PRIMARY','P4_REPLICA'):
                item['training_run_key']=('REPLICA_' if r['stage']=='P4_REPLICA' else '')+r['training_block']['arm']
            else:item['evaluation_candidate']=r['evaluation_candidate']
            value['PBS_submissions'].append(item);value['used']['PBS_submissions']=len(value['PBS_submissions'])
            value['reserved']['H200_s']=value['reserved'].get('H200_s',0)+r['walltime_s']+PAD
            value['storage_reserved_bytes']=sum(s['additional_output_upper_bytes'] for s in active_items(value))
        command=[PBS+'qsub']
        if r['dependency_afterok']:command+=['-W','depend=afterok:'+r['dependency_afterok']]
        command.append(str(directory/'job.pbs'))
        result=subprocess.run(command,capture_output=True,text=True,timeout=45)
        write(directory/'QSUB_NATIVE.private.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        job=result.stdout.strip()
        with ledger.transaction() as value:
            item=next(s for s in value['PBS_submissions'] if s['request_id']==r['request_id'])
            if item['status']!='QSUB_INTENT_OUTCOME_UNKNOWN_NO_RETRY':raise PermissionError('PV1_SUBMISSION_INTENT_CHANGED')
            if re.fullmatch(r'[0-9]+\.hopper-m-[0-9]+',job):
                item.update(status='SUBMITTED_PENDING_NATIVE',job_id=job,returned_epoch=time.time(),native_returncode=result.returncode)
            else:item.update(native_returncode=result.returncode,reply_job_id_unknown=True)
        if not re.fullmatch(r'[0-9]+\.hopper-m-[0-9]+',job):
            raise RuntimeError('PV1_UNKNOWN_QSUB_REPLY_PRESERVE_FULL_RESERVATION_AND_RECONCILE_NO_RESUBMIT')
        return dict(status='PV1_SUBMITTED_ONCE',job_id=job,submit_count=number,dependency_afterok=r['dependency_afterok'])
    finally:lock.close()


def publish_release_record(path,value):
    """Publish an already-0400 inode, exclusively; never expose a writable START."""
    path=Path(path);temporary=path.with_name('.'+path.name+'.new-'+uuid.uuid4().hex)
    try:
        with temporary.open('xb') as handle:
            handle.write((json.dumps(value,indent=2,allow_nan=False)+'\n').encode())
            handle.flush();os.fchmod(handle.fileno(),0o400);os.fsync(handle.fileno())
        os.link(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()
    directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)


def release(root,number):
    root=Path(root);validate(root);directory=root/'pbs'/('alloc-'+str(number).zfill(2))
    if directory.is_symlink() or not directory.resolve().is_relative_to(root.resolve()):
        raise PermissionError('PV1_RELEASE_DIRECTORY_MUST_REMAIN_IN_REGISTERED_ROOT')
    r=read(directory/'REQUEST.json')
    ready=read(directory/'PBS_READY.json');job=native(ready['job_id'])
    ledger=Ledger(root,final=True) if r['stage']=='P5_FINAL' else Ledger(root)
    item=next(s for s in read(ledger.path)['PBS_submissions'] if s['request_id']==r['request_id'])
    if (item['job_id']!=ready['job_id'] or job['job_state']!='R' or job['Job_Owner'].split('@')[0]!=os.environ['USER']
            or job['project']!=r['project'] or job['Job_Name']!=r['job_name']
            or not native_request_binding(job,r)
            or ready['request_sha256']!=sha(directory/'REQUEST.json')):
        raise PermissionError('PV1_EXACT_OWN_RUNNING_JOB_SINGLE_FRESH_RELEASE')
    lock=OwnerLock(directory/'PV1_RELEASE_OWNER.lock');attempt=None;failure=None
    try:
        proof=directory/'RUNNING_PROJECT_ATTESTATION.json';start=directory/'START_NEURAL.json'
        def sealed(path):
            if path.is_symlink() or not path.is_file() or path.stat().st_uid!=os.getuid() or path.stat().st_mode&0o777!=0o400:
                raise PermissionError('PV1_PRESERVE_EXACT_OWN_SEALED_RELEASE_RECORD')
            return read(path)
        def original_window():
            if not 0<=time.time()-ready['observed_epoch']<175:
                raise PermissionError('PV1_ORIGINAL_175_SECOND_FIRST_RELEASE_WINDOW')
        def bound_proof(value):
            if (value.get('schema')!='PI0294_PV1_RUNNING_JOB_PROJECT_ATTESTATION_V1'
                    or value.get('source')!=PROJECT_SOURCE or value.get('run_uuid')!=root.name
                    or value.get('request_id')!=r['request_id'] or value.get('job_id')!=ready['job_id']
                    or value.get('submission_nonce')!=r['submission_nonce']
                    or value.get('authentication_context_fingerprint')!=r.get('authentication_context_fingerprint')
                    or value['project']['name']!=r['project']
                    or not ready['observed_epoch']<=value['observed_epoch']<ready['observed_epoch']+175):
                raise PermissionError('PV1_EXISTING_RELEASE_PROOF_EXACT_JOB_NONCE_CONTEXT_AND_TIME')
        if start.exists():
            existing=sealed(start);attestation=sealed(proof);bound_proof(attestation)
            if (existing['run_uuid']!=root.name or existing['job_id']!=ready['job_id']
                    or existing['request_sha256']!=sha(directory/'REQUEST.json')
                    or existing['submission_nonce']!=r['submission_nonce']
                    or existing['project_attestation_sha256']!=sha(proof)
                    or not attestation['observed_epoch']<=existing['observed_epoch']<ready['observed_epoch']+175):
                raise PermissionError('PV1_EXISTING_START_NOT_AN_EXACT_ORIGINAL_RELEASE')
            return dict(status='PV1_ALREADY_RELEASED_EXACT_NO_SECOND_WRITE',job_id=ready['job_id'],project=r['project'])
        original_window();context=auth_context()
        if (context['fingerprint']!=r.get('authentication_context_fingerprint')
                or context['principal']!=r.get('authentication_principal') or context['principal']!=os.environ['USER']):
            raise PermissionError('PV1_SUBMIT_HELPER_RELEASE_AUTH_CONTEXT_MISMATCH')
        recovered=proof.exists()
        if recovered:bound_proof(sealed(proof))
        attempts=directory/'release_attempts'
        if attempts.is_symlink():raise PermissionError('PV1_RELEASE_JOURNAL_MUST_BE_IN_REGISTERED_DIRECTORY')
        attempts.mkdir(exist_ok=True,mode=0o700)
        attempt=attempts/(uuid.uuid4().hex+'.json')
        publish_release_record(attempt,dict(schema='PV1_JOB_BOUND_RELEASE_ATTEMPT_V1',run_uuid=root.name,
            request_sha256=sha(directory/'REQUEST.json'),request_id=r['request_id'],submission_nonce=r['submission_nonce'],
            job_id=ready['job_id'],hostname=os.uname().nodename,pid=os.getpid(),
            proc_start=Path('/proc/self/stat').read_text().split()[21],observed_epoch=time.time(),
            authentication_context_fingerprint=context['fingerprint'],models_loaded_by_control=0))
        # This is always a new site query after READY, even on partial-proof recovery.
        def release_query(arguments,**kwargs):
            remaining=175-(time.time()-ready['observed_epoch'])
            upper=min(10.,(remaining-5)/2)
            if upper<=0:raise TimeoutError('PV1_NO_PROJECT_QUERY_PAST_ORIGINAL_RELEASE_WINDOW')
            kwargs['timeout']=min(kwargs.get('timeout',30),upper)
            return subprocess.run(arguments,**kwargs)
        value=project(r['project'],query=release_query);original_window()
        if (auth_context()['fingerprint']!=context['fingerprint']
                or not ready['observed_epoch']<=value['observed_epoch']<=time.time()
                or not 0<=time.time()-value['observed_epoch']<=30):
            raise PermissionError('PV1_FRESH_RUNNING_PROJECT_AND_UNCHANGED_AUTH_CONTEXT')
        if not recovered:
            value.update(schema='PI0294_PV1_RUNNING_JOB_PROJECT_ATTESTATION_V1',run_uuid=root.name,
                request_id=r['request_id'],job_id=ready['job_id'],submission_nonce=r['submission_nonce'],
                authentication_context_fingerprint=context['fingerprint'])
            publish_release_record(proof,value)
        original_window()
        publish_release_record(start,dict(run_uuid=root.name,job_id=ready['job_id'],request_sha256=sha(directory/'REQUEST.json'),
            submission_nonce=r['submission_nonce'],project_attestation_sha256=sha(proof),observed_epoch=time.time()))
        return dict(status='PV1_CURRENT_PROJECT_RELEASED_REQUEST_UNCHANGED',job_id=ready['job_id'],project=r['project'],
                    recovered_existing_project_attestation=recovered)
    except BaseException as exc:
        failure=type(exc).__name__;raise
    finally:
        try:
            if attempt is not None and attempt.exists():
                publish_release_record(attempt.with_suffix('.FINAL.json'),dict(
                    attempt_sha256=sha(attempt),failure_type=failure,completed_epoch=time.time(),
                    start_sha256=sha(directory/'START_NEURAL.json') if (directory/'START_NEURAL.json').is_file() else None,
                    proof_sha256=sha(directory/'RUNNING_PROJECT_ATTESTATION.json') if (directory/'RUNNING_PROJECT_ATTESTATION.json').is_file() else None,
                    reconcile_exact_files_before_any_retry=True))
        finally:lock.close()


def release_readback(root,number):
    """Read exact publication and live PID evidence after an uncertain control reply."""
    root=Path(root);validate(root,closeout=True);d=root/'pbs'/('alloc-'+str(number).zfill(2))
    if d.is_symlink() or not d.resolve().is_relative_to(root.resolve()):
        raise PermissionError('PV1_RELEASE_READBACK_DIRECTORY_ESCAPE')
    request=read(d/'REQUEST.json');ready=read(d/'PBS_READY.json');entries=[];active=[];locked=False
    lock_path=d/'PV1_RELEASE_OWNER.lock'
    if lock_path.exists():
        with lock_path.open('r+') as handle:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:locked=True
    for p in (d/'release_attempts').glob('*.json'):
        if p.name.endswith('.FINAL.json'):continue
        if p.is_symlink() or p.stat().st_uid!=os.getuid() or p.stat().st_mode&0o777!=0o400:
            raise PermissionError('PV1_RELEASE_JOURNAL_UNSEALED')
        v=read(p)
        if (v['run_uuid']!=root.name or v['request_sha256']!=sha(d/'REQUEST.json')
                or v['request_id']!=request['request_id'] or v['submission_nonce']!=request['submission_nonce']
                or v['job_id']!=ready['job_id'] or v['hostname']!=os.uname().nodename):
            raise PermissionError('PV1_RELEASE_READBACK_CHANGED_JOB_OR_HOST_NO_REPLAY')
        q=Path('/proc',str(v['pid']),'stat');fields=q.read_text().split() if q.exists() else None
        live=bool(fields and fields[2]!='Z' and fields[21]==str(v['proc_start']))
        final=p.with_suffix('.FINAL.json')
        if final.exists() and (final.is_symlink() or final.stat().st_uid!=os.getuid()
                or final.stat().st_mode&0o777!=0o400 or read(final)['attempt_sha256']!=sha(p)):
            raise PermissionError('PV1_RELEASE_FINAL_JOURNAL_INTEGRITY_STOP')
        if live and not final.exists():active.append(dict(pid=v['pid'],proc_start=v['proc_start']))
        entries.append(dict(sha256=sha(p),final_present=final.exists()))
    publications={}
    for name in ('RUNNING_PROJECT_ATTESTATION.json','START_NEURAL.json'):
        p=d/name
        if p.exists():
            if p.is_symlink() or p.stat().st_uid!=os.getuid() or p.stat().st_mode&0o777!=0o400:
                raise PermissionError('PV1_RELEASE_READBACK_PUBLICATION_UNSEALED')
            publications[name]=dict(sha256=sha(p),record=read(p))
    if 'START_NEURAL.json' in publications and 'RUNNING_PROJECT_ATTESTATION.json' not in publications:
        raise PermissionError('PV1_RELEASE_START_WITHOUT_PROOF_INTEGRITY_STOP')
    status=('LIVE_RELEASE_PROCESS_DO_NOT_REPLAY' if active else 'UNKNOWN_INFLIGHT_RELEASE_LOCK_NO_REPLAY' if locked
            else 'TERMINAL_RELEASE_WRITER_EXACT_FILES_FOR_RECONCILIATION')
    return dict(status=status,
                job_id=ready['job_id'],active=active,attempts=entries,publications=publications,
                new_release=False,new_model_loads=0)


def settle(root):
    root=Path(root);validate(root,closeout=True);ledger=Ledger(root,final=True);rows=[]
    snapshot=read(ledger.path);facts={}
    for item in active_items(snapshot):
        if not item.get('job_id'):raise PermissionError('PV1_UNKNOWN_QSUB_INTENT_NOT_A_ZERO_COST_JOB')
        facts[item['request_id']]=native(item['job_id'])
    allocated=None
    if any(j['job_state']=='F' for j in facts.values()):
        allocated=int(subprocess.check_output(['du','-s','-B1',str(root.parent.parent)],text=True,timeout=180).split()[0])
    with ledger.transaction() as value:
        for item in value['PBS_submissions']:
            if item['status'] in ('NATIVE_TERMINAL','NATIVE_QSUB_REJECTED_NO_JOB'):continue
            if not item.get('job_id'):raise PermissionError('PV1_UNKNOWN_QSUB_INTENT_NOT_A_ZERO_COST_JOB')
            job=facts.get(item['request_id'])
            if job is None:raise PermissionError('PV1_SETTLEMENT_SNAPSHOT_CHANGED_READ_ONLY_REFRESH_REQUIRED')
            if job['Job_Owner'].split('@')[0]!=os.environ['USER'] or job['project']!=item['project'] or job['Job_Name']!=item['job_name']:
                raise PermissionError('PV1_SETTLE_ONLY_EXACT_OWN_REGISTERED_JOB')
            if job['job_state']!='F':continue
            wall=sum(int(x)*scale for x,scale in zip(job.get('resources_used',{}).get('walltime','00:00:00').split(':'),(3600,60,1)))
            if job.get('stime'):
                if not job.get('obittime') or int(job['Resource_List']['ngpus'])!=1:
                    raise PermissionError('PV1_NATIVE_TERMINAL_TIMESTAMPS_AND_ONE_GPU')
                upper=max(wall,pbs_epoch(job['obittime'])-pbs_epoch(job['stime']))
            else:
                if job.get('exec_host') or job.get('run_count',0) or wall:raise PermissionError('PV1_UNOBSERVED_ALLOCATION_CANNOT_BE_FREE')
                upper=0
            item.update(status='NATIVE_TERMINAL',native_exit_status=job.get('Exit_status'),native_allocated_wall_s=wall,
                conservative_H200_s=upper,settled_epoch=time.time())
            value['used']['H200_s']=value['used'].get('H200_s',0)+upper
            value['reserved']['H200_s']-=item['reserved_H200_s']+item['accounting_pad_s']
            rows.append(dict(job_id=item['job_id'],native_GPU_s=wall,conservative_GPU_s=upper,exit_status=job.get('Exit_status')))
        value['storage_reserved_bytes']=sum(s['additional_output_upper_bytes'] for s in active_items(value))
        if allocated is not None:
            value['new_storage_current_bytes']=allocated
            value['new_storage_peak_bytes']=max(allocated,value['new_storage_peak_bytes'])
    return dict(status='PV1_TERMINAL_JOBS_SETTLED_QUEUED_RESERVATIONS_PRESERVED',rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=('prepare','submit','release','settle'));p.add_argument('--root',required=True)
    p.add_argument('--number',type=int);p.add_argument('--stage');p.add_argument('--project');p.add_argument('--walltime-s',type=int,default=14400)
    p.add_argument('--additional-output-upper-bytes',type=int,default=80*1024**3);p.add_argument('--memory-gb',type=int,default=128);a=p.parse_args()
    if a.operation=='prepare':result=prepare(a.root,stage=a.stage,project_name=a.project,walltime_s=a.walltime_s,
        additional_output_upper_bytes=a.additional_output_upper_bytes,memory_gb=a.memory_gb)
    elif a.operation=='settle':result=settle(a.root)
    else:result=globals()[a.operation](a.root,a.number)
    print(json.dumps(result),flush=True)
