"""Native E1 scheduler accounting and second finite workload, never GPU on login."""
import argparse
from datetime import date
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import uuid

from scripts.pi0292.allocation_admission import owned_jobs,personal_quota,current_project,PBS
from scripts.pi0292.durable import read,write,OwnerLock
from scripts.pi0294.e0_scope import pbs_epoch
from scripts.pi0294.native_dataset import sha
from .scope import validate,Ledger


def native(job):
    return json.loads(subprocess.check_output([PBS+'qstat','-x','-f','-F','json',job],text=True,timeout=20))['Jobs'][job]


def settle(root):
    root=Path(root);validate(root,'hopper-neural');ledger=Ledger(root,'hopper');allocations=[]
    with ledger.transaction() as l:
        for item in l['PBS_submissions']:
            if not item.get('job_id'):
                if item['status']=='NATIVE_QSUB_REJECTED_NO_JOB':continue
                raise PermissionError('E1_UNKNOWN_QSUB_INTENT_NOT_ZERO_COST')
            j=native(item['job_id'])
            if j['job_state']!='F' or j['Job_Owner'].split('@')[0]!=os.environ['USER'] or j.get('project')!='CFP04-CF-007':raise PermissionError('E1_SETTLE_ONLY_OWN_NATIVE_TERMINAL')
            if not j.get('stime'):
                if j.get('exec_host') or j.get('run_count',0):raise ValueError('E1_UNOBSERVED_GPU_COST_NOT_ZERO')
                wall=upper=0
            else:
                wall=sum(int(x)*m for x,m in zip(j['resources_used']['walltime'].split(':'),(3600,60,1)))
                if not j.get('obittime'):raise PermissionError('E1_NATIVE_TERMINAL_TIMESTAMP_REQUIRED')
                upper=max(wall,pbs_epoch(j['obittime'])-pbs_epoch(j['stime']))
            item.update(status='NATIVE_TERMINAL',native_exit_status=j.get('Exit_status'),native_allocated_wall_s=wall,GPU_conservative_charge_s=upper)
            allocations.append(dict(job_id=item['job_id'],native_state='F',exit=j.get('Exit_status'),GPU_native_s=wall,GPU_conservative_s=upper))
        total=sum(x['GPU_conservative_s'] for x in allocations)
        if total>28800:raise PermissionError('E1_GPU_BUDGET_EXCEEDED_PRESERVE_ACTUAL')
        l.update(GPU_allocated_s=total,GPU_allocated_s_native=sum(x['GPU_native_s'] for x in allocations),GPU_reserved_s=0)
    write(root/'evidence'/('GPU_SETTLEMENT_'+str(len(allocations))+'.json'),dict(status='NATIVE_TERMINAL_ALLOCATIONS_ACCOUNTED',allocations=allocations,GPU_conservative_s=total))
    return dict(status='E1_TERMINAL_GPU_ACCOUNTED',allocations=allocations,GPU_conservative_s=total)


def project_proof(root):
    value=current_project(executable='/opt/am/python/bin/amgr');p=value['project']
    if p['name']!='CFP04-CF-007' or p['active'] is not True or os.environ['USER'] not in p['users'] or value['balance_su']<=0:
        raise PermissionError('E1_NATIVE_ACTIVE_PROJECT_MEMBER_BALANCE')
    return dict(run_uuid=Path(root).name,**value)


def prepare_second(root):
    root=Path(root);m=validate(root,'hopper-neural');a=read(root/'E1_ADOPTION_AND_BINDING.json');l=read(root/'HOPPER_RESOURCE_LEDGER.json')
    if time.time()>=a['allocation_admission_cutoff_epoch'] or len(l['PBS_submissions'])!=1 or l['GPU_reserved_s'] or l['PBS_submissions'][0]['status']!='NATIVE_TERMINAL':raise PermissionError('E1_NEW_SCOPE_NO_UNSETTLED_OR_ACTIVE_PRIOR_JOB')
    if l['optimizer_updates']!=1024 or l['pending_updates']:raise PermissionError('E1_COMPLETE_M1_NO_PENDING_UPDATES')
    train=read(root/'training/M1/session-start/TRAINING_RUN.json');complete=root/'training/M1/checkpoint-1024/CHECKPOINT_COMPLETE.json'
    if train['global_step']!=1024 or not train.get('frozen_parameters_unchanged') or not read(complete)['full_resume']:
        raise PermissionError('E1_M1_COMPLETE_NATIVE_FROZEN_WEIGHT_PROOF')
    c0=read(root/'plans/C0_BASELINE_BINDING.private.json')
    if c0['status']!='PASS_ACTUAL_HOPPER_CORRECTION_PROCESSOR_LOADER' or c0['independent_reader']['full_windows']<8:raise PermissionError('E1_ACTUAL_NONEMPTY_CONSUMED_CORRECTION_BANK')
    history=read(root/'evidence/ORACLE_HISTORY_ADJUDICATION.json')
    if history['status']!='C0_ALL_USED_CORRECTION_ANCHORS_EXACT_ORIGINAL_STATIC_HISTORY':raise PermissionError('E1_EXACT_USED_STATE_HISTORY_REQUIRED')
    plan=read(root/'plans/D_COLLECTION_PLAN.json')
    if len(plan['cards'])!=4 or {c['task'] for c in plan['cards']}!={'T1'} or plan['qualified_tasks']!=['T1']:raise PermissionError('E1_FOUR_SMALL8_T1_COLLECTION_CONDITIONS_ONLY')
    if l['GPU_allocated_s']+14400>28800:raise PermissionError('E1_SECOND_FOUR_HOUR_ALLOCATION_CAP')
    if any(j.get('Job_Name','').startswith('pi0294_e1') for j in owned_jobs(os.environ['USER']).values()):raise PermissionError('E1_NO_OVERLAPPING_GPU')
    q=personal_quota(os.environ['USER'])
    if not 0<=time.time()-q['report_epoch']<=900 or q['hard_bytes']-q['used_bytes']<120*1024**3:raise PermissionError('E1_ACTUAL_FRESH_QUOTA_HEADROOM')
    d=root/'pbs/alloc-02';d.mkdir(mode=0o700);proof=project_proof(root);write(d/'PROJECT_ADMISSION.private.json',proof)
    names=['plans/SMALL8_BINDING.json','plans/PROBE_MANIFEST.json','plans/M1_RECIPE_RESOLVED.json','plans/STUDENT_PROBES.private.json',
        'plans/C0_BASELINE_BINDING.private.json','plans/D_COLLECTION_PLAN.json','plans/M2_RECIPE_RESOLVED.json',
        'evidence/ORACLE_HISTORY_ADJUDICATION.json','NEURAL_BINDINGS.private.json','training/M1/checkpoint-1024/CHECKPOINT_COMPLETE.json']
    b=read(root/'NEURAL_BINDINGS.private.json');code=root/m['code']['capsule_relative']
    request=dict(run_uuid=root.name,source_sha=m['code']['vla_commit'],run_manifest_sha256=sha(root/'RUN_MANIFEST.json'),mode='E1_T1_COLLECTION_MATCHED_M2',
        numerical_mode='DETERMINISTIC_CUDA_MATH_SDP_V1',sealed_inputs={n:sha(root/n) for n in names},submit_count=2,prior_GPU_allocated_s=l['GPU_allocated_s'],walltime_s=14400,
        python=b['python'],project_admission_relative=str((d/'PROJECT_ADMISSION.private.json').relative_to(root)),project_admission_sha256=sha(d/'PROJECT_ADMISSION.private.json'),request_nonce=str(uuid.uuid4()))
    write(d/'REQUEST.json',request)
    lines=['#!/bin/bash -l','#PBS -N pi0294_e1_2','#PBS -P CFP04-CF-007','#PBS -q auto','#PBS -l select=1:ncpus=12:mem=225gb:ngpus=1','#PBS -l walltime=04:00:00','#PBS -m n','#PBS -j oe','#PBS -o '+str(d/'pbs.log'),
        'set -eu','export PYTHONDONTWRITEBYTECODE=1','export PYTHONPATH='+str(code)+':'+str(code/'src'),
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1',
        'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1','cd '+str(code)]
    bootstrap="import os,runpy;os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));runpy.run_module('scripts.pi0294.e1.launcher',run_name='__main__')"
    lines.append('exec '+shlex.join([b['python'],'-c',bootstrap,'--root',str(root),'--request',str(d/'REQUEST.json')]))
    (d/'job.pbs').write_text('\n'.join(lines)+'\n');(d/'job.pbs').chmod(0o400)
    return dict(status='E1_SECOND_WORKLOAD_PREPARED_NOT_SUBMITTED',request_sha256=sha(d/'REQUEST.json'),script_sha256=sha(d/'job.pbs'),walltime_s=14400)


def submit_second(root):
    root=Path(root);validate(root,'hopper-neural');owner=OwnerLock(root/'E1_QSUB_OWNER.lock');ledger=Ledger(root,'hopper')
    try:
        with ledger.transaction() as l:
            a=read(root/'E1_ADOPTION_AND_BINDING.json');d=root/'pbs/alloc-02';r=read(d/'REQUEST.json')
            if len(l['PBS_submissions'])!=1 or l['GPU_reserved_s'] or l['PBS_submissions'][0]['status']!='NATIVE_TERMINAL' or time.time()>=a['allocation_admission_cutoff_epoch']:raise PermissionError('E1_EXACT_SECOND_SUBMISSION_ONLY')
            if l['GPU_allocated_s']!=r['prior_GPU_allocated_s'] or l['GPU_allocated_s']+r['walltime_s']>28800:raise PermissionError('E1_TOTAL_GPU_BOUND')
            if time.time()-read(d/'PROJECT_ADMISSION.private.json')['observed_epoch']>=900:raise PermissionError('E1_FRESH_PRE_QSUB_PROJECT_PROOF')
            if any(j.get('Job_Name','').startswith('pi0294_e1') for j in owned_jobs(os.environ['USER']).values()):raise PermissionError('E1_NO_DUPLICATE_QUEUE_OR_ALLOCATION')
            item=dict(status='QSUB_INTENT_OUTCOME_UNKNOWN_NO_RETRY',request_id='alloc-02',started_epoch=time.time(),walltime_s=14400)
            l['PBS_submissions'].append(item);l['GPU_reserved_s']=14400
            # Persist the intent before the native external mutation.
            write(ledger.path,l)
            result=subprocess.run([PBS+'qsub',str(d/'job.pbs')],capture_output=True,text=True,timeout=45)
            write(d/'QSUB_NATIVE.private.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
            if result.returncode:
                item['status']='NATIVE_QSUB_REJECTED_NO_JOB';l['GPU_reserved_s']=0;write(ledger.path,l);raise RuntimeError('E1_NATIVE_QSUB_REJECTED_NO_RETRY')
            job=result.stdout.strip()
            if not re.fullmatch(r'\d+\.hopper-m-\d+',job):raise RuntimeError('E1_UNKNOWN_NATIVE_QSUB_REPLY_PRESERVE_INTENT')
            item.update(status='SUBMITTED_PENDING_NATIVE',job_id=job,returned_epoch=time.time())
            return dict(status='E1_SECOND_PBS_SUBMITTED_ONCE',job_id=job,PBS_used=2,walltime_s=14400)
    finally:owner.close()


def release_second(root):
    root=Path(root);validate(root,'hopper-neural');d=root/'pbs/alloc-02';ready=read(d/'PBS_READY.json');l=read(root/'HOPPER_RESOURCE_LEDGER.json');job=native(ready['job_id'])
    if l['PBS_submissions'][-1]['job_id']!=ready['job_id'] or job['job_state']!='R' or job['Job_Owner'].split('@')[0]!=os.environ['USER'] or job.get('project')!='CFP04-CF-007':raise PermissionError('E1_EXACT_OWN_RUNNING_SECOND_PBS')
    if (d/'START_NEURAL.json').exists() or time.time()-ready['observed_epoch']>=175:raise PermissionError('E1_SINGLE_FRESH_SECOND_RELEASE')
    write(d/'PROJECT_ADMISSION.private.json',project_proof(root));r=read(d/'REQUEST.json');r['project_admission_sha256']=sha(d/'PROJECT_ADMISSION.private.json');write(d/'REQUEST.json',r)
    write(d/'START_NEURAL.json',dict(run_uuid=root.name,job_id=ready['job_id'],request_sha256=sha(d/'REQUEST.json'),observed_epoch=time.time()))
    return dict(status='E1_SECOND_RELEASED_FRESH_NATIVE_PROJECT',job_id=ready['job_id'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['settle','prepare-second','submit-second','release-second']);p.add_argument('--root',required=True);a=p.parse_args()
    print(json.dumps({'settle':settle,'prepare-second':prepare_second,'submit-second':submit_second,'release-second':release_second}[a.operation](a.root)),flush=True)
