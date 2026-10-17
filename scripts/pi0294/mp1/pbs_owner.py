"""Finite source-study submissions through the existing native PBS primitives."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import uuid

from scripts.pi0292.allocation_admission import owned_jobs,personal_quota,PBS
from scripts.pi0292.durable import read,write,OwnerLock
from scripts.pi0294.e0_scope import pbs_epoch
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.e1.pbs_owner import native,project_proof
from .science import validate_runtime,Ledger
from .recovery_config import limits
from .n1_limits import clock


def native_queue_limit(query=subprocess.check_output):
    def get(name):
        value=json.loads(query([PBS+'qstat','-Qf','-F','json',name],text=True,timeout=20))
        return value['Queue'][name]
    auto=get('auto');small=get('small')
    if (auto['queue_type']!='Route' or 'small' not in auto['route_destinations'].split(',')
            or auto['enabled']!='True' or auto['started']!='True'
            or small['queue_type']!='Execution' or small['enabled']!='True' or small['started']!='True'
            or not int(small['resources_min']['ngpus'])<=1<=int(small['resources_max']['ngpus'])):
        raise PermissionError('N1_NATIVE_AUTO_SINGLE_GPU_DESTINATION_NOT_PROVEN')
    text=small['resources_max']['walltime']
    wall=sum(int(v)*k for v,k in zip(text.split(':'),(3600,60,1)))
    if wall<=2760:raise PermissionError('N1_NATIVE_QUEUE_CANNOT_COVER_ORIGINAL_CLOSEOUT')
    return dict(queue='auto',destination='small',observed_epoch=time.time(),
        maximum_wall_s=wall,auto=auto,destination_definition=small,read_only=True)


def old_scope_closed(root):
    old=read(root/'plans/OLD_E1_BINDING.private.json');lease=read(old['lease']);l=read(Path(old['root'])/'HOPPER_RESOURCE_LEDGER.json')
    if (lease['status']!='CLOSED' or lease['run_uuid']!='20261016T150228Z-localpc-e1'
            or l['GPU_reserved_s'] or any(v['status']!='NATIVE_TERMINAL' for v in l['PBS_submissions'])):
        raise PermissionError('MP1_DOES_NOT_REUSE_ACTIVE_OLD_E1_LEASE_OR_RESERVATION')


def no_overlap():
    if any(j.get('Job_Name','').startswith(('pi0294_e1','pi0294_mp1')) for j in owned_jobs(os.environ['USER']).values()):
        raise PermissionError('MP1_ONLY_ONE_OWN_H200_QUEUE_OR_ALLOCATION')


def prepare(root,*,output_rebinding=None,memory_gb=225):
    if type(memory_gb) is not int or memory_gb not in [128,225]:
        raise PermissionError('ONLY_EXISTING_SOURCE_ALLOCATION_MEMORY_PROFILES')
    root=Path(root);m=validate_runtime(root,'hopper-neural');old_scope_closed(root);no_overlap()
    a=clock(root);l=read(root/'HOPPER_RESOURCE_LEDGER.json')
    bound=limits(read(root/'MP1_STAGE_BINDING.json'),root=root)
    from .n1_limits import resolve
    if resolve(root,m) is not None:
        from .transport_fixture import prepared_plan
        prepared_plan(root,m)
    if (time.time()>=a['new_submission_and_candidate_cutoff_epoch'] or len(l['PBS_submissions'])>=bound['PBS'] or l['GPU_reserved_s']
            or any(v['status']!='NATIVE_TERMINAL' for v in l['PBS_submissions'])):raise PermissionError('MP1_NEW_SUBMISSION_CLOCK_AND_PRIOR_TERMINALS')
    queue_proof=native_queue_limit() if bound.get('job_wall_s_max',14400)>14400 else None
    wall=min(bound.get('job_wall_s_max',14400),queue_proof['maximum_wall_s'] if queue_proof is not None else 14400,
        (bound['GPU_s']-l['GPU_allocated_s'])//60*60,
        int(a['new_experiment_cutoff_epoch']-time.time())//60*60)
    if wall<=2760:raise PermissionError('MP1_PRESERVE_45_MIN_ALLOCATION_CLOSEOUT')
    upper=bound['output_bytes'];rebinding=None;management_source=m['code']['vla_commit']
    if output_rebinding is not None:
        from .output_capacity import checked_estimate
        rebinding=Path(output_rebinding)
        upper,m=checked_estimate(root,rebinding,m)
    used=int(subprocess.check_output(['du','-s','-B1',str(root)],text=True,timeout=30).split()[0])
    if used>upper:raise PermissionError('MP1_SOURCE_STUDY_OUTPUT_ESTIMATE_REQUIRES_EVIDENCED_REBINDING')
    additional=max(8*1024**3,upper-used);q=personal_quota(os.environ['USER'])
    if not 0<=time.time()-q['report_epoch']<=900 or q['hard_bytes']-q['used_bytes']<additional:raise PermissionError('MP1_NATIVE_QUOTA_FOR_NEW_SOURCE_OUTPUT')
    number=len(l['PBS_submissions'])+1;d=root/'pbs'/('alloc-'+str(number).zfill(2));d.mkdir(mode=0o700)
    if queue_proof is not None:write(d/'NATIVE_QUEUE_LIMIT.json',queue_proof)
    write(d/'PROJECT_ADMISSION.private.json',project_proof(root));b=read(root/'NEURAL_BINDINGS.private.json');code=root/m['code']['capsule_relative']
    names=list(m['execution']['sealed_inputs'])+['NEURAL_BINDINGS.private.json','plans/OLD_E1_BINDING.private.json']
    if rebinding is not None:names.append(str(rebinding.relative_to(root)))
    if queue_proof is not None:names.append(str((d/'NATIVE_QUEUE_LIMIT.json').relative_to(root)))
    r=dict(run_uuid=root.name,source_sha=m['code']['vla_commit'],run_manifest_sha256=sha(root/'RUN_MANIFEST.json'),mode=bound['mode'],
        numerical_mode='DETERMINISTIC_CUDA_MATH_SDP_V1',sealed_inputs={n:sha(root/n) for n in sorted(set(names))},submit_count=number,
        prior_GPU_allocated_s=l['GPU_allocated_s'],walltime_s=wall,additional_output_upper_bytes=additional,created_epoch=time.time(),python=b['python'],
        project_admission_relative=str((d/'PROJECT_ADMISSION.private.json').relative_to(root)),project_admission_sha256=sha(d/'PROJECT_ADMISSION.private.json'),request_nonce=str(uuid.uuid4()))
    if rebinding is not None:r.update(output_estimate_binding_relative=str(rebinding.relative_to(root)),output_estimate_binding_sha256=sha(rebinding),management_source_sha=management_source,scientific_launch_source_unchanged=True)
    if queue_proof is not None:r.update(queue_limit_relative=str((d/'NATIVE_QUEUE_LIMIT.json').relative_to(root)),queue_limit_sha256=sha(d/'NATIVE_QUEUE_LIMIT.json'))
    write(d/'REQUEST.json',r);h,rem=divmod(wall,3600);minutes,seconds=divmod(rem,60)
    lines=['#!/bin/bash -l','#PBS -N pi0294_mp1_s'+str(number),'#PBS -P CFP04-CF-007','#PBS -q auto',
        f'#PBS -l select=1:ncpus=12:mem={memory_gb}gb:ngpus=1',f'#PBS -l walltime={h:02d}:{minutes:02d}:{seconds:02d}',
        '#PBS -m n','#PBS -j oe','#PBS -o '+str(d/'pbs.log'),'set -eu','export PYTHONDONTWRITEBYTECODE=1',
        'export PYTHONPATH='+str(code)+':'+str(code/'src'),
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1',
        'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1','cd '+str(code)]
    if m['execution'].get('cold_engineering_binding_relative'):
        lines.append('export MP1_RUN_MANIFEST='+shlex.quote(str(code.parent/'WORKER_RUN_MANIFEST.json')))
    bootstrap="import os,runpy;os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));runpy.run_module('scripts.pi0294.mp1.launcher',run_name='__main__')"
    lines.append('exec '+shlex.join([b['python'],'-c',bootstrap,'--root',str(root),'--request',str(d/'REQUEST.json')]))
    (d/'job.pbs').write_text('\n'.join(lines)+'\n');(d/'job.pbs').chmod(0o400)
    write(d/'OUTPUT_CAPACITY_PRECHECK.json',dict(observed_new_bytes=used,additional_output_upper_bytes=additional,study_estimated_new_peak_bytes=upper,protocol_Hopper_peak_limit_bytes=400*1024**3))
    return dict(status='MP1_SOURCE_NATIVE_REQUEST_PREPARED_NOT_SUBMITTED',submit_count=number,walltime_s=wall,request_sha256=sha(d/'REQUEST.json'))


def submit(root):
    root=Path(root);validate_runtime(root,'hopper-neural');old_scope_closed(root);no_overlap();owner=OwnerLock(root/'MP1_QSUB_OWNER.lock');ledger=Ledger(root,'hopper')
    try:
        with ledger.transaction() as l:
            number=len(l['PBS_submissions'])+1;d=root/'pbs'/('alloc-'+str(number).zfill(2));r=read(d/'REQUEST.json');a=clock(root)
            bound=limits(read(root/'MP1_STAGE_BINDING.json'),root=root)
            if (number>bound['PBS'] or r['submit_count']!=number or r['mode']!=bound['mode'] or l['GPU_reserved_s'] or any(v['status']!='NATIVE_TERMINAL' for v in l['PBS_submissions'])
                    or time.time()>=a['new_submission_and_candidate_cutoff_epoch'] or l['GPU_allocated_s']!=r['prior_GPU_allocated_s']
                    or l['GPU_allocated_s']+r['walltime_s']>bound['GPU_s'] or time.time()-read(d/'PROJECT_ADMISSION.private.json')['observed_epoch']>=900):
                raise PermissionError('MP1_SINGLE_FRESH_NATIVE_INTENT_WITHIN_RESERVED_STUDY_BUDGET')
            from .allocation import request_contract
            prospective=dict(l,GPU_reserved_s=r['walltime_s'],PBS_submissions=l['PBS_submissions']+[{}])
            request_contract(root,r,read(root/'RUN_MANIFEST.json'),prospective,time.time())
            item=dict(status='QSUB_INTENT_OUTCOME_UNKNOWN_NO_RETRY',request_id=d.name,started_epoch=time.time(),walltime_s=r['walltime_s'])
            l['PBS_submissions'].append(item);l['GPU_reserved_s']=r['walltime_s'];write(ledger.path,l)
            result=subprocess.run([PBS+'qsub',str(d/'job.pbs')],capture_output=True,text=True,timeout=45)
            write(d/'QSUB_NATIVE.private.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
            if result.returncode:
                item['status']='NATIVE_QSUB_REJECTED_NO_JOB';l['GPU_reserved_s']=0;write(ledger.path,l);raise RuntimeError('MP1_NATIVE_QSUB_REJECTED_NO_BLIND_RETRY')
            job=result.stdout.strip()
            if not re.fullmatch(r'\d+\.hopper-m-\d+',job):raise RuntimeError('MP1_UNKNOWN_NATIVE_QSUB_REPLY_PRESERVE_INTENT')
            item.update(status='SUBMITTED_PENDING_NATIVE',job_id=job,returned_epoch=time.time())
            return dict(status='MP1_SOURCE_PBS_SUBMITTED_ONCE',job_id=job,submit_count=number,walltime_s=r['walltime_s'])
    finally:owner.close()


def release(root):
    root=Path(root);validate_runtime(root,'hopper-neural');old_scope_closed(root);l=read(root/'HOPPER_RESOURCE_LEDGER.json')
    number=len(l['PBS_submissions']);d=root/'pbs'/('alloc-'+str(number).zfill(2));ready=read(d/'PBS_READY.json');j=native(ready['job_id'])
    if (l['PBS_submissions'][-1]['job_id']!=ready['job_id'] or j['job_state']!='R' or j['Job_Owner'].split('@')[0]!=os.environ['USER']
            or j.get('project')!='CFP04-CF-007' or (d/'START_NEURAL.json').exists() or time.time()-ready['observed_epoch']>=175):
        raise PermissionError('MP1_EXACT_OWN_RUNNING_ALLOCATION_SINGLE_FRESH_RELEASE')
    write(d/'PROJECT_ADMISSION.private.json',project_proof(root));r=read(d/'REQUEST.json');r['project_admission_sha256']=sha(d/'PROJECT_ADMISSION.private.json');write(d/'REQUEST.json',r)
    write(d/'START_NEURAL.json',dict(run_uuid=root.name,job_id=ready['job_id'],request_sha256=sha(d/'REQUEST.json'),observed_epoch=time.time()))
    return dict(status='MP1_SOURCE_NATIVE_RELEASED_CURRENT_PROJECT',job_id=ready['job_id'])


def settle(root):
    root=Path(root);validate_runtime(root,'hopper-neural',closeout=True);ledger=Ledger(root,'hopper');rows=[]
    with ledger.transaction() as l:
        for item in l['PBS_submissions']:
            if item['status']=='NATIVE_QSUB_REJECTED_NO_JOB':continue
            j=native(item['job_id'])
            if j['job_state']!='F' or j['Job_Owner'].split('@')[0]!=os.environ['USER'] or j.get('project')!='CFP04-CF-007':raise PermissionError('MP1_SETTLE_ONLY_OWN_TERMINAL_NATIVE_JOB')
            wall=sum(int(x)*m for x,m in zip(j['resources_used'].get('walltime','00:00:00').split(':'),(3600,60,1)))
            if j.get('stime'):
                if not j.get('obittime'):raise PermissionError('MP1_TERMINAL_TIMESTAMP_REQUIRED')
                upper=max(wall,pbs_epoch(j['obittime'])-pbs_epoch(j['stime']))
            else:
                if j.get('exec_host') or j.get('run_count',0) or wall:raise PermissionError('MP1_UNOBSERVED_ALLOCATION_NOT_ZERO')
                upper=0
            item.update(status='NATIVE_TERMINAL',native_exit_status=j.get('Exit_status'),native_allocated_wall_s=wall,GPU_conservative_charge_s=upper)
            rows.append(dict(job_id=item['job_id'],GPU_native_s=wall,GPU_conservative_s=upper,exit_status=j.get('Exit_status')))
        total=sum(v['GPU_conservative_s'] for v in rows)
        if total>limits(read(root/'MP1_STAGE_BINDING.json'),root=root)['GPU_s']:raise PermissionError('MP1_SOURCE_STUDY_ALLOCATION_CAP_EXCEEDED_PRESERVE_ACTUAL')
        l.update(GPU_allocated_s=total,GPU_reserved_s=0)
    value=dict(status='MP1_SOURCE_NATIVE_ALLOCATIONS_SETTLED',rows=rows,GPU_conservative_s=total)
    write(root/'evidence'/('GPU_SETTLEMENT_'+str(len(rows))+'.json'),value);return value


if __name__=='__main__':
    operations=dict(prepare=prepare,submit=submit,release=release,settle=settle)
    p=argparse.ArgumentParser();p.add_argument('operation',choices=operations);p.add_argument('--root',required=True);p.add_argument('--output-rebinding');p.add_argument('--memory-gb',type=int,choices=[128,225]);a=p.parse_args()
    if (a.output_rebinding or a.memory_gb is not None) and a.operation!='prepare':p.error('output/memory preparation options are only valid for prepare')
    print(json.dumps(prepare(a.root,output_rebinding=a.output_rebinding,memory_gb=a.memory_gb or 225) if a.operation=='prepare' else operations[a.operation](a.root)))
