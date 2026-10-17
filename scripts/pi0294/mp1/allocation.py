"""Native source-study admission, reusing the established scheduler readers."""
from datetime import datetime,timedelta,timezone,date
import os
from pathlib import Path
import socket
import time

from scripts.pi0292.allocation_admission import owned_jobs,personal_quota
from scripts.pi0292.durable import read,write
from scripts.pi0294.e0_scope import pbs_epoch,inside
from scripts.pi0294.native_dataset import sha
from .science import validate_runtime
from .recovery_config import limits


def request_contract(root,request,manifest,ledger,now):
    root=Path(root);a=read(root.parent.parent/'MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json')
    from .n1_limits import clock
    a=clock(root,a,manifest)
    p=root/'MP1_STAGE_BINDING.json';bound=limits(read(p) if p.exists() else {},root=root)
    if bound.get('job_wall_s_max',14400)>14400:
        from .n1_boundary import target
        if target(manifest['execution'].get('cold_engineering_binding_relative'))!=request['submit_count']:
            raise PermissionError('N1_REQUEST_MUST_USE_EXACT_CURRENT_COLD_FENCE')
        proof_path=inside(root/request['queue_limit_relative'],root);proof=read(proof_path)
        if (sha(proof_path)!=request['queue_limit_sha256']
                or request['sealed_inputs'].get(request['queue_limit_relative'])!=sha(proof_path)
                or proof['queue']!='auto' or proof['destination']!='small'
                or not 0<=request['created_epoch']-proof['observed_epoch']<=120
                or request['walltime_s']>proof['maximum_wall_s']):
            raise PermissionError('N1_NATIVE_SMALLER_QUEUE_WALLTIME_PROOF')
    if (request['run_uuid']!=root.name or request['source_sha']!=manifest['code']['vla_commit']
            or request['run_manifest_sha256']!=sha(root/'RUN_MANIFEST.json')
            or request['mode']!=bound['mode'] or request['numerical_mode']!='DETERMINISTIC_CUDA_MATH_SDP_V1'
            or type(request['submit_count']) is not int or not 1<=request['submit_count']<=bound['PBS']
            or type(request['walltime_s']) is not int or not 2760<request['walltime_s']<=bound.get('job_wall_s_max',14400)
            or request['walltime_s']+request['prior_GPU_allocated_s']>bound['GPU_s']
            or not a['T0_epoch']<=now<a['new_experiment_cutoff_epoch']
            or not a['T0_epoch']<=request['created_epoch']<a['new_submission_and_candidate_cutoff_epoch']):
        raise PermissionError('MP1_FIXED_SOURCE_NATIVE_REQUEST_AND_CLOCK')
    if (ledger['GPU_reserved_s']!=request['walltime_s'] or len(ledger['PBS_submissions'])!=request['submit_count']
            or ledger['GPU_allocated_s']!=request['prior_GPU_allocated_s']):
        raise PermissionError('MP1_SINGLE_REGISTERED_NATIVE_SUBMISSION_INTENT')
    for name,digest in request['sealed_inputs'].items():
        if sha(inside(root/name,root))!=digest:raise PermissionError('MP1_SEALED_NATIVE_INPUT_CHANGED:'+name)
    old=read(root/'plans/OLD_E1_BINDING.private.json');lease=read(old['lease'])
    if lease['status']!='CLOSED' or lease['run_uuid']!='20261016T150228Z-localpc-e1':
        raise PermissionError('MP1_SOURCE_GPU_AFTER_OLD_E1_HOPPER_LEASE_CLOSED_ONLY')
    return a


def admit(root,request_path):
    root=Path(root);m=validate_runtime(root,'hopper-neural');request_path=inside(request_path,root);r=read(request_path);now=time.time()
    ledger=read(root/'HOPPER_RESOURCE_LEDGER.json');a=request_contract(root,r,m,ledger,now)
    user=os.environ['USER'];job_id=os.environ.get('PBS_JOBID');jobs=owned_jobs(user);job=jobs.get(job_id)
    if (not job or job['job_state']!='R' or job.get('project')!='CFP04-CF-007'
            or job['Job_Owner'].split('@')[0]!=user or int(job['Resource_List']['ngpus'])!=1
            or ledger['PBS_submissions'][-1]['job_id']!=job_id):
        raise PermissionError('MP1_OWN_RUNNING_SINGLE_GPU_PROJECT_PBS')
    if any(k!=job_id and (v.get('Job_Name','').startswith('pi0294_e1') or v.get('Job_Name','').startswith('pi0294_mp1')) for k,v in jobs.items()):
        raise PermissionError('MP1_NO_OVERLAPPING_E1_OR_MP1_H200_ALLOCATION')
    nodes={n.split('/')[0].split('.')[0] for n in job['exec_host'].split('+')}
    if socket.gethostname().split('.')[0] not in nodes:raise PermissionError('MP1_NO_LOGIN_OR_FOREIGN_COMPUTE_NODE')
    wall=sum(int(x)*k for x,k in zip(job['Resource_List']['walltime'].split(':'),(3600,60,1)));started=pbs_epoch(job['stime'])
    if wall!=r['walltime_s'] or not a['T0_epoch']<=started<a['new_experiment_cutoff_epoch']:
        raise PermissionError('MP1_NATIVE_WALLTIME_OR_ADMISSION_CUTOFF')
    proof_path=inside(root/r['project_admission_relative'],root)
    if sha(proof_path)!=r['project_admission_sha256']:raise PermissionError('MP1_FRESH_LOGIN_SITE_PROOF_SEAL')
    proof=read(proof_path);project=proof['project'];today=datetime.fromtimestamp(now,timezone(timedelta(hours=8))).date()
    if (proof['run_uuid']!=root.name or not 0<=now-proof['observed_epoch']<=1800
            or project['name']!='CFP04-CF-007' or project['active'] is not True or user not in project['users']
            or not date.fromisoformat(project['start_date'])<=today<=date.fromisoformat(project['end_date']) or proof['balance_su']<=0):
        raise PermissionError('MP1_NATIVE_CURRENT_PROJECT_MEMBER_AND_BALANCE')
    q=personal_quota(user)
    if (not 0<=now-q['report_epoch']<=900 or q['hard_bytes']-q['used_bytes']<r['additional_output_upper_bytes']
            or r['additional_output_upper_bytes']>400*1024**3):raise PermissionError('MP1_FRESH_NATIVE_CAPACITY_FOR_ACTUAL_NEW_OUTPUT')
    out=dict(status='PASS_MP1_OWN_SOURCE_PBS',job_id=job_id,observed_epoch=now,allocation_start_epoch=started,
        allocation_end_epoch=min(started+wall,a['new_experiment_cutoff_epoch']),project='CFP04-CF-007',hostname=socket.gethostname(),request_sha256=sha(request_path))
    write(request_path.parent/'ALLOCATION_ADMISSION.json',out);return out
