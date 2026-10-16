"""E1 own-PBS admission; never invokes account login on a compute node."""
from datetime import date,datetime,timedelta,timezone
import os
from pathlib import Path
import socket
import time

from scripts.pi0292.allocation_admission import owned_jobs,personal_quota
from scripts.pi0292.durable import read,write
from scripts.pi0294.e0_scope import pbs_epoch,inside
from scripts.pi0294.native_dataset import sha
from .scope import validate


def admit(root,request_path):
    root=Path(root);m=validate(root,'hopper-neural');a=read(root/'E1_ADOPTION_AND_BINDING.json')
    request_path=inside(request_path,root);r=read(request_path);now=time.time()
    if (r['run_uuid']!=root.name or r['source_sha']!=m['code']['vla_commit']
            or r['run_manifest_sha256']!=sha(root/'RUN_MANIFEST.json')
            or r['mode']!='E1_M1_TRAIN_PROBE_EVAL' or r['numerical_mode']!='DETERMINISTIC_CUDA_MATH_SDP_V1'
            or not 1<=r['submit_count']<=3 or not 0<r['walltime_s']<=14400
            or r['walltime_s']+r['prior_GPU_allocated_s']>28800):
        raise PermissionError('EXACT_E1_ALLOCATION_REQUEST')
    for relative,digest in r['sealed_inputs'].items():
        if sha(inside(root/relative,root))!=digest:raise PermissionError('E1_INPUT_SEAL_CHANGED:'+relative)
    user=os.environ['USER'];job_id=os.environ.get('PBS_JOBID');jobs=owned_jobs(user);job=jobs.get(job_id)
    if (not job or job['job_state']!='R' or job.get('project')!='CFP04-CF-007'
            or job['Job_Owner'].split('@')[0]!=user or int(job['Resource_List']['ngpus'])!=1):
        raise PermissionError('E1_OWN_RUNNING_SINGLE_GPU_PROJECT_PBS_REQUIRED')
    if any(k!=job_id and v.get('Job_Name','').startswith('pi0294_e1') for k,v in jobs.items()):
        raise PermissionError('E1_ONE_GPU_ALLOCATION_ONLY')
    nodes={n.split('/')[0].split('.')[0] for n in job['exec_host'].split('+')}
    if socket.gethostname().split('.')[0] not in nodes:raise PermissionError('E1_NO_LOGIN_OR_FOREIGN_GPU')
    wall=sum(int(x)*k for x,k in zip(job['Resource_List']['walltime'].split(':'),(3600,60,1)))
    started=pbs_epoch(job['stime'])
    if wall!=r['walltime_s'] or not a['T0_epoch']<=started<a['allocation_admission_cutoff_epoch']:
        raise PermissionError('E1_ADMISSION_CLOCK_OR_WALLTIME')
    p=inside(root/r['project_admission_relative'],root)
    if sha(p)!=r['project_admission_sha256']:raise PermissionError('E1_BOUND_FRESH_LOGIN_PROJECT_RECEIPT')
    proof=read(p);project=proof['project'];today=datetime.fromtimestamp(now,timezone(timedelta(hours=8))).date()
    if (proof['run_uuid']!=root.name or not 0<=now-proof['observed_epoch']<=1800
            or project['name']!='CFP04-CF-007' or project['active'] is not True or user not in project['users']
            or not date.fromisoformat(project['start_date'])<=today<=date.fromisoformat(project['end_date'])
            or proof['balance_su']<=0):raise PermissionError('E1_NATIVE_CURRENT_PROJECT_MEMBERSHIP_BALANCE')
    q=personal_quota(user)
    if not 0<=now-q['report_epoch']<=900 or q['hard_bytes']-q['used_bytes']<120*1024**3:
        raise PermissionError('E1_FRESH_QUOTA_120GIB_HEADROOM')
    l=read(root/'HOPPER_RESOURCE_LEDGER.json')
    if (l['PBS_submissions'][-1]['job_id']!=job_id or l['GPU_reserved_s']!=wall
            or len(l['PBS_submissions'])!=r['submit_count']):raise PermissionError('E1_ACTUAL_ONE_SHOT_SUBMISSION_LEDGER')
    out=dict(status='PASS_E1_OWN_PBS',job_id=job_id,observed_epoch=now,allocation_start_epoch=started,
        allocation_end_epoch=min(started+wall,a['hard_closeout_epoch']),project='CFP04-CF-007',
        hostname=socket.gethostname(),request_sha256=sha(request_path))
    write(request_path.parent/'ALLOCATION_ADMISSION.json',out);return out
