"""PI0294 fresh login attestation plus native PBS identity, before model import."""
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import uuid

from scripts.pi0292.allocation_admission import owned_jobs, personal_quota, current_project
from scripts.pi0292.durable import read, write

PROJECT = 'CFP04-CF-007'
T0, CUTOFF, DEADLINE = 1791901497, 1791939600, 1791943200
GIB = 1024**3


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''):
            value.update(block)
    return value.hexdigest()


def validate(*, now,user,job_id,jobs,quota,project_receipt,adoption,request,hostname):
    if (adoption['pi'] != 'PI0294' or adoption['T0_epoch'] != T0
        or adoption['new_work_cutoff_epoch'] != CUTOFF or adoption['deadline_epoch'] != DEADLINE
        or adoption['reset_on_resume'] or now >= CUTOFF):
        raise PermissionError('PI0294_ORIGINAL_AUTHORITY_CLOCK_REQUIRED')
    if request['stage'] not in ('smoke','benchmark','main','resident'):
        raise PermissionError('UNKNOWN_STAGE')
    if request['submit_count'] > 4 or request['prior_GPU_allocated_s']+request['walltime_s'] > 43200:
        raise PermissionError('NIGHT_SUBMISSION_OR_GPU_ALLOCATION_CAP')
    nonce = uuid.UUID(request['request_nonce'])
    if nonce.version != 4 or str(nonce) != request['request_nonce']:
        raise PermissionError('CANONICAL_REQUEST_NONCE_REQUIRED')
    job = jobs.get(job_id)
    if job is None or job['job_state'] != 'R' or job['Job_Owner'].split('@')[0] != user or job.get('project') != PROJECT:
        raise PermissionError('OWN_RUNNING_PROJECT_ALLOCATION_REQUIRED')
    resources = job['Resource_List']
    wall = sum(int(v)*m for v,m in zip(resources['walltime'].split(':'),(3600,60,1)))
    if int(resources['ngpus']) != 1 or wall != request['walltime_s']:
        raise PermissionError('EXACT_SINGLE_GPU_REQUEST_REQUIRED')
    nodes = {p.split('/')[0].split('.')[0] for p in job['exec_host'].split('+')}
    if hostname.split('.')[0] not in nodes:
        raise PermissionError('OWN_ALLOCATED_COMPUTE_NODE_REQUIRED')
    peers = {k:v for k,v in jobs.items() if k != job_id and int(v.get('Resource_List',{}).get('ngpus',0)) > 0}
    if len(peers) > 1 or not set(peers).issubset(set(request['allowed_peer_ids'])):
        raise PermissionError('ACCOUNT_MAX_TWO_INDEPENDENT_GPU_ALLOCATIONS')
    if any(v['Job_Owner'].split('@')[0] != user or int(v['Resource_List']['ngpus']) != 1 for v in peers.values()):
        raise PermissionError('PEER_SINGLE_OWN_GPU_REQUIRED')
    project = project_receipt['project']
    day = datetime.fromtimestamp(now,timezone(timedelta(hours=8))).date()
    if (project['name'] != PROJECT or not project['active'] or user not in project['users']
        or 'hopper-m-02' not in project['cluster']
        or not date.fromisoformat(project['start_date']) <= day <= date.fromisoformat(project['end_date'])
        or not 0 <= now-project_receipt['observed_epoch'] <= 1800
        or not 0 < project_receipt['balance_su'] < float('inf')):
        raise PermissionError('FRESH_NATIVE_LOGIN_PROJECT_PROOF_REQUIRED')
    if quota['user'] != user or not 0 <= now-quota['report_epoch'] <= 900 or quota['hard_bytes']-quota['used_bytes'] < request['predicted_remaining_peak_bytes']+20*GIB:
        raise PermissionError('CURRENT_PERSONAL_QUOTA_HEADROOM_REQUIRED')
    return dict(status='PASS_PI0294_OWN_NATIVE_PBS_ADMISSION_NOT_MODEL_QUALIFICATION',
                job_id=job_id,hostname=hostname,project=PROJECT,observed_epoch=now)


def admit(root,request_path):
    root,request_path = Path(root),Path(request_path)
    from .run_admission import validate as entry_validate
    entry_validate(root,profile='hopper-neural')
    if not request_path.resolve().is_relative_to(root.resolve()):
        raise PermissionError('REQUEST_OUTSIDE_REGISTERED_RUN')
    request,adoption = read(request_path),read(root/'ADOPTION.json')
    for name,key in (('RUN_MANIFEST.json','run_manifest_sha256'),('CODE_SEAL.json','code_seal_sha256')):
        if sha(root/name) != request[key]:
            raise PermissionError('REQUEST_BOUND_IDENTITY_CHANGED')
    seal = read(root/'CODE_SEAL.json')
    if seal['source_sha'] != request['source_sha'] or seal['protocol_release'] != adoption['protocol_release']:
        raise PermissionError('NEW_PROTOCOL_AND_CODE_BINDING_REQUIRED')
    for name,digest in seal['files'].items():
        path = root/'code'/name
        if not path.resolve().is_relative_to((root/'code').resolve()) or sha(path) != digest:
            raise PermissionError('FIXED_CAPSULE_CHANGED')
    if sha(request['bindings']) != request['bindings_sha256']:
        raise PermissionError('PRIVATE_BINDING_CHANGED')
    project_path = request_path.parent/'PROJECT_ADMISSION.private.json'
    if sha(project_path) != request['project_admission_sha256']:
        raise PermissionError('LOGIN_ATTESTATION_CHANGED')
    project = read(project_path)
    if project['request_nonce'] != request['request_nonce'] or project['run_uuid'] != root.name:
        raise PermissionError('LOGIN_ATTESTATION_REQUEST_MISMATCH')
    job_id = os.environ.get('PBS_JOBID')
    if not job_id:
        raise PermissionError('NO_NEURAL_ON_LOGIN')
    jobs=owned_jobs(os.environ['USER'])
    result = validate(now=time.time(),user=os.environ['USER'],job_id=job_id,
        jobs=jobs,quota=personal_quota(os.environ['USER']),
        project_receipt=project,adoption=adoption,request=request,hostname=socket.gethostname())
    elapsed=jobs[job_id].get('resources_used',{}).get('walltime','00:00:00')
    elapsed_s=sum(int(v)*m for v,m in zip(elapsed.split(':'),(3600,60,1)))
    result['allocation_end_epoch']=min(DEADLINE,time.time()+request['walltime_s']-elapsed_s)
    remaining = request['predicted_remaining_peak_bytes']
    owned = sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
    if remaining <= 0 or owned+remaining > 180*GIB or shutil.disk_usage(root).free < remaining+20*GIB:
        raise PermissionError('HOPPER_PEAK_SPACE_BOUND')
    result['request_sha256'] = sha(request_path)
    return result


def main():
    request_path = Path(sys.argv[1]);request = read(request_path);root = Path(request['root'])
    result = admit(root,request_path)
    write(request_path.parent/'ALLOCATION_ADMISSION.json',result)
    authorization = dict(job_id=os.environ['PBS_JOBID'],request_sha256=sha(request_path),
        allowed_modes=request['allowed_modes'],allocation_end_epoch=result['allocation_end_epoch'],
        allocation_walltime_s=request['walltime_s'],prior_GPU_allocated_s=request['prior_GPU_allocated_s'],
        predicted_remaining_peak_bytes=request['predicted_remaining_peak_bytes'])
    write(root/'NEURAL_AUTHORIZATION.json',authorization)
    os.execv(request['python'],[request['python'],'-m','scripts.pi0294.neural','--root',str(root),
        '--request',str(request_path),'--bindings',request['bindings'],'--mode',request['mode'],
        '--microbatch',str(request.get('microbatch',2)),'--workers',str(request.get('workers',2))])


if __name__ == '__main__':
    main()
