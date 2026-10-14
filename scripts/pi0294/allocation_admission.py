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


def remaining_peak(request,owned):
    if 'predicted_total_owned_peak_bytes' not in request:return request['predicted_remaining_peak_bytes']
    total=request['predicted_total_owned_peak_bytes']
    if type(total) is not int or not owned<total<=180*GIB:
        raise PermissionError('SEALED_TOTAL_PEAK_EXCEEDS_CAP_OR_ALREADY_EXHAUSTED')
    return total-owned


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''):
            value.update(block)
    return value.hexdigest()


def validate(*, now,user,job_id,jobs,quota,project_receipt,adoption,request,hostname,
             continued_allocation=None):
    closeout=request['stage'] in ('resident','checkpoint')
    if (adoption['pi'] != 'PI0294' or adoption['T0_epoch'] != T0
        or adoption['new_work_cutoff_epoch'] != CUTOFF or adoption['deadline_epoch'] != DEADLINE
        or adoption['reset_on_resume'] or now < T0 or now >= (DEADLINE if closeout else CUTOFF)):
        raise PermissionError('PI0294_ORIGINAL_AUTHORITY_CLOCK_REQUIRED')
    if request['stage'] not in ('smoke','benchmark','main','resident','night-final','checkpoint'):
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
    fresh_project = 0 <= now-project_receipt['observed_epoch'] <= 1800
    if continued_allocation is not None:
        proof=continued_allocation
        if (proof.get('status')!='PASS_PI0294_BOUND_LIVE_NIGHT_PARENT'
                or proof.get('job_id')!=job_id or proof.get('project')!=PROJECT
                or proof.get('project_attestation_sha256')!=request.get('project_admission_sha256')
                or not 0 <= proof['initial_admission_epoch']-project_receipt['observed_epoch'] <= 1800
                or not proof['initial_admission_epoch'] <= now < proof['allocation_end_epoch']
                or request['stage'] not in ('main','resident','checkpoint')):
            raise PermissionError('EXACT_INITIAL_PROJECT_PROOF_AND_LIVE_ALLOCATION_REQUIRED')
        fresh_project=True
    if (project['name'] != PROJECT or not project['active'] or user not in project['users']
        or 'hopper-m-02' not in project['cluster']
        or not date.fromisoformat(project['start_date']) <= day <= date.fromisoformat(project['end_date'])
        or not fresh_project
        or not 0 < project_receipt['balance_su'] < float('inf')):
        raise PermissionError('FRESH_NATIVE_LOGIN_PROJECT_PROOF_REQUIRED')
    if quota['user'] != user or not 0 <= now-quota['report_epoch'] <= 900 or quota['hard_bytes']-quota['used_bytes'] < request['predicted_remaining_peak_bytes']+20*GIB:
        raise PermissionError('CURRENT_PERSONAL_QUOTA_HEADROOM_REQUIRED')
    return dict(status='PASS_PI0294_OWN_NATIVE_PBS_ADMISSION_NOT_MODEL_QUALIFICATION',
                job_id=job_id,hostname=hostname,project=PROJECT,observed_epoch=now,
                project_proof_mode='SAME_LIVE_NATIVE_ALLOCATION_INITIAL_ENTITLEMENT'
                    if continued_allocation is not None else 'FRESH_LOGIN_PROJECT_AT_INITIAL_ADMISSION')


def night_parent_proof(root,request_path,request,*,job_id,now=None):
    """A live same-job parent can continue, never authorize another allocation."""
    now=time.time() if now is None else now
    names=('night_parent_request_relative','night_parent_request_sha256')
    if not any(k in request for k in names):return None
    if not all(k in request for k in names):
        raise PermissionError('COMPLETE_BOUND_NIGHT_PARENT_REFERENCE_REQUIRED')
    relative=Path(request[names[0]])
    root,request_path=Path(root),Path(request_path)
    path=root/relative
    if (relative.is_absolute() or '..' in relative.parts or path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve()) or path==request_path
            or not path.is_file() or sha(path)!=request[names[1]]):
        raise PermissionError('EXACT_PARENT_REQUEST_INSIDE_OWN_RUN_REQUIRED')
    parent=read(path)
    if parent.get('stage')!='night-final' or parent.get('mode')!='night-final':
        raise PermissionError('INITIAL_NIGHT_ALLOCATION_ONLY')
    for key in ('root','source_sha','run_manifest_sha256','code_seal_sha256','bindings','bindings_sha256',
                'walltime_s','submit_count','prior_GPU_allocated_s','numerical_mode','microbatch','workers',
                'project_admission_sha256','request_nonce'):
        if request.get(key)!=parent.get(key):
            raise PermissionError('PARENT_ALLOCATION_IDENTITY_CHANGED:'+key)
    admission_path=path.parent/'ALLOCATION_ADMISSION.json'
    state_path=path.parent/'night-session/STATE.json'
    if any(p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root.resolve())
           for p in (admission_path,state_path)):
        raise PermissionError('ACTUAL_PARENT_ADMISSION_AND_STATE_REQUIRED')
    admission,state=read(admission_path),read(state_path)
    if (admission.get('status')!='PASS_PI0294_OWN_NATIVE_PBS_ADMISSION_NOT_MODEL_QUALIFICATION'
            or admission.get('project_proof_mode')!='FRESH_LOGIN_PROJECT_AT_INITIAL_ADMISSION'
            or admission.get('job_id')!=job_id or admission.get('project')!=PROJECT
            or admission.get('request_sha256')!=sha(path)
            or admission.get('project_attestation_sha256')!=parent['project_admission_sha256']
            or state.get('job_id')!=job_id or state.get('run_uuid')!=root.name
            or state.get('parent_request_sha256')!=sha(path)
            or state.get('active_request_sha256')!=sha(request_path)
            or state.get('status')!='RUNNING_'+request['stage'].upper()
            or not admission['observed_epoch']<=now<admission['allocation_end_epoch']):
        raise PermissionError('ACTUAL_SAME_JOB_PARENT_SUBSTAGE_REQUIRED')
    pid=state.get('pid');start=state.get('proc_start')
    if type(pid) is not int or pid<=0 or not isinstance(start,str):
        raise PermissionError('EXACT_LIVE_PARENT_PROCESS_IDENTITY_REQUIRED')
    proc=Path('/proc')/str(pid)
    try:
        fields=(proc/'stat').read_text().split()
        command=(proc/'cmdline').read_bytes().split(b'\0')
        if (proc.stat().st_uid!=os.getuid() or fields[21]!=start or fields[2] in ('Z','X')
                or b'scripts.pi0294.night_session' not in command or str(path).encode() not in command):
            raise PermissionError('PARENT_PID_REUSED_TERMINAL_OR_WRONG_COMMAND')
    except (FileNotFoundError,ProcessLookupError):
        raise PermissionError('NIGHT_PARENT_NOT_LIVE') from None
    return dict(status='PASS_PI0294_BOUND_LIVE_NIGHT_PARENT',job_id=job_id,project=PROJECT,
        initial_admission_epoch=admission['observed_epoch'],allocation_end_epoch=admission['allocation_end_epoch'],
        project_attestation_sha256=admission['project_attestation_sha256'],parent_request_sha256=sha(path),
        parent_pid=pid,parent_proc_start=start)


def admit(root,request_path):
    from .continuation import RUN as CONTINUATION_RUN, admit as admit_continuation
    if Path(root).name == CONTINUATION_RUN:
        return admit_continuation(root, request_path)
    root,request_path = Path(root),Path(request_path)
    from .run_admission import validate as entry_validate
    if not request_path.resolve().is_relative_to(root.resolve()):
        raise PermissionError('REQUEST_OUTSIDE_REGISTERED_RUN')
    request=read(request_path)
    manifest=entry_validate(root,profile='hopper-neural',closeout=request.get('stage') in ('resident','checkpoint'))
    request,adoption = read(request_path),read(root/'ADOPTION.json')
    seal_path=root/manifest['code'].get('source_seal_relative','SOURCE_SEAL.json')
    capsule=root/manifest['code'].get('capsule_relative','code')
    if (sha(root/'RUN_MANIFEST.json')!=request['run_manifest_sha256']
        or sha(seal_path)!=request['code_seal_sha256']
        or request['source_sha']!=manifest['code']['vla_commit']):
        raise PermissionError('REQUEST_BOUND_IDENTITY_CHANGED')
    seal = read(seal_path)
    if seal['source_sha'] != request['source_sha'] or seal['protocol_release'] != adoption['protocol_release']:
        raise PermissionError('NEW_PROTOCOL_AND_CODE_BINDING_REQUIRED')
    for name,digest in seal['files'].items():
        path = capsule/name
        if not path.resolve().is_relative_to(capsule.resolve()) or path.is_symlink() or sha(path) != digest:
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
    owned = sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
    admission_request=dict(request)
    admission_request['predicted_remaining_peak_bytes']=remaining_peak(request,owned)
    continuation=night_parent_proof(root,request_path,request,job_id=job_id)
    result = validate(now=time.time(),user=os.environ['USER'],job_id=job_id,
        jobs=jobs,quota=personal_quota(os.environ['USER']),
        project_receipt=project,adoption=adoption,request=admission_request,hostname=socket.gethostname(),
        continued_allocation=continuation)
    elapsed=jobs[job_id].get('resources_used',{}).get('walltime','00:00:00')
    elapsed_s=sum(int(v)*m for v,m in zip(elapsed.split(':'),(3600,60,1)))
    result['allocation_end_epoch']=min(DEADLINE,time.time()+request['walltime_s']-elapsed_s)
    if continuation is not None:
        result['allocation_end_epoch']=min(result['allocation_end_epoch'],continuation['allocation_end_epoch'])
        result['continued_allocation']=continuation
    remaining = admission_request['predicted_remaining_peak_bytes']
    if remaining <= 0 or owned+remaining > 180*GIB or shutil.disk_usage(root).free < remaining+20*GIB:
        raise PermissionError('HOPPER_PEAK_SPACE_BOUND')
    result['request_sha256'] = sha(request_path)
    result['project_attestation_sha256']=request['project_admission_sha256']
    result.update(current_owned_bytes=owned,predicted_remaining_peak_bytes=remaining)
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
    if request['mode']=='continuation':
        command=[request['python'],'-m','scripts.pi0294.continuation_session',
            '--root',str(root),'--request',str(request_path),'--bindings',request['bindings']]
        os.execv(request['python'],command)
    if request['mode']=='night-final':
        command=[request['python'],'-m','scripts.pi0294.night_session','--root',str(root),
            '--request',str(request_path),'--bindings',request['bindings'],
            '--microbatch',str(request.get('microbatch',2)),'--workers',str(request.get('workers',2))]
        os.execv(request['python'],command)
    if request['mode']=='resident':
        if request.get('resident_operation')=='TRAIN_DIAGNOSTIC':
            from .continuation import RUN as CONTINUATION_RUN
            if root.name!=CONTINUATION_RUN:
                raise PermissionError('DIAGNOSTIC_ONLY_UNDER_NEW_EXPLICIT_AUTHORITY')
            command=[request['python'],'-m','scripts.pi0294.policy_diagnostic',
                '--root',str(root),'--request',str(request_path),'--bindings',request['bindings']]
            os.execv(request['python'],command)
        command=[request['python'],'-m','scripts.pi0294.resident','--root',str(root),
            '--request',str(request_path),'--bindings',request['bindings']]
        for key in ('checkpoint','registry','identity','directory'):
            command+=['--'+key,request[key]]
        os.execv(request['python'],command)
    if request['mode']=='checkpoint':
        command=[request['python'],'-m','scripts.pi0294.checkpoint_session','--root',str(root),
            '--request',str(request_path)]
        os.execv(request['python'],command)
    from .continuation import RUN as CONTINUATION_RUN
    if root.name==CONTINUATION_RUN and request['mode']=='main':
        command=[request['python'],'-m','scripts.pi0294.r3_offline','--root',str(root),
                 '--request',str(request_path),'--bindings',request['bindings'],'--resume',request['resume']]
        os.execv(request['python'],command)
    command=[request['python'],'-m','scripts.pi0294.neural','--root',str(root),
        '--request',str(request_path),'--bindings',request['bindings'],'--mode',request['mode'],
        '--microbatch',str(request.get('microbatch',2)),'--workers',str(request.get('workers',2)),
        '--max-steps',str(request.get('max_steps',4096)),'--stop-at',str(request.get('stop_at',4096))]
    if request.get('resume'):command+=['--resume',request['resume']]
    os.execv(request['python'],command)


if __name__ == '__main__':
    main()
