"""Last-allocation composition around unchanged sealed training/evaluation.

Not a submitter. Admission requires actual successful S3 and a separately sealed
real relay endpoint. CPU fixtures cannot satisfy these runtime requirements.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import socket
import subprocess
import time

from .allocation_admission import PARALLEL_AUTHORITY, current_project, owned_jobs, parallel_peer_jobs, personal_quota
from .durable import OwnerLock, read, write
from .pbs_smoke import pbs_epoch, seconds
from .smoke_capacity_receipt import build
from .train_adapter import sha

SOURCE = 'c187d9573bdff302783fb5f41d06fcaa1753d2a8'
GIB = 1024**3


def admit(*, request, adoption, ledger, job_id, jobs, user, host, project, quota, plan, now):
    if (adoption['release'] != '261bc5b37f20aaff1ad7d5742252d3dc1a91fa65'
            or adoption['T0_epoch'] != 1791860304 or adoption['new_work_cutoff_epoch'] != 1791943104
            or adoption['deadline_epoch'] != 1791946704 or adoption['reset_on_resume']):
        raise PermissionError('ORIGINAL_SCIENCE_AUTHORITY_AND_CLOCK_REQUIRED')
    if (request['request_id'] != 'S4-S6-main-eval-01' or request['stage'] != 'main-eval'
            or request['source_sha'] != SOURCE or request['parallel_authority'] != PARALLEL_AUTHORITY
            or type(request['walltime_s']) is not int or not 300 < request['walltime_s'] <= 28800):
        raise PermissionError('EXACT_LAST_ALLOCATION_REQUEST_REQUIRED')
    if len(ledger['PBS_submissions']) != 3:
        raise PermissionError('THIRD_AND_LAST_SUBMISSION_ONLY')
    prior = ledger['PBS_submissions'][:2]
    if (prior[0]['request_id'] != 'S3-smoke-01' or prior[0]['status'] != 'SUBMIT_FAILED_PRESERVED'
            or prior[0].get('job_id') is not None or prior[1]['request_id'] != 'S3-smoke-02'
            or prior[1]['status'] != 'RECONCILED_TERMINAL'):
        raise PermissionError('EXACT_S3_FAILURE_AND_SUCCESS_CHAIN_REQUIRED')
    if any(row.get('GPU_reserved_s', 0) or row.get('CPU_reserved_s', 0) for row in prior):
        raise PermissionError('PREVIOUS_ALLOCATIONS_MUST_BE_RECONCILED')
    entry = ledger['PBS_submissions'][-1]
    if (entry['request_id'] != request['request_id'] or entry['job_id'] != job_id
            or entry['status'] != 'SUBMITTED' or ledger['main_updates'] != 0 or ledger.get('pending_updates', 0)):
        raise PermissionError('EXACT_SUBMITTED_NEW_PRISTINE_MAIN_REQUIRED')
    if entry.get('GPU_reserved_s') != request['walltime_s'] or entry.get('CPU_reserved_s') != 4*request['walltime_s']:
        raise PermissionError('DURABLE_LAST_ALLOCATION_RESERVATIONS_REQUIRED')
    peers = {key: value for key, value in jobs.items() if key != job_id}
    parallel_peer_jobs(jobs=peers, user=user, allowed_ids=request['parallel_peer_ids'], authority=request['parallel_authority'])
    if job_id not in jobs:
        raise PermissionError('EXACT_RUNNING_MAIN_JOB_REQUIRED')
    job = jobs[job_id]
    if (job['job_state'] != 'R' or job['Job_Owner'].split('@')[0] != user
            or job['project'] != 'CFP04-CF-007' or int(job['Resource_List']['ngpus']) != 1
            or seconds(job['Resource_List']['walltime']) != request['walltime_s']
            or host.split('.')[0] not in {part.split('/')[0].split('.')[0] for part in job['exec_host'].split('+')}):
        raise PermissionError('OWN_PROJECT_SINGLE_GPU_COMPUTE_NODE_REQUIRED')
    p = project['project']
    from datetime import date, datetime, timedelta, timezone
    today = datetime.fromtimestamp(now, timezone(timedelta(hours=8))).date()
    if (p['name'] != 'CFP04-CF-007' or p['active'] is not True or user not in p['users']
            or 'hopper-m-02' not in p['cluster'] or not date.fromisoformat(p['start_date']) <= today <= date.fromisoformat(p['end_date'])
            or not 0 <= now-project['observed_epoch'] <= 1800
            or type(project['balance_su']) not in (int,float) or not 0 < project['balance_su'] < float('inf')):
        raise PermissionError('FRESH_MAIN_PROJECT_MEMBERSHIP_DATES_BALANCE_REQUIRED')
    if (quota['user'] != user or not 0 <= now-quota['report_epoch'] <= 900
            or any(type(quota[k]) is not int or quota[k] < 0 for k in ('used_bytes','hard_bytes'))
            or quota['hard_bytes']-quota['used_bytes'] < plan['required_free_headroom_bytes']):
        raise PermissionError('FRESH_PERSONAL_MAIN_QUOTA_REQUIRED')
    if (plan['status'] != 'CAPACITY_ESTIMATE_PASSES_REQUIRES_BOUND_LIVE_ADMISSION'
            or plan['blockers'] or not plan['main_and_eval_share_next_allocation']
            or plan['main_optimizer_updates'] != 4096 or plan['effective_batch'] != 32
            or plan['combined_walltime_seconds'] > request['walltime_s']
            or plan['remaining_peak_bytes'] != request['predicted_remaining_peak_bytes']):
        raise PermissionError('ACTUAL_FULL_4096_AND_EVAL_CAPACITY_REQUIRED')
    wall = request['walltime_s']
    if ledger['GPU_allocated_s']+wall > 8*3600:
        raise PermissionError('ORIGINAL_8_HOUR_ALLOCATED_GPU_BUDGET')
    stages = sum(row.get('CPU_reserved_s', 0) for row in ledger['stages'])
    if ledger['CPU_s']+ledger.get('CPU_upper_s', 0)+stages+4*wall > 64*3600:
        raise PermissionError('ORIGINAL_HOPPER_CPU_BUDGET')
    start = pbs_epoch(job['stime'])
    if start+wall >= adoption['new_work_cutoff_epoch'] or now >= start+wall-300:
        raise PermissionError('ORIGINAL_CLOCK_AND_ALLOCATION_WINDOW')
    return dict(status='LAST_ALLOCATION_LIVE_MAIN_ADMISSION_PASS',job_id=job_id,
        allocation_start_epoch=start,allocation_end_epoch=start+wall,
        prior_GPU_allocated_s=ledger['GPU_allocated_s'],observed_epoch=now)


def capacity_ledger_view(ledger):
    """Exclude only this just-started allocation from the prior-smoke planner."""
    import copy
    view = copy.deepcopy(ledger)
    if len(view['PBS_submissions']) != 3 or view['PBS_submissions'][-1]['stage'] != 'main-eval':
        raise PermissionError('EXACT_THIRD_SUBMISSION_REQUIRED')
    view['PBS_submissions'] = view['PBS_submissions'][:2]
    return view


def run(args):
    root = Path(args.root)
    request_path = Path(args.request)
    request = read(request_path)
    if (root.name != '20261013T025824Z-localpc-pi0292'
            or not request_path.resolve().is_relative_to(root.resolve()) or not os.environ.get('PBS_JOBID')):
        raise PermissionError('EXISTING_REGISTERED_RUN_AND_PBS_REQUIRED')
    owner = OwnerLock(root/'PBS_OWNER.lock')
    if (sha(__file__) != request['wrapper_sha256']
            or sha(request_path.parent/'RUN_MANIFEST.json') != request['run_manifest_sha256']):
        raise PermissionError('FIXED_LAST_ALLOCATION_WRAPPER_AND_MANIFEST_REQUIRED')
    state_path = root/'evidence/PBS_MAIN_EVAL_S4-S6-main-eval-01.json'
    if state_path.exists():
        raise FileExistsError('Inspect the original allocation; never restart wrapper')
    operations = Path(request['operations'])
    seal = read(operations/'OPERATIONS_SEAL.json')
    if seal['source_sha'] != SOURCE:
        raise PermissionError('UNCHANGED_SCIENTIFIC_SOURCE_REQUIRED')
    for name, digest in seal['files'].items():
        if sha(operations/name) != digest:
            raise PermissionError('SEALED_SCIENTIFIC_SOURCE_CHANGED')
    bindings = read(request['bindings'])
    if sha(request['bindings']) != request['bindings_sha256']:
        raise PermissionError('CURRENT_QUALIFIED_BINDINGS_REQUIRED')
    controller = Path(request['controller_operations'])
    for name,digest in request['controller_files'].items():
        path = controller/name
        if not path.resolve().is_relative_to(controller.resolve()) or sha(path) != digest:
            raise PermissionError('OUTER_CONTROLLER_CHANGED')
    # S6 must have an actual presealed endpoint before spending the final call.
    relay = controller/'scripts/pi0292/neural_relay.py'
    if not relay.is_file() or 'scripts/pi0292/neural_relay.py' not in request['controller_files']:
        raise PermissionError('REAL_RELAY_ENDPOINT_NOT_IMPLEMENTED_OR_SEALED')
    endpoint_receipt_path = Path(request['relay_endpoint_qualification'])
    if (not endpoint_receipt_path.resolve().is_relative_to(root.resolve())
            or sha(endpoint_receipt_path) != request['relay_endpoint_qualification_sha256']):
        raise PermissionError('HASH_BOUND_RELAY_IMPLEMENTATION_QUALIFICATION_REQUIRED')
    endpoint_receipt = read(endpoint_receipt_path)
    if (endpoint_receipt['status'] != 'PASS_REAL_RELAY_ENDPOINT_CPU_QUALIFICATION_NOT_PHYSICS_OR_MODEL'
            or endpoint_receipt['endpoint_sha256'] != sha(relay)
            or endpoint_receipt['implementation'] != 'ACTUAL_ENDPOINT_NOT_FIXTURE'
            or endpoint_receipt['original_T0'] != 1791860304):
        raise PermissionError('REAL_ENDPOINT_IMPLEMENTATION_AND_CPU_QUALIFICATION_REQUIRED')
    lease = read(root/'DEPENDENCY_LEASE.json')
    if (lease['status'] != 'ACTIVE' or lease['expires'] != 1791946704 or not lease['no_move_or_delete']
            or sha(root/'DEPENDENCY_LEASE.json') != request['lease_sha256']):
        raise PermissionError('CURRENT_PROTECTED_DEPENDENCIES_REQUIRED')
    for _ in range(20):
        ledger = read(root/'HOPPER_RESOURCE_LEDGER.json')
        if ledger['PBS_submissions'][-1].get('job_id') == os.environ['PBS_JOBID']:
            break
        time.sleep(.5)
    else:
        raise PermissionError('EXACT_SUBMISSION_RECEIPT_PENDING_NO_MODEL_LOAD')
    # build() hashes the real ledger; use a read-only logical view of prior
    # submissions for capacity, never rewrite the authoritative ledger to fit.
    quota = personal_quota(os.environ['USER'])
    plan = build(root, owned_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink()),
        filesystem_free_bytes=shutil.disk_usage(root).free,quota_free_bytes=quota['hard_bytes']-quota['used_bytes'],
        extra_peak_bytes=request['extra_peak_bytes'],now=time.time(),prior_ledger=capacity_ledger_view(ledger))
    admitted = admit(request=request,adoption=read(root/'ADOPTION.json'),ledger=ledger,
        job_id=os.environ['PBS_JOBID'],jobs=owned_jobs(os.environ['USER']),user=os.environ['USER'],
        host=socket.gethostname(),project=current_project(),quota=quota,plan=plan,now=time.time())
    write(root/'evidence/MAIN_CAPACITY_PREFLIGHT.json',dict(plan,
        status='PASS_FULL_4096_WITHIN_REMAINING_BUDGET',live_admission=admitted))
    end = admitted['allocation_end_epoch']
    write(root/'NEURAL_AUTHORIZATION.json',dict(job_id=os.environ['PBS_JOBID'],allowed_modes=['main','offline','relay'],
        allocation_end_epoch=end,allocation_walltime_s=request['walltime_s'],prior_GPU_allocated_s=ledger['GPU_allocated_s'],
        predicted_remaining_peak_bytes=plan['remaining_peak_bytes'],source_sha=SOURCE))
    entry = ledger['PBS_submissions'][-1]
    entry.update(status='RUNNING',allocation_start_epoch=admitted['allocation_start_epoch'],allocation_end_epoch=end)
    write(root/'HOPPER_RESOURCE_LEDGER.json',ledger)
    state = dict(status='RUNNING_MAIN_EVAL',pid=os.getpid(),job_id=os.environ['PBS_JOBID'],
        proc_start=Path('/proc/self/stat').read_text().split()[21],started_epoch=time.time(),steps=[],current=None)
    write(state_path,state)
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    env = dict(os.environ,PYTHONPATH=str(operations)+':'+str(root/'data/bank_v2/frozen_source/src'),
        PATH='/cm/shared/apps/pbspro/current/bin:'+os.environ.get('PATH',''),
        PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',NO_ALBUMENTATIONS_UPDATE='1')
    child = None
    def stop(signum,frame):
        raise TimeoutError('OWN_MAIN_EVAL_ALLOCATION_CLOSEOUT')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop)
    signal.alarm(max(1,int(end-30-time.time())))
    def execute(stage,command,environment):
        nonlocal child
        if time.time() >= end-180:
            raise TimeoutError('RESERVE_ALLOCATION_CLOSEOUT')
        state['current']=stage;write(state_path,state);began=time.time()
        with (root/'logs'/('main-eval-'+stage+'.stdout')).open('xb') as out, (root/'logs'/('main-eval-'+stage+'.stderr')).open('xb') as err:
            child=subprocess.Popen(command,cwd=operations,env=environment,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
            state.update(child_pid=child.pid,child_start=Path('/proc').joinpath(str(child.pid),'stat').read_text().split()[21]);write(state_path,state)
            if stage!='live-relay':
                code=child.wait(timeout=max(1,end-90-time.time()))
            else:
                while child.poll() is None:
                    if time.time()>=end-90:raise TimeoutError('LIVE_SESSION_ALLOCATION_CLOSEOUT')
                    query_state=root/'relay/neural-session-01/QUERY_STATE.json'
                    if query_state.exists():
                        pending=read(query_state).get('pending')
                        if pending and time.time()>=pending['expires']+1:
                            raise TimeoutError('OWN_LIVE_QUERY_CPU_WATCHDOG_60_SECOND_LIMIT')
                    time.sleep(.1)
                code=child.returncode
        child=None
        state['steps'].append(dict(stage=stage,exit=code,wall_s=time.time()-began));write(state_path,state)
        if code:
            raise RuntimeError('MAIN_EVAL_CHILD_FAILED_'+stage)
    try:
        execute('main',[request['python'],'-m','scripts.pi0292.neural_train','--root',str(root),'--bindings',request['bindings'],'--mode','main'],env)
        training=read(root/'training/main/session-start/TRAINING_RUN.json')
        if training['status']!='MAIN_COMPLETED_FIXED_S4096' or training['global_step']!=4096:
            state['status']='MAIN_PARTIAL_B1_NOT_AVAILABLE'
            return state
        from .checkpoint_artifacts import inventory
        data=read(root/'evidence/HOPPER_DATA_QUALIFICATION.json')
        checkpoints=[inventory(root/'training/main'/('checkpoint-'+str(step)),expected_step=step,recipe=training['recipe'],
            statistics_sha256=data['statistics_sha256'],manifest_sha256=data['manifest_sha256']) for step in (2048,4096)]
        write(root/'evidence/MAIN_FULL_CHECKPOINT_INVENTORIES.json',dict(checkpoints=checkpoints,selected='NONE',deployment='NONE'))
        for profile in ('B0','S2048','S4096'):
            execute('offline-'+profile,[request['python'],'-m','scripts.pi0292.generate_probes',
                '--root',str(root),'--bindings',request['bindings'],'--profile',profile],env)
        relay_env=dict(env,PYTHONPATH=str(controller)+':'+str(root/'data/bank_v2/frozen_source/src'))
        execute('paired-offline-analysis',[request['python'],'-m','scripts.pi0292.compare_profiles',
            '--root',str(root),'--frozen-code',str(root/'data/bank_v2/frozen_source')],relay_env)
        execute('live-relay',[request['python'],'-m','scripts.pi0292.neural_relay',
            '--root',str(root),'--bindings',request['bindings'],'--request',str(request_path)],relay_env)
        # The real relay endpoint reports gates and attempted/remaining cases;
        # allocation completion is never itself a task-success declaration.
        state['status']='MAIN_OFFLINE_AND_RELAY_TERMINAL_REPORT_PENDING'
        return state
    except BaseException as error:
        state.update(status='FAILED_PRESERVED_NO_AUTOMATIC_RETRY',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        signal.alarm(0)
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        u=resource.getrusage(resource.RUSAGE_SELF);c=resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu=u.ru_utime+u.ru_stime+c.ru_utime+c.ru_stime
        state.update(finished_epoch=time.time(),CPU_s=cpu,child_pid=None);write(state_path,state)
        ledger=read(root/'HOPPER_RESOURCE_LEDGER.json');entry=ledger['PBS_submissions'][-1]
        entry.update(status='WRAPPER_TERMINAL_SCHEDULER_RECONCILIATION_PENDING',wrapper_status=state['status'],
            CPU_s=cpu,CPU_reserved_s=0,GPU_reserved_s=request['walltime_s'])
        ledger['CPU_s']+=cpu;write(root/'HOPPER_RESOURCE_LEDGER.json',ledger);owner.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--request',required=True)
    print(json.dumps(run(parser.parse_args())))
