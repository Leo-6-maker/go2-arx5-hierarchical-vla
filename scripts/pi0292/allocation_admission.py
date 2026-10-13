"""Stdlib-only outer admission for the unchanged sealed S3 allocation wrapper."""
import ast
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = 'CFP04-CF-007'
RELEASE = '261bc5b37f20aaff1ad7d5742252d3dc1a91fa65'
SOURCE = 'c187d9573bdff302783fb5f41d06fcaa1753d2a8'
GIB = 1024**3
PBS = '/cm/shared/apps/pbspro/current/bin/'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def owned_jobs(user, query=subprocess.run):
    """The user-filtered qstat view is not JSON; select IDs, then read each ID."""
    result = query([PBS+'qselect', '-u', user], capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise PermissionError('OWN_JOB_SELECTION_UNKNOWN')
    jobs = {}
    for job_id in result.stdout.split():
        if job_id in jobs:
            raise ValueError('DUPLICATE_SCHEDULER_JOB_ID')
        row = query([PBS+'qstat', '-f', '-F', 'json', job_id],
                    capture_output=True, text=True, timeout=20)
        if row.returncode:
            raise PermissionError('SCHEDULER_CHANGED_REPEAT_READ_ONLY_PREFLIGHT')
        job = json.loads(row.stdout)['Jobs'][job_id]
        if job['Job_Owner'].split('@')[0] != user:
            raise PermissionError('FOREIGN_JOB_IN_OWN_SELECTION')
        if job['job_state'] not in ('F', 'X'):
            jobs[job_id] = job
    return jobs


def personal_quota(user, path='/home/svu/ccepbs/nushpc/quota_hw_report/hpfs_quota_raw.txt'):
    path = Path(path)
    rows = []
    with path.open() as handle:
        for line in handle:
            fields = line.split()
            if '/'+user in fields:
                rows.append(fields)
    if len(rows) != 1 or rows[0][3] != '/'+user:
        raise PermissionError('EXACT_OWN_PERSONAL_QUOTA_ROW_REQUIRED')
    return dict(user=user, used_bytes=int(rows[0][4]), hard_bytes=int(rows[0][5]),
                report_epoch=path.stat().st_mtime)


def current_project(query=subprocess.check_output):
    projects = json.loads(query(['bash', '-lc', 'amgr ls project --name CFP04-CF-007 --json-info'],
                                text=True, timeout=30))
    balance = ast.literal_eval(query(['bash', '-lc', 'amgr checkbalance project --name CFP04-CF-007'],
                                     text=True, timeout=30))
    if len(projects) != 1:
        raise PermissionError('EXACT_CURRENT_PROJECT_REQUIRED')
    return dict(project=projects[0], balance_su=balance['su'], observed_epoch=time.time())


def validate(*, now, user, job_id, jobs, quota, adoption, project_receipt,
             dependency, request, binding_sha256):
    if (adoption['release'] != RELEASE or adoption['T0_epoch'] != 1791860304
            or adoption['new_work_cutoff_epoch'] != 1791943104
            or adoption['deadline_epoch'] != 1791946704 or adoption['reset_on_resume']):
        raise PermissionError('ORIGINAL_AUTHORITY_AND_CLOCK_REQUIRED')
    if now >= adoption['new_work_cutoff_epoch']:
        raise PermissionError('ORIGINAL_NEW_WORK_CUTOFF')
    if set(jobs) != {job_id}:
        raise PermissionError('NO_OVERLAPPING_OWN_ALLOCATION')
    job = jobs[job_id]
    if (job['job_state'] != 'R' or job['Job_Owner'].split('@')[0] != user
            or job.get('project') != PROJECT or int(job['Resource_List']['ngpus']) != 1
            or job['Resource_List']['walltime'] != '00:30:00'):
        raise PermissionError('EXACT_RUNNING_PROJECT_SINGLE_GPU_SMOKE_ALLOCATION_REQUIRED')
    project = project_receipt['project']
    day = datetime.fromtimestamp(now, timezone(timedelta(hours=8))).date()
    if (project['name'] != PROJECT or project['active'] is not True
            or user not in project['users'] or 'hopper-m-02' not in project['cluster']
            or not date.fromisoformat(project['start_date']) <= day <= date.fromisoformat(project['end_date'])
            or not 0 <= now-project_receipt['observed_epoch'] <= 1800
            or type(project_receipt['balance_su']) not in (int, float)
            or not 0 < project_receipt['balance_su'] < float('inf')):
        raise PermissionError('FRESH_PROJECT_MEMBERSHIP_BALANCE_AND_DATES_REQUIRED')
    if (quota['user'] != user or not 0 <= now-quota['report_epoch'] <= 900
            or any(type(quota[k]) is not int or quota[k] < 0 for k in ('used_bytes', 'hard_bytes'))
            or quota['hard_bytes']-quota['used_bytes'] < 100*GIB):
        raise PermissionError('FRESH_100_GIB_PERSONAL_QUOTA_HEADROOM_REQUIRED')
    if (dependency['status'] != 'PASS_MIGRATED_DEPENDENCY_FILES_AND_CPU_PROCESSOR'
            or dependency['model_qualification'] is not False
            or dependency['bindings_sha256'] != binding_sha256
            or dependency['original_T0'] != adoption['T0_epoch']
            or dependency['original_deadline'] != adoption['deadline_epoch']):
        raise PermissionError('ACTUAL_MIGRATED_DEPENDENCY_QUALIFICATION_REQUIRED')
    if (request['request_id'] != 'S3-smoke-02' or request['source_sha'] != SOURCE
            or request['stage'] != 'smoke' or request['walltime_s'] != 1800
            or request['predicted_remaining_peak_bytes'] != 80*GIB):
        raise PermissionError('EXACT_UNCHANGED_S3_RETRY_REQUIRED')
    return dict(status='OUTER_S3_ALLOCATION_ADMISSION_PASS_NOT_MODEL_QUALIFICATION',
                job_id=job_id, project=PROJECT, observed_epoch=now,
                personal_free_bytes=quota['hard_bytes']-quota['used_bytes'])


def main():
    request_path = Path(sys.argv[1])
    request = read(request_path)
    root = Path(request['root'])
    if root.name != '20261013T025824Z-localpc-pi0292' or not request_path.resolve().is_relative_to(root.resolve()):
        raise PermissionError('EXISTING_REGISTERED_RUN_REQUIRED')
    if sha(__file__) != request['guard_sha256']:
        raise PermissionError('OUTER_GUARD_HASH_CHANGED')
    for name, key in [('SUBMISSION_RUN_MANIFEST.json', 'run_manifest_sha256'),
                      ('PATH_ADOPTION_DELTA.json', 'adoption_delta_sha256')]:
        if sha(request_path.parent/name) != request[key]:
            raise PermissionError('BOUND_MANIFEST_OR_ADOPTION_DELTA_CHANGED')
    user = os.environ['USER']
    job_id = os.environ.get('PBS_JOBID')
    if not job_id:
        raise PermissionError('NO_MODEL_ADMISSION_ON_LOGIN')
    dependency_path = root/'evidence/dependency-adoption-02/STATE.json'
    if sha(dependency_path) != request['dependency_qualification_sha256']:
        raise PermissionError('DEPENDENCY_RECEIPT_CHANGED')
    project_path = request_path.parent/'PROJECT_ADMISSION.private.json'
    if sha(project_path) != request['project_admission_sha256']:
        raise PermissionError('PROJECT_RECEIPT_CHANGED')
    lease = read(root/'DEPENDENCY_LEASE.json')
    if (lease['status'] != 'ACTIVE' or lease['expires'] != 1791946704
            or not lease['no_move_or_delete'] or str(root) not in lease['protected']
            or sha(root/'DEPENDENCY_LEASE.json') != request['lease_sha256']):
        raise PermissionError('CURRENT_DEPENDENCY_LEASE_REQUIRED')
    # Queue time does not extend a login-node project observation. Re-query
    # inside the actual allocation; unavailable authentication fails closed.
    project = current_project()
    result = validate(now=time.time(), user=user, job_id=job_id, jobs=owned_jobs(user),
        quota=personal_quota(user), adoption=read(root/'ADOPTION.json'),
        project_receipt=project, dependency=read(dependency_path), request=request,
        binding_sha256=sha(request['bindings']))
    destination = request_path.parent/'ALLOCATION_ADMISSION.json'
    with destination.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.flush(); os.fsync(handle.fileno())
    # Only after CPU admission, replace this process with the original sealed
    # launcher. It independently checks job, node, H200, code, data and budgets.
    os.execv(request['python'], [request['python'], '-m', 'scripts.pi0292.pbs_smoke',
                               '--root', str(root), '--request', str(request_path)])


if __name__ == '__main__':
    main()
