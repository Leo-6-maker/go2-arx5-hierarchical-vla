"""PBS-only E0 resident launcher with a fresh owner release and CPU closeout."""
import argparse
import os
from pathlib import Path
import resource
import subprocess
import time

from scripts.pi0292.allocation_admission import owned_jobs
from scripts.pi0292.durable import read, write
from .e0_scope import RUN, ADMISSION_END, validate_entry
from .native_dataset import sha


def run(root, request_path):
    root, request_path = Path(root), Path(request_path)
    validate_entry(root, profile='hopper-neural')
    job_id = os.environ.get('PBS_JOBID')
    if not job_id:
        raise PermissionError('E0_PBS_LAUNCHER_ONLY')
    job = owned_jobs(os.environ['USER']).get(job_id)
    if not job or job['job_state'] != 'R' or job.get('project') != 'CFP04-CF-007':
        raise PermissionError('E0_NATIVE_RUNNING_OWN_PROJECT_JOB_REQUIRED')
    if time.time() >= ADMISSION_END:
        raise PermissionError('E0_NEW_ALLOCATION_AFTER_ADMISSION_CUTOFF')
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:4]))
    ledger = read(root/'HOPPER_RESOURCE_LEDGER.json')
    remaining = int(86400-ledger['CPU_s'])
    if remaining <= 0:
        raise PermissionError('E0_HOPPER_CPU_CAP')
    resource.setrlimit(resource.RLIMIT_CPU, (remaining, remaining))
    directory = request_path.parent
    ready = dict(status='WAIT_FRESH_OWNER_INFERENCE_RELEASE', run_uuid=RUN, job_id=job_id,
        pid=os.getpid(), proc_start=Path('/proc/self/stat').read_text().split()[21],
        observed_epoch=time.time(), model_loaded=False)
    write(directory/'PBS_READY.json', ready)
    cpu0, child0 = time.process_time(), resource.getrusage(resource.RUSAGE_CHILDREN)
    status, result = 'FAILED_PRESERVED', None
    try:
        release_path = directory/'START_INFERENCE.json'
        end = time.time()+180
        while not release_path.exists() and time.time() < end:
            time.sleep(.2)
        if not release_path.exists():
            raise TimeoutError('E0_OWNER_NOT_READY_NO_IDLE_GPU_MODEL_LOAD')
        release = read(release_path)
        if (release['run_uuid'] != RUN or release['job_id'] != job_id
                or release['request_sha256'] != sha(request_path)):
            raise PermissionError('E0_EXACT_OWN_JOB_AND_REQUEST_RELEASE')
        request = read(request_path)
        if request['allowed_modes'] != ['resident'] or request['mode'] != 'resident':
            raise PermissionError('E0_CANNOT_DISPATCH_TRAINING')
        argv = [request['python'], '-m', 'scripts.pi0294.resident', '--root', str(root),
            '--request', str(request_path), '--bindings', request['bindings'],
            '--checkpoint', request['checkpoint'], '--identity', request['identity'],
            '--registry', request['registry'], '--directory', request['directory']]
        result = subprocess.run(argv, check=False)
        status = 'OWN_E0_RESIDENT_TERMINAL' if result.returncode == 0 else 'RESIDENT_FAILURE_PRESERVED'
        return result.returncode
    finally:
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = time.process_time()-cpu0+child.ru_utime+child.ru_stime-child0.ru_utime-child0.ru_stime
        ledger = read(root/'HOPPER_RESOURCE_LEDGER.json')
        ledger['CPU_s'] += cpu
        write(root/'HOPPER_RESOURCE_LEDGER.json', ledger)
        write(directory/'PBS_PROCESS_FINAL.json', dict(status=status, run_uuid=RUN, job_id=job_id,
            completed_epoch=time.time(), CPU_s=cpu, exit=None if result is None else result.returncode,
            new_training_updates=0))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--request', required=True)
    a = p.parse_args()
    raise SystemExit(run(a.root, a.request))
