"""Bounded Vanda host/software qualification. No policy or robot execution.

The private manifest is supplied by the existing local-PC owner. This is an
engineering bootstrap adapter, not the E0 launcher or a scientific evaluator.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def validate(manifest, *, compute=False):
    m = read(manifest)
    root = Path(m['output_root']).resolve()
    assert root == Path(manifest).resolve().parent, 'OUTPUT_BINDING'
    assert root.is_relative_to(Path(m['registered_runs_root']).resolve()), 'OUTPUT_ESCAPE'
    assert m['owner_uid'] == os.getuid(), 'OWNER_UID'
    assert m['authority']['direct_user_project_gpu_authorization'], 'AUTHORITY'
    assert time.time() < m['deadline_epoch'], 'EXPIRED'
    assert m['execution']['profile'] == 'vanda-qualification', 'PROFILE'
    assert m['scope'] == 'HOST_SOFTWARE_ONLY_NO_MODEL_OR_ROBOT', 'SCOPE'
    assert m['limits']['ngpus'] == 1 and m['limits']['walltime_s'] <= 900, 'BOUNDS'
    assert read(m['lease'])['run_uuid'] == m['run_uuid'], 'LEASE_ID'
    assert read(m['lease'])['status'] == 'ACTIVE', 'LEASE_INACTIVE'
    for path, expected in m['sealed_files'].items():
        assert digest(path) == expected, 'SOURCE_SEAL:' + path
    host = socket.gethostname().split('.')[0].lower()
    if compute:
        assert host.startswith('gn-a40-'), 'COMPUTE_HOST_REQUIRED'
        jobid = os.environ.get('PBS_JOBID')
        assert jobid, 'PBS_REQUIRED'
        result = subprocess.run(['/opt/pbs/bin/qstat', '-f', '-F', 'json', jobid],
                                capture_output=True, text=True, check=True, timeout=20)
        job = json.loads(result.stdout)['Jobs'][jobid]
        assert job['job_state'] == 'R' and job['project'] == m['project'], 'RUNNING_PROJECT'
        assert job['Job_Owner'].split('@')[0] == m['owner_user'], 'PBS_OWNER'
        assert int(job['Resource_List']['ngpus']) == 1, 'PBS_GPU_LIMIT'
        hours, mins, secs = map(int, job['Resource_List']['walltime'].split(':'))
        assert hours * 3600 + mins * 60 + secs <= 900, 'PBS_TIME_LIMIT'
        submission = read(root / 'SUBMISSION.private.json')
        assert submission['job_id'] == jobid, 'JOB_ID_BINDING'
    else:
        assert host in m['execution']['login_hosts'], 'LOGIN_HOST_REQUIRED'
    return m


def query(argv, timeout=90, env=None):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
    return dict(argv=argv, exit_code=result.returncode,
                stdout=result.stdout[-24000:], stderr=result.stderr[-4000:])


def execute(manifest):
    m = validate(manifest, compute=True)
    root = Path(m['output_root'])
    with (root / 'QUALIFICATION_OWNER.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (root / 'RESULT.private.json').exists(), 'DO_NOT_REPEAT_COMPLETED_RUN'
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:4]))
        resource.setrlimit(resource.RLIMIT_CPU, (1800, 1800))
        started = time.time()
        rows = []
        result = dict(run_uuid=m['run_uuid'], job_id=os.environ['PBS_JOBID'],
                      hostname=socket.gethostname(), started_epoch=started,
                      source_sha=m['code']['vla_commit'], checks=rows,
                      model_loads=0, policy_queries=0, optimizer_steps=0,
                      robot_physics_steps=0, WBC_launches=0)
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        try:
            commands = [
                ('gpu', ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap', '--format=csv']),
                ('cpu', ['lscpu']),
                ('ram', ['free', '-b']),
                ('cuda_torch', [m['python'], '-c',
                  'import torch,json; '
                  'print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,'
                  'available=torch.cuda.is_available(),count=torch.cuda.device_count(),'
                  'device=torch.cuda.get_device_name(0),capability=torch.cuda.get_device_capability(0)))); '
                  'x=torch.ones((64,64),device="cuda"); y=x@x; torch.cuda.synchronize(); '
                  'assert y[0,0].item()==64; print("CUDA_MATMUL_PASS")']),
                ('imports', [m['python'], '-c',
                  'import importlib,json; r={}; '
                  '\nfor n in ["torchvision","transformers","flash_attn","gr00t","mujoco","OpenGL"]:'
                  '\n try: r[n]=dict(file=importlib.import_module(n).__file__)'
                  '\n except Exception as e: r[n]=dict(error=type(e).__name__+":"+str(e))'
                  '\nprint(json.dumps(r))']),
            ]
            for label, command in commands:
                rows.append(dict(check=label, **query(command, env=env)))
            for image in m['containers']:
                command = [m['apptainer'], 'exec', '--cleanenv', '--nv', image,
                           '/bin/bash', '-c',
                           'cat /etc/os-release; ls /opt/ros 2>/dev/null; '
                           'ldconfig -p 2>/dev/null | grep -E "libEGL|libGLX|libOpenGL"; '
                           'command -v cmake; command -v colcon; command -v g++']
                rows.append(dict(check='container', **query(command, timeout=90, env=env)))
            result['status'] = 'HOST_CHECKS_COMPLETED_SOFTWARE_GAPS_RECORDED'
        except BaseException as error:
            result.update(status='ENGINEERING_INCOMPLETE', error=type(error).__name__ + ':' + str(error))
            raise
        finally:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            children = resource.getrusage(resource.RUSAGE_CHILDREN)
            result.update(completed_epoch=time.time(), allocated_elapsed_s=time.time()-started,
                          cpu_s=usage.ru_utime+usage.ru_stime+children.ru_utime+children.ru_stime)
            path = root / 'RESULT.private.json'
            with path.open('x') as output:
                json.dump(result, output, indent=2)
                output.write('\n')
            lease = read(m['lease'])
            lease.update(status='CLOSED', result_status=result['status'], completed_epoch=time.time())
            Path(m['lease']).write_text(json.dumps(lease, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        result = execute(args.manifest)
    else:
        m = validate(args.manifest)
        result = dict(status='MANIFEST_VALIDATED_NO_COMPUTE', run_uuid=m['run_uuid'])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
