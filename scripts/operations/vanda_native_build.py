"""PBS CPU-only build of the user-selected frozen native WBC archive.

Never starts a ROS node, simulator, policy forward, or optimizer. Every
external command and result stays in the registered private build run.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import socket
import subprocess
import tarfile
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def validate(path, execute=False):
    m = read(path)
    root = Path(m['output_root']).resolve()
    assert Path(path).resolve().parent == root, 'OUTPUT_BINDING'
    assert root.is_relative_to(Path(m['registered_runs_root']).resolve()), 'OUTPUT_ESCAPE'
    assert m['owner_uid'] == os.getuid() and m['owner'] == 'EXISTING_LOCAL_PC', 'OWNER'
    assert m['authority']['direct_user_vanda_closed_loop_GO'], 'AUTHORITY'
    assert m['scope'] == 'VANDA_FROZEN_WBC_CPU_BUILD_NO_EXECUTION', 'SCOPE'
    assert time.time() < m['deadline_epoch'], 'DEADLINE'
    assert m['execution']['profile'] == 'vanda-native-build', 'PROFILE'
    assert m['limits']['ngpus'] == 0 and m['limits']['active_cpus'] == 4, 'CPU_ONLY'
    assert m['limits']['physics_steps'] == 0 and m['limits']['model_queries'] == 0, 'NO_EXECUTION'
    lease = read(m['lease'])
    assert lease['status'] == 'ACTIVE' and lease['run_uuid'] == m['run_uuid'], 'LEASE'
    assert lease['owner'] == m['owner'] and lease['deadline_epoch'] == m['deadline_epoch'], 'LEASE_BINDING'
    binding_path = root/'OUTPUT_BINDING.private.json'
    assert sha(binding_path) == m['execution']['output_binding_sha256'], 'OUTPUT_BINDING_SHA'
    binding = read(binding_path)
    assert binding['run_uuid'] == m['run_uuid'] and Path(binding['run_root']).resolve() == root, 'OUTPUT_IDENTITY'
    registry = read(binding['registry'])
    assert any(row['role'] == binding['role'] and Path(row['path']).resolve() == root
               for row in registry['entries']), 'REGISTERED_OUTPUT_REQUIRED'
    for filename, digest in m['sealed_files'].items():
        assert sha(filename) == digest, 'SEALED_SOURCE_CHANGED'
    if execute:
        host = socket.gethostname().split('.')[0].lower()
        assert not host.startswith(('stdct-login', 'vanda')), 'LOGIN_NODE_REFUSED'
        jobid = os.environ.get('PBS_JOBID')
        assert jobid and read(root/'SUBMISSION.private.json')['job_id'] == jobid, 'OWN_PBS'
        result = subprocess.run(['/opt/pbs/bin/qstat', '-f', '-F', 'json', jobid],
                                capture_output=True, text=True, timeout=25, check=True)
        job = json.loads(result.stdout)['Jobs'][jobid]
        assert job['job_state'] == 'R' and job['project'] == m['project'], 'RUNNING_PROJECT'
        assert job['Job_Owner'].split('@')[0] == m['owner_user'], 'PBS_OWNER'
        allocated_hosts = {chunk.split('/')[0].split('.')[0].lower()
                           for chunk in job['exec_host'].split('+')}
        assert host in allocated_hosts, 'EXACT_ALLOCATED_HOST_REQUIRED'
        assert int(job['Resource_List'].get('ngpus', 0)) == 0, 'NO_GPU_ALLOCATION'
        h, minute, sec = map(int, job['Resource_List']['walltime'].split(':'))
        assert h*3600 + minute*60 + sec <= 10800, 'PBS_WALLTIME'
    return m


def unpack(archive, destination):
    assert not destination.exists(), 'PRESERVE_EXISTING_EXTRACTED_SOURCE'
    destination.mkdir(mode=0o700)
    with tarfile.open(archive, 'r:gz') as source:
        members = source.getmembers()
        assert len(members) < 15000, 'ARCHIVE_MEMBER_LIMIT'
        assert sum(m.size for m in members if m.isfile()) < 2*1024**3, 'ARCHIVE_SIZE_LIMIT'
        for member in members:
            assert member.name.split('/')[0] == 'wbc_mujoco_minimal', 'ARCHIVE_ROOT'
            assert not (member.issym() or member.islnk() or member.isdev() or member.isfifo()), 'ARCHIVE_LINK_OR_DEVICE'
            assert (destination/member.name).resolve().is_relative_to(destination.resolve()), 'ARCHIVE_ESCAPE'
        source.extractall(destination)
    return destination/'wbc_mujoco_minimal'


def build(path):
    m = validate(path, execute=True)
    root = Path(m['output_root'])
    with (root/'BUILD_OWNER.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (root/'RESULT.private.json').exists(), 'NO_REPEATED_BUILD_RUN'
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:4]))
        os.umask(0o077)
        resource.setrlimit(resource.RLIMIT_CPU, (43200, 43200))
        started = time.time()
        result = dict(run_uuid=m['run_uuid'], job_id=os.environ['PBS_JOBID'],
                      hostname=socket.gethostname(), status='RUNNING', commands=[],
                      model_queries=0, physics_steps=0, WBC_launches=0, optimizer_steps=0)
        def stop(signum, frame):
            raise TimeoutError('BUILD_ALLOCATION_END')
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGALRM, stop)
        signal.alarm(10400)
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', WBC_BUILD_JOBS='2',
                   APPTAINER_CACHEDIR=str(root/'container-cache'))
        def command(label, argv, timeout, cwd=None):
            row = dict(label=label, argv=argv, started_epoch=time.time())
            result['commands'].append(row)
            write(root/'STATE.private.json', result)
            with (root/(label+'.log')).open('x') as log:
                process = subprocess.Popen(argv, cwd=cwd, env=env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    rc = process.wait(timeout=timeout)
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
            row.update(exit_code=rc, completed_epoch=time.time())
            write(root/'STATE.private.json', result)
            if rc:
                raise RuntimeError('COMMAND_FAILED:'+label)
        try:
            assert sha(m['archive']) == m['archive_sha256'], 'USER_DELIVERY_SHA'
            native = unpack(m['archive'], root/'source')
            result['native_source_root'] = str(native)
            command('source-sha-before', ['sha256sum', '--quiet', '-c', 'SHA256SUMS'], 600, native)
            temporary = Path(os.environ.get('PBS_TMPDIR', '/tmp'))/('gr00t-wbc-'+str(os.getuid())+'-'+os.environ['PBS_JOBID'])
            assert shutil.disk_usage(temporary.parent).free > 24*1024**3, 'NODE_LOCAL_TMP_CAPACITY'
            temporary.mkdir(mode=0o700)
            env['APPTAINER_TMPDIR'] = str(temporary)
            env['TMPDIR'] = str(temporary)
            image = root/'native-wbc-jazzy.sif'
            command('container-build', [m['apptainer'], 'build', '--fakeroot',
                '--build-arg', 'ros_image='+m['ros_image'],
                '--mksquashfs-args', '-processors 2 -comp gzip -Xcompression-level 1',
                str(image), m['definition']], 4500)
            result['image_sha256'] = sha(image)
            result['image_bytes'] = image.stat().st_size
            command('native-compile', [m['apptainer'], 'exec', '--cleanenv',
                '--bind', str(root)+':'+str(root), str(image),
                '/bin/bash', '-c', 'export WBC_BUILD_JOBS=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; exec ./build.sh'],
                4800, native)
            command('source-sha-after', ['sha256sum', '--quiet', '-c', 'SHA256SUMS'], 600, native)
            simulator = native/'build/mujoco/bin/rl_sim_mujoco'
            assert simulator.is_file(), 'SIMULATOR_BINARY_ABSENT'
            command('linked-libraries', [m['apptainer'], 'exec', '--cleanenv', str(image),
                '/bin/bash', '-c', 'source ./env.sh; ldd build/mujoco/bin/rl_sim_mujoco; '
                'find install/lib -type f -executable -name "*mpc*" -o -name "*ocs2*node"'], 60, native)
            assert 'not found' not in (root/'linked-libraries.log').read_text(), 'MISSING_SHARED_LIBRARY'
            nodes = [native/'install/lib/go2_x5_ocs2/go2_x5_ocs2_node',
                     native/'install/lib/ocs2_mobile_manipulator_ros/mobile_manipulator_mpc_node']
            assert all(node.is_file() for node in nodes), 'CONTROLLER_BINARY_ABSENT'
            result.update(status='PASS_NATIVE_CONTAINER_AND_BUILD_NO_PHYSICS', image=str(image),
                          simulator_sha256=sha(simulator),
                          controller_sha256={str(node.relative_to(native)):sha(node) for node in nodes},
                          native_controller_sources_unchanged=True)
        except BaseException as error:
            result.update(status='ENGINEERING_INCOMPLETE_PRESERVED',
                          error_type=type(error).__name__, error=str(error)[:1200])
            raise
        finally:
            signal.alarm(0)
            usage = resource.getrusage(resource.RUSAGE_SELF)
            children = resource.getrusage(resource.RUSAGE_CHILDREN)
            result.update(completed_epoch=time.time(), elapsed_s=time.time()-started,
                CPU_s=usage.ru_utime+usage.ru_stime+children.ru_utime+children.ru_stime)
            write(root/'RESULT.private.json', result)
            lease = read(m['lease'])
            lease.update(status='CLOSED', result_status=result['status'])
            write(m['lease'], lease)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    value = build(args.manifest) if args.execute else dict(status='PREPARED_NO_EXECUTION', run_uuid=validate(args.manifest)['run_uuid'])
    print(json.dumps(value, indent=2))
