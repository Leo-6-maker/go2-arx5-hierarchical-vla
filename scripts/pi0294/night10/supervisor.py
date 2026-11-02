"""Finite CoreLab overnight DAG, not a replacement for native leases/owners.

Only launches explicitly pinned existing native commands. No scientific model,
physics, checkpoint, scoring or data exporter is reimplemented here.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

SCHEMA = 'CORE_V7_NIGHT10_DAG_V1'
TERMINAL = frozenset({'COMPLETE', 'FAILED', 'SKIPPED', 'UNRUN', 'UNKNOWN'})
LANES = frozenset({'gate', 'cpu', 'gpu_train', 'gpu_inference', 'physics'})
ID = re.compile(r'^[A-Za-z][A-Za-z0-9_-]{1,63}$')


def atomic(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    temp = path.with_name(path.name + '.new-' + str(os.getpid()))
    with temp.open('xb') as f:
        f.write(encoded)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def read_json(path):
    with Path(path).open('rb') as f:
        return json.load(f)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def field(obj, name):
    value = obj
    for key in name.split('.'):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(name)
        value = value[key]
    return value


def verify_file(rule):
    path = Path(rule['path'])
    if not path.is_file() or path.is_symlink():
        return False
    if 'sha256' in rule and sha256(path) != rule['sha256']:
        return False
    try:
        obj = read_json(path)
        for key, wanted in rule.get('equals', {}).items():
            if field(obj, key) != wanted:
                return False
        for key, minimum in rule.get('minimum', {}).items():
            actual = field(obj, key)
            if type(actual) not in (int, float) or actual < minimum:
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def check_plan(plan):
    if '__BIND__' in json.dumps(plan):
        raise ValueError('TEMPLATE_NOT_A_NATIVE_EXECUTION_PLAN_BIND_REAL_CORE_COMMANDS')
    if plan.get('schema') != SCHEMA or plan.get('duration_s') != 36000:
        raise ValueError('EXACT_FINITE_TEN_HOUR_SCIENTIFIC_WINDOW')
    if not (1 <= plan.get('max_cpu_parallel', 0) <= 8):
        raise ValueError('BOUNDED_CPU_PARALLELISM_1_TO_8')
    if not (30 <= plan.get('disk_margin_gib', 0) <= 1024):
        raise ValueError('REAL_SSD_RESERVE_AT_LEAST_30GIB')
    if not isinstance(plan.get('tasks'), list) or not 1 <= len(plan['tasks']) <= 40:
        raise ValueError('FINITE_ONE_TO_FORTY_TASKS')
    ids = [t.get('id') for t in plan['tasks']]
    if any(type(x) is not str or not ID.fullmatch(x) for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('UNIQUE_SAFE_TASK_IDS')
    tasks = {t['id']: t for t in plan['tasks']}
    for t in tasks.values():
        if t.get('lane') not in LANES:
            raise ValueError('EXPLICIT_APPROVED_LANE:' + t['id'])
        after = t.get('after', [])
        needs = t.get('needs', [])
        if (not isinstance(after, list) or not isinstance(needs, list)
                or any(x not in tasks or x == t['id'] for x in after + needs)
                or len(set(after + needs)) != len(after + needs)):
            raise ValueError('EXACT_EXISTING_DAG_DEPENDENCIES:' + t['id'])
        for rule in t.get('readiness', []) + t.get('admission', []):
            if not isinstance(rule, dict) or not Path(rule.get('path', '')).is_absolute():
                raise ValueError('ABSOLUTE_EXPLICIT_NATIVE_RECEIPT:' + t['id'])
            if not rule.get('equals') and not rule.get('minimum') and not rule.get('sha256'):
                raise ValueError('RECEIPT_MUST_HAVE_TYPED_CONDITION:' + t['id'])
        if t.get('closure_receipt'):
            rule=t['closure_receipt']
            if (not isinstance(rule,dict)or not Path(rule.get('path','')).is_absolute()
                    or not rule.get('equals')):
                raise ValueError('EXACT_NATIVE_CLOSURE_RECEIPT:' + t['id'])
        if t['lane'] == 'gate':
            if t.get('command') or not t.get('readiness') or t.get('receipt'):
                raise ValueError('GATE_IS_READ_ONLY_NATIVE_CLOSURE:' + t['id'])
            continue
        if (not isinstance(t.get('command'), list) or not t['command']
                or not all(type(x) is str and x for x in t['command'])
                or not Path(t['command'][0]).is_absolute()
                or not Path(t.get('cwd', '')).is_absolute()):
            raise ValueError('ABSOLUTE_NATIVE_EXECUTABLE_AND_WORKDIR:' + t['id'])
        # A readonly CPU monitor may cover the dispatch window through its
        # final summary reserve. GPU/native task bounds remain unchanged.
        maximum_runtime = 35900 if t['lane'] == 'cpu' else 35000
        if not (10 <= t.get('max_seconds', 0) <= maximum_runtime):
            raise ValueError('FINITE_TASK_RUNTIME:' + t['id'])
        if (not isinstance(t.get('receipt'), dict)
                or not Path(t['receipt'].get('path', '')).is_absolute()
                or not t['receipt'].get('equals')):
            raise ValueError('EXACT_NATIVE_TERMINAL_RECEIPT_REQUIRED:' + t['id'])
        if (type(t.get('write_gib')) not in (int, float)
                or not 0 <= t['write_gib'] <= 200):
            raise ValueError('FINITE_FORECAST_NEW_WRITES:' + t['id'])
        cpus = t.get('cpus', [])
        if not isinstance(cpus, list) or not cpus or any(type(c) is not int or c < 0 for c in cpus) or len(set(cpus)) != len(cpus):
            raise ValueError('DECLARED_DISJOINT_CPU_AFFINITY:' + t['id'])
        if t['lane'] in ('gpu_train', 'gpu_inference', 'physics'):
            if t.get('gpu') not in (0, 1) or type(t.get('gpu')) is not int:
                raise ValueError('EXPLICIT_PHYSICAL_GPU_INDEX:' + t['id'])
            if not isinstance(t.get('gpu_uuid'), str) or not t['gpu_uuid'].startswith('GPU-'):
                raise ValueError('ACTUAL_GPU_UUID_REQUIRED:' + t['id'])
            if not isinstance(t.get('min_free_mib'), int) or t['min_free_mib'] < 1024:
                raise ValueError('NATIVE_FREE_VRAM_REQUIREMENT:' + t['id'])
            if not t.get('readiness'):
                raise ValueError('GPU_TASK_REQUIRES_COLD_OR_OWNERSHIP_GATE:' + t['id'])
            if t['lane'] == 'gpu_train' and t['gpu'] == 1:
                if plan.get('gpu1_training_admitted') is not True or not t.get('admission'):
                    raise PermissionError('GPU1_NATIVE_TEN_UPDATE_CAPACITY_NOT_ADMITTED')
            if type(t.get('gpu_stable_seconds',0)) not in (int,float) or not 0<=t.get('gpu_stable_seconds',0)<=300:
                raise ValueError('FINITE_GPU_STABILITY_OBSERVATION:' + t['id'])
            if t['lane'] == 'physics' and (not t.get('readiness') or not t.get('preflight')
                    or (t['gpu']==1 and not t.get('admission'))):
                raise ValueError('SOURCE_PHYSICS_REQUIRES_ORIGINAL_COLD_GATE_AND_OWNER:' + t['id'])
        elif t.get('gpu') is not None or t.get('gpu_uuid') is not None:
            raise ValueError('CPU_CANNOT_CLAIM_GPU:' + t['id'])
        for name, val in t.get('env', {}).items():
            if (not re.fullmatch(r'[A-Z][A-Z0-9_]*', name)
                    or not isinstance(val, str) or name in ('CUDA_VISIBLE_DEVICES',)):
                raise ValueError('INVALID_TASK_ENVIRONMENT_OVERRIDE:' + t['id'])
        if t.get('preflight'):
            c = t['preflight']
            if not isinstance(c, list) or not c or not Path(c[0]).is_absolute() or not all(isinstance(x, str) for x in c):
                raise ValueError('EXPLICIT_EXISTING_NATIVE_PREFLIGHT:' + t['id'])
    visiting, visited = set(), set()
    def visit(key):
        if key in visiting:
            raise ValueError('ACYCLIC_DAG_ONLY')
        if key in visited:
            return
        visiting.add(key)
        for x in tasks[key].get('needs', []) + tasks[key].get('after', []):
            visit(x)
        visiting.remove(key)
        visited.add(key)
    for key in tasks:
        visit(key)
    return tasks


def gpu_snapshot(index):
    proc = subprocess.run(['nvidia-smi', '-i', str(index), '--query-gpu=uuid,memory.free',
                           '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=8)
    if proc.returncode != 0 or len(proc.stdout.strip().splitlines()) != 1:
        raise RuntimeError('NVIDIA_SMI_NATIVE_UNAVAILABLE')
    uuid, free = [s.strip() for s in proc.stdout.strip().split(',')]
    tasks = subprocess.run(['nvidia-smi', '-i', str(index), '--query-compute-apps=pid',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=8)
    if tasks.returncode != 0:
        raise RuntimeError('GPU_COMPUTE_OWNERS_NOT_VERIFIED')
    pids = [int(x.strip()) for x in tasks.stdout.splitlines() if x.strip().isdigit()]
    return uuid, int(free), pids


def available_memory_gib():
    with Path('/proc/meminfo').open() as f:
        line = next((s for s in f if s.startswith('MemAvailable:')), None)
    if line is None:
        raise RuntimeError('MEMAVAILABLE_UNKNOWN')
    return int(line.split()[1]) / 1024**2


def start_ticks(pid):
    try:
        # after the last ')', field 22 = index 19 of remaining tokens
        return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


class Supervisor:
    def __init__(self, plan, output, *, resume=False, gpu_probe=gpu_snapshot, time_source=time.time):
        self.plan, self.tasks = plan, check_plan(plan)
        self.output = Path(output)
        self.gpu_probe, self.now = gpu_probe, time_source
        self.processes = {}
        self.gpu_stable_since = {}
        if not resume:
            self.output.mkdir(mode=0o700, parents=False, exist_ok=False)
            self.started = self.now()
            self.states = {key: dict(state='PENDING') for key in self.tasks}
            atomic(self.output/'PLAN.json', plan)
            self.save()
        else:
            stored = read_json(self.output/'PLAN.json')
            if stored != plan:
                raise PermissionError('DO_NOT_CHANGE_ACTIVE_OR_RESUMED_PLAN')
            state = read_json(self.output/'STATE.json')
            self.started, self.states = state['started'], state['tasks']
            if any(row['state'] in ('LAUNCHING', 'RUNNING', 'STOPPING') for row in self.states.values()):
                raise PermissionError('EXISTING_CHILD_EFFECT_UNKNOWN_REQUIRES_NATIVE_RECONCILIATION')
        self.deadline = self.started + plan['duration_s']
        self.last_heartbeat = self.now()
        self.owner_fd = (self.output/'OWNER.lock').open('a+')
        fcntl.flock(self.owner_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def close(self):
        fd = getattr(self, 'owner_fd', None)
        if fd is not None and not fd.closed:
            fd.close()

    def __del__(self):
        self.close()

    def save(self):
        atomic(self.output/'STATE.json', dict(schema=SCHEMA, started=self.started,
            deadline=self.started+self.plan['duration_s'], observed=self.now(), tasks=self.states))

    def change(self, key, state, **kwargs):
        self.states[key] = dict(self.states[key], state=state, observed=self.now(), **kwargs)
        self.save()

    def prerequisites(self, key):
        task = self.tasks[key]
        for other in task.get('after', []) + task.get('needs', []):
            if self.states[other]['state'] not in TERMINAL:
                return 'WAIT'
        for other in task.get('needs', []):
            if self.states[other]['state'] != 'COMPLETE':
                return 'SKIP'
        return 'READY'

    def capacity(self, key):
        task = self.tasks[key]
        if task['lane'] == 'gate':
            return True
        used_cpu = {c for key in self.processes for c in self.tasks[key].get('cpus', [])}
        cpus = set(task['cpus'])
        if used_cpu & cpus or not cpus <= os.sched_getaffinity(0):
            return False
        cpu_running = sum(1 for k in self.processes if self.tasks[k]['lane'] == 'cpu')
        if task['lane'] == 'cpu' and cpu_running >= self.plan['max_cpu_parallel']:
            return False
        gpu = task.get('gpu')
        if gpu is not None:
            if any(self.tasks[k].get('gpu') == gpu for k in self.processes):
                return False
            if task['lane'] == 'physics' and any(self.tasks[k]['lane'] == 'physics' for k in self.processes):
                return False
            try:
                uuid, free, compute_pids = self.gpu_probe(gpu)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
                self.gpu_stable_since.pop(key,None)
                return False
            if (uuid != task['gpu_uuid'] or free < task['min_free_mib']
                    or (compute_pids and task.get('allow_shared_gpu') is not True)):
                self.gpu_stable_since.pop(key,None)
                return False
            since=self.gpu_stable_since.setdefault(key,self.now())
            if self.now()-since<task.get('gpu_stable_seconds',0):
                return False
        reserved_gib = sum(self.tasks[k]['write_gib'] for k in self.processes)
        disk_gib = shutil.disk_usage(self.output).free / 1024**3
        if disk_gib < self.plan['disk_margin_gib'] + reserved_gib + task['write_gib']:
            return False
        if available_memory_gib() < max(2, len(cpus) * .5,task.get('min_available_memory_gib',0)):
            return False
        return True

    def launch(self, key):
        task = self.tasks[key]
        if Path(task['receipt']['path']).exists():
            self.change(key, 'UNKNOWN', reason='OUTPUT_ALREADY_EXISTS_NO_IMPLICIT_ADOPTION')
            return
        if not Path(task['cwd']).is_dir():
            self.change(key, 'FAILED', reason='WORKDIR_MISSING')
            return
        if task.get('preflight'):
            try:
                pre = subprocess.run(task['preflight'], cwd=task['cwd'], capture_output=True, timeout=90)
            except (OSError, subprocess.TimeoutExpired) as error:
                self.change(key, 'SKIPPED', reason='NATIVE_COLD_PREFLIGHT_NOT_QUALIFIED',
                            detail=type(error).__name__)
                return
            if pre.returncode != 0:
                self.change(key, 'SKIPPED', reason='NATIVE_OWNER_OR_COLD_SAFETY_PREFLIGHT_FAILED',
                            preflight_returncode=pre.returncode)
                return
        # Immutable intent BEFORE starting any potentially mutating command.
        self.change(key, 'LAUNCHING', command_sha256=hashlib.sha256(
            json.dumps(task['command'], separators=(',', ':')).encode()).hexdigest())
        log = (self.output / (key + '.log')).open('xb')
        env = dict(os.environ)
        for name in list(env):
            if name.startswith('SSH_') or name.startswith('KRB5'):
                env.pop(name, None)
        env['CUDA_VISIBLE_DEVICES'] = '' if task.get('gpu') is None else str(task['gpu'])
        env['OMP_NUM_THREADS'] = str(len(task['cpus']))
        env['MKL_NUM_THREADS'] = str(len(task['cpus']))
        env['OPENBLAS_NUM_THREADS'] = str(len(task['cpus']))
        env['PYTHONNOUSERSITE'] = '1'
        for name, val in task.get('env', {}).items():
            env[name] = val
        def affinity():
            os.sched_setaffinity(0, set(task['cpus']))
        try:
            process = subprocess.Popen(task['command'], cwd=task['cwd'], env=env,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, preexec_fn=affinity)
        except OSError as e:
            log.close()
            self.change(key, 'FAILED', reason='NATIVE_EXEC_START_FAILED', detail=str(e))
            return
        log.close()
        self.processes[key] = process
        self.change(key, 'RUNNING', pid=process.pid, proc_start=start_ticks(process.pid),
                    start_time=self.now(), gpu=task.get('gpu'), lane=task['lane'])

    def finish(self, key, process, *, timeout=False):
        code = process.poll()
        if code is None:
            return False
        self.processes.pop(key)
        t = self.tasks[key]
        if t.get('closure_receipt')and not verify_file(t['closure_receipt']):
            self.change(key,'UNKNOWN',exit_code=code,reason='NATIVE_EFFECT_NOT_INDEPENDENTLY_CLOSED')
            return True
        receipt_ok = verify_file(t['receipt'])
        timeout=timeout or self.states[key]['state']=='STOPPING'
        self.change(key, 'COMPLETE' if code == 0 and receipt_ok and not timeout else 'FAILED',
                    exit_code=code, receipt_verified=receipt_ok, timeout=timeout,
                    reason=('NATIVE_COMPLETED_RECEIPT' if code == 0 and receipt_ok and not timeout
                            else 'NATIVE_ERROR_OR_INVALID_RECEIPT'))
        return True

    def poll(self):
        for key, proc in list(self.processes.items()):
            if self.finish(key, proc):
                continue
            task = self.tasks[key]
            if self.now() - self.states[key]['start_time'] > task['max_seconds']:
                if self.states[key]['state'] != 'STOPPING':
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    self.change(key, 'STOPPING', reason='FINITE_NATIVE_RUNTIME_EXCEEDED',
                                sigterm_at=self.now())
                elif self.now()-self.states[key]['sigterm_at'] > 60:
                    # Do not kill, replay or release a native effect of unknown state.
                    self.change(key, 'UNKNOWN', reason='TERMINATION_NOT_CONFIRMED_NATIVE_RECONCILIATION_REQUIRED')
                    self.processes.pop(key)
                    return 'UNKNOWN'
        return None

    def tick(self):
        if self.poll() == 'UNKNOWN':
            return False
        # Unknown native effects prohibit further launches, even on an idle GPU.
        if any(x['state'] == 'UNKNOWN' for x in self.states.values()):
            return False
        for key, task in self.tasks.items():
            if self.states[key]['state'] != 'PENDING':
                continue
            p = self.prerequisites(key)
            if p == 'WAIT':
                continue
            if p == 'SKIP':
                self.change(key, 'SKIPPED', reason='REQUIRED_PREDECESSOR_DID_NOT_COMPLETE')
                continue
            if not all(verify_file(rule) for rule in task.get('readiness', [])):
                continue
            if task['lane'] == 'gate':
                self.change(key, 'COMPLETE', reason='READ_ONLY_ORIGINAL_NATIVE_GATE_VERIFIED')
                continue
            if not all(verify_file(rule) for rule in task.get('admission', [])):
                self.change(key, 'SKIPPED', reason='MISSING_EXACT_DATA_OR_TRAINING_ADMISSION')
                continue
            if self.now() + task['max_seconds'] + 60 > self.deadline:
                continue
            if self.capacity(key):
                self.launch(key)
        return True

    def run(self, poll_s=3):
        if not 1 <= poll_s <= 30:
            raise ValueError('LIMITED_HEARTBEAT')
        try:
            while self.now() < self.deadline:
                if not self.tick():
                    break
                if self.now() - self.last_heartbeat >= 30:
                    self.save()
                    self.last_heartbeat = self.now()
                if all(x['state'] in TERMINAL for x in self.states.values()):
                    break
                time.sleep(poll_s)
            if self.now() >= self.deadline:
                for key in self.tasks:
                    if self.states[key]['state'] == 'PENDING':
                        self.change(key, 'UNRUN', reason='TEN_HOUR_WINDOW_ENDED')
            # New work is always stopped at the ten-hour boundary. Children
            # already launched may use their predeclared finite runtime/drain.
            while self.processes:
                if self.poll() == 'UNKNOWN':
                    break
                if self.processes:
                    time.sleep(poll_s)
            # Tasks whose prerequisites were not satisfied remain explicitly unrun.
            for key in self.tasks:
                if self.states[key]['state'] == 'PENDING':
                    self.change(key, 'UNRUN', reason='DEPENDENCY_OR_RESOURCE_NOT_READY')
        finally:
            atomic(self.output/'SUMMARY.json', dict(schema=SCHEMA,
                plan_sha256=sha256(self.output/'PLAN.json'), started=self.started,
                finished=self.now(), deadline=self.deadline, tasks=self.states,
                scientific_success_not_inferred_from_process_exit=True))
            self.close()
        return 0 if not any(v['state'] == 'UNKNOWN' for v in self.states.values()) else 3


def cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='operation', required=True)
    check = sub.add_parser('check');check.add_argument('--plan', required=True)
    run = sub.add_parser('run');run.add_argument('--plan', required=True)
    run.add_argument('--output', required=True);run.add_argument('--arm', action='store_true')
    run.add_argument('--resume', action='store_true');run.add_argument('--poll-seconds', type=int, default=3)
    status = sub.add_parser('status');status.add_argument('--output', required=True)
    a = p.parse_args(argv)
    if a.operation == 'status':
        print(json.dumps(read_json(Path(a.output)/'STATE.json'), indent=2, sort_keys=True))
        return 0
    plan = read_json(a.plan)
    tasks = check_plan(plan)
    if a.operation == 'check':
        print(json.dumps(dict(status='VALID_CONFIGURATION_NOT_NATIVE_ADMISSION',
                              tasks=list(tasks),duration_s=plan['duration_s'],
                              gpu1_training_admitted=plan.get('gpu1_training_admitted') is True), indent=2))
        return 0
    if not a.arm:
        raise PermissionError('NO_MUTATING_CORE_TASK_WITHOUT_EXPLICIT_ARM')
    runner = Supervisor(plan, a.output, resume=a.resume)
    return runner.run(poll_s=a.poll_seconds)


if __name__ == '__main__':
    sys.exit(cli())
