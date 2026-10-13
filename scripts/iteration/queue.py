"""One coordinator owns reservation state. Workers cannot mutate this ledger.

This is a tested state machine, not a replacement for PI0292's serial physics
guard. Unknown or crashed attempts keep their reservations until reconciliation.
"""
from __future__ import annotations

import copy
import math
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import time

from scripts.pi0292.durable import OwnerLock, read, write

KEYS=('physics_steps','rgb_pairs','cpu_s','output_bytes')


class Coordinator:
    def __init__(self, root, plan, *, fixture=False):
        if not fixture:raise ValueError('LIVE_ADAPTER_NOT_ADMITTED_PREPARATION_ONLY')
        if plan['authorizes_execution'] or plan['status']!='PREPARED_NOT_AUTHORIZED':
            raise ValueError('FIXTURE_PLAN_REQUIRED')
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.root=self.root.resolve(strict=True)
        self.lock=OwnerLock(self.root/'COORDINATOR.lock')
        self.path=self.root/'QUEUE.json'
        if self.path.exists():
            self.lock.close();raise FileExistsError('INSPECT_OR_RECONCILE_NEVER_RESTART')
        self.plan=copy.deepcopy(plan)
        self.value=dict(status='FIXTURE_RUNNING_ZERO_PHYSICS', plan=self.plan,
            reservations={},records=[],actual={key:0 for key in KEYS},
            fixture=True,physics_executed=0,model_calls=0)
        write(self.path,self.value)

    def reserve(self, identifier):
        v=copy.deepcopy(self.value)
        if v['status']!='FIXTURE_RUNNING_ZERO_PHYSICS':raise ValueError('STOPPED_OWNER')
        if identifier not in [r['id'] for r in self.plan['episodes']]:raise ValueError('UNREGISTERED_ID')
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',identifier):raise ValueError('UNSAFE_ID')
        if identifier in v['reservations'] or any(r['id']==identifier for r in v['records']):
            raise ValueError('FIRST_ATTEMPT_ALREADY_RESERVED_OR_FINISHED')
        if len(v['reservations'])>=self.plan['workers']:raise ValueError('ALL_WORKER_SLOTS_RESERVED')
        cost=self.plan['cost']
        if any(v['actual'][k]+sum(r[k] for r in v['reservations'].values())+cost[k]>self.plan['limits'][k] for k in KEYS):
            raise ValueError('GLOBAL_BUDGET_RESERVATION_EXCEEDED')
        output=self.root/'episodes'/identifier
        if not output.resolve().is_relative_to(self.root):raise ValueError('OUTPUT_PATH_ESCAPE')
        output.mkdir(parents=True,exist_ok=False)
        v['reservations'][identifier]=copy.deepcopy(cost)
        self.value=v;write(self.path,v)
        row=next(r for r in self.plan['episodes'] if r['id']==identifier)
        return dict(**row,output=str(output),reservation=copy.deepcopy(cost),fixture=True)

    def complete(self, receipt):
        identifier=receipt['id'];v=copy.deepcopy(self.value)
        if identifier not in v['reservations']:raise ValueError('UNKNOWN_OR_DUPLICATE_RECEIPT')
        if (receipt['fixture'] is not True or receipt['physics_steps']!=0 or receipt['rgb_pairs']!=0
                or receipt['model_calls']!=0):
            raise ValueError('FIXTURE_MUST_NOT_EXECUTE_PHYSICS')
        expected=self.root/'episodes'/identifier/'WORKER_RECEIPT.json'
        if read(expected)!=receipt:raise ValueError('DURABLE_WORKER_RECEIPT_REQUIRED')
        for key in KEYS:
            x=receipt[key]
            if type(x) not in (int,float) or not math.isfinite(x) or x<0:raise ValueError('INVALID_ACTUAL_COST')
            if key!='cpu_s' and type(x) is not int:raise ValueError('INTEGER_ACTUAL_COUNT_REQUIRED')
        breach=any(receipt[k]>v['reservations'][identifier][k] for k in KEYS)
        for key in KEYS:v['actual'][key]+=receipt[key]
        del v['reservations'][identifier]
        v['records'].append(copy.deepcopy(receipt))
        if breach or receipt['status']!='FIXTURE_COMPLETE':
            v['status']='FIXTURE_STOP_NO_AUTOMATIC_RETRY'
        self.value=v;write(self.path,v)

    def unknown(self, identifier):
        if identifier not in self.value['reservations']:raise ValueError('UNKNOWN_ID')
        self.value['status']='FIXTURE_STOP_UNRECONCILED_RESERVATION_RETAINED'
        write(self.path,self.value)

    def finish(self):
        if self.value['reservations']:raise ValueError('UNRECONCILED_WORKERS')
        if self.value['status']=='FIXTURE_RUNNING_ZERO_PHYSICS':
            if len(self.value['records'])!=len(self.plan['episodes']):raise ValueError('INCOMPLETE_PLAN')
            self.value['status']='FIXTURE_TERMINAL_ZERO_PHYSICS'
            write(self.path,self.value)
        return copy.deepcopy(self.value)

    def close(self):self.lock.close()


LIVE_KEYS = ('physics_starts', 'physics_steps', 'rgb_pairs', 'cpu_s', 'output_bytes')
LIVE_LIMITS = dict(physics_starts=400, physics_steps=6400000, rgb_pairs=256400,
                   cpu_s=160 * 3600, output_bytes=192 * 1024**3)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def value_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


class LiveCoordinator:
    """A single admitted owner reserves before spawn, and charges durable receipts.

    A missing or malformed receipt retains the full upper bound. No restart or
    implicit repair exists; failed first attempts keep their logical identity.
    """
    def __init__(self, root, plan_path, admission_path, *, workers=2, now=None,
                 stage_boundary=False):
        self.root = Path(root).resolve(strict=True)
        self.plan_path, self.admission_path = Path(plan_path), Path(admission_path)
        self.plan, self.admission = read(plan_path), read(admission_path)
        self.now = now or time.time
        self.lock = OwnerLock(self.root / 'COORDINATOR.lock')
        try:
            self._admit(workers)
            queue_name = self.admission.get('queue_name', 'live')
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', queue_name):
                raise ValueError('SAFE_QUEUE_NAME_REQUIRED')
            self.path = self.root / ('LIVE_QUEUE.json' if queue_name == 'live' else 'LIVE_QUEUE_' + queue_name + '.json')
            self.workers = workers
            self.base_records = 0
            if self.path.exists():
                if not stage_boundary:
                    raise FileExistsError('INSPECT_OR_RECONCILE_NEVER_RESTART')
                prior = read(self.path)
                if (prior['status'] != 'LIVE_TERMINAL_ALL_FIRST_ATTEMPTS'
                        or prior['reservations'] or prior['auxiliary']):
                    raise ValueError('ONLY_RECONCILED_TERMINAL_STAGE_BOUNDARY')
                if prior['actual'] != self.admission['prior_actual']:
                    raise ValueError('AUTHORITATIVE_CUMULATIVE_COST_CHANGED')
                if prior['run_uuid'] != self.admission['run_uuid'] or prior['source_sha'] != self.admission['source_sha']:
                    raise ValueError('ACTIVE_RUN_SOURCE_CANNOT_CHANGE')
                if {r['id'] for r in prior['records']} & {r['id'] for r in self.plan['episodes']}:
                    raise ValueError('FIRST_ATTEMPT_REUSE_FORBIDDEN')
                self.base_records = len(prior['records'])
                prior['status'] = 'LIVE_RUNNING'
                prior['stages'].append(dict(plan_sha256=digest(plan_path),
                    admission_sha256=digest(admission_path), started_epoch=self.now(),
                    expected_ids=[r['id'] for r in self.plan['episodes']]))
                prior['plan_sha256'] = digest(plan_path)
                prior['admission_sha256'] = digest(admission_path)
                prior['owner_pid'] = os.getpid()
                prior['owner_proc_start'] = Path('/proc/self/stat').read_text().split()[21]
                self.value = prior
                write(self.path, prior)
                return
            self.value = dict(schema='PI0294_LIVE_QUEUE_V1', status='LIVE_RUNNING',
                fixture=False, model_calls=0, owner_pid=os.getpid(),
                run_uuid=self.admission['run_uuid'], source_sha=self.admission['source_sha'],
                owner_proc_start=Path('/proc/self/stat').read_text().split()[21],
                plan_sha256=digest(plan_path), admission_sha256=digest(admission_path),
                started_epoch=self.now(), actual=copy.deepcopy(self.admission['prior_actual']),
                reservations={}, records=[], auxiliary={}, stages=[dict(
                    plan_sha256=digest(plan_path), admission_sha256=digest(admission_path),
                    started_epoch=self.now(), expected_ids=[r['id'] for r in self.plan['episodes']])])
            (self.root / 'reservations').mkdir(exist_ok=True)
            (self.root / 'episodes').mkdir(exist_ok=True)
            (self.root / 'worker_logs').mkdir(exist_ok=True)
            (self.root / 'worker_processes').mkdir(exist_ok=True)
            write(self.path, self.value)
        except BaseException:
            self.lock.close()
            raise

    def _admit(self, workers):
        a, p = self.admission, self.plan
        if a.get('schema') != 'PI0294_LIVE_ADMISSION_V1' or a.get('status') != 'LIVE_ADMITTED_ACTUAL_RUNTIME':
            raise ValueError('ACTUAL_RUNTIME_ADMISSION_REQUIRED')
        if a.get('fixture') is not False or p.get('fixture') is not False or not p.get('authorizes_execution'):
            raise ValueError('FIXTURE_NEVER_BECOMES_LIVE')
        if a.get('hostname') != socket.gethostname() or a.get('profile') != '4080-simulation':
            raise ValueError('HOST_PROFILE_BINDING_REQUIRED')
        if Path(a['root']).resolve() != self.root or a['plan_sha256'] != digest(self.plan_path):
            raise ValueError('ROOT_OR_PLAN_BINDING_CHANGED')
        manifest = read(self.root / 'RUN_MANIFEST.json')
        if digest(self.root / 'RUN_MANIFEST.json') != a['run_manifest_sha256']:
            raise ValueError('RUN_MANIFEST_CHANGED')
        if manifest['pi'] != 'PI0294' or manifest['run_uuid'] != a['run_uuid'] or not manifest.get('authority'):
            raise ValueError('NEW_PI0294_AUTHORITY_REQUIRED')
        seal = read(self.root / 'SOURCE_SEAL.json')
        if digest(self.root / 'SOURCE_SEAL.json') != a['source_seal_sha256'] or seal['source_sha'] != a['source_sha']:
            raise ValueError('SOURCE_SEAL_CHANGED')
        source = Path(a['source_root']).resolve(strict=True)
        for name, expected in seal['files'].items():
            path = (source / name).resolve(strict=True)
            if not path.is_relative_to(source) or digest(path) != expected:
                raise ValueError('SEALED_SOURCE_BYTES_CHANGED')
        if a.get('T0_epoch') != 1791901497 or a.get('cutoff_epoch') != 1791939600 or a.get('deadline_epoch') != 1791943200:
            raise ValueError('CLOCK_RESET_OR_DEADLINE_CHANGED')
        if self.now() >= a['cutoff_epoch'] or self.now() >= a['lease_expires_epoch']:
            raise TimeoutError('LIVE_LEASE_OR_NEW_WORK_CUTOFF')
        diagnostic = p.get('kind') == 'PARALLEL_QUALIFICATION'
        if diagnostic and any(row['card']['split'] in ('TRAIN', 'DEV') for row in p['episodes']):
            raise ValueError('PARALLEL_DIAGNOSTICS_NEVER_FORMAL_BANK')
        ceiling = a.get('diagnostic_workers', 0) if diagnostic else a['qualified_workers']
        if workers not in (1, 2, 4) or workers > ceiling or ceiling > 4:
            raise ValueError('PARALLEL_LEVEL_NOT_QUALIFIED')
        if workers == 4 and not diagnostic and a.get('parallel4_receipt_sha256') is None:
            raise ValueError('FOUR_WORKER_ACTUAL_QUALIFICATION_REQUIRED')
        cpus = a['allowed_cpus']
        if len(cpus) != len(set(cpus)) or len(cpus) > 16 or not set(cpus).issubset(os.sched_getaffinity(0)):
            raise ValueError('ACTUAL_ALLOWED_CPU_BINDING_REQUIRED')
        if len(cpus) < 3 * workers + a.get('auxiliary_cpu_count', 1):
            raise ValueError('DISJOINT_WORKER_AND_AUDITOR_CORES_REQUIRED')
        if any(a['limits'][k] > LIVE_LIMITS[k] for k in LIVE_KEYS):
            raise ValueError('PROTOCOL_LIMIT_EXCEEDED')
        if a['memory_limit_bytes'] > 192 * 1024**3 or a['memory_limit_bytes'] <= 0:
            raise ValueError('BOUNDED_MEMORY_REQUIRED')
        for k in LIVE_KEYS:
            self._cost(k, a['prior_actual'][k])
        ids = [row['id'] for row in p['episodes']]
        if len(ids) != len(set(ids)) or not ids or any(not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', i) for i in ids):
            raise ValueError('UNIQUE_SAFE_FIRST_ATTEMPT_PLAN_REQUIRED')

    @staticmethod
    def _cost(key, value):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('INVALID_ACTUAL_COST:' + key)
        if key != 'cpu_s' and type(value) is not int:
            raise ValueError('INTEGER_ACTUAL_COUNT_REQUIRED:' + key)

    def _capacity(self, cost, memory):
        v, a = self.value, self.admission
        outstanding = list(v['reservations'].values()) + list(v['auxiliary'].values())
        if any(v['actual'][k] + sum(r['reservation'][k] for r in outstanding) + cost[k] > a['limits'][k] for k in LIVE_KEYS):
            raise ValueError('GLOBAL_BUDGET_RESERVATION_EXCEEDED')
        if sum(r['reservation']['memory_bytes'] for r in outstanding) + memory > a['memory_limit_bytes']:
            raise ValueError('PEAK_MEMORY_RESERVATION_EXCEEDED')
        if shutil.disk_usage(self.root).free < cost['output_bytes'] + a['free_space_margin_bytes']:
            raise ValueError('REAL_FREE_SPACE_MARGIN_REQUIRED')

    def reserve(self, identifier, *, timeout_s=600, output_bytes=512 * 1024**2,
                memory_bytes=4 * 1024**3):
        v = copy.deepcopy(self.value)
        if v['status'] != 'LIVE_RUNNING':
            raise ValueError('STOPPED_OWNER')
        now = self.now()
        if now >= min(self.admission['cutoff_epoch'], self.admission['lease_expires_epoch']):
            raise TimeoutError('NEW_WORK_CUTOFF')
        rows = [r for r in self.plan['episodes'] if r['id'] == identifier]
        if len(rows) != 1:
            raise ValueError('UNREGISTERED_ID')
        if identifier in v['reservations'] or any(r['id'] == identifier for r in v['records']):
            raise ValueError('FIRST_ATTEMPT_ALREADY_RESERVED_OR_FINISHED')
        used_slots = {r['slot'] for r in v['reservations'].values()}
        slots = sorted(set(range(self.workers)) - used_slots)
        if not slots:
            raise ValueError('ALL_WORKER_SLOTS_RESERVED')
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('BOUNDED_TIMEOUT_REQUIRED')
        deadline = min(now + timeout_s, self.admission['cutoff_epoch'], self.admission['lease_expires_epoch'])
        cost = dict(physics_starts=1, physics_steps=16000, rgb_pairs=641,
                    cpu_s=3 * (deadline - now), output_bytes=output_bytes,
                    memory_bytes=memory_bytes)
        for k in LIVE_KEYS:
            self._cost(k, cost[k])
        self._cost('memory_bytes', memory_bytes)
        self._capacity(cost, memory_bytes)
        row = rows[0]
        card = row['card']
        if card['id'] != identifier:
            raise ValueError('CARD_ID_CHANGED')
        slot = slots[0]
        output = self.root / 'episodes' / identifier
        if output.exists():
            raise FileExistsError('FIRST_ATTEMPT_OUTPUT_PRESERVED')
        ticket = dict(schema='PI0294_LIVE_RESERVATION_V1', id=identifier, fixture=False,
            card=copy.deepcopy(card), card_sha256=value_digest(card), slot=slot,
            root=str(self.root), output=str(output), asset_root=self.admission['asset_root'],
            source_sha=self.admission['source_sha'], source_root=self.admission['source_root'],
            cpus=self.admission['allowed_cpus'][3 * slot:3 * slot + 3], reservation=cost,
            deadline_epoch=deadline, reserved_epoch=now, run_manifest_sha256=self.admission['run_manifest_sha256'],
            plan_sha256=self.value['plan_sha256'])
        # Persist the reservation before publishing a ticket or creating output.
        v['reservations'][identifier] = ticket
        write(self.path, v)
        self.value = v
        write(self.root / 'reservations' / (identifier + '.json'), ticket)
        return copy.deepcopy(ticket)

    def complete(self, receipt):
        v = copy.deepcopy(self.value)
        identifier = receipt['id']
        if identifier not in v['reservations']:
            raise ValueError('UNKNOWN_OR_DUPLICATE_RECEIPT')
        ticket = v['reservations'][identifier]
        if read(Path(ticket['output']) / 'WORKER_RECEIPT.json') != receipt:
            raise ValueError('DURABLE_WORKER_RECEIPT_REQUIRED')
        if receipt.get('fixture') is not False or receipt.get('model_calls', 0) != 0:
            raise ValueError('ACTUAL_PHYSICS_ONLY_RECEIPT_REQUIRED')
        if receipt.get('card_sha256') != ticket['card_sha256']:
            raise ValueError('WORKER_CARD_IDENTITY_CHANGED')
        for k in LIVE_KEYS + ('memory_bytes',):
            self._cost(k, receipt[k])
        if receipt['status'] not in ('VALID_COMPLETE', 'PHYSICS_COMPLETE', 'FAILED_PRESERVED', 'SAFETY_STOP_PRESERVED'):
            raise ValueError('FIRST_ATTEMPT_TERMINAL_REQUIRED')
        breach = any(receipt[k] > ticket['reservation'][k] for k in LIVE_KEYS + ('memory_bytes',))
        for k in LIVE_KEYS:
            v['actual'][k] += receipt[k]
        del v['reservations'][identifier]
        v['records'].append(copy.deepcopy(receipt))
        if breach:
            v['status'] = 'LIVE_STOP_RESOURCE_BREACH'
        elif receipt['status'] == 'SAFETY_STOP_PRESERVED':
            v['status'] = 'LIVE_STOP_SAFETY'
        write(self.path, v)
        self.value = v

    def reserve_auxiliary(self, identifier, *, cpu_count, timeout_s, output_bytes,
                          memory_bytes=1024**3):
        if identifier in self.value['auxiliary']:
            raise ValueError('AUXILIARY_ALREADY_RESERVED')
        available = self.admission['allowed_cpus'][3 * self.workers:]
        occupied = {c for row in self.value['auxiliary'].values() for c in row['cpus']}
        cpus = [c for c in available if c not in occupied][:cpu_count]
        if cpu_count <= 0 or len(cpus) != cpu_count:
            raise ValueError('DISJOINT_AUXILIARY_CPU_REQUIRED')
        duration = min(timeout_s, self.admission['cutoff_epoch'] - self.now())
        if duration <= 0:
            raise TimeoutError('AUXILIARY_CUTOFF')
        cost = dict(physics_starts=0, physics_steps=0, rgb_pairs=0,
                    cpu_s=cpu_count * duration, output_bytes=output_bytes, memory_bytes=memory_bytes)
        for k in LIVE_KEYS + ('memory_bytes',):
            self._cost(k, cost[k])
        self._capacity(cost, memory_bytes)
        v = copy.deepcopy(self.value)
        v['auxiliary'][identifier] = dict(reservation=cost, cpus=cpus, deadline_epoch=self.now() + duration)
        write(self.path, v)
        self.value = v
        return copy.deepcopy(v['auxiliary'][identifier])

    def complete_auxiliary(self, identifier, *, cpu_s, output_bytes):
        v = copy.deepcopy(self.value)
        row = v['auxiliary'][identifier]
        for k, actual in (('cpu_s', cpu_s), ('output_bytes', output_bytes)):
            self._cost(k, actual)
            v['actual'][k] += actual
            if actual > row['reservation'][k]:
                v['status'] = 'LIVE_STOP_RESOURCE_BREACH'
        del v['auxiliary'][identifier]
        write(self.path, v)
        self.value = v

    def unknown(self, identifier):
        if identifier not in self.value['reservations']:
            raise ValueError('UNKNOWN_ID')
        self.value['status'] = 'LIVE_STOP_UNRECONCILED_RESERVATION_RETAINED'
        write(self.path, self.value)

    def finish(self):
        if self.value['reservations'] or self.value['auxiliary']:
            raise ValueError('UNRECONCILED_WORKERS')
        if self.value['status'] == 'LIVE_RUNNING':
            if len(self.value['records']) != self.base_records + len(self.plan['episodes']):
                raise ValueError('INCOMPLETE_PLAN')
            self.value['status'] = 'LIVE_TERMINAL_ALL_FIRST_ATTEMPTS'
            write(self.path, self.value)
        return copy.deepcopy(self.value)

    def close(self):
        self.lock.close()
