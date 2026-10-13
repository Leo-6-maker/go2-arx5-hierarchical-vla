"""New authority ledger; reserve incomplete attempts until reconciled."""
from __future__ import annotations

import os
from pathlib import Path
import resource
import time

from .durable import OwnerLock, read, write

GIB = 1024 ** 3
EPISODE_BYTES = 512 * 1024 ** 2
EPISODE_CPU_UPPER = 4 * 900


def owned_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file() and not p.is_symlink())


def cpu():
    a = resource.getrusage(resource.RUSAGE_SELF)
    b = resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


class Budget:
    def __init__(self, root, gpu_service=False):
        if gpu_service:
            raise ValueError('CORELAB_GPU_FORBIDDEN')
        self.root = Path(root)
        self.path = self.root / 'RESOURCE_LEDGER.json'
        self.lock = OwnerLock(self.root / 'PHYSICS_OWNER.lock')
        self.adoption = read(self.root / 'ADOPTION.json')
        self.plan = read(self.root / 'ATTEMPT_PLAN.json')
        self.value = read(self.path)
        self.cpu0, self.began = cpu(), time.monotonic()
        self.identifier = None
        self.last_check = -float('inf')

    def start(self, identifier, source):
        self.guard()
        if time.time() >= self.adoption['new_work_cutoff_epoch']:
            raise RuntimeError('NEW_WORK_CUTOFF')
        if identifier not in self.plan['attempts'] or any(r['id'] == identifier for r in self.value['starts']):
            raise RuntimeError('UNREGISTERED_OR_ALREADY_STARTED')
        if any(r['status'] == 'STARTED' for r in self.value['starts']):
            raise RuntimeError('UNRECONCILED_PRIOR_ATTEMPT')
        if len(self.value['starts']) >= 54 or self.value['physics_steps'] + 16000 > 864000 or self.value['pairs'] + 641 > 34614:
            raise RuntimeError('COUNT_RESERVATION_EXCEEDED')
        if self.value['CPU_s'] + self.value['CPU_upper_s'] + EPISODE_CPU_UPPER >= 96 * 3600:
            raise RuntimeError('CPU_RESERVATION_EXCEEDED')
        size = owned_bytes(self.root)
        if size + EPISODE_BYTES + 2 * GIB > 40 * GIB:
            raise RuntimeError('SPACE_RESERVATION_EXCEEDED')
        fs = os.statvfs(self.root)
        if fs.f_bavail * fs.f_frsize < EPISODE_BYTES + 10 * GIB:
            raise RuntimeError('FILESYSTEM_MARGIN')
        self.identifier = identifier
        self.value['starts'].append(dict(id=identifier, status='STARTED', source_sha=source,
            attempt_uid=self.adoption['run_uuid']+'/'+identifier, steps=0, pairs=0, epoch=time.time(),
            reserved_steps=16000, reserved_pairs=641, CPU_upper_s=EPISODE_CPU_UPPER,
            pid=os.getpid(), proc_start=Path('/proc/self/stat').read_text().split()[21]))
        self.value['CPU_upper_s'] += EPISODE_CPU_UPPER
        self.value['owned_bytes_at_start'] = size
        write(self.path, self.value)

    def guard(self):
        if time.time() >= self.adoption['new_work_cutoff_epoch']:
            raise RuntimeError('PHYSICS_CUTOFF_CLOSEOUT_RESERVE')
        if time.monotonic() - self.began >= 890:
            raise RuntimeError('EPISODE_WALL_LIMIT')
        if cpu() - self.cpu0 >= EPISODE_CPU_UPPER:
            raise RuntimeError('EPISODE_CPU_LIMIT')
        if self.identifier and time.monotonic() - self.last_check >= 5:
            if owned_bytes(self.root / 'episodes' / self.identifier) >= EPISODE_BYTES - 32 * 1024 ** 2:
                raise RuntimeError('EPISODE_OUTPUT_RESERVATION')
            fs = os.statvfs(self.root)
            if fs.f_bavail * fs.f_frsize < 10 * GIB:
                raise RuntimeError('FILESYSTEM_MARGIN')
            self.last_check = time.monotonic()

    def progress(self, identifier, steps, pairs):
        row = next(r for r in self.value['starts'] if r['id'] == identifier)
        if not row['steps'] <= steps <= 16000 or not row['pairs'] <= pairs <= 641:
            raise RuntimeError('NONMONOTONIC_OR_EXCESS_EPISODE_COUNTS')
        self.value['physics_steps'] += steps - row['steps']
        self.value['pairs'] += pairs - row['pairs']
        row.update(steps=steps, pairs=pairs)
        write(self.path, self.value)

    def finish(self, identifier, status, GPU_s):
        assert GPU_s == 0
        row = next(r for r in self.value['starts'] if r['id'] == identifier)
        used = cpu() - self.cpu0
        self.value['CPU_s'] += used
        self.value['CPU_upper_s'] -= row['CPU_upper_s']
        row.update(status=status, CPU_s=used, CPU_upper_s=0, reserved_steps=0, reserved_pairs=0)
        self.value['owned_bytes'] = owned_bytes(self.root)
        write(self.path, self.value)
        self.lock.close()

    def finish_cpu(self, name):
        used = cpu() - self.cpu0
        self.value['CPU_s'] += used
        self.value['events'].append(dict(stage=name,CPU_s=used,physics_steps=0))
        self.value['owned_bytes'] = owned_bytes(self.root)
        write(self.path,self.value)
        self.lock.close()
