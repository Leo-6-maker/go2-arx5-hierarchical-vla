"""Persistent single-owner jobs with explicit terminal and message identities."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    data = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    temporary = path.with_name(path.name + '.pending.' + str(os.getpid()))
    with temporary.open('xb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class OwnerLock:
    def __init__(self, path):
        self.handle = Path(path).open('a')
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise

    def close(self):
        self.handle.close()


class MessageGate:
    """A rejected message never advances the committed sequence."""
    def __init__(self, run, episode):
        self.run, self.episode, self.sequence = run, episode, 0

    def accept(self, packet, now):
        expected = hashlib.sha256(packet['payload'].encode()).hexdigest()
        if packet['run'] != self.run or packet['episode'] != self.episode:
            raise ValueError('WRONG_IDENTITY')
        if packet['sequence'] != self.sequence:
            raise ValueError('DUPLICATE_OR_REORDERED')
        if now > packet['expires']:
            raise ValueError('EXPIRED')
        if packet['sha256'] != expected:
            raise ValueError('CRC_MISMATCH')
        self.sequence += 1


def fake(root, duration):
    root = Path(root)
    root.mkdir(exist_ok=True, parents=True)
    try:
        lock = OwnerLock(root / 'owner.lock')
    except BlockingIOError:
        print('DUPLICATE_OWNER_REJECTED', flush=True)
        return 73
    path = root / 'STATE.json'
    if path.exists():
        print('EXISTING_JOB_REJECTED', flush=True)
        return 74
    state = dict(pid=os.getpid(), proc_start=Path('/proc/self/stat').read_text().split()[21],
                 status='RUNNING', sequence=0, started=time.time(), physics_steps=0, model_calls=0)
    write(path, state)
    stop = time.monotonic() + duration
    while time.monotonic() < stop:
        state.update(sequence=state['sequence'] + 1, updated=time.time())
        write(path, state)
        print(json.dumps(state), flush=True)
        time.sleep(.25)
    state.update(status='TERMINAL_SUCCESS', updated=time.time())
    write(path, state)
    lock.close()
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fake-root', required=True)
    p.add_argument('--duration', type=float, default=15)
    a = p.parse_args()
    raise SystemExit(fake(a.fake_root, a.duration))
