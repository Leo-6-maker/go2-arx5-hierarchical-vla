"""Owned bounded tmpfs IPC, with immutable post-request disk evidence.

No scheduler, model, actuator or physics API. Live files survive a worker exit;
the host is not a power-loss archive. A lost/unarchived attempt is UNKNOWN and
must never be blindly submitted again.
"""
import hashlib
import os
from pathlib import Path
import stat
from scripts.pi0292.durable import write

MAX_BYTES = 64 * 1024**2


def initialize(root):
    root = Path(root).resolve(strict=True)
    mount = Path('/dev/shm')
    rows = Path('/proc/self/mountinfo').read_text().splitlines()
    if not any(r.split()[4] == str(mount) and r.split(' - ', 1)[1].split()[0] == 'tmpfs' for r in rows):
        raise RuntimeError('ACTUAL_TMPFS_REQUIRED')
    v = os.statvfs(mount)
    if v.f_bavail*v.f_frsize < 2*MAX_BYTES:
        raise MemoryError('PRIVATE_IPC_HEADROOM_REQUIRED')
    parent = mount / ('sla-ipc-' + str(os.getuid()))
    parent.mkdir(mode=0o700, exist_ok=True)
    if parent.is_symlink() or parent.stat().st_uid != os.getuid() or parent.stat().st_mode & 0o077:
        raise PermissionError('OWNED_PRIVATE_IPC_PARENT_REQUIRED')
    live = parent / hashlib.sha256(str(root).encode()).hexdigest()[:24]
    live.mkdir(mode=0o700)  # Never adopt an unknown or historical live capsule.
    write(root/'LIVE_IPC.private.json', dict(live_root=str(live), max_bytes=MAX_BYTES,
        backing='ACTUAL_TMPFS', archive='POST_REQUEST_RAW_BYTES_TO_REGISTERED_DISK',
        automatic_retry=False, native_requests=0))
    return live


def check_size(live):
    total = 0
    for p in Path(live).rglob('*'):
        mode = p.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise RuntimeError('REGULAR_PRIVATE_IPC_FILES_ONLY')
        if stat.S_ISREG(mode):
            total += p.stat().st_size
    if total > MAX_BYTES:
        raise MemoryError('BOUNDED_IPC_BYTES_EXCEEDED')
    return total


def archive_request(source, destination):
    """Caller must have received the immutable response and ceased mutations."""
    source, destination = Path(source), Path(destination)
    destination.mkdir()  # No replacement of a prior request/evidence identity.
    rows = []
    for p in sorted(source.rglob('*')):
        relative = p.relative_to(source)
        target = destination/relative
        if p.is_symlink():
            raise RuntimeError('NO_IPC_ARCHIVE_SYMLINKS')
        if p.is_dir():
            target.mkdir()
            continue
        raw = p.read_bytes()
        with target.open('xb') as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        pin = hashlib.sha256(raw).hexdigest()
        if hashlib.sha256(target.read_bytes()).hexdigest() != pin:
            raise IOError('IPC_ARCHIVE_RAW_BYTE_MISMATCH')
        rows.append(dict(file=str(relative), bytes=len(raw), sha256=pin))
    write(destination/'ARCHIVE.private.json', dict(files=rows, complete=True,
        source_live=str(source), original_references_retained=True, new_generations=0))
    return dict(path=str(destination/'ARCHIVE.private.json'),
        sha256=hashlib.sha256((destination/'ARCHIVE.private.json').read_bytes()).hexdigest())
