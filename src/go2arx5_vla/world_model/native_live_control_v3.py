"""Opt-in tmpfs control/trace; original step and ROS journals stay on disk."""
import hashlib
import json
import os
import stat
from pathlib import Path

VERSION = 'CORELAB_TMPFS_LIVE_AND_TRACE_CONTROL_V3'
CONTROL_FILES = ('state.bin', 'pause_ack.json', 'step_quota', 'quota.binding', 'gripper.bin')
TRACE_FILES = ('init.trace', 'init.trace.snapshots', 'init.trace.summary', 'trace_seal_request')
FILES = CONTROL_FILES + TRACE_FILES
MAX_BYTES = 8 * 1024**2
TRACE_MAX_BYTES = 768 * 1024**2


def configuration(spec):
    value = spec.get('native_live_control_v3')
    if value is None:
        return None
    from .corelab_expert_semantics_v3 import ROLE, recorded_tail
    if (value != dict(version=VERSION)
            or spec.get('phase') != 'collect-target-task-learning-grasp-v1'
            or spec.get('task_learning_role') != ROLE or not recorded_tail(spec)):
        raise ValueError('TMPFS_CONTROL_REQUIRES_NEW_COMPLETE_EXPERT_CAPSULE')
    return value


def live_path(adapter, name):
    if name not in FILES:
        raise ValueError('BOUNDED_LIVE_CONTROL_FILENAME')
    return Path(getattr(adapter, 'live_control_output', None) or adapter.output) / name


def acknowledgement_wait(spec, original_seconds):
    # A measured WBC mode notification arrived ~18 wall seconds after the
    # paused request on the shared FUSE disk. Wait for the original receipt;
    # never repeat a command or substitute a previous mode/source packet.
    return 90 if configuration(spec) is not None else original_seconds


def maximum_wall_seconds(spec, original_seconds):
    return 3600 if configuration(spec) is not None else original_seconds


def initialize(adapter):
    if configuration(adapter.spec) is None:
        return
    output = Path(adapter.output).resolve(strict=True)
    shm = Path('/dev/shm')
    mounts = Path('/proc/self/mountinfo').read_text().splitlines()
    if not any(row.split()[4] == str(shm) and row.split(' - ', 1)[1].split()[0] == 'tmpfs'
               for row in mounts):
        raise RuntimeError('ACTUAL_TMPFS_REQUIRED')
    if os.statvfs(shm).f_bavail * os.statvfs(shm).f_frsize < 4 * (MAX_BYTES + TRACE_MAX_BYTES):
        raise MemoryError('TMPFS_LIVE_CONTROL_CAPACITY')
    parent = shm / ('sla-pt-' + str(os.getuid()))
    parent.mkdir(mode=0o700, exist_ok=True)
    if parent.is_symlink() or parent.stat().st_uid != os.getuid() or parent.stat().st_mode & 0o077:
        raise PermissionError('OWNED_PRIVATE_TMPFS_PARENT_REQUIRED')
    identity = hashlib.sha256(str(output).encode()).hexdigest()
    root = parent / identity[:24]
    root.mkdir(mode=0o700)  # No adoption or overwrite of an unknown live world.
    adapter.live_control_output = root
    from .storage import write_json
    write_json(output / 'LIVE_CONTROL.private.json', dict(version=VERSION,
        output=str(output), live_output=str(root), output_identity_sha256=identity,
        files=list(FILES), max_bytes=MAX_BYTES, trace_max_bytes=TRACE_MAX_BYTES,
        journal='DURABLE_ORIGINAL_STATE_JOURNAL',
        trace_persistence='RAW_BYTES_ARCHIVED_AFTER_ALL_OWNED_CHILDREN_TERMINATE'))
    # The frozen native recorder derives this filename from state.bin. Keep
    # that append stream on the original disk, outside the bounded tmpfs set.
    (output / 'state.bin.journal').open('xb').close()
    (root / 'state.bin.journal').symlink_to(output / 'state.bin.journal')
    for name in FILES:
        (output / name).symlink_to(root / name)


def check(adapter):
    root = getattr(adapter, 'live_control_output', None)
    if root is None:
        return
    control_bytes = trace_bytes = 0
    for path in root.iterdir():
        if path.name == 'state.bin.journal':
            if not path.is_symlink() or path.resolve() != (Path(adapter.output) / path.name).resolve():
                raise RuntimeError('ORIGINAL_DURABLE_JOURNAL_ROUTE_CHANGED')
            continue
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(mode):
            raise RuntimeError('UNEXPECTED_LIVE_CONTROL_MEMBER')
        # Both frozen C++ and existing Python publish with .pending + rename.
        if path.name not in FILES and path.name not in {name + '.pending' for name in FILES} | {'step_quota.pending', 'gripper.pending'}:
            raise RuntimeError('UNEXPECTED_LIVE_CONTROL_MEMBER')
        try:
            size = path.stat().st_size
            if path.name in TRACE_FILES or path.name=='trace_seal_request.pending':
                trace_bytes += size
            else:
                control_bytes += size
        except FileNotFoundError:
            continue  # A same-directory atomic publication just completed.
    if control_bytes > MAX_BYTES or trace_bytes > TRACE_MAX_BYTES:
        raise MemoryError('BOUNDED_LIVE_CONTROL_BYTES_EXCEEDED')


def archive(adapter):
    root = getattr(adapter, 'live_control_output', None)
    if root is None:
        return
    if any(process.poll() is None for _, process in adapter.processes):
        raise RuntimeError('LIVE_CONTROL_ARCHIVE_REQUIRES_OWNED_CHILDREN_TERMINAL')
    check(adapter)
    output = Path(adapter.output)
    rows = []
    for name in FILES:
        source, destination = root / name, output / name
        if not destination.is_symlink() or destination.readlink() != source:
            raise RuntimeError('CANONICAL_LIVE_CONTROL_ALIAS_CHANGED')
        if source.exists():
            raw = source.read_bytes()
            temporary = output / (name + '.archive-pending')
            with temporary.open('xb') as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            if temporary.read_bytes() != raw:
                raise IOError('LIVE_CONTROL_ARCHIVE_READBACK')
            os.replace(temporary, destination)
            rows.append(dict(name=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        else:
            destination.unlink()
            rows.append(dict(name=name, absent_before_native_publication=True))
    from .storage import write_json
    write_json(output / 'LIVE_CONTROL_ARCHIVE.private.json', dict(version=VERSION,
        children_terminal=True, files=rows, journal_retained_on_original_disk=True,
        raw_trace_archived_without_reencoding=True))
    # Leave tmpfs evidence intact if any archive step above failed.
    for path in root.iterdir():
        path.unlink()
    root.rmdir()
    adapter.live_control_output = None
