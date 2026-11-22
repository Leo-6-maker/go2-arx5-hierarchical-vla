"""Opt-in tmpfs control/trace; original step and ROS journals stay on disk."""
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path

VERSION = 'CORELAB_TMPFS_LIVE_AND_TRACE_CONTROL_V3'
CONTROL_FILES = ('state.bin', 'pause_ack.json', 'step_quota', 'quota.binding', 'gripper.bin')
TRACE_FILES = ('init.trace', 'init.trace.snapshots', 'init.trace.summary', 'trace_seal_request')
FILES = CONTROL_FILES + TRACE_FILES
MAX_BYTES = 8 * 1024**2
TRACE_MAX_BYTES = 768 * 1024**2


def configuration(spec):
    if spec.get('d2_isolated_live_control') is not None:
        from .multitask_scope_d2 import PLAN, SCOPE

        if (spec['d2_isolated_live_control'] != VERSION
                or spec.get('scope') != SCOPE or spec.get('plan_commit') != PLAN
                or spec.get('sim_only_role') not in ('INTERFACE','EXPERT')
                or not spec.get('d2_world_binding')):
            raise PermissionError('D2_ISOLATED_MOBILE_WORLD_REQUIRED_FOR_TMPFS')
        return dict(version=VERSION)
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
    return 90 if spec.get('d2_isolated_live_control') is None and configuration(spec) is not None else original_seconds


def maximum_wall_seconds(spec, original_seconds):
    from . import task_learning_grasp_v1 as task
    if spec.get('r1_condition_wall_repair_v5') is not None:
        from .stable_task_scene_v1 import pinned
        from .sla_r1_diagnostic_v5 import validate_panel
        from .sla_r1_suffix_v5 import window
        pin=spec['r1_condition_wall_repair_v5'];panel=pinned(pin)
        rows=validate_panel(panel)
        task.need(spec.get('task_learning_role')==task.SLA_ROLE
            and spec.get('task_learning_grasp')==task.protocol(task.SLA_ROLE)
            and spec.get('stable_task_panel')==pin['path']
            and spec.get('stable_task_panel_sha256')==pin['sha256']
            and Path(spec['registered_root']).name in {r['native_run']for r in rows[13:]},
            'R1_ONLY_DECLARED_UNSTARTED_SUFFIX_NATIVE_WALL_REPAIR')
        return window(panel)
    if task.dense_sla_role(spec.get('task_learning_role')):
        if spec.get('task_learning_grasp') != task.protocol(spec['task_learning_role']):
            raise ValueError('DENSE_P10_FINITE_NATIVE_PROTOCOL_REQUIRED')
        return 3600
    return 3600 if configuration(spec) is not None else original_seconds


def initialize(adapter):
    persistent = None
    if configuration(adapter.spec) is None:
        storage = adapter.spec.get('multienv_payload_storage')
        if not storage or 'live_control' not in storage:
            return
        from scripts.sla.core_multienv_payload import validate_live_control
        persistent = validate_live_control(adapter.spec, Path(adapter.output))
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
    journal = output / 'state.bin.journal'
    journal_target = journal
    journal_details = dict(journal='DURABLE_ORIGINAL_STATE_JOURNAL')
    if persistent is not None:
        journal_target, cap = persistent
        adapter.local_state_journal = persistent
        journal_details = dict(journal='DURABLE_LOCAL_STATE_JOURNAL',journal_live_path=str(journal_target),
                               journal_max_bytes=cap)
        journal.symlink_to(journal_target)
    journal_target.open('xb').close()
    from .storage import write_json
    write_json(output / 'LIVE_CONTROL.private.json', dict(version=VERSION,
        output=str(output), live_output=str(root), output_identity_sha256=identity,
        files=list(FILES), max_bytes=MAX_BYTES, trace_max_bytes=TRACE_MAX_BYTES,
        **journal_details,
        trace_persistence='RAW_BYTES_ARCHIVED_AFTER_ALL_OWNED_CHILDREN_TERMINATE'))
    # The frozen native recorder derives this filename from state.bin. Keep
    # that append stream on the original disk, outside the bounded tmpfs set.
    (root / journal.name).symlink_to(journal_target)
    for name in FILES:
        (output / name).symlink_to(root / name)


def check(adapter):
    root = getattr(adapter, 'live_control_output', None)
    if root is None:
        return
    control_bytes = trace_bytes = 0
    for path in root.iterdir():
        if path.name == 'state.bin.journal':
            persistent = getattr(adapter,'local_state_journal',None)
            if persistent is not None:
                journal, cap = persistent
                info = journal.lstat()
                if path.readlink() != journal or not stat.S_ISREG(info.st_mode) or info.st_size > cap:
                    raise RuntimeError('BOUNDED_LOCAL_STATE_JOURNAL_CHANGED')
                continue
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


def archive_local_journal(source, destination):
    from .storage import digest

    if destination.readlink() != source:
        raise RuntimeError('CANONICAL_DURABLE_JOURNAL_ALIAS_CHANGED')
    expected = digest(source)
    size = source.stat().st_size
    temporary = destination.with_suffix('.journal.archive-pending')
    with source.open('rb') as reader, temporary.open('xb') as writer:
        os.fsync(reader.fileno())
        shutil.copyfileobj(reader,writer,length=1024**2)
        writer.flush()
        os.fsync(writer.fileno())
    if temporary.stat().st_size != size or digest(temporary) != expected:
        raise IOError('LOCAL_STATE_JOURNAL_ARCHIVE_READBACK')
    os.replace(temporary,destination)
    return dict(name=destination.name,bytes=size,sha256=expected)


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
    persistent = getattr(adapter,'local_state_journal',None)
    if persistent is not None:
        state_journal, _ = persistent
        rows.append(archive_local_journal(state_journal,output/state_journal.name))
    write_json(output / 'LIVE_CONTROL_ARCHIVE.private.json', dict(version=VERSION,
        children_terminal=True, files=rows, journal_retained_on_original_disk=persistent is None,
        raw_trace_archived_without_reencoding=True))
    if persistent is not None:
        fd = os.open(output,os.O_RDONLY|os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        state_journal.unlink()
        state_journal.parent.rmdir()
        adapter.local_state_journal = None
    # Leave tmpfs evidence intact if any archive step above failed.
    for path in root.iterdir():
        path.unlink()
    root.rmdir()
    adapter.live_control_output = None
