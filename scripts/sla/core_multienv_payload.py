"""Bind a cold Core collector to a separate payload volume with local ownership."""
import os
from pathlib import Path
import shutil
import stat

from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
from scripts.sla.core_multienv_scope import PLAN, SCOPE

GIB = 1024**3
LOCAL_CASE_RESERVE = GIB


def payload_volume(control, storage):
    if os.uname().nodename != 'corelab-4080s' or os.getuid() != 2009:
        raise PermissionError('The Core owner must perform the payload handoff')
    project = Path('/home/liuyu-nfs/Projects/GR00T-WBC/runs')
    root = Path(storage['root'])
    evidence = read_pin(storage['preflight'])
    old = Path(storage['preflight']['path']).parent
    before = root.stat()
    if (storage['plan'] != PLAN or root.parent != project or root == old
            or root != root.resolve(strict=True) or not stat.S_ISDIR(before.st_mode)
            or before.st_uid != os.getuid() or stat.S_IMODE(before.st_mode) != 0o700
            or control.stat().st_dev == before.st_dev or before.st_dev != evidence['device']
            or evidence['owner'] != os.getuid() or evidence['native_requests'] or evidence['native_ticks']
            or evidence['status'] != 'EXISTING_PRIVATE_PROJECT_NFS_IO_PREFLIGHT_PASS'
            or evidence['held_flock_child_result'] != 23 or not evidence['close_then_atomic_rename']):
        raise PermissionError('Expected a fresh private directory on the validated Core payload volume')
    return root


def measure(root, reserve):
    disk = shutil.disk_usage(root)
    floor = max(50*GIB, int(.15*disk.total)) + 8*GIB
    return dict(free_disk=disk.free, floor=floor, reserve=reserve,
                passed=disk.free >= floor+reserve)


def capacity(control, storage, limits):
    target = payload_volume(control, storage)
    local = measure(control, limits['snapshot_reserve_bytes']+limits['transfer_reserve_bytes']+LOCAL_CASE_RESERVE)
    payload = measure(target, limits['trajectory_reserve_bytes'])
    return dict(passed=local['passed'] and payload['passed'], local=local, payload=payload)


def create_output(case, local):
    """Create the raw directory before startup, keeping the registry and UNIX sockets local."""
    storage = case['multienv_payload_storage']
    folder = Path(case['registered_root'])
    campaign = Path(case['campaign_root'])
    condition = case['panel_entry_id']
    if (case['scope'] != SCOPE or case['goal_plan_sha'] != PLAN or case['sim_only_role'] != 'EXPERT'
            or case['hardware_connections_allowed'] is not False or folder != campaign/'cases'/condition
            or folder.name != condition or local != folder/'native-01'):
        raise PermissionError('Payload storage is limited to one registered Core expert case')
    root = payload_volume(folder, storage)
    if not capacity(campaign, storage, case['multienv'])['passed']:
        raise PermissionError('Both payload and local control volume reserves are required')
    target = root/condition
    if os.path.lexists(local) or os.path.lexists(target):
        raise FileExistsError('The raw payload identity already exists')
    target.mkdir(mode=0o700)
    local.symlink_to(target, target_is_directory=True)
    return local
