"""Keep one N1 world on local storage and durably archive its complete output."""
import hashlib
import json
import os
from pathlib import Path
import shutil

from go2arx5_vla.world_model.storage import digest
from scripts.pi0292.durable import write
from scripts.sla.core_multienv_payload import measure, payload_volume


def capacity(control, storage, limits):
    target = payload_volume(control, storage)
    trajectory = limits['trajectory_reserve_bytes']
    local = measure(control, trajectory + limits['snapshot_reserve_bytes'])
    payload = measure(target, trajectory + limits['transfer_reserve_bytes'])
    return dict(passed=local['passed'] and payload['passed'], local=local, payload=payload)


def inventory(root):
    members = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise IOError('Unexpected member in the closed N1 native output: '+str(path))
        if path.is_file():
            members[str(path.relative_to(root))] = dict(bytes=path.stat().st_size, sha256=digest(path))
    return members


def sync_dir(path):
    fd = os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def copy_member(source, destination):
    with open(source,'rb') as reader, open(destination,'xb') as writer:
        os.fsync(reader.fileno())
        shutil.copyfileobj(reader,writer,length=4*1024**2)
        writer.flush()
        os.fsync(writer.fileno())
    shutil.copystat(source,destination,follow_symlinks=False)


def archive(case, storage, limits):
    """Publish exact bytes after child exit. A failed handoff retains the local raw directory."""
    case = Path(case)
    raw = case/'native-01'
    target = payload_volume(case,storage)/case.name
    pending = target.with_name(target.name+'.incoming')
    retained = case/'native-01.local-sealed'
    alias = case/'native-01.link-pending'
    if (raw.is_symlink() or not raw.is_dir() or raw.stat().st_uid != os.getuid()
            or any(os.path.lexists(path) for path in (target,pending,retained,alias))):
        raise FileExistsError('Expected one fresh local N1 trajectory and NFS destination')
    members = inventory(raw)
    size = sum(row['bytes'] for row in members.values())
    if not 0 < size <= limits['trajectory_reserve_bytes']:
        raise OSError('Closed N1 trajectory exceeded its reserved archive bytes')
    required = limits['trajectory_reserve_bytes']+limits['transfer_reserve_bytes']
    if not measure(target.parent,required)['passed']:
        raise OSError('NFS does not have the original archive and transfer reserves')
    shutil.copytree(raw,pending,copy_function=copy_member)
    if inventory(pending) != members:
        raise IOError('N1 trajectory bytes differ after NFS copy')
    for directory, _, _ in os.walk(pending, topdown=False):
        sync_dir(directory)
    os.replace(pending,target)
    sync_dir(target.parent)
    alias.symlink_to(target,target_is_directory=True)
    os.replace(raw,retained)
    try:
        os.replace(alias,raw)
        sync_dir(case)
    except OSError:
        if not os.path.lexists(raw):
            os.replace(retained,raw)
        raise
    receipt = dict(condition=case.name,members=members,bytes=size,count=len(members),
                   tree_sha256=hashlib.sha256(json.dumps(members,sort_keys=True).encode()).hexdigest(),
                   source_device=retained.stat().st_dev,archive_device=target.stat().st_dev)
    write(case/'ARCHIVE.private.json',receipt)
    shutil.rmtree(retained)
    sync_dir(case)
    return receipt
