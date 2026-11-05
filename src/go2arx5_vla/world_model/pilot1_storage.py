"""Accept the measured PILOT1 local data disk without changing legacy mounts."""
import json
from pathlib import Path
import stat

from .storage import digest


def qualified_block_path(path, proof_path, proof_sha256):
    path=Path(path).resolve(strict=True);proof=Path(proof_path).resolve(strict=True)
    value=json.loads(proof.read_text());root=Path(value['root']).resolve(strict=True)
    if (digest(proof)!=proof_sha256 or proof.parent!=root
            or value.get('status')!='PASSED_NODE_LOCAL_HDD_IO'
            or value.get('same_root_device')is not False
            or value.get('independent_lock_conflict_exit')!=73
            or value.get('atomic_rename_and_directory_fsync')is not True
            or not value.get('roundtrip_sha256')or value.get('roundtrip_sha256')!=value.get('expected_sha256')
            or not path.is_relative_to(root.parent)
            or path.stat().st_dev!=value['device']or root.stat().st_dev!=value['device']
            or path.stat().st_dev==Path('/').stat().st_dev):
        raise PermissionError('ACTUAL_HASHED_NON_ROOT_PILOT1_DISK_PROOF_REQUIRED')
    mounts=[]
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left,right=line.split(' - ',1);mount=left.split()[4];fs,source=right.split()[:2]
        if path==Path(mount)or path.is_relative_to(mount):mounts.append((len(mount),fs,source,mount))
    if not mounts:raise PermissionError('LIVE_LOCAL_BLOCK_MOUNT_REQUIRED')
    _,fs,source,mount=max(mounts);device=Path(source)
    if (fs!='fuseblk'or not source.startswith('/dev/')or not device.exists()
            or not stat.S_ISBLK(device.stat().st_mode)or device.stat().st_rdev!=path.stat().st_dev
            or value.get('mount')!=' '.join([source,fs,mount])):
        raise PermissionError('MEASURED_LOCAL_BLOCK_DEVICE_MUST_STILL_MATCH')
    return path


def local_path(spec, path):
    from scripts.pi0294.e1.smoke import node_local
    try:return node_local(path)
    except PermissionError:
        return qualified_block_path(path,spec['output_storage_proof'],spec['output_storage_proof_sha256'])
