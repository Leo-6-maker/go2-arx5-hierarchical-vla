"""Cold check of the frozen loader's exact local alias, without loading weights."""
import os
import hashlib
from pathlib import Path

from scripts.pi0294.local_backbone import local_model_name


def prepare_alias(root,bindings):
    """Create only the named link in a fresh cold binder, never copy tensor files."""
    root=Path(root)
    if root.is_symlink() or (root/'RUN_MANIFEST.json').exists() or any((root/'worker_processes').glob('*.json')):
        raise PermissionError('PV1_MODEL_ALIAS_CREATION_ONLY_BEFORE_FINAL_COLD_BINDING')
    snapshot=Path(bindings['cosmos']).resolve(strict=True)
    if not snapshot.is_dir() or not (snapshot/'config.json').is_file():
        raise PermissionError('PV1_EXISTING_PINNED_SNAPSHOT_ONLY_NO_DOWNLOAD')
    for p in (root/'model_inputs',root/'model_inputs/nvidia'):
        if p.is_symlink():raise PermissionError('PV1_MODEL_ALIAS_PARENT_ESCAPE')
        p.mkdir(exist_ok=True,parents=True,mode=0o700)
    alias=root/'model_inputs/nvidia/Cosmos-Reason2-2B'
    if os.path.lexists(alias):
        raise FileExistsError('PV1_PRESERVE_EXISTING_MODEL_INPUT_NO_REPLACEMENT_OR_DELETE')
    alias.symlink_to(snapshot,target_is_directory=True)
    return preflight(root,bindings)


def preflight(root,bindings):
    root=Path(root)
    relative='model_inputs/nvidia/Cosmos-Reason2-2B'
    alias=root/relative
    for parent in (root/'model_inputs',root/'model_inputs/nvidia'):
        if parent.is_symlink() or not parent.is_dir():
            raise PermissionError('PV1_LOCAL_MODEL_ALIAS_PARENT_NOT_A_REAL_DIRECTORY')
    # Preserve the original selector: an equivalent copied directory is not an alias.
    selected=Path(local_model_name(root,bindings))
    if alias.lstat().st_uid!=os.getuid() or selected!=alias:
        raise PermissionError('PV1_LOCAL_MODEL_ALIAS_OWNER_OR_NAME_CHANGED')
    snapshot=Path(bindings['cosmos']).resolve(strict=True)
    config=snapshot/'config.json'
    if not snapshot.is_dir() or not config.is_file():
        raise PermissionError('PV1_PINNED_COSMOS_SNAPSHOT_METADATA_MISSING')
    return dict(status='PASS_EXACT_PINNED_LOCAL_COSMOS_ALIAS_NO_MODEL_LOAD',alias_relative=relative,
        exact_snapshot_resolved=str(snapshot),config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        symlink_target_readback=os.readlink(alias),new_model_loads=0,new_download_bytes=0)
