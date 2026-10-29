"""Physical storage proof for the one closed-predecessor replacement.

Pure filesystem inspection; no ledger changes, dispatch or deletion. The old
roots remain in the paid control envelope. Every new-root allocated block is
counted against the same already-charged main output upper. Symlink targets
are never traversed and cannot be used for generated output.
"""
import hashlib,json,os,re,stat
from pathlib import Path


def _need(value,why):
    if not value:raise PermissionError('ARX5_RETRY_PHYSICAL_'+why)


def _scan(root,*,permitted_links,transient_bytecode_retry=False):
    """Re-enumerate once after a vanished atomic bytecode temporary file.

    Never skip a missing entry in the accepted inventory. Historical scans
    remain strict; the current E2 root alone opts into a complete fresh scan.
    The caller still reserves control/temp growth in the physical envelope.
    """
    _need(type(transient_bytecode_retry)is bool,'EXPLICIT_TRANSIENT_RESCAN')
    try:return _scan_once(root,permitted_links=permitted_links)
    except FileNotFoundError as error:
        path=Path(error.filename)if error.filename is not None else None
        if not(transient_bytecode_retry and path is not None
                and path.is_relative_to(Path(root)) and path.parent.name=='__pycache__'
                and re.fullmatch(r'[^/]+\.cpython-\d+(?:\.opt-\d+)?\.pyc\.\d+',path.name)is not None):
            raise
        return _scan_once(root,permitted_links=permitted_links)


def _scan_once(root,*,permitted_links):
    root=Path(root);_need(root.is_dir() and not root.is_symlink(),'OWN_REAL_ROOT')
    files={};links={}
    for p in [root,*root.rglob('*')]:
        s=p.lstat();name='.' if p==root else p.relative_to(root).as_posix()
        if stat.S_ISLNK(s.st_mode):
            _need(name in permitted_links,'NO_OUTPUT_OR_UNREGISTERED_SYMLINK:'+name)
            links[name]=os.readlink(p);continue
        _need(stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode),'ONLY_REGULAR_FILES_AND_DIRECTORIES')
        _need(s.st_uid==os.getuid(),'OWN_FILES_ONLY')
        key=(s.st_dev,s.st_ino);row=dict(path=name,kind='file' if stat.S_ISREG(s.st_mode) else 'directory',
            dev=s.st_dev,inode=s.st_ino,logical=s.st_size,allocated=s.st_blocks*512,links=s.st_nlink,
            mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
        _need(key not in files,'NO_INTERNAL_HARDLINK_ALIASES')
        files[key]=row
    return files,links


def check(old_roots,new_root,*,old_metadata_ceiling,paid_output_ceiling,next_output_upper_bytes=0):
    _need(len(old_roots)==2 and type(old_metadata_ceiling)is int and old_metadata_ceiling>0
        and type(paid_output_ceiling)is int and paid_output_ceiling>0
        and type(next_output_upper_bytes)is int and 0<=next_output_upper_bytes<=paid_output_ceiling,
        'EXACT_TWO_ROOTS_AND_FINITE_BOUNDS')
    allowed={'model_inputs/nvidia/Cosmos-Reason2-2B'};old={};old_rows=[]
    for root in old_roots:
        rows,links=_scan(root,permitted_links=allowed)
        for key,row in rows.items():
            _need(key not in old,'NO_CROSS_ROOT_HARDLINKS');old[key]=row
            old_rows.append(dict(root=Path(root).name,**row))
            _need(not(row['kind']=='file' and row['path'].endswith(('.safetensors','.pt','.pth','.ckpt','.bin'))),
                'OLD_ROOTS_REMAIN_ZERO_MODEL_PAYLOAD')
    new,new_links=_scan(new_root,permitted_links=allowed)
    _need(not(set(old)&set(new)),'NEW_OUTPUT_DOES_NOT_ALIAS_RETAINED_OLD_FILES')
    old_bytes=sum(row['allocated']for row in old.values());new_bytes=sum(row['allocated']for row in new.values())
    _need(old_bytes<=old_metadata_ceiling,'RETAINED_OLD_CONTROL_OVER_PAID_ENVELOPE')
    _need(new_bytes+next_output_upper_bytes<=paid_output_ceiling,'NEW_FULL_OUTPUT_OVER_PAID_GROSS_ENVELOPE')
    wire=json.dumps(sorted(old_rows,key=lambda row:(row['root'],row['path'])),sort_keys=True,separators=(',',':')).encode()
    return dict(status='ACTUAL_RETRY_PHYSICAL_BOUNDS_ONLY_NO_PERMISSION',old_allocated_bytes=old_bytes,
        new_allocated_bytes=new_bytes,next_output_upper_bytes=next_output_upper_bytes,
        old_metadata_ceiling=old_metadata_ceiling,paid_output_ceiling=paid_output_ceiling,
        old_inventory_sha256=hashlib.sha256(wire).hexdigest(),old_roots=[str(Path(x))for x in old_roots],
        new_root=str(Path(new_root)),input_symlinks=new_links,ledger_changed=False,native_permission=False)
