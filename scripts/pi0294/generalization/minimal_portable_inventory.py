"""Closed metadata identity across two clients of the same mounted filesystem.

Kernel st_dev is used for alias detection inside one observation. It is not a
filesystem identifier across hosts. The portable join additionally requires
the same backing filesystem, exact paths/inodes/stat fields and full contents.
This module reads metadata-only closed roots; it grants no native permission.
"""
import hashlib
import json
import os
from pathlib import Path
import stat

from . import minimal_retry_physical_guard as original

SCHEMA = 'ARX5_CLOSED_METADATA_PORTABLE_FILESYSTEM_IDENTITY_V1'


def need(ok, why):
    if not ok:
        raise PermissionError('MINIMAL_PORTABLE_' + why)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _unescape(value):
    for escaped, text in (('\\040', ' '), ('\\011', '\t'), ('\\012', '\n'), ('\\134', '\\')):
        value = value.replace(escaped, text)
    return value


def filesystem(root):
    """Use real mount backing identity, not a caller-supplied filesystem name."""
    root = Path(root).resolve(strict=True)
    matches = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        before, after = line.split(' - ', 1); left, right = before.split(), after.split()
        mountpoint = Path(_unescape(left[4]))
        if root == mountpoint or root.is_relative_to(mountpoint):
            matches.append((len(mountpoint.parts), left, right))
    need(matches, 'ACTUAL_MOUNT_FOR_ROOT')
    _, left, right = max(matches, key=lambda item:item[0])
    fsid = os.statvfs(root).f_fsid
    need(type(fsid) is int and fsid >= 0, 'ACTUAL_FILESYSTEM_ID')
    return dict(filesystem_type=right[0], filesystem_source=_unescape(right[1]),
                filesystem_root=_unescape(left[3]), filesystem_id=fsid)


def capture(roots, *, metadata_ceiling, file_hash_cache=None):
    need(type(roots) is list and 1 <= len(roots) <= 4
         and type(metadata_ceiling) is int and metadata_ceiling > 0,
         'FINITE_CLOSED_ROOTS_AND_BOUND')
    paths = [Path(value) for value in roots]
    need(len({str(p) for p in paths}) == len(paths)
         and len({p.name for p in paths}) == len(paths), 'DISTINCT_ROOTS')
    rows, seen, allocated, contents = [], set(), 0, 0
    bindings = {}
    for root in paths:
        scanned, links = original._scan(root, permitted_links={'model_inputs/nvidia/Cosmos-Reason2-2B'})
        fs = filesystem(root)
        bindings[root.name] = dict(path=str(root), filesystem=fs, input_symlinks=links)
        for key, row in scanned.items():
            need(key not in seen, 'NO_CROSS_ROOT_HARDLINK_ALIAS'); seen.add(key)
            allocated += row['allocated']; need(allocated <= metadata_ceiling, 'METADATA_LIMIT')
            need(not (row['kind'] == 'file' and row['path'].endswith(
                ('.safetensors', '.pt', '.pth', '.ckpt', '.bin'))), 'NO_RETAINED_MODEL_PAYLOAD')
            path = root / row['path']; first = path.lstat()
            item = {k:value for k, value in row.items() if k != 'dev'}
            item.update(root=root.name, uid=first.st_uid, mode=stat.S_IMODE(first.st_mode))
            if row['kind'] == 'file':
                need(first.st_nlink == 1, 'NO_UNCAPTURED_HARDLINK_CONSUMER')
                key = (str(path), first.st_dev, first.st_ino, first.st_size,
                       first.st_blocks, first.st_nlink, first.st_mtime_ns,
                       first.st_ctime_ns, first.st_uid, first.st_mode)
                cached = file_hash_cache.get(key) if file_hash_cache is not None else None
                if cached is None:
                    h = hashlib.sha256()
                    with path.open('rb') as handle:
                        need(os.fstat(handle.fileno()).st_ino == first.st_ino, 'SAME_OPEN_INODE')
                        while True:
                            block = handle.read(1024**2)
                            if not block: break
                            contents += len(block); need(contents <= metadata_ceiling, 'BOUNDED_CONTENT_READ')
                            h.update(block)
                    cached = h.hexdigest()
                    if file_hash_cache is not None:
                        need(len(file_hash_cache) < 16000, 'FINITE_PRIVATE_HASH_CACHE')
                        file_hash_cache[key] = cached
                item['sha256'] = cached
            last = path.lstat()
            changed = {field:dict(before=getattr(first, field), after=getattr(last, field))
                for field in ('st_dev', 'st_ino', 'st_size', 'st_nlink',
                              'st_mtime_ns', 'st_ctime_ns', 'st_uid', 'st_mode')
                if getattr(first, field) != getattr(last, field)}
            need(not changed, 'UNCHANGED_DURING_FULL_CONTENT_READ:' + root.name + ':'
                 + row['path'] + ':' + json.dumps(changed, sort_keys=True))
            # NFS may update block-allocation metadata without changing file
            # identity/content. Retain the largest actually observed physical
            # allocation and bound it independently; never infer a refund.
            current_allocated=max(item['allocated'],first.st_blocks*512,last.st_blocks*512)
            allocated+=current_allocated-item['allocated'];item['allocated']=current_allocated
            need(allocated<=metadata_ceiling,'METADATA_LIMIT')
            rows.append(item)
    rows.sort(key=lambda row:(row['root'], row['path']))
    value = dict(schema=SCHEMA, roots=bindings, rows=rows, allocated_bytes=allocated,
                 metadata_ceiling=metadata_ceiling)
    return dict(value=value, portable_sha256=digest(value),
                content_bytes_read=contents,
                kernel_device_ids=sorted({key[0] for key in seen}),
                hostname=os.uname().nodename, native_permission=False)


def identity_value(value):
    """File identity excludes mutable filesystem block-allocation reports.

    Every observation still carries, sums and bounds its actual allocation.
    Full bytes, inode, logical size, link count, times, mode and uid stay exact.
    """
    return dict(value, allocated_bytes=None,
                rows=[{k:v for k,v in row.items() if k != 'allocated'}
                      for row in value['rows']])


def identity_digest(value):
    return digest(identity_value(value))


def authorized_failed_source_cleanup(expected, actual):
    """One observed PR176 cleanup; every surviving byte/stat remains bound.

    These full identities include all four filesystem/root bindings and rows.
    This is no permission for later deletion, restamping or inventory adoption.
    The original gross ceiling and historical costs remain with the caller.
    """
    need(identity_digest(expected)=='b69bf8c72decf9e47412391e5ec7001d6fee3464bf6d020ce1c699b6a5727f3b'
         and identity_digest(actual)=='8ff9c5dddc32c2c5cf06cd2206ba4473730ea5f016b9786adc7aad4e4645f982',
         'ONLY_EXACT_AUTHORIZED_20261030_FAILED_SOURCE_CLEANUP')
    root='arx5-minimal-raw-parent-first512-02'
    prefix='capsules/a01875902809f814b2bd03d095528cdaed84668c/code/'
    before={(v['root'],v['path']):v for v in identity_value(expected)['rows']}
    after={(v['root'],v['path']):v for v in identity_value(actual)['rows']}
    removed=set(before)-set(after);added=set(after)-set(before)
    marker=(root,'FAILED_SOURCE_PAYLOAD_REMOVED_20261030.json')
    need(len(removed)==1493 and all(r==root and p.startswith(prefix)
         and before[(r,p)]['kind']=='file'for r,p in removed)
         and added=={marker}and after[marker]['sha256']=='8d0d0805e1f3ba6a251288ce354d6c00ed7e0b97017ab17c8053591cd81d60a0',
         'EXACT1493_OBSOLETE_SOURCE_LEAVES_AND_ORIGINAL_DELETE_RECORD')
    changed=0
    for key in set(before)&set(after):
        a,b=before[key],after[key]
        fields={k for k in set(a)|set(b)if a.get(k)!=b.get(k)}
        if fields:
            changed+=1
            need(a['kind']==b['kind']=='directory'and key[0]==root
                 and(key[1]=='.'or key[1].startswith(prefix))
                 and fields<={'ctime_ns','mtime_ns','logical'},
                 'ONLY_KNOWN_CLEANUP_PARENT_DIRECTORY_METADATA')
    need(changed==53 and expected['roots']==actual['roots'],
         'ALL_SURVIVING_FILES_AND_FILESYSTEM_BINDINGS_UNCHANGED')
    return dict(authority_sha256='49506e3a135ad6ad931cd672cb9f0b9f6214e47fb072fd160b3f29b5932f007b',
                removed_obsolete_source_files=1493,deleted_raw_source_is_not_recomputable=True,
                original_historical_gross_ceiling_and_costs_retained=True,native_permission=False)


def verify(expected, actual):
    for proof in (expected, actual):
        need(proof.get('native_permission') is False
             and proof['value']['schema'] == SCHEMA
             and proof['portable_sha256'] == digest(proof['value'])
             and all(type(row.get('allocated'))is int and row['allocated']>=0
                     for row in proof['value']['rows'])
             and proof['value']['allocated_bytes']==sum(row['allocated']for row in proof['value']['rows'])
             and proof['value']['allocated_bytes'] <= proof['value']['metadata_ceiling'],
             'COMPLETE_HASHED_METADATA_PROOF')
    cleanup=None
    if identity_value(expected['value'])!=identity_value(actual['value']):
        cleanup=authorized_failed_source_cleanup(expected['value'],actual['value'])
    result=dict(portable_sha256=actual['portable_sha256'],
                portable_identity_sha256=identity_digest(actual['value']),
                historical_allocated_bytes=expected['value']['allocated_bytes'],
                current_allocated_bytes=actual['value']['allocated_bytes'],
                kernel_device_namespace_changed=expected['kernel_device_ids'] != actual['kernel_device_ids'],
                native_permission=False, ledger_changed=False)
    if cleanup is not None:result['already_authorized_failed_source_cleanup']=cleanup
    return result
