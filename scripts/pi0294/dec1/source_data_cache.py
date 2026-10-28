"""Reuse a closed failed staging attempt without changing any old payload."""
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile

from scripts.pi0292.durable import read, write
from scripts.pi0294.fd1 import sha

PLAN='plans/CLOSED_STAGING_CACHE_REUSE.json'


def previous(root):
    root=Path(root);m=read(root/'RUN_MANIFEST.json');plan=read(root/PLAN)
    p=root/PLAN
    if p.is_symlink() or p.stat().st_mode&0o222 or m['sealed_inputs'].get(PLAN)!=sha(p):
        raise PermissionError('SOURCE_CACHE_MANIFEST_SEALED_PLAN')
    old=Path(plan['source_root'])
    if (old.is_symlink() or old.parent.resolve()!=root.parent.resolve() or old==root
            or old.stat().st_uid!=os.getuid() or plan.get('schema')!='CLOSED_FAILED_TRAIN139_STAGING_CACHE_V1'
            or sha(old/'RUN_MANIFEST.json')!=plan['source_manifest_sha256']
            or sha(old/'RESULT.private.json')!=plan['source_result_sha256']):
        raise PermissionError('SOURCE_CACHE_EXACT_OWN_CLOSED_SIBLING')
    binding=read(old/'OUTPUT_BINDING.private.json');lease=read(binding['lease'])
    process=read(old/'PROCESS.private.json');stat=Path('/proc')/str(process['pid'])/'stat'
    fields=stat.read_text().rsplit(')',1)[1].split() if stat.exists() else None
    result=read(old/'RESULT.private.json')
    if (lease.get('status')!='CLOSED_FAILED_CPU_STAGING_PRESERVED'
            or sha(binding['lease'])!=plan['source_lease_sha256']
            or fields and fields[0]!='Z' and fields[19]==str(process['proc_start'])
            or result.get('status')!='FAILED_PRESERVED_NO_MODEL_OR_HF_MUTATION'
            or result.get('failure',{}).get('error')!='SOURCE_DATA_FINITE_WALL_CUTOFF'
            or (old/'STOP.json').exists()):
        raise PermissionError('SOURCE_CACHE_NO_LIVE_UNKNOWN_OR_SAFETY_STOP_RESTART')
    return old


class OutputBudget:
    def __init__(self, upper):
        self.upper=upper;self.reserved=64*1024**2

    def reserve(self, size):
        if type(size) is not int or size<0 or self.reserved+size>self.upper:
            raise PermissionError('SOURCE_CACHE_CONSERVATIVE_OUTPUT_BOUND')
        self.reserved+=size


def cached_download(root, old, receipt, get, budget, guard):
    root,old=Path(root),Path(old);index={r['path']:r for r in receipt['files']}
    reused=[];downloaded=[]
    def fetch(repo,path,*,revision,local_dir,token):
        if repo!=receipt['repo_id'] or revision!=receipt['revision'] or token is not False or path not in index:
            raise PermissionError('SOURCE_CACHE_ONLY_ORIGINAL_ANONYMOUS_FIXED_DELIVERY')
        relative=PurePosixPath(path)
        if relative.is_absolute() or '..' in relative.parts:raise PermissionError('SOURCE_CACHE_DOWNLOAD_PATH')
        row=index[path];budget.reserve(row['bytes']);guard(row['bytes'])
        source=old/'download'/path;target=Path(local_dir)/path
        if (source.is_file() and not source.is_symlink()
                and source.resolve().is_relative_to((old/'download').resolve())
                and source.stat().st_uid==os.getuid() and source.stat().st_size==row['bytes']
                and sha(source)==row['sha256']):
            target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target);target.chmod(0o400)
            reused.append(path);return target
        downloaded.append(path)
        return get(repo,path,revision=revision,local_dir=local_dir,token=False)
    return fetch,reused,downloaded


def extract(path,target,expected,guard,*,old_target,budget,incremental_accounting=None):
    target,old_target=Path(target),Path(old_target);seen=set();reused=0;written=0
    target.mkdir(parents=True,exist_ok=True)
    with tarfile.open(path,'r|gz') as archive:
        for member in archive:
            name=PurePosixPath(member.name)
            if (not member.isfile() or name.is_absolute() or '..' in name.parts
                    or member.name in seen or member.name not in expected):
                raise PermissionError('SOURCE_CACHE_UNEXPECTED_ARCHIVE_MEMBER')
            output=target/member.name;old=old_target/member.name
            if (not output.resolve().is_relative_to(target.resolve()) or output.exists()
                    or old.is_symlink()):
                raise PermissionError('SOURCE_CACHE_NO_OUTPUT_REPLAY_OR_PATH_ESCAPE')
            budget.reserve(member.size);guard(member.size);output.parent.mkdir(parents=True,exist_ok=True)
            if (old.is_file() and old.resolve().is_relative_to(old_target.resolve())
                    and old.stat().st_uid==os.getuid() and old.stat().st_size==member.size
                    and sha(old)==expected[member.name]):
                os.link(old,output);output.chmod(0o400);reused+=1
            else:
                with output.open('xb') as stream:
                    shutil.copyfileobj(archive.extractfile(member),stream,1<<20)
                if sha(output)!=expected[member.name]:raise ValueError('SOURCE_CACHE_NEW_MEMBER_HASH_MISMATCH')
                output.chmod(0o400);written+=1
                if incremental_accounting is not None:
                    stat=output.stat()
                    incremental_accounting['new_regular_file_allocated_bytes']+=max(stat.st_size,stat.st_blocks*512)
                    incremental_accounting['new_regular_files']+=1
            seen.add(member.name)
    if seen!=set(expected):raise ValueError('SOURCE_CACHE_INCOMPLETE_ARCHIVE')
    return dict(members=len(seen),reused_hash_verified_members=reused,new_members=written)
