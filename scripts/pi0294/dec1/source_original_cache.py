"""Opt-in original TRAIN/C1 staging cache, not a model or dataset change."""
from pathlib import Path
import os

from scripts.pi0292.durable import read
from scripts.pi0294.fd1 import sha
from .source_data_cache import OutputBudget, extract

PLAN='plans/CLOSED_ORIGINAL126_STAGING_CACHE_REUSE.json'
REASON='ENGINEERING_NFS_GUARD_FULL_TREE_RESCAN'
HISTORY_REASON='ENGINEERING_RETIRED_PBS_HISTORY_PRIORITY_GUARD'


def reusable_failure(result,reason):
    failure=result.get('failure',{})
    if reason==REASON:return failure.get('error')=='SOURCE_DATA_CPU_STORAGE_CLOCK_OR_STOP_BOUND'
    if reason==HISTORY_REASON:
        return (failure.get('error_type')=='CalledProcessError'
            and failure.get('error')=="Command '['/opt/pbs/bin/qstat', '-x', '-f', '-F', 'json', '1471903.stdct-mgmt-02']' returned non-zero exit status 1.")
    return False


def previous(root):
    root=Path(root);m=read(root/'RUN_MANIFEST.json');p=root/PLAN;plan=read(p)
    if (p.is_symlink() or p.stat().st_mode&0o222 or m['sealed_inputs'].get(PLAN)!=sha(p)
            or m.get('source_data_profile')!='ORIGINAL126_C1'
            or plan.get('schema')!='CLOSED_ORIGINAL126_C1_STAGING_CACHE_V1'):
        raise PermissionError('ORIGINAL_CACHE_EXACT_OPT_IN_PLAN')
    old=Path(plan['source_root'])
    if (old.is_symlink() or old==root or old.parent.resolve()!=root.parent.resolve()
            or old.stat().st_uid!=os.getuid()
            or sha(old/'RUN_MANIFEST.json')!=plan['source_manifest_sha256']
            or sha(old/'RESULT.private.json')!=plan['source_result_sha256']):
        raise PermissionError('ORIGINAL_CACHE_OWN_FIXED_CLOSED_SIBLING')
    b=read(old/'OUTPUT_BINDING.private.json');lease=read(b['lease']);om=read(old/'RUN_MANIFEST.json')
    result=read(old/'RESULT.private.json');stop_path=old/'STOP.json';stop=read(stop_path)
    process=read(old/'PROCESS.private.json');stat=Path('/proc')/str(process['pid'])/'stat'
    f=stat.read_text().rsplit(')',1)[1].split() if stat.exists() else None
    if (sha(b['lease'])!=plan['source_lease_sha256'] or lease.get('status')!='CLOSED_FAILED_CPU_STAGING_PRESERVED'
            or lease.get('run_uuid')!=old.name or lease.get('owner')!='EXISTING_LOCAL_PC'
            or lease.get('authority_release')!=m['authority_release']
            or f and f[0]!='Z' and f[19]==str(process['proc_start'])
            or om.get('source_data_profile')!='ORIGINAL126_C1'
            or om.get('authority_release')!=m['authority_release']
            or om.get('original_T0_epoch')!=m['original_T0_epoch']
            or om.get('zero_effects')!=dict(GPU=0,PBS=0,models=0,queries=0,updates=0,physics=0)
            or result.get('status')!='FAILED_PRESERVED_NO_MODEL_OR_HF_MUTATION'
            or not reusable_failure(result,stop.get('reason'))
            or stop_path.is_symlink() or stop_path.stat().st_mode&0o222
            or sha(stop_path)!=plan.get('source_stop_sha256')
            or stop.get('reason') not in (REASON,HISTORY_REASON) or stop.get('administrative_CPU_only') is not True
            or stop.get('new_GPU_or_PBS_or_physics')!=0):
        raise PermissionError('ORIGINAL_CACHE_NO_LIVE_UNKNOWN_OR_SCIENTIFIC_STOP')
    for name in ('plans/V3_TRANSFER.json','plans/C0_TRANSFER.json','plans/C1_TRANSFER.json'):
        if sha(old/name)!=sha(root/name):raise PermissionError('ORIGINAL_CACHE_FROZEN_DELIVERY_CHANGED')
    return old


class Cache:
    def __init__(self,root,old,guard,*,upper):
        self.root,self.old,self.guard=Path(root),Path(old),guard
        self.budget=OutputBudget(upper);self.reused_downloads=[];self.new_downloads=[];self.archives=[]
        self.incremental=dict(new_regular_file_allocated_bytes=0,new_regular_files=0)

    def download(self,client,name,row,get):
        relative=client.root.relative_to(self.root);member=Path(row['path'])
        if (member.is_absolute() or '..' in member.parts or not client.root.resolve().is_relative_to(self.root.resolve())):
            raise PermissionError('ORIGINAL_CACHE_OWN_FIXED_DOWNLOAD_PATH')
        source=self.old/relative/member
        target=client.root/row['path'];self.budget.reserve(row.get('bytes',1024**2))
        if target.exists() or target.is_symlink():raise FileExistsError('ORIGINAL_CACHE_NO_OUTPUT_REPLAY')
        if (source.is_file() and not source.is_symlink()
                and source.resolve().is_relative_to((self.old/relative).resolve())
                and source.stat().st_uid==os.getuid() and not source.stat().st_mode&0o222
                and sha(source)==row['sha256']
                and source.stat().st_size==row.get('bytes',source.stat().st_size)):
            target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target)
            self.reused_downloads.append(dict(relative=str(target.relative_to(self.root)),bytes=target.stat().st_size))
            return target,False
        target=Path(get(client.spec['repo_id'],row['path'],revision=client.spec['revision'],
            local_dir=client.root,token=None))
        stat=target.stat()
        self.incremental['new_regular_file_allocated_bytes']+=max(stat.st_size,stat.st_blocks*512)
        self.incremental['new_regular_files']+=1
        self.new_downloads.append(dict(relative=str(target.relative_to(self.root)),bytes=target.stat().st_size))
        return target,True

    def extract(self,path,target,expected):
        relative=Path(target).relative_to(self.root)
        result=extract(path,target,expected,self.guard,old_target=self.old/relative,budget=self.budget,
            incremental_accounting=self.incremental)
        self.archives.append(dict(target_relative=str(relative),archive_sha256=sha(path),**result))
        return result
