"""Finite V6 source depot. Reuse first12; never inspect TEST labels here."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from scripts.sla.robodojo_source_audit_v1 import read, write, sha, download, profile

PLAN='ae3d3c0dceade437b0104a150cd0402fef650052'
REVISION='1127bda94f172d757386945af3eb3a6754d1a4e6'


def schema_only(path):
    import h5py
    result={}
    with h5py.File(path,'r')as f:
        def visit(name,item):
            if isinstance(item,h5py.Dataset):
                if item.external or item.is_virtual:
                    raise ValueError('SELF_CONTAINED_SOURCE_BYTES_REQUIRED')
                result[name]=dict(shape=list(item.shape),dtype=str(item.dtype))
        f.visititems(visit)
        frequency=float(f['additional_info/frequency'][()])
    return dict(schema=result,frequency=frequency,source_TEST_labels_read=False,
        source_TEST_relationship_or_error_measured=False,model_outputs_read=False)


def bind_runtime(spec,spec_path):
    runtime=spec['trusted_runtime']
    if sha(runtime['path'])!=runtime['sha256']:raise ValueError('TRUSTED_RUNTIME_SPEC_CHANGED')
    old=read(runtime['path']);base=old['trusted_source_parent']
    if sha(base['spec_path'])!=base['spec_sha256']:raise ValueError('TRUSTED_BASE_SPEC_CHANGED')
    roots=[Path(spec['source_root']),Path(old['source_root']),Path(base['source_root'])]
    for name,pin in spec['source_files'].items():
        if sha(roots[0]/name)!=pin:raise ValueError('NEW_V6_SOURCE_CHANGED')
    sys.path[:0]=[str(r/p)for r in roots for p in('','src')]
    import go2arx5_vla.world_model,scripts.wm02
    for module,suffix in((go2arx5_vla.world_model,'src/go2arx5_vla/world_model'),(scripts.wm02,'scripts/wm02')):
        module.__path__=[str(r/suffix)for r in roots]+list(module.__path__)
    from scripts.wm02.mobile_boundary_runner_v1 import registered
    registered(spec,spec_path)


def main(spec_path):
    spec=read(spec_path);root=Path(spec['registered_root'])
    if(spec['plan_commit']!=PLAN or spec['phase']!='V6_SOURCE_COMPLETE_FIXED60' or
       spec['new_download_cap']!=48 or spec['episode_cap']!=60 or
       spec['new_model_generations_budget']or spec['optimizer_updates']or
       spec['new_physical_requests_budget']or os.environ.get('CUDA_VISIBLE_DEVICES')!=''):
        raise PermissionError('CPU_SOURCE_DEPOT_ONLY_FIXED_V6')
    bind_runtime(spec,spec_path)
    selection=spec['selection']
    if sha(selection['path'])!=selection['sha256']:raise ValueError('ORIGINAL_SELECTED60_CHANGED')
    population=read(selection['path'])
    if population['revision']!=REVISION or len(population['candidates'])!=60:
        raise ValueError('FIXED_ORIGINAL60_REQUIRED')
    previous=spec['previous_first12']
    if sha(previous['result']['path'])!=previous['result']['sha256']:raise ValueError('FIRST12_RESULT_CHANGED')
    prior=read(previous['result']['path'])
    if prior['profiled_episodes']!=12 or prior['exit_code']:raise ValueError('VALID_EXISTING12_REQUIRED')
    records={}
    for entry in prior['results']:
        if sha(entry['path'])!=entry['sha256']:raise ValueError('FIRST12_PROFILE_CHANGED')
        r=read(entry['path']);p=Path(r['transfer']['path'])
        if sha(p)!=r['source']['sha256']:raise ValueError('REUSED_RAW12_BYTES_CHANGED_NO_REDOWNLOAD')
        records[r['source']['path']]=dict(source=r['source'],raw_path=str(p),raw_sha256=r['source']['sha256'],
            bytes=p.stat().st_size,profile=r['profile'],reused_first12=True)
    first=set(population['first_paths'])
    if set(records)!=first:raise ValueError('EXACT_FIRST12_REUSE_REQUIRED')
    rows=[r for r in population['candidates']if r['path']not in first]
    if len(rows)!=48 or sum(r['size']for r in population['candidates'])>32*1024**3:
        raise PermissionError('TOTAL60_RAW32G_CAP')
    if(root/'SUPERVISOR.private.json').exists():raise FileExistsError('NEW_ONCE_ONLY_DEPOT_REQUIRED')
    write(root/'SUPERVISOR.private.json',dict(pid=os.getpid(),epoch=time.time(),source_sha=spec['source_sha']))
    from scripts.pi0294.e1.journal import Journal
    journal=Journal(root/'control/journal',root.name);start=time.monotonic();cpu=time.process_time();failure=None;code=0;new_bytes=0
    try:
        for i,row in enumerate(rows):
            if time.time()>=spec['deadline_epoch']-60 or(root/'STOP.json').exists():raise TimeoutError('FINITE_DEPOT_STOP_OR_DEADLINE')
            write(root/'PROGRESS.private.json',dict(epoch=time.time(),new_downloads_completed=i,reused=12,
                current=row['path'],proposed_split=row['proposed_split'],model_generations=0,native_requests=0))
            transfer=journal.once('DOWNLOAD-%02d'%i,row,lambda:download(row,root,population['repository'],REVISION,until=spec['deadline_epoch']-60))
            new_bytes+=transfer['bytes']
            detail=schema_only(transfer['path'])if row['proposed_split']=='SOURCE_TEST'else profile(transfer['path'])
            if detail['frequency']!=30.:raise ValueError('DECLARED_NATIVE30HZ_CONTRACT_MISMATCH')
            record=dict(source=row,raw_path=transfer['path'],raw_sha256=transfer['sha256'],bytes=transfer['bytes'],
                profile=detail,reused_first12=False)
            write(root/('FILE-%02d.private.json'%i),record);records[row['path']]=record
            write(root/'SOURCE_MAP.private.json',dict(repository=population['repository'],revision=REVISION,
                records=[records[r['path']]for r in population['candidates']if r['path']in records],
                TEST_labels_read=False,split='ORIGINAL_EPISODE_HOLDOUT_ONLY_NO_SESSION_INDEPENDENCE_CLAIM'))
        if len(records)!=60:raise ValueError('FULL60_INVENTORY_REQUIRED')
    except BaseException as error:
        code=1;failure=dict(type=type(error).__name__,reason=str(error).split('?')[0][:1200])
    result=dict(schema='SLA_V6_SOURCE_DEPOT_ACTUAL',exit_code=code,status='FIXED60_READY_FOR_TYPED_SOURCE_READER'if code==0 else'PRESERVED_FINITE_TECHNICAL_STOP',
        plan_commit=PLAN,source_sha=spec['source_sha'],revision=REVISION,reused_first12=12,new_downloads_completed=len(records)-12,
        new_verified_raw_bytes=new_bytes,all_verified_raw_bytes=sum(r['bytes']for r in records.values()),failure=failure,
        source_map=dict(path=str(root/'SOURCE_MAP.private.json'),sha256=sha(root/'SOURCE_MAP.private.json'))if(root/'SOURCE_MAP.private.json').exists()else None,
        TEST_labels_read=False,model_generations=0,optimizer_updates=0,optimization_visits=0,native_requests=0,
        wall_s=time.monotonic()-start,CPU_s=time.process_time()-cpu)
    write(root/'RESULT.private.json',result)
    from go2arx5_vla.world_model.mfree_unattended_runtime_v1 import close
    close(spec,code);return code


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);raise SystemExit(main(p.parse_args().spec))
