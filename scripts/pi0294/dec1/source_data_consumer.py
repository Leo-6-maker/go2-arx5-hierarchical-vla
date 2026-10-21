"""CPU-only fixed PV1 TRAIN-bank staging for the admitted alternate site.

This future cold scope does not change the H200-only consumer or the active
A40 probe. Source training remains separately allocation/recipe qualified.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import re
import shutil
import signal
import subprocess
import time

from scripts.pi0291_arx5_data.native_dataset import cpu_guard
from scripts.pi0292.data_consumer import extract_checked
from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0294.fd1 import sha
from scripts.pi0294.pv1.data_delivery import bank_members, download_dataset, official_consumption
from .authority import AUTHORITY, PARENT_AUTHORITY, R0, BANK, HARD_CLOSE
from scripts.operations.read_observation import observe

VAP1='43f7fe6f86c683f2e38e5c8802eb8cde208e575c'
SCHEMA='DEC1_VAP1_SOURCE_DATA_CPU_CHILD_V1'
DELIVERY='2c4e587c95beb5c2be3c592790d535015182a098ce06bd1ca7762edc3db4d7ee'
SOURCE_READY='e896959171cd66394b2eba60c081d035bd2688be2f874a887077ed8b5c4a5cbc'
ADOPTION='a26aeb5076e075791e1dfea4e5da0ccaae05ccbb7fa578a1583dea1c4764ed89'
REVISION='e4f603a1b768018077a8b5c9640250be67500902'
PREFIX='PI0294/20261018T182956Z-localpc-pv1/PV1_NEW_TRAIN_BANK_V1'
CPU=4500
TRANSFER=8*1024**3
OUTPUT=24*1024**3
PRIORITY='plans/CLOSED_HISTORICAL_PROBE_PRIORITY.json'


def bulk_priority(root=None,*, query=subprocess.check_output):
    if root is not None and PRIORITY in read(Path(root)/'RUN_MANIFEST.json')['sealed_inputs']:
        r=Path(root);m=read(r/'RUN_MANIFEST.json');v=sealed(r,PRIORITY,m['sealed_inputs'][PRIORITY])
        old=Path(v['root']);b=read(old/'OUTPUT_BINDING.private.json');lease=read(b['lease'])
        terminal=v['native_proof']
        if (v.get('schema')!='CLOSED_HISTORICAL_A40_PRIORITY_ONLY_V1'
                or v.get('not_model_qualification') is not True
                or old.is_symlink() or old.stat().st_uid!=os.getuid()
                or not old.resolve().is_relative_to(Path(m['registered_runs_root']).resolve())
                or sha(old/'RUN_MANIFEST.json')!=v['manifest_sha256']
                or sha(b['lease'])!=v['lease_sha256'] or lease.get('status')!='CLOSED'
                or lease.get('run_uuid')!=old.name
                or lease.get('owner') not in ('EXISTING_LOCAL_PC','EXISTING_SINGLE_LOCAL_PC')
                or terminal.get('job_id')!='1471903.stdct-mgmt-02' or terminal.get('job_state')!='F'
                or terminal.get('project')!='CFP04-CF-007' or terminal.get('owner')!='e1192548'
                or terminal.get('recorded_process_terminal') is not True):
            raise PermissionError('SOURCE_DATA_EXACT_CLOSED_HISTORICAL_PRIORITY_BINDING')
        # Current complete native selection still protects a live old probe.
        # Its terminal receipt does not expire with PBS history retention.
        ids=query(['/opt/pbs/bin/qselect','-u','e1192548'],text=True,timeout=25).split()
        if len(ids)!=len(set(ids)) or any(re.fullmatch(r'[0-9]+(?:\[[0-9]*\])?\.stdct-mgmt-[0-9]+',x) is None for x in ids):
            raise PermissionError('SOURCE_DATA_CURRENT_NATIVE_SELECTION_NOT_A_TABLE_OR_BANNER')
        if '1471903.stdct-mgmt-02' not in ids:return True
    raw=observe(lambda:query(['/opt/pbs/bin/qstat','-x','-f','-F','json','1471903.stdct-mgmt-02'],
        text=True,timeout=25))
    job=json.loads(raw)['Jobs']['1471903.stdct-mgmt-02']
    if (job.get('project')!='CFP04-CF-007' or job.get('Job_Owner','').split('@')[0]!='e1192548'
            or job.get('job_state') not in ('Q','R','F')):
        raise PermissionError('SOURCE_DATA_CURRENT_PROBE_PRIORITY_UNKNOWN_NO_BULK')
    return job['job_state']!='R'


def sealed(root, name, expected=None, *, parse=True):
    root=Path(root);p=root/name
    if (p.is_symlink() or not p.resolve().is_relative_to(root.resolve())
            or not p.is_file() or p.stat().st_uid!=os.getuid() or p.stat().st_mode&0o222
            or expected is not None and sha(p)!=expected):
        raise PermissionError('SOURCE_DATA_OWN_SEALED_INPUT')
    return read(p) if parse else p


def validate(root, *, now=time.time):
    r=Path(root);m=sealed(r,'RUN_MANIFEST.json');b=sealed(r,'OUTPUT_BINDING.private.json')
    profile=m.get('source_data_profile','PV1_139')
    if profile not in ('PV1_139','ORIGINAL126_C1'):raise PermissionError('SOURCE_DATA_EXACT_TWO_SAVED_TRAIN_PROFILES')
    if (r.is_symlink() or r.stat().st_uid!=os.getuid()
            or m.get('schema')!=SCHEMA or m.get('run_uuid')!=r.name
            or m.get('authority_release')!=AUTHORITY or m.get('amendment')!=VAP1
            or m.get('parent_authority_release')!=PARENT_AUTHORITY
            or m.get('owner')!='EXISTING_LOCAL_PC' or m.get('owner_uid')!=os.getuid()
            or m.get('campaign_run_uuid')!='20261018T182956Z-localpc-pv1'
            or m.get('original_T0_epoch')!=1792348196
            or not now()<m['deadline_epoch']<=min(HARD_CLOSE,1792866596)
            or m.get('CPU_upper_s')!=CPU or m.get('transfer_upper_bytes')!=TRANSFER
            or m.get('output_upper_bytes')!=OUTPUT
            or m.get('zero_effects')!=dict(GPU=0,PBS=0,models=0,queries=0,updates=0,physics=0)
            or m.get('execution_profile')!='vanda-neural-probe'
            or not r.resolve().is_relative_to(Path(m['registered_runs_root']).resolve())
            or b.get('uid')!=os.getuid() or b.get('host')!=os.uname().nodename
            or b.get('run_root')!=str(r) or b.get('run_manifest_sha256')!=sha(r/'RUN_MANIFEST.json')
            or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('PBS_JOBID')
            or (r/'STOP.json').exists()):
        raise PermissionError('SOURCE_DATA_STANDARD_CPU_AUTHORITY_HOST_AND_FINITE_BOUNDS')
    lease=read(b['lease']);matches=[x for x in read(b['registry'])['entries'] if x.get('role')==b['role']]
    if (lease.get('status')!='ACTIVE' or lease.get('owner')!='EXISTING_LOCAL_PC'
            or lease.get('authority_release')!=AUTHORITY or lease.get('amendment')!=VAP1
            or lease.get('run_uuid')!=r.name or lease.get('deadline_epoch')!=m['deadline_epoch']
            or len(matches)!=1 or Path(matches[0]['path']).resolve()!=r.resolve()):
        raise PermissionError('SOURCE_DATA_UNIQUE_NATIVE_REGISTRY_AND_ACTIVE_LEASE')
    for name,digest in m['sealed_inputs'].items():sealed(r,name,digest,parse=False)
    required={'BINDINGS.private.json','CPU_BUDGET_PREPAID.json','PUBLICATION_AND_CI.json',
        'plans/VAP1_ADOPTION_AND_DELTA_BUDGET.json',
        'capsule/code/scripts/pi0294/dec1/source_data_consumer.py','SOURCE_SEAL.json'}
    if profile=='PV1_139':required.add('plans/PV1_HF_BANK_DELIVERY.json')
    else:
        from .source_data_original import INPUTS
        required.update(INPUTS)
        required.add('capsule/code/scripts/pi0294/dec1/source_data_original.py')
    if not required<=set(m['sealed_inputs']):raise PermissionError('SOURCE_DATA_EXPLICIT_PREPAID_INPUTS')
    if 'plans/CLOSED_STAGING_CACHE_REUSE.json' in m['sealed_inputs']:
        if profile!='PV1_139':raise PermissionError('SOURCE_DATA_CACHE_ONLY_FIXED_139_BANK')
        from .source_data_cache import previous
        previous(r)
    if 'plans/CLOSED_ORIGINAL126_STAGING_CACHE_REUSE.json' in m['sealed_inputs']:
        if profile!='ORIGINAL126_C1':raise PermissionError('SOURCE_DATA_ORIGINAL_CACHE_ONLY_FIXED_ORIGINAL126_C1')
        sealed(r,'capsule/code/scripts/pi0294/dec1/source_original_cache.py',
            m['sealed_inputs'].get('capsule/code/scripts/pi0294/dec1/source_original_cache.py','MISSING'),parse=False)
        from .source_original_cache import previous
        previous(r)
    sealed(r,'plans/VAP1_ADOPTION_AND_DELTA_BUDGET.json',ADOPTION)
    proof=read(r/'PUBLICATION_AND_CI.json');source=m['code']['vla_commit']
    if (proof.get('source_sha')!=source or proof.get('CI_head_sha')!=source
            or proof.get('CI_status')!='completed' or proof.get('CI_conclusion')!='success'
            or proof.get('files_exact_readback') is not True
            or Path(__file__).resolve()!=(r/'capsule/code/scripts/pi0294/dec1/source_data_consumer.py').resolve()):
        raise PermissionError('SOURCE_DATA_EXACT_COLD_PUBLISHED_CAPSULE')
    paid=read(r/'CPU_BUDGET_PREPAID.json')
    if (paid.get('authority_release')!=AUTHORITY or paid.get('amendment')!=VAP1
            or paid.get('original_T0_epoch')!=1792348196
            or paid.get('campaign_run_uuid')!=m['campaign_run_uuid']
            or paid.get('costs_inside_original_PV1_and_DEC1') is not True
            or paid.get('CPU_upper_s')!=CPU or paid.get('transfer_reserved_bytes')!=TRANSFER
            or paid.get('output_reserved_bytes')!=OUTPUT or paid.get('unknown_effects')!=[]
            or paid.get('source_data_profile','PV1_139')!=profile):
        raise PermissionError('SOURCE_DATA_NONADDITIVE_PREPAYMENT_REQUIRED')
    if profile=='ORIGINAL126_C1':
        for name,digest in INPUTS.items():sealed(r,name,digest)
        inputs=read(r/'BINDINGS.private.json');source=inputs['official_source']
        actual=subprocess.check_output(['git','-C',source,'rev-parse','HEAD'],text=True,timeout=20).strip()
        clean=subprocess.run(['git','-C',source,'diff','--quiet','HEAD','--'],timeout=20).returncode==0
        if actual!='51d4c89f72fda44cbf77285c6a8114b52676b8a1' or not clean:
            raise PermissionError('SOURCE_DATA_EXACT_CLEAN_OFFICIAL_PROCESSOR_SOURCE')
        return m,inputs,None
    receipt=sealed(r,'plans/PV1_HF_BANK_DELIVERY.json',DELIVERY)
    if (receipt.get('status')!='PASS_FIXED_REVISION_READY_LAST_EXACT_FULL_SET_SIZE_SHA256'
            or receipt.get('repo_id')!='leoyuer/pi0294-public-artifacts' or receipt.get('prefix')!=PREFIX
            or receipt.get('revision')!=REVISION or receipt.get('dataset_manifest_sha256')!=BANK):
        raise PermissionError('SOURCE_DATA_EXACT_EXISTING_R0_139_TRAIN_DELIVERY')
    return m,read(r/'BINDINGS.private.json'),receipt


def consume(root, receipt, inputs, guard, *, get=download_dataset, cache=None):
    """Reuse mature full member extraction and official reader qualification."""
    r=Path(root);folder=r/'download';target=r/'datasets/FROZEN_TRAIN_BANK'
    folder.mkdir();target.parent.mkdir();prefix=receipt['prefix']+'/'
    files={x['path'][len(prefix):]:x for x in receipt['files'] if x['path'].startswith(prefix)}
    if (len(files)!=len(receipt['files']) or sum(x['bytes'] for x in files.values())>TRANSFER
            or any('/' in n or n in ('.','..') for n in files)):
        raise PermissionError('SOURCE_DATA_FIXED_FLAT_DELIVERY_TRANSFER_BOUND')
    def fetch(name):
        row=files[name];guard(row['bytes'])
        path=Path(get(receipt['repo_id'],row['path'],revision=receipt['revision'],local_dir=folder,token=False))
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError('SOURCE_DATA_FIXED_REVISION_ACTUAL_DOWNLOAD_HASH')
        return path
    ready=read(fetch('READY.json'));index_path=fetch('ARCHIVE_MEMBER_MANIFEST.json');index=read(index_path)
    if (ready['archive_index_sha256']!=sha(index_path) or ready['statistics_sha256']!=R0
            or ready['dataset_manifest_sha256']!=BANK or index['dataset_manifest_sha256']!=BANK
            or index['source_ready_sha256']!=SOURCE_READY
            or set(files)!=set(index['archives'])|{'READY.json','ARCHIVE_MEMBER_MANIFEST.json'}):
        raise PermissionError('SOURCE_DATA_EXACT_READY_INDEX_AND_EXISTING_SOURCE_BANK')
    seen=set()
    for name,proof in index['archives'].items():
        if set(proof['member_sha256'])&seen:raise PermissionError('SOURCE_DATA_DISJOINT_ARCHIVES_REQUIRED')
        p=fetch(name)
        if sha(p)!=proof['sha256'] or p.stat().st_size!=proof['bytes']:
            raise ValueError('SOURCE_DATA_ARCHIVE_PROOF_CHANGED')
        if cache is None:extract_checked(p,target,proof['member_sha256'],guard)
        else:
            from .source_data_cache import extract
            progress=extract(p,target,proof['member_sha256'],guard,old_target=cache['old_target'],budget=cache['budget'])
            record=r/'archive_progress'/name.replace('.tar.gz','.json');record.parent.mkdir(exist_ok=True)
            write(record,dict(archive=name,archive_sha256=proof['sha256'],**progress,epoch=time.time()))
        seen.update(proof['member_sha256'])
    manifest,source_ready,members=bank_members(target,lambda:guard())
    if (seen!=set(members) or sha(target/'DATASET_MANIFEST.json')!=BANK
            or sha(target/'READY.json')!=SOURCE_READY or len(manifest['episodes'])!=139):
        raise PermissionError('SOURCE_DATA_FULL_BANK_AND_ALL_139_TRAIN_PARENTS')
    binding=r/'RAW_BINDINGS.private.json'
    write(binding,{row['id']:str(target/'raw'/row['id']) for row in manifest['episodes']})
    result=official_consumption(r,dict(job_id='CPU_ONLY_NOT_A_PBS_JOB'),lambda:guard(),
        dict(source=inputs['official_source'],cosmos=inputs['cosmos']),dict(new_raw_bindings=str(binding)),
        r/'plans/PV1_HF_BANK_DELIVERY.json',target,receipt,manifest,members)
    result['actual_Vanda_CPU_official_processor_and_collator_batches']=result.pop('actual_Hopper_official_processor_and_collator_batches')
    result.update(status='PASS_VANDA_CPU_FIXED_PV1_139_TRAIN_FULL_MEMBER_AND_READER',
        new_download_bytes=sum(x['bytes'] for x in files.values()),GPU=0,PBS=0,models=0,
        new_optimizer_updates=0,new_physics=0,training_qualification=False,statistics_sha256=R0)
    return result


def run(root):
    r=Path(root);m,inputs,receipt=validate(r);lock=OwnerLock(r/'CPU_CONSUMER.lock')
    if (r/'RESULT.private.json').exists() or (r/'CONSUMPTION_INTENT.private.json').exists():
        lock.close();raise FileExistsError('SOURCE_DATA_PRESERVE_PRIOR_ATTEMPT_NO_BLIND_RESTART')
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_CPU,(CPU,CPU))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));began=time.process_time();failure=None;result={};last_size=0.;resident=0;last_priority=0.
    def expired(*unused):raise TimeoutError('SOURCE_DATA_FINITE_WALL_CUTOFF')
    previous=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,m['deadline_epoch']-time.time())
    cache_mode='plans/CLOSED_STAGING_CACHE_REUSE.json' in m['sealed_inputs']
    original_cache_mode='plans/CLOSED_ORIGINAL126_STAGING_CACHE_REUSE.json' in m['sealed_inputs']
    budget=None
    def guard(extra=0):
        nonlocal last_size,resident,last_priority
        if time.monotonic()-last_priority>15:
            while not bulk_priority(r):
                if time.time()+20>=m['deadline_epoch']:raise TimeoutError('SOURCE_DATA_PRIORITY_WAIT_FINITE_CUTOFF')
                time.sleep(15)
            last_priority=time.monotonic()
        if cache_mode or original_cache_mode:
            resident=budget.reserved if budget is not None else 0
        elif time.monotonic()-last_size>15:
            resident=sum(p.stat().st_size for p in r.rglob('*') if p.is_file() and not p.is_symlink());last_size=time.monotonic()
        if (time.time()>=m['deadline_epoch'] or time.process_time()-began>=CPU-60
                or resident+extra>OUTPUT or shutil.disk_usage(r).free<extra+40*1024**3 or (r/'STOP.json').exists()):
            raise PermissionError('SOURCE_DATA_CPU_STORAGE_CLOCK_OR_STOP_BOUND')
    try:
        cpu_guard();guard()
        write(r/'CONSUMPTION_INTENT.private.json',dict(epoch=time.time(),CPU_upper_s=CPU,
            revision=REVISION,model_loads=0,PBS=0,physics=0,no_blind_restart=True))
        if cache_mode:
            from .source_data_cache import previous as prior,OutputBudget,cached_download
            old=prior(r);budget=OutputBudget(OUTPUT)
            getter,reused,downloaded=cached_download(r,old,receipt,download_dataset,budget,guard)
            result=consume(r,receipt,inputs,guard,get=getter,cache=dict(old_target=old/'datasets/FROZEN_TRAIN_BANK',budget=budget))
            index={v['path']:v for v in receipt['files']}
            result.update(reused_closed_cache_files=reused,new_download_files=downloaded,
                new_download_bytes=sum(index[n]['bytes'] for n in downloaded),
                output_logical_upper_bytes=budget.reserved,old_failure_preserved=True)
        elif m.get('source_data_profile','PV1_139')=='PV1_139':result=consume(r,receipt,inputs,guard)
        else:
            from .source_data_original import consume as original
            if original_cache_mode:
                from .source_original_cache import previous as prior,Cache
                cache=Cache(r,prior(r),guard,upper=OUTPUT);budget=cache.budget
                result=original(r,receipt,inputs,guard,cache=cache)
                allocated_upper=cache.incremental['new_regular_file_allocated_bytes']+1024**3
                if allocated_upper>OUTPUT or sum(row['members'] for row in cache.archives)>500000:
                    raise PermissionError('SOURCE_DATA_INCREMENTAL_METADATA_AND_OUTPUT_UPPER')
                result.update(output_logical_upper_bytes=budget.reserved,
                    reused_closed_downloads=cache.reused_downloads,new_downloads=cache.new_downloads,
                    cache_archives=cache.archives,old_failure_preserved=True,
                    incremental_cache_allocation=dict(cache.incremental,
                        prior_payload_still_charged=True,reused_hardlink_payload_is_not_new_bytes=True,
                        directory_link_HF_metadata_and_control_upper_bytes=1024**3,
                        new_output_allocated_upper_bytes=allocated_upper,
                        is_independent_backup=False))
            else:result=original(r,receipt,inputs,guard)
    except BaseException as e:failure=dict(error_type=type(e).__name__,error=str(e)[-800:]);raise
    finally:
        result.update(CPU_s=time.process_time()-began,finished_epoch=time.time(),failure=failure)
        if failure:result['status']='FAILED_PRESERVED_NO_MODEL_OR_HF_MUTATION'
        write(r/'RESULT.private.json',result);signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous);lock.close()
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    value=run(parser.parse_args().root);print(value['status'],flush=True)
