"""CPU-only fixed PV1 TRAIN-bank staging for the admitted alternate site.

This future cold scope does not change the H200-only consumer or the active
A40 probe. Source training remains separately allocation/recipe qualified.
"""
import argparse
import json
import os
from pathlib import Path
import resource
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


def bulk_priority(*, query=subprocess.check_output):
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
        'plans/PV1_HF_BANK_DELIVERY.json','plans/VAP1_ADOPTION_AND_DELTA_BUDGET.json',
        'capsule/code/scripts/pi0294/dec1/source_data_consumer.py','SOURCE_SEAL.json'}
    if not required<=set(m['sealed_inputs']):raise PermissionError('SOURCE_DATA_EXPLICIT_PREPAID_INPUTS')
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
            or paid.get('output_reserved_bytes')!=OUTPUT or paid.get('unknown_effects')!=[]):
        raise PermissionError('SOURCE_DATA_NONADDITIVE_PREPAYMENT_REQUIRED')
    receipt=sealed(r,'plans/PV1_HF_BANK_DELIVERY.json',DELIVERY)
    if (receipt.get('status')!='PASS_FIXED_REVISION_READY_LAST_EXACT_FULL_SET_SIZE_SHA256'
            or receipt.get('repo_id')!='leoyuer/pi0294-public-artifacts' or receipt.get('prefix')!=PREFIX
            or receipt.get('revision')!=REVISION or receipt.get('dataset_manifest_sha256')!=BANK):
        raise PermissionError('SOURCE_DATA_EXACT_EXISTING_R0_139_TRAIN_DELIVERY')
    return m,read(r/'BINDINGS.private.json'),receipt


def consume(root, receipt, inputs, guard, *, get=download_dataset):
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
        extract_checked(p,target,proof['member_sha256'],guard);seen.update(proof['member_sha256'])
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
    def guard(extra=0):
        nonlocal last_size,resident,last_priority
        if time.monotonic()-last_priority>15:
            while not bulk_priority():
                if time.time()+20>=m['deadline_epoch']:raise TimeoutError('SOURCE_DATA_PRIORITY_WAIT_FINITE_CUTOFF')
                time.sleep(15)
            last_priority=time.monotonic()
        if time.monotonic()-last_size>15:
            resident=sum(p.stat().st_size for p in r.rglob('*') if p.is_file() and not p.is_symlink());last_size=time.monotonic()
        if (time.time()>=m['deadline_epoch'] or time.process_time()-began>=CPU-60
                or resident+extra>OUTPUT or shutil.disk_usage(r).free<extra+40*1024**3 or (r/'STOP.json').exists()):
            raise PermissionError('SOURCE_DATA_CPU_STORAGE_CLOCK_OR_STOP_BOUND')
    try:
        cpu_guard();guard()
        write(r/'CONSUMPTION_INTENT.private.json',dict(epoch=time.time(),CPU_upper_s=CPU,
            revision=REVISION,model_loads=0,PBS=0,physics=0,no_blind_restart=True))
        result=consume(r,receipt,inputs,guard)
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
