"""One predeclared source arm on its own legal A40 allocation.

The model, Trainer, matched dataset, fixed probes and full-state saves are the
existing implementations. This entry never serves actions or starts physics.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import time

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.pv1.pbs_owner import publish_release_record as once
from . import a40_allocation as native
from . import a40_training_context as context
from .a40_native_ledger import Ledger


def fail(message):raise PermissionError('VAP1_A40_RUNTIME_'+message)


def native_job(r):
    job_id=os.environ.get('PBS_JOBID');host=socket.gethostname()
    if not job_id or not host.split('.')[0].lower().startswith('gn-a40-'):
        fail('NO_MODEL_ON_LOGIN_OR_WITHOUT_OWN_A40_ALLOCATION')
    p=subprocess.run(['/opt/pbs/bin/qstat','-f','-F','json',job_id],
        capture_output=True,text=True,timeout=25,check=True)
    job=json.loads(p.stdout)['Jobs'][job_id]
    if (job.get('job_state')!='R' or job.get('Job_Owner','').split('@')[0]!=os.environ['USER']
            or job.get('Job_Name')!=r['job_name'] or job.get('project')!=r['project']
            or job.get('Rerunable') not in (False,'False') or int(job.get('run_count',1))!=1
            or job.get('Variable_List',{}).get('PV1_REQUEST_ID')!=r['request_id']
            or job.get('Variable_List',{}).get('PV1_SUBMISSION_NONCE')!=r['submission_nonce']
            or host.split('.')[0] not in {v.split('/')[0].split('.')[0] for v in job['exec_host'].split('+')}):
        fail('OWN_ACTUAL_RUNNING_JOB_REQUEST_AND_NODE_BEFORE_RELEASE')
    native.select_profile(job['Resource_List'],r)
    return job_id,host,job


def allocation(root,directory,r):
    root=Path(root);job_id,host,job=native_job(r);ready=directory/'PBS_READY.json'
    started=time.time();stat=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    once(ready,dict(run_uuid=root.name,job_id=job_id,hostname=host,pid=os.getpid(),proc_start=stat[19],
        observed_epoch=started,request_sha256=sha(directory/'REQUEST.json'),models_loaded=0,
        source_sha=r['source_sha'],amgr_login_in_compute=False))
    release=directory/'START_NEURAL.json';m=read(root/'RUN_MANIFEST.json');binding=read(root/'OUTPUT_BINDING.private.json')
    while not release.exists():
        if ((root/'STOP.json').exists() or read(binding['lease'])['status']!='ACTIVE'
                or time.time()>=native.T0+156*3600):fail('ORIGINAL_STOP_AND_LEASE_BEFORE_MODEL_RELEASE')
        if time.time()-started>=180:raise TimeoutError('A40_OWNER_FRESH_RELEASE_MISSING_NO_IDLE_MODEL')
        time.sleep(.2)
    v=context.sealed(root,str(release.relative_to(root)))
    material=context.sealed(root,v['material_relative'])
    if (v.get('job_id')!=job_id or v.get('request_sha256')!=sha(directory/'REQUEST.json')
            or v.get('submission_nonce')!=r['submission_nonce']
            or v.get('ready_sha256')!=sha(ready)
            or not started<=v['observed_epoch']<=time.time() or time.time()-started>=180
            or sha(root/v['material_relative'])!=v.get('material_sha256')):
        fail('SAME_JOB_FIRST_OWNER_RELEASE_WITH_ACTUAL_NATIVE_INPUTS')
    h=json.loads(base64.b64decode(material['ledger_base64'],validate=True))
    # Queue and project observations come from the owner, but this own R row is
    # independently queried on the actual compute node before any CUDA work.
    receipt=native.validate_native(now=time.time(),user=os.environ['USER'],hostname=host,job_id=job_id,
        job=job,request_bytes=(directory/'REQUEST.json').read_bytes(),ledger=h,
        project_proof=material['project_proof'],quota=material['quota'],queue_proof=context.sealed(root,v['queue_relative']),
        budget_proof=material['budget_proof'],grant_bytes=(root/'plans/VAP1_DUAL_PLATFORM_CONCURRENCY_ADOPTION.json').read_bytes(),
        campaign_view=material['campaign_view'])
    if m['code']['vla_commit']!=receipt['source_sha']:fail('ORIGINAL_REGISTERED_SOURCE_CAPSULE')
    once(directory/'ALLOCATION_ADMISSION.json',receipt)
    return receipt


def compatibility(root,instance,allocation,ledger,guard):
    import numpy as np
    from scripts.pi0294.e1.model import fixed_draw
    root=Path(root);folder=root/'A40_COMPATIBILITY';folder.mkdir(mode=0o700)
    saved=context.sealed(root,'plans/PROBE_OBSERVATION.private.json')
    if saved.get('provenance',{}).get('split')!='TRAIN':fail('ORIGINAL_SAVED_TRAIN_COMPATIBILITY_ONLY')
    records=[]
    for index in range(17):
        guard();target=folder/('query-'+str(index).zfill(2)+'.npz');seed=29400+index
        began=time.monotonic()
        physical,metadata=fixed_draw(root,'A40-COMPATIBILITY','draw-'+str(index),instance,
            saved['payload'],seed,target,ledger)
        with np.load(target,allow_pickle=False) as z:
            normalized=z['normalized']
            if normalized.shape!=(1,40,132) or not np.isfinite(normalized).all():fail('FULL_NORMALIZED_FINITE_H40')
        if physical.shape!=(40,10) or not np.isfinite(physical).all():fail('PHYSICAL_FINITE_EEF10_H40')
        target.chmod(0o400)
        records.append(dict(index=index,seed=seed,arrays_sha256=metadata['arrays_sha256'],
            finite_shapes_verified=True,wall_s=time.monotonic()-began,
            inference_precision=metadata['inference_precision']))
    result=dict(status='PASS_ACTUAL_SAME_JOB_17_SAVED_TRAIN_DRAWS_NOT_TASK_SUCCESS',
        job_id=allocation['job_id'],request_sha256=allocation['request_sha256'],
        parent_checkpoint_manifest_sha256=native.PARENT,statistics_sha256=native.R0,
        records=records,new_physics=0,new_tasks=0,new_optimizer_updates=0)
    once(folder/'RESULT.json',result);return result


def fixed_probes(root,arm,instance,ledger,guard):
    from scripts.pi0294.pv1.neural import train_dataset
    from scripts.pi0294.pv1.probes import run as probes
    root=Path(root);dataset=train_dataset(root);dataset.initialize()
    result=probes(root,arm+'-1024',instance,dataset,ledger,guard)
    folder=root/'evidence'/(arm+'-1024')/'TRAIN_PROBES'
    for path in folder.iterdir():
        if path.is_file() and not path.is_symlink() and path.name!='SESSION.lock':path.chmod(0o400)
    return result


def run(root,request_path):
    root,request_path=Path(root),Path(request_path)
    m,r=context.validate_root(root,request_path,verify_all=True)
    lock=OwnerLock(root/'A40_NATIVE_ALLOCATION_OWNER.lock');directory=request_path.parent
    began=time.process_time();allocated=None;ledger=None;instance=None;records=[];phase='BOOTSTRAP';error=None
    status='FAILED_PRESERVED';final_step=r['expected_native_start_step']
    try:
        if (directory/'PBS_PROCESS_FINAL.json').exists():raise FileExistsError('A40_NATIVE_PROCESS_NO_RESTART')
        allocated=allocation(root,directory,r)
        os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
        if len(os.sched_getaffinity(0))!=4:fail('FOUR_OWN_ACTIVE_PROCESS_CPUS')
        resource.setrlimit(resource.RLIMIT_CPU,(r['CPU_upper_s'],r['CPU_upper_s']))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        ledger=Ledger(root,allocated);ledger.initialize()
        guard=ledger.guard;guard()
        phase='ACTUAL_NATIVE_SOURCE_PREMODEL'
        from scripts.pi0294.numerics import prepare_environment,configure_torch
        prepare_environment(r['numerical_mode'])
        import torch
        if torch.cuda.device_count()!=1 or 'A40' not in torch.cuda.get_device_name(0):fail('ONE_REAL_VISIBLE_A40')
        configure_torch(torch,r['numerical_mode']);torch.set_num_threads(4)
        closed=context.verify_consumers(root)
        from scripts.pi0294.pv1.neural import stage_work
        stage_work(root,r)
        once(directory/context.QUALIFICATION,dict(status='PASS_A40_NATIVE_SOURCE_PREMODEL_NO_CORE_TRANSPORT_USED',
            job_id=allocated['job_id'],request_sha256=allocated['request_sha256'],source_sha=r['source_sha'],
            native_GPU='NVIDIA_A40',visible_GPU_count=1,closed_consumers=closed,models_loaded=0,queries=0,physics=0))
        from scripts.pi0294.e1 import model
        from scripts.pi0294.e1.neural import checkpoint_identity
        identity=context.sealed(root,'POLICY_IDENTITY.json')
        consumer=context.sealed(root,'ASSET_CONSUMPTION.json')
        if (identity.get('checkpoint_manifest_sha256')!=native.PARENT or identity.get('statistics_sha256')!=native.R0
                or identity.get('global_step')!=2048 or identity.get('files')!=consumer.get('inference_files_sha256')
                or consumer.get('status')!='PASS_FULL_CONSUMER_BYTES_AND_REUSED_FROZEN_SHARD_NO_MODEL'
                or consumer.get('original_checkpoint_manifest_sha256')!=native.PARENT):
            fail('ORIGINAL_CORR_DERIVED_INFERENCE_BYTES_NOT_ANOTHER_PARENT_OR_FULL_RESUME_CLAIM')
        if identity.get('relay_identity',{}).get('source_sha')!=r['source_sha']:
            fail('CURRENT_RUNTIME_SOURCE_IN_ORIGINAL_PARENT_RELAY_IDENTITY')
        inputs=read(root/'plans/PV1_TRAIN_INPUTS.private.json');parent=Path(inputs['CORR_parent_checkpoint'])
        def load(checkpoint,label,*,trainable,original=False):
            nonlocal instance
            guard();context.require_premodel(root,allocated)
            value=identity if original else checkpoint_identity(checkpoint,label,r['source_sha'])
            folder=root/'MODEL_LOAD_INTENTS';folder.mkdir(exist_ok=True)
            once(folder/(label+'.json'),dict(job_id=allocated['job_id'],request_sha256=allocated['request_sha256'],
                checkpoint_manifest_sha256=value['checkpoint_manifest_sha256']))
            ledger.charge('model_loads',1);instance=model.load(root,checkpoint,value,trainable=trainable)
        def unload():
            nonlocal instance
            if instance is not None:model.unload(instance);instance=None
        phase='CURRENT_CORR_17_SAVED_TRAIN_COMPATIBILITY'
        load(parent,'COMPATIBILITY',trainable=False,original=True)
        records.append(compatibility(root,instance,allocated,ledger,guard));unload()
        arm=r['training_block']['arm'];relative=r['training_block'].get('resume_checkpoint_relative')
        checkpoint=root/relative if relative else parent
        load(checkpoint,'TRAIN-INPUT',trainable=True,original=not relative)
        from scripts.pi0294.pv1.training import run as train
        from .long_training import execute
        def training(*,stop,phase,checkpoint):
            return train(root,instance,allocated,guard,ledger,arm=arm,request_id=r['request_id'],
                stop_after_step=stop,resume_checkpoint=checkpoint or (root/relative if relative else None),long_phase=phase)
        def probe():
            return fixed_probes(root,arm,instance,ledger,guard)
        phase='SOURCE_LONG_TRAIN_SAME_ALLOCATION'
        final,history=execute(start_step=r['expected_native_start_step'],train=training,probe=probe,unload=unload,
            load=lambda path,trainable:load(path,'PROBES' if not trainable else 'TRAIN-REENTRY',trainable=trainable),
            guard=guard,checkpoint=root/'training'/arm/'checkpoint-1024')
        records.extend(history);final_step=final['global_step']
        status='A40_SOURCE_4096_COMPLETE' if final_step==4096 else 'A40_KNOWN_PARTIAL_FULL_STATE_PRESERVED'
        return dict(status=status,global_step=final_step,records=records)
    except BaseException as e:error=dict(type=type(e).__name__,message=str(e)[-1000:]);raise
    finally:
        result=dict(status=status,phase=phase,error=error,job_id=os.environ.get('PBS_JOBID'),
            source_sha=r['source_sha'],request_sha256=sha(request_path),CPU_actual_s=time.process_time()-began,
            global_step=final_step,records=records,new_physics=0,new_episodes=0,selected='NONE',deployment='NONE')
        if ledger is not None:
            try:result['effects']=ledger.summary()
            except BaseException as e:result['effect_summary_error']=dict(type=type(e).__name__)
        if instance is not None and allocated is not None and time.time()<allocated['allocation_end_epoch']:
            try:
                from scripts.pi0294.e1.model import unload
                unload(instance)
            except BaseException as e:result['unload_error']=type(e).__name__
        once(directory/'PBS_PROCESS_FINAL.json',result);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--request',required=True)
    args=p.parse_args();run(args.root,args.request)
