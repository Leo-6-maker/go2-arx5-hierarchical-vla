"""Finite E1 PBS workload: fixed M0 probes, real M1, then fixed 256/1024 eval."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import time

import numpy as np

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0292.relay_protocol import PROTOCOL,digest,sealed,unseal,uuid_string,validate_identity,validate_observation,exact_keys
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.resident import QueryReservations
from scripts.pi0294.numerics import prepare_environment,configure_torch
from .allocation import admit
from .artifacts import atomic_npz
from .scope import Ledger,STATS
from .control_read import read


def checkpoint_identity(checkpoint,label,source_sha):
    complete=read(checkpoint/'CHECKPOINT_COMPLETE.json')
    files={n:s for n,s in complete['files'].items() if n.endswith('.safetensors') or n in ['config.json','model.safetensors.index.json','statistics.json'] or n.startswith('processor/')}
    canonical=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    processor=hashlib.sha256(json.dumps({k:v for k,v in files.items() if k.startswith('processor/')},sort_keys=True).encode()).hexdigest()
    identity=dict(files=files,label=label,global_step=complete['global_step'],statistics_sha256=STATS,
        checkpoint_manifest_sha256=sha(checkpoint/'CHECKPOINT_COMPLETE.json'),
        relay_identity=dict(profile='B1',model_revision=canonical[:40],source_sha=source_sha,
            config_sha256=files['config.json'],processor_sha256=processor,statistics_sha256=STATS))
    validate_identity(identity['relay_identity']);return identity


def serve(root,label,instance,allocation,ledger,guard,*,max_queries=320,ready_status='E1_NATIVE_MODEL_READY',noise_namespace='PI0294',terminal_status='E1_STAGE_TERMINAL_NOT_TASK_SUCCESS',native_pre_model_qualification=None,visibility_refresh=None,mapped_ack_contracts=None):
    if ready_status=='PV1_NATIVE_MODEL_READY':
        proof=native_pre_model_qualification
        if (not callable(visibility_refresh) or not isinstance(proof,dict) or proof.get('status')!='PASS_SAME_JOB_NATIVE_QUALIFICATION_READBACK'
                or not re.fullmatch('[0-9a-f]{64}',proof.get('receipt_sha256',''))
                or proof.get('heartbeat_elapsed_s',0)<600 or proof.get('transport_cycles')!=80
                or proof.get('new_model_queries')!=0 or proof.get('new_physics')!=0):
            raise PermissionError('PV1_MODEL_READY_NEEDS_SAME_JOB_NATIVE_PREMODEL_READBACK')
    elif native_pre_model_qualification is not None or visibility_refresh is not None:
        raise PermissionError('PV1_PROOF_CANNOT_BE_ATTACHED_TO_LEGACY_MODEL_READY')
    directory=root/'evidence'/label/'resident';directory.mkdir(parents=True,exist_ok=False)
    for n in ['inbox','outbox','receipts']:(directory/n).mkdir()
    state=dict(status=ready_status,label=label,run_uuid=root.name,job_id=os.environ['PBS_JOBID'],
        pid=os.getpid(),identity=instance.identity,allocation_end_epoch=allocation['allocation_end_epoch'],queries=0)
    if native_pre_model_qualification is not None:
        state['native_pre_model_qualification']=native_pre_model_qualification
        state['NFS_lookup_cache_assistance']='QUALIFIED_STATX_FORCE_SYNC_AND_DIRECTORY_ENUMERATION_V2'
    write(root/'pbs/MODEL_STATE.json',state);write(directory/'STATE.json',state)
    registry_path=root/'plans'/label/'REGISTRY.json';wait=time.time()+180
    while not registry_path.exists():
        guard()
        if visibility_refresh is not None:visibility_refresh(directory)
        done=directory/'STAGE_DONE.json'
        if done.exists():
            value=read(done)
            if value['run_uuid']!=root.name or value['job_id']!=os.environ['PBS_JOBID'] or value['label']!=label:
                raise PermissionError('E1_EXACT_DISABLED_STAGE_TERMINAL_REQUEST')
            state['status']='PHYSICS_NOT_RUN_OWNER_TERMINAL_BEFORE_REGISTRY';write(directory/'STATE.json',state);return state
        if time.time()>=wait:raise TimeoutError('E1_OWNER_REGISTRY_NOT_READY_NO_IDLE_GPU')
        time.sleep(.2)
    registry=read(registry_path)
    if registry['checkpoint_identity']!=instance.identity or registry['run_uuid']!=root.name:raise PermissionError('E1_EXACT_STAGE_MODEL_REGISTRY')
    # Only the already admitted cold caller supplies these owner-pinned objects.
    # Default legacy callers still construct the original strict RAW consumer.
    if mapped_ack_contracts is not None and set(mapped_ack_contracts)!=set(registry['cases']):
        raise PermissionError('COLD_MAPPED_ACK_CONTRACTS_MUST_COVER_EXACT_REGISTERED_CASES')
    reservation_options=dict(max_queries=max_queries)
    if mapped_ack_contracts is not None:reservation_options['mapped_ack_contracts']=mapped_ack_contracts
    reservations=QueryReservations(directory,root.name,**reservation_options)
    seen=set()
    def expire(signum,frame):raise TimeoutError('E1_FRESH_MODEL_DRAW_EXPIRED_NO_RESAMPLE')
    signal.signal(signal.SIGALRM,expire)
    while True:
        guard()
        if visibility_refresh is not None:
            visibility_refresh(directory);visibility_refresh(directory/'inbox')
        done=directory/'STAGE_DONE.json'
        if done.exists():
            value=read(done)
            if value['run_uuid']!=root.name or value['job_id']!=os.environ['PBS_JOBID'] or value['label']!=label:
                raise PermissionError('E1_EXACT_STAGE_TERMINAL_REQUEST')
            break
        for path in sorted((directory/'inbox').glob('*.json')):
            if path.name in seen:continue
            request=read(path);body=unseal(request)
            from scripts.pi0292.neural_relay import REQUEST_KEYS
            exact_keys(body,REQUEST_KEYS)
            for key in ['episode_uuid','query_uuid','reference_uuid']:uuid_string(body[key])
            case=registry['cases'].get(body['episode_uuid'])
            if (case is None or body['run_uuid']!=root.name or body['protocol']!=PROTOCOL or body['kind']!='student_request'
                    or path.stem!=body['query_uuid'] or body['identity']!=instance.identity['relay_identity']
                    or type(body['sequence']) is not int or not 0<=body['sequence']<80 or body['tick']!=body['sequence']*200
                    or not time.time()<body['expires']<=time.time()+60):raise PermissionError('E1_EXACT_FRESH_TRAIN_REQUEST')
            validate_observation(body['payload'])
            observation=sealed(dict(protocol=PROTOCOL,kind='observation',run_uuid=root.name,episode_uuid=body['episode_uuid'],sequence=body['sequence'],tick=body['tick'],payload=body['payload']))
            if body['observation_sha256']!=observation['sha256'] or body['anchor_sha256']!=digest(body['payload']['state_eef10']):raise ValueError('E1_OBSERVATION_HASH_CHANGED')
            if (body['sequence']==0)!=(body['previous_ack'] is None):raise ValueError('E1_PREVIOUS_ACK_REQUIRED_AFTER_PREFIX')
            reservations.begin(body,time.time());ledger.charge('generation_queries',1);seen.add(path.name)
            try:
                signal.alarm(max(1,math.ceil(body['expires']-time.time())))
                seed=int.from_bytes(hashlib.sha256((noise_namespace+'/'+case['layout_id']+'/'+str(body['sequence'])).encode()).digest()[:8],'big')%(2**63)
                value=instance.generate(body['payload'],seed)
                if time.time()>=body['expires']:raise TimeoutError('E1_DRAW_EXPIRED_NO_PHYSICS')
                physical=value['physical_H40'].tolist()
                action=sealed(dict(protocol=PROTOCOL,kind='action',run_uuid=root.name,episode_uuid=body['episode_uuid'],sequence=body['sequence'],tick=body['tick'],
                    payload={k:body[k] for k in ['query_uuid','reference_uuid','observation_sha256','anchor_sha256','identity']}|
                    dict(trajectory_eef10=physical,offsets_ns=[i*50000000 for i in range(40)])))
                command=sealed(dict(protocol=PROTOCOL,kind='install_prefix8',run_uuid=root.name,episode_uuid=body['episode_uuid'],sequence=body['sequence'],tick=body['tick'],
                    **{k:body[k] for k in ['query_uuid','reference_uuid','observation_sha256','anchor_sha256']},
                    prefix_eef10=physical[:8],prefix_sha256=digest(physical[:8]),advance_integrations=200,
                    frame=body['payload']['frame'],tcp=body['payload']['tcp'],expires=body['expires']))
                arrays=directory/'receipts'/(path.stem+'.npz');atomic_npz(arrays,normalized=value['normalized'],physical=value['physical_H40'])
                write(directory/'receipts'/path.name,dict(seed=seed,noise=value['noise'],inference_precision=value['inference_precision'],action_sha256=action['sha256'],arrays_sha256=sha(arrays),physics_steps=0,extra_correction=False))
                completion_options={} if mapped_ack_contracts is None else dict(action_packet=action,command_packet=command)
                reservations.complete(body,action['sha256'],prefix_sha256=digest(physical[:8]),command_sha256=command['sha256'],
                    **completion_options)
                write(directory/'outbox'/path.name,action);state['queries']+=1;write(directory/'STATE.json',state)
            except BaseException as exc:
                reservations.fail(body,type(exc).__name__+': '+str(exc),known_terminal=False);raise
            finally:signal.alarm(0)
        time.sleep(.05)
    state['status']=terminal_status;write(directory/'STATE.json',state)


def run(root,request_path):
    root=Path(root);allocation=admit(root,request_path);request=read(request_path);ledger=Ledger(root,'hopper')
    prepare_environment(request['numerical_mode'])
    import torch
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count()!=1 or 'H200' not in torch.cuda.get_device_name(0):
        raise PermissionError('E1_ONE_VISIBLE_H200_IN_OWN_ALLOCATION')
    configure_torch(torch,request['numerical_mode']);torch.set_num_threads(4)
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    resource.setrlimit(resource.RLIMIT_CPU,(int(115200-read(ledger.path)['CPU_s']),)*2)
    lock=OwnerLock(root/'NEURAL_OWNER.lock');started=time.process_time();status='FAILED_PRESERVED';stage='INITIAL'
    def guard():
        ledger.guard()
        if time.time()>=allocation['allocation_end_epoch']-2700:raise TimeoutError('E1_45_MIN_CLOSEOUT_RESERVE')
        heartbeat=read(root/'OWNER_HEARTBEAT.json')
        if heartbeat['run_uuid']!=root.name or not 0<=time.time()-heartbeat['observed_epoch']<=300:
            raise TimeoutError('E1_OWNER_HEARTBEAT_LOST_NO_IDLE_GPU')
    try:
        from . import model
        from .dataset import SmallDataset
        from .train import run as train
        b=read(root/'NEURAL_BINDINGS.private.json');inputs=read(root/'E1_INPUTS.private.json');identity=read(root/'evidence/M0_IDENTITY.private.json')
        with ledger.transaction() as l:l['model_loads']+=1
        instance=model.load(root,inputs['M0_checkpoint'],identity)
        dataset=SmallDataset(b['dataset'],b['source'],b['cosmos'],for_optimization=False,raw_bindings=read(b['raw_bindings']),
            selected_ids=[r['id'] for r in read(root/'plans/SMALL8_BINDING.json')['episodes']])
        dataset.initialize();stage='M0_PROBES';model.probes(root,'M0',instance,dataset,ledger,guard)
        stage='M1_TRAIN';result=train(root,instance,allocation,guard);model.unload(instance)
        if result['global_step']!=1024:raise TimeoutError('E1_PARTIAL_M1_PRESERVED_NO_RESELECT')
        for step in [256,1024]:
            label='M1-'+str(step);stage=label;checkpoint=root/'training/M1'/('checkpoint-'+str(step))
            identity=checkpoint_identity(checkpoint,label,request['source_sha'])
            with ledger.transaction() as l:l['model_loads']+=1
            instance=model.load(root,checkpoint,identity,trainable=False)
            model.probes(root,label,instance,dataset,ledger,guard)
            serve(root,label,instance,allocation,ledger,guard);model.unload(instance)
        status='M1_TRAIN_AND_FIXED_PROBE_EVAL_STAGES_TERMINAL'
        return dict(status=status)
    finally:
        cpu=time.process_time()-started;ledger.cpu(cpu)
        write(Path(request_path).parent/'PBS_PROCESS_FINAL.json',dict(status=status,stage=stage,CPU_s=cpu,
            job_id=os.environ['PBS_JOBID'],completed_epoch=time.time(),selected='NONE',deployment='NONE'))
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--request',required=True);a=p.parse_args();print(json.dumps(run(a.root,a.request)),flush=True)
