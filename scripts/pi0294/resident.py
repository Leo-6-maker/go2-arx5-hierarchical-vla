"""One shared checkpoint inference primitive; caller owns sealed ACK/case protocol."""
import hashlib
from pathlib import Path

import numpy as np
import argparse
import math
import os
import signal
import time

from scripts.pi0292.durable import read,write,OwnerLock
from scripts.pi0292.relay_protocol import PROTOCOL,digest,exact_keys,sealed,unseal,uuid_string,validate_identity,validate_observation
from scripts.pi0292.live_inputs import prepare
from .native_dataset import sha, official_import
from .neural import verify_allocation, verify_inputs, COSMOS_REVISION


class ResidentModel:
    def __init__(self,root,request_path,bindings,checkpoint,identity):
        self.root,self.bindings = Path(root),bindings
        self.allocation = verify_allocation(self.root,'resident',request_path)
        verify_inputs(self.root,bindings,'resident')
        checkpoint=Path(checkpoint)
        complete=read(checkpoint/'CHECKPOINT_COMPLETE.json')
        if complete['global_step'] not in (512,2048,4096,6144):
            raise PermissionError('ONLY_REGISTERED_FORMAL_CHECKPOINTS')
        inventory=identity['files']
        if not inventory or not any(name.endswith('.safetensors') for name in inventory):
            raise ValueError('ACTUAL_FULL_MODEL_WEIGHT_INVENTORY_REQUIRED')
        for name,digest in complete['required_files'].items():
            if inventory.get(name)!=digest:raise ValueError('CHECKPOINT_COMPLETE_INVENTORY_MISMATCH')
        for name,digest in inventory.items():
            path=checkpoint/name
            if not path.resolve().is_relative_to(checkpoint.resolve()) or sha(path)!=digest:
                raise ValueError('FULL_SHARED_CHECKPOINT_CHANGED')
        if identity['statistics_sha256']!=sha(Path(bindings['dataset'])/'statistics.json'):
            raise ValueError('SHARED_FROZEN_STATISTICS_CHANGED')
        official_import(bindings['source'])
        import torch
        from transformers import AutoModel
        import gr00t.model.gr00t_n1d7.gr00t_n1d7
        from scripts.pi0292.train_adapter import build_processor
        from .local_backbone import local_model_name
        self.model,loading=AutoModel.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=True,
            output_loading_info=True,model_name=local_model_name(root,bindings),transformers_loading_kwargs=dict(local_files_only=True,trust_remote_code=True,
                cache_dir=bindings['cosmos_hub'],revision=COSMOS_REVISION),tune_llm=False,tune_visual=False,
            tune_projector=True,tune_diffusion_model=True,tune_vlln=True,state_dropout_prob=0.,
            backbone_trainable_params_fp32=True,load_bf16=False)
        if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys')):
            raise ValueError('FORMAL_FULL_CHECKPOINT_LOADING_MISMATCH')
        self.model.to('cuda').eval()
        if self.model.action_head.num_inference_timesteps!=4:raise ValueError('NATIVE_FOUR_STEP_SAMPLER_REQUIRED')
        self.model.requires_grad_(False)
        self.processor=build_processor(bindings['source'],bindings['cosmos'],read(Path(bindings['dataset'])/'statistics.json'))
        self.identity=identity

    def generate(self,payload,seed):
        """Exactly one native draw from a fresh caller-validated real observation."""
        import torch
        from transformers import set_seed
        from gr00t.data.embodiment_tags import EmbodimentTag
        from .native_train import tensor_identity
        batch=prepare(self.processor,payload)
        batch={k:{n:v.to('cuda') if torch.is_tensor(v) else v for n,v in values.items()} for k,values in batch.items()}
        noise=[]
        def record(module,args):
            if not noise:
                noise.append(args[0].detach().clone())
        hook=self.model.action_head.action_encoder.register_forward_pre_hook(record)
        try:
            set_seed(int(seed))
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                normalized=self.model.get_action(**batch)['action_pred'].detach().float().cpu().numpy()
            actual=np.asarray(payload['state_eef10'],dtype=np.float64)
            decoded=self.processor.decode_action(normalized,EmbodimentTag.NEW_EMBODIMENT,
                {'eef_9d':actual[None,None,:9],'gripper_position':actual[None,None,9:]})
            physical=np.concatenate([decoded[k] for k in ('eef_9d','gripper_position')],axis=-1)[0]
            if physical.shape!=(40,10) or not np.isfinite(physical).all() or len(noise)!=1:
                raise ValueError('INVALID_SHARED_NATIVE_GENERATION')
            return dict(normalized=normalized,physical_H40=physical,noise=tensor_identity(noise[0]),
                identity=self.identity,extra_correction=False,physics_integrations=0,
                normalized_outside_unit_components=int(np.count_nonzero(np.abs(normalized[...,:10])>1)))
        finally:
            hook.remove()

class QueryReservations:
    """A durable intent makes a lost response non-retryable, not replayable."""
    def __init__(self,directory,run,max_queries=32000):
        if type(max_queries) is not int or not 0<max_queries<=32000:
            raise ValueError('BOUNDED_REGISTERED_RESIDENT_QUERY_COUNT_REQUIRED')
        self.max_queries=max_queries
        self.directory=Path(directory);self.run=run;self.path=self.directory/'QUERY_STATE.json'
        if self.path.exists():raise FileExistsError('Never restart an existing live model session')
        self.state=dict(run_uuid=run,episodes={},queries=[],pending=None)
        write(self.path,self.state)


    def begin(self,body,now):
        if self.state['pending'] is not None:raise ValueError('PRIOR_GENERATION_OUTCOME_UNKNOWN_NO_RETRY')
        if any(row['query_uuid']==body['query_uuid'] for row in self.state['queries']):
            raise ValueError('DUPLICATE_GENERATION_QUERY')
        episode=self.state['episodes'].setdefault(body['episode_uuid'],dict(next_sequence=0,
            started_epoch=now,snapshots=[],stopped=False))
        if (episode['stopped'] or body['sequence']!=episode['next_sequence'] or now>=episode['started_epoch']+900
                or body['payload']['snapshot_uuid'] in episode['snapshots']):
            raise ValueError('STOPPED_STALE_REORDERED_OR_REUSED_LIVE_OBSERVATION')
        if body['sequence']:
            previous=next(row for row in reversed(self.state['queries']) if row['episode_uuid']==body['episode_uuid'])
            ack=unseal(body['previous_ack'])
            exact_keys(ack,('protocol','kind','run_uuid','episode_uuid','sequence','tick','payload'))
            actual=ack['payload'];exact_keys(actual,('query_uuid','reference_uuid','command_sha256','prefix_sha256',
                'tick_after','integrations','physics_receipt_sha256'))
            if (any(type(ack[key]) is not int for key in ('sequence','tick'))
                    or any(type(actual[key]) is not int for key in ('tick_after','integrations'))
                    or not isinstance(actual['physics_receipt_sha256'],str)
                    or any(c not in '0123456789abcdef' for c in actual['physics_receipt_sha256'])):
                raise ValueError('ACTUAL_PHYSICS_ACK_INTEGER_AND_HASH_FIELDS_REQUIRED')
            if (ack['protocol']!=PROTOCOL or ack['kind']!='ack' or ack['run_uuid']!=self.run
                    or ack['episode_uuid']!=body['episode_uuid'] or ack['sequence']!=body['sequence']-1
                    or ack['tick']!=body['tick']-200 or actual['tick_after']!=body['tick']
                    or type(actual['integrations']) is not int or actual['integrations']!=200
                    or actual['query_uuid']!=previous['query_uuid'] or actual['reference_uuid']!=previous['reference_uuid']
                    or actual['command_sha256']!=previous['expected_command_sha256']
                    or actual['prefix_sha256']!=previous['expected_prefix_sha256']
                    or len(actual['physics_receipt_sha256'])!=64):
                raise ValueError('PREVIOUS_ACTUAL_PREFIX_ACK_DOES_NOT_BIND_LAST_GENERATION')
        if len(self.state['queries'])>=self.max_queries:raise PermissionError('PI0294_REGISTERED_GENERATION_LIMIT')
        self.state['pending']=dict(query_uuid=body['query_uuid'],episode_uuid=body['episode_uuid'],
            reference_uuid=body['reference_uuid'],sequence=body['sequence'],observation_sha256=body['observation_sha256'],started_epoch=now,expires=body['expires'])
        self.state['queries'].append(dict(self.state['pending']));episode['snapshots'].append(body['payload']['snapshot_uuid'])
        write(self.path,self.state)

    def complete(self,body,response_sha256,*,prefix_sha256,command_sha256):
        if self.state['pending']['query_uuid']!=body['query_uuid']:raise ValueError('WRONG_GENERATION_COMPLETION')
        episode=self.state['episodes'][body['episode_uuid']];episode['next_sequence']+=1
        self.state['queries'][-1].update(response_sha256=response_sha256,
            expected_prefix_sha256=prefix_sha256,expected_command_sha256=command_sha256)
        self.state['pending']=None
        write(self.path,self.state)

    def fail(self,body,error,*,known_terminal=False):
        self.state['episodes'][body['episode_uuid']]['stopped']=True
        self.state['queries'][-1]['error']=error
        # Retain intent: a generation that failed after reservation cannot be
        # retried by the same service or silently resampled in a new process.
        if known_terminal:
            self.state['queries'][-1]['terminal_outcome']='FAILED_PRESERVED_NO_RESAMPLING'
            self.state['pending']=None
        write(self.path,self.state)




def serve(args):
    root,request_path=Path(args.root),Path(args.request)
    request=read(request_path)
    allocation=verify_allocation(root,'resident',request_path)
    registry_path,identity_path=Path(args.registry),Path(args.identity)
    if sha(registry_path)!=request['resident_registry_sha256'] or sha(identity_path)!=request['resident_identity_sha256']:
        raise PermissionError('SEALED_SHARED_CHECKPOINT_CASES_REQUIRED')
    registry,identity=read(registry_path),read(identity_path)
    if registry['run_uuid']!=root.name or registry['checkpoint_identity']!=identity:
        raise PermissionError('ONE_SHARED_CHECKPOINT_FOR_ALL_REGISTERED_CASES')
    validate_identity(identity['relay_identity'])
    ready=read(root/'evidence/LIVE_READINESS.json')
    from .continuation import RUN as CONTINUATION_RUN, PARENT
    readiness_run=PARENT if root.name==CONTINUATION_RUN else root.name
    if (root.name==CONTINUATION_RUN and request['live_readiness_sha256']!='c2ca87bda6942f17b037d47bdacf33f069f2bd1ecb2f8d57126d87ba8a69ed5c'):
        raise PermissionError('EXACT_REUSED_PARENT_ROUTE_QUALIFICATION_REQUIRED')
    if sha(root/'evidence/LIVE_READINESS.json')!=request['live_readiness_sha256'] or ready['run_uuid']!=readiness_run or ready['status']!='PASS_ACTUAL_TEACHER_AND_DUAL_SSH_ROUTE':
        raise PermissionError('ACTUAL_TEACHER_AND_ROUTE_PROOF_REQUIRED')
    directory=Path(args.directory)
    if not directory.resolve().is_relative_to(root.resolve()) or directory.exists():
        raise PermissionError('NEW_REGISTERED_SESSION_DIRECTORY_REQUIRED')
    directory.mkdir(parents=True)
    for name in ('inbox','outbox','receipts'):(directory/name).mkdir()
    owner=OwnerLock(root/'NEURAL_OWNER.lock')
    reservations=QueryReservations(directory,root.name,request['max_queries'])
    state=dict(status='LOADING_ONE_SHARED_MODEL',job_id=os.environ['PBS_JOBID'],pid=os.getpid(),queries=0,
               checkpoint_identity=identity,selected='NONE',deployment='NONE')
    write(directory/'STATE.json',state)
    try:
        ledger_path=root/'HOPPER_RESOURCE_LEDGER.json'
        ledger=read(ledger_path);ledger['model_loads']+=1;write(ledger_path,ledger)
        model=ResidentModel(root,request_path,read(args.bindings),args.checkpoint,identity)
        def expire(signum,frame):raise TimeoutError('FRESH_QUERY_TIMEOUT_NO_RESAMPLE')
        signal.signal(signal.SIGALRM,expire)
        seen=set()
        from .continuation import RUN as CONTINUATION_RUN
        boundary=read(root/'ADOPTION.json')['new_work_cutoff_epoch'] if root.name==CONTINUATION_RUN else 1791943200
        end=min(allocation['allocation_end_epoch']-120,boundary)
        state['status']='RESIDENT_WAIT_FRESH_REAL_OBSERVATION';write(directory/'STATE.json',state)
        while time.time()<end:
            if (directory/'STOP.json').exists():
                stop=read(directory/'STOP.json')
                if stop['job_id']!=state['job_id'] or stop['run_uuid']!=root.name:
                    raise PermissionError('EXACT_OWNER_SESSION_STOP_REQUIRED')
                break
            for path in sorted((directory/'inbox').glob('*.json')):
                if path.name in seen:continue
                body=unseal(read(path))
                exact_keys(body,('protocol','kind','run_uuid','episode_uuid','sequence','tick','query_uuid',
                    'reference_uuid','expires','observation_sha256','anchor_sha256','identity','payload','previous_ack'))
                for key in ('episode_uuid','query_uuid','reference_uuid'):uuid_string(body[key])
                case=registry['cases'].get(body['episode_uuid'])
                if case is None or body['run_uuid']!=root.name or body['protocol']!=PROTOCOL or body['kind']!='student_request':
                    raise PermissionError('REGISTERED_PI0294_REAL_QUERY_REQUIRED')
                if path.stem!=body['query_uuid'] or body['identity']!=identity['relay_identity']:
                    raise ValueError('EXACT_SHARED_IDENTITY_AND_FILENAME_REQUIRED')
                if type(body['sequence']) is not int or not 0<=body['sequence']<80 or body['tick']!=body['sequence']*200:
                    raise ValueError('ACTUAL_PREFIX_OBSERVATION_ORDER_REQUIRED')
                if not time.time()<body['expires']<=time.time()+60:
                    raise TimeoutError('EXPIRED_OBSERVATION')
                validate_observation(body['payload'])
                observation=sealed(dict(protocol=PROTOCOL,kind='observation',run_uuid=root.name,
                    episode_uuid=body['episode_uuid'],sequence=body['sequence'],tick=body['tick'],payload=body['payload']))
                if body['observation_sha256']!=observation['sha256'] or body['anchor_sha256']!=digest(body['payload']['state_eef10']):
                    raise ValueError('ACTUAL_OBSERVATION_ANCHOR_BINDING_REQUIRED')
                if body['sequence']==0 and body['previous_ack'] is not None:
                    raise ValueError('INITIAL_QUERY_HAS_NO_ACK')
                if body['sequence'] and body['previous_ack'] is None:
                    raise ValueError('FRESH_QUERY_NEEDS_PREVIOUS_PHYSICS_ACK')
                from .continuation import RUN as CONTINUATION_RUN, LIMITS
                if root.name==CONTINUATION_RUN:
                    ledger=read(ledger_path)
                    if ledger['generation_queries']>=LIMITS['generation_queries']:
                        raise PermissionError('CONTINUATION_GLOBAL_QUERY_CAP')
                    reservations.begin(body,time.time())
                    ledger['generation_queries']+=1;write(ledger_path,ledger)
                else:reservations.begin(body,time.time())
                seen.add(path.name)
                try:
                    signal.alarm(max(1,math.ceil(body['expires']-time.time())))
                    seed=int.from_bytes(hashlib.sha256(('PI0294/'+case['layout_id']+'/'+str(body['sequence'])).encode()).digest()[:8],'big')%(2**63)
                    value=model.generate(body['payload'],seed)
                    if time.time()>=body['expires']:raise TimeoutError('QUERY_EXPIRED_AFTER_GENERATION')
                    physical=value['physical_H40'].tolist()
                    action=sealed(dict(protocol=PROTOCOL,kind='action',run_uuid=root.name,
                        episode_uuid=body['episode_uuid'],sequence=body['sequence'],tick=body['tick'],
                        payload={k:body[k] for k in ('query_uuid','reference_uuid','observation_sha256','anchor_sha256','identity')}|
                        dict(trajectory_eef10=physical,offsets_ns=[i*50000000 for i in range(40)])))
                    prefix=physical[:8];prefix_sha=digest(prefix)
                    command=sealed(dict(protocol=PROTOCOL,kind='install_prefix8',run_uuid=root.name,
                        episode_uuid=body['episode_uuid'],sequence=body['sequence'],tick=body['tick'],
                        query_uuid=body['query_uuid'],reference_uuid=body['reference_uuid'],
                        observation_sha256=body['observation_sha256'],anchor_sha256=body['anchor_sha256'],
                        prefix_eef10=prefix,prefix_sha256=prefix_sha,advance_integrations=200,
                        frame=body['payload']['frame'],tcp=body['payload']['tcp'],expires=body['expires']))
                    np.savez_compressed(directory/'receipts'/(path.stem+'.npz'),normalized=value['normalized'],physical=value['physical_H40'])
                    write(directory/'receipts'/path.name,dict(seed=seed,noise=value['noise'],action_sha256=action['sha256'],
                        arrays_sha256=sha(directory/'receipts'/(path.stem+'.npz')),physics_steps=0,extra_correction=False))
                    reservations.complete(body,action['sha256'],prefix_sha256=prefix_sha,command_sha256=command['sha256'])
                    write(directory/'outbox'/path.name,action)
                    state['queries']+=1;write(directory/'STATE.json',state)
                except BaseException as error:
                    reservations.fail(body,type(error).__name__+': '+str(error),known_terminal=False)
                    raise
                finally:signal.alarm(0)
            time.sleep(.05)
        state['status']='OWN_SESSION_TERMINAL_NOT_TASK_SUCCESS'
        return state
    finally:
        write(directory/'STATE.json',state);owner.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('root','request','bindings','checkpoint','identity','registry','directory'):
        parser.add_argument('--'+name,required=True)
    serve(parser.parse_args())


