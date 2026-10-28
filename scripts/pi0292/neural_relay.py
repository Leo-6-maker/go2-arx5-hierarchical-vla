"""Resident, file-transport live inference on the verified own PBS compute node.

The local-PC owner moves fresh packets over its two SSH routes. This module
opens no port/tunnel, advances no physics and never loads a model on login.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import socket
import time

from .allocation_admission import owned_jobs, parallel_peer_jobs
from .durable import OwnerLock,read,write
from .live_inputs import prepare
from .relay_protocol import PROTOCOL,digest,exact_keys,sealed,unseal,uuid_string,validate_identity,validate_observation
from .train_adapter import build_processor,official_import,sha

REQUEST_KEYS=('protocol','kind','run_uuid','episode_uuid','sequence','tick','query_uuid','reference_uuid',
              'expires','observation_sha256','anchor_sha256','identity','payload','previous_ack')


def live_seed(layout_id,sequence):
    if type(sequence) is not int or not 0<=sequence<80 or not isinstance(layout_id,str) or not layout_id:
        raise ValueError('REGISTERED_LIVE_LAYOUT_AND_SEQUENCE_REQUIRED')
    return int.from_bytes(hashlib.sha256(('PI0292/LIVE/'+layout_id+'/'+str(sequence)).encode()).digest()[:8],'big')%(2**63)


def validate_request(packet,*,run,cases,identities,now):
    body=unseal(packet);exact_keys(body,REQUEST_KEYS)
    if body['protocol']!=PROTOCOL or body['kind']!='student_request' or body['run_uuid']!=run:
        raise ValueError('LIVE_REQUEST_PROTOCOL_OR_RUN_CHANGED')
    for key in ('episode_uuid','query_uuid','reference_uuid'):uuid_string(body[key])
    case=cases.get(body['episode_uuid'])
    if case is None or case['profile'] not in ('B0','B1'):
        raise ValueError('UNREGISTERED_STUDENT_EPISODE')
    if (type(body['sequence']) is not int or not 0<=body['sequence']<80
            or type(body['tick']) is not int or body['tick']!=body['sequence']*200):
        raise ValueError('LIVE_QUERY_TICK_OR_SEQUENCE_CHANGED')
    if type(body['expires']) not in (int,float) or not now<body['expires']<=now+60:
        raise ValueError('EXPIRED_OR_EXTENDED_LIVE_QUERY')
    validate_identity(body['identity'])
    if body['identity']!=identities[case['profile']]:
        raise ValueError('EXACT_LIVE_MODEL_IDENTITY_REQUIRED')
    validate_observation(body['payload'])
    observation=sealed(dict(protocol=PROTOCOL,kind='observation',run_uuid=run,episode_uuid=body['episode_uuid'],
        sequence=body['sequence'],tick=body['tick'],payload=body['payload']))
    if body['observation_sha256']!=observation['sha256'] or body['anchor_sha256']!=digest(body['payload']['state_eef10']):
        raise ValueError('OBSERVATION_OR_ACTUAL_ANCHOR_HASH_CHANGED')
    if body['sequence']==0 and body['previous_ack'] is not None:
        raise ValueError('INITIAL_OBSERVATION_HAS_NO_PREVIOUS_PHYSICS')
    if body['sequence'] and body['previous_ack'] is None:
        raise ValueError('NEW_OBSERVATION_REQUIRES_PREVIOUS_REAL_PREFIX_ACK')
    return body,case


def require_teacher_proof(value,run):
    if (value['status']!='PASS_ACTUAL_CORELAB_TEACHER_REFERENCE_RELAY_2_OF_2'
            or value['run_uuid']!=run or value['physics_source_sha']!='35290803cd82e10a164daaa53af5743dd43054ee'
            or value['proof_kind']!='REAL_PHYSICS_AND_INDEPENDENT_SCORE_NOT_PACKET_FIXTURE'
            or len(value['records'])!=2):
        raise PermissionError('REAL_TEACHER_RELAY_QUALIFICATION_REQUIRED_BEFORE_MODEL_QUERIES')
    for row in value['records']:
        if (row['physics_steps']!=16000 or row['paired_samples']!=641 or row['acknowledged_commands']!=80
                or row['persistent_success'] is not True or row['safety_valid'] is not True
                or len(row['physics_sha256'])!=64 or len(row['score_sha256'])!=64):
            raise PermissionError('COMPLETE_SAFE_TEACHER_REFERENCE_PHYSICS_REQUIRED')
    if {row['layout_id'] for row in value['records']}!= {'T291000','T291001'}:
        raise PermissionError('EXACT_TWO_REGISTERED_TRAIN_LAYOUTS_REQUIRED')


def require_case_phase(case,proofs):
    if case['profile'] in proofs.get('stopped_models',[]):
        raise PermissionError('REAL_SAFETY_VIOLATION_STOPPED_THIS_MODEL_ARM')
    if case['kind']=='TRAIN_CANARY':
        if case['profile']!='B1' or case['layout_id'] not in ('T291000','T291001'):
            raise PermissionError('ONLY_TWO_PREREGISTERED_B1_TRAIN_CANARIES')
        return
    if case['kind']!='CONFIRM' or case['layout_id'] not in {f'E292{i:03d}' for i in range(8)}:
        raise PermissionError('EXACT_PREREGISTERED_CONFIRM_CASE_REQUIRED')
    canaries=proofs.get('B1_canaries',[]);teachers=proofs.get('confirm_teacher',[])
    if (len(canaries)!=2 or {row['layout_id'] for row in canaries}!= {'T291000','T291001'}
            or not all(row['safety_valid'] is True and row['attempted'] is True for row in canaries)
            or sum(row['persistent_success'] is True for row in canaries)<1):
        raise PermissionError('B1_TWO_SAFE_CANARIES_AND_AT_LEAST_ONE_SUCCESS_REQUIRED')
    if (len(teachers)!=8 or {row['layout_id'] for row in teachers}!= {f'E292{i:03d}' for i in range(8)}
            or not all(row['safety_valid'] is True and row['attempted'] is True for row in teachers)
            or sum(row['persistent_success'] is True for row in teachers)<7):
        raise PermissionError('CONFIRM_TEACHER_AT_LEAST_7_OF_8_SAFE_REQUIRED')
    for row in canaries+teachers:
        if (row['proof_kind']!='REAL_CORELAB_ATTEMPT_NOT_PACKET_FIXTURE'
                or len(row['result_sha256'])!=64 or len(row['score_sha256'])!=64):
            raise PermissionError('HASH_BOUND_REAL_CASE_RESULTS_REQUIRED')
    if any(row['physics_steps']!=16000 or row['paired_samples']!=641 or row['acknowledged_commands']!=80 for row in teachers):
        raise PermissionError('ALL_EIGHT_TEACHER_CONFIRMATIONS_MUST_BE_FULL_PHYSICS')


def require_case_order(case,*,cases,query_state,proofs):
    prior=query_state['episodes']
    if case['episode_uuid'] in prior:return
    remaining=[row for row in cases.values() if row['episode_uuid'] not in prior
        and row['profile'] not in proofs.get('stopped_models',[])]
    if not remaining or case['schedule_index']!=min(row['schedule_index'] for row in remaining):
        raise PermissionError('PREREGISTERED_CANARY_AND_ALTERNATING_CASE_ORDER_REQUIRED')
    finished=proofs.get('finished_episodes',{})
    for episode_id,episode in prior.items():
        if episode_id not in finished or finished[episode_id]['proof_kind']!='REAL_CORELAB_ATTEMPT_NOT_PACKET_FIXTURE':
            raise PermissionError('PREVIOUS_PHYSICAL_ATTEMPT_TERMINAL_RECEIPT_REQUIRED')


class QueryReservations:
    """A durable intent makes a lost response non-retryable, not replayable."""
    def __init__(self,directory,run,*,mapped_ack_contracts=None):
        self.mapped_ack_contracts=dict(mapped_ack_contracts or {})
        if self.mapped_ack_contracts:
            from scripts.pi0294.dec1.mapped_ack import FixedMappedAckContract
            for episode,contract in self.mapped_ack_contracts.items():
                if (not isinstance(contract,FixedMappedAckContract)
                        or contract.condition['episode_uuid']!=episode or contract.condition['run_uuid']!=run):
                    raise TypeError('OWNER_FIXED_EPISODE_ACK_CONTRACTS_REQUIRED')
        self.directory=Path(directory);self.run=run;self.path=self.directory/'QUERY_STATE.json'
        if self.path.exists():raise FileExistsError('Never restart an existing live model session')
        self.state=dict(run_uuid=run,episodes={},queries=[],pending=None)
        if self.mapped_ack_contracts:
            self.state['mapped_ack_conditions']={k:v.condition_sha256 for k,v in self.mapped_ack_contracts.items()}
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
        contract=self.mapped_ack_contracts.get(body['episode_uuid']);initial_adapter_ack_id=None
        if contract is not None and body['identity']!=contract.condition['identity']:
            raise ValueError('MAPPED_ACK_FIXED_MODEL_IDENTITY_CHANGED')
        if contract is not None and body['sequence']==0 and body['previous_ack'] is not None:
            raise ValueError('INITIAL_MAPPED_QUERY_HAS_NO_PREVIOUS_ACK')
        if body['sequence']:
            previous=next(row for row in reversed(self.state['queries']) if row['episode_uuid']==body['episode_uuid'])
            ack=unseal(body['previous_ack'])
            exact_keys(ack,('protocol','kind','run_uuid','episode_uuid','sequence','tick','payload'))
            actual=ack['payload']
            if contract is None:
                exact_keys(actual,('query_uuid','reference_uuid','command_sha256','prefix_sha256',
                    'tick_after','integrations','physics_receipt_sha256'))
            else:
                from scripts.pi0294.dec1.mapped_ack import ValidatedMappedAck
                if previous['mapped_ack_binding']['response_sha256']!=previous['response_sha256']:
                    raise ValueError('ORIGINAL_COMPLETED_RESPONSE_SHA_CHANGED')
                checked=contract.validate(body['previous_ack'],expected=previous['mapped_ack_binding'])
                if not isinstance(checked,ValidatedMappedAck) or checked.packet_sha256!=body['previous_ack']['sha256']:
                    raise ValueError('FULL_PREVIOUS_MAPPED_ACK_PACKET_REQUIRED')
                initial_adapter_ack_id=checked.final_adapter_ack_id
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
        if len(self.state['queries'])>=1440:raise PermissionError('LIVE_1440_GENERATION_LIMIT')
        self.state['pending']=dict(query_uuid=body['query_uuid'],episode_uuid=body['episode_uuid'],
            reference_uuid=body['reference_uuid'],sequence=body['sequence'],observation_sha256=body['observation_sha256'],started_epoch=now,expires=body['expires'])
        if contract is not None:self.state['pending']['initial_adapter_ack_id']=initial_adapter_ack_id
        self.state['queries'].append(dict(self.state['pending']));episode['snapshots'].append(body['payload']['snapshot_uuid'])
        write(self.path,self.state)

    def complete(self,body,response_sha256,*,prefix_sha256,command_sha256,action_packet=None,command_packet=None):
        if self.state['pending']['query_uuid']!=body['query_uuid']:raise ValueError('WRONG_GENERATION_COMPLETION')
        contract=self.mapped_ack_contracts.get(body['episode_uuid'])
        if contract is not None:
            binding=contract.bind_action(action_packet,command_packet,
                initial_adapter_ack_id=self.state['pending']['initial_adapter_ack_id'])
            if (binding['response_sha256']!=response_sha256 or binding['command_sha256']!=command_sha256
                    or binding['prefix_sha256']!=prefix_sha256 or binding['query_uuid']!=body['query_uuid']
                    or binding['reference_uuid']!=body['reference_uuid'] or binding['sequence']!=body['sequence']
                    or binding['tick']!=body['tick']):
                raise ValueError('MAPPED_ACK_ORIGINAL_GENERATION_BINDING_REQUIRED')
            self.state['queries'][-1]['mapped_ack_binding']=binding
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


def run(args):
    if not os.environ.get('PBS_JOBID'):raise PermissionError('NO_LIVE_MODELS_ON_LOGIN_OR_CORELAB')
    root=Path(args.root);request=read(args.request);bindings=read(args.bindings)
    from .neural_train import verify_allocation,verify_inputs,COSMOS_REVISION
    allocation=verify_allocation(root,'relay')
    if request['source_sha']!='c187d9573bdff302783fb5f41d06fcaa1753d2a8':
        raise PermissionError('ORIGINAL_SCIENTIFIC_MODEL_SOURCE_REQUIRED')
    parallel_peer_jobs(jobs={k:v for k,v in owned_jobs(os.environ['USER']).items() if k!=os.environ['PBS_JOBID']},
        user=os.environ['USER'],allowed_ids=request['parallel_peer_ids'],authority=request['parallel_authority'])
    if read(root/'training/main/session-start/TRAINING_RUN.json')['status']!='MAIN_COMPLETED_FIXED_S4096':
        raise PermissionError('FIXED_FULL_MAIN_REQUIRED')
    if read(root/'evidence/PAIRED_OFFLINE_ATTRIBUTION.json')['status']!='PAIRED_OFFLINE_RECORDS_ANALYZED_NOT_TASK_SUCCESS':
        raise PermissionError('ACTUAL_THREE_PROFILE_OFFLINE_ATTRIBUTION_REQUIRED')
    verify_inputs(root,bindings,'relay');official_import(bindings['source'])
    registry_path=Path(request['live_registry'])
    if not registry_path.resolve().is_relative_to(root.resolve()) or sha(registry_path)!=request['live_registry_sha256']:
        raise PermissionError('SEALED_MODEL_CASE_AND_IDENTITY_REGISTRY_REQUIRED')
    registry=read(registry_path)
    if registry['run_uuid']!=root.name or registry['original_T0']!=1791860304 or len(registry['cases'])!=18:
        raise PermissionError('EXACT_TWO_CANARIES_AND_SIXTEEN_MODEL_CASES_REQUIRED')
    identities=registry['identities'];cases=registry['cases']
    for identity in identities.values():validate_identity(identity)
    if set(identities)!= {'B0','B1'} or {row['profile'] for row in cases.values()}!= {'B0','B1'}:
        raise PermissionError('FIXED_B0_AND_FINAL_B1_ONLY')
    expected={('B1','TRAIN_CANARY',name) for name in ('T291000','T291001')}
    expected|={(profile,'CONFIRM',f'E292{i:03d}') for profile in ('B0','B1') for i in range(8)}
    if {(row['profile'],row['kind'],row['layout_id']) for row in cases.values()}!=expected:
        raise PermissionError('EXACT_REGISTERED_LIVE_CASE_SET_REQUIRED')
    order=[('B1','TRAIN_CANARY',name) for name in ('T291000','T291001')]
    for i in range(8):
        order.extend((profile,'CONFIRM',f'E292{i:03d}') for profile in (('B0','B1') if i%2==0 else ('B1','B0')))
    for episode_id,row in cases.items():
        uuid_string(episode_id)
        if (row['episode_uuid']!=episode_id or type(row['schedule_index']) is not int
                or not 0<=row['schedule_index']<18 or tuple(order[row['schedule_index']])!=(row['profile'],row['kind'],row['layout_id'])):
            raise PermissionError('EXACT_ALTERNATING_MODEL_CASE_ORDER_REQUIRED')
    base_files={row['name']:row['sha256'] for row in read(root/'evidence/PRETRAINED_FILES.private.json')['files']}
    full=read(root/'evidence/MAIN_FULL_CHECKPOINT_INVENTORIES.json')['checkpoints']
    final=next(row for row in full if row['global_step']==4096)
    candidate_files={name:value['sha256'] for name,value in final['files'].items()}
    if registry['profile_files']!= {'B0':base_files,'B1':candidate_files}:
        raise PermissionError('EXACT_PRISTINE_AND_FULL_FINAL_CHECKPOINT_IDENTITIES_REQUIRED')
    candidate_revision=hashlib.sha1(json.dumps(candidate_files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if (identities['B0']['model_revision']!='2fc962b973bccdd5d8ce4f67cc63b264d6886495'
            or identities['B1']['model_revision']!=candidate_revision
            or registry['model_revision_kind']['B1']!='LOCAL_FULL_CHECKPOINT_SHA1_NOT_HF_REVISION'):
        raise PermissionError('TRUTHFUL_FIXED_BASE_AND_LOCAL_CANDIDATE_REVISION_REQUIRED')
    for profile in ('B0','B1'):
        path=Path(bindings['base']) if profile=='B0' else root/'training/main/checkpoint-4096'
        if (identities[profile]['config_sha256']!=sha(path/'config.json')
                or identities[profile]['statistics_sha256']!=sha(Path(bindings['dataset'])/'statistics.json')
                or identities[profile]['source_sha']!='51d4c89f72fda44cbf77285c6a8114b52676b8a1'):
            raise PermissionError('ACTUAL_MODEL_CONFIG_OFFICIAL_SOURCE_AND_FROZEN_STATS_REQUIRED')
    directory=root/'relay/neural-session-01';directory.mkdir(parents=True,exist_ok=False)
    for name in ('inbox','outbox','receipts'): (directory/name).mkdir()
    owner=OwnerLock(root/'NEURAL_OWNER.lock');queries=QueryReservations(directory,root.name)
    state=dict(status='WAIT_REAL_TEACHER_PROOF_AND_ROUTE',pid=os.getpid(),
        proc_start=Path('/proc/self/stat').read_text().split()[21],job_id=os.environ['PBS_JOBID'],
        hostname=socket.gethostname(),source_sha=registry['controller_source_sha'],allocation=allocation,
        queries=0,model_profiles_loaded=[],started_epoch=time.time(),selected='NONE',deployment='NONE')
    write(directory/'STATE.json',state)
    import torch
    from transformers import AutoModel,set_seed
    from gr00t.data.embodiment_tags import EmbodimentTag
    import gr00t.model.gr00t_n1d7.gr00t_n1d7
    from .trainer_adapter import tensor_identity
    import numpy as np
    torch.set_num_threads(4);os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',NO_ALBUMENTATIONS_UPDATE='1')
    end=min(allocation['allocation_end_epoch']-120,read(root/'ADOPTION.json')['new_work_cutoff_epoch'])
    models={};processor=None;processed=set()
    proof_path=directory/'TEACHER_RELAY_PROOF.json'
    def query_expired(signum,frame):raise TimeoutError('LIVE_QUERY_60_SECOND_TIMEOUT')
    signal.signal(signal.SIGALRM,query_expired)
    try:
        while time.time()<end:
            state['heartbeat_epoch']=time.time();write(directory/'STATE.json',state)
            stop_path=directory/'STOP.json'
            if stop_path.exists():
                stop=read(stop_path)
                if stop['run_uuid']!=root.name or stop['job_id']!=state['job_id']:
                    raise PermissionError('WRONG_SESSION_STOP_IDENTITY')
                state.update(status='LIVE_MODEL_SESSION_TERMINAL_NOT_TASK_SUCCESS',coordinator_closeout=stop)
                return state
            if not proof_path.exists():time.sleep(.1);continue
            require_teacher_proof(read(proof_path),root.name)
            if not models:
                for profile in ('B0','B1'):
                    path=Path(bindings['base']) if profile=='B0' else root/'training/main/checkpoint-4096'
                    for name,digest_value in registry['profile_files'][profile].items():
                        target=path/name
                        if not target.resolve().is_relative_to(path.resolve()) or sha(target)!=digest_value:
                            raise ValueError('FULL_LIVE_MODEL_PROFILE_CHANGED')
                    ledger=read(root/'HOPPER_RESOURCE_LEDGER.json');ledger['model_loads']+=1;write(root/'HOPPER_RESOURCE_LEDGER.json',ledger)
                    verify_allocation(root,'relay');set_seed(29200)
                    model,loading=AutoModel.from_pretrained(path,local_files_only=True,trust_remote_code=True,output_loading_info=True,
                        transformers_loading_kwargs=dict(local_files_only=True,trust_remote_code=True,cache_dir=bindings['cosmos_hub'],revision=COSMOS_REVISION),
                        tune_llm=False,tune_visual=False,tune_projector=True,tune_diffusion_model=True,tune_vlln=True,
                        state_dropout_prob=0.,backbone_trainable_params_fp32=True,load_bf16=False)
                    missing=loading.get('missing_keys',[])
                    if any('mask_token' not in name for name in missing) or loading.get('unexpected_keys') or loading.get('mismatched_keys'):
                        raise ValueError('LIVE_PROFILE_LOADING_MISMATCH')
                    if missing and model.action_head.mask_token is not None:
                        with torch.no_grad():model.action_head.mask_token.copy_(.02*torch.randn_like(model.action_head.mask_token))
                    if model.action_head.num_inference_timesteps!=4:raise ValueError('FIXED_NATIVE_FOUR_STEP_SAMPLER_REQUIRED')
                    model.to('cuda').eval();models[profile]=model;state['model_profiles_loaded'].append(profile)
                    write(directory/'receipts'/('LOAD_'+profile+'.json'),dict(loading=loading,profile=profile,identity=identities[profile]))
                processor=build_processor(bindings['source'],bindings['cosmos'],read(Path(bindings['dataset'])/'statistics.json'))
                processor_directory=directory/'processor';processor.save_pretrained(processor_directory)
                actual_processor={name:sha(processor_directory/name) for name in ('processor_config.json','statistics.json','embodiment_id.json')}
                if any(identities[profile]['processor_sha256']!=digest(actual_processor) for profile in ('B0','B1')):
                    raise ValueError('ACTUAL_SHARED_OFFICIAL_PROCESSOR_IDENTITY_CHANGED')
                state['status']='REAL_MODELS_RESIDENT_WAIT_FRESH_OBSERVATION';write(directory/'STATE.json',state)
            paths=sorted((directory/'inbox').glob('*.json'))
            for path in paths:
                if path.name in processed:continue
                uuid_string(path.stem);packet=read(path)
                body,case=validate_request(packet,run=root.name,cases=cases,identities=identities,now=time.time())
                if path.stem!=body['query_uuid']:raise ValueError('REQUEST_FILENAME_IDENTITY_CHANGED')
                phase_path=directory/'CASE_PHASE_PROOFS.json'
                proofs=read(phase_path) if phase_path.exists() else {}
                if proofs.get('run_uuid',root.name)!=root.name:raise ValueError('WRONG_CASE_PHASE_RUN')
                require_case_phase(case,proofs)
                require_case_order(case,cases=cases,query_state=queries.state,proofs=proofs)
                queries.begin(body,time.time());processed.add(path.name)
                ledger=read(root/'HOPPER_RESOURCE_LEDGER.json')
                if ledger.get('live_queries',0)>=1440 or ledger['generation_queries']>=2624:
                    raise PermissionError('ORIGINAL_GENERATION_BUDGET')
                ledger['live_queries']=ledger.get('live_queries',0)+1;ledger['generation_queries']+=1
                write(root/'HOPPER_RESOURCE_LEDGER.json',ledger)
                captured=[];model=models[case['profile']]
                def initial_noise(module,inputs):
                    if not captured:captured.append(inputs[0].detach().clone())
                hook=model.action_head.action_encoder.register_forward_pre_hook(initial_noise)
                try:
                    import math
                    signal.alarm(max(1,math.ceil(body['expires']-time.time())))
                    batch=prepare(processor,body['payload'])
                    batch={k:{name:value.to('cuda') if torch.is_tensor(value) else value for name,value in values.items()} for k,values in batch.items()}
                    seed=live_seed(case['layout_id'],body['sequence']);set_seed(seed)
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        normalized=model.get_action(**batch)['action_pred'].detach().float().cpu().numpy()
                    if len(captured)!=1:raise ValueError('ACTUAL_INITIAL_NATIVE_NOISE_NOT_OBSERVED')
                    actual=np.asarray(body['payload']['state_eef10'],dtype=np.float64)
                    decoded=processor.decode_action(normalized,EmbodimentTag.NEW_EMBODIMENT,
                        {'eef_9d':actual[None,None,:9],'gripper_position':actual[None,None,9:]})
                    physical=np.concatenate([decoded[k] for k in ('eef_9d','gripper_position')],axis=-1)[0]
                    if physical.shape!=(40,10) or not np.isfinite(physical).all():raise ValueError('NONFINITE_OR_WRONG_H40_LIVE_DECODE')
                    if time.time()>=body['expires']:raise TimeoutError('LIVE_QUERY_60_SECOND_TIMEOUT')
                    action=sealed(dict(protocol=PROTOCOL,kind='action',run_uuid=root.name,episode_uuid=body['episode_uuid'],
                        sequence=body['sequence'],tick=body['tick'],payload={key:body[key] for key in
                            ('query_uuid','reference_uuid','observation_sha256','anchor_sha256','identity')}|
                        dict(trajectory_eef10=physical.tolist(),offsets_ns=[i*50000000 for i in range(40)])))
                    output=directory/'outbox'/path.name
                    if output.exists():raise FileExistsError('NEVER_OVERWRITE_A_LIVE_RESPONSE')
                    arrays_path=directory/'receipts'/(path.stem+'.npz')
                    np.savez_compressed(arrays_path,normalized_action=normalized,decoded_action=physical,
                        actual_state=actual,noise_bits=captured[0].contiguous().view(torch.uint8).cpu().numpy())
                    with arrays_path.open('rb') as handle:os.fsync(handle.fileno())
                    write(directory/'receipts'/path.name,dict(seed=seed,noise=tensor_identity(captured[0]),
                        observation_sha256=body['observation_sha256'],action_sha256=action['sha256'],
                        raw_arrays_sha256=sha(directory/'receipts'/(path.stem+'.npz')),native_inverse_clip=True,
                        normalized_outside_unit_components=int(np.count_nonzero(np.abs(normalized[...,:10])>1)),
                        extra_correction=False,model_inputs_future_action=False,physics_integrations=0))
                    # Publish only after actual arrays/noise/provenance are durable.
                    write(output,action)
                    prefix=physical[:8].tolist();prefix_sha=digest(prefix)
                    expected_command=sealed(dict(protocol=PROTOCOL,kind='install_prefix8',run_uuid=root.name,
                        episode_uuid=body['episode_uuid'],sequence=body['sequence'],tick=body['tick'],
                        query_uuid=body['query_uuid'],reference_uuid=body['reference_uuid'],
                        observation_sha256=body['observation_sha256'],anchor_sha256=body['anchor_sha256'],
                        prefix_eef10=prefix,prefix_sha256=prefix_sha,advance_integrations=200,
                        frame=body['payload']['frame'],tcp=body['payload']['tcp'],expires=body['expires']))
                    queries.complete(body,action['sha256'],prefix_sha256=prefix_sha,command_sha256=expected_command['sha256'],
                        action_packet=action,command_packet=expected_command)
                    state['queries']+=1;write(directory/'STATE.json',state)
                except BaseException as error:
                    known=isinstance(error,(ValueError,AssertionError,FloatingPointError,TimeoutError))
                    if (directory/'outbox'/path.name).exists():
                        known=False
                        state['published_response_finalization']='UNKNOWN_DO_NOT_RESEND_OR_RESAMPLE'
                    else:
                        write(directory/'outbox'/(path.stem+'.ERROR.json'),dict(status='LIVE_QUERY_FAILED_PRESERVED_NO_RESAMPLING',
                            query_uuid=body['query_uuid'],episode_uuid=body['episode_uuid'],error_type=type(error).__name__,error=str(error)))
                    queries.fail(body,type(error).__name__+': '+str(error),known_terminal=known)
                    if not known:raise
                finally:signal.alarm(0);hook.remove()
            time.sleep(.05)
        state['status']='ALLOCATION_CUTOFF_LIVE_SESSION_PARTIAL';return state
    except BaseException as error:
        state.update(status='FAILED_PRESERVED_NO_MODEL_OR_QUERY_RETRY',error_type=type(error).__name__,error=str(error));raise
    finally:
        u=resource.getrusage(resource.RUSAGE_SELF)
        state.update(finished_epoch=time.time(),CPU_s=u.ru_utime+u.ru_stime)
        write(directory/'STATE.json',state);owner.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('root','bindings','request'):parser.add_argument('--'+name,required=True)
    print(json.dumps(run(parser.parse_args())))
