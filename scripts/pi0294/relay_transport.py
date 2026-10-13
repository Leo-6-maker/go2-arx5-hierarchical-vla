"""Local owner orchestration of the existing ACK exchange, never physics/model."""
from pathlib import Path
import copy
import hashlib
import json
import time
import uuid

from scripts.pi0292.durable import read,write
from scripts.pi0292.relay_protocol import RelayExchange,PROTOCOL,sealed,unseal,exact_keys,digest,validate_identity


def model_registry(panel,cards,inventory,delivery,*,step,run_uuid,core,hopper,deadline_epoch,
                   inventory_sha256,processor_sha256,live_readiness_sha256,now=None):
    """Prepare local-owner cases; never load, select, dispatch or infer a model."""
    now=time.time() if now is None else now
    category='TRAIN_DEVELOPMENT' if step==512 else 'DEV'
    name='M512_TRAIN' if step==512 else 'SHARED_DEV'
    if (step not in (512,2048,4096,6144) or panel['run_uuid']!=run_uuid
            or panel['status']!='FROZEN_BEFORE_FIRST_V3_NEURAL_REQUALIFICATION_AND_MAIN_UPDATE'
            or not panel['frozen_epoch']<=now<deadline_epoch<=1791939600):
        raise PermissionError('FIXED_PANELS_AND_ORIGINAL_PHYSICAL_CUTOFF_REQUIRED')
    if (inventory['kind']!='main' or inventory['global_step']!=step
            or inventory['status']!='FULL_STATE_BYTES_INVENTORIED_NOT_INDEPENDENT_RESUME_PROOF'
            or inventory['model_loaded'] or inventory['optimizer_pickle_loaded']
            or delivery['status']!='PASS_COMPLETE_PRIVATE_CHECKPOINT_FIXED_REVISION_READBACK'
            or delivery['kind']!='main' or delivery['global_step']!=step
            or delivery['manifest_sha256']!=inventory_sha256
            or inventory['manifest_sha256']!=hopper['data_manifest_sha256']
            or inventory['statistics_sha256']!=hopper['statistics_sha256']):
        raise PermissionError('ACTUAL_COMPLETE_MAIN_CHECKPOINT_DELIVERY_REQUIRED')
    files=inventory['files']
    if (not any(n.endswith('.safetensors') for n in files)
            or any(n not in files for n in ('config.json','optimizer.pt','scheduler.pt','rng_state.pth','SAMPLER_STATE.json'))
            or any(delivery['files'].get(n)!=row for n,row in files.items())):
        raise PermissionError('ONE_FULL_DELIVERED_MODEL_AND_RESUME_INVENTORY_REQUIRED')
    identity=dict(profile='B1',model_revision=delivery['revision'],source_sha=hopper['source_sha'],
        config_sha256=files['config.json']['sha256'],processor_sha256=processor_sha256,
        statistics_sha256=inventory['statistics_sha256'])
    validate_identity(identity)
    rows=panel['panels'][name]
    if len(rows)!=(4 if step==512 else 8) or len({r['original_attempt_id'] for r in rows})!=len(rows):
        raise PermissionError('EXACT_REGISTERED_DEVELOPMENT_OR_DEV_DENOMINATOR')
    byid={c['id']:c for c in cards};pairs={};cases={}
    for row in rows:
        original=byid[row['original_attempt_id']]
        if (digest(original)!=row['original_card_sha256']
                or original['initial_state_sha256']!=row['initial_state_sha256']
                or original['scene_group']!=row['scene_group'] or original['task']!=row['task']
                or original['split']!=('TRAIN' if step==512 else 'DEV')):
            raise PermissionError('ORIGINAL_CARD_LANGUAGE_SPLIT_AND_INITIAL_STATE_CHANGED')
        pair=pairs.setdefault(original['scene_group'],[]);pair.append(original)
        card=copy.deepcopy(original)
        card['id']='PI0294-M'+str(step)+'-'+original['id'].removeprefix('PI0294-')
        episode=str(uuid.uuid4());relative='evidence/model-M'+str(step)+'/'+card['id']+'-REQUEST.json'
        request=dict(run_uuid=run_uuid,attempt_id=card['id'],source_sha=core['source_sha'],
            run_manifest_sha256=core['manifest_sha256'],initial_state_sha256=card['initial_state_sha256'],
            profile='B1',category=category,identity=identity,episode_uuid=episode,
            deadline_epoch=deadline_epoch,live_readiness_sha256=live_readiness_sha256)
        request_sha=hashlib.sha256((json.dumps(request,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
        cases[episode]=dict(run_uuid=run_uuid,episode_uuid=episode,attempt_id=card['id'],
            identity=identity,deadline_epoch=deadline_epoch,layout_id=original['scene_group'],
            core_request_relative=relative,core_request_sha256=request_sha,core_request=request,
            card=card,original_attempt_id=original['id'],hopper_session_relative='evidence/model-M'+str(step)+'/resident')
    ignored={'id','task','task_id','instruction'}
    for pair in pairs.values():
        if (len(pair)!=2 or {c['task'] for c in pair}!={'T1','T2'}
                or {k:v for k,v in pair[0].items() if k not in ignored}!={k:v for k,v in pair[1].items() if k not in ignored}
                or pair[0]['instruction']==pair[1]['instruction']):
            raise PermissionError('SAME_INITIAL_SCENE_ONLY_LANGUAGE_DIFFERS')
    checkpoint_identity=dict(files={n:r['sha256'] for n,r in files.items()},relay_identity=identity,
        statistics_sha256=inventory['statistics_sha256'],checkpoint_manifest_sha256=inventory_sha256,
        HF_revision=delivery['revision'],HF_prefix=delivery['prefix'],global_step=step,
        revision_kind='PRIVATE_HF_FULL_STATE_READY_LAST_ARTIFACT_REVISION')
    return dict(run_uuid=run_uuid,core_source_sha=core['source_sha'],hopper_source_sha=hopper['source_sha'],
        core_manifest_sha256=core['manifest_sha256'],hopper_manifest_sha256=hopper['manifest_sha256'],
        deadline_epoch=deadline_epoch,cases=cases,checkpoint_identity=checkpoint_identity,
        status='PLANNED_NOT_DISPATCHED_OR_MODEL_BEHAVIOR_QUALIFIED',category=category,
        frozen_panel_semantic_sha256=digest(panel),selected='NONE',deployment='NONE')


def teacher_action(query,teacher):
    body=unseal(query)
    if body['identity']['profile']!='TEACHER_REFERENCE' or body['kind']!='student_request' or body['protocol']!=PROTOCOL:
        raise PermissionError('ACTUAL_TEACHER_QUALIFICATION_ONLY_NOT_MODEL_POLICY')
    exact_keys(teacher,('observation_sha256','trajectory_eef10','source'))
    if teacher['observation_sha256']!=body['observation_sha256'] or teacher['source']!='ACTUAL_CURRENT_FEEDBACK_TEACHER_NOT_MODEL':
        raise ValueError('TEACHER_MUST_BIND_THIS_FRESH_OBSERVATION')
    return sealed(dict(protocol=PROTOCOL,kind='action',run_uuid=body['run_uuid'],episode_uuid=body['episode_uuid'],
        sequence=body['sequence'],tick=body['tick'],payload={k:body[k] for k in
        ('query_uuid','reference_uuid','observation_sha256','anchor_sha256','identity')}|
        dict(trajectory_eef10=teacher['trajectory_eef10'],offsets_ns=[i*50000000 for i in range(40)])))


def run_episode(directory,case,core,hopper,*,now=time.time,sleep=time.sleep):
    """Injected endpoints must implement bounded actual SSH reads/writes.

    A dispatched command is never resent after an uncertain network outcome.
    Read-only polling is allowed; generation and command sends are one-shot.
    """
    directory=Path(directory)
    exchange=RelayExchange(directory,run_uuid=case['run_uuid'],episode_uuid=case['episode_uuid'],
        identity=case['identity'],now=now())
    state=dict(status='ACTUAL_ROUTE_START_NOT_TASK_SUCCESS',episode_uuid=case['episode_uuid'],
        profile=case['identity']['profile'],fresh_observations=0,commands_sent=0,acknowledgements=0,
        network_outcome_unknown=False,model_calls_local=0,physics_integrations_local=0)
    path=directory/'ROUTE.json'
    write(path,state)
    try:
        for sequence in range(80):
            end=min(case['deadline_epoch'],now()+60)
            packet=None
            while packet is None and now()<end:
                packet=core.observation(case,sequence)
                if packet is None:sleep(.05)
            if packet is None:raise TimeoutError('NO_FRESH_CORE_OBSERVATION')
            query=exchange.observe(packet,now());state['fresh_observations']+=1
            # Durable exchange state already contains the query before network dispatch.
            state['status']='GENERATION_DISPATCH_INTENT_NO_RESEND';write(path,state)
            action=hopper.query(case,query,core.teacher(case,sequence) if case['identity']['profile']=='TEACHER_REFERENCE' else None)
            if action is None:
                stop=min(end,unseal(query)['expires'])
                while action is None and now()<stop:
                    action=hopper.response(case,unseal(query)['query_uuid'])
                    if action is None:sleep(.05)
            if action is None:raise TimeoutError('GENERATION_RESPONSE_UNKNOWN_NO_RESAMPLE')
            command=exchange.action(action,now())
            state['status']='COMMAND_DISPATCH_INTENT_NO_REPLAY';write(path,state)
            core.install(case,sequence,dict(command=command,action=action,request=query))
            state['commands_sent']+=1;state['status']='WAIT_ACTUAL_PREFIX_ACK';write(path,state)
            ack=None;end=min(case['deadline_epoch'],unseal(query)['expires'])
            while ack is None and now()<end:
                ack=core.ack(case,sequence)
                if ack is None:sleep(.05)
            if ack is None:raise TimeoutError('PHYSICAL_PROGRESS_UNKNOWN_DO_NOT_RESEND')
            exchange.acknowledge(ack,now());state['acknowledgements']+=1
            write(path,state)
        if exchange.state['phase']!='DONE' or exchange.state['tick']!=16000:
            raise ValueError('ALL_EIGHTY_ACTUAL_PREFIX_ACKS_REQUIRED')
        state['status']='PASS_ACTUAL_MESSAGE_ACK_CHAIN_PENDING_INDEPENDENT_PHYSICAL_AUDIT'
        return state
    except BaseException as error:
        state.update(status='STOPPED_PRESERVED_NO_AUTOMATIC_RETRY',
            failure=dict(type=type(error).__name__,error=str(error)),
            network_outcome_unknown=exchange.state['phase'] in ('WAIT_ACTION','WAIT_ACK'))
        raise
    finally:
        write(path,state);exchange.close()
