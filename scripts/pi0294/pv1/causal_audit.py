"""Read-only per-condition joins of saved cameras, draws, commands and ACKs."""
import hashlib
import math
from pathlib import Path
import re

from scripts.pi0292.durable import read
from scripts.pi0292.eval_contract import gate_action
from scripts.pi0292.relay_protocol import digest,unseal
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.mp1.effect_closure import rejected_prefix
from .audit import terminal_attempt


def same_gate(original,recomputed):
    """Allow cross-platform ULPs in diagnostic metrics, not gate semantics."""
    def numeric(a,b):
        if isinstance(a,dict) and isinstance(b,dict):
            return a.keys()==b.keys() and all(numeric(a[k],b[k]) for k in a)
        if isinstance(a,list) and isinstance(b,list):
            return len(a)==len(b) and all(numeric(x,y) for x,y in zip(a,b))
        if type(a) in (int,float) and type(b) in (int,float):
            return math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
        return type(a) is type(b) and a==b
    if (original.keys()!=recomputed.keys() or original['decision']!=recomputed['decision']
            or original['provenance']!=recomputed['provenance']
            or len(original['issues'])!=len(recomputed['issues'])):return False
    for a,b in zip(original['issues'],recomputed['issues']):
        if ({k:v for k,v in a.items() if k!='observed'}!={k:v for k,v in b.items() if k!='observed'}
                or ('observed' in a)!=('observed' in b)
                or 'observed' in a and not numeric(a['observed'],b['observed'])):return False
    return numeric(original['metrics'],recomputed['metrics'])


def _names(folder,count):
    if {p.name for p in folder.glob('*.json')}!={f'{i:03}.json' for i in range(count)}:
        raise PermissionError('PV1_CAUSAL_AUDIT_EXACT_CONTIGUOUS_SAVED_SEQUENCES')


def core(root,identifier,*,engineering=False):
    if type(engineering) is not bool:raise PermissionError('PV1_EXPLICIT_ENGINEERING_READ_SIDE_ONLY')
    root=Path(root);before=terminal_attempt(root,identifier);folder=root/'episodes'/identifier
    row=read(folder/'PV1_ADJUDICATION.json');card=read(folder/'CARD.json');relay=folder/'relay'
    audit=read(root/'evidence/PV1_AUDIT_RECEIPTS'/(identifier+'.json'))
    valid=(row['status']=='ENGINEERING_INCOMPLETE' and row['pipeline_trustworthy'] is False
        if engineering else row['status'] in ('SUCCESS','VALID_TASK_FAIL') and row['pipeline_trustworthy'] is True)
    if (not valid
            or row['live_native_failures'] or row['teacher_takeover']
            or audit['status']!='PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT' or audit['adjudication']!=row):
        raise PermissionError('PV1_CAUSAL_AUDIT_VALID_TERMINAL_INDEPENDENTLY_AUDITED_CONDITION')
    ready=read(root/'plans'/card['pv1_session_label']/'REMOTE_READY.json')
    steps=row['main_integrations'];executed=steps//200
    reference_veto=row['failure_stage']=='ORIGINAL_REFERENCE_GATE_REJECT'
    s1_veto=row['failure_stage']=='S1_PRE_EXECUTION_REJECT'
    if (type(steps) is not int or not 0<=steps<=16000 or steps%200
            or row['S1_rejections']!=int(s1_veto)
            or not (reference_veto or s1_veto or steps==16000 or engineering and 0<steps<16000)):
        raise PermissionError('PV1_CAUSAL_AUDIT_EXACT_FULL_TASK_OR_ORIGINAL_VETO')
    count=executed+int(reference_veto or s1_veto)
    if not 0<count<=80:raise PermissionError('PV1_CAUSAL_AUDIT_ORIGINAL_EIGHTY_DRAW_BOUND')
    terminal_extra=None
    observations=len(list((relay/'observations').glob('*.json')))
    if engineering:
        if reference_veto or s1_veto or observations not in (count,count+1):
            raise PermissionError('PV1_ENGINEERING_ONLY_COMPLETE_ACK_PREFIXES_AND_OPTIONAL_UNUSED_OBSERVATION')
        if observations==count+1:
            packet=read(relay/'observations'/f'{count:03}.json');body=unseal(packet)
            if (body['run_uuid']!=root.name or body['sequence']!=count or body['tick']!=steps
                    or len(body['payload']['images'])!=2):
                raise PermissionError('PV1_ENGINEERING_TERMINAL_UNUSED_OBSERVATION_IDENTITY')
            for image,camera in zip(body['payload']['images'],('body_rgb','wrist_rgb')):
                if image['sha256']!=sha(folder/camera/f'{count*8:04}.png'):
                    raise PermissionError('PV1_ENGINEERING_UNUSED_OBSERVATION_ACTUAL_CAMERA_HASH')
            terminal_extra=dict(sequence=count,observation_sha256=packet['sha256'],
                episode_uuid=body['episode_uuid'],snapshot_uuid=body['payload']['snapshot_uuid'],
                payload_sha256=digest(body['payload']),anchor_sha256=digest(body['payload']['state_eef10']),
                state_eef10=body['payload']['state_eef10'],
                model_query_generated='UNKNOWN_UNTIL_NATIVE_READBACK',action_installed=False,live_integrations=0)
    for kind,n in [('observations',observations if engineering else count),('commands',executed+int(s1_veto)),
            ('intents',executed+int(s1_veto)),('acks',executed)]:_names(relay/kind,n)
    previous=None;entries=[];snapshots=set();episode=None
    for sequence in range(count):
        name=f'{sequence:03}.json';packet=read(relay/'observations'/name);observation=unseal(packet)
        if (observation['run_uuid']!=root.name or observation['sequence']!=sequence or observation['tick']!=sequence*200
                or episode is not None and observation['episode_uuid']!=episode
                or observation['payload']['snapshot_uuid'] in snapshots):
            raise PermissionError('PV1_CAUSAL_AUDIT_UNIQUE_ACTUAL_FRESH_SNAPSHOT')
        episode=observation['episode_uuid'];snapshots.add(observation['payload']['snapshot_uuid'])
        images=observation['payload']['images']
        if len(images)!=2:raise PermissionError('PV1_CAUSAL_AUDIT_ACTUAL_TWO_CAMERAS')
        for image,camera in zip(images,('body_rgb','wrist_rgb')):
            if image['sha256']!=sha(folder/camera/f'{sequence*8:04}.png'):
                raise PermissionError('PV1_CAUSAL_AUDIT_OBSERVATION_EQUALS_ORIGINAL_CAMERA_BYTES')
        entry=dict(sequence=sequence,observation_sha256=packet['sha256'],
            payload_sha256=digest(observation['payload']),anchor_sha256=digest(observation['payload']['state_eef10']),
            state_eef10=observation['payload']['state_eef10'],previous_ACK_sha256=previous,
            classification='EXECUTED_ACKED_PREFIX' if sequence<executed else
                'ORIGINAL_REFERENCE_GATE_UNINSTALLED_ZERO_LIVE' if reference_veto else 'ORIGINAL_S1_REJECTED_INSTALLED_ZERO_LIVE_PREFIX')
        if sequence==executed and reference_veto:
            reject=read(folder/'MODEL_REJECT.json')
            if reject['original_gate']['decision']!='REJECT' or (relay/f'physics-{sequence:03}.npz').exists():
                raise PermissionError('PV1_CAUSAL_AUDIT_REFERENCE_VETO_HAS_NO_PHYSICAL_EXECUTION')
            entry.update(query_uuid=reject['query_uuid'],action_sha256=reject['action_sha256'],
                request_sha256=None,original_reference_gate=reject['original_gate'],ACK_sha256=None)
        else:
            bundle=read(relay/'commands'/name);query=unseal(bundle['request']);action=unseal(bundle['action']);command=unseal(bundle['command'])
            if (query['run_uuid']!=root.name or query['previous_ack'] is None and sequence!=0
                    or (None if query['previous_ack'] is None else query['previous_ack']['sha256'])!=previous
                    or query['episode_uuid']!=episode or query['sequence']!=sequence or query['tick']!=sequence*200
                    or query['payload']!=observation['payload'] or query['observation_sha256']!=packet['sha256']
                    or query['anchor_sha256']!=entry['anchor_sha256'] or action['sequence']!=sequence
                    or command['sequence']!=sequence or command['prefix_sha256']!=digest(action['payload']['trajectory_eef10'][:8])):
                raise PermissionError('PV1_CAUSAL_AUDIT_QUERY_ACTION_PREVIOUS_ACK_AND_OBSERVATION')
            ack_packet=None
            if sequence<executed:
                ack_packet=read(relay/'acks'/name);ack=unseal(ack_packet)
                if (ack['episode_uuid']!=episode or ack['sequence']!=sequence or ack['tick']!=sequence*200
                        or ack['payload']['tick_after']!=(sequence+1)*200 or ack['payload']['integrations']!=200
                        or ack['payload']['query_uuid']!=query['query_uuid']
                        or ack['payload']['reference_uuid']!=query['reference_uuid']
                        or ack['payload']['prefix_sha256']!=command['prefix_sha256']
                        or ack['payload']['command_sha256']!=bundle['command']['sha256']
                        or ack['payload']['physics_receipt_sha256']!=sha(relay/f'physics-{sequence:03}.npz')):
                    raise PermissionError('PV1_CAUSAL_AUDIT_ORIGINAL_COMMAND_ACK_AND_ACTUAL_PHYSICS')
            else:rejected_prefix(folder,row,name,bundle)
            entry.update(query_uuid=query['query_uuid'],request_sha256=bundle['request']['sha256'],
                action_sha256=bundle['action']['sha256'],ACK_sha256=None if ack_packet is None else ack_packet['sha256'])
            if ack_packet is not None:previous=ack_packet['sha256']
        entries.append(entry)
    if (terminal_extra and (terminal_extra['episode_uuid']!=episode or terminal_extra['snapshot_uuid'] in snapshots)
            or len({v['query_uuid'] for v in entries})!=count or terminal_attempt(root,identifier)!=before):
        raise PermissionError('PV1_CAUSAL_AUDIT_UNIQUE_QUERIES_AND_UNCHANGED_TERMINAL_INPUTS')
    return dict(status='PASS_ACTUAL_ENGINEERING_PREFIX_EFFECTS_NOT_VALID_POLICY_OUTCOME' if engineering
            else 'PASS_ACTUAL_TERMINAL_CORE_FRESH_CAMERA_COMMAND_ACK_CHAIN',attempt_id=identifier,
        episode_uuid=episode,card=card,adjudication=row,raw_hashes=before,entries=entries,
        model_ready=ready,
        executed_ACKs=executed,draws=count,raw_camera_hashes_matched=count*2,
        new_integrations=0,new_model_queries=0,terminal_unused_observation=terminal_extra)


def hopper(root,core_proof):
    import numpy as np
    root=Path(root);card=core_proof['card'];label=card['pv1_session_label'];folder=root/'evidence'/label/'resident'
    model=read(folder/'STATE.json');registry=read(root/'plans'/label/'REGISTRY.json')
    row=core_proof['adjudication'];episode=core_proof['episode_uuid'];identity=model['identity']
    case=registry['cases'][episode]
    if (model['run_uuid']!=root.name or registry['run_uuid']!=root.name or registry['job_id']!=model['job_id']
            or model['status'] not in ('PV1_NATIVE_MODEL_READY','PV1_NATIVE_PANEL_TERMINAL_OUTCOMES_AUDITED_SEPARATELY')
            or registry['checkpoint_identity']!=identity or identity['label']!=row['model']
            or identity!=core_proof['model_ready']['identity']
            or model['job_id']!=core_proof['model_ready']['job_id']
            or case['attempt_id']!=row['attempt_id'] or case['card']!=card
            or case['layout_id']!=card['scene_group']+'/'+card['task']):
        raise PermissionError('PV1_CAUSAL_AUDIT_ACTUAL_FROZEN_NATIVE_SHARED_CHECKPOINT_AND_CASE')
    proof=model['native_pre_model_qualification']
    allocation=[p for p in (root/'pbs').glob('alloc-*/ALLOCATION_ADMISSION.json') if read(p)['job_id']==model['job_id']]
    if (len(allocation)!=1 or proof['status']!='PASS_SAME_JOB_NATIVE_QUALIFICATION_READBACK'
            or sha(allocation[0].parent/'PV1_NATIVE_ENDURANCE_AND_TRANSPORT.json')!=proof['receipt_sha256']):
        raise PermissionError('PV1_CAUSAL_AUDIT_SAME_JOB_ORIGINAL_QUALIFICATION_RECEIPT')
    state=read(folder/'QUERY_STATE.json');draws=[v for v in state['queries'] if v['episode_uuid']==episode]
    entries=list(core_proof['entries']);extra=core_proof.get('terminal_unused_observation')
    engineering=core_proof['status']=='PASS_ACTUAL_ENGINEERING_PREFIX_EFFECTS_NOT_VALID_POLICY_OUTCOME'
    tail=int(engineering and extra is not None and len(draws)==core_proof['draws']+1)
    if tail:
        candidates=[v for v in draws if v.get('sequence')==extra['sequence']]
        if len(candidates)!=1 or 'response_sha256' not in candidates[0]:
            raise PermissionError('PV1_ENGINEERING_ONLY_ONE_COMPLETE_GENERATED_UNINSTALLED_TAIL')
        previous=entries[-1]['ACK_sha256'] if entries else None
        entries.append(dict(extra,previous_ACK_sha256=previous,query_uuid=candidates[0]['query_uuid'],
            action_sha256=candidates[0]['response_sha256'],request_sha256=None,ACK_sha256=None,
            classification='ENGINEERING_GENERATED_UNINSTALLED_ZERO_LIVE_TAIL'))
    if (len(draws)!=core_proof['draws']+tail or len({v['query_uuid'] for v in draws})!=len(draws)
            or state['episodes'][episode]['next_sequence']!=len(draws)
            or any('error' in v or 'response_sha256' not in v for v in draws)):
        raise PermissionError('PV1_CAUSAL_AUDIT_ALL_AND_ONLY_COMPLETED_NATIVE_CONDITION_DRAWS')
    draws={v['query_uuid']:v for v in draws};records=[]
    for entry in entries:
        key=entry['query_uuid'];query_packet=read(folder/'inbox'/(key+'.json'));query=unseal(query_packet)
        action_packet=read(folder/'outbox'/(key+'.json'));action=unseal(action_packet)
        receipt=read(folder/'receipts'/(key+'.json'));record=draws[key]
        if (query['run_uuid']!=root.name or action['run_uuid']!=root.name or action['episode_uuid']!=episode
                or query['episode_uuid']!=episode or query['query_uuid']!=key or query['sequence']!=entry['sequence']
                or query['tick']!=entry['sequence']*200 or query['identity']!=identity['relay_identity']
                or query['observation_sha256']!=entry['observation_sha256']
                or query['anchor_sha256']!=entry['anchor_sha256'] or digest(query['payload'])!=entry['payload_sha256']
                or (None if query['previous_ack'] is None else query['previous_ack']['sha256'])!=entry['previous_ACK_sha256']
                or entry['request_sha256'] is not None and query_packet['sha256']!=entry['request_sha256']
                or action_packet['sha256']!=entry['action_sha256'] or action_packet['sha256']!=record['response_sha256']
                or action_packet['sha256']!=receipt['action_sha256'] or action['sequence']!=entry['sequence']
                or record['sequence']!=entry['sequence'] or record['episode_uuid']!=episode
                or any(action['payload'][k]!=query[k] for k in ('query_uuid','reference_uuid','observation_sha256','anchor_sha256','identity'))
                or sha(folder/'receipts'/(key+'.npz'))!=receipt['arrays_sha256']):
            raise PermissionError('PV1_CAUSAL_AUDIT_ORIGINAL_NATIVE_INPUT_OUTPUT_AND_PREVIOUS_EXECUTION')
        with np.load(folder/'receipts'/(key+'.npz'),allow_pickle=False) as arrays:
            if not np.array_equal(arrays['physical'],np.asarray(action['payload']['trajectory_eef10'])):
                raise PermissionError('PV1_CAUSAL_AUDIT_ACTION_EQUALS_ORIGINAL_NATIVE_ARRAY')
        namespace=card['noise_namespace'];seed=int.from_bytes(hashlib.sha256(
            (namespace+'/'+case['layout_id']+'/'+str(entry['sequence'])).encode()).digest()[:8],'big')%(2**63)
        noise=receipt['noise'];precision=receipt['inference_precision']
        if (receipt['seed']!=seed or noise['shape']!=[1,40,132] or noise['dtype']!='torch.float32'
                or not re.fullmatch('[0-9a-f]{64}',noise['sha256'])
                or precision['mode']!='ORIGINAL_E0_BF16_AUTOCAST_FP32_MATMUL_HIGHEST'
                or precision['bf16_autocast'] is not True or precision['cuda_matmul_allow_tf32'] is not False):
            raise PermissionError('PV1_CAUSAL_AUDIT_FIXED_NOISE_SEED_AND_ORIGINAL_PRECISION')
        if entry['classification']=='ORIGINAL_REFERENCE_GATE_UNINSTALLED_ZERO_LIVE':
            gate=gate_action(action['payload']['trajectory_eef10'],entry['state_eef10'],query['tick']*.002,query['reference_uuid'])
            if not same_gate(entry['original_reference_gate'],gate) or gate['decision']!='REJECT':
                raise PermissionError('PV1_CAUSAL_AUDIT_ORIGINAL_REFERENCE_VETO_RECOMPUTED_FROM_SAVED_ACTION')
        records.append(dict(sequence=entry['sequence'],query_uuid=key,seed=seed,noise_sha256=noise['sha256'],
            action_sha256=action_packet['sha256'],arrays_sha256=receipt['arrays_sha256'],classification=entry['classification']))
    return dict(status='PASS_ACTUAL_NATIVE_GENERATION_FIXED_NOISE_AND_CAUSAL_ACTION_CHAIN',
        attempt_id=row['attempt_id'],job_id=model['job_id'],identity=identity,draws=len(records),
        records=records,executed_ACKs=core_proof['executed_ACKs'],Teacher_takeover=False,
        engineering_generated_uninstalled_draws=tail,
        terminal_unused_observation_model_query_generated=bool(tail) if engineering and extra is not None else None,
        new_integrations=0,new_model_queries=0,noise_RNG_reexecuted=False)


def _join(core_proof,hopper_proof,engineering):
    expected='PASS_ACTUAL_ENGINEERING_PREFIX_EFFECTS_NOT_VALID_POLICY_OUTCOME' if engineering else 'PASS_ACTUAL_TERMINAL_CORE_FRESH_CAMERA_COMMAND_ACK_CHAIN'
    if (core_proof['status']!=expected
            or hopper_proof['status']!='PASS_ACTUAL_NATIVE_GENERATION_FIXED_NOISE_AND_CAUSAL_ACTION_CHAIN'
            or core_proof['attempt_id']!=hopper_proof['attempt_id']
            or core_proof['draws']+(hopper_proof.get('engineering_generated_uninstalled_draws',0) if engineering else 0)!=hopper_proof['draws']
            or core_proof['executed_ACKs']!=hopper_proof['executed_ACKs']
            or core_proof['model_ready']['identity']!=hopper_proof['identity']):
        raise PermissionError('PV1_CAUSAL_AUDIT_EXACT_TWO_SITE_READ_SIDE_PROOFS')
    return dict(status='PASS_INDEPENDENT_ACTUAL_ENGINEERING_EFFECT_CLOSURE_NOT_POLICY_RESULT' if engineering
            else 'PASS_INDEPENDENT_ACTUAL_TWO_SITE_CAUSAL_CHAIN',
        attempt_id=core_proof['attempt_id'],adjudication_sha256=core_proof['raw_hashes']['PV1_ADJUDICATION.json'],
        identity=hopper_proof['identity'],draws=hopper_proof['draws'],executed_ACKs=hopper_proof['executed_ACKs'],
        core_proof_sha256=digest(core_proof),hopper_proof_sha256=digest(hopper_proof),
        new_integrations=0,new_model_queries=0)


def join(core_proof,hopper_proof):return _join(core_proof,hopper_proof,False)


def join_engineering(core_proof,hopper_proof):
    row=core_proof['adjudication']
    if row['status']!='ENGINEERING_INCOMPLETE' or row['pipeline_trustworthy'] is not False:
        raise PermissionError('PV1_ENGINEERING_EFFECT_CLOSURE_CANNOT_RELABEL_POLICY_OUTCOME')
    result=_join(core_proof,hopper_proof,True)
    tail=hopper_proof.get('engineering_generated_uninstalled_draws',0)
    if tail not in (0,1) or tail and (not core_proof.get('terminal_unused_observation')
            or hopper_proof['records'][-1]['classification']!='ENGINEERING_GENERATED_UNINSTALLED_ZERO_LIVE_TAIL'
            or core_proof['terminal_unused_observation']['action_installed'] is not False
            or core_proof['terminal_unused_observation']['live_integrations']!=0):
        raise PermissionError('PV1_ENGINEERING_TAIL_MUST_HAVE_EXACT_UNUSED_OBSERVATION_AND_ZERO_LIVE_EFFECTS')
    return dict(result,policy_outcome_valid=False,replacement_admitted=False,
        engineering_generated_uninstalled_draws=tail,executed_draws=core_proof['executed_ACKs'])
