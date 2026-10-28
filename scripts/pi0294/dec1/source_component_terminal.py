"""Pure supplied-bytes SOURCE Teacher component audit; no native admission.

Requires the small strict-reader extraction patch and an independently pinned
owner terminal byte anchor. 80df does not yet emit that anchor: absent anchor
means UNKNOWN, never COMPLETE. No IO/model/physics/ledger writes are performed.
"""
from collections import Counter
from io import BytesIO
import hashlib
import json
import zipfile
from types import SimpleNamespace

import numpy as np
from scripts.pi0294.dec1 import source_commissioning_plan as plans
from scripts.pi0294.dec1.source_cpu_intake import PLAN_SHA256, PLAN_OWNER_RELEASE, ORIGINAL_13, ROLE
from scripts.pi0294.dec1.source_component_evidence import verify_event_bytes
from scripts.pi0294.dec1.source_component_ack import SourceComponentAcknowledger as Ack
from scripts.pi0294.dec1.source_component_ack import POLICY, PURPOSE, SCHEMA as ACK_SCHEMA, PLAN_SCHEMA, _bytes
from scripts.pi0294.dec1.mapped_ack import _hash, _int, raw_h40_hash, SEMANTICS
from scripts.pi0294.dec1.source_sync_stop import SCHEMA as STOP_SCHEMA
from scripts.pi0294.s1_preflight import VERSION as ORIGINAL_S1_VERSION
from scripts.pi0292.eval_contract import FRAME,TCP

ANCHOR_SCHEMA='DEC1_SOURCE_COMPONENT_TERMINAL_BYTE_ANCHOR_V1'
PREFIX='source_component_ack/'


def require(value, reason):
    if not value: raise ValueError(reason)


def sha(raw):
    require(type(raw) is bytes, 'EXACT_SUPPLIED_BYTES_REQUIRED')
    return hashlib.sha256(raw).hexdigest()


def document(raw):
    def unique(pairs):
        out={}
        for key,value in pairs:
            require(key not in out, 'DUPLICATE_JSON_KEY');out[key]=value
        return out
    return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('NONFINITE_JSON')))


def arrays(raw):
    # Caller bounds bytes/uncompressed NPZ in the read-only adapter; no paths.
    with np.load(BytesIO(raw),allow_pickle=False) as saved:
        return {key:saved[key].copy() for key in saved.files}


def scalar_index(value):
    require(value.shape==() and value.dtype.kind in 'iu','EXACT_INTEGER_NPZ_CLOCK_OR_DIMENSION')
    return _int(int(value))


def _attempt(condition, files, anchor_bytes, expected_anchor_sha, manifest, manifest_sha):
    require(sha(anchor_bytes)==_hash(expected_anchor_sha), 'OWNER_TERMINAL_ANCHOR_BYTES_CHANGED')
    anchor=document(anchor_bytes);cid=condition['condition_id'];runtime=manifest['code']['vla_commit']
    require(isinstance(runtime,str) and len(runtime)==40 and all(c in '0123456789abcdef' for c in runtime),
        'EXACT_RUNTIME_SOURCE_COMMIT_REQUIRED')
    require(anchor['schema']==ANCHOR_SCHEMA and anchor['condition_id']==cid
        and anchor['component_plan_sha256']==PLAN_SHA256 and anchor['manifest_sha256']==manifest_sha
        and anchor['runtime_release']==runtime and anchor['engine_sha256']==ORIGINAL_13['scripts/pi0291_arx5_data/engine.py']
        and anchor['owner_closeout_known'] is True, 'EXACT_KNOWN_OWNER_TERMINAL_ANCHOR_REQUIRED')
    require(set(anchor['files'])==set(files)-{'DEC1_SOURCE_COMPONENT_ATTEMPT.json'}
        and all(sha(files[path])==_hash(value) for path,value in anchor['files'].items()),
        'MISSING_CHANGED_OR_EXTRA_AUDIT_INPUT_BYTES')
    get=lambda path:document(files[path])
    result=get('RESULT.json');s1=get('S1_RESULT.json');row=get('DEC1_SOURCE_COMPONENT_ATTEMPT.json')
    stop=get('SOURCE_COMPONENT_STOP.json');owner=get(PREFIX+'OWNER_COMPONENT_BINDING.json')
    episode=s1['episode_id'];executor=s1['executor_id'];steps=_int(result['physics_steps'])
    require(episode==cid and result['id']==cid and result['source_sha']==runtime
        and result['card_sha256']==condition['card_sha256'] and result['fixture'] is False
        and s1['policy']==POLICY and s1['purpose']==PURPOSE and s1['interface_mode']==condition['interface']
        and s1['runtime_release']==runtime and s1['component_plan_sha256']==PLAN_SHA256
        and s1['version']==ORIGINAL_S1_VERSION and s1['execution_path']=='DEC1_SOURCE_SINGLE_POSE_MAPPED_PREFIX8'
        and s1['plan_owner_release']==PLAN_OWNER_RELEASE and s1['engineering_unknown'] is False
        and s1['events_readback_valid'] is True and s1['terminal_stop_error'] is None
        and not any(s1['execution_state']['unknown'].values())
        and not s1['execution_state'].get('protocol_ack_unknown')
        and not s1['execution_state'].get('finish_unknown') and not s1['execution_state'].get('finish_error')
        and s1['component_meter_unknown'] is False and s1['component_ack_state']['unknown'] is False,
        'RESULT_SOURCE_CONDITION_OR_KNOWN_S1_CLOSEOUT_CHANGED')
    require(row['schema']=='DEC1_SOURCE_COMPONENT_ATTEMPT_V1' and row['attempt_id']==cid
        and row['condition_id']==cid and row['interface']==condition['interface'] and row['policy']==POLICY
        and row['component_plan_sha256']==PLAN_SHA256 and row['result_sha256']==sha(files['RESULT.json'])
        and row['S1_sha256']==sha(files['S1_RESULT.json'])
        and row['terminal_anchor_sha256']==sha(anchor_bytes)
        and row['native_STOP_return_and_owner_event_confirmed'] is True
        and row['student_neural_ACK_qualification'] is False and row['GR00T_queries']==0
        and row['new_training_data'] is False and row['selected']==row['deployment']=='NONE',
        'OWNER_ATTEMPT_BINDING_OR_COMPONENT_ONLY_PURPOSE_CHANGED')
    require(owner['component_plan_sha256']==PLAN_SHA256 and owner['plan_owner_release']==PLAN_OWNER_RELEASE
        and owner['runtime_release']==runtime and owner['condition_id']==cid
        and owner['card_sha256']==condition['card_sha256'] and owner['interface_mode']==condition['interface']
        and owner['episode_id']==episode and owner['executor_id']==executor and owner['policy']==POLICY
        and owner['purpose']==PURPOSE and owner['execution_semantics']==SEMANTICS
        and owner['frame']==FRAME and owner['tcp']==TCP
        and owner['student_neural_ACK_qualification'] is False, 'EXACT_COMPONENT_OWNER_BINDING_REQUIRED')
    ea=anchor['events']
    require(ea['bytes_sha256']==sha(files['SOURCE_COMPONENT_EVENTS.jsonl'])
        and anchor['episode_id']==episode and anchor['executor_id']==executor
        and anchor['interface']==condition['interface'], 'OWNER_EVENT_BYTES_IDENTITY_ANCHOR_CHANGED')
    events=verify_event_bytes(files['SOURCE_COMPONENT_EVENTS.jsonl'],episode_id=episode,executor_id=executor,
        maximum_bytes=128*1024**2,expected_offset=ea['offset'],expected_sequence=ea['sequence'],expected_previous=ea['previous'])
    stopping=[e for e in events if e.get('kind') in ('SOURCE_STOP_ACK','SOURCE_NORMAL_STOP_ACK',
        'SOURCE_TERMINAL_STOP_ACK','SOURCE_FACTORY_STOP_ACK')]
    receipt='SOURCE_COMPONENT_STOP.json#sha256='+sha(files['SOURCE_COMPONENT_STOP.json'])
    require(len(stopping)==1 and stopping[0]==s1['actual_stop_event'] and stopping[0]['receipt_id']==receipt
        and stopping[0]['episode_id']==episode and stopping[0]['executor_id']==executor
        and stop['schema']==STOP_SCHEMA and stop['status']=='LATCHED_READBACK_RECORD_ACK_REQUIRES_OWNER_RETURN'
        and stop['episode_id']==episode and stop['executor_id']==executor and stop['stopped'] is True
        and stop['owned_live_dispatch_latched'] is True and stop['native_frame_inflight'] is False
        and stop['source_step_sha256']==anchor['engine_sha256'] and _int(stop['native_steps'])==steps
        and abs(float(stop['sim_time_s'])-steps*.002)<=1e-8
        and s1['actual_stop_state']['stop_acknowledged'] is True
        and s1['actual_stop_state']['unknown'] is False and s1['actual_stop_state']['latched'] is True,
        'ACTUAL_STOP_RETURN_AND_UNIQUE_OWNER_EVENT_REQUIRED')
    _hash(stop['native_state_sha256'])
    high=arrays(files['highrate.npz']);main=high['rows']
    require(main.ndim==2 and len(main)==steps and np.isfinite(main).all()
        and main.shape[1]==1+scalar_index(high['nq'])+scalar_index(high['nv'])+8+6+1
        and np.all(np.abs(main[:,0]-np.arange(1,steps+1)*.002)<=1e-8), 'ACTUAL_MAIN_HIGH_RATE_COUNTER_CLOCK_REQUIRED')
    finished=[e for e in events if e.get('kind')=='COMPONENT_PREFIX_FINISHED']
    reserves=[e['token'] for e in events if e.get('kind')=='COMPONENT_PREFIX_RESERVED']
    require(finished==s1['shadow_and_live_finish_records'] and len({e['token'] for e in finished})==len(finished)
        and len(set(reserves))==len(reserves) and set(reserves)=={e['token'] for e in finished}
        and all(e['reservation_retained'] is False for e in finished), 'UNKNOWN_OR_DUPLICATE_PREFIX_SETTLEMENT')
    totals=Counter()
    for end in finished:
        effects=[e for e in events if e.get('kind')=='COMPONENT_PREFIX_NATIVE_EFFECT' and e['token']==end['token']]
        require(len(effects)==_int(end['known_ticks']) and all(e['actual_increment']==1 and e['unknown'] is False
            and e['actual_lower_bound']==i+1 and e['domain']==end['domain'] for i,e in enumerate(effects)),
            'ACTUAL_PARTIAL_PREFIX_LOWER_BOUND_CHANGED')
        totals[end['domain']]+=end['known_ticks']
    require(totals['live']==steps==s1['native_live_integrations']==s1['live_integrations_lower_bound']==row['main_integrations']
        and totals['shadow']==s1['shadow_integrations']==row['shadow_integrations'], 'ACTUAL_LIVE_SHADOW_COUNTERS_CHANGED')
    complete=[e for e in events if e.get('kind')=='SOURCE_PREFIX200_ACK']
    actual=[e for e in events if e.get('kind')=='SOURCE_ACTUAL_REFERENCE_ACK']
    require(complete==s1['actual_completed_prefixes'] and actual==s1['actual_25tick_ACKs']
        and len(complete)==row['completed_prefixes'] and len(complete)<=80, 'ACTUAL_PREFIX_ACK_SUMMARY_CHANGED')
    for folder in ('acks','physics'):
        require({p for p in files if p.startswith(PREFIX+folder+'/')}=={
            PREFIX+folder+f'/{i:03d}.'+('json' if folder=='acks' else 'npz') for i in range(len(complete))},
            'EXTRA_OR_MISSING_PREFIX_ACK_PHYSICS_ARTIFACT')
    gates=[e for e in events if e.get('kind')=='SOURCE_FULL_H40_GATE']
    require(gates==s1['full_H40_gate_results'] and all(e['decision']['decision'] in ('ACCEPT','WARN','REJECT') for e in gates),
        'ORIGINAL_FULL_H40_GATE_EVIDENCE_REQUIRED')
    if any(e['decision']['decision']=='REJECT' for e in gates) or any(e['status']!='PASS' for e in finished):
        pending=get(PREFIX+f'plans/{len(complete):03d}.json')
        require(pending['schema']==PLAN_SCHEMA and pending['condition_id']==cid
            and pending['component_plan_sha256']==PLAN_SHA256 and pending['episode_id']==episode
            and pending['executor_id']==executor and pending['interface_mode']==condition['interface']
            and pending['tick_before']==len(complete)*200 and pending['policy']==POLICY
            and pending['frame']==FRAME and pending['tcp']==TCP
            and raw_h40_hash(pending['raw_H40_eef10'])==pending['raw_h40_sha256'],
            'FAILED_PREFIX_ORIGINAL_TEACHER_PREBIND_REQUIRED')
    prior=None;seen=set()
    for sequence,event in enumerate(complete):
        proof={k:v for k,v in event.items() if k!='kind'};stem=f'{sequence:03d}'
        binding=get(PREFIX+'plans/'+stem+'.json');ack_path=PREFIX+'acks/'+stem+'.json';ack=get(ack_path)
        require(binding['schema']==PLAN_SCHEMA and binding['sequence']==sequence and binding['tick_before']==sequence*200
            and binding['component_plan_sha256']==PLAN_SHA256 and binding['condition_id']==cid
            and binding['interface_mode']==condition['interface'] and binding['episode_id']==episode
            and binding['executor_id']==executor and raw_h40_hash(binding['raw_H40_eef10'])==binding['raw_h40_sha256']
            and binding['policy']==POLICY and binding['purpose']==PURPOSE and binding['frame']==FRAME and binding['tcp']==TCP
            and abs(float(binding['observation_timestamp'])-sequence*.4)<=1e-8
            and binding['offsets_s']==[i/20 for i in range(40)], 'ORIGINAL_TEACHER_PREBOUND_RAW_H40_CHANGED')
        verifier=SimpleNamespace(episode_id=episode,executor_id=executor,_prior_ack=prior)
        prior=Ack._proof(verifier,proof,binding)
        require(proof['tick_before']==sequence*200 and proof['nonce'] not in seen, 'REPEATED_OR_SKIPPED_PREFIX')
        seen.add(proof['nonce'])
        receipts=[(i,e) for i,e in enumerate(events) if e.get('kind')=='SOURCE_PROTOCOL_ACK_READBACK' and e['nonce']==proof['nonce']]
        require(len(receipts)==1, 'ACTUAL_FINAL_PROTOCOL_ACK_RETURN_REQUIRED');index,returned=receipts[0]
        verifier.read_events=lambda end=index:events[:end]
        event_sha=Ack._events(verifier,proof)
        require(ack['schema']==ACK_SCHEMA and ack['proof']==proof and ack['component_plan_sha256']==PLAN_SHA256
            and ack['condition_id']==cid and ack['interface_mode']==condition['interface']
            and ack['episode_id']==episode and ack['executor_id']==executor and ack['runtime_release']==runtime
            and ack['teacher_raw_plan_sha256']==sha(files[PREFIX+'plans/'+stem+'.json'])
            and ack['verified_source_event_list_sha256']==event_sha and ack['policy']==POLICY
            and ack['purpose']==PURPOSE and ack['student_neural_ACK_qualification'] is False
            and ack['plan_owner_release']==PLAN_OWNER_RELEASE and ack['frame']==FRAME and ack['tcp']==TCP
            and ack['execution_semantics']==SEMANTICS and ack['sequence']==sequence and ack['GR00T_queries']==0
            and returned['acknowledgement_sha256']==sha(files[ack_path])
            and returned['episode_id']==episode and returned['executor_id']==executor
            and returned['receipt_id']=='acks/'+stem+'.json', 'COMPLETE_ACTUAL_COMPONENT_ACK_READBACK_REQUIRED')
        physics_path=PREFIX+'physics/'+stem+'.npz';physics=arrays(files[physics_path])
        require(sha(files[physics_path])==ack['physics_receipt_sha256']==returned['physics_receipt_sha256']
            and set(physics)=={'rows','tick_before','tick_after'}
            and scalar_index(physics['tick_before'])==sequence*200 and scalar_index(physics['tick_after'])==(sequence+1)*200
            and np.array_equal(physics['rows'],main[sequence*200:(sequence+1)*200])
            and physics['rows'].shape==(200,main.shape[1]) and np.all(physics['rows'][:,-1]==0),
            'ACTUAL_PREFIX200_PHYSICS_BYTES_OR_MAIN_SLICE_CHANGED')
        require(sum(e['reference_id']==binding['raw_chunk_id'] for e in gates)==17
            and all(sum(e['reference_id']==r['command_id'] for e in gates)==2 for r in proof['actual_reference_acks'])
            and all(e['decision']['decision'] in ('ACCEPT','WARN') for e in gates if e['reference_id'] in
                {binding['raw_chunk_id']}|{r['command_id'] for r in proof['actual_reference_acks']}),
            'ORIGINAL_RAW_AND_MAPPED_FULL_GATES_FOR_SHADOW_AND_LIVE_REQUIRED')
    rejects=[e for e in gates if e['decision']['decision']=='REJECT']
    s1rejects=[e for e in finished if e['domain']=='shadow' and e['status']=='S1_REJECT']
    require(len(s1rejects)==s1['pre_execution_rejections']==s1['known_partial_S1_rejections'], 'KNOWN_S1_REJECT_COUNT_CHANGED')
    if result['status']=='SAFETY_STOP_PRESERVED' or s1['live_native_failures']:
        require(steps>0 and bool(main[-1,-1]), 'ACTUAL_NATIVE_SAFETY_FLAG_REQUIRED');status='NATIVE_SAFETY_FAIL'
    elif rejects or s1rejects: status='KNOWN_REFERENCE_OR_S1_REJECT'
    elif (result['status']=='VALID_COMPLETE' and steps==16000 and result['rgb_pairs']==row['RGB_pairs']==641
            and totals['shadow']==16000 and len(complete)==80 and len(actual)==640
            and stopping[0]['kind']=='SOURCE_NORMAL_STOP_ACK'
            and s1['component_ack_state']['sequence']==80 and s1['component_ack_state']['pending_raw_plan'] is False
            and s1['maximum_live_shadow_difference']==max(e['maximum_live_shadow_difference'] for e in complete)):
        status='COMPONENT_COMPLETE'
    else: status='ENGINEERING_INCOMPLETE_UNKNOWN'
    expected={'COMPONENT_COMPLETE':'COMPONENT_COMPLETE_NOT_STUDENT_EVALUATION',
        'KNOWN_REFERENCE_OR_S1_REJECT':'KNOWN_COMPONENT_SAFETY_OR_REFERENCE_REJECT',
        'NATIVE_SAFETY_FAIL':'COMPONENT_NATIVE_SAFETY_FAIL',
        'ENGINEERING_INCOMPLETE_UNKNOWN':'COMPONENT_ENGINEERING_INCOMPLETE'}
    require(row['status']==expected[status], 'OWNER_TERMINAL_CLASSIFICATION_DISAGREES_WITH_BYTES')
    return dict(condition_id=cid,status=status,main_integrations=steps,shadow_integrations=totals['shadow'],
        verified_completed_prefixes=len(complete),Teacher_task_success_diagnostic=result.get('score',{}).get('persistent_success'))


def audit(plan_bytes,population_bytes,manifest_bytes,*,manifest_sha256,attempts,anchors,anchor_sha256):
    """attempts[cid] maps FIXED relative audit paths to immutable raw bytes.

    anchors are separately owner-pinned terminal readback bytes, not generated
    from the supplied event file by this auditor. See IO contract companion.
    """
    require(sha(plan_bytes)==PLAN_SHA256 and sha(manifest_bytes)==_hash(manifest_sha256), 'FIXED_PLAN_AND_MANIFEST_BYTES_REQUIRED')
    plan=plans.validate(document(plan_bytes),population_bytes,owner_release=PLAN_OWNER_RELEASE)
    manifest=document(manifest_bytes);require(manifest['scope']==ROLE,'SOURCE_COMPONENT_ROLE_REQUIRED')
    ids={c['condition_id'] for c in plan['conditions']}
    require(set(attempts)<=ids and set(anchors)<=ids and set(anchor_sha256)<=ids,'UNREGISTERED_COMPONENT_CONDITION')
    rows=[]
    for condition in plan['conditions']:
        cid=condition['condition_id'];files=attempts.get(cid,{})
        if not files and cid not in anchors: rows.append(dict(condition_id=cid,status='NOT_RUN'));continue
        try: rows.append(_attempt(condition,files,anchors[cid],anchor_sha256[cid],manifest,manifest_sha256))
        except (KeyError,ValueError,TypeError,IndexError,OSError,AttributeError,EOFError,zipfile.BadZipFile) as error:
            rows.append(dict(condition_id=cid,status='ENGINEERING_INCOMPLETE_UNKNOWN',reason=type(error).__name__+':'+str(error)))
    return dict(schema='DEC1_SOURCE_COMPONENT_INDEPENDENT_TERMINAL_AUDIT_V1',conditions=rows,
        component_counts=dict(Counter(r['status'] for r in rows)),purpose=PURPOSE,
        student_neural_ACK_qualification=False,target_qualification=False,core64_cells=0,new_training_data=False)
