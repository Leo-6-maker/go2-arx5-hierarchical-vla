"""Offline physics-feedback records: measured / innovation / forecast.

This is a record-consistency prototype, not a runtime, WM, policy-input adapter,
legality oracle or execution grant. It reads no files and runs no model/FK or
physics. Saved producer/trace receipts must still be externally authenticated.
The native ActualState inputs are simulator measurements, not RGB deployment.
"""
import copy
import hashlib
import json

import numpy as np

from scripts.pi0292.eval_contract import matrix10
from scripts.pi0292.relay_protocol import digest as document_hash
from scripts.pi0294.dec1.target_observation import rigid
from scripts.pi0294.task_gripper_contract import require_working_target
from ..trajectory.so3 import matrix_to_quat, quat_to_matrix, quat_slerp
from .target_reference import require_install_clock, require_live_window, trajectory_identity
from .fusion_reference_qualification_native import require_original_install_ack

SCHEMA = 'WM02_FUSION_PHYSICS_FEEDBACK_PACKET_V1'
PREDICTION_SCHEMA = 'WM02_FUSION_SAVED_COMMAND_PREDICTION_V1'
CONDITION_SCHEMA = 'WM02_FUSION_ATOM_COMMAND_CONDITION_V1'
DT, ATOM_TICKS, ATOM_S = .002, 25, .05
IDENTITY_KEYS = {'run_id','native_run_id','native_spec_sha256','executor_version',
                 'scene_sha256','frame_id','tcp_semantic_id'}
FIELDS = ('tcp','base_pose','finger_q','object_pose')
COMMAND_DOMAINS = ('UPSTREAM_INSTALLED_REFERENCE','SOLVER_TARGET_CONFIRMED','CONTROLLER_SELECTED_POLICY')
DISCONTINUITIES = {'CANCEL','COMMAND_CHANGED','REFERENCE_CHANGED','PREFIX_TTL_EXPIRED',
                   'REFERENCE_EXPIRED','RESET','TAKEOVER','PARTIAL_INTERVAL','STOP','HOLD'}


def need(ok, reason):
    if not ok:
        raise ValueError('PHYSICS_FEEDBACK_' + reason)


def _hash(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:_plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_plain(v) for v in value]
    return value


def state_sha256(state):
    """Canonical measured-state document hash, not a raw-packet/file SHA."""
    return document_hash(_plain(state))


def _vector(value,n):
    a=np.asarray(value,float)
    need(a.shape==(n,) and np.isfinite(a).all(), 'FINITE_VECTOR_'+str(n))
    return a


def _step(state):
    step=state['native_step'];time=state['time_s']
    need(type(step) is int and step>=0 and type(time) in (int,float) and np.isfinite(time)
         and abs(time-step*DT)<=1e-8, 'ACTUAL_ABSOLUTE_NATIVE_CLOCK')
    return step


def _identity(value):
    need(isinstance(value,dict) and set(value)==IDENTITY_KEYS
         and all(isinstance(value[k],str) and value[k] for k in IDENTITY_KEYS)
         and all(_hash(value[k]) for k in ('native_spec_sha256','executor_version','scene_sha256')),
         'EXPLICIT_RUN_EXECUTOR_FRAME_TCP_IDENTITIES')
    return copy.deepcopy(value)


def _base(state):
    q=_vector(state['qpos'],34);rotation=q[3:7]
    need(abs(np.linalg.norm(rotation)-1)<=1e-6, 'ACTUAL_BASE_WXYZ_QUATERNION')
    p=np.eye(4);p[:3,3]=q[:3];p[:3,:3]=quat_to_matrix(rotation[[1,2,3,0]])
    return p


def _pose_error(reference,actual):
    reference,actual=rigid(reference),rigid(actual)
    relative=np.eye(4)
    relative[:3,:3]=reference[:3,:3].T@actual[:3,:3]
    relative[:3,3]=reference[:3,:3].T@(actual[:3,3]-reference[:3,3])
    q=matrix_to_quat(relative[:3,:3])
    if q[3]<0:q=-q
    n=float(np.linalg.norm(q[:3]));angle=2*np.arctan2(n,q[3])
    rotvec=np.zeros(3) if n<1e-15 else q[:3]*(angle/n)
    return dict(translation_world_m=(actual[:3,3]-reference[:3,3]).tolist(),
        translation_reference_frame_m=relative[:3,3].tolist(),
        rotation_reference_frame_rad=rotvec.tolist(),rotation_error_rad=float(angle),
        T_reference_from_actual=relative.tolist(),convention='INV_REFERENCE_TIMES_ACTUAL_NO_EULER_SUBTRACTION')


def planned_condition(*,program_sha256,program_start_native_step,native_start_step,
                      command_EEF10_BASE7,interval_end_EEF10_BASE7,
                      effective_reference_H40,reference_subatom_phase_s):
    """A planned command condition, never an observed response or permission.

    Content plus global atom index survive a proven same-program wire renewal.
    Local wire IDs/source clocks remain separate execution evidence.
    """
    need(_hash(program_sha256) and type(program_start_native_step) is int
         and type(native_start_step) is int and program_start_native_step>=0
         and native_start_step>=program_start_native_step
         and (native_start_step-program_start_native_step)%ATOM_TICKS==0, 'EXACT_PROGRAM_ATOM_IDENTITY')
    first,last=_vector(command_EEF10_BASE7,17),_vector(interval_end_EEF10_BASE7,17)
    rigid(matrix10(first[:10]));rigid(matrix10(last[:10]))
    require_working_target([first[9],last[9]])
    preview=np.asarray(effective_reference_H40,float)
    need(preview.shape==(40,10) and np.isfinite(preview).all()
         and np.array_equal(preview[0],first[:10]) and np.array_equal(preview[1],last[:10]),
         'FULL_CONDITIONED_H40_SUFFIX_NOT_ONLY_CURRENT_COMMAND')
    require_working_target(preview[:,9])
    need(type(reference_subatom_phase_s) in (int,float) and np.isfinite(reference_subatom_phase_s)
         and 0<=reference_subatom_phase_s<=.025+1e-8, 'EXPLICIT_ORIGINAL_SOURCE_CLOCK_SUBATOM_PHASE')
    return dict(schema=CONDITION_SCHEMA,program_sha256=program_sha256,
        program_start_native_step=program_start_native_step,native_start_step=native_start_step,
        native_end_step=native_start_step+ATOM_TICKS,
        global_atom_index=(native_start_step-program_start_native_step)//ATOM_TICKS,
        command_EEF10_BASE7=first.tolist(),interval_end_EEF10_BASE7=last.tolist(),
        effective_reference_H40_sha256=hashlib.sha256(preview.astype('<f8').tobytes()).hexdigest(),
        reference_subatom_phase_s=round(float(reference_subatom_phase_s),10),
        gripper_clock='ISSUED_FIRST_ROW_HELD_DURING_ATOM_NOT_NEXT_ROW',
        condition_is_measured_outcome=False)


def _wire_pose(part,elapsed):
    points=part['waypoints']
    need(part['frame']=='world' and part.get('start_delay',0)==0 and part.get('lead_in',0)==0
         and len(points)==41 and all(abs(p['time']-i*ATOM_S)<=1e-10 for i,p in enumerate(points)),
         'ORIGINAL_41_POINT_ABSOLUTE_WORLD_REFERENCE')
    need(0<=elapsed<=40*ATOM_S, 'REFERENCE_TIME_INSIDE_INSTALLED_HORIZON')
    i=min(int(np.floor(elapsed/ATOM_S)),39);u=(elapsed-i*ATOM_S)/ATOM_S
    a,b=points[i],points[i+1]
    qa,qb=_vector(a['quaternion_xyzw'],4),_vector(b['quaternion_xyzw'],4)
    need(abs(np.linalg.norm(qa)-1)<=1e-6 and abs(np.linalg.norm(qb)-1)<=1e-6,
         'ORIGINAL_UNIT_REFERENCE_QUATERNIONS')
    p=np.eye(4);p[:3,3]=(1-u)*_vector(a['position'],3)+u*_vector(b['position'],3)
    p[:3,:3]=quat_to_matrix(quat_slerp(qa,qb,u))
    return rigid(p)


def _consumption(record,identity):
    w,p=record['window'],record['proposed']
    start,end=w['native_start_step'],w['native_end_step']
    need(type(start) is int and type(end) is int and end-start==w['actual_ticks']==ATOM_TICKS,
         'ONE_COMPLETE_ACTUAL_ATOM')
    wire=p['wire'];pin=hashlib.sha256(wire.encode()).hexdigest();doc=json.loads(wire);meta=doc['_pi0294']
    need(pin==p['wire_sha256']==w['reference_sha256'] and _hash(w['target_trajectory_sha256'])
         and meta['episode']==identity['native_run_id'] and meta['operation']=='install'
         and meta['ttl_s']==.4 and meta['source_mpc_s']==p['source_mpc_s'], 'SAME_ORIGINAL_INSTALLED_WIRE')
    require_original_install_ack(p)
    installed=[r for r in p['receiver_events'] if r['kind']=='installed' and r.get('ref_id')==pin]
    need(w['target_trajectory_sha256'] in {trajectory_identity(r['times'],r['poses'],r['inputs']) for r in installed},
         'ACTUAL_INSTALLED_TARGET_ARRAYS_NOT_ONLY_A_PROPOSED_WIRE')
    receiver_events=record.get('reference_receiver_events',p['receiver_events'])
    sent_identities={}
    for event in receiver_events if isinstance(receiver_events,list) else ():
        try:
            if event.get('kind')=='installed' and event.get('ref_id')==pin:
                key=trajectory_identity(event['times'],event['poses'],event['inputs'])
                sent_identities.setdefault(key,[]).append(event.get('monotonic_s'))
        except (ValueError,KeyError,TypeError,AttributeError):pass
    install=p['native_step'];source=p['source_mpc_s']
    need(type(install) is int and start>=install and (start-install)%25==0, 'ACTUAL_LOCAL_CONSUMED_INDEX')
    require_install_clock(source,install*DT)
    clock=require_live_window(source,start*DT,ATOM_TICKS)
    need(abs(w['clock']['source_mpc_s']-source)<=1e-12
         and abs(w['clock']['native_start_s']-start*DT)<=1e-8
         and abs(w['clock']['native_end_s']-end*DT)<=1e-8, 'ORIGINAL_SOURCE_CLOCK_JOIN')
    matches=w.get('solver_policies_with_exact_installed_target',[])
    solver_confirmed=(isinstance(matches,list) and bool(matches) and all(isinstance(r,dict)
        and _hash(r.get('trajectory_sha256'))
        and type(r.get('receiver_send_monotonic_s')) in (int,float)
        and type(r.get('received_monotonic_s')) in (int,float)
        and np.isfinite([r['receiver_send_monotonic_s'],r['received_monotonic_s']]).all()
        and r['receiver_send_monotonic_s'] in sent_identities.get(r['trajectory_sha256'],())
        and 0<=r['receiver_send_monotonic_s']<=r['received_monotonic_s'] for r in matches))
    full=np.asarray(p['action_EEF10'],float);saved=np.asarray(w['full_final_installed_H40'],float)
    index=(start-install)//25
    need(full.shape==saved.shape==(40,10) and np.isfinite(full).all()
         and np.array_equal(full,saved) and 0<=index<40, 'EXACT_FULL_INSTALLED_H40_AND_INDEX')
    require_working_target(full[:,9])
    row=_vector(w['actual_installed_consumed_EEF10_BASE7'],17)
    need(np.array_equal(row[:10],full[index]), 'ACTUAL_CONSUMED_ROW_NOT_A_PROPOSED_CANDIDATE')
    # Audit the serialized EEF points against the original already-decoded H40.
    poses=matrix10(full)
    need(all(np.allclose(_wire_pose(doc['ee'],i*ATOM_S),poses[min(i,39)],rtol=0,atol=1e-10)
             for i in range(41)), 'WIRE_AND_RETAINED_H40_AGREE')
    need('base' in doc and all(np.allclose(np.r_[x['position'],x['quaternion_xyzw']],row[10:],rtol=0,atol=1e-12)
         for x in doc['base']['waypoints']), 'SAME_INSTALLED_FIXED_BASE7')
    grip=w['gripper'];sent=grip['submitted'];memory=grip['consumed_wrapper'];values=memory['values']
    require_install_clock(sent['source_mpc_s'],start*DT)
    need(grip['command_consumed'] is True and memory['status']=='ACTUAL_OWNED_CHILD_GLOBALS_READ_ONLY'
         and type(memory['pid']) is int and memory['pid']>0 and _hash(memory['library_sha256'])
         and values['initialized'] is True and values['has_command'] is True
         and values['steps']==end and type(sent['sequence']) is int and sent['sequence']>=0
         and values['sequence']==sent['sequence']
         and abs(sent['opening']-row[9])<=1e-12 and abs(values['opening']-sent['opening'])<=1e-12
         and abs(values['expiry']-sent['expiry'])<=1e-12
         and abs(sent['expiry']-sent['source_mpc_s']-.4)<=1e-9 and sent['native_start_step']==start
         and end*DT<=sent['expiry']+1e-9, 'SAME_REAL_GRIPPER_COMMAND_CONSUMPTION')
    next_row=np.r_[full[min(index+1,39)],row[10:]]
    program=record.get('program')
    condition=None if program is None else planned_condition(program_sha256=program['sha256'],
        program_start_native_step=program['start_native_step'],native_start_step=start,
        command_EEF10_BASE7=row,interval_end_EEF10_BASE7=next_row,
        effective_reference_H40=full[np.minimum(index+np.arange(40),39)],
        reference_subatom_phase_s=max(0.,start*DT-source-index*ATOM_S))
    return dict(condition=condition,reference_id=pin,actual_consumed_index=index,
        target_trajectory_sha256=w['target_trajectory_sha256'],reference_age_s=end*DT-source,
        source_mpc_s=source,source_clock=clock,reference_TCP=_wire_pose(doc['ee'],end*DT-source),
        reference_BASE=_wire_pose(doc['base'],end*DT-source),gripper_command_fraction=float(sent['opening']),
        native_start_step=start,native_end_step=end,
        gripper_sequence=sent['sequence'],
        gripper_wrapper_identity=dict(pid=memory['pid'],library_sha256=memory['library_sha256']),
        command_domain_evidence=dict(UPSTREAM_INSTALLED_REFERENCE=True,
            SOLVER_TARGET_CONFIRMED=bool(solver_confirmed),CONTROLLER_SELECTED_POLICY=False),
        controller_selected_reference='DERIVED_ONLY_NO_SELECTED_POLICY_ID_EXPORTED')


def command_condition(consumption,*,identity):
    """Join an actual window/proposed wire to its planned global atom content."""
    result=_consumption(consumption,_identity(identity))
    need(result['condition'] is not None, 'EXPLICIT_PINNED_GLOBAL_PROGRAM_IDENTITY_REQUIRED')
    return result['condition']


def _optional(values,key,reader):
    if key not in values or values[key] is None:return None,False
    try:return reader(values[key]),True
    except (ValueError,TypeError,KeyError,IndexError):return None,False


def _actual_value(state,name):
    if name=='base_pose':return _base(state)
    if name=='finger_q':return _vector(state[name],2)
    return rigid(state[name])


def _measured(before,after,command):
    values,mask={},{}
    for name in FIELDS:
        try:value=_actual_value(after,name);valid=True
        except (ValueError,TypeError,KeyError,IndexError):value=None;valid=False
        values[name]=None if value is None else value.tolist();mask[name]=valid
    try:
        delta=_pose_error(rigid(before['tcp']),rigid(after['tcp']))
        rotation=np.asarray(delta['rotation_reference_frame_rad'])/ATOM_S
        values['velocity']=dict(linear_world_m_s=(np.asarray(delta['translation_world_m'])/ATOM_S).tolist(),
            angular_previous_TCP_frame_rad_s=rotation.tolist(),method='TWO_ACTUAL_SAMPLES_FIXED_50MS')
        mask['velocity']=True
    except (ValueError,TypeError,KeyError,IndexError):values['velocity']=None;mask['velocity']=False
    tracking={}
    for name,field in (('tcp','reference_TCP'),('base_pose','reference_BASE')):
        valid=command is not None and mask[name]
        tracking[name]=_pose_error(command[field],_actual_value(after,name)) if valid else None
        mask['tracking_'+name]=valid
    if command is not None and mask['finger_q']:
        tracking['finger_error_m']=(np.asarray(values['finger_q'])-command['gripper_command_fraction']*.044).tolist()
        mask['tracking_gripper']=True
    else:tracking['finger_error_m']=None;mask['tracking_gripper']=False
    values['tracking_error']=tracking
    values['reference_age_s']=None if command is None else command['reference_age_s'];mask['reference_age_s']=command is not None
    values['events']={k:after[k] if type(after.get(k)) is bool else None
        for k in ('robot_object_contact','left_object_contact','right_object_contact','native_grasp','safety_valid','native_abort','reset','takeover')}
    mask['events']={k:v is not None for k,v in values['events'].items()}
    return dict(values=values,validity_mask=mask,
        information_origin='ACTUAL_NATIVE_SIMULATOR_STATE_AND_INSTALLED_REFERENCE',
        deployable_vs_privileged={k:('EXECUTION_METADATA' if k=='reference_age_s' else 'SIMULATOR_PRIVILEGED') for k in values},
        RGB_deployment_qualified=False,tracking_is_prediction_innovation=False,
        tracking_command_domain='UPSTREAM_INSTALLED_REFERENCE',
        command_domain_evidence={key:bool(command is not None and command['command_domain_evidence'][key])
            for key in COMMAND_DOMAINS},
        tracking_reference='INSTALLED_WIRE_AT_ACTUAL_OBSERVATION_TIME_NOT_SAME_INDEX_NEXT_DESIRED')


def _predicted_values(record):
    predicted=record.get('predicted',{});values={};masks={}
    need(isinstance(predicted,dict), 'PREDICTED_FIELDS_DICTIONARY')
    for name in FIELDS:
        reader=(lambda v:_vector(v,2)) if name=='finger_q' else rigid
        value,valid=_optional(predicted,name,reader)
        values[name]=None if value is None else value.tolist();masks[name]=valid
    events=predicted.get('event_probabilities',{})
    need(isinstance(events,dict), 'PREDICTED_EVENT_PROBABILITIES_DICTIONARY')
    values['event_probabilities']={};masks['event_probabilities']={}
    for name in ('contact','grasp','release'):
        value=events.get(name);valid=type(value) in (int,float) and np.isfinite(value) and 0<=value<=1
        values['event_probabilities'][name]=float(value) if valid else None
        masks['event_probabilities'][name]=bool(valid)
    return values,masks


def _innovation(prediction,identity,before,after,current_command,consumed,lifecycle,timing):
    reasons=[];record=None;pin=None
    if prediction is None:reasons.append('NO_SAVED_PREDICTION')
    else:
        try:
            raw=prediction['bytes'];pin=prediction['sha256']
            need(type(raw) is bytes and _hash(pin) and hashlib.sha256(raw).hexdigest()==pin,'PREDICTION_ARTIFACT_BYTES')
            parsed=json.loads(raw)
            document_hash(parsed)  # Reject non-finite JSON before retaining any metadata.
            record=parsed;receipt=prediction['persist_receipt']
            need(isinstance(record,dict) and record['schema']==PREDICTION_SCHEMA and record['identity']==identity
                 and _hash(record['model_sha256']) and _hash(record['issue_state_sha256']) and _hash(record['history_sha256'])
                 and record['future_response_observed'] is False,'SAME_SAVED_MODEL_RUN_EXECUTOR_FRAME')
            issue,target=record['issue_native_step'],record['target_native_step']
            need(type(issue) is int and type(target) is int and target>issue and (target-issue)%25==0
                 and target==after['native_step'] and abs(record['issue_time_s']-issue*DT)<=1e-8
                 and abs(record['target_time_s']-target*DT)<=1e-8
                 and isinstance(record['history_span'],list) and len(record['history_span'])==2
                 and all(type(s) is int for s in record['history_span'])
                 and record['history_span'][1]==issue and 0<=record['history_span'][0]<=issue,
                 'EXACT_PREDICTION_ISSUE_HISTORY_AND_TARGET_CLOCK')
            need(receipt['prediction_sha256']==pin and receipt['durability']=='ORIGINAL_FSYNC_COMPLETED_BEFORE_EFFECT'
                 and type(receipt['native_step_before']) is int and type(receipt['native_step_after']) is int
                 and receipt['native_step_before']==receipt['native_step_after']==issue
                 and _hash(receipt['producer_source_sha256']) and _hash(timing['trace_sha256'])
                 and timing['interval_start_native_step']==issue and timing['observation_native_step']==target
                 and all(type(x) is int and x>=0 for x in (receipt['persisted_monotonic_ns'],
                     timing['first_effect_monotonic_ns'],timing['observation_monotonic_ns']))
                 and receipt['persisted_monotonic_ns']<timing['first_effect_monotonic_ns']<=timing['observation_monotonic_ns'],
                 'SAVED_PREDICTION_PRECEDES_ACTUAL_EFFECT_NOT_JUST_FALSE_FUTURE_FLAG')
            need(lifecycle is not None and lifecycle['native_interval']==[issue,target]
                 and _hash(lifecycle['artifact_sha256']), 'COMPLETE_LIFECYCLE_INTERVAL_EVIDENCE')
            need(all(type(e['native_step']) is int and e['kind'] in DISCONTINUITIES|{'SAME_PROGRAM_RENEWAL'}
                     and issue<=e['native_step']<=target
                     for e in lifecycle['events']), 'KNOWN_COMPLETE_LIFECYCLE_EVENT_TYPES')
            need(not any(issue<=e['native_step']<=target and e['kind'] in DISCONTINUITIES for e in lifecycle['events']),
                 'COMMAND_CHANGE_CANCEL_EXPIRY_RESET_OR_PARTIAL_INVALIDATES_OLD_PREDICTION')
            commands=[_consumption(row,identity) for row in consumed]
            domain=record['conditioning_command_domain']
            need(domain in COMMAND_DOMAINS, 'EXPLICIT_MODEL_CONDITIONING_COMMAND_DOMAIN')
            need(domain!='CONTROLLER_SELECTED_POLICY', 'CONTROLLER_SELECTED_POLICY_ID_NOT_EXPORTED')
            need(all(row['command_domain_evidence'][domain] for row in commands),
                 'ACTUAL_COMMAND_LEDGER_MUST_PROVE_DECLARED_MODEL_CONDITIONING_DOMAIN')
            actual=[row['condition'] for row in commands]
            need(actual and all(x is not None for x in actual)
                 and [x['native_start_step'] for x in actual]==list(range(issue,target,25))
                 and document_hash(actual)==document_hash(record['conditioning_commands']),
                 'EVERY_ACTUAL_PROGRAM_SUFFIX_AND_GRIPPER_MUST_MATCH_PREDICTION_CONDITION')
            need(all(b['gripper_sequence']==a['gripper_sequence']+1 for a,b in zip(commands,commands[1:])),
                 'ORIGINAL_CONTIGUOUS_PER_ATOM_GRIPPER_SEQUENCE')
            need(all(row['gripper_wrapper_identity']==commands[0]['gripper_wrapper_identity'] for row in commands),
                 'SAME_ACTUAL_OWNED_GRIPPER_CHILD_AND_FROZEN_LIBRARY')
            need(consumed[0]['state_before_sha256']==record['issue_state_sha256']
                 and consumed[-1]['state_after_sha256']==state_sha256(after)
                 and all(_hash(r['state_before_sha256']) and _hash(r['state_after_sha256']) for r in consumed)
                 and all(a['state_after_sha256']==b['state_before_sha256'] for a,b in zip(consumed,consumed[1:])),
                 'SAME_ISSUE_AND_SUCCESSOR_STATE_CHAIN_NOT_ANOTHER_RESTART')
            need(current_command is not None and current_command['reference_id']==commands[-1]['reference_id']
                 and current_command['condition']==commands[-1]['condition'], 'SAME_CURRENT_ACTUAL_CONSUMPTION_NOT_ANOTHER_BRANCH')
            need(all(s.get(k) is False for s in (before,after) for k in ('reset','takeover','native_abort')),
                 'ACTUAL_RESET_TAKEOVER_ABORT_OR_UNKNOWN_FLAGS')
        except (ValueError,KeyError,TypeError,IndexError,AttributeError) as error:
            reasons.append(str(error))
            if not isinstance(record,dict):record=None
    values={name:None for name in FIELDS};masks={name:False for name in FIELDS}
    if record is not None and not reasons:
        try:predicted,available=_predicted_values(record)
        except (ValueError,KeyError,TypeError,IndexError):predicted={};available={}
        for name in FIELDS:
            try:
                need(available.get(name,False), 'MISSING_PREDICTED_'+name)
                actual=_actual_value(after,name)
                values[name]=(actual-np.asarray(predicted[name])).tolist() if name=='finger_q' else _pose_error(predicted[name],actual)
                masks[name]=True
            except (ValueError,KeyError,TypeError,IndexError):pass
    return dict(valid=not reasons and any(masks.values()),invalid_reasons=reasons,validity_mask=masks,values=values,
        prediction_artifact_sha256=pin,model_sha256=None if record is None else record.get('model_sha256'),
        issue_native_step=None if record is None else record.get('issue_native_step'),
        prediction_timestamp=None if record is None else record.get('issue_time_s'),
        target_native_step=None if record is None else record.get('target_native_step'),
        conditioning_command_domain=None if record is None else record.get('conditioning_command_domain'),
        history_artifact_authenticated_by_packet=False,
        hidden_controller_selection_observed=False,partial_observability_remains=True,
        conditioning_reference=None if record is None else record.get('conditioning_commands'),
        information_origin='PREDICTION_VS_ACTUAL_SIMULATOR_MEASUREMENT',expert_action_label=False,
        training_authorized=False,model_domain_qualification_established=False,controller_selected_policy_id_proven=False)


def _forecast(candidate,identity,step,state_hash):
    reasons=[];values={name:None for name in FIELDS};mask={name:False for name in FIELDS}
    values['event_probabilities']={name:None for name in ('contact','grasp','release')}
    mask['event_probabilities']={name:False for name in ('contact','grasp','release')}
    try:
        need(isinstance(candidate['candidate_id'],str) and bool(candidate['candidate_id'])
             and candidate['identity']==identity and type(candidate['issue_native_step']) is int
             and candidate['issue_native_step']==step
             and type(candidate['target_native_step']) is int and candidate['target_native_step']>step
             and (candidate['target_native_step']-step)%25==0 and _hash(candidate['model_sha256']),
             'SAME_CURRENT_CANDIDATE_FORECAST_IDENTITY_AND_CLOCK')
        values,mask=_predicted_values(candidate)
        proof=candidate['candidate_qualification'];preview=np.asarray(candidate['full_candidate_H40'],float)
        need(preview.shape==(40,10) and np.isfinite(preview).all(), 'FULL_CANDIDATE_H40')
        require_working_target(preview[:,9])
        certificate=proof['certificate'];prefix=certificate['executed_reference_history_points']
        need(_hash(proof['native_qualification_sha256']) and proof['actual_state_sha256']==state_hash
             and proof['native_step']==step
             and proof['full_preview_sha256']==hashlib.sha256(preview.astype('<f8').tobytes()).hexdigest()
             and certificate['status']=='PASS_CLONED_NATIVE_REFERENCE_CERTIFICATE_NOT_DYNAMIC_WBC_PROOF'
             and type(prefix) is int and prefix in (0,2)
             and type(certificate['points']) is int and certificate['points']==(40+prefix)*25+1
             and certificate['Cartesian_reference_values_changed'] is False
             and certificate['physics_integrations']==0 and certificate['actual_state_modified'] is False
             and proof['certificate_sha256']==document_hash(certificate),
             'PINNED_CANDIDATE_QUALIFICATION_AND_UNCHANGED_ORIGINAL_FULL_PREVIEW_CERTIFICATE')
        need(candidate['support_domain_flag']=='IN_DOMAIN_WITH_EXTERNAL_QUALIFICATION'
             and _hash(candidate['model_qualification_sha256']), 'UNKNOWN_OR_UNSUPPORTED_WM_NOT_PROMOTED')
    except (ValueError,KeyError,TypeError,IndexError,AttributeError) as error:reasons.append(str(error))
    available=copy.deepcopy(mask)
    valid_mask={name:bool(not reasons and mask.get(name,False)) for name in FIELDS}
    calibrated=not reasons and _hash(candidate.get('event_calibration_sha256'))
    valid_mask['event_probabilities']={k:bool(calibrated and v) for k,v in mask.get('event_probabilities',{}).items()}
    uncertainty=candidate.get('uncertainty') or {}
    if not isinstance(uncertainty,dict):uncertainty={}
    uncertainty_values={};uncertainty_masks={}
    for name,shape in (('empirical_tcp_error_interval_m',(3,2)),('ensemble_tcp_dispersion_m',(3,))):
        try:
            value=np.asarray(uncertainty[name],float)
            need(value.shape==shape and np.isfinite(value).all(), 'FINITE_UNCERTAINTY')
            need(np.all(value[:,0]<=value[:,1]) if len(shape)==2 else np.all(value>=0), 'ORDERED_UNCERTAINTY')
            uncertainty_values[name]=value.tolist()
            uncertainty_masks[name]=bool(not reasons and _hash(uncertainty.get('calibration_sha256')))
        except (ValueError,KeyError,TypeError,IndexError):uncertainty_values[name]=None;uncertainty_masks[name]=False
    return dict(candidate_id=(candidate.get('candidate_id') if isinstance(candidate.get('candidate_id'),str) else None),
        model_sha256=(candidate.get('model_sha256') if _hash(candidate.get('model_sha256')) else None),
        issue_native_step=(candidate.get('issue_native_step') if type(candidate.get('issue_native_step')) is int else None),
        target_native_step=(candidate.get('target_native_step') if type(candidate.get('target_native_step')) is int else None),
        forecast_horizon_seconds=((candidate['target_native_step']-step)*DT
            if type(candidate.get('target_native_step')) is int else None),values=values,validity_mask=valid_mask,
        field_availability_mask=available,
        record_consistent=not reasons,invalid_reasons=reasons,
        support_domain_flag=(candidate.get('support_domain_flag') if isinstance(candidate.get('support_domain_flag'),str) else 'UNKNOWN'),
        uncertainty=dict(values=uncertainty_values,validity_mask=uncertainty_masks,
            calibration_sha256=(uncertainty.get('calibration_sha256') if _hash(uncertainty.get('calibration_sha256')) else None)),
        uncertainty_calibration_verified=False,
        information_origin='MODEL_ESTIMATE_CONDITIONED_ON_PRIVILEGED_NATIVE_STATE_NOT_MEASUREMENT',
        conditioning_command_domain='UPSTREAM_CANDIDATE_REFERENCE',future_actual_consumption_proven=False,
        forecast_is_current_proprioception=False,forecast_is_command=False,
        model_or_candidate_qualified_by_packet=False,usable_for_execution=False)


def make_packet(*,identity,state_before,state_after,consumption=None,saved_prediction=None,
                consumed_since_prediction=(),lifecycle_evidence=None,causal_timing=None,forecast_candidates=()):
    """Assemble separate masked sections from existing native record fields.

    ``consumption`` joins the actual window to its original ``proposed`` wire,
    augmented with original accept/installed ``receiver_events``, and explicit
    global ``program`` SHA/start-step. Predictions need immutable
    bytes/pin and a separate post-fsync receipt, plus original STEP_BEFORE/AFTER
    timing. Missing proof masks innovation; no old artifact is upgraded.
    """
    identity=_identity(identity);start,end=_step(state_before),_step(state_after)
    need(end-start==ATOM_TICKS, 'TWO_CONTIGUOUS_ACTUAL_50MS_STATES')
    command=None;command_error=None
    if consumption is not None:
        try:
            command=_consumption(consumption,identity)
            need(command['native_start_step']==start and command['native_end_step']==end, 'SAME_ACTUAL_MEASUREMENT_WINDOW')
            need(consumption['state_before_sha256']==state_sha256(state_before)
                 and consumption['state_after_sha256']==state_sha256(state_after), 'SAME_ACTUAL_STATE_DOCUMENTS')
        except (ValueError,KeyError,TypeError,IndexError) as error:command=None;command_error=str(error)
    measured=_measured(state_before,state_after,command)
    need(isinstance(forecast_candidates,(tuple,list)) and all(isinstance(c,dict) for c in forecast_candidates),
         'EXPLICIT_CANDIDATE_FORECAST_RECORDS')
    innovation=_innovation(saved_prediction,identity,state_before,state_after,command,consumed_since_prediction,
                           lifecycle_evidence,causal_timing or {})
    forecasts=[_forecast(c,identity,end,state_sha256(state_after)) for c in forecast_candidates]
    reference=None if command is None else {k:command[k] for k in ('reference_id','actual_consumed_index',
        'target_trajectory_sha256','reference_age_s','source_mpc_s','controller_selected_reference',
        'command_domain_evidence','condition')}
    return dict(schema=SCHEMA,identity=identity,observation_timestamp=state_after['time_s'],native_step=end,
        actual_state_document_sha256=state_sha256(state_after),
        state_identity_kind='CANONICAL_ACTUAL_STATE_DOCUMENT_NOT_RAW_PACKET_FILE_SHA',
        history_span=[start,end],units=dict(position='m',rotation='SO3_rad',time='native_s',
            monotonic_time='ns',finger_opening='m',reference_gripper='mechanical_fraction_q_over_0.044'),
        measured=measured,innovation=innovation,forecast=forecasts,consumed_reference=reference,
        WM_checkpoint_sha256=sorted({x for x in [innovation['model_sha256']]+[c.get('model_sha256') for c in forecasts] if _hash(x)}),
        consumption_invalid_reason=command_error,missing_values_are_null_with_false_masks=True,
        scope='OFFLINE_RECORD_CONSISTENCY_PROTOTYPE_NOT_NATIVE_OR_MODEL_QUALIFICATION',
        external_artifact_and_producer_authentication_required=True,native_execution_authorized=False,
        online_deployment_qualified=False,GR00T_updates=0,WM_updates=0,Actor_updates=0,
        current_proprioception_modified=False,Processor_padding_modified=False,language_modified=False,
        forecast_action_installations=0,new_queries=0,new_integrations=0)
