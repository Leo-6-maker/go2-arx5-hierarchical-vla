"""Pure Teacher-assisted task-phase initialization, never an expert label.

No native entry, query, FK, renderer, scheduler, storage writer or execution
grant lives here. A future original-owner wrapper must freeze the actual 3703
packet/cube witness before installation, certify every complete preview, and
retain the ordinary ACK/TTL/gripper/trace/shutdown proofs independently.
"""
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.pi0292.eval_contract import matrix10
from scripts.pi0292.relay_protocol import digest as document_hash, sealed, unseal, validate_observation
from scripts.pi0294.dec1.target_observation import rigid, transform_hash
from scripts.pi0294.task_gripper_contract import teacher_physical_fraction, require_working_target
from scripts.pi0294.task_state import parse_state
from ..task_data.codec import pose10
from ..trajectory.so3 import geodesic_angle, matrix_to_quat, quat_to_matrix
from .fusion_trajectory_adapter import _slerp as _exact_slerp
from .target_stability import motion, screen

SCHEMA = 'WM02_FUSION_TASK_PHASE_INITIALIZATION_EPISODE_V1'
PHASE = 'collect-target-fusion-task-phase-initialization-v1'
INPUT_SCHEMA = 'WM02_FUSION_TASK_PHASE_INITIALIZATION_INPUT_V1'
RECIPE = 'WORLD_ABSOLUTE_CUBE_CENTER_Z140_SOURCE_ROTATION_FIXED_QUINTIC_V1'
PURPOSE = 'TEACHER_ASSISTED_TASK_PHASE_INPUT_INITIALIZATION_DIAGNOSTIC'
SOURCE_OBSERVATION = '71558ff5bcbc03e83eef9fbe79c868e75a2fece5c712f708682c8f23a090a594'
SOURCE_PAYLOAD = '867e229119dcade9183e0090705492e80429c6c32474d633b2d5fe389b42bb96'
SOURCE_STATE = '8a69ab7e96bc7ba2690b93914c0f1b29d7bf940f7eca559eb91cec91ee4ae880'
FRAME_MAP = 'fc347a59fe6b82f758d198f2db93955ba2697a55e8ec24e490faeda53c2761b1'
SCENE = '40408bbcbef2469396187c741750905a6ba9ed7161dc031d14bfab5fed62ddd2'
RAMPS = {.03:128, .1:256, .25:512, .5:768, 1.:1024}
INIT_STEP, DECISION_STEP, TAIL_ATOMS = 2503, 3703, 8
ATOM_DT, NATIVE_DT, TICKS_PER_ATOM = .05, .002, 25
STANDOFF_M = .14
OPEN = float(teacher_physical_fraction(1.))
FK_BINDING = 'CALLER_ACTUALSTATE_SAME_PACKET_FK_WITNESS_REQUIRED_NOT_PROVED_BY_PURE_CONTRACT'
THRESHOLDS = dict(position_error_m=.01, lateral_error_m=.01, rotation_error_rad=.05,
    minimum_actual_cube_distance_m=.12, hold_duration_s=.4,
    relative_TCP_diameter_m=.002, relative_TCP_max_speed_m_s=.025,
    minimum_actual_finger_opening_m=.038, maximum_cube_displacement_m=.0005)
INPUT_KEYS = {'schema','decision_packet_sha256','decision_step','decision_time_s','scene_sha256',
    'frame_map_sha256','T_source_from_target','source_observation_sha256','source_payload_sha256',
    'source_state_eef10','source_state_sha256','decision_TCP_target','decision_TCP_sha256',
    'frozen_cube_pose_target','frozen_cube_pose_sha256','full_goal_TCP_target','full_goal_TCP_sha256',
    'actual_state_vector_sha256','actual_cube_binding'}


def need(ok, reason):
    if not ok:
        raise ValueError('TASK_PHASE_INITIALIZATION_' + reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _matrix_sha(value):
    return _sha(rigid(value).astype('<f8').tobytes())


def _immutable(value):
    a = np.asarray(value)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _full_goal(cube, source_state, frame):
    target = np.eye(4)
    target[:3, 3] = rigid(cube)[:3, 3] + [0., 0., STANDOFF_M]
    # Fixed frame rotation only. Neither object truth nor a fitted transform
    # changes the original fc347 mapping or the saved source observation.
    target[:3, :3] = rigid(frame)[:3, :3].T @ matrix10(np.asarray(source_state, float))[:3, :3]
    return rigid(target)


def load_inputs(*, source_observation_bytes, decision_packet_bytes, measured_decision_state,
                T_source_from_target, scene_sha256):
    """Bind a caller's measured ActualState witness to the exact 3703 packet.

    This validates source bytes, packet ABI/clock and identical qpos/qvel/ctrl.
    It cannot prove that the supplied TCP/cube poses were computed by frozen FK;
    that original-owner witness must be saved by the future native wrapper.
    Geometry here is calibration/scoring metadata, never a policy observation.
    """
    need(type(source_observation_bytes) is bytes and _sha(source_observation_bytes)==SOURCE_OBSERVATION,
         'EXACT_ORIGINAL_71558_BYTES')
    source = unseal(json.loads(source_observation_bytes))['payload']
    validate_observation(source)
    need(document_hash(source)==SOURCE_PAYLOAD and document_hash(source['state_eef10'])==SOURCE_STATE,
         'EXACT_ORIGINAL_SOURCE_POLICY_STATE')
    need(type(decision_packet_bytes) is bytes, 'ACTUAL_PACKET_BYTES')
    packet = parse_state(decision_packet_bytes)
    need(packet['step']==DECISION_STEP and np.isfinite(packet['time_s'])
         and abs(packet['time_s']-DECISION_STEP*NATIVE_DT)<=1e-8,
         'EXACT_3703_ABSOLUTE_NATIVE_CLOCK')
    for contact in packet['contacts']:
        need(all(np.isfinite(np.asarray(contact[k])).all() for k in
                 ('distance_m','wrench','normal','frame','point')), 'FINITE_ACTUAL_CONTACT_PACKET')
    state = measured_decision_state
    need(type(state.get('native_step')) is int and state['native_step']==packet['step']
         and state.get('time_s')==packet['time_s'], 'SAME_MEASURED_DECISION_CLOCK')
    need(state.get('safety_valid') is True and state.get('native_abort') is False
         and state.get('robot_object_contact') is False and state.get('gripper_state_valid') is True
         and state.get('object_present') is True, 'ORIGINAL_SAFE_NONCONTACT_DECISION_WITNESS')
    for k in ('qpos','qvel','ctrl'):
        need(np.array_equal(np.asarray(state[k]),packet[k]), 'SAME_MEASURED_PACKET_'+k)
    tcp, cube, frame = rigid(state['tcp']), rigid(state['object_pose']), rigid(T_source_from_target)
    need(scene_sha256==SCENE and transform_hash(frame)==FRAME_MAP, 'UNCHANGED_T06_SCENE_AND_FC347_FRAME')
    goal = _full_goal(cube, source['state_eef10'], frame)
    body = dict(schema=INPUT_SCHEMA,decision_packet_sha256=_sha(decision_packet_bytes),
        decision_step=DECISION_STEP,decision_time_s=packet['time_s'],scene_sha256=scene_sha256,
        frame_map_sha256=FRAME_MAP,T_source_from_target=frame.tolist(),
        source_observation_sha256=SOURCE_OBSERVATION,source_payload_sha256=SOURCE_PAYLOAD,
        source_state_eef10=copy.deepcopy(source['state_eef10']),source_state_sha256=SOURCE_STATE,
        decision_TCP_target=tcp.tolist(),decision_TCP_sha256=_matrix_sha(tcp),
        frozen_cube_pose_target=cube.tolist(),frozen_cube_pose_sha256=_matrix_sha(cube),
        full_goal_TCP_target=goal.tolist(),full_goal_TCP_sha256=_matrix_sha(goal),
        actual_state_vector_sha256=_sha(np.concatenate([packet[k] for k in ('qpos','qvel','ctrl')]).astype('<f8').tobytes()),
        actual_cube_binding=FK_BINDING)
    raw = np.repeat(pose10(goal,OPEN)[None],40,axis=0)
    return dict(frozen=sealed(body),raw=_immutable(raw),provenance=_provenance(body))


def _provenance(body):
    return dict(purpose=PURPOSE,experiment_role='NON_VLA_CALIBRATION',desired_outcome_provenance='CALIBRATION_INTENT',
        source_observation_sha256=body['source_observation_sha256'],decision_packet_sha256=body['decision_packet_sha256'],
        frozen_cube_pose_sha256=body['frozen_cube_pose_sha256'],full_goal_TCP_sha256=body['full_goal_TCP_sha256'],
        frame_map_sha256=body['frame_map_sha256'],source_pose_is_expert_action_label=False,
        raw_generated_H40=True,raw_is_VLA_action=False,object_and_goal_are_policy_inputs=False,
        original_source_observation_unmodified=True,actual_cube_FK_verified_by_this_module=False,
        native_execution_authorized=False,GR00T_updates=0,WM_updates=0,Actor_updates=0)


def _inputs(inputs):
    need(isinstance(inputs,dict) and set(inputs)=={'frozen','raw','provenance'}, 'EXACT_CALIBRATION_INPUT_FIELDS')
    body = unseal(inputs['frozen'])
    need(set(body)==INPUT_KEYS and body['schema']==INPUT_SCHEMA
         and body['actual_cube_binding']==FK_BINDING
         and body['scene_sha256']==SCENE and body['frame_map_sha256']==FRAME_MAP
         and transform_hash(body['T_source_from_target'])==FRAME_MAP
         and body['source_observation_sha256']==SOURCE_OBSERVATION
         and body['source_payload_sha256']==SOURCE_PAYLOAD
         and body['source_state_sha256']==document_hash(body['source_state_eef10'])==SOURCE_STATE,
         'IMMUTABLE_SOURCE_SCENE_FRAME_INPUT_IDENTITIES')
    need(type(body['decision_step']) is int and body['decision_step']==DECISION_STEP
         and type(body['decision_time_s']) in (int,float) and np.isfinite(body['decision_time_s'])
         and abs(body['decision_time_s']-DECISION_STEP*NATIVE_DT)<=1e-8
         and all(isinstance(body[k],str) and len(body[k])==64 and all(c in '0123456789abcdef' for c in body[k])
                 for k in ('decision_packet_sha256','actual_state_vector_sha256')),
         'FROZEN_ACTUAL_3703_PACKET_IDENTITY')
    need(_matrix_sha(body['decision_TCP_target'])==body['decision_TCP_sha256']
         and _matrix_sha(body['frozen_cube_pose_target'])==body['frozen_cube_pose_sha256'], 'FROZEN_MEASURED_POSES')
    goal = _full_goal(body['frozen_cube_pose_target'],body['source_state_eef10'],body['T_source_from_target'])
    need(np.array_equal(goal,np.asarray(body['full_goal_TCP_target']))
         and _matrix_sha(goal)==body['full_goal_TCP_sha256']
         and np.array_equal(inputs['raw'],np.repeat(pose10(goal,OPEN)[None],40,axis=0))
         and inputs['provenance']==_provenance(body), 'DERIVED_KNOWN_GOAL_NOT_VLA_OR_PRIVILEGED_POLICY_INPUT')
    return body


def _fixed():
    return dict(schema=SCHEMA,phase=PHASE,purpose=PURPOSE,reference_rule=RECIPE,split='TRAIN',
        scene_id='F2T06',parent='FUSION_F2_T06_ROOT_XY_V1',
        experiment_role='NON_VLA_CALIBRATION',desired_outcome_provenance='CALIBRATION_INTENT',
        goal='FROZEN_3703_ACTUAL_CUBE_CENTER_PLUS_WORLD_Z_0P14M',
        orientation='ORIGINAL_71558_SOURCE_ROTATION_MAPPED_BY_UNCHANGED_FC347',
        grade_goal='DECISION_TO_FULL_WORLD_GOAL_QUINTIC_ENDPOINT_SCALE',
        reference_anchor='ORIGINAL_MEASURED_INIT_TCP_2503',desired_anchor='FROZEN_MEASURED_DECISION_TCP_3703',
        endpoint_transport='NONE_BOTH_PATHS_SHARE_ABSOLUTE_GRADED_WORLD_GOAL',
        source_observation_sha256=SOURCE_OBSERVATION,frame_map_sha256=FRAME_MAP,scene_sha256=SCENE,
        query_pins=dict(VLA_query_present=False,calibration_source_observation_sha256=SOURCE_OBSERVATION,
                        frame_map_sha256=FRAME_MAP),
        prehold_atoms=40,screen_atom=48,tail_hold_atoms=TAIL_ATOMS,physics_dt_s=NATIVE_DT,
        atom_dt_s=ATOM_DT,native_ticks_per_atom=TICKS_PER_ATOM,thresholds=copy.deepcopy(THRESHOLDS),
        thresholds_are_native_safety=False,source_pose_is_expert_action_label=False,
        calibration_goal_is_VLA_action=False,object_and_goal_are_policy_inputs=False,
        GR00T_updates=0,WM_updates=0,Actor_updates=0,WBC_changes=False,full_task_success_claim=False)


def make_episode(*, run, geometry_scale):
    """Static manifest: no future packet, cube pose, goal or runtime hash.

    The caller can seal this before physics. ``load_inputs`` produces a separate
    runtime goal binding at the actual 3703 pause without changing this episode.
    """
    need(type(geometry_scale) in (int,float) and geometry_scale in RAMPS, 'PREDECLARED_GEOMETRY_GRADE')
    episode = dict(_fixed(),run=str(run),geometry_scale=float(geometry_scale),
                   ramp_atoms=RAMPS[geometry_scale],planned_end_atom=RAMPS[geometry_scale])
    episode_contract(episode)
    return episode


def episode_contract(episode):
    fixed = _fixed()
    need(isinstance(episode,dict) and set(episode)==set(fixed)|{'run','geometry_scale','ramp_atoms','planned_end_atom'}
         and all(document_hash(episode[k])==document_hash(v) for k,v in fixed.items()), 'EXACT_VERSIONED_EPISODE')
    alpha = episode['geometry_scale']
    need(type(alpha) in (int,float) and alpha in RAMPS and isinstance(episode['run'],str)
         and Path(episode['run']).is_absolute() and type(episode['ramp_atoms']) is int
         and type(episode['planned_end_atom']) is int
         and episode['ramp_atoms']==episode['planned_end_atom']==RAMPS[alpha], 'FIXED_GRADE_CLOCK_NO_RUNTIME_RETIME')
    endpoint(episode)
    return episode


def endpoint(episode):
    n=episode['planned_end_atom']
    need(type(n) is int and n in RAMPS.values(), 'PREDECLARED_INTEGER_RAMP')
    end=DECISION_STEP+TICKS_PER_ATOM*(n+TAIL_ATOMS)
    need(end<=33000, 'FINITE_ENDPOINT_AT_MOST_33000')
    return end


def _interpolate(first,last,weights):
    path=np.repeat(rigid(first)[None],len(weights),axis=0)
    path[:,:3,3]=first[:3,3]+weights[:,None]*(last[:3,3]-first[:3,3])
    q0=np.broadcast_to(matrix_to_quat(first[:3,:3]),(len(weights),4))
    q1=np.broadcast_to(matrix_to_quat(last[:3,:3]),(len(weights),4))
    path[:,:3,:3]=quat_to_matrix(_exact_slerp(q0,q1,weights[:,None]))
    path[0]=first;path[-1]=last
    return path


@dataclass(frozen=True,init=False,eq=False)
class InitializationProgram:
    """Persistent fixed-clock calibration; readonly arrays, no achieved reanchor."""
    def __init__(self,inputs,episode,initial_tcp,decision_tcp,base):
        body=_inputs(inputs);episode_contract(episode)
        initial,decision=rigid(initial_tcp),rigid(decision_tcp)
        need(np.array_equal(decision,np.asarray(body['decision_TCP_target'])), 'SAME_FROZEN_ACTUAL_DECISION_TCP')
        base=np.asarray(base,float)
        need(base.shape==(7,) and np.isfinite(base).all()
             and abs(np.linalg.norm(base[3:])-1)<=1e-6, 'FIXED_MEASURED_BASE7')
        full=rigid(body['full_goal_TCP_target']);alpha=float(episode['geometry_scale'])
        goal=_interpolate(decision,full,np.array([0.,alpha,1.]))[1]
        if alpha==1.:goal=full.copy()
        n=episode['ramp_atoms'];u=np.arange(n+1,dtype=float)/n;weight=10*u**3-15*u**4+6*u**5
        desired=_interpolate(decision,goal,weight);reference=_interpolate(initial,goal,weight)
        duration=n*ATOM_DT
        rates={}
        for name,start in (('reference',initial),('desired',decision)):
            distance=float(np.linalg.norm(goal[:3,3]-start[:3,3]));angle=float(geodesic_angle(start[:3,:3],goal[:3,:3]))
            r=dict(peak_translation_speed_m_s=1.875*distance/duration,
                peak_translation_acceleration_m_s2=(10*np.sqrt(3)/3)*distance/duration**2,
                peak_angular_speed_rad_s=1.875*angle/duration,
                peak_angular_acceleration_rad_s2=(10*np.sqrt(3)/3)*angle/duration**2)
            need(r['peak_translation_speed_m_s']<=.05 and r['peak_translation_acceleration_m_s2']<=.25
                 and r['peak_angular_speed_rad_s']<=.25 and r['peak_angular_acceleration_rad_s2']<=1.,
                 'PREDECLARED_FIXED_RAMP_RATE_REJECT_NO_RETIME_NO_CLIP')
            rates[name]=r
        for name,value in dict(desired_path=_immutable(np.stack([pose10(p,OPEN) for p in desired])),
            reference_path=_immutable(np.stack([pose10(p,OPEN) for p in reference])),base_pose_hold=_immutable(base),
            full_goal_TCP_target=_immutable(full),graded_goal_TCP_target=_immutable(goal),
            initial_reference_TCP_target=_immutable(initial),decision_TCP_target=_immutable(decision),
            knot_indices=tuple(int(i) for i in np.rint(np.linspace(0,n,40))),end_atom=n).items():
            object.__setattr__(self,name,value)
        evidence=dict(schema=SCHEMA,recipe=RECIPE,geometry_scale=alpha,ramp_atoms=n,
            native_endpoint_step=endpoint(episode),tail_hold_atoms=TAIL_ATOMS,rest_to_rest_time_law='10u3-15u4+6u5',
            reference_start_TCP=initial.tolist(),desired_start_TCP=decision.tolist(),
            full_goal_TCP_target=full.tolist(),graded_goal_TCP_target=goal.tolist(),
            reference_endpoint_TCP=reference[-1].tolist(),desired_endpoint_TCP=desired[-1].tolist(),
            reference_path_sha256=_sha(self.reference_path.astype('<f8').tobytes()),
            desired_path_sha256=_sha(self.desired_path.astype('<f8').tobytes()),
            frozen_input_body_sha256=inputs['frozen']['sha256'],clock=dict(atom_dt_s=ATOM_DT,native_dt_s=NATIVE_DT,
                native_ticks_per_atom=TICKS_PER_ATOM,end_atom=n,knot_indices=list(self.knot_indices)),
            analytical_rates=rates,generated_calibration_waypoints=40,raw_VLA_knots=0,
            gripper_physical_fraction=OPEN,full_preview_native_certificate_required=True,
            endpoint_transport_applied=False,achieved_state_reanchoring=False,native_execution_authorized=False,
            native_qualification_established=False,object_and_goal_are_policy_inputs=False)
        object.__setattr__(self,'_evidence',evidence)
        require_working_target(self.reference_path[:,9])

    @property
    def nominal_reference_path(self):
        return self.reference_path

    def preview(self,elapsed_atoms=0):
        need(type(elapsed_atoms) is int and elapsed_atoms>=0, 'NONNEGATIVE_INTEGER_PREVIEW_OFFSET')
        return self.reference_path[np.minimum(elapsed_atoms+np.arange(40),self.end_atom)].copy()

    def desired_at(self,elapsed_atoms):
        need(type(elapsed_atoms) is int and elapsed_atoms>=0, 'NONNEGATIVE_INTEGER_DESIRED_OFFSET')
        return self.desired_path[min(elapsed_atoms,self.end_atom)].copy()

    def evidence(self):
        return copy.deepcopy(self._evidence)


def make_program(inputs,episode,initial_tcp,decision_tcp,base):
    return InitializationProgram(inputs,episode,initial_tcp,decision_tcp,base)


def validate_install_preview(action,*,initial_tcp,program,step,base,frozen_base,end_step):
    """Exact new-program suffix only; the original native certifier still runs."""
    need(type(step) is int and INIT_STEP<=step<end_step and (step-INIT_STEP)%25==0,
         'ORIGINAL_ACTUAL_ATOM_BOUNDARY')
    base,frozen_base=np.asarray(base,float),np.asarray(frozen_base,float)
    need(base.shape==frozen_base.shape==(7,) and np.isfinite([base,frozen_base]).all()
         and np.allclose(base,frozen_base,rtol=0,atol=1e-10), 'UNCHANGED_MEASURED_BASE')
    if step<DECISION_STEP:
        need(program is None, 'PREDECISION_INITIAL_HOLD_ONLY')
        expected=np.repeat(pose10(rigid(initial_tcp),OPEN)[None],40,axis=0)
    else:
        need(isinstance(program,InitializationProgram)
             and np.allclose(program.initial_reference_TCP_target,rigid(initial_tcp),rtol=0,atol=1e-12)
             and np.allclose(program.base_pose_hold,frozen_base,rtol=0,atol=1e-10)
             and end_step==DECISION_STEP+25*(program.end_atom+TAIL_ATOMS),
             'EXACT_FROZEN_INIT_BASE_AND_FIXED_PROGRAM_ENDPOINT')
        expected=program.preview((step-DECISION_STEP)//25)
    action=np.asarray(action,float)
    need(action.shape==(40,10) and np.isfinite(action).all()
         and np.allclose(action,expected,rtol=0,atol=1e-12), 'ENTIRE_H40_PERSISTENT_SUFFIX_NOT_SINGLE_ROW')
    require_working_target(action[:,9])
    return dict(status='EXACT_CALIBRATION_SUFFIX_NOT_NATIVE_QUALIFIED',native_step=step,points_checked=40,
        original_certify_chunk_required=True,original_install_ACK_required=True,
        physics_advance_authorized=False,native_execution_authorized=False)


def query_readiness(samples,episode,inputs):
    """Check exactly B-200..B actual samples, never reference/model positions.

    ``query_ready`` is only this new initialization measurement predicate. It
    does not supply original owner/admission/certificate/native-PASS authority.
    Small grades can never be query-ready, even if already near the full goal.
    """
    episode_contract(episode)
    return _query_readiness_at_endpoint(samples,episode,inputs,end=endpoint(episode))


def _query_readiness_at_endpoint(samples,episode,inputs,*,end,maximum_endpoint=33000):
    """Pure measurement math at an explicit finite caller-owned endpoint.

    The public V1 entry retains its exact manifest/clock. A versioned wrapper
    can use this math after validating its own actual episode, without
    relabelling that episode or granting execution.
    """
    body=_inputs(inputs);episode_contract(episode)
    need(type(maximum_endpoint) is int and maximum_endpoint in (33000,35103,46303)
         and type(end) is int and endpoint(episode)<=end<=maximum_endpoint
         and (end-DECISION_STEP)%TICKS_PER_ATOM==0, 'FINITE_EXPLICIT_MEASUREMENT_ENDPOINT')
    required={'native_step','time_s','tcp','object_pose','qpos','qvel','ctrl','finger_q',
        'safety_valid','native_abort','robot_object_contact','left_object_contact','right_object_contact',
        'native_grasp','contact_valid','gripper_state_valid','object_present','illegal_contact',
        'joint_limit_penetration','reset','takeover'}
    flags=required-{'native_step','time_s','tcp','object_pose','qpos','qvel','ctrl','finger_q'}
    need(isinstance(samples,list) and len(samples)==9
         and all(isinstance(s,dict) and required<=set(s) for s in samples), 'EXACT_NINE_COMPLETE_ACTUAL_SAMPLES')
    steps=[s['native_step'] for s in samples]
    need(all(type(k) is int for k in steps) and steps==list(range(end-200,end+1,25))
         and all(type(s['time_s']) in (int,float) and np.isfinite(s['time_s'])
                 and abs(s['time_s']-s['native_step']*NATIVE_DT)<=1e-8 for s in samples),
         'EXACT_B_MINUS200_TO_B_ABSOLUTE_CLOCK_NO_NEAREST_OR_FUTURE')
    need(all(type(s[k]) is bool for s in samples for k in flags), 'ACTUAL_BOOLEAN_CONTACT_AND_SAFETY_FIELDS')
    poses=np.stack([rigid(s['tcp']) for s in samples]);cubes=np.stack([rigid(s['object_pose']) for s in samples])
    vectors={k:np.asarray([s[k] for s in samples],float) for k in ('qpos','qvel','ctrl','finger_q')}
    need(all(vectors[k].shape==(9,n) and np.isfinite(vectors[k]).all()
             for k,n in (('qpos',34),('qvel',32),('ctrl',20),('finger_q',2)))
         and np.all((vectors['finger_q']>=0)&(vectors['finger_q']<=.044)), 'FINITE_ACTUAL_NATIVE_VECTORS_AND_MECHANICAL_FINGERS')
    relative=motion(vectors['qpos'][:,:3],vectors['qpos'][:,3:7],poses[:,:3,3])['relative_tcp']
    stability=screen([s['time_s'] for s in samples],relative)
    full=rigid(body['full_goal_TCP_target']);decision=rigid(body['decision_TCP_target'])
    alpha=float(episode['geometry_scale']);goal=_interpolate(decision,full,np.array([0.,alpha,1.]))[1]
    error=poses[:,:3,3]-goal[:3,3];position=np.linalg.norm(error,axis=1);lateral=np.linalg.norm(error[:,:2],axis=1)
    angle=geodesic_angle(poses[:,:3,:3],goal[:3,:3]);distance=np.linalg.norm(poses[:,:3,3]-cubes[:,:3,3],axis=1)
    frozen=rigid(body['frozen_cube_pose_target']);cube_move=np.linalg.norm(cubes[:,:3,3]-frozen[:3,3],axis=1)
    predicates=dict(full_grade=alpha==1.,position=bool(position.max()<=THRESHOLDS['position_error_m']),
        lateral=bool(lateral.max()<=THRESHOLDS['lateral_error_m']),orientation=bool(angle.max()<=THRESHOLDS['rotation_error_rad']),
        actual_cube_distance=bool(distance.min()>THRESHOLDS['minimum_actual_cube_distance_m']),
        continuous_relative_hold=bool(stability['qualified']),
        actual_open=bool(vectors['finger_q'].min()>=THRESHOLDS['minimum_actual_finger_opening_m']),
        cube_still=bool(cube_move.max()<=THRESHOLDS['maximum_cube_displacement_m']),
        no_contact=all(not s[k] for s in samples for k in ('robot_object_contact','left_object_contact','right_object_contact','native_grasp')),
        original_safety=all(s['safety_valid'] and s['contact_valid'] and s['gripper_state_valid'] and s['object_present']
            and not any(s[k] for k in ('native_abort','illegal_contact','joint_limit_penetration','reset','takeover')) for s in samples))
    return dict(schema='WM02_FUSION_TASK_PHASE_INITIALIZATION_READINESS_V1',
        status='PURE_MEASUREMENT_PREDICATES_ONLY_ORIGINAL_NATIVE_PROOFS_STILL_REQUIRED',
        query_ready=all(predicates.values()),predicates=predicates,thresholds=copy.deepcopy(THRESHOLDS),
        metrics=dict(maximum_position_error_m=float(position.max()),maximum_lateral_error_m=float(lateral.max()),
            maximum_orientation_error_rad=float(angle.max()),minimum_actual_cube_distance_m=float(distance.min()),
            minimum_actual_finger_opening_m=float(vectors['finger_q'].min()),maximum_cube_displacement_m=float(cube_move.max()),
            maximum_cube_rotation_from_frozen_rad=float(geodesic_angle(cubes[:,:3,:3],frozen[:3,:3]).max()),
            final_actual_TCP=poses[-1].tolist(),full_goal_TCP_target=full.tolist(),graded_goal_TCP_target=goal.tolist(),
            final_actual_cube_pose=cubes[-1].tolist(),relative_hold=stability),
        actual_sample_steps=steps,frozen_decision_packet_sha256=body['decision_packet_sha256'],
        thresholds_are_native_safety=False,original_native_safety_changed=False,
        native_execution_authorized=False,GPU_query_authorized=False,actual_native_qualification_established=False,
        object_and_goal_are_policy_inputs=False,source_pose_is_expert_action_label=False,
        source80mm_implies_target_skill_pass=False,full_task_success=False,new_queries=0,new_integrations=0)
