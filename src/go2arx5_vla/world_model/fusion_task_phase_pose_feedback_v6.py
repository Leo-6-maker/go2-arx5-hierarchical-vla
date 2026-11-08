"""Finite world-frame XYZ and SO(3) reference commissioning, without learning.

Each new reference starts at the preceding committed endpoint. Measurements
determine bounded corrections; achieved poses never become reference anchors,
expert actions, or future desired labels.
"""
from dataclasses import dataclass

import numpy as np

from scripts.pi0292.eval_contract import matrix10 as matrix10_pose
from scripts.pi0292.relay_protocol import digest as document_hash
from scripts.pi0294.dec1.target_observation import rigid
from ..task_data.codec import pose10
from ..trajectory.so3 import matrix_to_quat
from . import fusion_task_phase_initialization as original
from . import fusion_task_phase_xyz_feedback_v4 as engine
from . import fusion_task_phase_xyz_feedback_continuation_v5 as previous

PHASE = 'collect-target-fusion-task-phase-pose-feedback-v6'
SCHEMA = 'WM02_FUSION_TASK_PHASE_POSE_FEEDBACK_EPISODE_V6'
RECIPE = 'FROZEN_BIAS_FINITE_TARGET_WORLD_XYZ_LEFT_SO3_FEEDBACK_V6'
CONDITION = 'INIT_FROZEN_BIAS_ORDINARY_XYZ_SO3_FEEDBACK_V6'
BASELINE_CONDITION = previous.CONDITION
SCALES = (.25, .5, 1.)
INIT_STEP, DECISION_STEP = original.INIT_STEP, original.DECISION_STEP
TICKS_PER_ATOM, TAIL_ATOMS = 25, 8
GAIN, STEP_CAP_M, DEADBAND_M, CUMULATIVE_CAP_M = .5, .004, .005, .032
ANGULAR_GAIN, ANGULAR_STEP_CAP_RAD, ANGULAR_DEADBAND_RAD, ANGULAR_CUMULATIVE_CAP_RAD = .5, .01, .005, .08
MOVE_ATOMS, SETTLE_ATOMS, DECISIONS = 20, 8, 8
MAX_ENDPOINT = 35103
load_inputs, fixed = original.load_inputs, previous.fixed


def need(ok, reason):
    if not ok:
        raise ValueError('TASK_PHASE_POSE_FEEDBACK_V6_' + reason)


def _fixed():
    fields = dict(previous._fixed(), schema=SCHEMA, phase=PHASE, reference_rule=RECIPE,
        initialization_condition=CONDITION, baseline_condition=BASELINE_CONDITION,
        feedback_cumulative_path_length_cap_m=CUMULATIVE_CAP_M,
        feedback_reference_orientation='LEFT_EXP_TARGET_WORLD_DELTA_TIMES_PREVIOUS_COMMITTED_REFERENCE_R',
        feedback_angular_error='LOG_R_DESIRED_GOAL_TIMES_R_ACTUAL_TRANSPOSE_TARGET_WORLD',
        feedback_angular_gain=ANGULAR_GAIN, feedback_angular_step_cap_rad=ANGULAR_STEP_CAP_RAD,
        feedback_angular_deadband_rad=ANGULAR_DEADBAND_RAD,
        feedback_angular_cumulative_path_length_cap_rad=ANGULAR_CUMULATIVE_CAP_RAD,
        finite_feedback_decisions=DECISIONS, finite_native_endpoint_maximum=MAX_ENDPOINT,
        continuation='SAME_POINT25_FAILED_V5_REPAIR_THEN_ADJACENT_QUALIFIED_V6_ONLY')
    return fields


def feedback_steps(episode):
    start = DECISION_STEP+25*(episode['ramp_atoms']+8)
    return tuple(start+700*i for i in range(DECISIONS))


def make_episode(*, run, geometry_scale):
    need(type(geometry_scale) in (int,float) and geometry_scale in SCALES,
         'ONLY_POINT25_POINT5_FULL1_SAME_GRADE_REPAIR_FIRST')
    ancestor = original.make_episode(run=run, geometry_scale=geometry_scale)
    return episode_contract(dict(ancestor, **_fixed(),
        feedback_decision_native_steps=list(feedback_steps(ancestor)),
        planned_end_atom=ancestor['ramp_atoms']+DECISIONS*(MOVE_ATOMS+SETTLE_ATOMS)))


def episode_contract(episode):
    fields = _fixed()
    need(isinstance(episode,dict) and set(episode) == set(fields) | {
        'run','geometry_scale','ramp_atoms','planned_end_atom','feedback_decision_native_steps'}
        and all(document_hash(episode[key]) == document_hash(value) for key,value in fields.items()),
        'EXACT_ACTUAL_V6_MANIFEST_AND_FIXED_DOSE')
    need(type(episode['geometry_scale']) in (int,float) and episode['geometry_scale'] in SCALES
         and type(episode['ramp_atoms']) is int
         and episode['ramp_atoms'] == original.RAMPS[episode['geometry_scale']]
         and type(episode['planned_end_atom']) is int
         and episode['planned_end_atom'] == episode['ramp_atoms']+224
         and episode['feedback_decision_native_steps'] == list(feedback_steps(episode)),
         'ORIGINAL_RAMP_PLUS_EXACT_EIGHT_FINITE_SEGMENTS')
    original.episode_contract(parent_recipe_view(episode,checked=True))
    need(DECISION_STEP+25*(episode['planned_end_atom']+8) <= MAX_ENDPOINT,
         'NEW_EXPLICIT_FINITE_V6_ENDPOINT_AT_MOST_35103')
    return episode


def parent_recipe_view(episode, *, checked=False):
    """Desired-intent math only; the actual V6 spec/manifest is never changed."""
    if not checked:
        episode_contract(episode)
    return original.make_episode(run=episode['run'],geometry_scale=episode['geometry_scale'])


def endpoint(episode):
    episode_contract(episode)
    return DECISION_STEP+25*(episode['planned_end_atom']+8)


def _log_so3(rotation):
    """Principal world rotation vector, including stable zero and pi limits."""
    q = matrix_to_quat(np.asarray(rotation,float))
    if q[3] < 0:
        q = -q
    sine = float(np.linalg.norm(q[:3]))
    return 2*q[:3] if sine < 1e-12 else q[:3]*(2*np.arctan2(sine,q[3])/sine)


def _exp_so3(vector):
    vector = np.asarray(vector,float)
    need(vector.shape == (3,) and np.isfinite(vector).all(),'FINITE_WORLD_ROTATION_VECTOR')
    theta = float(np.linalg.norm(vector))
    x,y,z = vector
    skew = np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    a = 1-theta**2/6+theta**4/120 if theta < 1e-8 else np.sin(theta)/theta
    b = .5-theta**2/24+theta**4/720 if theta < 1e-8 else (1-np.cos(theta))/theta**2
    return np.eye(3)+a*skew+b*(skew@skew)


def _bounded(error, gain, deadband, cap):
    norm = float(np.linalg.norm(error))
    raw = gain*error
    applied = np.zeros(3) if norm <= deadband else raw.copy()
    length = float(np.linalg.norm(applied))
    if length > cap:
        applied *= cap/length
    return norm,raw,applied,float(np.linalg.norm(applied))


@dataclass(frozen=True,init=False,eq=False)
class FeedbackProgram(engine.FeedbackProgram):
    def __init__(self,inputs,episode,initial_tcp,decision_tcp,base):
        self._initialize_feedback(inputs,episode,initial_tcp,decision_tcp,base,
            episode_validator=episode_contract,ancestor_view=parent_recipe_view,
            schema=SCHEMA,recipe_name=RECIPE,condition=CONDITION,baseline=BASELINE_CONDITION,
            binding_schema='WM02_FROZEN_PREDECISION_REFERENCE_BINDING_V6',
            decision_schema='WM02_FINITE_CAUSAL_POSE_FEEDBACK_DECISION_V6',
            final_step=endpoint(episode),final_move_atom=episode['planned_end_atom'],
            ramp_atoms=episode['ramp_atoms'],feedback_steps=feedback_steps(episode))
        object.__setattr__(self,'_cumulative_feedback_rad',0.)

    def _law_parameters(self):
        return (GAIN,STEP_CAP_M,DEADBAND_M,CUMULATIVE_CAP_M,
                ANGULAR_GAIN,ANGULAR_STEP_CAP_RAD,ANGULAR_DEADBAND_RAD,ANGULAR_CUMULATIVE_CAP_RAD)

    def prepare_feedback(self,native_step,actual_tcp):
        need(type(native_step) is int and self.feedback_count < len(self._feedback_steps)
             and native_step == self._feedback_steps[self.feedback_count],'EXACT_NEXT_CAUSAL_FEEDBACK_STEP')
        actual = rigid(actual_tcp)
        gain,step_cap,deadband,total_cap,angular_gain,angular_cap,angular_deadband,angular_total_cap = self._law_parameters()
        error = self.graded_goal_TCP_target[:3,3]-actual[:3,3]
        angular_error = _log_so3(self.graded_goal_TCP_target[:3,:3]@actual[:3,:3].T)
        error_norm,raw_delta,delta,applied_norm = _bounded(error,gain,deadband,step_cap)
        angle,raw_rotation,rotation,applied_angle = _bounded(
            angular_error,angular_gain,angular_deadband,angular_cap)
        need(self._cumulative_feedback_m+applied_norm <= total_cap+1e-12
             and self._cumulative_feedback_rad+applied_angle <= angular_total_cap+1e-12,
             'FINITE_XYZ_AND_ANGULAR_SUM_OF_NORMS_NO_EXTRA_ATTEMPT')
        peak_speed,peak_acceleration = 1.875*applied_norm,(10*np.sqrt(3)/3)*applied_norm
        peak_angular_speed,peak_angular_acceleration = 1.875*applied_angle,(10*np.sqrt(3)/3)*applied_angle
        need(peak_speed <= .05 and peak_acceleration <= .25
             and peak_angular_speed <= .25 and peak_angular_acceleration <= 1.,
             'ORIGINAL_REFERENCE_RATE_DOMAIN_NO_RETIME')
        atom = (native_step-DECISION_STEP)//25
        start = self.reference_path[atom].copy()
        need(np.allclose(self.reference_path[atom:,:9],start[:9],rtol=0,atol=1e-12),
             'PREVIOUS_COMMITTED_ENDPOINT_HOLD_NO_PENDING_MOTION_OVERWRITE')
        u = np.arange(MOVE_ATOMS+1)/MOVE_ATOMS
        weight = 10*u**3-15*u**4+6*u**5
        path = self.reference_path.copy()
        if applied_norm > 0:
            path[atom:atom+MOVE_ATOMS+1,:3] = start[:3]+weight[:,None]*delta
            path[atom+MOVE_ATOMS:,:3] = start[:3]+delta
        if applied_angle > 0:
            start_pose = matrix10_pose(start)
            for offset,w in enumerate(weight[1:],1):
                pose = start_pose.copy()
                pose[:3,:3] = _exp_so3(w*rotation)@start_pose[:3,:3]
                path[atom+offset,3:9] = pose10(pose,original.OPEN)[3:9]
            path[atom+MOVE_ATOMS:,3:9] = path[atom+MOVE_ATOMS,3:9]
        need(np.array_equal(path[:atom+1],self.reference_path[:atom+1])
             and np.array_equal(path[:,9],self.reference_path[:,9]),
             'PAST_AND_START_REFERENCE_GRIPPER_UNCHANGED_NO_ACHIEVED_REANCHOR')
        preview = path[np.minimum(atom+np.arange(40),self.end_atom)]
        proof = dict(schema=self._feedback_decision_schema,decision_index=self.feedback_count,native_step=native_step,
            actual_TCP=actual.tolist(),unchanged_desired_goal_TCP=self.graded_goal_TCP_target.tolist(),
            error_world_xyz_m=error.tolist(),error_norm_m=error_norm,
            raw_gain_delta_world_m=raw_delta.tolist(),applied_delta_world_m=delta.tolist(),
            applied_delta_norm_m=applied_norm,deadband_hold=error_norm <= deadband,
            gain=gain,step_norm_cap_m=step_cap,deadband_m=deadband,
            cumulative_reference_path_length_m=self._cumulative_feedback_m+applied_norm,
            cumulative_reference_path_length_cap_m=total_cap,
            angular_error_target_world_rotvec_rad=angular_error.tolist(),angular_error_norm_rad=angle,
            angular_error_convention='LOG_R_GOAL_R_ACTUAL_TRANSPOSE_TARGET_WORLD_LEFT',
            raw_gain_rotation_target_world_rad=raw_rotation.tolist(),applied_rotation_target_world_rad=rotation.tolist(),
            applied_rotation_norm_rad=applied_angle,angular_deadband_hold=angle <= angular_deadband,
            angular_gain=angular_gain,angular_step_cap_rad=angular_cap,
            angular_deadband_rad=angular_deadband,
            cumulative_reference_rotation_path_rad=self._cumulative_feedback_rad+applied_angle,
            cumulative_reference_rotation_path_cap_rad=angular_total_cap,
            peak_translation_speed_m_s=peak_speed,peak_translation_acceleration_m_s2=peak_acceleration,
            peak_angular_speed_rad_s=peak_angular_speed,peak_angular_acceleration_rad_s2=peak_angular_acceleration,
            move_atoms=MOVE_ATOMS,settle_atoms=SETTLE_ATOMS,
            reference_path_before_sha256=fixed._sha(self.reference_path),reference_path_after_sha256=fixed._sha(path),
            new_H40_sha256=fixed._sha(preview),unchanged_desired_path_sha256=fixed._sha(self.desired_path),
            reference_endpoint_before_EEF10=start.tolist(),reference_endpoint_after_EEF10=path[-1].tolist(),
            reference_rotation_update='EXP_WORLD_DELTA_LEFT_TIMES_PREVIOUS_COMMITTED_REFERENCE_R',
            future_response_used=False,achieved_state_is_expert_label=False,
            ordinary_reference_feedback_not_GR00T_clipping=True,native_execution_authorized=False,WM_updates=0,VLA_queries=0)
        return engine.FeedbackUpdate(native_step,self.feedback_count,fixed._immutable(actual),fixed._immutable(path),proof)

    def commit_feedback(self,update,*,saved_decision_sha256):
        super().commit_feedback(update,saved_decision_sha256=saved_decision_sha256)
        object.__setattr__(self,'_cumulative_feedback_rad',update.evidence['cumulative_reference_rotation_path_rad'])

    def evidence(self):
        value = super().evidence()
        value.update(cumulative_feedback_reference_rotation_path_rad=self._cumulative_feedback_rad,
            reference_rotation_frame='ACTUAL_TARGET_WORLD_LEFT_MULTIPLICATION',
            ordinary_XYZ_SO3_feedback_not_model_benefit=True)
        return value


def make_program(inputs,episode,initial_tcp,decision_tcp,base):
    return FeedbackProgram(inputs,episode,initial_tcp,decision_tcp,base)


def validate_install_preview(action,*,initial_tcp,program,step,base,frozen_base,end_step):
    if step >= DECISION_STEP:
        need(type(program) is FeedbackProgram,'EXACT_NEW_TYPED_POSE_FEEDBACK_PROGRAM')
        program.require_decisions_through(step)
    return fixed._validate_install_preview(action,initial_tcp=initial_tcp,program=program,step=step,
        base=base,frozen_base=frozen_base,end_step=end_step,program_type=FeedbackProgram,
        allowed_endpoints=(22303,28703,35103),status='EXACT_FINITE_XYZ_SO3_FEEDBACK_SUFFIX_NOT_NATIVE_QUALIFIED')


def query_readiness(samples,episode,inputs):
    episode_contract(episode)
    value = original._query_readiness_at_endpoint(samples,parent_recipe_view(episode),inputs,
        end=endpoint(episode),maximum_endpoint=MAX_ENDPOINT)
    value.update(schema='WM02_FUSION_TASK_PHASE_POSE_FEEDBACK_READINESS_V6',actual_episode_schema=SCHEMA,
        ordinary_feedback_not_model_benefit=True)
    return value
