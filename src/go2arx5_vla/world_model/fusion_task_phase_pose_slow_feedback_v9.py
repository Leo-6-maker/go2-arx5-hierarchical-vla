"""Twelve slower ordinary pose corrections at the unchanged half-grade endpoint.

The desired path, frozen predecision bias and bounded correction law are the
V7 math. Only this version's feedback clock changes; the original native
certificate is still required for every installed preview.
"""
from dataclasses import dataclass

import numpy as np

from scripts.pi0292.eval_contract import matrix10
from scripts.pi0292.relay_protocol import digest as document_hash
from ..task_data.codec import pose10
from . import fusion_task_phase_pose_continuation_v7 as previous
from . import fusion_task_phase_pose_feedback_v6 as pose_feedback

original, engine, fixed = previous.original, previous.engine, previous.fixed
load_inputs = previous.load_inputs
PHASE = 'collect-target-fusion-task-phase-pose-slow-feedback-v9'
SCHEMA = 'WM02_FUSION_TASK_PHASE_POSE_SLOW_FEEDBACK_EPISODE_V9'
RECIPE = 'FROZEN_BIAS_FINITE_HALF_GRADE_SLOW_XYZ_LEFT_SO3_FEEDBACK_V9'
CONDITION = 'INIT_FROZEN_BIAS_ORDINARY_SLOW_XYZ_SO3_FEEDBACK_V9'
BASELINE_CONDITION = previous.CONDITION
SCALES = (.5,)
INIT_STEP, DECISION_STEP = original.INIT_STEP, original.DECISION_STEP
TICKS_PER_ATOM, TAIL_ATOMS, INITIAL_SETTLE_ATOMS = 25, 8, 8
MOVE_ATOMS, SETTLE_ATOMS, DECISIONS = 28, 8, 12
GAIN, STEP_CAP_M, DEADBAND_M = .5, .008, .005
ANGULAR_GAIN, ANGULAR_STEP_CAP_RAD, ANGULAR_DEADBAND_RAD = .5, .01, .005
CUMULATIVE_CAP_M = DECISIONS*STEP_CAP_M
ANGULAR_CUMULATIVE_CAP_RAD = DECISIONS*ANGULAR_STEP_CAP_RAD
FINAL_MOVE_ATOM, FINAL_STEP, MAX_ENDPOINT = 1216, 34303, 34303
FINAL_HOLD_ATOMS = 24


def need(ok, reason):
    if not ok:
        raise ValueError('TASK_PHASE_POSE_SLOW_FEEDBACK_V9_' + reason)


def _fixed():
    return dict(previous._fixed(.5), schema=SCHEMA, phase=PHASE,
        reference_rule=RECIPE, initialization_condition=CONDITION,
        baseline_condition=BASELINE_CONDITION,
        feedback_move_atoms=MOVE_ATOMS, feedback_settle_atoms=SETTLE_ATOMS,
        feedback_initial_settle_atoms=INITIAL_SETTLE_ATOMS,
        feedback_final_hold_atoms=FINAL_HOLD_ATOMS,
        feedback_cumulative_path_length_cap_m=CUMULATIVE_CAP_M,
        feedback_angular_cumulative_path_length_cap_rad=ANGULAR_CUMULATIVE_CAP_RAD,
        finite_feedback_decisions=DECISIONS, finite_native_endpoint_maximum=MAX_ENDPOINT,
        continuation='SAME_HALF_GRADE_DESIRED_AND_FROZEN_BIAS_SLOWER_FINITE_FEEDBACK_V9')


def feedback_steps(episode):
    start = DECISION_STEP+TICKS_PER_ATOM*(episode['ramp_atoms']+INITIAL_SETTLE_ATOMS)
    return tuple(start+TICKS_PER_ATOM*(MOVE_ATOMS+SETTLE_ATOMS)*i for i in range(DECISIONS))


def make_episode(*, run, geometry_scale):
    need(type(geometry_scale) in (int, float) and geometry_scale == .5, 'HALF_GRADE_ONLY')
    ancestor = original.make_episode(run=run, geometry_scale=.5)
    return episode_contract(dict(ancestor, **_fixed(), planned_end_atom=FINAL_MOVE_ATOM,
        feedback_decision_native_steps=list(feedback_steps(ancestor))))


def episode_contract(episode):
    fields = _fixed()
    need(isinstance(episode, dict) and set(episode) == set(fields) | {
        'run', 'geometry_scale', 'ramp_atoms', 'planned_end_atom', 'feedback_decision_native_steps'}
        and all(document_hash(episode[k]) == document_hash(v)
                for k, v in fields.items()), 'EXACT_VERSIONED_MANIFEST_AND_FIXED_CLOCK')
    need(type(episode['geometry_scale']) in (int, float) and episode['geometry_scale'] == .5
         and type(episode['ramp_atoms']) is int and episode['ramp_atoms'] == original.RAMPS[.5]
         and type(episode['planned_end_atom']) is int and episode['planned_end_atom'] == FINAL_MOVE_ATOM
         and type(episode['feedback_decision_native_steps']) is list
         and all(type(step) is int for step in episode['feedback_decision_native_steps'])
         and episode['feedback_decision_native_steps'] == list(feedback_steps(episode)),
         'ORIGINAL_HALF_RAMP_TWELVE_SLOTS_AND_UNCHANGED_ENDPOINT')
    original.episode_contract(parent_recipe_view(episode, checked=True))
    return episode


def parent_recipe_view(episode, *, checked=False):
    """Original half-grade desired intent only, never an operational manifest."""
    if not checked:
        episode_contract(episode)
    return original.make_episode(run=episode['run'], geometry_scale=.5)


def endpoint(episode):
    episode_contract(episode)
    return FINAL_STEP


@dataclass(frozen=True, init=False, eq=False)
class FeedbackProgram(previous.FeedbackProgram):
    def __init__(self, inputs, episode, initial_tcp, decision_tcp, base):
        self._initialize_feedback(inputs, episode, initial_tcp, decision_tcp, base,
            episode_validator=episode_contract, ancestor_view=parent_recipe_view,
            schema=SCHEMA, recipe_name=RECIPE, condition=CONDITION, baseline=BASELINE_CONDITION,
            binding_schema='WM02_FROZEN_PREDECISION_REFERENCE_BINDING_V9',
            decision_schema='WM02_FINITE_CAUSAL_SLOW_POSE_FEEDBACK_DECISION_V9',
            final_step=endpoint(episode), final_move_atom=FINAL_MOVE_ATOM,
            ramp_atoms=episode['ramp_atoms'], feedback_steps=feedback_steps(episode))
        object.__setattr__(self, '_cumulative_feedback_rad', 0.)
        self._evidence['clock'].update(feedback_move_atoms=MOVE_ATOMS,
            feedback_settle_atoms=SETTLE_ATOMS, feedback_initial_settle_atoms=INITIAL_SETTLE_ATOMS,
            feedback_interval_atoms=MOVE_ATOMS+SETTLE_ATOMS, final_hold_atoms=FINAL_HOLD_ATOMS)

    def _law_parameters(self):
        return (GAIN, STEP_CAP_M, DEADBAND_M, CUMULATIVE_CAP_M,
                ANGULAR_GAIN, ANGULAR_STEP_CAP_RAD, ANGULAR_DEADBAND_RAD, ANGULAR_CUMULATIVE_CAP_RAD)

    def prepare_feedback(self, native_step, actual_tcp):
        need(self.reference_path.shape == self.desired_path.shape == (FINAL_MOVE_ATOM+1, 10),
             'EXACT_ORIGINAL_END_SHAPE')
        # This pure call supplies the unchanged bounded correction. Its old
        # 20-atom candidate is discarded, never certified, installed or executed.
        bounded = super().prepare_feedback(native_step, actual_tcp)
        proof = dict(bounded.evidence)
        delta = np.asarray(proof['applied_delta_world_m'], float)
        rotation = np.asarray(proof['applied_rotation_target_world_rad'], float)
        atom = (native_step-DECISION_STEP)//TICKS_PER_ATOM
        need(atom+MOVE_ATOMS <= self.end_atom, 'WHOLE_SLOW_SEGMENT_INSIDE_FIXED_PATH')
        start = self.reference_path[atom].copy()
        path = self.reference_path.copy()
        u = np.arange(MOVE_ATOMS+1)/MOVE_ATOMS
        weight = 10*u**3-15*u**4+6*u**5
        if proof['applied_delta_norm_m'] > 0:
            path[atom:atom+MOVE_ATOMS+1, :3] = start[:3]+weight[:, None]*delta
            path[atom+MOVE_ATOMS:, :3] = start[:3]+delta
        if proof['applied_rotation_norm_rad'] > 0:
            start_pose = matrix10(start)
            for offset, w in enumerate(weight[1:], 1):
                pose = start_pose.copy()
                pose[:3, :3] = pose_feedback._exp_so3(w*rotation)@start_pose[:3, :3]
                path[atom+offset, 3:9] = pose10(pose, original.OPEN)[3:9]
            path[atom+MOVE_ATOMS:, 3:9] = path[atom+MOVE_ATOMS, 3:9]
        need(np.array_equal(path[:atom+1], self.reference_path[:atom+1])
             and np.array_equal(path[:, 9], self.reference_path[:, 9]),
             'PAST_START_AND_GRIPPER_UNCHANGED_NO_ACHIEVED_REANCHOR')
        duration = MOVE_ATOMS*.05
        speed = 1.875*proof['applied_delta_norm_m']/duration
        acceleration = (10*np.sqrt(3)/3)*proof['applied_delta_norm_m']/duration**2
        angular_speed = 1.875*proof['applied_rotation_norm_rad']/duration
        angular_acceleration = (10*np.sqrt(3)/3)*proof['applied_rotation_norm_rad']/duration**2
        need(speed <= .05 and acceleration <= .25
             and angular_speed <= .25 and angular_acceleration <= 1., 'ORIGINAL_REFERENCE_RATE_DOMAIN')
        preview = path[np.minimum(atom+np.arange(40), self.end_atom)]
        proof.update(move_atoms=MOVE_ATOMS, settle_atoms=SETTLE_ATOMS, move_duration_s=duration,
            feedback_interval_atoms=MOVE_ATOMS+SETTLE_ATOMS,
            peak_translation_speed_m_s=speed, peak_translation_acceleration_m_s2=acceleration,
            peak_angular_speed_rad_s=angular_speed, peak_angular_acceleration_rad_s2=angular_acceleration,
            reference_path_after_sha256=fixed._sha(path), new_H40_sha256=fixed._sha(preview),
            reference_endpoint_after_EEF10=path[-1].tolist())
        return engine.FeedbackUpdate(native_step, self.feedback_count, bounded.actual_tcp,
                                     fixed._immutable(path), proof)


def make_program(inputs, episode, initial_tcp, decision_tcp, base):
    return FeedbackProgram(inputs, episode, initial_tcp, decision_tcp, base)


def validate_install_preview(action, *, initial_tcp, program, step, base, frozen_base, end_step):
    if step >= DECISION_STEP:
        need(type(program) is FeedbackProgram, 'EXACT_TYPED_V9_PROGRAM')
        program.require_decisions_through(step)
    return fixed._validate_install_preview(action, initial_tcp=initial_tcp, program=program,
        step=step, base=base, frozen_base=frozen_base, end_step=end_step,
        program_type=FeedbackProgram, allowed_endpoints=(FINAL_STEP,),
        status='EXACT_FINITE_SLOW_HALF_POSE_SUFFIX_NOT_NATIVE_QUALIFIED')


def query_readiness(samples, episode, inputs):
    episode_contract(episode)
    # Keep the original V7 reader's accepted maximum enum; the V9 contract and
    # exact nine-sample clock above still bind the actual endpoint to 34303.
    value = original._query_readiness_at_endpoint(samples, parent_recipe_view(episode), inputs,
        end=endpoint(episode), maximum_endpoint=previous.MAX_ENDPOINT)
    value.update(schema='WM02_FUSION_TASK_PHASE_POSE_SLOW_FEEDBACK_READINESS_V9',
        actual_episode_schema=SCHEMA, ordinary_feedback_not_model_benefit=True)
    return value
