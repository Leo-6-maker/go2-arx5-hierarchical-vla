"""One declared full-grade translation-bias fade, without learning or clipping.

The V7 desired trajectory, frozen SO(3) transport, clock and ordinary feedback
law remain unchanged. Saved-state certificate diagnostics motivate this new
reference hypothesis; they do not establish dynamic qualification.
"""
import copy
from dataclasses import dataclass

import numpy as np

from scripts.pi0292.eval_contract import matrix10
from scripts.pi0292.relay_protocol import digest as document_hash
from . import fusion_task_phase_pose_continuation_v7 as previous

original, engine, fixed = previous.original, previous.engine, previous.fixed
load_inputs = previous.load_inputs
PHASE = 'collect-target-fusion-task-phase-bias-fade-v8'
SCHEMA = 'WM02_FUSION_TASK_PHASE_BIAS_FADE_EPISODE_V8'
RECIPE = 'FULL1_FIXED_QUINTIC_TRANSLATION_BIAS_FADE_THEN_24_POSE_FEEDBACK_V8'
CONDITION = 'PREDECLARED_TRANSLATION_BIAS_FADE_FROZEN_SO3_FINITE_POSE_FEEDBACK_V8'
BASELINE_CONDITION = previous.CONDITION
SCALES = (1.,)
INIT_STEP, DECISION_STEP, TICKS_PER_ATOM, TAIL_ATOMS = 2503, 3703, 25, 8
FADE_START_ATOM, FADE_END_ATOM = 552, 672
FADE_START_STEP, FADE_END_STEP = 17503, 20503
FADE_DURATION_S = 6.
GAIN, STEP_CAP_M, DEADBAND_M, CUMULATIVE_CAP_M = .5, .008, .005, .192
ANGULAR_GAIN, ANGULAR_STEP_CAP_RAD, ANGULAR_DEADBAND_RAD, ANGULAR_CUMULATIVE_CAP_RAD = .5, .01, .005, .24
MOVE_ATOMS, SETTLE_ATOMS, DECISIONS, MAX_ENDPOINT = 20, 8, 24, 46303


def need(ok, why):
    if not ok:
        raise ValueError('TASK_PHASE_BIAS_FADE_V8_' + why)


def _fade():
    return dict(start_elapsed_atom=FADE_START_ATOM, end_elapsed_atom=FADE_END_ATOM,
        start_native_step=FADE_START_STEP, end_native_step=FADE_END_STEP,
        duration_s=FADE_DURATION_S, time_law='10u3-15u4+6u5',
        translation_bias_remaining='ONE_BEFORE552_ONE_MINUS_QUINTIC_TO672_ZERO_AFTER672',
        before_start='BITWISE_ORIGINAL_V7_FULL_BIAS_REFERENCE',
        after_end='DESIRED_XYZ_UNTIL_LATER_SAVED_CAUSAL_FEEDBACK',
        frozen_left_SO3_transport='UNCHANGED_ORIGINAL_R_INIT_R_DEC_TRANSPOSE',
        geometry_intervention_not_WM=True, dynamic_success_established=False)


def _fixed():
    return dict(previous._fixed(1.), schema=SCHEMA, phase=PHASE, reference_rule=RECIPE,
        initialization_condition=CONDITION, baseline_condition=BASELINE_CONDITION,
        translation_bias_fade=_fade(),
        reference_correction='DECLARED_CLOCK_TRANSLATION_BIAS_FADE_WITH_UNCHANGED_FROZEN_LEFT_SO3',
        endpoint_transport='TRANSLATION_BIAS_ZERO_AFTER672_FROZEN_LEFT_SO3_RETAINED',
        continuation='EXACT_QUALIFIED_V7_HALF_WITH_RETAINED_REJECTED_FULL1_GEOMETRY_REPAIR',
        original_failed_full1_reclassified=False, saved_state_certificate_is_dynamic_proof=False)


def feedback_steps(episode):
    return tuple(29503+700*i for i in range(DECISIONS))


def make_episode(*, run, geometry_scale):
    need(type(geometry_scale) in (int, float) and geometry_scale == 1., 'ONLY_NEW_FULL1_REPAIR')
    ancestor = previous.make_episode(run=run, geometry_scale=1.)
    return episode_contract(dict(ancestor, **_fixed()))


def episode_contract(episode):
    fields = _fixed()
    need(isinstance(episode, dict) and set(episode) == set(fields) | {
        'run', 'geometry_scale', 'ramp_atoms', 'planned_end_atom', 'feedback_decision_native_steps'}
        and all(document_hash(episode[k]) == document_hash(v)
                for k, v in fields.items()), 'EXACT_VERSIONED_FULL1_FADE_AND_UNCHANGED_DOSE')
    need(type(episode['geometry_scale']) in (int, float) and episode['geometry_scale'] == 1.
         and type(episode['ramp_atoms']) is int and episode['ramp_atoms'] == 1024
         and type(episode['planned_end_atom']) is int and episode['planned_end_atom'] == 1696
         and episode['feedback_decision_native_steps'] == list(feedback_steps(episode)),
         'ORIGINAL_FULL1_RAMP_24_SLOTS_AND_FINITE_CLOCK')
    previous.episode_contract(previous.make_episode(run=episode['run'], geometry_scale=1.))
    return episode


def parent_recipe_view(episode, *, checked=False):
    if not checked:
        episode_contract(episode)
    return original.make_episode(run=episode['run'], geometry_scale=1.)


def endpoint(episode):
    episode_contract(episode)
    return MAX_ENDPOINT


@dataclass(frozen=True, init=False, eq=False)
class FeedbackProgram(previous.FeedbackProgram):
    def __init__(self, inputs, episode, initial_tcp, decision_tcp, base):
        self._initialize_feedback(inputs, episode, initial_tcp, decision_tcp, base,
            episode_validator=episode_contract, ancestor_view=parent_recipe_view,
            schema=SCHEMA, recipe_name=RECIPE, condition=CONDITION, baseline=BASELINE_CONDITION,
            binding_schema='WM02_FROZEN_PREDECISION_REFERENCE_BINDING_V8',
            decision_schema='WM02_FINITE_CAUSAL_POSE_FEEDBACK_DECISION_V8',
            final_step=MAX_ENDPOINT, final_move_atom=1696,
            ramp_atoms=1024, feedback_steps=feedback_steps(episode))
        object.__setattr__(self, '_cumulative_feedback_rad', 0.)
        old = self.reference_path
        path = old.copy()
        u = np.arange(FADE_END_ATOM-FADE_START_ATOM+1, dtype=float)/(FADE_END_ATOM-FADE_START_ATOM)
        weight = 10*u**3-15*u**4+6*u**5
        section = slice(FADE_START_ATOM, FADE_END_ATOM+1)
        path[section, :3] = old[section, :3] + weight[:, None]*self.frozen_translation_bias_m
        # Preserve the endpoints exactly; no cancellation roundoff after fade.
        path[:FADE_START_ATOM+1] = old[:FADE_START_ATOM+1]
        path[FADE_END_ATOM:, :3] = self.desired_path[FADE_END_ATOM:, :3]
        need(np.array_equal(path[:, 3:], old[:, 3:])
             and np.array_equal(path[:FADE_START_ATOM+1], old[:FADE_START_ATOM+1]),
             'ONLY_PREDECLARED_FUTURE_TRANSLATION_BIAS_CHANGES')
        object.__setattr__(self, 'reference_path', fixed._immutable(path))
        object.__setattr__(self, 'reference_goal_TCP_target', fixed._immutable(
            matrix10(path[-1])))
        value = copy.deepcopy(self._evidence)
        bound = copy.deepcopy(value['analytical_rates']['desired'])
        bias = float(np.linalg.norm(self.frozen_translation_bias_m))
        bound['peak_translation_speed_m_s'] += 1.875*bias/FADE_DURATION_S
        bound['peak_translation_acceleration_m_s2'] += (10*np.sqrt(3)/3)*bias/FADE_DURATION_S**2
        need(bound['peak_translation_speed_m_s'] <= .05
             and bound['peak_translation_acceleration_m_s2'] <= .25,
             'FADE_CONSERVATIVE_CARTESIAN_RATE_BOUND_NO_RETIME_OR_CLIP')
        bound['bound_kind'] = 'TRIANGLE_BOUND_DESIRED_PLUS_QUINTIC_FROZEN_TRANSLATION_REMOVAL'
        value['analytical_rates']['reference'] = bound
        value.update(translation_bias_fade=_fade(), reference_path_sha256=fixed._sha(path),
            endpoint_transport_definition='PREDECLARED_TRANSLATION_FADE_ONLY_SO3_STAYS_FROZEN_NO_FUTURE_RESPONSE_FIT',
            reference_endpoint_TCP=self.reference_goal_TCP_target.tolist(),
            initial_static_reference_endpoint_TCP=self.reference_goal_TCP_target.tolist(),
            original_full_bias_reference_path_sha256=fixed._sha(old),
            unchanged_desired_path_sha256=fixed._sha(self.desired_path),
            reference_calibration_hypothesis='FADE_TRANSLATION_ONLY_SO3_BIAS_UNCHANGED',
            saved_state_geometry_diagnostic_is_dynamic_success=False)
        value['frozen_predecision_reference_binding'].update(
            schema='WM02_FROZEN_PREDECISION_REFERENCE_BINDING_V8', condition=CONDITION,
            translation_bias_fade=_fade(), reference_path_sha256=fixed._sha(path),
            original_full_bias_reference_path_sha256=fixed._sha(old),
            frozen_SO3_bias_unchanged=True)
        object.__setattr__(self, '_evidence', value)


def make_program(inputs, episode, initial_tcp, decision_tcp, base):
    return FeedbackProgram(inputs, episode, initial_tcp, decision_tcp, base)


def validate_install_preview(action, *, initial_tcp, program, step, base, frozen_base, end_step):
    if step >= DECISION_STEP:
        need(type(program) is FeedbackProgram, 'EXACT_NEW_TYPED_FADE_PROGRAM')
        program.require_decisions_through(step)
    return fixed._validate_install_preview(action, initial_tcp=initial_tcp, program=program, step=step,
        base=base, frozen_base=frozen_base, end_step=end_step, program_type=FeedbackProgram,
        allowed_endpoints=(MAX_ENDPOINT,), status='EXACT_FADE_FULL1_SUFFIX_NOT_NATIVE_QUALIFIED')


def query_readiness(samples, episode, inputs):
    episode_contract(episode)
    value = original._query_readiness_at_endpoint(samples, parent_recipe_view(episode), inputs,
        end=MAX_ENDPOINT, maximum_endpoint=MAX_ENDPOINT)
    value.update(schema='WM02_FUSION_TASK_PHASE_BIAS_FADE_READINESS_V8', actual_episode_schema=SCHEMA,
        ordinary_feedback_and_declared_geometry_not_model_benefit=True)
    return value
