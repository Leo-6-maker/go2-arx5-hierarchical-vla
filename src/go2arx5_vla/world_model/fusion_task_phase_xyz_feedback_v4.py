"""Finite ordinary XYZ reference feedback, separate from WM/VLA experiments.

The desired intent and predecision bias are fixed by the original math. Four
causal commissioning decisions may change only the reference XYZ, after their
actual packet/window evidence has been saved by the native wrapper.
"""
import copy
from dataclasses import dataclass

import numpy as np

from scripts.pi0292.eval_contract import matrix10 as matrix10_pose
from scripts.pi0292.relay_protocol import digest as document_hash
from scripts.pi0294.dec1.target_observation import rigid
from . import fusion_task_phase_initialization as original
from . import fusion_task_phase_bias_calibration_v2 as fixed

PHASE = 'collect-target-fusion-task-phase-xyz-feedback-v4'
SCHEMA = 'WM02_FUSION_TASK_PHASE_XYZ_FEEDBACK_EPISODE_V4'
RECIPE = 'FROZEN_PREDECISION_BIAS_PLUS_FINITE_CAUSAL_XYZ_FEEDBACK_V4'
CONDITION = 'INIT_FROZEN_BIAS_ORDINARY_XYZ_FEEDBACK_V4'
BASELINE_CONDITION = 'INIT_FROZEN_PREDECISION_BIAS_GRADED_V3'
INIT_STEP, DECISION_STEP = original.INIT_STEP, original.DECISION_STEP
TICKS_PER_ATOM, TAIL_ATOMS = 25, 8
FEEDBACK_STEPS = (10303, 11003, 11703, 12403)
FINAL_STEP, FINAL_MOVE_ATOM = 13103, 368
GAIN, STEP_CAP_M, DEADBAND_M, CUMULATIVE_CAP_M = .5, .004, .005, .016
MOVE_ATOMS, SETTLE_ATOMS = 20, 8
load_inputs = original.load_inputs


def need(ok, reason):
    if not ok:
        raise ValueError('TASK_PHASE_XYZ_FEEDBACK_V4_' + reason)


def _fixed():
    return dict(fixed._fixed(), schema=SCHEMA, phase=PHASE, reference_rule=RECIPE,
        initialization_condition=CONDITION, baseline_condition=BASELINE_CONDITION,
        feedback_decision_native_steps=list(FEEDBACK_STEPS), feedback_gain=GAIN,
        feedback_step_norm_cap_m=STEP_CAP_M, feedback_deadband_m=DEADBAND_M,
        feedback_cumulative_path_length_cap_m=CUMULATIVE_CAP_M,
        feedback_move_atoms=MOVE_ATOMS, feedback_settle_atoms=SETTLE_ATOMS,
        feedback_reference_orientation='UNCHANGED_STATIC_BIAS_REFERENCE_ENDPOINT_ROTATION',
        feedback_goal='UNCHANGED_CURRENT_RESTART_POINT1_DESIRED_ENDPOINT',
        feedback_reference_start='PREVIOUS_COMMITTED_REFERENCE_ENDPOINT_NOT_ACHIEVED_TCP',
        feedback_measurements='CAUSAL_ACTUAL_COMPLETED_NATIVE_WINDOWS_ONLY',
        future_reference_placeholder='HOLD_CURRENT_COMMITTED_ENDPOINT_NOT_A_FUTURE_RESPONSE_LABEL',
        ordinary_feedback_not_WM_or_VLA_gain=True)


def make_episode(*, run, geometry_scale=.1):
    need(type(geometry_scale) in (int, float) and geometry_scale == .1, 'ONLY_POINT1_COMMISSIONING')
    return episode_contract(dict(original.make_episode(run=run, geometry_scale=.1),
        **_fixed(), planned_end_atom=FINAL_MOVE_ATOM))


def episode_contract(episode):
    fields = _fixed()
    need(isinstance(episode, dict) and set(episode) == set(fields) | {'run', 'geometry_scale', 'ramp_atoms', 'planned_end_atom'}
         and all(document_hash(episode[key]) == document_hash(value) for key, value in fields.items()),
         'EXACT_NEW_V4_STATIC_MANIFEST')
    need(type(episode['geometry_scale']) in (int, float) and episode['geometry_scale'] == .1
         and type(episode['ramp_atoms']) is int and episode['ramp_atoms'] == 256
         and type(episode['planned_end_atom']) is int and episode['planned_end_atom'] == FINAL_MOVE_ATOM,
         'UNCHANGED_POINT1_RAMP_AND_PREDECLARED_FOUR_SEGMENT_CLOCK')
    original.episode_contract(parent_recipe_view(episode, checked=True))
    return episode


def parent_recipe_view(episode, *, checked=False):
    """Pure desired-intent ancestor math, never an operational spec/manifest."""
    if not checked:
        episode_contract(episode)
    return original.make_episode(run=episode['run'], geometry_scale=.1)


def endpoint(episode):
    episode_contract(episode)
    return FINAL_STEP


@dataclass(frozen=True, eq=False)
class FeedbackUpdate:
    native_step: int
    revision_before: int
    actual_tcp: np.ndarray
    reference_path: np.ndarray
    evidence: dict


@dataclass(frozen=True, init=False, eq=False)
class FeedbackProgram(fixed.FrozenBiasProgram):
    def __init__(self, inputs, episode, initial_tcp, decision_tcp, base):
        self._initialize_feedback(inputs, episode, initial_tcp, decision_tcp, base,
            episode_validator=episode_contract, ancestor_view=parent_recipe_view,
            schema=SCHEMA, recipe_name=RECIPE, condition=CONDITION, baseline=BASELINE_CONDITION,
            binding_schema='WM02_FROZEN_PREDECISION_REFERENCE_BINDING_V4',
            decision_schema='WM02_FINITE_CAUSAL_XYZ_FEEDBACK_DECISION_V4',
            final_step=FINAL_STEP, final_move_atom=FINAL_MOVE_ATOM,
            ramp_atoms=256, feedback_steps=FEEDBACK_STEPS)

    def _initialize_feedback(self, inputs, episode, initial_tcp, decision_tcp, base, *,
                             episode_validator, ancestor_view, schema, recipe_name, condition, baseline,
                             binding_schema, decision_schema, final_step, final_move_atom,
                             ramp_atoms, feedback_steps):
        """Fixed-law math shared only by explicit versioned typed recipes."""
        self._initialize(inputs, episode, initial_tcp, decision_tcp, base,
            episode_validator=episode_validator, ancestor_view=ancestor_view,
            schema=schema, recipe_name=recipe_name, condition=condition, binding_schema=binding_schema)
        n = final_move_atom+1
        desired = self.desired_path[np.minimum(np.arange(n), ramp_atoms)]
        reference = self.reference_path[np.minimum(np.arange(n), ramp_atoms)]
        object.__setattr__(self, 'desired_path', fixed._immutable(desired))
        object.__setattr__(self, 'reference_path', fixed._immutable(reference))
        object.__setattr__(self, 'end_atom', final_move_atom)
        object.__setattr__(self, '_feedback_steps', tuple(feedback_steps))
        object.__setattr__(self, '_feedback_decision_schema', decision_schema)
        object.__setattr__(self, '_feedback_receipts', ())
        object.__setattr__(self, '_cumulative_feedback_m', 0.)
        evidence = copy.deepcopy(self._evidence)
        evidence.update(native_endpoint_step=final_step,
            baseline_condition=baseline,
            static_component_absolute_ancestor_condition=fixed.BASELINE_CONDITION,
            initial_static_reference_endpoint_TCP=matrix10_pose(reference[-1]).tolist(),
            clock=dict(atom_dt_s=.05, native_dt_s=.002, native_ticks_per_atom=25,
                end_atom=final_move_atom, knot_indices=list(self.knot_indices)),
            reference_path_sha256=fixed._sha(reference), desired_path_sha256=fixed._sha(desired),
            online_feedback_uses_completed_actual_measurements=True,
            original_static_ramp_atoms=ramp_atoms, feedback_decision_native_steps=list(feedback_steps),
            future_reference_placeholder='HOLD_ONLY_UNTIL_NEXT_SAVED_CAUSAL_DECISION')
        evidence['frozen_predecision_reference_binding'].update(
            baseline_condition=baseline,
            static_component_absolute_ancestor_condition=fixed.BASELINE_CONDITION,
            reference_path_sha256=fixed._sha(reference), desired_path_sha256=fixed._sha(desired),
            future_online_feedback_path_known_at3703=False)
        object.__setattr__(self, '_evidence', evidence)

    @property
    def feedback_count(self):
        return len(self._feedback_receipts)

    def prepare_feedback(self, native_step, actual_tcp):
        """Calculate, but do not commit or install, one bounded reference update."""
        need(type(native_step) is int and self.feedback_count < len(self._feedback_steps)
             and native_step == self._feedback_steps[self.feedback_count], 'EXACT_NEXT_CAUSAL_FEEDBACK_STEP')
        actual = rigid(actual_tcp)
        error = self.graded_goal_TCP_target[:3, 3]-actual[:3, 3]
        error_norm = float(np.linalg.norm(error))
        raw_delta = GAIN*error
        delta = np.zeros(3) if error_norm <= DEADBAND_M else raw_delta.copy()
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > STEP_CAP_M:
            delta *= STEP_CAP_M/delta_norm
        applied_norm = float(np.linalg.norm(delta))
        peak_speed = 1.875*applied_norm
        peak_acceleration = (10*np.sqrt(3)/3)*applied_norm
        need(peak_speed <= .05 and peak_acceleration <= .25,
             'ORIGINAL_REFERENCE_RATE_DOMAIN_NO_RETIME')
        need(self._cumulative_feedback_m+applied_norm <= CUMULATIVE_CAP_M+1e-12,
             'FINITE_CUMULATIVE_REFERENCE_PATH_LENGTH_NO_EXTRA_ATTEMPT')
        atom = (native_step-DECISION_STEP)//25
        start = self.reference_path[atom].copy()
        need(np.allclose(self.reference_path[atom:, 3:9], start[3:9], rtol=0, atol=1e-12)
             and np.allclose(self.reference_path[atom:, :3], start[:3], rtol=0, atol=1e-12),
             'PREVIOUS_COMMITTED_ENDPOINT_HOLD_REQUIRED_NO_OVERWRITE_OF_PENDING_MOTION')
        u = np.arange(MOVE_ATOMS+1)/MOVE_ATOMS
        weight = 10*u**3-15*u**4+6*u**5
        path = self.reference_path.copy()
        path[atom:atom+MOVE_ATOMS+1, :3] = start[:3]+weight[:, None]*delta
        path[atom+MOVE_ATOMS:, :3] = start[:3]+delta
        need(np.array_equal(path[:, 3:], self.reference_path[:, 3:])
             and np.array_equal(path[:atom], self.reference_path[:atom]),
             'XYZ_ONLY_FUTURE_REFERENCE_SUFFIX_PAST_ORIENTATION_GRIPPER_UNCHANGED')
        preview = path[np.minimum(atom+np.arange(40), self.end_atom)]
        proof = dict(schema=self._feedback_decision_schema,
            decision_index=self.feedback_count, native_step=native_step,
            actual_TCP=actual.tolist(), unchanged_desired_goal_TCP=self.graded_goal_TCP_target.tolist(),
            error_world_xyz_m=error.tolist(), error_norm_m=error_norm,
            raw_gain_delta_world_m=raw_delta.tolist(), applied_delta_world_m=delta.tolist(),
            applied_delta_norm_m=applied_norm, deadband_hold=error_norm <= DEADBAND_M,
            gain=GAIN, step_norm_cap_m=STEP_CAP_M, deadband_m=DEADBAND_M,
            cumulative_reference_path_length_m=self._cumulative_feedback_m+applied_norm,
            cumulative_reference_path_length_cap_m=CUMULATIVE_CAP_M,
            peak_translation_speed_m_s=peak_speed,
            peak_translation_acceleration_m_s2=peak_acceleration,
            move_atoms=MOVE_ATOMS, settle_atoms=SETTLE_ATOMS,
            reference_path_before_sha256=fixed._sha(self.reference_path),
            reference_path_after_sha256=fixed._sha(path), new_H40_sha256=fixed._sha(preview),
            unchanged_desired_path_sha256=fixed._sha(self.desired_path),
            reference_endpoint_before_EEF10=start.tolist(), reference_endpoint_after_EEF10=path[-1].tolist(),
            future_response_used=False, achieved_state_is_expert_label=False,
            ordinary_reference_feedback_not_GR00T_clipping=True,
            native_execution_authorized=False, WM_updates=0, VLA_queries=0)
        return FeedbackUpdate(native_step, self.feedback_count, fixed._immutable(actual),
                              fixed._immutable(path), proof)

    def commit_feedback(self, update, *, saved_decision_sha256):
        """Commit only after the wrapper independently saved/read back evidence."""
        need(type(update) is FeedbackUpdate and update.revision_before == self.feedback_count
             and isinstance(saved_decision_sha256, str) and len(saved_decision_sha256) == 64
             and all(c in '0123456789abcdef' for c in saved_decision_sha256),
             'NEW_TYPED_UPDATE_AND_SAVED_DECISION_FILE_SHA')
        checked = self.prepare_feedback(update.native_step, update.actual_tcp)
        need(document_hash(checked.evidence) == document_hash(update.evidence)
             and np.array_equal(checked.reference_path, update.reference_path), 'RECOMPUTED_EXACT_PENDING_UPDATE')
        object.__setattr__(self, 'reference_path', fixed._immutable(update.reference_path))
        object.__setattr__(self, '_cumulative_feedback_m', update.evidence['cumulative_reference_path_length_m'])
        object.__setattr__(self, '_feedback_receipts', (*self._feedback_receipts,
            dict(copy.deepcopy(update.evidence), saved_decision_sha256=saved_decision_sha256)))

    def require_decisions_through(self, native_step):
        need(self.feedback_count == sum(step <= native_step for step in self._feedback_steps),
             'NO_SCHEDULED_REFERENCE_WITHOUT_COMMITTED_SAVED_CAUSAL_DECISION')

    def evidence(self):
        value = copy.deepcopy(self._evidence)
        value.update(reference_path_sha256=fixed._sha(self.reference_path),
            reference_endpoint_TCP=matrix10_pose(self.reference_path[-1]).tolist(),
            current_reference_goal_TCP_target=matrix10_pose(self.reference_path[-1]).tolist(),
            feedback_decisions=copy.deepcopy(list(self._feedback_receipts)),
            feedback_decisions_committed=self.feedback_count,
            cumulative_feedback_reference_path_length_m=self._cumulative_feedback_m)
        return value


def make_program(inputs, episode, initial_tcp, decision_tcp, base):
    return FeedbackProgram(inputs, episode, initial_tcp, decision_tcp, base)


def validate_install_preview(action, *, initial_tcp, program, step, base, frozen_base, end_step):
    if step >= DECISION_STEP:
        need(type(program) is FeedbackProgram, 'EXACT_NEW_TYPED_FEEDBACK_PROGRAM')
        program.require_decisions_through(step)
    return fixed._validate_install_preview(action, initial_tcp=initial_tcp, program=program, step=step,
        base=base, frozen_base=frozen_base, end_step=end_step, program_type=FeedbackProgram,
        allowed_endpoints=(FINAL_STEP,), status='EXACT_FINITE_XYZ_FEEDBACK_SUFFIX_NOT_NATIVE_QUALIFIED')


def query_readiness(samples, episode, inputs):
    episode_contract(episode)
    value = original._query_readiness_at_endpoint(samples, parent_recipe_view(episode), inputs, end=FINAL_STEP)
    value.update(schema='WM02_FUSION_TASK_PHASE_XYZ_FEEDBACK_READINESS_V4',
        actual_episode_schema=SCHEMA, ordinary_feedback_not_model_benefit=True)
    return value
