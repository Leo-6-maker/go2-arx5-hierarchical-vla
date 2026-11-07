"""Pure frozen predecision reference calibration, separate from VLA/WM trials.

Desired intent uses the unchanged V1 construction. Only the reference uses the
two actual predecision TCP measurements; future tracking error is never read.
Independent restarts are not a full-MPC-state fork or byte-identical inputs.
"""
import copy
from dataclasses import dataclass
import hashlib

import numpy as np

from scripts.pi0292.eval_contract import matrix10
from scripts.pi0292.relay_protocol import digest as document_hash
from scripts.pi0294.dec1.target_observation import rigid
from scripts.pi0294.task_gripper_contract import require_working_target
from ..task_data.codec import pose10
from . import fusion_task_phase_initialization as original

PHASE = 'collect-target-fusion-task-phase-bias-calibration-v2'
SCHEMA = 'WM02_FUSION_TASK_PHASE_BIAS_CALIBRATION_EPISODE_V2'
RECIPE = 'FROZEN_PREDECISION_XYZ_SO3_BIAS_REFERENCE_FIXED_QUINTIC_V2'
CONDITION = 'INIT_FROZEN_PREDECISION_BIAS_V2'
BASELINE_CONDITION = 'INIT_ABSOLUTE_V1'
DECISION_STEP, INIT_STEP = original.DECISION_STEP, original.INIT_STEP
TICKS_PER_ATOM, TAIL_ATOMS = original.TICKS_PER_ATOM, original.TAIL_ATOMS
load_inputs = original.load_inputs


def need(ok, reason):
    if not ok:
        raise ValueError('TASK_PHASE_FROZEN_BIAS_V2_' + reason)


def _fixed():
    return dict(original._fixed(), schema=SCHEMA, phase=PHASE, reference_rule=RECIPE,
        initialization_condition=CONDITION, baseline_condition=BASELINE_CONDITION,
        desired_path_construction=original.RECIPE,
        endpoint_transport='FROZEN_PREDECISION_XYZ_OFFSET_AND_LEFT_SO3_REFERENCE_ENDPOINT',
        reference_correction='XYZ_MINUS_DEC_MINUS_INIT_AND_R_INIT_R_DEC_T_LEFT_MULTIPLY',
        translation_frame='ACTUAL_TARGET_WORLD_COMPONENT_OFFSET_WITH_SEPARATE_LEFT_SO3_ROTATION',
        original_TrajectoryAdaptation_byte_equivalence_claim=False,
        calibration_measurement_native_steps=[INIT_STEP, DECISION_STEP],
        calibration_frozen_before_generated_reference_install=True,
        future_actual_tracking_error_used=False, learned_model_used=False,
        independent_restart_not_complete_MPC_state_fork=True,
        same_algorithm_does_not_claim_identical_restart_path_bytes=True)


def make_episode(*, run, geometry_scale=.03):
    need(type(geometry_scale) in (int, float) and geometry_scale == .03,
         'ONLY_FIRST_POINT03_CONDITION_BEFORE_MEASURED_QUALIFICATION')
    episode = dict(original.make_episode(run=run, geometry_scale=.03), **_fixed())
    return episode_contract(episode)


def episode_contract(episode):
    fixed = _fixed()
    need(isinstance(episode, dict) and set(episode) == set(fixed) | {'run', 'geometry_scale', 'ramp_atoms', 'planned_end_atom'}
         and all(document_hash(episode[k]) == document_hash(v) for k, v in fixed.items()),
         'EXACT_NEW_STATIC_EPISODE_NOT_OLD_RECIPE_DISGUISE')
    need(type(episode['geometry_scale']) in (int, float) and episode['geometry_scale'] == .03,
         'FIRST_POINT03_ONLY')
    original.episode_contract(parent_recipe_view(episode, checked=True))
    return episode


def parent_recipe_view(episode, *, checked=False):
    """Explicit pure mathematical ancestor view, never a spec or disk manifest."""
    if not checked:
        episode_contract(episode)
    view = dict(original._fixed(), **{key: copy.deepcopy(episode[key])
        for key in ('run', 'geometry_scale', 'ramp_atoms', 'planned_end_atom')})
    return original.episode_contract(view)


def endpoint(episode):
    episode_contract(episode)
    return original.endpoint(parent_recipe_view(episode, checked=True))


def _immutable(value):
    array = np.asarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _sha(array):
    return hashlib.sha256(np.asarray(array, dtype='<f8').tobytes()).hexdigest()


@dataclass(frozen=True, init=False, eq=False)
class FrozenBiasProgram:
    def __init__(self, inputs, episode, initial_tcp, decision_tcp, base):
        self._initialize(inputs, episode, initial_tcp, decision_tcp, base,
            episode_validator=episode_contract, ancestor_view=parent_recipe_view,
            schema=SCHEMA, recipe_name=RECIPE, condition=CONDITION,
            binding_schema='WM02_FROZEN_PREDECISION_REFERENCE_BINDING_V2')

    def _initialize(self, inputs, episode, initial_tcp, decision_tcp, base, *,
                    episode_validator, ancestor_view, schema, recipe_name, condition, binding_schema):
        """Shared fixed math; each version keeps its actual manifest and type."""
        episode_validator(episode)
        initial, decision = rigid(initial_tcp), rigid(decision_tcp)
        ancestor = original.make_program(inputs, ancestor_view(episode), initial, decision, base)
        bias = decision[:3, 3] - initial[:3, 3]
        left_rotation = initial[:3, :3] @ decision[:3, :3].T
        desired_poses = matrix10(ancestor.desired_path)
        references = desired_poses.copy()
        references[:, :3, 3] -= bias
        references[:, :3, :3] = np.einsum('ij,njk->nik', left_rotation, desired_poses[:, :3, :3])
        references[0] = initial
        path = np.stack([pose10(rigid(pose), original.OPEN) for pose in references])
        # Left rotation/constant translation preserve desired path rates. The
        # ancestor also checks its original rate envelope; the native certifier
        # must still check every transformed full40 preview, without clipping.
        for key, value in dict(desired_path=ancestor.desired_path, reference_path=_immutable(path),
            base_pose_hold=ancestor.base_pose_hold, full_goal_TCP_target=ancestor.full_goal_TCP_target,
            graded_goal_TCP_target=ancestor.graded_goal_TCP_target,
            reference_goal_TCP_target=_immutable(references[-1]),
            initial_reference_TCP_target=_immutable(initial), decision_TCP_target=_immutable(decision),
            frozen_translation_bias_m=_immutable(bias), frozen_left_rotation=_immutable(left_rotation),
            knot_indices=ancestor.knot_indices, end_atom=ancestor.end_atom).items():
            object.__setattr__(self, key, value)
        binding = dict(schema=binding_schema,
            condition=condition, baseline_condition=BASELINE_CONDITION,
            initial_native_step=INIT_STEP, decision_native_step=DECISION_STEP,
            largest_measurement_native_step=DECISION_STEP,
            initial_TCP_target=initial.tolist(), decision_TCP_target=decision.tolist(),
            initial_TCP_sha256=_sha(initial), decision_TCP_sha256=_sha(decision),
            translation_bias_DEC_minus_INIT_m=bias.tolist(), translation_bias_sha256=_sha(bias),
            left_rotation_R_INIT_R_DEC_transpose=left_rotation.tolist(), left_rotation_sha256=_sha(left_rotation),
            reference_path_sha256=_sha(self.reference_path), desired_path_sha256=_sha(self.desired_path),
            initial_source='ORIGINAL_SHARED_LOOP_ACTUALSTATE_SAMPLE_AT2503',
            decision_source='ORIGINAL_ACTUALSTATE_SAME_PACKET_FK_AT3703',
            decision_packet_sha256=inputs['provenance']['decision_packet_sha256'],
            frozen_before_generated_reference_install=True, future_tracking_error_used=False,
            historical_terminal_error_used=False, learned_model_used=False,
            translation_frame='ACTUAL_TARGET_WORLD_COMPONENT_OFFSET_NOT_COUPLED_SE3_TRANSLATION',
            orientation_frame='TARGET_WORLD_LEFT_R_INIT_R_DEC_TRANSPOSE',
            original_TrajectoryAdaptation_byte_equivalence_claim=False,
            source_or_target_achieved_pose_used_as_expert_label=False,
            calibration_is_native_or_GPU_grant=False)
        evidence = ancestor.evidence()
        evidence.update(schema=schema, recipe=recipe_name, initialization_condition=condition,
            baseline_condition=BASELINE_CONDITION,
            desired_path_construction=original.RECIPE,
            reference_path_sha256=_sha(self.reference_path), desired_path_sha256=_sha(self.desired_path),
            reference_endpoint_TCP=references[-1].tolist(),
            endpoint_transport_applied=True,
            endpoint_transport_definition='FROZEN_PREDECISION_BIAS_ONLY_NOT_FUTURE_RESPONSE_FIT',
            frozen_predecision_reference_binding=binding,
            native_original_full40_certifier_required=True, no_reference_clipping_or_retime=True,
            independent_restart_not_complete_MPC_state_fork=True,
            initialization_calibration_not_VLA_task_transfer_result=True)
        # The calibrated reference is a rigid image of the desired path; report
        # those rates rather than the ancestor's different absolute reference.
        evidence['analytical_rates']['reference'] = copy.deepcopy(evidence['analytical_rates']['desired'])
        object.__setattr__(self, '_evidence', evidence)

    def preview(self, elapsed_atoms=0):
        need(type(elapsed_atoms) is int and elapsed_atoms >= 0, 'INTEGER_PERSISTENT_PREVIEW_OFFSET')
        return self.reference_path[np.minimum(elapsed_atoms + np.arange(40), self.end_atom)].copy()

    def desired_at(self, elapsed_atoms):
        need(type(elapsed_atoms) is int and elapsed_atoms >= 0, 'INTEGER_DESIRED_OFFSET')
        return self.desired_path[min(elapsed_atoms, self.end_atom)].copy()

    def evidence(self):
        return copy.deepcopy(self._evidence)


def make_program(inputs, episode, initial_tcp, decision_tcp, base):
    return FrozenBiasProgram(inputs, episode, initial_tcp, decision_tcp, base)


def validate_install_preview(action, *, initial_tcp, program, step, base, frozen_base, end_step):
    return _validate_install_preview(action, initial_tcp=initial_tcp, program=program, step=step,
        base=base, frozen_base=frozen_base, end_step=end_step, program_type=FrozenBiasProgram,
        allowed_endpoints=(7103,), status='EXACT_FROZEN_BIAS_CALIBRATION_SUFFIX_NOT_NATIVE_QUALIFIED')


def _validate_install_preview(action, *, initial_tcp, program, step, base, frozen_base, end_step,
                              program_type, allowed_endpoints, status):
    need(type(step) is int and INIT_STEP <= step < end_step and (step-INIT_STEP) % 25 == 0,
         'ORIGINAL_ACTUAL_ATOM_BOUNDARY')
    base, frozen_base = np.asarray(base, float), np.asarray(frozen_base, float)
    need(base.shape == frozen_base.shape == (7,) and np.isfinite([base, frozen_base]).all()
         and np.allclose(base, frozen_base, rtol=0, atol=1e-10), 'UNCHANGED_MEASURED_BASE')
    if step < DECISION_STEP:
        need(program is None, 'PREDECISION_INIT_HOLD_ONLY')
        expected = np.repeat(pose10(rigid(initial_tcp), original.OPEN)[None], 40, axis=0)
    else:
        need(type(program) is program_type
             and np.allclose(program.initial_reference_TCP_target, rigid(initial_tcp), rtol=0, atol=1e-12)
             and np.allclose(program.base_pose_hold, frozen_base, rtol=0, atol=1e-10)
             and end_step in allowed_endpoints
             and end_step == DECISION_STEP + 25*(program.end_atom+TAIL_ATOMS),
             'EXACT_NEW_TYPED_FROZEN_BIAS_PROGRAM')
        expected = program.preview((step-DECISION_STEP)//25)
    action = np.asarray(action, float)
    need(action.shape == (40, 10) and np.isfinite(action).all()
         and np.allclose(action, expected, rtol=0, atol=1e-12), 'ENTIRE_TRANSFORMED_H40_SUFFIX_NO_CLIP')
    require_working_target(action[:, 9])
    return dict(status=status, native_step=step,
        points_checked=40, original_certify_chunk_required=True, original_install_ACK_required=True,
        physics_advance_authorized=False, native_execution_authorized=False)


def query_readiness(samples, episode, inputs):
    return original.query_readiness(samples, parent_recipe_view(episode), inputs)
