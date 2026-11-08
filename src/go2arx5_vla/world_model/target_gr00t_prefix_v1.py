"""Finite target GR00T reference prefixes, using the original native consumer.

This library supplies no submission, dynamic permission, GPU query or Teacher
fallback. The future paired native/GPU entry must verify the actual model and
fresh observation before calling it. A CPU check is not native qualification.
The original three-query external-adapter protocol remains separately typed.
"""
import numpy as np

from .storage import digest
from . import target_gr00t_collection_v1 as collection

PHASE='collect-target-gr00t-prefix-loop-v1'
KIND='TARGET_GR00T_PREFIX_LOOP_V1'
B=collection.B
PREFIX_ATOMS,POLICY_QUERIES,FINAL_HOLD_ATOMS=120,2,8
E=B+25*(PREFIX_ATOMS*POLICY_QUERIES+FINAL_HOLD_ATOMS)


def need(ok,why):
    if not ok:raise ValueError('TARGET_GR00T_PREFIX_'+why)


def protocol():
    return dict(schema='WM02_TARGET_GR00T_TWO_QUERY_PREFIX_PROTOCOL_V1',
        prefix_atoms=PREFIX_ATOMS,policy_queries=POLICY_QUERIES,
        final_hold_atoms=FINAL_HOLD_ATOMS,
        assistance='NATIVE_QUOTA_PAUSED_REAL_OBSERVATION',
        full_H40_retained_and_screened_before_each_prefix=True,
        cancelled_suffix='SUPERSEDED_ONLY_AFTER_ORIGINAL_NEW_REFERENCE_ACK',
        full_controller_preview_retained=True,
        observation_timestamp_rebased=False,original_per_install_MPC_clock=True,
        fresh_query_per_cycle=True,teacher_fallback=False,online_WM=False,
        geometry_scale=1.,time_stretch_factor=2,physics_dt_s=.002,WBC_changes=False,
        task_direction_is_an_outcome_not_an_admission_gate=True,
        derivation='REAL_TRAIN_120_ATOM_TIMING_AUDIT_NOT_STUDENT_DEV_SELECTION')


def declared(spec):
    need(spec.get('phase')==PHASE and spec.get('kind')==KIND
        and spec.get('target_gr00t_prefix_protocol')==protocol()
        and not spec.get('stable_task_closed_loop')and not spec.get('stable_task_candidate')
        and spec.get('live_skill_start_step')==B and spec.get('max_ticks')==E
        and spec.get('physics_dt_s')==.002 and spec.get('WBC_changes')is False,
        'EXACT_TWO_QUERY_FINITE_PROTOCOL')


def compile_proposal(raw, *, query_state10, source_from_target, actual_TCP,
                     consumed_reference_TCP, base_hold, actual_cube):
    """Compile all already decoded knots before selecting the fixed prefix.

    Cube geometry is used only by the established noncontact screen. The raw
    proposal and every timing/rotation/gripper mapping remain unchanged. In
    particular, the Teacher's identity world map must not replace the policy's
    actual observation frame map.
    """
    from scripts.pi0292.eval_contract import matrix10
    from .fusion_trajectory_adapter import TrajectoryAdaptation
    from .fusion_bounded_reference_tracking_v1 import screen
    state=np.asarray(query_state10,float);frame=np.asarray(source_from_target,float)
    actual=np.asarray(actual_TCP,float)
    need(state.shape==(10,)and frame.shape==actual.shape==(4,4)
        and np.isfinite(state).all()and np.isfinite(frame).all()
        and np.isfinite(actual).all(),'FINITE_ACTUAL_SOURCE_AND_TARGET_POSES')
    query=matrix10(state)
    need(np.allclose(query,frame@actual,rtol=0,atol=1e-7),
         'SAME_CURRENT_QUERY_AND_NATIVE_TCP_WITH_ACTUAL_FRAME')
    program=TrajectoryAdaptation(raw,query,frame,actual,consumed_reference_TCP,
        base_hold,1.,time_stretch_factor=2)
    qualification=screen(program,actual_cube)
    return program,dict(schema='WM02_TARGET_GR00T_FULL_PROPOSAL_PREFIX_SCREEN_V1',
        mechanical_screen=qualification,raw_knots=40,raw_H40=program.decoded_H40.tolist(),
        adaptation=program.evidence(),fixed_execution_prefix_atoms=PREFIX_ATOMS,
        raw_knot_consumption_if_prefix_completes=[int(k)<PREFIX_ATOMS for k in program.knot_indices],
        unexecuted_raw_knot_indices=[i for i,k in enumerate(program.knot_indices)if int(k)>=PREFIX_ATOMS],
        full_controller_preview_may_influence_MPC=True,
        native_execution_authorized=False,native_safety_established=False,
        extra_Processor_decodes=0,command_clips=0,teacher_fallback=False,
        model_updates=0,native_integrations=0)


def screened_program(*args,**kwargs):
    program,proof=compile_proposal(*args,**kwargs)
    need(proof['mechanical_screen']['eligible_for_original_native_certificate'],
         'FULL_PROPOSAL_MECHANICAL_DOMAIN_REJECT_NO_CLIPPING_OR_PREFIX_CHOICE')
    return program,proof


def load_generated_proposal(action, pin, **bindings):
    """A paired entry must already bind this file to its actual GPU response."""
    need(digest(action)==pin,'ACTUAL_GPU_ACTION_BYTES')
    with np.load(action,allow_pickle=False)as z:
        raw=z['physical_H40'].copy();normalized=z['normalized']
        need(raw.shape==(40,10)and normalized.shape==(1,40,132)
            and np.isfinite(raw).all()and np.isfinite(normalized).all(),
            'COMPLETE_FINITE_GPU_PROPOSAL_NOT_PARTIAL_OR_REDECODED')
    return screened_program(raw,**bindings)
