"""Slower ordinary-feedback prefix; unchanged complete generated-skill consumer.

The source prompt, full H40 adapter and native safety are unchanged. This is a
new V9 initialization condition, never a relabelled V7/V8 physical result.
"""
import numpy as np
from . import fusion_task_phase_pose_slow_feedback_v9 as prefix
from .fusion_task_phase_live_skill_contract_v9 import PHASE,KIND,B,E,need,episode_config
from .fusion_task_phase_live_skill_native_v8 import _continue
from .fusion_trajectory_adapter import TrajectoryAdaptation


def validate_adapter_install(adapter,action):
    need(adapter.spec.get('phase')==PHASE and adapter.spec.get('kind')==KIND,
         'ACTUAL_TYPED_V9_NATIVE_SCOPE')
    step=adapter.observe()['step'];program=getattr(adapter,'live_skill_program',None)
    if program is None:
        need(step<B,'NO_SUFFIX_INSTALL_BEFORE_SAVED_QUERY_AND_FULL_PROPOSAL')
        return prefix.validate_install_preview(action,initial_tcp=adapter.trajectory_initial_tcp,
            program=adapter.trajectory_program,step=step,base=adapter.base_pose_hold,
            frozen_base=adapter.trajectory_initial_base,end_step=B)
    need(type(program)is TrajectoryAdaptation and B<=step<E and (step-B)%25==0
         and program.geometry_scale==1.
         and type(adapter.spec.get('live_skill_time_stretch_factor'))is int
         and program.time_stretch_factor==adapter.spec['live_skill_time_stretch_factor']
         and np.array_equal(np.asarray(action),program.preview((step-B)//25))
         and np.array_equal(program.base_pose_hold,np.asarray(adapter.base_pose_hold))
         and np.array_equal(np.asarray(adapter.base_pose_hold),adapter.trajectory_initial_base),
         'WHOLE_LIVE_SUFFIX_EXACT_LOCAL_CLOCK_AND_UNCHANGED_BASE')
    return dict(status='EXACT_GENERATED_FULL_H40_SUFFIX_NOT_NATIVE_QUALIFIED',native_step=step,
        geometry_scale=program.geometry_scale,time_stretch_factor=program.time_stretch_factor,
        points_checked=40,original_certify_chunk_required=True,original_install_ACK_required=True,
        physics_advance_authorized=False)


def run(adapter):
    from .fusion_task_phase_xyz_feedback_native_v4 import run_feedback_program
    episode,source,task,camera=episode_config(adapter.spec)
    def continuation(current,report,samples,windows):
        return _continue(current,report,samples,windows,task=task,camera=camera)
    return run_feedback_program(adapter,episode,source,recipe_module=prefix,actual_phase=PHASE,
        report_schema='WM02_NATIVE_GRADED_INITIALIZATION_PREFIX_LIVE_V9',
        goal_schema='WM02_NATIVE_GRADED_INITIALIZATION_GOAL_BINDING_LIVE_V9',
        decision_schema='WM02_NATIVE_FINITE_POSE_FEEDBACK_DECISION_LIVE_V9',
        feedback_report_schema='WM02_NATIVE_FINITE_POSE_FEEDBACK_REPORT_LIVE_V9',
        continuation=continuation)
