"""Typed pose-feedback entry on the original finite native lifecycle."""
from . import fusion_task_phase_pose_feedback_v6 as recipe
from .fusion_task_phase_pose_feedback_contract_v6 import PHASE,need


def validate_adapter_install(adapter,action):
    need(adapter.spec.get('phase') == PHASE and hasattr(adapter,'trajectory_initial_tcp'),
         'ACTUAL_V6_PHASE_AND_MEASURED_INIT')
    return recipe.validate_install_preview(action,initial_tcp=adapter.trajectory_initial_tcp,
        program=adapter.trajectory_program,step=adapter.observe()['step'],base=adapter.base_pose_hold,
        frozen_base=adapter.trajectory_initial_base,end_step=adapter.spec['max_ticks'])


def run(adapter):
    from .fusion_task_phase_pose_feedback_contract_v6 import episode_config
    from .fusion_task_phase_xyz_feedback_native_v4 import run_feedback_program
    episode,source = episode_config(adapter.spec)
    return run_feedback_program(adapter,episode,source,recipe_module=recipe,actual_phase=PHASE,
        report_schema='WM02_NATIVE_TASK_PHASE_POSE_FEEDBACK_V6',
        goal_schema='WM02_NATIVE_TASK_PHASE_POSE_FEEDBACK_GOAL_BINDING_V6',
        decision_schema='WM02_NATIVE_FINITE_POSE_FEEDBACK_DECISION_V6',
        feedback_report_schema='WM02_NATIVE_FINITE_POSE_FEEDBACK_REPORT_V6')
