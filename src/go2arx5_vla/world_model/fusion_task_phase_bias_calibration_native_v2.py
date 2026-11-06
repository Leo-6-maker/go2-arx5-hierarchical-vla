"""Original native lifecycle with a new frozen-predecision reference condition."""
from . import fusion_task_phase_bias_calibration_v2 as recipe
from .fusion_task_phase_bias_calibration_contract_v2 import PHASE, need


def validate_adapter_install(adapter, action):
    need(adapter.spec.get('phase') == PHASE and hasattr(adapter, 'trajectory_initial_tcp'),
         'NEW_PHASE_AND_MEASURED_INIT_REQUIRED')
    return recipe.validate_install_preview(action, initial_tcp=adapter.trajectory_initial_tcp,
        program=adapter.trajectory_program, step=adapter.observe()['step'],
        base=adapter.base_pose_hold, frozen_base=adapter.trajectory_initial_base,
        end_step=adapter.spec['max_ticks'])


def run(adapter):
    from .fusion_task_phase_bias_calibration_contract_v2 import episode_config
    from .fusion_task_phase_initialization_native import run_initialization_program
    episode, source = episode_config(adapter.spec)

    def freeze(inputs, actual_episode, initial_tcp, decision_tcp, base):
        program = recipe.make_program(inputs, actual_episode, initial_tcp, decision_tcp, base)
        return program, program.evidence()['frozen_predecision_reference_binding']

    return run_initialization_program(adapter, episode, source, recipe_module=recipe,
        actual_phase=PHASE, report_schema='WM02_NATIVE_TASK_PHASE_FROZEN_BIAS_CALIBRATION_V2',
        goal_schema='WM02_NATIVE_TASK_PHASE_FROZEN_BIAS_GOAL_BINDING_V2',
        reference_calibration_factory=freeze,
        report_context=dict(initialization_condition=recipe.CONDITION,
            baseline_condition=recipe.BASELINE_CONDITION,
            historical_terminal_tracking_error_used=False, learned_reference_adapter_used=False,
            independent_restart_not_complete_MPC_state_fork=True,
            initialization_calibration_not_VLA_task_transfer_result=True))
