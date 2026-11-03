"""WM02 bounded world-task primitive, with separate native backend time maps."""
import numpy as np
from .contracts import compile_action, FRAME, TCP

SOURCE = 'ARX5_SOURCE_MUJOCO'
TARGET = 'GO2_ARX5_MUJOCO_OCS2'


def compile_reference(raw, anchor, time_s, *, backend, frame, tcp):
    if backend not in (SOURCE, TARGET):
        raise ValueError('Backend must be bound explicitly; Isaac and OCS2 are not interchangeable')
    if backend == SOURCE and (frame != FRAME or tcp != TCP):
        raise ValueError('Source frame/TCP mismatch')
    if backend == TARGET and (frame != 'world' or tcp != 'x5_ee'):
        raise ValueError('Target reference requires already qualified native world/x5_ee')
    plan = compile_action(raw, anchor, time_s)
    plan.update(frame=frame, tcp=tcp, backend=backend, compiler='WM02_WORLD_DELTA_H40_V1')
    plan['ACTION_TIME_MAP'] = dict(
        raw_shape=[8,7], command_rows=40, atomic_intervals=8, first_node_offset_s=0.,
        raw_to_node=list(range(8)), raw_to_control_interval=list(range(8)),
        command_offsets_s=(np.arange(40)*.05).tolist(),
        interval_bounds_s=[[k*.05,(k+1)*.05] for k in range(8)],
        suffix='rows8..39 hold command7; conditions, not future physical targets',
        first_sample='command0 retained by existing receiver bind_packet; no measured-start substitution' if backend==TARGET else 'command0 installed at t+0',
        native_wire_points=41 if backend==TARGET else 40,
        native_wire_final_hold_offset_s=2. if backend==TARGET else 1.95,
        native_consumption='UNKNOWN_UNTIL_ACTUAL_REFERENCE_ACK_AND_PROGRESS_TRACE' if backend==TARGET else '25 native2ms ticks per50ms row',
        full_preview_may_affect_control=backend==TARGET,
        target_dynamic_qualified=False)
    if backend == TARGET:
        from scripts.pi0294.base_intent import eef_document
        from scripts.pi0294.task_gripper_contract import require_working_target
        # Raw units remain q/.044; reject mechanical endpoints outside the
        # original working stroke, never rescale or silently clip a label.
        require_working_target(plan['opening'])
        plan['wire_document'] = eef_document(plan['rows10'])
    return plan


def decision_due(atomic_index):
    if type(atomic_index) is not int or atomic_index < 0:
        raise ValueError('Nonnegative integer atomic clock required')
    return atomic_index % 8 == 0


def discounted_block(rewards):
    values = np.asarray(rewards, dtype=float)
    if values.ndim != 1 or not 1 <= len(values) <= 8 or not np.isfinite(values).all():
        raise ValueError('One to eight actually executed rewards required')
    return float(np.dot(.997**np.arange(len(values)), values)), .997**len(values)
