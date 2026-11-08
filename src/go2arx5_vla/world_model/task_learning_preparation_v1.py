"""Task-local reference hold after the unchanged native startup and steady gate.

This removes the unrelated source-pose calibration from NEW grasp pilots.
It does not replace reset, WBC, integration, certificates or consumed evidence.
"""
import numpy as np
from . import task_learning_grasp_v1 as task
from .fusion_trajectory_qualification_contract import hold_preview
from ..task_data.codec import pose10


def episode(spec):
    from .storage import digest
    return dict(schema='TASK_LEARNING_LOCAL_PREPARATION_EPISODE_V4',run=spec['registered_root'],
        phase=task.PHASE,scene_sha256=digest(spec['model_path']),parent=spec['data_parent'],
        data_role=spec['data_role'],protocol=task.protocol(),
        query_pins=dict(role='ORIGINAL_NATIVE_REFERENCE_HOLD_NO_MODEL_QUERY_OR_EXPERT_LABEL'),
        source_pose_alignment=False,end_native_step=task.CALIBRATION_B)


def endpoint(document):
    task.need(document.get('schema')=='TASK_LEARNING_LOCAL_PREPARATION_EPISODE_V4'
        and document.get('protocol')==task.protocol()and document.get('source_pose_alignment')is False
        and document.get('end_native_step')==task.CALIBRATION_B,'EXACT_TASK_LOCAL_PREPARATION')
    return task.CALIBRATION_B


class HoldProgram:
    knot_indices=tuple(range(40))
    end_atom=39

    def __init__(self,initial,actual):
        self.reference_path=hold_preview(initial)
        self.desired_path=np.repeat(pose10(actual,1.)[None,:],40,axis=0)
        self.reference_path.setflags(write=False);self.desired_path.setflags(write=False)

    def preview(self,elapsed):
        task.need(type(elapsed)is int and elapsed>=0,'EXACT_PREPARATION_HOLD_CLOCK')
        return self.reference_path.copy()

    def desired_at(self,elapsed):
        return self.desired_path[min(elapsed,39)].copy()

    def evidence(self):
        return dict(schema='TASK_LEARNING_ORIGINAL_INITIAL_REFERENCE_HOLD_V4',
            generated_reference='UNCHANGED_ACTUAL_2503_TCP_REFERENCE',
            source_pose_alignment=False,expert_training_eligible=False,
            startup_WBC_native_safety_or_dynamics_changed=False)


def validate_preview(adapter,action):
    step=adapter.observe()['step']
    task.need(2503<=step<task.CALIBRATION_B and (step-2503)%25==0
        and np.array_equal(action,hold_preview(adapter.trajectory_initial_tcp))
        and np.array_equal(adapter.base_pose_hold,adapter.trajectory_initial_base),
        'EXACT_UNCHANGED_INITIAL_REFERENCE_DURING_TASK_PREPARATION')
    return dict(status='ORIGINAL_NATIVE_CERTIFICATE_AND_ACK_REQUIRED',native_step=step,
        source_pose_alignment=False,physics_advance_authorized=False)
