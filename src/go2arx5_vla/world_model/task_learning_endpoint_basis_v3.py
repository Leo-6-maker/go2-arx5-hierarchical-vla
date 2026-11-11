"""Explicit predicted-goal representation study; no native execution authority.

Raw C500 predictions stay immutable. A current-state anchor and the model's
last decoded pose/grip define a full40 minimum-jerk curve at the original50ms
clock. This is a different composed policy, not a repaired raw-C500 result.
No object truth, Teacher, per-knot clipping, retiming or WBC change is used.
"""
import numpy as np
from ..trajectory.so3 import rot6d_rows_to_matrix,rotation_matrix_to_rot6d_rows
from ..trajectory.so3 import matrix_to_quat,quat_to_matrix,quat_slerp

VERSION='H40_CURRENT_STATE_PREDICTED_GOAL_QUINTIC_REPRESENTATION_V3_01'
DT=.05


def decode(raw,state10,*,consumed_nominal_grip):
    raw=np.asarray(raw,float);state=np.asarray(state10,float)
    if raw.shape!=(40,10)or state.shape!=(10,)or not np.isfinite(raw).all()or not np.isfinite(state).all():
        raise ValueError('Original complete finite H40 and current policy state required')
    if not np.all((raw[:,9]>=0)&(raw[:,9]<=1))or not np.isfinite(consumed_nominal_grip)or not 0<=consumed_nominal_grip<=1:
        raise ValueError('Original nominal gripper range; no clipping or fallback')
    rotations=rot6d_rows_to_matrix(np.stack([state[3:9],raw[-1,3:9]]))
    u=np.linspace(0.,1.,40);w=u**3*(10.+u*(-15.+6.*u))
    result=np.empty_like(raw)
    result[:,:3]=state[:3]+w[:,None]*(raw[-1,:3]-state[:3])
    q=quat_slerp(matrix_to_quat(rotations[0]),matrix_to_quat(rotations[1]),w[:,None])
    result[:,3:9]=rotation_matrix_to_rot6d_rows(quat_to_matrix(q))
    result[:,9]=consumed_nominal_grip+w*(raw[-1,9]-consumed_nominal_grip)
    # Preserve the two declared representation endpoints exactly. This is one
    # global basis construction, never a componentwise action correction.
    result[0,:9]=state[:9];result[0,9]=consumed_nominal_grip;result[-1]=raw[-1]
    return result,dict(version=VERSION,raw_GR00T_promoted=False,raw_output_preserved=True,
        representation='CURRENT_EEF9_AND_CONSUMED_NOMINAL_GRIP_TO_PREDICTED_FINAL_EEF9_GRIP1',
        goal_source='SAME_ORIGINAL_PROCESSOR_DECODE_LAST_KNOT',knots=40,action_period_s=DT,
        last_knot_time_s=39*DT,full_prefix_intervals=40,time_retiming=False,
        command_clips=0,Teacher_fallback=False,object_truth_used=False,
        original_native_certificate_required=True,native_execution_authority=False,
        maximum_XYZ_representation_difference_m=float(np.linalg.norm(result[:,:3]-raw[:,:3],axis=1).max()),
        maximum_gripper_representation_difference=float(np.abs(result[:,9]-raw[:,9]).max()))
