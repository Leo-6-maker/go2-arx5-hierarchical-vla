"""Observed native motion, tracking and consumption; no model capability proxy."""
import json
from pathlib import Path
import numpy as np
from .mobile_wbc_reference_v1 import START


def metrics(output,condition):
    path=Path(output)/'MOBILE_SAMPLES.private.jsonl'
    rows=[json.loads(x)for x in path.read_text().splitlines()]if path.exists()else[]
    task=[r for r in rows if r['native_start_step']>=START]
    if not task:return dict(task_atoms=0,movement_demonstrated=False)
    base=np.asarray([r['actual_mount']for r in task]);tcp=np.asarray([r['actual_TCP']for r in task])
    ref=np.asarray([r['target_endpoint_EEF10']for r in task])
    from scripts.pi0292.eval_contract import matrix10
    from ..trajectory.so3 import geodesic_angle
    desired=matrix10(ref)
    xyz=np.linalg.norm(tcp[:,:3,3]-desired[:,:3,3],axis=1)
    angle=geodesic_angle(desired[:,:3,:3],tcp[:,:3,:3])
    yaw=np.unwrap(np.arctan2(base[:,1,0],base[:,0,0]))
    # Initial pose is the actual sample immediately before the mobile window.
    initial=next((r for r in reversed(rows)if r['native_end_step']==START),None)
    origin=np.asarray(initial['actual_mount'])if initial else base[0]
    delta=base[-1,:3,3]-origin[:3,3]
    yaw_delta=float(np.arctan2(np.sin(yaw[-1]-np.arctan2(origin[1,0],origin[0,0])),
                              np.cos(yaw[-1]-np.arctan2(origin[1,0],origin[0,0]))))
    axis=condition['axis'];sign=1 if axis[0]=='+'else -1
    directional=float(sign*(yaw_delta if axis.endswith('yaw')else delta[0 if axis.endswith('x')else 1]))
    movement=directional>=(np.deg2rad(5)if axis.endswith('yaw')else .05)
    guard=Path(output)/'MOBILE_SAFETY_TRACE.private.jsonl'
    contacts=[json.loads(x)for x in guard.read_text().splitlines()if x.strip()]if guard.exists()else[]
    contacts=[r for r in contacts if r['kind']=='mobile_actual'and r['tick']>=START]
    return dict(task_atoms=len(task),task_integrations=sum(r['ticks']for r in task),
        actual_base_translation_m=delta.tolist(),actual_base_yaw_rad=yaw_delta,
        actual_base_path_m=float(np.linalg.norm(np.diff(base[:,:3,3],axis=0),axis=1).sum()),
        directional_motion=directional,movement_demonstrated=bool(movement),
        EEF_translation_RMS_m=float(np.sqrt(np.mean(xyz**2))),EEF_rotation_RMS_rad=float(np.sqrt(np.mean(angle**2))),
        exact_solver_target_confirmed_atoms=sum(bool(r['solver_exact_installed_targets'])for r in task),
        arm_request_atoms=sum(r['arm_requests']>0 for r in task),base_command_atoms=sum(r['base_commands']>0 for r in task),
        all_sampled_contacts_valid=all(r['safety_valid']for r in task),
        foot_contact_counts=sorted(set(r['foot_contacts']for r in contacts)),
        contact_tangent_speed_max_m_s=max((v for r in contacts for v in r['stance_tangent_speed_m_s']if v is not None),default=None),
        grasped_atoms=sum(r['native_grasp']for r in task),task_success='NOT_A_GRASP_TASK_M0_INTERFACE_ONLY',
        controller_selected_policy_identity='NOT_EXPORTED_DERIVED_ONLY')
