"""Read original 500Hz endpoints and 20Hz divergence, without forward/integration."""
from collections import defaultdict
import itertools

import numpy as np

from scripts.pi0294.fd1 import read,sha,actual_phase


def native_object_pose(qpos):
    from go2arx5_vla.trajectory.so3 import quat_to_matrix
    qpos=np.asarray(qpos)
    if qpos.shape!=(15,) or not np.isfinite(qpos).all():
        raise ValueError('FD1_ORIGINAL_6ARM_2FINGER_LAST7_CUBE_QPOS')
    pose=np.eye(4)
    pose[:3,3]=qpos[-7:-4]
    wxyz=qpos[-4:]
    pose[:3,:3]=quat_to_matrix(np.r_[wxyz[1:],wxyz[0]])
    return pose


def in_zone(pose,goal):
    corners=np.asarray(list(itertools.product((-.02,.02),repeat=3))) @ pose[:3,:3].T + pose[:3,3]
    return bool((np.abs(corners[:,:2]-np.asarray(goal))<=.07).all())


def sample_divergence(left,right):
    n=min(len(left),len(right));state_index=image_index=None;difference=None
    for i in range(n):
        a,b=left[i],right[i]
        x=np.asarray(a['execution']['actual_tcp']);y=np.asarray(b['execution']['actual_tcp'])
        gap=abs(a['execution']['actual_gripper_gap_m']-b['execution']['actual_gripper_gap_m'])
        error=max(float(np.max(np.abs(x-y))),gap)
        if state_index is None and error>1e-12:
            state_index=i;difference=error
        if image_index is None and any(a['policy_obs'][c]['sha256']!=b['policy_obs'][c]['sha256'] for c in ['body_rgb','wrist_rgb']):
            image_index=i
    return dict(first_original_20Hz_camera_difference_s=None if image_index is None else image_index*.05,
        first_original_20Hz_TCP_or_gripper_difference_s=None if state_index is None else state_index*.05,
        first_state_maximum_component_difference=difference,state_difference_threshold=1e-12,
        first_different_camera_frame=image_index,compared_original_paired_frames=n,temporal_resolution_s=.05,
        new_render_or_physics=0)


def run_endpoints(meter):
    atlas=read(meter.root/'evidence/ATLAS_RESULT.private.json');rows=[];samples={};groups=defaultdict(list)
    for row in atlas['actual_rows']:
        meter.guard();identifier=row['attempt_id'];p=meter.source/'episodes'/identifier
        for n in ['highrate.npz','physics.npz','samples.json','CARD.json']:
            if sha(p/n)!=row['original_files_sha256'][n]:
                raise ValueError('FD1_ORIGINAL_ENDPOINT_MEMBER_CHANGED')
        card=read(p/'CARD.json');samples[identifier]=read(p/'samples.json')
        with np.load(p/'highrate.npz',allow_pickle=False) as z:
            high=z['rows'].copy();nq=int(z['nq'])
        with np.load(p/'physics.npz',allow_pickle=False) as z:
            micro=z['contact_audit'].copy();times=z['contact_time_s'].copy();frame_t=z['time_s'].copy()
        if nq!=15 or high.shape[0]!=len(micro) or len(high)==0:
            raise ValueError('FD1_ORIGINAL_NATIVE_ENDPOINT_SHAPE')
        pose=native_object_pose(high[-1,1:1+nq])
        if abs(pose[2,3]-micro[-1,3])>1e-12:
            raise ValueError('FD1_ORIGINAL_CUBE_QPOS_HIGH_CONTACT_HEIGHT_PARITY')
        contact=bool(micro[-1,0]);grasp=bool(micro[-1,1]);zone=in_zone(pose,card['goal_xy'])
        phase=actual_phase(card['task'],bool(micro[:,0].any()),bool(micro[:,2].max()>=.001),
            bool(micro[:,2].max()>=.1),grasp,not grasp,not contact,zone)
        rows.append(dict(attempt_id=identifier,actual_endpoint_time_s=float(times[-1]),
            last_original_20Hz_pose_time_s=float(frame_t[-1]),actual_endpoint_robot_contact=contact,
            actual_endpoint_native_grasp=grasp,actual_endpoint_object_in_zone=zone,
            actual_endpoint_object_XYZ_m=pose[:3,3].tolist(),
            actual_endpoint_goal_center_error_m=float(np.linalg.norm(pose[:2,3]-card['goal_xy'])),
            actual_phase=phase,previous_coarse_20Hz_phase=row['actual_events']['actual_phase'],
            actual_phase_changed_vs_coarse=phase!=row['actual_events']['actual_phase'],
            actual_endpoint_evidence='ORIGINAL_500HZ_QPOS_AND_CONTACT_NO_NEW_FORWARD',
            original_highrate_sha256=row['original_files_sha256']['highrate.npz']))
        if row['adjudication']['status']!='ENGINEERING_INCOMPLETE':
            groups[(card['task'],card['scene_group'])].append(identifier)
    pairs=[]
    for (task,scene),ids in sorted(groups.items()):
        if len(ids)!=2:raise ValueError('FD1_ORIGINAL_TWO_BANKS_AT_SAMPLE_GRAIN')
        ids=sorted(ids)
        pairs.append(dict(task=task,scene_group=scene,attempt_ids=ids,**sample_divergence(samples[ids[0]],samples[ids[1]])))
    return dict(status='CPU_ORIGINAL_NATIVE_ENDPOINT_AND_20HZ_DIVERGENCE_PASS',attempts=rows,pairs=pairs,
        original_attempts=42,original_conditions=40,new_model_queries=0,new_physics=0,shadow_starts=0)
