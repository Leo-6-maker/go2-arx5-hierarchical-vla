"""Read-only arithmetic for a proposal. No solver, renderer or runtime imports.

Input: six privately retained retarget NPZs and the sealed extraction/gate index.
Output: timing lower bounds and source joins, never a physical admission receipt.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def so3_angles(a, b):
    r=np.swapaxes(a,-1,-2)@b
    skew=np.stack([r[...,2,1]-r[...,1,2],r[...,0,2]-r[...,2,0],r[...,1,0]-r[...,0,1]],axis=-1)/2
    return np.arctan2(np.linalg.norm(skew,axis=-1),np.clip((np.trace(r,axis1=-2,axis2=-1)-1)/2,-1,1))


def analyze(folder):
    started=time.process_time();folder=Path(folder)
    identity=json.loads((folder/'IDENTITY.private.json').read_text(encoding='utf-8'))
    rows=[]
    for record in identity['gate']['rows']:
        file=folder/record['retarget_file'];data=file.read_bytes()
        if hashlib.sha256(data).hexdigest()!=record['retarget_sha256']:
            raise ValueError('SEALED_INPUT_CHANGED')
        with np.load(file,allow_pickle=False) as f:
            poses=f['poses'].copy();gripper=f['gripper'].copy();source_indices=f['source_indices'].copy()
        base=dict(id=record['id'],task=record['task'],input_sha256=record['retarget_sha256'],
            old_uniform_duration_s=record['active_duration_s'],physical_admission=False)
        if record['task']=='T2':
            distance=np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=1)
            angles=so3_angles(poses[:-1,:3,:3],poses[1:,:3,:3])
            dt=np.maximum.reduce([np.full(len(distance),.05),distance/.06,angles/.3])
            # A rounding allowance only for representation of exact grid ratios;
            # the resulting rate is independently checked, never clipped.
            ticks=np.maximum(1,np.ceil(dt/.05-1e-10).astype(np.int64))
            durations=ticks*.05;time_grid=np.r_[0.,np.cumsum(durations)]
            rate_v=float(np.max(distance/durations));rate_w=float(np.max(angles/durations))
            if rate_v>.06*(1+1e-10) or rate_w>.3*(1+1e-10):
                raise ValueError('ROUNDING_FAILED_UNCHANGED_RATE_BOUND')
            base.update(proposed_change='LOCAL_INTERVAL_STRETCH_ONLY',
                source_knots=len(poses),source_start_index=int(source_indices[0]),source_end_index=int(source_indices[-1]),
                unconstrained_by_joints_cartesian_lower_bound_s=float(dt.sum()),
                quantized_cartesian_schedule_s=float(ticks.sum()*.05),
                possible_entry_time_before_joint_checks_s=float(min(10.,28-ticks.sum()*.05)),
                max_linear_m_s=rate_v,max_angular_rad_s=rate_w,extra_20Hz_intervals=int((ticks-1).sum()),
                timestamp_f64le_sha256=hashlib.sha256(time_grid.astype('<f8').tobytes()).hexdigest(),
                source_waypoints_deleted=0,pose_values_changed=False,gripper_values_changed=False,
                joint_velocity_acceleration_or_collision_proven=False)
        else:
            extraction=next(r for r in identity['source']['records'] if r['id']==record['id'])
            close=next(e['time_s'] for e in extraction['events'] if e['stage']=='close')
            index=int(np.searchsorted(np.arange(640)*.05,close))
            local=int(np.searchsorted(source_indices,index))
            duration=(int(source_indices[-1])-index)*.05*record['slowdown']
            base.update(proposed_change='DIRECT_JOIN_FIRST_CLOSE_KEEP_UNIFORM_TIMING',
                old_cut_start_index=int(source_indices[0]),new_cut_start_index=index,new_cut_source_time_s=index*.05,
                end_source_index=int(source_indices[-1]),retained_submitted_knots=len(poses)-local,
                removed_source_alignment_knots=local,join_original_logical_gripper=float(gripper[local]),
                retained_uniform_slowdown=record['slowdown'],suffix_uniform_duration_s=float(duration),
                join_pose_f64le_sha256=hashlib.sha256(poses[local].astype('<f8').tobytes()).hexdigest(),
                suffix_pose_f64le_sha256=hashlib.sha256(poses[local:].astype('<f8').tobytes()).hexdigest(),
                new_goal_IK_calls=0,route_or_dynamic_tracking_proven=False)
        rows.append(base)
    return dict(schema='SOURCE_LOCAL_RETIME_DIRECT_JOIN_V2_PREPARATION_ARITHMETIC',
        status='ANALYSIS_ONLY_NOT_RUNTIME_OR_PHYSICS_ADMISSION',rows=rows,
        source_result_release='770989874a59447449d649ed3401340b016b66c1',
        new_IK_calls=0,new_RRT_calls=0,new_physics=0,new_render=0,new_neural_queries=0,
        authorizes_execution=False,launch_allowed=False,PC_CPU_process_s=time.process_time()-started)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    output=Path(a.output)
    if output.exists():raise FileExistsError('DO_NOT_OVERWRITE_FIRST_ANALYSIS')
    output.write_text(json.dumps(analyze(a.input),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
