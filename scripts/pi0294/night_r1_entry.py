"""Declared finite precontact connector; immutable skill and original gates."""
import numpy as np


def package(selected, design):
    if (design['schema']!='TFNR1_PRECONTACT_ENTRY_V1'
            or design['source_id']!=selected['source_id']
            or design['source_program_sha256']!=selected['source_program_sha256']
            or design['distance_m'] not in (.01,.02,.04)):
        raise ValueError('FROZEN_PRECONTACT_DESIGN_BINDING_REQUIRED')
    direction=np.asarray(design['source_approach_unit_world'],dtype=float)
    if direction.shape!=(3,) or not np.isfinite(direction).all() or abs(np.linalg.norm(direction)-1)>1e-10:
        raise ValueError('SOURCE_APPROACH_DIRECTION_REQUIRED')
    pre=np.asarray(selected['contact']).copy()
    pre[:3,3]-=design['distance_m']*direction
    return dict(selected,approach=pre,direct_join=False,entry_contact_stage='contact',
        entry_design_sha256=design['design_sha256'])


def phases(entry, response, certificate):
    links=response['links']
    if len(links)!=2 or any(x['status']!='Success' for x in links):
        raise ValueError('PAIRED_APPROACH_CONTACT_ENTRY_REQUIRED')
    boundary=(float(entry['time'][len(links[0]['time'])-1]) if 'entry_timing' in entry
        else float(links[0]['time'][-1])*certificate['uniform_time_scale'])
    if not 0<boundary<float(entry['time'][-1]):
        raise ValueError('ENTRY_PHASE_BOUNDARY_REQUIRED')
    return dict(entry,free_approach_end_s=boundary,contact_connector_end_s=float(entry['time'][-1]))


def stage(entry, at, task):
    if at<entry.get('free_approach_end_s',entry['time'][-1])-1e-12:
        return 'approach' if task=='T1' else 'contact'
    return 'contact'


def retime(entry, joint_path):
    """Use the native interpolation peak and real 50ms grid, not just knot averages."""
    from .night_r1_source import local_schedule,sample
    from .source_retime_v2_analysis import so3_angles
    from go2arx5_vla.trajectory.so3 import matrix_to_quat
    before=np.asarray(entry['time']);poses=np.asarray(entry['poses']);dt=np.diff(before)
    angles=so3_angles(poses[:-1,:3,:3],poses[1:,:3,:3])
    quats=matrix_to_quat(poses[:,:3,:3]);dots=np.abs(np.sum(quats[:-1]*quats[1:],axis=1))
    # The existing quat_slerp uses normalized linear interpolation above .9995.
    # Its peak angular derivative is 4*tan(rotation_angle/4), not angle.
    angular_peak=np.where(dots>.9995,4*np.tan(angles/4),angles)
    needed=np.maximum.reduce([dt,np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=1)/.06,angular_peak/.3])
    seeded=np.r_[0.,np.cumsum(needed)]
    times,proof=local_schedule(poses,np.asarray(joint_path),seeded)
    grid=np.arange(int(np.ceil(times[-1]/.05))+1)*.05
    sampled,_=sample(dict(time=times,poses=poses,gripper=np.ones(len(times))),grid)
    speed=np.linalg.norm(np.diff(sampled[:,:3,3],axis=0),axis=1)/.05
    omega=so3_angles(sampled[:-1,:3,:3],sampled[1:,:3,:3])/.05
    if speed.max()>.06*(1+1e-10) or omega.max()>.3*(1+1e-10):
        raise ValueError('FULL_NATIVE_INTERPOLATION_RATE_REJECT')
    proof.update(schema='QUANTIZED_NATIVE_SLERP_R1',original_duration_s=float(before[-1]),
        duration_s=float(times[-1]),full_grid_max_linear_speed=float(speed.max()),full_grid_max_angular_speed=float(omega.max()),
        native_interpolation_unchanged=True,entry_poses_changed=False)
    return dict(entry,time=times,entry_timing=proof)
