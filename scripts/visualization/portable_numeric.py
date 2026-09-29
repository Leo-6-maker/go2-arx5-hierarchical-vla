"""Saved-array CPU checks; no processor, model, controller or remote workspace."""
import numpy as np
from evidence_so3 import geodesic_angle,rot6d_rows_to_matrix,quat_to_matrix

# Same formulas as pi0243_staged_learning_to_sim._screen_metrics, H40 dt=.05.
def screen_metrics(model,expert):
    xyz_sq=np.sum((model[:,:3]-expert[:,:3])**2,axis=1)
    rotations=rot6d_rows_to_matrix(model[:,3:9]);gt=rot6d_rows_to_matrix(expert[:,3:9]);theta=geodesic_angle(rotations,gt)
    return dict(xyz_rmse_m=float(np.sqrt(np.mean(xyz_sq))),so3_rmse_rad=float(np.sqrt(np.mean(theta**2))),gripper_mae=float(np.abs(model[:,9]-expert[:,9]).mean()),first_xyz_error_m=float(np.sqrt(xyz_sq[0])),endpoint_xyz_error_m=float(np.sqrt(xyz_sq[-1])),max_linear_speed_mps=float(np.linalg.norm(np.diff(model[:,:3],axis=0),axis=1).max()/.05),max_angular_speed_radps=float(np.max(geodesic_angle(rotations[:-1],rotations[1:]))/.05))

def preparation_series(ticks):
    points=[];stable=None
    for r in ticks:
        p=r['post_step'];before=r['pre_step'];dt=p['time_s']-before['time_s']
        values=dict(position_error_m=float(np.linalg.norm(np.asarray(p['TCP_world'])-p['desired_world'])),rotation_error_rad=float(geodesic_angle(quat_to_matrix(p['quaternion_xyzw']),quat_to_matrix(p['desired_quaternion']))),speed_mps=float(np.linalg.norm(np.asarray(p['TCP_world'])-before['TCP_world'])/dt),angular_speed_radps=float(geodesic_angle(quat_to_matrix(p['quaternion_xyzw']),quat_to_matrix(before['quaternion_xyzw']))/dt))
        # Historical collector/replay used the previous post-step state for velocity.
        if points:
            prev=ticks[len(points)-1]['post_step'];dt=p['time_s']-prev['time_s']
            values['speed_mps']=float(np.linalg.norm(np.asarray(p['TCP_world'])-prev['TCP_world'])/dt)
            values['angular_speed_radps']=float(geodesic_angle(quat_to_matrix(p['quaternion_xyzw']),quat_to_matrix(prev['quaternion_xyzw']))/dt)
        for k,v in values.items():assert np.isclose(v,r[k],atol=1e-10,rtol=1e-10),(k,v,r[k])
        qualified=r['fresh'] and r['clean'] and not r['joint_limit_violation'] and all(values[k]<=limit for k,limit in [('speed_mps',.02),('angular_speed_radps',.20),('position_error_m',.035),('rotation_error_rad',.10)])
        stable=(r['time_s'] if stable is None else stable) if qualified else None
        assert stable==r['stable_since']
        points.append(dict(time_s=r['time_s'],**values,qualified=qualified,qualified_continuous_s=0. if stable is None else r['time_s']-stable,clean=r['clean'],joint_limit_violation=r['joint_limit_violation'],post_reset=r['pre_autoreset_event'] is not None))
    return points
