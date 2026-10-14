"""EEF13 world reference conversion; no controller, physics or model calls."""
import math
import numpy as np
from scripts.pi0292.eval_contract import matrix10
from go2arx5_vla.trajectory.so3 import matrix_to_quat

DT=.05
SPEED=np.array([.15,.10,.30])
ACCEL=np.array([.30,.20,.60])


def rotation(yaw):
    c,s=math.cos(yaw),math.sin(yaw)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])


def mount_pose(value):
    m=np.asarray(value,dtype=float)
    if m.shape!=(4,4) or not np.isfinite(m).all():raise ValueError('FINITE_MOUNT_REQUIRED')
    r=m[:3,:3]
    if not np.allclose(r.T@r,np.eye(3),atol=1e-10) or not np.allclose(m[3],[0,0,0,1],atol=1e-12) or abs(np.linalg.det(r)-1)>1e-10:
        raise ValueError('ACTUAL_SE3_MOUNT_REQUIRED')
    return m


def base_endpoints(mount,velocity):
    """Exact SE2 exponential; preserve measured height/roll/pitch."""
    mount=mount_pose(mount);v=np.asarray(velocity,dtype=float)
    if v.shape!=(40,3) or not np.isfinite(v).all():
        raise ValueError('FINITE_MOUNT_AND_FORTY_PLANAR_INTERVALS_REQUIRED')
    r=mount[:3,:3]
    yaw0=math.atan2(r[1,0],r[0,0]);yaw=yaw0
    result=[mount.copy()]
    for vx,vy,w in v:
        angle=w*DT
        # numpy sinc keeps the exact limit at zero without cancellation.
        a=DT*np.sinc(angle/np.pi)
        b=DT*(angle/2)*np.sinc(angle/(2*np.pi))**2
        delta=np.array([a*vx-b*vy,b*vx+a*vy,0])
        nxt=result[-1].copy();nxt[:3,3]+=rotation(yaw)@delta
        yaw+=angle;nxt[:3,:3]=rotation(yaw-yaw0)@r;result.append(nxt)
    return np.asarray(result)


def native_document(action,mount,*,base_valid,previous_velocity,speed=SPEED,acceleration=ACCEL):
    """No implicit clipping, body attachment, or second relative EEF transform."""
    a=np.asarray(action,dtype=float);previous=np.asarray(previous_velocity,dtype=float)
    speed=np.asarray(speed,dtype=float);acceleration=np.asarray(acceleration,dtype=float)
    if (a.shape!=(40,13) or previous.shape!=(3,) or type(base_valid) is not bool
            or not np.isfinite(a).all() or not np.isfinite(previous).all()
            or speed.shape!=(3,) or acceleration.shape!=(3,)
            or not np.isfinite(speed).all() or not np.isfinite(acceleration).all()
            or np.any(speed<=0) or np.any(speed>SPEED) or np.any(acceleration<=0) or np.any(acceleration>ACCEL)):
        raise ValueError('FIXED_EEF13_AND_TIGHTER_QUALIFIED_ENVELOPE_REQUIRED')
    v=a[:,10:13]
    if (np.any(a[:,9]<0) or np.any(a[:,9]>1) or np.any(np.abs(v)>speed+1e-12)
            or np.any(np.abs(np.diff(np.vstack([previous,v]),axis=0))>acceleration*DT+1e-12)):
        raise ValueError('OUT_OF_ENVELOPE_REJECT_WITHOUT_CORRECTION')
    eef=matrix10(a[:,:10]);eef=np.concatenate([eef,eef[-1:]])
    def waypoints(poses):
        return [dict(time=k*DT,position=p[:3,3].tolist(),quaternion_xyzw=matrix_to_quat(p[:3,:3]).tolist()) for k,p in enumerate(poses)]
    document=dict(version=1,ee=dict(frame='world',start_delay=0.,lead_in=0.,waypoints=waypoints(eef)))
    if base_valid:document['base']=dict(frame='world',start_delay=0.,lead_in=0.,waypoints=waypoints(base_endpoints(mount,v)))
    return document


def measured_base(imu,linear_body,omega_body,mount_offset):
    """Move velocity origin before projecting to yaw; heading rate is not omega-z."""
    imu=mount_pose(imu);linear=np.asarray(linear_body,dtype=float);omega=np.asarray(omega_body,dtype=float);offset=np.asarray(mount_offset,dtype=float)
    if imu.shape!=(4,4) or any(x.shape!=(3,) for x in (linear,omega,offset)) or not all(np.isfinite(x).all() for x in (imu,linear,omega,offset)):
        raise ValueError('FINITE_MEASURED_BASE_FIELDS_REQUIRED')
    r=imu[:3,:3];yaw=math.atan2(r[1,0],r[0,0]);pitch=math.asin(np.clip(-r[2,0],-1,1));roll=math.atan2(r[2,1],r[2,2])
    if abs(math.cos(pitch))<1e-6:raise ValueError('HEADING_SINGULARITY')
    mount=imu.copy();mount[:3,3]+=r@offset
    velocity=rotation(-yaw)@(r@(linear+np.cross(omega,offset)))
    heading_rate=(math.sin(roll)*omega[1]+math.cos(roll)*omega[2])/math.cos(pitch)
    return mount,np.array([velocity[0],velocity[1],heading_rate])
