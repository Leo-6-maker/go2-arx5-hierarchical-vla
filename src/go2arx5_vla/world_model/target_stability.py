"""Recorded target motion decomposition; never constructs or integrates physics."""
import numpy as np


def rotation_wxyz(q):
    q=np.asarray(q,dtype=float)
    if q.shape[-1]!=4 or not np.isfinite(q).all() or not np.allclose(np.linalg.norm(q,axis=-1),1.,atol=1e-6,rtol=0):
        raise ValueError('UNIT_WXYZ_BASE_QUATERNION_REQUIRED')
    w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1).reshape(q.shape[:-1]+(3,3))


def diameter(points):
    x=np.asarray(points,dtype=float)
    if x.ndim!=2 or x.shape[1]!=3 or not len(x) or not np.isfinite(x).all():
        raise ValueError('FINITE_POSITION_SEQUENCE_REQUIRED')
    return float(np.linalg.norm(x[:,None]-x[None,:],axis=-1).max())


def motion(base_position,base_wxyz,tcp):
    b=np.asarray(base_position);p=np.asarray(tcp);r=rotation_wxyz(base_wxyz)
    if b.shape!=p.shape or p.shape!=(len(r),3) or not np.isfinite([b,p]).all():
        raise ValueError('MATCHED_BASE_TCP_SEQUENCE_REQUIRED')
    relative=np.einsum('nji,nj->ni',r,p-b)
    translation=b-b[0]
    rotation=np.einsum('nij,j->ni',r-r[0],relative[0])
    arm=np.einsum('nij,nj->ni',r,relative-relative[0])
    residual=p-p[0]-translation-rotation-arm
    if np.abs(residual).max()>1e-10:raise ValueError('BASE_ARM_DECOMPOSITION_IDENTITY_FAILED')
    return dict(relative_tcp=relative,base_translation=translation,base_rotation=rotation,relative_arm_world=arm,identity_residual_max_m=float(np.abs(residual).max()))


def screen(time_s,relative_tcp,*,duration_s=.4,diameter_max_m=.002,max_speed_m_s=.025):
    """Scientific prefix screen, never a replacement for native safety checks."""
    t=np.asarray(time_s,dtype=float);x=np.asarray(relative_tcp,dtype=float)
    if len(t)!=len(x) or len(t)<2 or not np.all(np.diff(t)>0):raise ValueError('MONOTONE_REAL_SCREEN_TIMES_REQUIRED')
    begin=np.searchsorted(t,t[-1]-duration_s-1e-9)
    if t[-1]-t[begin]<duration_s-1e-8:return dict(qualified=False,reason='INSUFFICIENT_REAL_WINDOW')
    d=diameter(x[begin:]);speed=np.linalg.norm(np.diff(x[begin:],axis=0)/np.diff(t[begin:])[:,None],axis=-1)
    vmax=float(speed.max())
    return dict(qualified=d<=diameter_max_m and vmax<=max_speed_m_s,relative_TCP_diameter_m=d,relative_TCP_max_speed_m_s=vmax,window_s=float(t[-1]-t[begin]),diameter_threshold_m=diameter_max_m,speed_threshold_m_s=max_speed_m_s,screen_is_native_safety=False)
