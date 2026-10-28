"""Source ABI conversions reused from the sealed PI0280 bridge."""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
BASE=Path('/home/liuyu-nfs/Projects/go2-arx5')
sys.path.insert(0,str(BASE/'PI0280/code'))
from source_control import encode,decode,next_tick,self_check as previous_check

def tool():
    result=np.eye(4);result[:3,:3]=Rotation.from_euler('z',np.pi/4).as_matrix();result[2,3]=-.1034
    return result

def native_to_hand(eef,root):
    poses=np.tile(np.eye(4),(len(eef),1,1));poses[:,:3,3]=eef[:,:3];poses[:,:3,:3]=decode(eef[:,3:9])
    return root@poses@np.linalg.inv(tool())

def hand_to_native(poses,root):
    native=np.linalg.inv(root)@poses@tool()
    return np.stack([np.r_[p[:3,3],encode(p[:3,:3])] for p in native])

def interpolate(poses,times,clock):
    clock=np.clip(clock,times[0],times[-1]);result=np.tile(np.eye(4),(len(clock),1,1))
    result[:,:3,3]=np.stack([np.interp(clock,times,poses[:,i,3]) for i in range(3)],axis=-1)
    result[:,:3,:3]=Slerp(times,Rotation.from_matrix(poses[:,:3,:3]))(clock).as_matrix()
    return result

def native_index(now,anchor):return min(7,max(0,int(np.searchsorted(anchor+np.arange(8)/15,now,side='right')-1)))

def expert_bridge(commands,root):
    assert commands.shape==(640,10)
    poses=np.tile(np.eye(4),(640,1,1));poses[:,:3,3]=commands[:,:3]
    from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix
    poses[:,:3,:3]=rot6d_rows_to_matrix(commands[:,3:9])
    times=np.arange(640)/20;native_times=np.arange(481)/15
    native_poses=interpolate(poses,times,native_times)
    encoded=hand_to_native(native_poses,root);decoded=native_to_hand(encoded,root)
    bridge=interpolate(decoded,native_times,times)
    # ponytail: retain original 20Hz gripper events; 15Hz pose sampling must not erase grasp/release.
    positions=np.linalg.norm(bridge[:,:3,3]-poses[:,:3,3],axis=-1)
    rotations=Rotation.from_matrix(bridge[:,:3,:3]@poses[:,:3,:3].transpose(0,2,1)).magnitude()
    report=dict(pose_samples20Hz=640,pose_samples15Hz=481,grid='20Hz ->15Hz throughSE3 interpolation -> officialDROID frame roundtrip ->20Hz',
        translation_RMS_m=float(np.sqrt(np.mean(positions**2))),translation_max_m=float(positions.max()),
        rotation_RMS_rad=float(np.sqrt(np.mean(rotations**2))),rotation_max_rad=float(rotations.max()),
        exact_gripper_event_ticks=np.flatnonzero(np.diff(commands[:,-1])!=0).tolist(),gripper_resampling='NONE_ORIGINAL20HZ_EVENT_HOLD',
        roundtrip_position_max_m=float(np.max(abs(decoded[:,:3,3]-native_poses[:,:3,3]))),
        roundtrip_rotation_max_rad=float(Rotation.from_matrix(decoded[:,:3,:3]@native_poses[:,:3,:3].transpose(0,2,1)).magnitude().max()))
    return bridge,commands[:,-1].copy(),report

def self_check():
    previous_check();root=np.eye(4);root[:3,3]=[-.615,.1,.2];root[:3,:3]=Rotation.from_euler('xyz',[.1,-.2,.3]).as_matrix()
    poses=np.tile(np.eye(4),(3,1,1));poses[:,:3,3]=[[.1,.2,.3],[.2,.4,.1],[-.1,.4,.2]]
    poses[:,:3,:3]=Rotation.from_euler('xyz',[[.3,-.2,1.1],[-1.,.3,-.2],[.1,.2,.3]]).as_matrix()
    assert np.max(abs(native_to_hand(hand_to_native(poses,root),root)-poses))<1e-12
    assert np.array_equal(interpolate(poses,np.array([0.,1.,2.]),np.array([0.,1.,2.])),poses) or np.max(abs(interpolate(poses,np.array([0.,1.,2.]),np.array([0.,1.,2.]))-poses))<1e-12
    assert native_index(.5,0)==7 and native_index(0,0)==0 and native_index(7/15,0)==7
    assert [next_tick(i) for i in [0,1,2,59]]==[31,42,52,660]

if __name__=='__main__':self_check();print('SOURCE_ABI_CPU_SELF_CHECK_PASS')
