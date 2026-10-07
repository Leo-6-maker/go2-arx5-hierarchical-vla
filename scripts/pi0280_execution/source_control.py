"""Static source-only frame and clock conversion; no model or physics."""
import numpy as np
from scipy.spatial.transform import Rotation

C=np.array([[0,0,-1],[-1,0,0],[0,1,0.]])

def encode(rotation):
    return (Rotation.from_euler('XYZ',Rotation.from_matrix(rotation).as_euler('xyz')).as_matrix()@C)[:2].reshape(6)

def decode(rot6):
    a,b=rot6[...,:3],rot6[...,3:6]
    a=a/np.linalg.norm(a,axis=-1)[...,None]
    b=b-(a*b).sum(-1)[...,None]*a;b=b/np.linalg.norm(b,axis=-1)[...,None]
    matrix=np.stack([a,b,np.cross(a,b)],axis=-2)@C.T
    return Rotation.from_euler('xyz',Rotation.from_matrix(matrix).as_euler('XYZ')).as_matrix()

def next_tick(index):
    # ponytail: integer ceiling avoids floating-point boundary drift at 20/15 Hz.
    return 20+((index+1)*32+2)//3

def self_check():
    for angles in ([.34,-.23,1.12],[-1.1,.4,-2.2],[0,0,0]):
        r=Rotation.from_euler('xyz',angles).as_matrix()
        assert np.max(abs(decode(encode(r))-r))<1e-12
    ticks=np.array([next_tick(i) for i in range(60)])
    assert ticks[-1]==660 and set(np.diff(np.r_[20,ticks]))=={10,11}
    quantization=(ticks-20)*.05-np.arange(1,61)*8/15
    assert quantization.min()>-1e-12 and quantization.max()<.05
    assert np.allclose([1-(q-.00004)/.07996 for q in [.00004,.08]],[1,0])

if __name__=='__main__':self_check();print('SOURCE_FRAME_CLOCK_GRIPPER_CPU_CHECK_PASS')
