"""Independent noncommuting decoder boundary, using non-learned CPU tensors."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation

def check():
    import pi0260_r3_physical as physical
    import pi0248_rotation_cpu as rotation
    physical.install()
    anchor=Rotation.from_euler('xyz',[.31,-.47,.83]).as_matrix()
    relative=Rotation.from_euler('xyz',[-.19,.37,.41]).as_matrix()
    values=np.repeat((relative[:2].reshape(6)*1.7)[None],40,axis=0).astype(np.float32)
    state=torch.tensor(np.r_[.3,-.2,.1,anchor[:2].reshape(6)][None],dtype=torch.float32)
    lo=torch.full((40,6),-1.);hi=-lo
    got,detail=rotation.decode_relative_rotation(torch.tensor(values[None]),state[:,3:9],lo,hi)
    assert np.max(abs(values))>1
    assert np.max(abs(detail['relative_physical_rot6d'].numpy()[0]-values))<2e-7
    expected=np.repeat((anchor@relative)[None],40,axis=0)
    error=float(np.max(abs(got.numpy()[0]-expected)));assert error<4e-7
    assert np.max(abs(anchor@relative-relative@anchor))>.01
    normalized=torch.full((1,40,3),1.7);scale=torch.tensor(np.repeat([[.03,.04,.05]],40,axis=0),dtype=torch.float32)
    offset=torch.tensor(np.repeat([[.002,-.003,.001]],40,axis=0),dtype=torch.float32)
    xyz=physical.decode_xyz(normalized,state,scale,offset).numpy()[0]
    reference=(normalized.numpy()[0]*scale.numpy()+offset.numpy())@anchor.T+state.numpy()[0,:3]
    xyz_error=float(np.max(abs(xyz-reference)));assert xyz_error<1e-7
    return dict(status='PASS_NONCOMMUTING_UNCLAMPED_ROT_XYZ_FP32',rotation_max_abs=error,xyz_max_abs=xyz_error,normalized_abs_max=float(abs(values).max()))

if __name__=='__main__':print(check())
