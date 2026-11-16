import numpy as np
from scipy.special import expit
from go2arx5_vla.sla_adapter.close_temporal_v1 import physical_gripper_terms


def test_physical_target_penalizes_open_forever_and_ignores_high_tau():
    target=np.zeros((2,40,32));target[:,:,6]=np.linspace(1,-1,40)
    stuck=target.copy();stuck[:,:,6]=1
    point,delta=physical_gripper_terms(stuck,target,np.array([.2,.9]),-9.21024,9.21024,xp=np,sigmoid=expit)
    assert point>0 and delta>0
    point2,delta2=physical_gripper_terms(stuck,target,np.array([.8,.9]),-9.21024,9.21024,xp=np,sigmoid=expit)
    assert point2==delta2==0
    perfect=physical_gripper_terms(target,target,np.array([.2,.4]),-9.21024,9.21024,xp=np,sigmoid=expit)
    assert perfect==(0.,0.)
