"""Nonlinear cloned-FK fixture exercises the fifth update and remaining guards.

This is a software certificate fixture, not a native robot model or execution.
"""
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.pi0294 import task_path_model as cert


def fixture(monkeypatch, start=.05, limit=100., contact=False):
    model=SimpleNamespace(nv=6,jnt_qposadr=np.arange(6),jnt_dofadr=np.arange(6),geom_bodyid=np.array([0,1]),
        jnt_range=np.tile([-100.,100.],(6,1)))
    model.jnt_range[0,1]=limit
    def data(model):
        return SimpleNamespace(qpos=np.zeros(6),qvel=np.zeros(6),ctrl=np.zeros(0),
            site_xpos=np.zeros((1,3)),site_xmat=np.eye(3).reshape(1,9),
            contact=[SimpleNamespace(geom1=0,geom2=1,dist=-.001)]if contact else [])
    def forward(model,d):
        d.site_xpos[0]=[d.qpos[0]**2,0,0]
    def jac(model,d,jp,jr,site):
        jp[:]=0;jr[:]=0;jp[0,0]=2*d.qpos[0]
    def name2id(model,kind,name):
        return int(name.removeprefix('x5_joint'))-1 if name.startswith('x5_joint')else 0
    fake=SimpleNamespace(MjData=data,mj_forward=forward,mj_jacSite=jac,mj_name2id=name2id,
        mj_id2name=lambda model,kind,index:['arm','task_cube'][index],
        mjtObj=SimpleNamespace(mjOBJ_JOINT=0,mjOBJ_SITE=1,mjOBJ_GEOM=2,mjOBJ_BODY=3))
    monkeypatch.setitem(sys.modules,'mujoco',fake)
    monkeypatch.setattr(cert,'contact_permitted',lambda names,dist,stage:False)
    q=np.zeros(6);q[0]=start
    sample=dict(qpos=q,qvel=np.zeros(6),ctrl=np.zeros(0),native_step=0)
    poses=np.tile(np.eye(4),(40,1,1));poses[:,0,3]=.04
    return model,sample,poses


def test_fifth_update_is_checked_at_original_tolerance_without_sixth_update(monkeypatch):
    m,s,p=fixture(monkeypatch)
    initial=s['qpos'].copy()
    # This exact nonlinear case needs five Newton updates: after four, the
    # residual is above 1e-5; after five, below it.
    q=.05;errors=[]
    for i in range(5):
        q+=(.04-q*q)/(2*q);errors.append(abs(.04-q*q))
    assert errors[3]>1e-5 and errors[4]<=1e-5
    r=cert.certify_chunk(m,s,p,stage='approach')
    assert r['maximum_IK_updates_per_point']==5 and r['position_tolerance_m']==1e-5
    assert r['rotation_tolerance_rad']==1e-4 and r['physics_integrations']==0
    assert r['local_IK_iterations']==5+1000*2
    np.testing.assert_array_equal(s['qpos'],initial)


@pytest.mark.parametrize('fault',('not_converged','joint_limit','contact'))
def test_final_update_fix_still_rejects_original_failures(monkeypatch,fault):
    m,s,p=fixture(monkeypatch,start=.01 if fault=='not_converged'else .05,
        limit=.1 if fault=='joint_limit'else 100.,contact=fault=='contact')
    reason={'not_converged':'FIVE_ITERATION','joint_limit':'JOINT_LIMIT','contact':'CONTACT_REJECT'}[fault]
    with pytest.raises(ValueError,match=reason):cert.certify_chunk(m,s,p,stage='approach')
