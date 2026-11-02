"""The diagnostic must execute native control and never its integration tail."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.pi0294.multitask.controller_replay import control_prefix,replay_tick

SOURCE=b'''class Engine:
    def step(self,target,opening):
        self.command_ik(target)
        self.arm_targets[:]=target
        self.data.ctrl[:]=target+opening
        mujoco.mj_step(self.model,self.data)
        raise RuntimeError('POST_INTEGRATION_TAIL_MUST_NOT_RUN')
'''


def test_exact_control_prefix_excludes_integration_and_tail():
    calls=[]
    def forbidden(*args):raise AssertionError('PHYSICS_WAS_INTEGRATED')
    prefix=control_prefix(SOURCE,hashlib.sha256(SOURCE).hexdigest(),dict(mujoco=SimpleNamespace(mj_step=forbidden)))
    e=SimpleNamespace(arm_targets=np.zeros(2),data=SimpleNamespace(ctrl=np.zeros(2)),command_ik=lambda t:calls.append(t.copy()))
    prefix(e,np.array([1.,2.]),.5)
    np.testing.assert_array_equal(e.data.ctrl,[1.5,2.5]);assert len(calls)==1
    with pytest.raises(PermissionError):control_prefix(SOURCE,'0'*64,{})
    with pytest.raises(ValueError):
        body=SOURCE.replace(b'mujoco.mj_step(self.model,self.data)',b'mujoco.mj_step1(self.model,self.data)')
        control_prefix(body,hashlib.sha256(body).hexdigest(),{})


def test_force_validation_uses_saved_states_without_integrating():
    e=SimpleNamespace(model=SimpleNamespace(nq=2,nv=2,na=0),data=SimpleNamespace(qpos=np.zeros(2),qvel=np.zeros(2),time=0.,ctrl=np.zeros(2),actuator_force=np.zeros(2)),
        arm_targets=np.zeros(2),finger_targets=np.zeros(2),arm_bounds=np.array([[-2.,2.],[-2.,2.]]),_cached_ik_solution=np.array([1.,1.]),ik_solve_calls=1,ik_cache_hits=0)
    def forward(model,data):data.actuator_force[:]=data.ctrl*2
    def prefix(engine,target,opening):engine.arm_targets[:]=target;engine.data.ctrl[:]=[3.,4.]
    before=dict(qpos=[0.,0.],qvel=[0.,0.],time_s=0.)
    after=dict(qpos=[.9,1.1],qvel=[.2,.3],time_s=.002,arm_q=[.9,1.1],actuator_force=[6.,8.])
    result=replay_tick(e,prefix,[1.,1.],0.,before,after,SimpleNamespace(mj_forward=forward))
    assert result['force_max_abs_error']==0.
    np.testing.assert_allclose(result['actual_minus_command'],[-.1,.1])
    e.model.na=1
    with pytest.raises(PermissionError):replay_tick(e,prefix,[1.,1.],0.,before,after,SimpleNamespace(mj_forward=forward))
