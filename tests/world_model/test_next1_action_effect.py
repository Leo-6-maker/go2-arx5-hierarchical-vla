import unittest
from unittest.mock import patch
import numpy as np
from scripts.wm02.audit_action_effect import run


class ActionEffectGate(unittest.TestCase):
    def fixture(self,response):
        offsets=np.asarray([[0,0,0],[.001,0,0],[-.001,0,0],[0,.001,0],[0,0,0]]*4)
        state=np.zeros((20,25,34));state[:,:,3]=1
        tcp=np.tile(np.eye(4),(20,25,1,1))
        for i in range(20):
            for h in range(1,17):
                noise=(1 if i%5==0 else-1)*.00001 if i%5 in(0,4)else 0.
                tcp[i,8+h,:3,3]=(offsets[i]*response+noise)*h/16
        data=dict(state=state,tcp=tcp,goal=np.c_[offsets,np.zeros((20,14))],history_goal=np.zeros((20,8,17)))
        meta=dict(collection_protocol='FIXED_HOLD40_BURN8_DECIDE_RAMP8_V1',episodes=[dict(episode=str(i),parent=str(i//5),split='TRAIN')for i in range(20)])
        return data,meta

    def test_real_response_gate_does_not_pass_zero_effect_or_invent_hold_noise(self):
        with patch('go2arx5_vla.world_model.storage.digest',return_value='fixture'):
            with patch('go2arx5_vla.world_model.target_replay.load',return_value=self.fixture(1.)):
                self.assertTrue(run('fixture')['gate_passed'])
            with patch('go2arx5_vla.world_model.target_replay.load',return_value=self.fixture(0.)):
                self.assertFalse(run('fixture')['gate_passed'])
            data,meta=self.fixture(1.);meta['episodes'][-1]['split']='DEV'
            with patch('go2arx5_vla.world_model.target_replay.load',return_value=(data,meta)):
                with self.assertRaises(ValueError):run('fixture')
