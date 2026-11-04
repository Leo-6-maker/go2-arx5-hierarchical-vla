import unittest
import numpy as np
from go2arx5_vla.world_model.target_arx import CausalARX


class CausalARXBaseline(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(7)
        state=np.zeros((12,25,6));goal=np.zeros((12,17));state[:,0]=rng.normal(0,.001,(12,6));goal[:,:3]=rng.normal(0,.001,(12,3))
        for t in range(24):
            state[:,t+1]=state[:,t]
            state[:,t+1,3:]+=.15*(goal[:,:3]-state[:,t,3:])
        return dict(state=state,goal=goal,source_age=np.zeros((12,24)),future_prefix_remaining=np.full(12,7.)),dict(state_fields=[dict(name='qpos',size=3),dict(name='tcp_position_m',size=3)])

    def test_inference_is_unchanged_by_future_truth_and_ages(self):
        data,meta=self.fixture();ids=np.arange(10);m=CausalARX(meta).fit(data,ids);pred=m.predict(data,np.array([10,11]))
        data['state'][10:,9:]=100.;data['source_age'][10:,9:]=999.
        np.testing.assert_array_equal(pred,m.predict(data,np.array([10,11])))

    def test_DEV_poison_does_not_change_TRAIN_fit(self):
        data,meta=self.fixture();ids=np.arange(10);m=CausalARX(meta).fit(data,ids)
        data['state'][10:]=1e8;data['goal'][10:]=1e8
        other=CausalARX(meta).fit(data,ids)
        for key,value in m.arrays().items():np.testing.assert_array_equal(value,other.arrays()[key])

    def test_action_conditioned_plant_beats_hold_without_DEV_fit(self):
        data,meta=self.fixture();m=CausalARX(meta).fit(data,np.arange(10));ids=np.array([10,11]);truth=data['state'][ids,9:,3:]
        pred=m.predict(data,ids)[:,:,3:];hold=data['state'][ids,8,None,3:]
        self.assertLess(np.mean((pred-truth)**2),np.mean((hold-truth)**2)/4)
