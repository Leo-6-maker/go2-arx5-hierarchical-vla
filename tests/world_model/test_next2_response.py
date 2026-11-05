import unittest
import numpy as np
from go2arx5_vla.world_model.next2_response import future_choice_mask,action,HoldResponse
class CausalMask(unittest.TestCase):
    def test_only_future_chosen_position_is_removed(self):
        data=dict(goal=np.ones((2,17)),history_goal=np.full((2,8,17),2.),state=np.arange(50).reshape(2,25));masked=future_choice_mask(data)
        np.testing.assert_array_equal(masked['goal'][:,:3],2.);np.testing.assert_array_equal(masked['goal'][:,3:],data['goal'][:,3:]);np.testing.assert_array_equal(masked['history_goal'],data['history_goal']);np.testing.assert_array_equal(data['goal'],1.)
    def test_one_mm_local_model_rejects_big_or_unidentified_action(self):
        data=dict(goal=np.zeros((1,17)),history_goal=np.zeros((1,8,17)))
        for delta in [[.001,0,0],[-.001,0,0],[0,.001,0],[0,0,0]]:data['goal'][0,:3]=delta;action(data,[0])
        for delta in [[.1,0,0],[0,0,.001],[0,-.001,0]]:
            data['goal'][0,:3]=delta
            with self.assertRaises(ValueError):action(data,[0])

class ExplicitResponse(unittest.TestCase):
    def test_zero_anchor_reload_and_future_truth_poison(self):
        n=20;state=np.zeros((n,25,10));state[:,:,3]=1.;goal=np.zeros((n,17));history=np.zeros((n,8,17))
        offsets=np.array([[0,0],[.001,0],[-.001,0],[0,.001],[0,0]])
        goal[:,:2]=np.tile(offsets,(4,1));goal[:,0]+=.4;history[:,:,0]=.4
        for t in range(25):
            state[:,t,7:]=[.4+t*.00001,0,.7]
            if t>8:state[:,t,7:9]+=np.tile(offsets,(4,1))*min((t-8)/8,1)*.5
        data=dict(state=state,goal=goal,history_goal=history);meta=dict(state_fields=[dict(name='qpos',size=7),dict(name='tcp_position_m',size=3)])
        m=HoldResponse(meta,baseline='constant',ridge=.01).fit(data,np.arange(15));ids=np.arange(15,20);a=m.predict(data,ids)
        frozen=HoldResponse.from_arrays(meta,m.arrays(),baseline='constant',ridge=.01)
        np.testing.assert_array_equal(a,frozen.predict(data,ids))
        data['state'][ids,9:]=1e8
        np.testing.assert_array_equal(a,m.predict(data,ids))
        same={k:v[15:16].repeat(2,axis=0).copy()for k,v in data.items()};same['goal'][1,0]+=.001
        p=m.predict(same,[0,1]);self.assertGreater(p[1,-1,0]-p[0,-1,0],.0004)
        masked=future_choice_mask(same);q=m.predict(masked,[0,1]);np.testing.assert_array_equal(q[0],q[1])
