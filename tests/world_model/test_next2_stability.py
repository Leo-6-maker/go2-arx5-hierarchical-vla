import unittest
import numpy as np
from go2arx5_vla.world_model.target_arx import CausalARX
from go2arx5_vla.world_model.next2_stability import companion
class ExactLinearization(unittest.TestCase):
    def test_operator_matches_independent_feature_finite_difference(self):
        rng=np.random.default_rng(21);m=CausalARX(dict(state_fields=[dict(name='qpos',size=7),dict(name='tcp_position_m',size=3)]));n=m.dimension;f=2*n+21;m.mean=rng.normal(size=n);m.scale=rng.uniform(.1,2,n);m.feature_mean=rng.normal(size=f);m.feature_scale=rng.uniform(.1,2,f);m.weight=rng.normal(size=(f+1,n))*.01;current=rng.normal(size=(1,n));previous=rng.normal(size=(1,n));goal=rng.normal(size=(1,17));delta=rng.normal(size=2*n)*1e-6
        def step(c,p):return c+m.design(m.feature(c,p,goal,np.zeros(1),np.zeros(1),np.ones(1)*7,0))@m.weight*m.scale
        actual=step(current+delta[:n],previous+delta[n:])-step(current,previous);np.testing.assert_allclose(actual[0],(companion(m)@delta)[:n],atol=2e-15,rtol=1e-7)
if __name__=='__main__':unittest.main()
