import numpy as np
from go2arx5_vla.sla_adapter.mobile_endpoint_v52 import clean_estimate,endpoint_loss
from scripts.wm02.mobile_d0_v52 import boundary


def test_clean_identity_and_physical_units():
 rng=np.random.default_rng(52);actions=rng.normal(size=(4,40,32));noise=rng.normal(size=actions.shape);tau=np.array([0.,.25,.5,.9]);x=tau[:,None,None]*noise+(1-tau[:,None,None])*actions
 np.testing.assert_allclose(clean_estimate(x,tau,noise-actions),actions,atol=1e-15)
 low=np.zeros(7);high=np.full(7,2.-1e-6);a=np.zeros((4,40,32));a[:,1,0]=.001
 # physical endpoint acceleration .4 -> hinge (.4-.2)/.25=.8;
 # one endpoint and three of four eligible -> .64/2*3/4.
 assert np.isclose(endpoint_loss(a,tau,low,high,xp=np),.24)
 assert endpoint_loss(np.zeros_like(a),tau,low,high,xp=np)==0


def test_full_batch_mask_and_micro4_reduction_are_equivalent():
 rng=np.random.default_rng(53);a=rng.normal(size=(32,40,32))*.001;tau=rng.random(32);lo=np.arange(7)*.1;hi=lo+2
 whole=endpoint_loss(a,tau,lo,hi,xp=np);micro=np.mean([endpoint_loss(a[i:i+4],tau[i:i+4],lo,hi,xp=np)for i in range(0,32,4)])
 assert np.isclose(whole,micro,rtol=1e-14)


def test_boundary_index_uses_original_zero_endpoint_velocity():
 a=np.zeros((40,10));a[:,0]=np.arange(40)*.0007;b=boundary(a)
 assert b['peak_index']in(0,39) and np.isclose(b['start'],.28)and np.isclose(b['end'],.28)
 assert b['interior']<1e-12
