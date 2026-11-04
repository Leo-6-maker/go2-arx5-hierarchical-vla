import unittest
import numpy as np
import torch
from go2arx5_vla.world_model.temporal_training import multitime_effect,fit_scales,HORIZONS,P
class TemporalLoss(unittest.TestCase):
    def test_multitime_targets_and_gradients_use_only_declared_horizons(self):
        state=torch.zeros(2,25,50);state[1,9,P]=.001;state[1,12,P]=.002;state[1,16,P]=.004;state[1,24,P]=.006
        b=dict(state=state,reference=torch.zeros(2,24,40,10),clock=torch.zeros(2,24,2),context=torch.zeros(2,5));p=torch.zeros(2,16,50,requires_grad=True);config={'horizons':list(HORIZONS),'sigma_m':[.004]*4,'weights':[.25]*4,'lambda_effect':1.}
        loss,terms=multitime_effect(p,b,[(0,1)],config);self.assertAlmostEqual(float(loss.detach()),sum(x*x for x in(.25,.5,1,1.5))/4);loss.backward();self.assertEqual(set(torch.nonzero(p.grad.abs().sum((0,2))).flatten().tolist()),{0,3,7,15})
        b['state'][1,8,0]=1
        with self.assertRaises(ValueError):multitime_effect(p,b,[(0,1)],config)
    def test_fit_scales_cannot_pair_across_parent_or_phase(self):
        state=torch.zeros(4,25,50);state[1,9:,P]=.004;state[2,9:,P]=100;state[3,9:,P]=200
        s=fit_scales({'state':state},np.array([0,0,1,0]),np.array(['transport','transport','transport','release']))
        self.assertEqual(s['FIT_pairs'],1);self.assertEqual(s['horizons'],[1,4,8,16]);self.assertTrue(all(abs(x-.004)<1e-8 for x in s['sigma_m']));self.assertAlmostEqual(sum(s['weights']),1)
if __name__=='__main__':unittest.main()
