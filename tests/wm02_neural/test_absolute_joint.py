import unittest
from types import SimpleNamespace
import torch
from go2arx5_vla.world_model.absolute_joint_training import joint_terms
from go2arx5_vla.world_model.response_training import P
class AbsoluteJoint(unittest.TestCase):
    def test_common_candidate_bias_is_penalized_and_event_logits_have_gradient(self):
        p=torch.zeros(2,16,50,requires_grad=True);p.data[:,:,P]=.001;logits=torch.zeros(2,16,4,requires_grad=True);b=dict(state=torch.zeros(2,25,50),valid=torch.ones(2,24),events=torch.ones(2,24,4));m=SimpleNamespace(normalizer=SimpleNamespace(std=torch.ones(50)));c=dict(horizons=[1,4,8,16],sigma_m=[.001]*4,weights=[.25]*4)
        pos,full,event=joint_terms(m,p,logits,b,c);self.assertAlmostEqual(float(pos.detach()),1.);(pos+.1*full+event).backward();self.assertGreater(float(p.grad.sum()),0);self.assertLess(float(logits.grad.sum()),0)
        # Pair differences would cancel this common offset, absolute labels do not.
        self.assertTrue(torch.equal(p[0]-p[1],torch.zeros_like(p[0])))
        b['valid'][0,23]=0
        with self.assertRaises(ValueError):joint_terms(m,p,logits,b,c)
if __name__=='__main__':unittest.main()
