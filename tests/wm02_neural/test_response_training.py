import unittest
import numpy as np
import torch
from go2arx5_vla.world_model.response_training import first_block_pairs,pair_loss,P
from go2arx5_vla.world_model.contact_training import supervised_loss
from go2arx5_vla.world_model.neural import WorldModel
from go2arx5_vla.world_model.replay import GROUPS

class ResponseTraining(unittest.TestCase):
    def test_pair_supervision_never_mixes_parents_or_second_block_starts(self):
        keys=np.array([(0,'transport',0),(0,'transport',0),(1,'transport',0),(0,'transport',1),(0,'preclose',0)],dtype=object)
        self.assertEqual(first_block_pairs(keys),[(0,1)])
        pred=torch.zeros(5,8,50,requires_grad=True);state=torch.zeros(5,17,50);state[1,16,P]=.004
        b=dict(state=state,reference=torch.zeros(5,16,40,10));loss=pair_loss(pred,b,[(0,1)]);loss.backward();self.assertGreater(float(pred.grad.abs().sum()),0)
        b['state'][1,8,0]=1
        with self.assertRaises(ValueError):pair_loss(pred,b,[(0,1)])
    def test_reusing_supervised_rollout_preserves_original_loss_and_gradients(self):
        torch.set_num_threads(1);torch.manual_seed(3);m=WorldModel(hidden=(16,),history_hidden=8,groups=GROUPS,context_dim=5)
        b=dict(state=torch.zeros(2,17,50),reference=torch.zeros(2,16,40,10),context=torch.zeros(2,5),clock=torch.zeros(2,16,2),valid=torch.ones(2,16),events=torch.zeros(2,16,4),done=torch.zeros(2,16),timeout=torch.zeros(2,16))
        a=supervised_loss(m,b,False);v,p=supervised_loss(m,b,False,return_prediction=True)
        self.assertTrue(torch.equal(a,v));self.assertEqual(p.shape,(2,8,50))
        ga=torch.autograd.grad(a,list(m.parameters()));gv=torch.autograd.grad(v,list(m.parameters()));self.assertTrue(all(torch.equal(x,y)for x,y in zip(ga,gv)))

if __name__=='__main__':unittest.main()
