"""Generated engineering fixtures, never reported as native branch evidence."""
import unittest
import numpy as np
try:import torch
except ModuleNotFoundError as e:
    if e.name!='torch':raise
    raise unittest.SkipTest('Isolated Torch environment required')
from go2arx5_vla.world_model.neural import WorldModel
from go2arx5_vla.world_model.replay import GROUPS,SLICES,STATE_DIM
from go2arx5_vla.world_model.contact_training import predict,cv_prediction,supervised_loss,training_view,select


class ContactLearning(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(27)
        self.model=WorldModel(hidden=(32,32),history_hidden=16,groups=GROUPS,context_dim=5)
        s=torch.zeros(2,STATE_DIM)
        for name,n in GROUPS:
            if name.endswith('rotation'):s[:,SLICES[name]]=torch.eye(3).flatten()
        s[:,SLICES['tcp_position']]=torch.tensor([.4,0,.5]);s[:,SLICES['object_position']]=torch.tensor([.4,0,.46]);s[:,SLICES['gripper_fraction']]=.45
        self.s=s;self.history=s[:,None].expand(-1,8,-1).clone();self.context=torch.tensor([[.5,0,.46,.44,.04]]).expand(2,-1)
        self.reference=torch.cat((s[:,SLICES['tcp_position']],torch.eye(3)[:2].flatten().expand(2,-1),s[:,SLICES['gripper_fraction']]),-1)[:,None,None].expand(-1,8,40,-1).clone()
        self.clocks=torch.tensor([[[i*.05/32,(i%8)*.05]for i in range(8)]]).expand(2,-1,-1)
        self.start=torch.tensor([[.4/32,0.]]).expand(2,-1);self.raw=torch.zeros(2,8,7);self.raw[:,:,6]=.45;self.raw[1,:,0]=.0005
    def test_future_interventions_keep_history_and_blind_prediction_identical(self):
        args=(self.history,self.s,self.reference,self.context,self.clocks,self.start)
        a,e,h=predict(self.model,*args,self.raw)
        b,f,k=predict(self.model,*args,self.raw.flip(0));self.assertTrue(torch.equal(h,k))
        x,_,_=predict(self.model,*args,self.raw,blind=True);y,_,_=predict(self.model,*args,self.raw.flip(0),blind=True)
        self.assertTrue(torch.equal(x,y));self.assertGreater(float((a-b).abs().max().detach()),0)
        tail,_,ht=predict(self.model,*args,self.raw,tail=True);self.assertTrue(torch.equal(ht,h))
        self.assertEqual(tuple(a.shape),(2,16,50))
        for name in ('tcp_rotation','object_rotation'):
            r=a[...,SLICES[name]].reshape(2,16,3,3);self.assertTrue(torch.allclose(r.transpose(-1,-2)@r,torch.eye(3),atol=1e-6))
    def test_cv_uses_actual_linear_and_angular_velocity_in_world_units(self):
        s=self.s.clone();s[:,SLICES['object_velocity']]=torch.tensor([.1,-.2,0,0,0,.3])
        result=cv_prediction(s);np.testing.assert_allclose(result[0,-1,SLICES['object_position']].numpy(),[.48,-.16,.46],atol=1e-7)
        r=result[0,-1,SLICES['object_rotation']].reshape(3,3);self.assertAlmostEqual(float(torch.atan2(r[1,0],r[0,0])),.24,places=6)
    def test_two_causal_training_blocks_and_terminated_targets_masked(self):
        b=dict(state=self.s[:,None].expand(-1,25,-1).clone(),reference=self.reference[:,0:1].expand(-1,24,-1,-1).clone(),
            context=self.context,events=torch.zeros(2,24,4),done=torch.zeros(2,24),timeout=torch.zeros(2,24),
            valid=torch.ones(2,24),clock=torch.tensor([[[i*.05/32,(i%8)*.05]for i in range(24)]]).expand(2,-1,-1),raw=self.raw,initial_event=torch.zeros(2,4))
        view=training_view(b);self.assertEqual(tuple(view['state'].shape),(4,17,50))
        view['done'][:,10]=1;view['valid'][:,11:]=0
        other={k:v.clone()for k,v in view.items()};other['state'][:,12:]+=100
        self.assertEqual(float(supervised_loss(self.model,view,False).detach()),float(supervised_loss(self.model,other,False).detach()))
        supervised_loss(self.model,view,False).backward();self.assertTrue(all(torch.isfinite(p.grad).all()for p in self.model.parameters()if p.grad is not None))

if __name__=='__main__':unittest.main()
