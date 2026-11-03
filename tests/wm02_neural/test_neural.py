import copy
import random
import tempfile
import subprocess
import sys
from pathlib import Path
import unittest

import numpy as np
try:
    import torch
except ModuleNotFoundError as error:
    if error.name!='torch':raise
    raise unittest.SkipTest('Neural tests require the isolated E-G PyTorch environment')

from go2arx5_vla.world_model.neural import (WorldModel,Actor,STATE_DIM,SLICES,
    so3_exp,compile_reference,actor_observation,imagine)
from go2arx5_vla.world_model.neural_training import (fixture,read_fixture,selected,
    loss,gradients,update,checkpoint,load_checkpoint,tree_error,parameter_hash)


class NeuralContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(19)
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        fixture(self.root/'train.npz',count=8)
        self.data=read_fixture(self.root/'train.npz',torch.device('cpu'))
        self.model=WorldModel(hidden=(32,32),history_hidden=16)
        self.model.normalizer.fit(self.data['state'])
    def tearDown(self):self.temp.cleanup()

    def test_SO3_and_preview_have_finite_action_gradients_at_zero(self):
        vector=torch.zeros(2,3,requires_grad=True);rot=so3_exp(vector)
        self.assertTrue(torch.allclose(torch.linalg.det(rot),torch.ones(2)))
        rot[...,0,1].sum().backward();self.assertTrue(torch.isfinite(vector.grad).all())
        state=self.data['state'][:2,8];raw=torch.zeros(2,56,requires_grad=True)
        ref=compile_reference(state,raw)
        self.assertEqual(tuple(ref.shape),(2,40,10))
        self.assertTrue(torch.equal(ref[:,8:],ref[:,7:8].expand(-1,32,-1)))
        ref[:,:,:3].sum().backward();self.assertGreater(float(raw.grad.norm()),0)

    def test_microbatch_accumulation_preserves_sequence_loss_gradient(self):
        a=gradients(self.model,self.data,0,2,effective=8)
        g=[p.grad.clone()for p in self.model.parameters()]
        b=gradients(self.model,self.data,0,8,effective=8)
        self.assertAlmostEqual(a,b,places=5)
        self.assertLess(max(float((p.grad-x).abs().max())for p,x in zip(self.model.parameters(),g)),2e-5)

    def test_post_terminal_targets_do_not_cross_into_loss(self):
        a=selected(self.data,0,1);b={k:v.clone()for k,v in a.items()}
        b['state'][:,14:]=torch.randn_like(b['state'][:,14:])*100
        self.assertEqual(float(loss(self.model,a).detach()),float(loss(self.model,b).detach()))
        b['state'][:,9,0]+=1
        self.assertNotEqual(float(loss(self.model,a).detach()),float(loss(self.model,b).detach()))

    def test_neural_help_does_not_import_torch_or_native_modules(self):
        code="""import builtins
original=builtins.__import__
def guard(name,*a,**kw):
    if name.split('.')[0] in ('torch','mujoco','rclpy'): raise RuntimeError('Unwanted help import:'+name)
    return original(name,*a,**kw)
builtins.__import__=guard
from scripts.wm02.neural_smoke import main
main(['--help'])
"""
        q=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(q.returncode,0,q.stderr)

    def test_full_checkpoint_continuation_and_rng(self):
        opt=torch.optim.AdamW(self.model.parameters(),lr=2e-4,weight_decay=1e-5)
        update(self.model,opt,self.data,0,2,effective=8)
        p=self.root/'model.pt';identity={'data':'synthetic','source':'test'};config={'hidden':[32,32],'history_hidden':16}
        checkpoint(p,self.model,opt,1,8,identity,config)
        expected=(random.random(),float(np.random.rand()),torch.rand(3))
        restored,ro,c=load_checkpoint(p,torch.device('cpu'),identity)
        actual=(random.random(),float(np.random.rand()),torch.rand(3))
        self.assertEqual(expected[:2],actual[:2]);self.assertTrue(torch.equal(expected[2],actual[2]))
        self.assertEqual(c['replay_cursor'],8)
        update(self.model,opt,self.data,8,2,effective=8);update(restored,ro,self.data,8,2,effective=8)
        self.assertEqual(tree_error(self.model.state_dict(),restored.state_dict()),0)
        self.assertEqual(tree_error(opt.state_dict(),ro.state_dict()),0)
        with self.assertRaises(ValueError):load_checkpoint(p,torch.device('cpu'),{'data':'other'})

    def test_H16_two_decisions_freezes_model_and_actor_has_no_joint_input(self):
        b=selected(self.data,0,2);s=b['state'][:,8];ctx=b['context'];last=s.new_zeros((2,56))
        a=actor_observation(s,ctx,last,.4);other=s.clone();other[:,SLICES['joint_position']]+=20
        self.assertTrue(torch.equal(a,actor_observation(other,ctx,last,.4)))
        self.model.requires_grad_(False);before=parameter_hash(self.model)
        hidden=self.model.burn(b['state'][:,:8],b['reference'][:,:8],ctx)
        actor=Actor();ret,hist,actions,decisions,end=imagine(self.model,actor,s,ctx,hidden,16)
        self.assertEqual(decisions,[0,8]);self.assertEqual(hist.shape[-1],98)
        action_grads=torch.autograd.grad(-ret.mean(),actions,retain_graph=True)
        self.assertTrue(all(float(g.norm())>0 for g in action_grads))
        (-ret.mean()).backward()
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertEqual(before,parameter_hash(self.model))
        self.assertGreater(sum(float(p.grad.norm())for p in actor.parameters()if p.grad is not None),0)
        for name in ['base_rotation','tcp_rotation','object_rotation']:
            r=end[:,SLICES[name]].reshape(-1,3,3)
            self.assertTrue(torch.allclose(r.transpose(-1,-2)@r,torch.eye(3),atol=1e-5))


if __name__=='__main__':unittest.main()
