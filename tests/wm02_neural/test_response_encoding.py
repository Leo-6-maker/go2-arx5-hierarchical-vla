import unittest
import torch
from go2arx5_vla.world_model.neural import WorldModel,relative_reference,so3_exp
from go2arx5_vla.world_model.replay import GROUPS,SLICES

class ResponseEncoding(unittest.TestCase):
    def setUp(self):
        self.s=torch.zeros(1,50,dtype=torch.float64);self.s[:,SLICES['tcp_rotation']]=torch.eye(3).flatten();self.s[:,SLICES['tcp_position']]=torch.tensor([.4,.1,.6]);self.s[:,SLICES['gripper_fraction']]=.5
        self.r=torch.zeros(1,40,10,dtype=torch.float64);self.r[...,:3]=self.s[:,SLICES['tcp_position']][:,None];self.r[...,3:9]=torch.eye(3)[:2].flatten();self.r[...,9]=.5
    def test_internal_encoding_units_pose_equivariance_and_physical_immutability(self):
        r=self.r.clone();r[...,0]+=.004;r[...,9]+=.04/.044
        raw=r.clone();a=relative_reference(self.s,r,SLICES);self.assertTrue(torch.allclose(a[...,0],torch.full((1,40),.4,dtype=torch.float64)))
        self.assertTrue(torch.allclose(a[...,9],torch.ones(1,40,dtype=torch.float64)));self.assertTrue(torch.equal(r,raw))
        R=so3_exp(torch.tensor([[.3,-.2,.4]],dtype=torch.float64));s=self.s.clone();s[:,SLICES['tcp_position']]=(R@self.s[:,SLICES['tcp_position']][...,None]).squeeze(-1)+.2;s[:,SLICES['tcp_rotation']]=R.flatten(-2)
        rr=r.clone();rr[...,:3]=(R[:,None]@r[...,:3,None]).squeeze(-1)+.2;rr[...,3:9]=R[:,:2].flatten(-2)[:,None]
        self.assertTrue(torch.allclose(a,relative_reference(s,rr,SLICES),atol=1e-12))
    def test_graph_has_action_gradient_and_old_default_parameters_stay_identical(self):
        raw=self.r.clone().requires_grad_();relative_reference(self.s,raw,SLICES).sum().backward();self.assertEqual(float(raw.grad[0,0,0]),100.)
        torch.manual_seed(19);a=WorldModel(hidden=(16,),history_hidden=8,groups=GROUPS,context_dim=5)
        torch.manual_seed(19);b=WorldModel(hidden=(16,),history_hidden=8,groups=GROUPS,context_dim=5,reference_encoding='relative_tcp_v1')
        self.assertTrue(all(torch.equal(x,y)for x,y in zip(a.state_dict().values(),b.state_dict().values())))

if __name__=='__main__':unittest.main()
