import unittest
import torch
from go2arx5_vla.world_model.neural import WorldModel,so3_exp
from go2arx5_vla.world_model.replay import GROUPS,SLICES
from go2arx5_vla.world_model.structured_interaction import attachment_gate,conditional_rigid_residual

class StructuredInteraction(unittest.TestCase):
    def fixture(self):
        s=torch.zeros(3,50,dtype=torch.float64)
        for name in ('tcp_rotation','object_rotation'):s[:,SLICES[name]]=torch.eye(3).flatten()
        s[:,SLICES['tcp_position']]=torch.tensor([.4,.1,.6]);s[:,SLICES['object_position']]=torch.tensor([.404,.102,.6]);s[:,SLICES['gripper_fraction']]=.433
        r=torch.zeros(3,40,10,dtype=torch.float64);r[...,9]=.04545;r[1,3,9]=.9545
        c=torch.zeros(3,2,dtype=torch.float64);c[:,1]=.15
        x=torch.zeros(3,5,dtype=torch.float64);x[:,4]=.02
        s[2,SLICES['gripper_fraction']]=1.
        return s,r,c,x
    def test_consumed_command_gate_se3_carry_and_open_mode_unchanged(self):
        s,r,c,x=self.fixture();n=s.clone();n[:,SLICES['tcp_position']]+=.003
        out=conditional_rigid_residual(s,n,r,c,x,SLICES)
        self.assertEqual(attachment_gate(s,r,c,x,SLICES).tolist(),[True,False,False])
        self.assertTrue(torch.allclose(out[0,SLICES['object_position']],s[0,SLICES['object_position']]+.003))
        self.assertTrue(torch.equal(out[1:],n[1:]));self.assertTrue(torch.equal(out[:,:SLICES['object_position'].start],n[:,:SLICES['object_position'].start]))
        # A rigid world transform rotates carry and slip together.
        R=so3_exp(torch.tensor([[.2,-.4,.3]],dtype=torch.float64));ss=s.clone();nn=n.clone()
        for v in(ss,nn):
            for name in('tcp_position','object_position'):v[:,SLICES[name]]=(R@v[:,SLICES[name]][...,None]).squeeze(-1)+.2
            for name in('tcp_rotation','object_rotation'):v[:,SLICES[name]]=(R@v[:,SLICES[name]].reshape(-1,3,3)).flatten(-2)
        oo=conditional_rigid_residual(ss,nn,r,c,x,SLICES)
        self.assertTrue(torch.allclose(oo[:,SLICES['object_position']],(R@out[:,SLICES['object_position']][...,None]).squeeze(-1)+.2,atol=1e-14))
        c[0,1]=.4
        with self.assertRaises(ValueError):attachment_gate(s,r,c,x,SLICES)
    def test_parameter_identity_and_tcp_and_slip_gradients(self):
        torch.manual_seed(19);a=WorldModel(hidden=(16,),history_hidden=8,groups=GROUPS,context_dim=5,reference_encoding='relative_tcp_v1')
        torch.manual_seed(19);b=WorldModel(hidden=(16,),history_hidden=8,groups=GROUPS,context_dim=5,reference_encoding='relative_tcp_v1',interaction_encoding='conditional_rigid_residual_v1')
        self.assertEqual(set(a.state_dict()),set(b.state_dict()));self.assertTrue(all(torch.equal(a.state_dict()[k],v)for k,v in b.state_dict().items()))
        s,r,c,x=self.fixture();n=s.clone().requires_grad_();out=conditional_rigid_residual(s,n,r,c,x,SLICES);out[0,SLICES['object_position']].sum().backward()
        self.assertTrue(torch.equal(n.grad[0,SLICES['tcp_position']],torch.ones(3,dtype=n.dtype)));self.assertTrue(torch.equal(n.grad[0,SLICES['object_position']],torch.ones(3,dtype=n.dtype)))

if __name__=='__main__':unittest.main()
