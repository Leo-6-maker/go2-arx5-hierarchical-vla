import unittest
import torch
from go2arx5_vla.world_model.contact_training import predict
from go2arx5_vla.world_model.temporal_diagnostics import rigid_object
from go2arx5_vla.world_model.replay import SLICES
class FakeModel:
    def __init__(self):self.references=[]
    def burn(self,states,refs,context,clock):return states.new_zeros((len(states),2))
    def step(self,state,reference,context,clock,hidden):
        self.references.append(reference);new=state.clone();new[:,SLICES['tcp_position']]=state[:,SLICES['tcp_position']]+.5*(reference[:,0,:3]-state[:,SLICES['tcp_position']]);return new,state.new_zeros((len(state),4)),state.new_zeros(len(state)),hidden
class TemporalCausality(unittest.TestCase):
    def test_second_block_anchor_is_predicted_and_gradient_is_not_detached(self):
        s=torch.zeros(1,50,dtype=torch.float64);s[:,SLICES['tcp_rotation']]=torch.eye(3).flatten();s[:,SLICES['object_rotation']]=torch.eye(3).flatten();raw=torch.zeros(1,8,7,dtype=torch.float64);raw[:,:,0]=.001;raw[:,:,6]=.5;raw.requires_grad_();m=FakeModel()
        pred,_,_=predict(m,s[:,None].expand(-1,8,-1),s,s.new_zeros(1,8,40,10),s.new_zeros(1,5),s.new_zeros(1,8,2),s.new_zeros(1,2),raw)
        self.assertTrue(torch.equal(m.references[8][:,0,:3],pred[:,7,SLICES['tcp_position']]))
        self.assertTrue(torch.equal(m.references[8][:,0,:3],m.references[15][:,0,:3]));self.assertFalse(torch.equal(m.references[8][:,0,:3],m.references[0][:,7,:3]))
        grad=torch.autograd.grad(pred[:,15,SLICES['tcp_position']].sum(),raw)[0];self.assertGreater(float(grad[0,0,0]),.9)
    def test_rigid_baseline_uses_initial_tcp_frame_rotation(self):
        s=torch.zeros(1,50);s[:,SLICES['tcp_rotation']]=torch.eye(3).flatten();s[:,SLICES['object_position']]=torch.tensor([[.01,0,0]])
        p=s[:,None].expand(-1,16,-1).clone();p[:,:,SLICES['tcp_rotation']]=torch.tensor([[0.,-1,0],[1,0,0],[0,0,1]]).flatten();p[:,:,SLICES['tcp_position']]=torch.tensor([1.,2,3])
        x=rigid_object(s,p);self.assertTrue(torch.allclose(x,torch.tensor([1.,2.01,3.]).expand_as(x)))
if __name__=='__main__':unittest.main()
