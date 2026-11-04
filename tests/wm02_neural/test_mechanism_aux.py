import unittest
import torch
from go2arx5_vla.world_model.contact_training import predict
from go2arx5_vla.world_model.mechanism_diagnostics import observable_features
from go2arx5_vla.world_model.neural import WorldModel
from go2arx5_vla.world_model.replay import SLICES,GROUPS
class MechanismAux(unittest.TestCase):
    def test_auxiliary_terminal_and_logits_do_not_change_free_states_or_gradients(self):
        torch.set_num_threads(1);torch.manual_seed(2);m=WorldModel(hidden=(16,),history_hidden=8,groups=GROUPS,context_dim=5,reference_encoding='relative_tcp_v1');s=torch.zeros(2,50);s[:,SLICES['tcp_rotation']]=torch.eye(3).flatten();s[:,SLICES['object_rotation']]=torch.eye(3).flatten();raw=torch.zeros(2,8,7);raw[:,:,6]=.5;raw.requires_grad_();args=(m,s[:,None].expand(-1,8,-1),s,torch.zeros(2,8,40,10),torch.zeros(2,5),torch.zeros(2,8,2),torch.zeros(2,2),raw)
        old=predict(*args);aux=predict(*args,return_terminal=True,return_logits=True);self.assertTrue(all(torch.equal(a,b)for a,b in zip(old,aux[:3])));self.assertTrue(torch.allclose(aux[1],aux[4]['event'].sigmoid(),atol=1e-7,rtol=0));self.assertTrue(torch.allclose(aux[3],aux[4]['terminal'].sigmoid(),atol=1e-7,rtol=0));a=torch.autograd.grad(old[0].sum(),raw)[0];b=torch.autograd.grad(aux[0].sum(),raw)[0];self.assertTrue(torch.equal(a,b))
    def test_observability_features_use_current_geometry_and_gripper_only(self):
        s=torch.zeros(1,50);s[:,SLICES['tcp_rotation']]=torch.eye(3).flatten();s[:,SLICES['object_position']]=torch.tensor([[.01,0,0]]);x=observable_features(s);self.assertEqual(x.shape,(1,9));self.assertAlmostEqual(float(x[0,1]),1.)
if __name__=='__main__':unittest.main()
