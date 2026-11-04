import unittest
import torch
from go2arx5_vla.world_model.neural import WorldModel
from go2arx5_vla.world_model.replay import GROUPS
from go2arx5_vla.world_model.contact_training import supervised_loss
from go2arx5_vla.world_model.loss_diagnostics import components

class LossComponents(unittest.TestCase):
    def test_components_and_gradient_match_original_masked_objective(self):
        torch.manual_seed(77);torch.set_num_threads(1)
        model=WorldModel(hidden=(16,),history_hidden=8,groups=GROUPS,context_dim=5)
        state=torch.randn(2,17,50);b=dict(state=state,reference=torch.randn(2,16,40,10),context=torch.randn(2,5),clock=torch.zeros(2,16,2),
            events=torch.randint(0,2,(2,16,4)).float(),valid=torch.ones(2,16),done=torch.zeros(2,16),timeout=torch.zeros(2,16))
        b['valid'][0,12:]=0;b['done'][0,11]=1
        for blind in (False,True):
            total=components(model,b,blind).sum();original=supervised_loss(model,b,blind)
            self.assertTrue(torch.allclose(total,original,atol=1e-6,rtol=1e-6))
            g=torch.autograd.grad(total,list(model.parameters()));h=torch.autograd.grad(original,list(model.parameters()))
            self.assertTrue(all(torch.allclose(a,c,atol=2e-6,rtol=2e-6)for a,c in zip(g,h)))

if __name__=='__main__':unittest.main()
