import unittest
from unittest.mock import patch
import torch
from go2arx5_vla.world_model.response_eval_audit import canonical_forecast
class CanonicalForecast(unittest.TestCase):
    def test_candidate_grouping_and_order_use_identical_singleton_arithmetic(self):
        b={'state':torch.arange(5.).reshape(5,1)}
        def fake(model,b):
            x=b['state'];return x,x+len(x),x*2
        with patch('go2arx5_vla.world_model.response_eval_audit.forecast',side_effect=fake):
            x=canonical_forecast(None,b);order=torch.tensor([4,2,0,3,1]);y=canonical_forecast(None,{'state':b['state'][order]});self.assertTrue(all(torch.equal(a,z[torch.argsort(order)])for a,z in zip(x,y)));self.assertTrue(torch.equal(x[1],b['state']+1))
if __name__=='__main__':unittest.main()
