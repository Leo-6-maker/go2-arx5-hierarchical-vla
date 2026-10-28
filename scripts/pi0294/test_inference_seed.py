"""CPU-only seed compatibility; never imports a model or enters CUDA."""
import unittest
import numpy as np
from .resident import seed_inference


class SeedCompatibilityTests(unittest.TestCase):
    def test_full_fixed_seed_retained_for_torch_numpy32_not_resampling(self):
        calls=[];large=2566782898028649176
        def framework(seed):
            np.random.seed(seed)
            calls.append(('framework',seed))
        value=seed_inference(large,framework,lambda x:calls.append(('torch',x)))
        self.assertEqual(calls,[('framework',large % 2**32),('torch',large)])
        self.assertEqual(value['torch_seed'],large)
        self.assertEqual(value['mode'],'FIXED_TORCH63_NUMPY32_COMPAT_V1')

    def test_small_seeds_unchanged(self):
        calls=[]
        seed_inference(29400,lambda x:calls.append(x),lambda x:calls.append(x))
        self.assertEqual(calls,[29400,29400])

    def test_bad_seed_rejected_before_mutating_any_rng(self):
        for seed in (-1,2**63,True,1.5,'29400'):
            calls=[]
            with self.assertRaises(ValueError):seed_inference(seed,lambda x:calls.append(x),lambda x:calls.append(x))
            self.assertFalse(calls)

    def test_actual_cpu_torch_noise_matches_direct_full_seed(self):
        import torch
        large=2566782898028649176
        torch.manual_seed(large);expected=torch.randn(1,40,132)
        def framework(seed):
            np.random.seed(seed)
            torch.manual_seed(seed)
        seed_inference(large,framework,torch.manual_seed)
        self.assertTrue(torch.equal(expected,torch.randn(1,40,132)))


if __name__=='__main__':unittest.main()
