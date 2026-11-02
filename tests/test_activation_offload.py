import copy
import unittest
try:
    import torch
except ImportError:
    torch=None
from scripts.pi0294.multitask.activation_offload import SavedActivationOffload


@unittest.skipIf(torch is None,'native PyTorch required; run in registered neural environment')
class ActivationOffloadTest(unittest.TestCase):
    def test_gradients_optimizer_rng_and_storage_views_preserved(self):
        torch.manual_seed(123)
        model=torch.nn.Sequential(torch.nn.Linear(8,16),torch.nn.SiLU(),torch.nn.Linear(16,3)).double()
        other=copy.deepcopy(model);x=torch.randn(5,8,dtype=torch.float64)
        a=torch.optim.AdamW(model.parameters());b=torch.optim.AdamW(other.parameters())
        off=SavedActivationOffload(other,device_type='cpu',minimum_bytes=1)
        before=torch.get_rng_state().clone()
        for _ in range(3):
            a.zero_grad();b.zero_grad()
            loss=model(x).square().mean();loss.backward()
            with off.context():out=other(x).square().mean();out.backward()
            self.assertTrue(torch.equal(loss,out))
            for pa,pb in zip(model.parameters(),other.parameters()):self.assertTrue(torch.equal(pa.grad,pb.grad))
            a.step();b.step()
            for pa,pb in zip(model.parameters(),other.parameters()):self.assertTrue(torch.equal(pa,pb))
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        view=next(other.parameters()).T
        self.assertIsNone(off.pack(view)[0])
        self.assertGreater(off.copied_tensors,0)
        # One explicit direct pack(view) is excluded from copies/restores.
        self.assertEqual(off.copied_tensors,off.restored_tensors)

    def test_roundtrip_dtypes_and_noncontiguous_shape(self):
        off=SavedActivationOffload(torch.nn.Identity(),device_type='cpu',minimum_bytes=1)
        for dtype in (torch.float64,torch.float32,torch.bfloat16,torch.int64):
            t=torch.arange(24).reshape(4,6).to(dtype).T
            device,cpu=off.pack(t)
            self.assertNotEqual(cpu.untyped_storage().data_ptr(),t.untyped_storage().data_ptr())
            restored=off.unpack((device,cpu))
            self.assertTrue(torch.equal(restored,t))
            self.assertEqual((restored.shape,restored.dtype,restored.device),(t.shape,t.dtype,t.device))


if __name__=='__main__':unittest.main()
