import copy
import unittest
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'registered native PyTorch required')
class CPUFrozenCategoryTest(unittest.TestCase):
    def layer(self):
        from scripts.pi0294.generalization.minimal_adapter import TargetCategoryLinear
        class Category(torch.nn.Module):
            def __init__(self):
                super().__init__();self.W=torch.nn.Parameter(torch.randn(32,8,12,dtype=torch.float64));self.b=torch.nn.Parameter(torch.randn(32,12,dtype=torch.float64))
        return TargetCategoryLinear(Category(),10)

    def test_exact_target_output_gradients_adam_and_frozen_values(self):
        torch.manual_seed(73);a=self.layer();b=copy.deepcopy(a)
        frozen={n:x.clone()for n,x in b.named_buffers()};rng=torch.get_rng_state().clone()
        retained=b.retain_frozen_tables_on_cpu()
        self.assertEqual(retained,sum(x.numel()*x.element_size()for x in frozen.values()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        x=torch.randn(2,5,8,dtype=torch.float64);cat=torch.tensor([10,10]);oa=torch.optim.AdamW(a.parameters());ob=torch.optim.AdamW(b.parameters())
        for _ in range(3):
            oa.zero_grad();ob.zero_grad();xa=x.clone().requires_grad_();xb=x.clone().requires_grad_()
            ya=a(xa,cat);yb=b(xb,cat);self.assertTrue(torch.equal(ya,yb))
            ya.square().mean().backward();yb.square().mean().backward();self.assertTrue(torch.equal(xa.grad,xb.grad))
            for pa,pb in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(pa.grad,pb.grad))
            oa.step();ob.step()
            for pa,pb in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(pa,pb))
        for name,buffer in b.named_buffers():self.assertTrue(torch.equal(buffer,frozen[name]))
        with self.assertRaisesRegex(ValueError,'EXACT_TARGET_ONLY'):b(xa,torch.tensor([10,9]))
        # The unchanged default path still handles other categories.
        self.assertEqual(a(x,torch.tensor([10,9])).shape,(2,5,12))

    def test_generic_model_placement_cannot_move_retained_buffers_back(self):
        m=self.layer();m.retain_frozen_tables_on_cpu();kept=dict(m.named_buffers());seen=[]
        def placement(tensor):seen.append(tensor);return tensor.clone()
        m._apply(placement)
        self.assertEqual(len(seen),2)
        for name,tensor in m.named_buffers():self.assertIs(tensor,kept[name]);self.assertEqual(tensor.device.type,'cpu')

    def test_saved_activation_copy_bound_resets_each_microbatch(self):
        from scripts.pi0294.multitask.activation_offload import SavedActivationOffload
        off=SavedActivationOffload(torch.nn.Identity(),device_type='cpu',minimum_bytes=1,
                                   policy={'schema':'TEST','per_microbatch_activation_offload_bytes':128})
        x=torch.arange(25,dtype=torch.float32)
        for _ in range(2):
            with off.context():
                packed=off.pack(x);self.assertIsNotNone(packed[0]);self.assertTrue(torch.equal(off.unpack(packed),x))
                self.assertIsNone(off.pack(x)[0])
        self.assertEqual(off.copied_bytes,200)
        self.assertEqual(off.copied_tensors,off.restored_tensors)


if __name__=='__main__':unittest.main()
