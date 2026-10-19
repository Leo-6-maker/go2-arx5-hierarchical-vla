"""CPU array backend verifies decisions/reduction order and scalar-transfer count."""
from types import SimpleNamespace
import unittest

import numpy as np

from scripts.pi0294.pv1.gradient_audit import aggregated,equivalent,legacy


class Tensor:
    def __init__(self,value,backend,device='fixture-device'):
        self.value=np.asarray(value);self.backend=backend;self.device=device
    def all(self):return Tensor(self.value.all(),self.backend,self.device)
    def float(self):return Tensor(self.value.astype(np.float32),self.backend,self.device)
    def square(self):return Tensor(np.square(self.value),self.backend,self.device)
    def sum(self):return Tensor(self.value.sum(),self.backend,self.device)
    def to(self,*,dtype):return Tensor(self.value.astype(dtype),self.backend,self.device)
    def cpu(self):self.backend.host_reads+=1;return self
    def tolist(self):return self.value.tolist()
    def __bool__(self):self.backend.host_reads+=1;return bool(self.value)
    def __float__(self):self.backend.host_reads+=1;return float(self.value)


class Backend:
    float64=np.float64
    def __init__(self):self.host_reads=0
    def isfinite(self,value):return Tensor(np.isfinite(value.value),self,value.device)
    def stack(self,values):return Tensor(np.stack([v.value for v in values]),self,values[0].device)
    def cat(self,values):return Tensor(np.concatenate([v.value for v in values]),self,values[0].device)


def parameters(torch):
    return [(name,SimpleNamespace(grad=Tensor(array,torch),requires_grad=True)) for name,array in [
        ('action_head.projector.a',np.array([1.e10,2],dtype=np.float32)),
        ('action_head.model.a',np.array([[.5,-3],[7,2]],dtype=np.float16)),
        ('adaptation.a',np.array([2.],dtype=np.float64)),
        ('action_head.projector.b',np.array([.001,-.002],dtype=np.float32)),
        ('action_head.model.b',np.arange(20,dtype=np.float32)[::2])]]


class GradientAudit(unittest.TestCase):
    def test_same_checks_exact_norms_and_python_order_one_host_read(self):
        torch=Backend();params=parameters(torch);before=[p.grad.value.copy() for _,p in params]
        reference=legacy(params,torch);old_reads=torch.host_reads;torch.host_reads=0
        candidate=aggregated(params,torch)
        self.assertTrue(equivalent(reference,candidate));self.assertEqual(old_reads,9)
        self.assertEqual(candidate['host_reads'],1);self.assertEqual(torch.host_reads,1)
        self.assertEqual(candidate['gradient_tensors_checked'],5)
        for (_,p),old in zip(params,before):self.assertTrue(np.array_equal(p.grad.value,old))

    def test_all_nonfinite_and_frozen_gradients_rejected_not_sampled(self):
        for value in (np.nan,np.inf,-np.inf):
            for index in range(5):
                for method in (legacy,aggregated):
                    with self.subTest(method=method.__name__,index=index,value=value):
                        torch=Backend();params=parameters(torch);params[index][1].grad.value.flat[0]=value
                        with self.assertRaisesRegex(ValueError,'NONFINITE_OR_FROZEN'):method(params,torch)
        for method in (legacy,aggregated):
            torch=Backend();params=parameters(torch);params[-1][1].requires_grad=False
            with self.assertRaisesRegex(ValueError,'NONFINITE_OR_FROZEN'):method(params,torch)

    def test_none_zero_and_non_action_gradients_keep_original_behavior(self):
        torch=Backend();params=parameters(torch)
        for name,p in params:p.grad=None
        for method in (legacy,aggregated):
            v=method(params,torch);self.assertEqual(v['norm_squared'],dict(projector=0.,diffusion=0.))
            self.assertEqual(v['host_reads'],0)
        params=[('adaptation.only',SimpleNamespace(grad=Tensor([1.],torch),requires_grad=True))]
        self.assertTrue(equivalent(legacy(params,torch),aggregated(params,torch)))

    def test_no_cross_device_fallback_or_changed_reduction_tree(self):
        torch=Backend();params=parameters(torch);params[-1][1].grad.device='OTHER'
        with self.assertRaisesRegex(ValueError,'SINGLE_DEVICE'):aggregated(params,torch)
        a=legacy(parameters(torch),torch);b=dict(a,norm_squared=dict(a['norm_squared'],projector=0.))
        self.assertFalse(equivalent(a,b))
