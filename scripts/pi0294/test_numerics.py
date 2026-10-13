"""CPU control fixtures; never proof that a CUDA backward is deterministic."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .numerics import MODE,prepare_environment,configure_torch


class NumericsTests(unittest.TestCase):
    def test_environment_is_explicit_and_rejects_unexpected_values(self):
        with patch.dict(os.environ,{},clear=True):
            prepare_environment(None);self.assertNotIn('CUBLAS_WORKSPACE_CONFIG',os.environ)
            prepare_environment(MODE);self.assertEqual(os.environ['CUBLAS_WORKSPACE_CONFIG'],':4096:8')
            os.environ['CUBLAS_WORKSPACE_CONFIG']='unexpected'
            with self.assertRaises(PermissionError):prepare_environment(MODE)

    def test_all_numerical_flags_are_required_and_no_gate_relaxed(self):
        flags=dict(deterministic=False,flash=True,mem=True,cudnn=True,math=False)
        backend=SimpleNamespace()
        for api,key in [('flash','flash'),('mem_efficient','mem'),('cudnn','cudnn'),('math','math')]:
            setattr(backend,'enable_'+api+'_sdp',lambda value,k=key:flags.update({k:value}))
            setattr(backend,api+'_sdp_enabled',lambda k=key:flags[k])
        torch=SimpleNamespace(backends=SimpleNamespace(cuda=backend,cudnn=SimpleNamespace()),
            use_deterministic_algorithms=lambda value,warn_only:flags.update(deterministic=value),
            are_deterministic_algorithms_enabled=lambda:flags['deterministic'])
        with patch.dict(os.environ,{},clear=True):
            receipt=configure_torch(torch,MODE)
        self.assertEqual(flags,dict(deterministic=True,flash=False,mem=False,cudnn=False,math=True))
        self.assertFalse(receipt['tolerances_relaxed'])
        self.assertTrue(receipt['bf16_unchanged'])
        with self.assertRaises(PermissionError):configure_torch(torch,'RELAX_TOLERANCE')


if __name__=='__main__':unittest.main()
