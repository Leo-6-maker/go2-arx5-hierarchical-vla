"""Captured actual terminal evidence; future cold actors are not native PASS."""
import os,json,base64
from copy import deepcopy
from pathlib import Path
import unittest
from scripts.pi0294.pv1 import mix_core_prebind_terminal as n,mix_accounting as a,mix_core_accounting as c,mix_core_desktop as desktop
class Terminal(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  p=os.environ.get('ARX5_CORE04_TERMINAL_CAPTURE')
  if not p:raise unittest.SkipTest('explicit private captured terminal required')
  cls.data=json.loads(Path(p).read_bytes())
 def setUp(self):
  p=self.data;self.gb=base64.b64decode(p['global_base64']);self.lb=base64.b64decode(p['LOCAL_base64']);self.proof=deepcopy(p['proof']);self.now=self.proof['observed_epoch']+1
 def propose(self):return n.propose(self.gb,self.lb,proof=self.proof,now_epoch=self.now)
 def test_actual_cost_not_fabricated_bind_or_zero_gpu(self):
  p=self.propose();g=p['proposed_global'];old=json.loads(self.gb);r=g[c.FIELD][n.FAILED];self.assertIsNone(r['native_handle']);self.assertEqual(p['receipt']['GPU_s'],62);self.assertEqual(p['receipt']['neural_CPU_s'],1);self.assertEqual(g['used']['H200_s'],old['used']['H200_s']+62);self.assertEqual(r['query_attempts'],0);c.paired(g,p['proposed_LOCAL']);self.assertEqual(g['PBS_submissions'],old['PBS_submissions']);self.assertEqual(g[a.PACKAGE]['completed_stages'],old[a.PACKAGE]['completed_stages'])
 def test_stale_actual_observation_rejected(self):
  self.now+=121
  with self.assertRaises(PermissionError):self.propose()
 def test_live_unit_is_not_terminal(self):
  next(iter(self.proof['units'].values()))['MainPID']='123'
  with self.assertRaises(PermissionError):self.propose()
 def test_repeated_financial_settlement_rejected(self):
  p=self.propose();self.gb=a.raw(p['proposed_global']);self.lb=a.raw(p['proposed_LOCAL'])
  with self.assertRaises(PermissionError):self.propose()
 def test_real_model_release_cannot_claim_premodel_zero(self):
  self.proof['qualification_or_model_or_phase_release']=True
  with self.assertRaises(PermissionError):self.propose()
 def test_unmatched_saved_bytes_rejected(self):
  self.proof['files']['NATIVE_FINAL']['base64']=base64.b64encode(b'{}').decode()
  with self.assertRaises(PermissionError):self.propose()
 def test_original_helper_cold_fix_is_narrow_and_all_owner_calls_bound(self):
  body=base64.b64decode(self.proof['files']['HELPERS']['base64']);new=desktop.owner_kernel_helpers(body);self.assertIn(b'desktop_request=r',new);self.assertIn(b'own_pid=pid if list',new)
  with self.assertRaises(PermissionError):desktop.owner_kernel_helpers(body+b'\n')
if __name__=='__main__':unittest.main()
