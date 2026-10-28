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

class Continuation(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  p=os.environ.get('ARX5_CORE05_CONTINUATION_CAPTURE')
  if not p:raise unittest.SkipTest('explicit actual closed predecessor capture required')
  cls.data=json.loads(Path(p).read_bytes())
 def setUp(self):
  import gzip
  p=self.data;self.capture=deepcopy(p['capture']);self.gb=gzip.decompress(base64.b64decode(self.capture['GLOBAL_gzip_base64']));self.lb=gzip.decompress(base64.b64decode(self.capture['LOCAL_gzip_base64']));self.ident=p['identity'];self.fixed=p['fixed_owner_helper_sha256'];self.now=self.capture['observed_epoch']+1
 def propose(self):return n.continue_proposal(self.gb,self.lb,closed_readback=self.capture,new_request_id=self.ident['request_id'],new_nonce=self.ident['nonce'],fixed_owner_helper_sha256=self.fixed,now_epoch=self.now)
 def test_all_costs_retained_both_panels_full_future_still_fit(self):
  v=self.propose();g=v['proposed_global'];l=v['proposed_LOCAL'];old=json.loads(self.gb);self.assertFalse(v['full_future_proof']['cap_changes_required']);self.assertEqual(g['used'],old['used']);self.assertEqual(g['PBS_submissions'],old['PBS_submissions']);self.assertEqual(g['new_storage_peak_bytes'],old['new_storage_peak_bytes']);self.assertEqual(g[a.PACKAGE]['completed_stages'],old[a.PACKAGE]['completed_stages']);self.assertEqual(g[a.PACKAGE]['remaining_resources']['H200_s'],57720);self.assertEqual(g[a.PACKAGE]['remaining_resources']['engine_starts'],4244);c.paired(g,l)
 def test_any_unclosed_predecessor_refused(self):
  self.capture['leases']['GPU']['status']='ACTIVE'
  with self.assertRaises(PermissionError):self.propose()
 def test_wrong_fixed_owner_code_refused(self):
  self.fixed='0'*64
  with self.assertRaises(PermissionError):self.propose()
 def test_old_request_not_reissued(self):
  self.ident=dict(self.ident,request_id=n.FAILED)
  with self.assertRaises(PermissionError):self.propose()
 def test_wrong_host_or_local_bytes_refused(self):
  self.capture['LOCAL_sha256']='0'*64
  with self.assertRaises(PermissionError):self.propose()
 def test_stale_or_duplicate_adoption_refused(self):
  self.now+=121
  with self.assertRaises(PermissionError):self.propose()
 def test_another_stage_lease_cannot_prove_closure(self):
  self.capture['leases']['physical']['run_uuid']='different-run'
  with self.assertRaises(PermissionError):self.propose()

 def test_historical_budget_projection_keeps_actual_core_failure_cpu(self):
  from scripts.pi0294.pv1.mix_vanda_first import budget_LOCAL
  path=Path(os.environ['ARX5_CORE05_CONTINUATION_CAPTURE']).parent/'FIRST512_HISTORICAL_LOCAL_BYTES.private.json'
  historical=json.loads(path.read_bytes());p=self.propose();g=p['proposed_global'];actual=p['proposed_LOCAL'];view=budget_LOCAL(g,historical)
  self.assertEqual(view['CPU_s']['CoreLab'],actual['CPU_s']['CoreLab'])
