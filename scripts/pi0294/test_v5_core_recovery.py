"""Actual captured recovery costs; no model, renderer, remote IO or permission."""
import base64,gzip,json,os
from copy import deepcopy
from pathlib import Path
import unittest
from scripts.pi0294.pv1 import mix_core_recovery as r,mix_core_accounting as c,mix_accounting as a,mix_core_premodel_repair as repair

class Recovery(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  path=os.environ.get('ARX5_CORE_RECOVERY_CAPTURE')
  if not path:raise unittest.SkipTest('explicit private captured readback required')
  cls.cap=json.loads(Path(path).read_bytes())
 def setUp(self):
  p=self.cap;self.gb=gzip.decompress(base64.b64decode(p['GLOBAL_before_gzip_base64']));self.lb=gzip.decompress(base64.b64decode(p['LOCAL_before_gzip_base64']))
  self.core=deepcopy(p['core']);self.closed=deepcopy(p['closed']);self.identity=p['identity'];self.now=self.core['observed_epoch']+1
 def propose(self,**kw):
  args=dict(core_readback=self.core,closed_Vanda_readback=self.closed,new_request_id=self.identity['request_id'],new_nonce=self.identity['nonce'],now_epoch=self.now);args.update(kw)
  return r.propose(self.gb,self.lb,**args)
 def test_full_future_preserves_costs_and_all_original_rows(self):
  p=self.propose();old=json.loads(self.gb);g=p['proposed_global'];l=p['proposed_LOCAL']
  self.assertFalse(p['full_future_proof']['cap_changes_required']);self.assertEqual(g['used'],old['used']);self.assertEqual(g['PBS_submissions'],old['PBS_submissions']);self.assertEqual(g['CPU_s'],old['CPU_s'])
  self.assertEqual(g['new_storage_peak_bytes'],old['new_storage_peak_bytes']);self.assertEqual(g['unknown_effects'],old['unknown_effects']);c.paired(g,l)
  self.assertEqual(l['optimizer_updates'],14786);self.assertEqual(len(l['PBS_submissions']),34)
 def test_global_only_cannot_admit_core(self):
  g=self.propose()['proposed_global']
  with self.assertRaises(PermissionError):c.paired(g,json.loads(self.lb))
 def test_no_native_permissions_from_proposal(self):
  p=self.propose();self.assertFalse(p['native_dispatch_admitted']);self.assertTrue(p['original_onceIO_required'])
 def test_no_unknown_effect_replay(self):
  g=json.loads(self.gb);g['unknown_effects']=[{'unknown':'keep'}];self.gb=a.raw(g)
  with self.assertRaises(PermissionError):self.propose()
 def test_no_valid_case_or_model_failure_is_replaced(self):
  x=next(v for v in self.closed['leases'].values());body=json.loads(base64.b64decode(x['base64']));body['status']='ACTIVE';b=a.raw(body);x.update(base64=base64.b64encode(b).decode(),sha256=a.sha(b))
  with self.assertRaises(PermissionError):self.propose()
 def test_old_identity_not_reissued(self):
  with self.assertRaises(PermissionError):self.propose(new_request_id=r.FAILED_VANDA)
 def test_changed_canonical_local_refused(self):
  self.lb=a.raw(dict(json.loads(self.lb),GPU_allocated_s=0))
  with self.assertRaises(PermissionError):self.propose()
 def test_stale_or_expired_native_read_refused(self):
  for now in (self.now+121,self.now-5):
   with self.subTest(now=now):
    with self.assertRaises(PermissionError):self.propose(now_epoch=now)
 def test_repeat_recovery_refused(self):
  self.gb=a.raw(self.propose()['proposed_global'])
  with self.assertRaises(PermissionError):self.propose()
 def test_exact_one_unconsumed_panel_credit(self):
  g=self.propose()['proposed_global'];m=r.validate(g);old=repair.validated(g)
  row=dict(request_id=m['replacement_request_id'],evaluation_candidate='C50',submission_nonce=m['replacement_submission_nonce'],child_run_uuid=m['replacement_run_uuid'],prepaid_panel_IO_credit_sha256=a.sha(a.raw(old)))
  self.assertEqual(repair.output_credit(g,row),old['paid_IO_credit']['Hopper_storage_bytes'])
  for k,v in [('request_id','other'),('evaluation_candidate','D25'),('submission_nonce','other')]:
   bad=dict(row,**{k:v})
   with self.subTest(k=k):
    with self.assertRaises(PermissionError):repair.output_credit(g,bad)
 def test_historical_budget_projection_is_same_conservative_floor(self):
  p=self.propose();g=p['proposed_global'];l=p['proposed_LOCAL'];self.assertEqual(r.budget_LOCAL(g,json.loads(self.lb)),l);self.assertEqual(r.budget_LOCAL(g,l),l)
  self.assertEqual(g[r.FIELD]['PBS_added'],0);self.assertEqual(g[a.PACKAGE]['remaining_resources']['H200_s'],57720)

if __name__=='__main__':unittest.main()
