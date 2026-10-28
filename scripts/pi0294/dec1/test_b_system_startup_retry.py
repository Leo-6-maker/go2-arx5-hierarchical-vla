"""PBS-only approval and mature full-runway integration; fixtures are CPU only."""
from copy import deepcopy
import json
import tempfile
from pathlib import Path
import unittest
from . import b_system_startup_retry as r,future_budget as fb
from .test_b_heartbeat_recovery import scope
from .test_b_recovery_forecast import inputs,forecast
from .test_a40_allocation import raw

GRANT=Path(__file__).resolve().parents[3]/'reports/PI0294/design/parallel-science-20261022/B2_SYSTEM_STARTUP_RETRY_PBS_AUTHORIZATION_20261023.json'

def full_retry():
 a,p=inputs();a['retry_bytes']=GRANT.read_bytes();now=r.APPROVED+100
 a['now_epoch']=now;a['snapshot']['observed_epoch']=p['observed_epoch']=now
 for v in (a['snapshot'],p,a['plan']):v.update(r.scope())
 a['snapshot']['used']['PBS_submissions']=22;a['snapshot']['dec1_used']['PBS_submissions']=11
 a['snapshot']['used']['optimizer_updates']=a['snapshot']['dec1_used']['optimizer_updates']=8744
 p['candidates']['B_COVERAGE']['optimizer_updates_committed']=2048
 a['plan']['training_jobs']=a['plan']['training_jobs'][1:]
 a['plan']['training_jobs'][0].update(r.scope());a['progress_bytes']=raw(p)
 return a,p

class StartupRetry(unittest.TestCase):
 def test_native_readonly_sealed_grant_and_paired_authority_match_before_state_import(self):
  for mode in ('ok','manifest','request','native','writable'):
   with self.subTest(mode=mode),tempfile.TemporaryDirectory() as td:
    parent=Path(td);root=parent/'children/retry';(root/'plans').mkdir(parents=True);(parent/'authority/DEC1').mkdir(parents=True)
    p=root/r.PLAN;p.write_bytes(GRANT.read_bytes());p.chmod(0o400)
    q=parent/'authority/DEC1'/Path(r.PLAN).name;q.write_bytes(GRANT.read_bytes());q.chmod(0o400)
    row=dict(scope(),**r.saved.scope(),**r.scope(),expected_native_start_step=2048,training_block=dict(arm='B_COVERAGE',stop_after_step=4096),sealed_inputs={r.PLAN:r.GRANT_SHA})
    manifest=dict(authority=r.scope(),execution=dict(sealed_inputs={r.PLAN:r.GRANT_SHA}))
    if mode=='manifest':manifest['execution']['sealed_inputs'][r.PLAN]='0'*64
    elif mode=='request':row['sealed_inputs'][r.PLAN]='0'*64
    elif mode=='native':q.chmod(0o600);q.write_bytes(b'wrong');q.chmod(0o400)
    elif mode=='writable':p.chmod(0o600)
    if mode=='ok':self.assertTrue(r.admitted(root,manifest,row))
    else:
     with self.assertRaises(PermissionError):r.admitted(root,manifest,row)
 def test_only_exact_published_PBS_two_change_authority(self):
  v=r.grant(GRANT.read_bytes());self.assertEqual(v['scope']['additional_unique_retry_submissions'],1)
  for k in ('scope','continuation','unchanged','human_answer'):
   wrong=deepcopy(v);wrong[k]={}
   with self.subTest(k=k),self.assertRaises(PermissionError):r.grant(raw(wrong))
  with self.assertRaises(PermissionError):r.grant(GRANT.read_bytes()+b' ')
 def test_only_new_original_B2048_to4096_request(self):
  row=dict(scope(),**r.saved.scope(),**r.scope(),expected_native_start_step=2048,training_block=dict(arm='B_COVERAGE',stop_after_step=4096),request_id='a-new-unique-request')
  self.assertTrue(r.request(row));self.assertFalse(r.enabled({}))
  for key,value in ((r.SCOPE_KEYS[0],'0'*40),('expected_native_start_step',1024),('training_lineage_key','A_CONTROL'),('request_id',r.REQUEST)):
   wrong=deepcopy(row);wrong[key]=value
   with self.subTest(key=key),self.assertRaises(PermissionError):r.request(wrong)
  wrong=deepcopy(row);wrong.pop(r.SCOPE_KEYS[1])
  with self.assertRaises(PermissionError):r.request(wrong)
 def test_full_remaining_original_three_matrix_and_final_protection_only_PBS_caps_change(self):
  a,p=full_retry();v=forecast(a);self.assertTrue(v['admissible_forecast'])
  self.assertEqual(v['future_required_additional']['PBS_submissions'],4);self.assertEqual(v['future_required_additional']['optimizer_updates'],2048)
  self.assertEqual(v['effective_DEC1_VAP1_caps']['PBS_submissions'],15);self.assertEqual(v['effective_global_caps']['PBS_submissions'],26)
  self.assertEqual(v['original_final_protection']['H200_s'],72000)
  for x in (a['snapshot'],p,a['plan']):
   for k in r.SCOPE_KEYS:x.pop(k)
  a.pop('retry_bytes');a['progress_bytes']=raw(p);old=forecast(a)
  for k in ('effective_DEC1_VAP1_caps','effective_global_caps'):
   self.assertEqual({n:z for n,z in old[k].items() if n!='PBS_submissions'},{n:z for n,z in v[k].items() if n!='PBS_submissions'})
  self.assertEqual(old['global_shortfalls'],dict(PBS_submissions=1));self.assertEqual(old['DEC1_shortfalls'],dict(PBS_submissions=1))
 def test_missing_grant_snapshot_plan_or_progress_reject(self):
  for mode in ('bytes','snapshot','plan','progress'):
   a,p=full_retry()
   if mode=='bytes':a.pop('retry_bytes')
   elif mode=='progress':p.pop(r.SCOPE_KEYS[1]);a['progress_bytes']=raw(p)
   else:a[mode].pop(r.SCOPE_KEYS[1])
   with self.subTest(mode=mode),self.assertRaises(PermissionError):forecast(a)
 def test_other_caps_and_original_deadlines_are_still_enforced(self):
  for mode in ('GPU','CPU','deadline'):
   a,p=full_retry()
   if mode=='GPU':a['snapshot']['dec1_used']['H200_s']+=100000
   elif mode=='CPU':a['snapshot']['dec1_used']['local_CPU_s']+=60000
   else:a['now_epoch']=r.b.CUTOFF
   with self.subTest(mode=mode):
    try:v=forecast(a)
    except PermissionError:continue
    self.assertFalse(v['admissible_forecast'])
if __name__=='__main__':unittest.main()
