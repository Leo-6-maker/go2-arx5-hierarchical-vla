"""Exact authorization plus tiny checkpoint fixtures, never native evidence."""
from contextlib import ExitStack,contextmanager
from copy import deepcopy
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from . import b_saved_state_reuse as saved,b_heartbeat_recovery as b,segment_intake as intake
from .test_segment_intake import fixture,put,cold_inputs
from .test_b_heartbeat_recovery import scope

GRANT=Path(__file__).parents[3]/'reports/PI0294/design/parallel-science-20261022/B_FAILED_SAVED2048_REUSE_AUTHORIZATION_20261023.json'

def bound():
 return dict(root='/registered/closed/B2048',arm='B_COVERAGE',job_id=saved.JOB,request_id=saved.REQUEST,request_sha256=saved.REQUEST_SHA,source_sha=saved.SOURCE,run_manifest_sha256='a'*64,terminal_sha256=saved.GPU_TERMINAL,lease_sha256=saved.GPU_LEASE,checkpoint1024_sha256=b.SEED_SHA,checkpoint2048_sha256=saved.CHECKPOINT,probe_result_sha256=b.PROBE_SHA,B_heartbeat_recovery_seed=intake.b_seed_identity(),**{saved.KEY:saved.identity()})

@contextmanager
def tiny_failed_saved():
 with fixture('B_COVERAGE') as (s,t,v,recipe,data),ExitStack() as stack:
  oldjob='88.stdct-mgmt-02';oldsource='e'*40
  cp=s/'training/B_COVERAGE/checkpoint-1024/CHECKPOINT_COMPLETE.json';c=intake.read(cp);c.update(origin_job_id=oldjob,origin_source_sha=oldsource);v['checkpoint1024_sha256']=intake.sha(put(cp,c))
  rp=s/'evidence/B_COVERAGE-1024/TRAIN_PROBES/RESULT.json';r=intake.read(rp);r['checkpoint_manifest_sha256']=v['checkpoint1024_sha256'];v['probe_result_sha256']=intake.sha(put(rp,r))
  m=intake.read(s/'RUN_MANIFEST.json');m['authority'].update(dict(zip(b.SCOPE_KEYS,(b.RELEASE,b.GRANT_SHA))));v['run_manifest_sha256']=intake.sha(put(s/'RUN_MANIFEST.json',m))
  c=intake.read(s/'TERMINAL_RECONCILIATION.json');c.pop('global_step');c.update(status='CLOSED_B_RECOVERY1_HEARTBEAT_FAILURE_SAVED2048_PRESERVED',native_terminal=dict(job_state='F',Exit_status=1),latest_complete_step=2048,committed_updates_without_new_complete_state=0,retired_or_deleted_files=0,settlement=dict(updates=8744,unknown_effects=[]),checkpoints={str(n):dict(complete_sha256=v['checkpoint'+str(n)+'_sha256'],members=15,actual_consumed_cursor=n*32) for n in (1024,2048)})
  c['effects'].update(start_native_step=1024,optimizer_updates_committed=1024,query_attempts=17,model_load_attempts=2,new_physics=0,new_episodes=0)
  v['terminal_sha256']=intake.sha(put(s/'TERMINAL_RECONCILIATION.json',c));lease=intake.read(s/'lease.json');lease['terminal_receipt_sha256']=v['terminal_sha256'];v['lease_sha256']=intake.sha(put(s/'lease.json',lease))
  out=intake.read(s/'OUTPUT_BINDING.private.json');out['run_manifest_sha256']=v['run_manifest_sha256'];put(s/'OUTPUT_BINDING.private.json',out)
  stack.enter_context(patch.multiple(b,FAILED_JOB=oldjob,FAILED_SOURCE=oldsource,SEED_SHA=v['checkpoint1024_sha256'],PROBE_SHA=v['probe_result_sha256'],RECIPE=intake.digest(recipe)))
  stack.enter_context(patch.multiple(saved,JOB=v['job_id'],REQUEST=v['request_id'],REQUEST_SHA=v['request_sha256'],SOURCE=v['source_sha'],CHECKPOINT=v['checkpoint2048_sha256'],GPU_TERMINAL=v['terminal_sha256'],GPU_LEASE=v['lease_sha256']))
  v['B_heartbeat_recovery_seed']=intake.b_seed_identity();v[saved.KEY]=saved.identity()
  yield s,t,v,recipe,data

class SavedStateReuse(unittest.TestCase):
 def test_finite_local_controller_preserves_all_old_remaining_ceilings(self):
  upper=dict(Hopper_CPU_s=5520,local_CPU_s=3600,transfer_bytes=805306368)
  v=saved.control_plan(local_charged_s=50063.974682509,future_local_s=7320,original_control_upper=upper)
  self.assertEqual(sum(v['LOCAL_phase_CPU_limits'].values()),180);self.assertEqual(v['LOCAL_phase_CPU_limits']['serve'],100)
  self.assertGreater(v['local_headroom_s'],36);self.assertFalse(v['native_admission'])
  for charged,future,control in ((50063.974682509,8520,upper),(50280,7320,upper),(50063,7320,dict(upper,local_CPU_s=2400))):
   with self.subTest(charged=charged,future=future),self.assertRaises(PermissionError):saved.control_plan(local_charged_s=charged,future_local_s=future,original_control_upper=control)
 def test_exact_published_human_grant_not_draft_or_changed_caps(self):
  raw=GRANT.read_bytes();v=saved.grant(raw);self.assertFalse(v['new_caps_or_budget_or_clock']);self.assertEqual(v['native_exit_status'],1)
  for k,n in (('human_answer','NO'),('native_failure_not_relabelled_success',False),('new_caps_or_budget_or_clock',True)):
   x=deepcopy(v);x[k]=n
   with self.subTest(k=k),self.assertRaises(PermissionError):saved.grant((json.dumps(x)+'\n').encode())
 def test_only_new_B2048_to4096_request_and_complete_scope(self):
  r=scope();r.update(saved.scope(),expected_native_start_step=2048,training_block=dict(arm='B_COVERAGE',stop_after_step=4096));self.assertTrue(saved.request(r))
  for kind in ('partial','other','start','stop','arm'):
   x=deepcopy(r)
   if kind=='partial':x.pop(saved.SCOPE_KEYS[1])
   elif kind=='other':x[saved.SCOPE_KEYS[0]]='0'*40
   elif kind=='start':x['expected_native_start_step']=1024
   elif kind=='stop':x['training_block']['stop_after_step']=2048
   else:x['training_block']['arm']='A_CONTROL'
   with self.subTest(kind=kind),self.assertRaises(PermissionError):saved.request(x)
 def test_exact_saved_binding_rejects_other_checkpoint_failure_job_or_seed(self):
  v=bound();intake.valid_binding(v,'B_COVERAGE')
  for key in ('checkpoint2048_sha256','checkpoint1024_sha256','job_id','source_sha','terminal_sha256','lease_sha256','probe_result_sha256'):
   x=deepcopy(v);x[key]='0'*64
   with self.subTest(key=key),self.assertRaises(PermissionError):intake.valid_binding(x,'B_COVERAGE')
  x=deepcopy(v);x.pop('B_heartbeat_recovery_seed')
  with self.assertRaises(PermissionError):intake.valid_binding(x,'B_COVERAGE')
 def test_failed_saved_source_uses_full_original_validator_with_grant_and_split_origins(self):
  with tiny_failed_saved() as (s,t,v,recipe,data):
   before={p:intake.sha(p) for p in s.rglob('*') if p.is_file()}
   with self.assertRaises(PermissionError):intake.audit_closed_source(v,recipe=recipe,dataset=data)
   proof=intake.audit_closed_source(v,recipe=recipe,dataset=data,reuse_grant_bytes=GRANT.read_bytes())
   self.assertEqual(proof['status'],'PASS_CLOSED_FAILED_SAVED2048_FULL_STATES_NOT_TRAINING_ADMISSION');self.assertEqual(proof['actual_consumed_cursor'],65536)
   self.assertEqual(before,{p:intake.sha(p) for p in s.rglob('*') if p.is_file()})
   cold_inputs(s,t,v,proof);put(t/saved.PLAN,saved.grant(GRANT.read_bytes()))
   # Preserve the exact grant bytes, rather than json fixture formatting.
   p=t/saved.PLAN;p.chmod(0o600);p.write_bytes(GRANT.read_bytes());p.chmod(0o400)
   shutil.copytree(s/'training',t/'training');shutil.copytree(s/'evidence',t/'evidence')
   self.assertEqual(intake.verify_copy(t,recipe=recipe,dataset=data)['global_step'],2048)
 def test_damaged_optimizer_rng_sampler_probe_or_failure_identity_reject(self):
  for name in ('optimizer.pt','rng_state.pth','SAMPLER_STATE.json','probe','exit','unknown'):
   with self.subTest(name=name),tiny_failed_saved() as (s,t,v,recipe,data):
    if name=='probe':p=s/'evidence/B_COVERAGE-1024/TRAIN_PROBES/probe-0.npz'
    elif name in ('exit','unknown'):
     c=intake.read(s/'TERMINAL_RECONCILIATION.json')
     if name=='exit':c['native_terminal']['Exit_status']=0
     else:c['effects']['pending_updates']=[{}]
     put(s/'TERMINAL_RECONCILIATION.json',c);p=None
    else:p=s/'training/B_COVERAGE/checkpoint-2048'/name
    if p:p.chmod(0o600);p.write_bytes(b'CORRUPTED_FIXTURE')
    with self.assertRaises((PermissionError,ValueError)):intake.audit_closed_source(v,recipe=recipe,dataset=data,reuse_grant_bytes=GRANT.read_bytes())
if __name__=='__main__':unittest.main()
