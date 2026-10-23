"""Synthetic endpoint identities only; no claim of a native terminal/audit."""
from copy import deepcopy
import unittest
from . import b_recovery_completion as c
from .test_b_heartbeat_recovery import scope

def expected():
 req=scope();req.update(source_sha='2ed8aea9f782741dfc0298438f5762c592e87149',project='CFP04-CF-007',job_name='pi0294_dec1_a40_train',request_id='CPU_REQUEST',submission_nonce='CPU_NONCE',expected_native_start_step=1024,training_block=dict(arm='B_COVERAGE',stop_after_step=2048))
 return dict(request=req,job_id='1.stdct-mgmt-02',request_sha256='1'*64)
def effects(e):
 return dict(job_id=e['job_id'],request_id=e['request']['request_id'],request_sha256=e['request_sha256'],source_sha=e['request']['source_sha'],arm='B_COVERAGE',start_native_step=1024,last_native_step=2048,optimizer_updates_committed=1024,query_attempts=17,model_load_attempts=2,recipe_sha256=c.b.RECIPE,pending_updates=[],unknown_query_intents=[],effect_journal_complete=True,new_physics=0,new_episodes=0)
def native(e):
 return dict(job_state='F',Exit_status=0,Job_Owner='e1192548@CPU_FIXTURE',project='CFP04-CF-007',Job_Name='pi0294_dec1_a40_train',Variable_List=dict(PV1_REQUEST_ID=e['request']['request_id'],PV1_SUBMISSION_NONCE=e['request']['submission_nonce']),Rerunable=False,run_count=1,Resource_List=dict(ngpus=1))
class Completion(unittest.TestCase):
 def test_failed_process_never_qualifies_success_even_at_target_step(self):
  e=expected();n=native(e);n['Exit_status']=1;c.failed_native(n,e)
  with self.assertRaises(PermissionError):c.native(n,e)
  for key,value in (('Exit_status',0),('job_state','R'),('run_count',2),('Job_Owner','other@CPU')):
   bad=deepcopy(n);bad[key]=value
   with self.subTest(key=key),self.assertRaises(PermissionError):c.failed_native(bad,e)
 def test_f_q_r_unknown_and_wrong_identity_never_mean_success(self):
  e=expected();c.native(native(e),e)
  for key,value in (('job_state','Q'),('job_state','R'),('job_state','H'),('Exit_status',1),('Job_Owner','other@CPU'),('run_count',2),('Rerunable',True)):
   n=native(e);n[key]=value
   with self.subTest(key=key,value=value),self.assertRaises(PermissionError):c.native(n,e)
 def test_exact_new1024_updates_and_preserved_probes_not_old_B1_effects(self):
  e=expected();v=effects(e);c.effects(v,e)
  for key,value in (('start_native_step',0),('optimizer_updates_committed',2048),('query_attempts',29),('model_load_attempts',4),('last_native_step',1829),('source_sha','0'*40),('recipe_sha256','0'*64),('pending_updates',[{}]),('new_physics',1)):
   w=dict(v);w[key]=value
   with self.subTest(key=key),self.assertRaises(PermissionError):c.effects(w,e)
 def test_split_origins_do_not_relabel_copied_seed_as_new_state(self):
  e=expected();seed=dict(full_resume=True,global_step=1024,files={str(n):'1'*64 for n in range(15)},origin_job_id=c.b.FAILED_JOB,origin_source_sha=c.b.FAILED_SOURCE)
  new=dict(seed,global_step=2048,origin_job_id=e['job_id'],origin_source_sha=e['request']['source_sha'])
  c.origin(1024,seed,e);c.origin(2048,new,e)
  for step,v in ((1024,dict(seed,origin_job_id=e['job_id'])),(2048,dict(new,origin_job_id=c.b.FAILED_JOB)),(2048,dict(new,files={}))):
   with self.subTest(step=step),self.assertRaises(PermissionError):c.origin(step,v,e)
if __name__=='__main__':unittest.main()
