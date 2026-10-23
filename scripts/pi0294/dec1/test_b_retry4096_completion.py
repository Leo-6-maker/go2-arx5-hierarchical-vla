"""CPU fixtures only; no simulated result is a native completion receipt."""
import base64,hashlib
from copy import deepcopy
import unittest
from unittest.mock import patch
from . import b_retry4096_completion as c
from .test_b_heartbeat_recovery import scope

def fixture():
 r=dict(scope(),**c.saved.scope(),**c.retry.scope(),request_id=c.REQUEST,source_sha=c.SOURCE,submission_nonce=c.NONCE,
  expected_native_start_step=2048,CPU_upper_s=54000,additional_output_upper_bytes=70000000000,
  training_block=dict(arm='B_COVERAGE',stop_after_step=4096),project='CFP04-CF-007',job_name='pi0294_dec1_a40_train')
 e=dict(request=r,job_id=c.JOB,request_sha256=c.REQUEST_SHA)
 j=dict(job_state='F',Exit_status=0,Job_Owner='e1192548@vanda',project=r['project'],Job_Name=r['job_name'],
  Variable_List=dict(PV1_REQUEST_ID=c.REQUEST,PV1_SUBMISSION_NONCE=c.NONCE),Rerunable=False,run_count=1,Resource_List=dict(ngpus=1))
 effects=dict(job_id=c.JOB,request_id=c.REQUEST,request_sha256=c.REQUEST_SHA,source_sha=c.SOURCE,arm='B_COVERAGE',
  start_native_step=2048,last_native_step=4096,optimizer_updates_committed=2048,query_attempts=17,model_load_attempts=2,
  recipe_sha256=c.b.RECIPE,pending_updates=[],unknown_query_intents=[],effect_journal_complete=True,new_physics=0,new_episodes=0)
 probe=b'{"CPU_fixture_only":true}\n';digest=hashlib.sha256(probe).hexdigest()
 members={x:'a'*64 for x in ('optimizer.pt','scheduler.pt','rng_state.pth','SAMPLER_STATE.json','trainer_state.json',
  'statistics.json','config.json','processor/tokenizer.json','model-00001-of-00002.safetensors',
  'model-00002-of-00002.safetensors','model.safetensors.index.json','metadata.json','recipe.json','data.json','extra.json')}
 v=dict(raw_native=j,effects=effects,native_proof=dict(job_id=c.JOB,request_id=c.REQUEST,job_state='F',Exit_status=0,
  recorded_process_terminal=True,new_resident_allocated_upper_bytes=70000000000),process_final=dict(status='A40_SOURCE_4096_COMPLETE',
  global_step=4096,last_returned_training_step=4096,error=None,source_sha=c.SOURCE,request_sha256=c.REQUEST_SHA,job_id=c.JOB,effects=deepcopy(effects)),
  wrapper_final=dict(job_id=c.JOB,returncode=0),checkpoints={str(s):dict(members=15,actual_consumed_cursor=s*32,
   complete_sha256=c.b.SEED_SHA if s==1024 else c.saved.CHECKPOINT if s==2048 else 'c'*64) for s in (1024,2048,4096)},
  checkpoint_members={str(s):deepcopy(members) for s in (1024,2048,4096)},probe_count=12,new_1024_probe_queries=0,
  imported1024_and2048_origin_preserved=True,models_loaded=0,new_queries=0,new_updates=0,new_physics=0,
  original_probe_result_base64=base64.b64encode(probe).decode(),probe_result_sha256=digest,
  probe_files={'evidence/B_COVERAGE-1024/TRAIN_PROBES/'+('RESULT.json' if i==0 else str(i)+'.npz'):digest if i==0 else 'b'*64 for i in range(13)})
 return e,v,digest

class Completion(unittest.TestCase):
 def test_owner_must_really_exit_naturally_and_successfully(self):
  c.owner(dict(MainPID='0',ActiveState='inactive',ExecMainStatus='0'))
  for state in (dict(MainPID='1',ActiveState='active',ExecMainStatus='0'),dict(MainPID='0',ActiveState='failed',ExecMainStatus='1'),{}):
   with self.subTest(state=state),self.assertRaises(PermissionError):c.owner(state)
 def test_exact_current_request_native_terminal_and_actual_effects(self):
  e,v,_=fixture();c.native(v['raw_native'],e);c.effects(v['effects'],e)
  for key,value in (('job_state','R'),('job_state','H'),('Exit_status',1),('run_count',21),('Job_Owner','other')):
   j=deepcopy(v['raw_native']);j[key]=value
   with self.subTest(key=key,value=value),self.assertRaises(PermissionError):c.native(j,e)
  for key,value in (('query_attempts',29),('model_load_attempts',4),('optimizer_updates_committed',1024),('pending_updates',[{}]),('recipe_sha256','0'*64)):
   z=deepcopy(v['effects']);z[key]=value
   with self.subTest(key=key),self.assertRaises(PermissionError):c.effects(z,e)
 def test_all_three_different_origins_and_full15_members(self):
  e,_,_=fixture()
  for s,job,source in ((1024,c.b.FAILED_JOB,c.b.FAILED_SOURCE),(2048,c.saved.JOB,c.saved.SOURCE),(4096,c.JOB,c.SOURCE)):
   v=dict(global_step=s,full_resume=True,files={str(i):'a'*64 for i in range(15)},origin_job_id=job,origin_source_sha=source);c.origin(s,v,e)
   for key,value in (('origin_job_id',c.saved.JOB if s==1024 else c.b.FAILED_JOB),('origin_source_sha','0'*40),('full_resume',False),('files',{})):
    z=deepcopy(v);z[key]=value
    with self.subTest(step=s,key=key),self.assertRaises(PermissionError):c.origin(s,z,e)
 def test_full_final_audit_original_probe_bytes_and_no_new_neural_work(self):
  e,v,digest=fixture()
  with patch.object(c.b,'PROBE_SHA',digest):self.assertIs(c.audit(v,e),v)
  for mode in ('optimizer','scheduler','RNG','sampler','processor','model','cursor','probe','output','wrapper','returned','query'):
   z=deepcopy(v)
   if mode in ('optimizer','scheduler','RNG','sampler'):z['checkpoint_members']['4096'].pop(dict(optimizer='optimizer.pt',scheduler='scheduler.pt',RNG='rng_state.pth',sampler='SAMPLER_STATE.json')[mode])
   elif mode=='processor':z['checkpoint_members']['4096'].pop('processor/tokenizer.json')
   elif mode=='model':z['checkpoint_members']['4096']['model-00002-of-00002.safetensors']='not-sha'
   elif mode=='cursor':z['checkpoints']['4096']['actual_consumed_cursor']=65536
   elif mode=='probe':z['original_probe_result_base64']=base64.b64encode(b'changed').decode()
   elif mode=='output':z['native_proof']['new_resident_allocated_upper_bytes']=0
   elif mode=='wrapper':z['wrapper_final']['returncode']=1
   elif mode=='returned':z['process_final']['last_returned_training_step']=2048
   else:z['new_queries']=1
   with self.subTest(mode=mode),patch.object(c.b,'PROBE_SHA',digest),self.assertRaises(PermissionError):c.audit(z,e)
 def test_other_request_or_unbound_grant_never_relabels_old_completion(self):
  for mode in ('job','request','source','nonce','scope'):
   e,_,_=fixture()
   if mode=='job':e['job_id']=c.saved.JOB
   elif mode=='scope':e['request'].pop(c.retry.SCOPE_KEYS[1])
   else:e['request'][dict(request='request_id',source='source_sha',nonce='submission_nonce')[mode]]='bad'
   with self.subTest(mode=mode),self.assertRaises(PermissionError):c.expected(e)
if __name__=='__main__':unittest.main()
