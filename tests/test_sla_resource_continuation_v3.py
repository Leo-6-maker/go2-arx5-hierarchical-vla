import copy
import pytest
from go2arx5_vla.world_model import sla_native_resource_continuation_v3 as resource


def snapshot():
 retained=[dict(layout='TRAIN01',hard_safety_stop=False),dict(layout='TRAIN07',hard_safety_stop=False)]
 rows=retained+[dict(layout='TRAIN13',native_run='s1-train13',attempted=True,technical_valid=True,
  policy_proposal_rejected=True,task_success=False,actual_SLA_queries=1,actual_native_integrations=5703,hard_safety_stop=False)]
 panel=dict(schema='fixed',task_learning_role='SLA_TARGET500_NATIVE_EVALUATION_V1',protocol={'fixed':True},population='TRAIN4',
  method='same',policies={'same':'weight'},preregistration={'same':'population'},assignments=['01','07','13','25'],
  registration=dict(source_root='/new',source_files={'code':'new'},native_base='original',gpu_base='original',
   capacity_wait_before_pair_v3=dict(version=resource.WAIT_VERSION,poll_s=60)))
 prior=copy.deepcopy(panel);prior['source_sha']=resource.SOURCE;prior['registration'].update(source_root='/old',source_files={'code':'old'});prior['registration'].pop('capacity_wait_before_pair_v3')
 result=dict(source_sha=resource.SOURCE,exit_code=1,failure='ValueError: '+resource.CAPACITY_REASON,assignment_slice=[2,4],
  retained_prefix_conditions=2,completed_in_group=1,cumulative_completed=3,summary=dict(completed=3,complete=False))
 lease=dict(status='CLOSED',worker_exit_code=1);inp=dict(source_sha=resource.SOURCE,assignment_slice=[2,4])
 return panel,retained,result,rows,lease,inp,prior


def test_retains_all_three_outcomes_with_same_policy_and_population():
 values=snapshot();assert resource.joined_prefix(*values)==values[3]
 assert resource.configuration(values[0])['version']==resource.WAIT_VERSION


@pytest.mark.parametrize('mutation',['swap_weight','remove_failure','technical','success','safety','source','slice','live_lease','reset_input'])
def test_cannot_promote_unknown_changed_or_retried_prefix(mutation):
 p,old,r,rows,l,i,prior=snapshot()
 if mutation=='swap_weight':p['policies']['same']='other'
 elif mutation=='remove_failure':old.pop()
 elif mutation=='technical':rows[2]['technical_valid']=False
 elif mutation=='success':rows[2]['task_success']=True
 elif mutation=='safety':rows[2]['hard_safety_stop']=True
 elif mutation=='source':r['source_sha']='different'
 elif mutation=='slice':i['assignment_slice']=[0,4]
 elif mutation=='live_lease':l['status']='ACTIVE'
 else:p['registration']['native_base']='different'
 with pytest.raises(ValueError):resource.joined_prefix(p,old,r,rows,l,i,prior)


def test_suffix_selects_only_train25_and_never_replays_train13(monkeypatch):
 from go2arx5_vla.world_model import sla_native_v1 as sla,sla_native_continuation_v2 as cold
 rows=[dict(layout=x)for x in ('TRAIN01','TRAIN07','TRAIN13','TRAIN25')]
 predecessor=dict(path='/actual3/RESULT.private.json',sha256=resource.RESULT)
 panel=dict(source_sha='new',resource_continuation=dict(result=predecessor),cold_continuation=dict(result='original2'))
 monkeypatch.setattr(sla,'validate_panel',lambda p:rows);monkeypatch.setattr(cold,'prefix',lambda p:rows[:3])
 inp=dict(schema=cold.INPUT_SCHEMA,source_sha='new',assignment_slice=[3,4],population_predecessor=predecessor)
 assert sla.selection(inp,panel)==rows[3:]
 for span in ([0,4],[2,4],[3,3]):
  with pytest.raises(ValueError):sla.selection(dict(inp,assignment_slice=span),panel)
 with pytest.raises(ValueError):sla.selection(dict(inp,population_predecessor='original2'),panel)


def test_wait_rechecks_only_effect_free_admission_and_allocates_nothing():
 clock=[0.];calls=[];saved=[]
 def attempt():
  calls.append(1)
  if len(calls)<=2:raise ValueError(resource.CAPACITY_REASON)
  return 'one-registration'
 value=resource.wait_for_pair(attempt,lambda:True,saved.append,180,now=lambda:clock[0],sleep=lambda n:clock.__setitem__(0,clock[0]+n))
 assert value=='one-registration' and len(calls)==3 and len(saved)==2
 assert all(not x['effect_started']and x['model_calls']==x['physical_requests']==0 for x in saved)


def test_effect_intent_or_unrelated_failure_is_never_replayed():
 calls=[]
 def attempt():calls.append(1);raise ValueError(resource.CAPACITY_REASON)
 with pytest.raises(RuntimeError,match='UNKNOWN_NO_ADMISSION_REPLAY'):
  resource.wait_for_pair(attempt,lambda:False,lambda x:pytest.fail('unknown cannot wait'),100,now=lambda:0)
 assert len(calls)==1
 with pytest.raises(ValueError,match='different_failure'):
  resource.wait_for_pair(lambda:(_ for _ in ()).throw(ValueError('different_failure')),lambda:True,lambda x:pytest.fail('unknown reason'),100,now=lambda:0)


def test_finite_wait_expiry_and_stop_never_start_an_actor():
 clock=[0.];saved=[]
 def attempt():raise ValueError(resource.CAPACITY_REASON)
 with pytest.raises(TimeoutError,match='NO_DISPATCH'):
  resource.wait_for_pair(attempt,lambda:True,saved.append,61,now=lambda:clock[0],sleep=lambda n:clock.__setitem__(0,clock[0]+n))
 assert len(saved)==2 and clock[0]==61
 with pytest.raises(PermissionError,match='STOP_RETAINED'):
  resource.wait_for_pair(lambda:pytest.fail('stopped'),lambda:True,saved.append,100,stopped=lambda:True,now=lambda:0)
