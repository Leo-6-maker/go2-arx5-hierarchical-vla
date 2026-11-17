import numpy as np
from go2arx5_vla.sla_adapter.mobile_endpoint_v52 import clean_estimate,endpoint_loss
from scripts.wm02.mobile_d0_v52 import boundary


def test_clean_identity_and_physical_units():
 rng=np.random.default_rng(52);actions=rng.normal(size=(4,40,32));noise=rng.normal(size=actions.shape);tau=np.array([0.,.25,.5,.9]);x=tau[:,None,None]*noise+(1-tau[:,None,None])*actions
 np.testing.assert_allclose(clean_estimate(x,tau,noise-actions),actions,atol=1e-15)
 low=np.zeros(7);high=np.full(7,2.-1e-6);a=np.zeros((4,40,32));a[:,1,0]=.001
 # physical endpoint acceleration .4 -> hinge (.4-.2)/.25=.8;
 # one endpoint and three of four eligible -> .64/2*3/4.
 assert np.isclose(endpoint_loss(a,tau,low,high,xp=np),.24)
 assert endpoint_loss(np.zeros_like(a),tau,low,high,xp=np)==0


def test_full_batch_mask_and_micro4_reduction_are_equivalent():
 rng=np.random.default_rng(53);a=rng.normal(size=(32,40,32))*.001;tau=rng.random(32);lo=np.arange(7)*.1;hi=lo+2
 whole=endpoint_loss(a,tau,lo,hi,xp=np);micro=np.mean([endpoint_loss(a[i:i+4],tau[i:i+4],lo,hi,xp=np)for i in range(0,32,4)])
 assert np.isclose(whole,micro,rtol=1e-14)


def test_boundary_index_uses_original_zero_endpoint_velocity():
 a=np.zeros((40,10));a[:,0]=np.arange(40)*.0007;b=boundary(a)
 assert b['peak_index']in(0,39) and np.isclose(b['start'],.28)and np.isclose(b['end'],.28)
 assert b['interior']<1e-12


def test_successor_package_rejects_old_parent_and_changed_objective():
 import copy
 import pytest
 from go2arx5_vla.sla_adapter.provider_mobile_v52 import contract,SCHEMA,ORDER
 from go2arx5_vla.sla_adapter.mobile_successor_v52 import RECIPE,PARENT,PLAN
 from go2arx5_vla.sla_adapter.gripper_time_shift7_v1 import VERSION,SHIFT,SLOTS
 from go2arx5_vla.sla_adapter.gripper_latent_v2 import VERSION as STATS
 b=dict(recipe=RECIPE,successor_parent_package=dict(sha256=PARENT),parent_cumulative_updates=1500,parent_cumulative_visits=48000,successor_cumulative_updates=2000,successor_cumulative_visits=64000,new_planned_updates=500,new_planned_visits=16000,endpoint_lambda=.1,endpoint_margin_m_s2=.2,endpoint_scale_m_s2=.25,endpoint_tau_max=.5,goal_plan_sha=PLAN,fresh_Adam=True,codec_version=VERSION,statistics_codec_version=STATS,inference_route='cached_actual')
 m=dict(schema=SCHEMA,codec=VERSION,decode_order=ORDER,step=500,cursor=16000,bindings=b,parameter_slot_shift=SHIFT,parameter_slots=[[int(i)for i in g]for g in SLOTS],previous_reference_required=True,teacher_required=False,optimizer_required=False,training_cache_required=False,physical_execution_admitted=False,endpoint_basis_applied=False)
 contract(m)
 for key,value in (('parent_cumulative_updates',1000),('endpoint_margin_m_s2',.25),('endpoint_lambda',.5),('fresh_Adam',False)):
  invalid=copy.deepcopy(m);invalid['bindings'][key]=value
  with pytest.raises(ValueError):contract(invalid)


def test_roll96_is_unique_balanced_and_keeps_all_failed_and_precontact_inputs():
 from scripts.wm02.mobile_prepare_v52 import balanced_roll
 rows=[]
 for family,cases in(('F1',[('09-student-f1',23),('01-student-f1',70),('03-student-f1',71),('05-student-f1',72)]),('F2',[('00-student-f2',45),('02-student-f2',3),('04-student-f2',13),('06-student-f2',22)])):
  for case,n in cases:
   for q in range(n):rows.append(dict(case=case,family=family,query=q,action_sha256='fixed',consumption=dict(full_H40_compiled=not(family=='F2'and case!='00-student-f2'and q==n-1))))
 selected=balanced_roll(rows);keys={(x['case'],x['query'])for x in selected}
 assert len(selected)==len(keys)==96
 assert all(sum(x['family']==f for x in selected)==48 for f in('F1','F2'))
 assert {('00-student-f2',q)for q in range(35,45)}<=keys
 assert {('02-student-f2',2),('04-student-f2',12),('06-student-f2',21)}<=keys
 assert selected==balanced_roll(list(reversed(rows)))


def test_followup_requires_actual_500_endpoints_and_matching_initial_randomness(tmp_path,monkeypatch):
 import json,time,pytest
 from scripts.wm02 import mobile_e1_followup_v52 as followup
 from go2arx5_vla.sla_adapter.mobile_successor_v52 import PARENT
 monkeypatch.setattr(followup,'unit_state',lambda unit:dict(pid=0,cgroup_pids=[]))
 monkeypatch.setattr(followup.subprocess,'Popen',lambda *a,**kw:pytest.fail('gate test must not launch any process'))
 initial=dict(parent_package=dict(sha256=PARENT),all_state_array_hashes=[str(i)for i in range(87)],visit_map_sha256='same-visits',LR_sha256='same-LR',noise_sha256='same-noise',tau_sha256='same-time',state_dropout=[True,False],first_indices=[1,2],fresh_Adam=True)
 specs={}
 for arm in('B1','B2'):
  folder=tmp_path/arm;folder.mkdir();(folder/'control').mkdir()
  for name,value in [('EXECUTION_SPEC.json',dict(arm=arm)),('control/LEASE.json',dict(status='CLOSED')),('INITIAL_MATCH.private.json',initial),('RESULT.private.json',dict(exit_code=0,status='FIXED500_COMPLETE',new_updates=500,new_visits=16000,full_restore=dict(bit_exact=True,arrays=87),parent_package=dict(sha256=PARENT),frozen_fingerprint_unchanged='same-frozen'))]:
   (folder/name).write_text(json.dumps(value))
  specs[arm]=followup.pin(folder/'EXECUTION_SPEC.json')
 s=dict(registered_root=str(tmp_path),E1_specs=specs,deadline_epoch=time.time()+3600)
 assert set(followup.wait_endpoints(s))=={'B1','B2'}
 changed=dict(initial,noise_sha256='different-noise');(tmp_path/'B2/INITIAL_MATCH.private.json').write_text(json.dumps(changed))
 with pytest.raises(ValueError,match='MATCHED_INITIAL_STATE_MISMATCH_noise'):followup.wait_endpoints(s)
 bad=json.loads((tmp_path/'B2/RESULT.private.json').read_text());bad['new_updates']=499;(tmp_path/'B2/RESULT.private.json').write_text(json.dumps(bad))
 with pytest.raises(ValueError,match='TWO_REAL_MATCHED500'):followup.wait_endpoints(s)
