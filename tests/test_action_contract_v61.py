import copy,importlib.abc,json,os,subprocess,sys
from types import SimpleNamespace
import pytest
from go2arx5_vla.sla_adapter.action_contract_v61 import final_plan,consumed_mask


def plan(valid=True):
    raw=[[0.,0.,.6,1.,0.,0.,0.,1.,0.,.5]for _ in range(40)]
    return final_plan(SimpleNamespace(decoded_H40=raw,reference_path=raw),query_id='original-query',
        native_step=5703,planner_sha256='a'*64,adapter_sha256='b'*64,normalizer_sha256='c'*64,full_H40_valid=valid)


def event(p,k):
    return dict(kind='NATIVE_ATOM',origin_query_id=p['query_id'],command_id=p['command_ids'][k],k=k,
        native_start_step=5703+25*k,native_end_step=5703+25*(k+1),actual_ticks=25,
        solver_policies_with_exact_installed_target=True,gripper_command_consumed=True)


def test_final_plan_snapshot_and_consumption_are_separate():
    p=plan();events=[event(p,k)for k in range(10)]
    events.extend([dict(kind='REFERENCE_RENEWAL_ACK')]*20)
    assert p['supervision_mask']==[True]*40 and p['consumed_mask']==[False]*40
    assert consumed_mask(p,events)==[True]*10+[False]*30
    assert consumed_mask(p,events+events)==consumed_mask(p,events)
    assert not p['task_success_claim']


@pytest.mark.parametrize('field,value',[('origin_query_id','later-query'),('command_id','equal-value-different-command'),('native_start_step',5953),('native_end_step',6003)])
def test_equal_values_or_changed_clock_cannot_backfill_original_query(field,value):
    p=plan();e=event(p,0);e[field]=value
    with pytest.raises(ValueError):consumed_mask(p,[e])


def test_partial_or_unconfirmed_solver_gripper_is_not_consumption():
    p=plan();events=[event(p,k)for k in range(10)]
    for e in events[6:]:e['gripper_command_consumed']=False
    assert sum(consumed_mask(p,events))==6
    events[5]['actual_ticks']=24
    assert sum(consumed_mask(p,events))==5
    assert plan(False)['supervision_mask']==[False]*40


def test_snapshot_does_not_alias_mutable_program_and_no_heavy_imports():
    raw=[[0.,0.,.6,1.,0.,0.,0.,1.,0.,.5]for _ in range(40)]
    p=final_plan(SimpleNamespace(decoded_H40=raw,reference_path=raw),query_id='query',native_step=5703,
        planner_sha256='a'*64,adapter_sha256='b'*64,normalizer_sha256='c'*64,full_H40_valid=True)
    raw[0][0]=999;assert p['final_H40'][0][0]==0
    code="""
import sys,importlib.abc
class Finder(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('numpy','scipy','jax','flax','torch'):raise ImportError(fullname)
sys.meta_path.insert(0,Finder())
import importlib.util
spec=importlib.util.spec_from_file_location('v61_metadata','src/go2arx5_vla/sla_adapter/action_contract_v61.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
"""
    subprocess.run([sys.executable,'-c',code],env=dict(os.environ),check=True,capture_output=True)


def test_selection_keeps_all_stages_and_distinct_replan_events():
    from go2arx5_vla.sla_adapter.action_contract_v61 import selection,STATS_SHA
    stages=['PREGRASP']*3+['DESCEND']*20+['CLOSE']*4+['LIFT']*6+['HOLD']*28
    episodes=[dict(id=f'TRAIN{i:02d}-r{r}',layout=f'TRAIN{i:02d}',restart=r,
        expert_stage_audit_only=stages)for r in range(3)for i in range(8)][:20]
    m=dict(dataset_id='T_PICK_HOLD_N20_ANCHOR_EQUIVALENT_CONSUMED_REFERENCE_V6',
        statistics=dict(sha256=STATS_SHA),episodes=episodes,supervised_windows=1220)
    s=selection(m);assert len({r['id']for r in s['S16']})==16
    assert s['selected_layouts']==8 and len(s['RP8'])==8
    assert [r['stage']for r in s['S16'][:10]]==[x for phase in ('PREGRASP','DESCEND','CLOSE','LIFT','HOLD')for x in(phase,phase)]
    assert selection(dict(m,episodes=list(reversed(episodes))))==s
    bad=copy.deepcopy(m)
    for e in bad['episodes']:e['expert_stage_audit_only']=['HOLD']*61
    with pytest.raises(ValueError,match='MISSING_STAGE'):selection(bad)


def test_certification_status_keeps_original_command_ids_and_raw_stats_can_be_unfitted():
    a,b=plan(False),plan(True)
    assert a['plan_sha256']==b['plan_sha256'] and a['command_ids']==b['command_ids']
    raw=[[0.,0.,.6,1.,0.,0.,0.,1.,0.,.5]for _ in range(40)]
    p=final_plan(SimpleNamespace(decoded_H40=raw,reference_path=raw),query_id='raw-plan',native_step=5703,
        planner_sha256='a'*64,adapter_sha256='b'*64,normalizer_sha256=None,full_H40_valid=False)
    assert p['normalizer_sha256'] is None and not any(p['supervision_mask'])
