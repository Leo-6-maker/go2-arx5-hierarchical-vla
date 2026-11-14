import copy
from pathlib import Path

import pytest

from go2arx5_vla.world_model import sla_n20_suffix_v5 as suffix
from go2arx5_vla.world_model import task_learning_grasp_v1 as task


def prefix_fixture():
    assignments=[dict(native_run=f'fixed-{i}',method='N20_FM3K_RAW_P10',
        layout=layout,restart=restart,query_seeds=[i])for i,(layout,restart)in enumerate(
        [('TRAIN01',3),('TRAIN01',4),('TRAIN07',3),('TRAIN07',4),
         ('TRAIN13',3),('TRAIN13',4),('TRAIN19',3),('TRAIN19',4)])]
    old=dict(schema='SLA_FRESH_N20_RAW_P10_TRAIN8_PANEL_V5',source_sha=suffix.OLD_SOURCE,
        task_learning_role=task.SLA_DENSE_ROLE,protocol={'unchanged':True},population='TRAIN8',
        B1_global_step=3000,training_seed=20261112,method='N20_FM3K_RAW_P10',
        policies={'unchanged':'full3K-package'},preregistration={'unchanged':'fixed8'},assignments=assignments)
    panel=copy.deepcopy(old);panel['source_sha']='new-source'
    rows=[dict(row,technical_valid=i<2,policy_proposal_rejected=i<2,task_success=False,
        actual_native_integrations=5703 if i<2 else 2450,actual_SLA_queries=1 if i<2 else 0,
        first_failure='raw-envelope'if i<2 else'RuntimeError: EXACT_NATIVE_PAUSE_NOT_REACHED',
        hard_safety_stop=False)for i,row in enumerate(assignments[:3])]
    result=dict(source_sha=suffix.OLD_SOURCE,exit_code=1,unresolved_pair=None,
        completed_in_group=3,cumulative_completed=3,summary=dict(requested=8,complete=False))
    lease=dict(status='CLOSED',worker_exit_code=1)
    inp=dict(source_sha=suffix.OLD_SOURCE,assignment_slice=[0,8])
    return panel,result,rows,lease,inp,old


def test_exact_prefix_retains_both_policy_negatives_and_technical_failure():
    args=prefix_fixture();assert suffix.joined(*args)==args[2]
    for change in ['discard','policy','seed','technical','unknown','safety']:
        a=copy.deepcopy(args)
        if change=='discard':a[2].pop(0)
        elif change=='policy':a[0]['policies']['unchanged']='different-package'
        elif change=='seed':a[0]['assignments'][0]['query_seeds'][0]+=1
        elif change=='technical':a[2][2]['actual_native_integrations']=2500
        elif change=='unknown':a[1]['unresolved_pair']='pending'
        else:a[2][0]['hard_safety_stop']=True
        with pytest.raises(ValueError):suffix.joined(*a)


def test_suffix_selection_cannot_repeat_failed_or_omit_unstarted_conditions(monkeypatch):
    panel,_,rows,*_=prefix_fixture();panel[suffix.KEY]=dict(result={'sha256':'retained'})
    monkeypatch.setattr(suffix,'prefix',lambda p:rows)
    inp=dict(schema='SLA_FRESH_N20_RAW_P10_TRAIN8_GROUP_INPUT_V5',source_sha=panel['source_sha'],
        assignment_slice=[3,8],population_predecessor=panel[suffix.KEY]['result'])
    assert suffix.selection(inp,panel,panel['assignments'])==panel['assignments'][3:]
    for span in [[0,8],[2,8],[3,7],[4,8]]:
        with pytest.raises(ValueError):suffix.selection(dict(inp,assignment_slice=span),panel,panel['assignments'])


def test_late_ack_must_belong_to_the_same_single_grant():
    f=dict(requested_step=2450,pause_ack=dict(step=2425),control=dict(running=True),automatic_ENTER_retries=0)
    a=dict(step=2450,clock_paused=True,running=False,time_s=4.9)
    requests=[dict(requested_step=25*(i+1),publication_count=1,matched_subscriptions=1)for i in range(98)]
    suffix.clock_evidence(f,a,requests)
    for change in ['old_ack','running','extra_enter','repeated_grant']:
        ff,aa,rr=copy.deepcopy((f,a,requests))
        if change=='old_ack':aa['step']=2425
        elif change=='running':aa['clock_paused']=False
        elif change=='extra_enter':rr[-1]['publication_count']=2
        else:rr.append(rr[-1])
        with pytest.raises(ValueError):suffix.clock_evidence(ff,aa,rr)


def test_dense_reuses_exact_existing_read_allowance_other_roles_stay_rejected():
    from go2arx5_vla.world_model.sla_native_continuation_v2 import wait_seconds
    from go2arx5_vla.world_model.storage import digest
    path=Path(__file__).resolve().parents[1]/'experiments/sla_go2_posttrain/native_pause_read_wait_v1.json'
    spec=dict(task_learning_role=task.SLA_DENSE_ROLE,native_pause_read_wait=dict(path=str(path),sha256=digest(path)))
    assert wait_seconds(spec)==90 and wait_seconds(dict(task_learning_role=task.SLA_DENSE_ROLE))==35
    with pytest.raises(ValueError):wait_seconds(dict(spec,task_learning_role='UNRELATED_ROLE'))
