import copy

import pytest

from go2arx5_vla.world_model import sla_r1_suffix_v5 as suffix
from go2arx5_vla.world_model.task_learning_student_batch_v1 import condition_window


def population():
    assignments=[dict(native_run=f'fixed-{i}',method='R0'if i<8 else'R1',
        layout='TRAIN13'if i==12 else'TRAIN01',restart=3,query_seeds=list(range(15)))for i in range(16)]
    old=dict(schema='fixed-r1',task_learning_role='SLA',protocol={'unchanged':True},population='TRAIN8',
        B1_global_step=500,policies={'same':True},preregistration={'same':'hash'},assignments=assignments,
        source_sha='old')
    panel=copy.deepcopy(old);panel.update(source_sha='cold',r1_cold_suffix_v5={'schema':suffix.SCHEMA},
        registration={'condition_wall_window_v5':dict(original_s=1200,new_s=3600,
            reason='ACTUAL_20128_OF21203_TICKS_AT1200S_UNDER_PARALLEL_LOAD',
            physics_clock_changed=False,model_changed=False,score_changed=False)})
    rows=[dict(r,technical_valid=True,hard_safety_stop=False,actual_SLA_queries=1,
               actual_native_integrations=5703,first_failure=None)for r in assignments[:13]]
    rows[-1].update(technical_valid=False,actual_SLA_queries=15,actual_native_integrations=20128,
                   first_failure='TimeoutError: AUTO1_FINITE_ENDPOINT_OR_OPERATOR_STOP')
    result=dict(exit_code=1,unresolved_pair=None,completed_in_group=13,source_sha='old',
                summary=dict(requested=16,complete=False))
    lease=dict(status='CLOSED',worker_exit_code=1)
    inp=dict(assignment_slice=[0,16],source_sha='old')
    return panel,result,rows,lease,inp,old


def test_timeout_and_every_other_outcome_remain_in_full_fixed_population():
    p,r,rows,l,i,old=population()
    assert suffix.joined(p,r,rows,l,i,old)==rows
    assert condition_window(p)==3600
    assert condition_window(old)==1200
    assert [a['native_run']for a in p['assignments'][13:]]==['fixed-13','fixed-14','fixed-15']


@pytest.mark.parametrize('mutation',['drop_failure','changed_seed','changed_policy','not_closed','unknown_pair','hard_stop'])
def test_suffix_cannot_erase_retry_or_relax_an_actual_condition(mutation):
    p,r,rows,l,i,old=population()
    if mutation=='drop_failure':rows.pop()
    elif mutation=='changed_seed':p['assignments'][13]['query_seeds'][0]=99
    elif mutation=='changed_policy':p['policies']={'changed':True}
    elif mutation=='not_closed':l['status']='ACTIVE'
    elif mutation=='unknown_pair':r['unresolved_pair']={'unknown':True}
    else:rows[0]['hard_safety_stop']=True
    with pytest.raises(ValueError):suffix.joined(p,r,rows,l,i,old)


def test_wall_allowance_cannot_change_physics_or_score():
    p,*_=population()
    p['registration']['condition_wall_window_v5']['physics_clock_changed']=True
    with pytest.raises(ValueError):condition_window(p)


def test_selection_runs_only_never_started_suffix(monkeypatch):
    p,_,rows,*_=population();p['r1_cold_suffix_v5']['result']={'path':'closed','sha256':'a'*64}
    monkeypatch.setattr(suffix,'prefix',lambda panel:rows)
    inp=dict(schema=suffix.SCHEMA,source_sha='cold',assignment_slice=[13,16],
             population_predecessor=p['r1_cold_suffix_v5']['result'])
    assert suffix.selection(inp,p,p['assignments'])==p['assignments'][13:]
    inp['assignment_slice']=[12,16]
    with pytest.raises(ValueError):suffix.selection(inp,p,p['assignments'])


def test_second_zero_tick_failure_stays_in_denominator_and_only_last_two_run(monkeypatch):
    p,_,rows,*_=population();p['r1_cold_suffix_v5']['result']={'path':'closed','sha256':'a'*64}
    rows.append(dict(p['assignments'][13],technical_valid=False,actual_native_integrations=0))
    monkeypatch.setattr(suffix,'prefix',lambda panel:rows)
    inp=dict(schema=suffix.SCHEMA,source_sha='cold',assignment_slice=[14,16],
        population_predecessor=p['r1_cold_suffix_v5']['result'])
    assert suffix.selection(inp,p,p['assignments'])==p['assignments'][14:]
    inp['assignment_slice']=[13,16]
    with pytest.raises(ValueError):suffix.selection(inp,p,p['assignments'])


def test_native_entry_accepts_only_declared_suffix_wall_repair(tmp_path,monkeypatch):
    import json
    from go2arx5_vla.world_model.storage import digest
    from go2arx5_vla.world_model import task_learning_grasp_v1 as task
    from go2arx5_vla.world_model import sla_r1_diagnostic_v5 as diagnostic
    from go2arx5_vla.world_model.native_live_control_v3 import maximum_wall_seconds
    panel,*_=population()
    path=tmp_path/'panel.json';path.write_text(json.dumps(panel))
    monkeypatch.setattr(diagnostic,'validate_panel',lambda p:p['assignments'])
    pin=dict(path=str(path),sha256=digest(path))
    spec=dict(task_learning_role=task.SLA_ROLE,task_learning_grasp=task.protocol(task.SLA_ROLE),
        r1_condition_wall_repair_v5=pin,stable_task_panel=str(path),stable_task_panel_sha256=pin['sha256'],
        registered_root=str(tmp_path/'fixed-13'))
    assert maximum_wall_seconds(spec,1200)==3600
    assert maximum_wall_seconds({},1200)==1200
    spec['registered_root']=str(tmp_path/'fixed-12')
    with pytest.raises(ValueError):maximum_wall_seconds(spec,1200)


def test_campaign_wall_allowance_is_independent_of_tmpfs_transport(tmp_path,monkeypatch):
    from go2arx5_vla.world_model import auto1_native,native_live_control_v3 as live
    from go2arx5_vla.world_model.storage import digest
    binding=tmp_path/'binding.json';binding.write_text('{}')
    spec=dict(campaign_root=str(tmp_path),campaign_binding=str(binding),
        phase='collect-target-task-learning-grasp-v1',domain='WM02_AUTO1_TARGET_COMMISSION_V1',
        physics=True,site_profile='4080-simulation',owner='EXISTING_LOCAL_PC',
        host='corelab-4080s',uid=2009,wall_limit_s=3600,r1_condition_wall_repair_v5={'typed':'proof'})
    lease=dict(status='ACTIVE',scope='WM02_AUTO1_SIM_ONLY',owner=spec['owner'],run_uuid=tmp_path.name,
               binding_sha256=digest(binding),deadline_epoch=100)
    registry=dict(entries=[dict(scope=lease['scope'],owner=spec['owner'],path=str(tmp_path),
                              local_lease=str(tmp_path/'control/LEASE.json'))])
    calls=[]
    monkeypatch.setattr(live,'maximum_wall_seconds',lambda s,old:calls.append((s,old))or 3600)
    auto1_native.validate_scope(spec,lease,registry,host='corelab-4080s',uid=2009,now=50)
    assert calls==[(spec,1200)] and 'native_live_control_v3'not in spec
