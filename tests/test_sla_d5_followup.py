from collections import Counter

import numpy as np
import pytest

from go2arx5_vla.sla_adapter.grasp_supervision_d5 import width_losses
from scripts.wm02.sla_followup_population_v1 import population, validate_frozen


def test_width_and_delta_supervise_first_point_and_suffix():
    target = np.full((1, 40), .022)
    previous = np.array([.022])
    prediction = target.copy()
    prediction[0, 0] += .002
    prediction[0, 39] -= .002
    width, delta = width_losses(prediction, target, previous, xp=np)
    assert width == pytest.approx(2*(.002/.040)**2/40)
    assert delta == pytest.approx(3/40)
    assert width_losses(target, target, previous, xp=np) == (0., 0.)


def test_fresh_populations_balance_arm_positions_and_resets():
    cards = {k: dict(object_positions_world=dict(red=[0.,.03,.4], blue=[0.,-.03,.4])) for k in ('F1','F2')}
    bank = population(cards, [])
    for c in bank['clusters']:
        c['scene'] = dict(sha256=c['id'])
    validate_frozen(bank)
    assert {k: sum(len(r['arm_order']) for r in rows) for k,rows in bank['conditions'].items()} == {
        'D5_GRASP24':24, 'D5_CONFIRM32':32, 'J_SCREEN48':48}
    for arm in ('D4-B','D5-W','D5-T'):
        positions = Counter(r['arm_order'].index(arm) for r in bank['conditions']['D5_GRASP24'])
        assert max(positions.values())-min(positions.values()) <= 1
    with pytest.raises(ValueError, match='OVERLAPS_AN_ARCHIVED'):
        population(cards, [bank])


def test_resume_only_the_cold_counted_suffix(tmp_path, monkeypatch):
    from pathlib import Path

    from go2arx5_vla.world_model import sla_followup_native_v1 as plan
    from go2arx5_vla.world_model.storage import digest
    from scripts.pi0292.durable import write
    from scripts.wm02 import d416_core_native_launch, d416_j_screen_suffix, multitask_panel_suffix as suffix

    arms = ('J0','J1')
    conditions = [(dict(id=f'condition{i}',instruction='HOLD-red'),arm)
                  for i in (0,1) for arm in arms]
    root = tmp_path/'initial'

    def pinned(path, value):
        path.parent.mkdir(parents=True,exist_ok=True)
        write(path,value)
        return dict(path=str(path),sha256=digest(path))

    cards, rows = [], []
    for i,(condition,arm) in enumerate(conditions):
        name = condition['id']+'-'+arm
        parent = root/'cases'/name
        card = pinned(parent/'EXECUTION_SPEC.json',dict(arm=arm))
        cards.append(card['path'])
        if i >= 2:
            continue
        source = pinned(parent/'RESULT.private.json',dict(physical_requests=1,actual_native_integrations=5953,
                       model_generations=1,technical_failure=False,source_sha='first'))
        pinned(parent/'control/LEASE.json',dict(status='CLOSED',spec_sha256=card['sha256']))
        rows.append(dict(id=name,role='STUDENT',result_sha256=source['sha256'],native_ticks=5953,
                         actual_model_generations=1,technical_failure=False,integrations_after_STOP=0,
                         score=dict(native_step=5953,missing_native_intervals=0)))
    population_card = pinned(tmp_path/'population.json',dict(frozen=True))
    registry = pinned(tmp_path/'registry.json',dict(entries=[]))
    old_authority = pinned(tmp_path/'first-authority.json',dict(first=True))
    new_authority = pinned(tmp_path/'next-authority.json',dict(first=False))
    spec = dict(scope=plan.SCOPE,plan_commit=plan.PLAN,evaluation_stage=plan.JOINT,compiled_population=population_card,
                packages={arm:arm for arm in arms},interface_predecessor='interface',mobile_world_lock='lock',
                ROS_DOMAIN_ID=198,owner='owner',registry=registry['path'],source_root=str(tmp_path/'source'),
                sim_only_project_authorization=old_authority)
    prev = pinned(root/'EXECUTION_SPEC.json',dict(spec,registered_root=str(root),source_sha='first',cases=cards))
    pinned(root/'control/LEASE.json',dict(status='CLOSED',spec_sha256=prev['sha256']))
    pinned(root/'PROGRESS.private.json',dict(phase='FINITE_STOP_OR_DEADLINE'))
    result = dict(source_sha='first',exit_code=1,unresolved_case_launches=0,failure=None,records=rows,
                  attempted_case_launches=2,new_verified_native_requests=2,
                  model_service=dict(exit_code=0,attempted_generations=2,completed_generations=2))
    actual = pinned(root/'RESULT.private.json',result)
    spec['previous_panel_segments'] = [dict(spec=prev,result=actual)]
    monkeypatch.setattr(suffix,'unit_state',lambda _:dict(pid=0,cgroup_pids=[]))
    assert suffix.verified_prefix(spec,conditions) == rows
    assert suffix.budgets(conditions[len(rows):]) == (2,71406,240)
    config = pinned(tmp_path/'initial-config.json',dict(stage=plan.JOINT,source_sha='first',
        source_root=spec['source_root'],population=population_card,E0_interface=spec['interface_predecessor']))
    monkeypatch.setattr(plan,'conditions',lambda *_:conditions)
    monkeypatch.setattr(d416_core_native_launch,'unit_state',lambda _:dict(pid=0,cgroup_pids=[]))
    next_root = tmp_path/'next'
    prepared = d416_j_screen_suffix.prepare(config,prev,next_root,new_authority)
    assert prepared['pending_science_requests'] == 2
    assert (prepared['first'],prepared['last']) == ('condition1-J0','condition1-J1')
    assert prepared['config']['previous_panel_segments'] == [dict(spec=prev,result=actual)]
    assert prepared['config']['authority'] == new_authority and not next_root.exists()
    with pytest.raises(PermissionError,match='NEW_AUTHORIZED_FINITE_WINDOW'):
        d416_j_screen_suffix.prepare(config,prev,next_root,old_authority)
    monkeypatch.setattr(d416_core_native_launch,'unit_state',lambda _:dict(pid=42,cgroup_pids=[42]))
    with pytest.raises(PermissionError,match='COLD_TERMINAL'):
        d416_j_screen_suffix.prepare(config,prev,next_root,new_authority)
    monkeypatch.setattr(suffix,'unit_state',lambda _:dict(pid=42,cgroup_pids=[42]))
    with pytest.raises(ValueError,match='ONLY_CLOSED_RECONCILED'):
        suffix.verified_prefix(spec,conditions)
    monkeypatch.setattr(suffix,'unit_state',lambda _:dict(pid=0,cgroup_pids=[]))
    part = pinned(Path(actual['path']),dict(result,records=rows[:1],attempted_case_launches=1,new_verified_native_requests=1))
    spec['previous_panel_segments'] = [dict(spec=prev,result=part)]
    with pytest.raises(ValueError,match='EXACT_MATCHED_CONDITION_PREFIX'):
        suffix.verified_prefix(spec,conditions)
