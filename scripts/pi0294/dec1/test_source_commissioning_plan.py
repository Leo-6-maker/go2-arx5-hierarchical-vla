"""Synthetic plan binding tests; not native admission or geometry qualification."""
from copy import deepcopy
import hashlib,json
import pytest
from scripts.pi0294.scene_cards import scene_group,task_card,digest
from scripts.pi0294.dec1 import source_commissioning_plan as unit


@pytest.fixture
def population(monkeypatch):
    groups=[];receipts=[]
    for family,group_id in zip(('F0','F1'),unit.GROUPS):
        group=scene_group('TRAIN',family,0);group.update(group_id=group_id,scene_group=group_id)
        groups.append(group)
        for task in ('T1','T2'):
            receipts.append(dict(scene_group=group_id,task=task,status='STATIC_PASS_PENDING_LIVE_CONTACT',
                geometry_passed=True,field_of_view_passed=True,card_sha256=digest(task_card(group,task)),
                receipt_sha256='a'*64))
    value=dict(schema='PI0294_PV1_DATA_PLAN_AND_SPLIT_SEAL_V1',
        status='SEALED_STATIC_QUALIFIED_BEFORE_TASK_RESULTS',evaluation_success_screening=False,
        groups=groups,static_receipts=receipts)
    def encode(v):
        raw=(json.dumps(v,sort_keys=True)+'\n').encode()
        monkeypatch.setattr(unit,'POPULATION',hashlib.sha256(raw).hexdigest())
        return raw
    return value,encode


def test_four_source_slots_balance_modes_tasks_and_preserve_geometry(population):
    value,encode=population;raw=encode(value);plan=unit.build(raw,owner_release='f'*40)
    assert len(plan['conditions'])==4 and plan['source_component_upper']+plan['target_component_reserved']==8
    assert {(c['task'],c['interface']) for c in plan['conditions']}=={
        ('T1','I_TIME'),('T1','I_PROGRESS'),('T2','I_TIME'),('T2','I_PROGRESS')}
    assert plan['GR00T_queries']==plan['core64_cells']==0
    assert not plan['student_neural_ACK_qualification'] and not plan['new_training_data']
    for condition in plan['conditions']:
        original=task_card(next(g for g in value['groups'] if g['group_id']==condition['scene_group']),condition['task'])
        card=deepcopy(condition['card']);card['id']=original['id']
        assert card==original and condition['original_TRAIN_card_sha256']==digest(original)
    assert unit.validate(plan,raw,owner_release='f'*40)==plan


@pytest.mark.parametrize('change',['bytes','split','duplicate','static-missing','static-card','screening','release'])
def test_missing_or_changed_original_bindings_are_rejected(population,change):
    value,encode=population;raw=encode(value)
    if change=='bytes':raw+=b' '
    elif change=='release':pass
    else:
        if change=='split':value['groups'][0]['split']='DEV'
        if change=='duplicate':value['groups'].append(deepcopy(value['groups'][0]))
        if change=='static-missing':value['static_receipts'].pop()
        if change=='static-card':value['static_receipts'][0]['card_sha256']='b'*64
        if change=='screening':value['evaluation_success_screening']=True
        raw=encode(value)
    with pytest.raises(PermissionError):unit.build(raw,owner_release='invalid' if change=='release' else 'f'*40)


@pytest.mark.parametrize('change',['mode','geometry','student','upper','card','target-slots','bool-counter'])
def test_plan_cannot_expand_scope_or_become_student_evidence(population,change):
    value,encode=population;raw=encode(value);plan=unit.build(raw,owner_release='f'*40)
    if change=='mode':plan['conditions'][0]['interface']='I_PROGRESS'
    if change=='geometry':plan['conditions'][0]['scene_group']='unsealed'
    if change=='student':plan['conditions'][0]['policy']='CORR2048'
    if change=='upper':plan['source_component_upper']=8
    if change=='card':plan['conditions'][0]['card']['goal_xy'][0]+=.001
    if change=='target-slots':plan['target_component_reserved']=0
    if change=='bool-counter':plan['GR00T_queries']=False
    with pytest.raises(PermissionError):unit.validate(plan,raw,owner_release='f'*40)
