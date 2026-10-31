"""Exercise fixture, manifest, payload and export identity at the actual seams."""
from copy import deepcopy
import hashlib
import numpy as np
import pytest

from scripts.pi0294.multitask import source_stage_bundle as bundle
from scripts.pi0294.multitask.cabinet_braked_turn import cards, VERSION as PORT08
from scripts.pi0294.multitask.cabinet_seed_cards import seed_card, OFFSETS_M
from scripts.pi0294.multitask.cabinet_repair02 import checked_card, attempt_operation
from scripts.pi0294.multitask.libero_book_cards import cards as books
from scripts.pi0294.multitask.source_export import source_identity


def manifest(index=1):
    return dict(task_kind='ROBOCASA_CABINET',condition_side='right',cabinet_port_revision=PORT08,
                cabinet_seed_index=index,predeclared_tasks=[dict(task_kind='ROBOCASA_CABINET',side='right')],
                attempts_this_chain_max=1)


def payload(index):
    c=cards();c['right']=seed_card(index)
    return dict(task_kind='ROBOCASA_CABINET',condition_side='right',cabinet_port_revision=PORT08,
                cabinet_seed_index=index,cards=dict(BOOK=books(),ROBOCASA_CABINET=c),source_sha='source',
                qualification=dict(cabinet_port_revision=PORT08,condition_side='right',cabinet_seed_index=index,
                    fixed_task_cards_sha256=bundle.CABINET_SEED_CARD_SHA[index],source_sha='source',
                    independent_raw_auditors=['scripts/pi0294/multitask/night_book_audit.py',
                                              'scripts/pi0294/multitask/night_cabinet_audit.py']),
                definition_files={},libero_binary_files={},robocasa_definition_files={},robocasa_texture_files={})


def test_whole_rigid_fixture_translation_preserves_home_mechanics_limits_score_and_base_identity():
    original=cards();base=deepcopy(original['right'])
    assert hashlib.sha256(bundle.raw(dict(BOOK=books(),ROBOCASA_CABINET=original))).hexdigest()==bundle.PORT08_CARD_SHA
    for index,offset in enumerate(OFFSETS_M,1):
        card=seed_card(index);assert checked_card(card,'right')
        changed=deepcopy(card)
        for frame,old in [(changed['cabinet']['cabinet_world'],base['cabinet']['cabinet_world']),
                          (changed['door']['hinge_world'],base['door']['hinge_world'])]:
            np.testing.assert_allclose(np.array(frame)[:3,3]-np.array(old)[:3,3],[*offset,0],atol=1e-16)
            for axis in range(2):frame[axis][3]=old[axis][3]
        changed['source_base_card']['goal_xy']=base['source_base_card']['goal_xy'][:]
        for key in ('cabinet_seed_index','source_seed_set','fixture_translation_world_m'):changed.pop(key)
        changed['id']=base['id'];assert changed==base
        assert attempt_operation(card,'right')==f'ROBOCASA_RIGHT_PORT08_BRAKED_TURN_SEED{index:03}'
        c=cards();c['right']=card
        assert hashlib.sha256(bundle.raw(dict(BOOK=books(),ROBOCASA_CABINET=c))).hexdigest()==bundle.CABINET_SEED_CARD_SHA[index]
        assert source_identity(card,dict(schema='NIGHT_SOURCE_CABINET_INDEPENDENT_RAW_AUDIT_V1',
            native_ROBOCASA_goal_final=True))['episode_id']==f'CUSTOM_CABINET_PORT08_SOURCE_SEED{index:03}'
    assert cards()==original and attempt_operation(base,'right')=='ROBOCASA_RIGHT_PORT08_BRAKED_TURN'


@pytest.mark.parametrize('index',[True,False,0,5,1.,'1',None])
def test_undeclared_seed_rejected_before_native_work(index):
    with pytest.raises(ValueError):seed_card(index)
    with pytest.raises(PermissionError):checked_card(dict(seed_card(1),cabinet_seed_index=index),'right')
    if index is not None:
        with pytest.raises(PermissionError):bundle.checked_manifest(manifest(index))


@pytest.mark.parametrize('fault',['hinge','fixture','grasp','speed','goal','time','side'])
def test_no_unbound_card_changes_enter_teacher_or_export(fault):
    card=seed_card(1);side='right'
    if fault=='hinge':card['door']['hinge_world'][0][3]+=.001
    elif fault=='fixture':card['cabinet']['cabinet_world'][1][3]+=.001
    elif fault=='grasp':card['door']['object_to_tcp_grasp'][0][3]+=.001
    elif fault=='speed':card['turn_reference_limits']['max_linear_m_s']*=2
    elif fault=='goal':card['independent_demo_conditions']['withdrawal_m']/=2
    elif fault=='time':card['task_duration_s']=31.
    else:side='left'
    with pytest.raises(PermissionError):checked_card(card,side)
    if fault!='side':
        with pytest.raises(PermissionError):source_identity(card,dict(schema='NIGHT_SOURCE_CABINET_INDEPENDENT_RAW_AUDIT_V1',native_ROBOCASA_goal_final=True))


def test_payload_and_qualification_seed_cannot_be_dropped_rebound_or_replaced_by_book(monkeypatch):
    # Asset decoder isolation exercises the real identity seam, not native asset qualification.
    monkeypatch.setattr(bundle,'_definitions',lambda encoded,expected:{})
    monkeypatch.setattr(bundle,'_binary',lambda encoded,expected:{})
    for index in range(1,5):
        assert bundle.checked_manifest(manifest(index))[1]==manifest(index)['predeclared_tasks']
        good=payload(index);assert bundle.unpack(good)['plan']==manifest(index)['predeclared_tasks']
        for fault in ('drop_seed','qualification','wrong_seed','book','wrong_side','changed_card'):
            bad=deepcopy(good)
            if fault=='drop_seed':bad.pop('cabinet_seed_index')
            elif fault=='qualification':bad['qualification'].pop('cabinet_seed_index')
            elif fault=='wrong_seed':bad['cabinet_seed_index']=index%4+1
            elif fault=='book':bad['task_kind']='BOOK'
            elif fault=='wrong_side':bad['condition_side']='left'
            else:
                bad['cards']['ROBOCASA_CABINET']['right']['turn_reference_limits']['max_linear_m_s']*=2
                bad['qualification']['fixed_task_cards_sha256']=hashlib.sha256(bundle.raw(bad['cards'])).hexdigest()
            with pytest.raises(PermissionError):bundle.unpack(bad)
