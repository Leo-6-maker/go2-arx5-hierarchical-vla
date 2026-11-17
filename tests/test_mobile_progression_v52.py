import copy

import pytest

from go2arx5_vla.sla_adapter.mobile_selection_v52 import OFFLINE_GATES
from go2arx5_vla.sla_adapter.mobile_progression_v52 import (
    authorize_next, confirmation_decision, generalization_decision, screen_decision,
)


def panel(stage='CONFIRM'):
    result = []
    for family in ('F1', 'F2'):
        for layout in range(4 if stage == 'CONFIRM' else 8 if stage == 'DEV' else 10):
            for noise in range(1 if stage == 'CONFIRM' else 2):
                label = 'T%d' % layout if stage == 'CONFIRM' else '%s-%s-%02d' % (stage, family, layout)
                key = '%s-%s-n%d' % (family, label, noise)
                result.append(dict(id=key, episode_id=key, split=stage, family=family,
                    layout=label, layout_id=label, reset=len(result) + 100,
                    noise_seed=len(result) + 200, scene=dict(sha256=label)))
    return result


def rows(entries, arm='B2', failures=()):
    return [dict(role='STUDENT', arm=arm, panel_entry_id=e['id'], family=e['family'],
        condition=dict(family=e['family'], layout=e['layout'], reset=e['reset'], seed=e['noise_seed']),
        scene_sha256=e['scene']['sha256'], actual_native_requests=1,
        technical_failure=False, hard_STOP=False, completed=True,
        task_and_mobile_success=i not in failures, task_ticks=12000,
        native_ticks=17703, valid_policy_rejection=None)
        for i, e in enumerate(entries)]


def screen():
    native = sum((rows(panel(), a) for a in ('B0', 'B1', 'B2')), [])
    offline = {a: {k: True for k in OFFLINE_GATES} for a in ('B0', 'B1', 'B2')}
    offline['B0']['saved_reject3_all'] = offline['B1']['saved_reject3_all'] = False
    return native, offline


def test_no_selection_from_one_completed_arm_or_an_incomplete_control_grid():
    native, offline = screen()
    with pytest.raises(ValueError, match='COMPLETE24'):
        screen_decision(native[8:], offline)
    native[0]['condition']['layout'] = 'T1'
    with pytest.raises(ValueError, match='CONTROL_GRIDS'):
        screen_decision(native, offline)


def test_a_failed_control_remains_without_vetoing_a_different_qualified_arm():
    native, offline = screen()
    native[0].update(technical_failure=True, completed=False)
    decision = screen_decision(native, offline)
    assert decision['unique_confirmation_candidate'] == 'B2'
    assert not decision['by_arm']['B0']['identity_complete']
    assert decision['by_arm']['B0']['attempts'] == 8
    assert authorize_next('CONFIRM', 'B2', screen=decision)['requests'] == 8
    with pytest.raises(PermissionError):
        authorize_next('DEV', 'B2', screen=decision)
    with pytest.raises(PermissionError):
        authorize_next('CONFIRM', 'B1', screen=decision)


def test_confirmation_original_family_and_stop_gates_are_retained():
    entries = panel()
    native = rows(entries, failures=(3, 7))
    passed = confirmation_decision(native, entries, 'B2')
    assert passed['passed'] and passed['by_family'] == {'F1': 3, 'F2': 3}
    assert passed['old_V51_pilot_GO'] is False
    native[0]['hard_STOP'] = True
    assert not confirmation_decision(native, entries, 'B2')['passed']
    native = rows(entries, failures=(0, 1))
    assert not confirmation_decision(native, entries, 'B2')['passed']


def test_confirmation_rejects_missing_native_attempt_or_seed_and_model_switch():
    entries = panel()
    for mutate in (
        lambda r: r.update(actual_native_requests=0),
        lambda r: r.update(actual_native_requests=True),
        lambda r: r['condition'].update(seed=99999),
        lambda r: r.update(arm='B1'),
    ):
        native = rows(entries)
        mutate(native[0])
        with pytest.raises(ValueError):
            confirmation_decision(native, entries, 'B2')


def test_dev_test_use_actual_sealed_layouts_and_do_not_open_test_on_failed_dev():
    split = dict(frozen_before_training=True, training_updates_before_freeze=0,
        episodes=panel('DEV') + panel('TEST'))
    dev = rows(panel('DEV'), failures=(0, 1, 2, 3, 16, 17, 18, 19))
    decision = generalization_decision(dev, split, 'DEV', 'B2')
    assert decision['passed'] and decision['successes'] == 24
    assert decision['by_family'] == {'F1': 12, 'F2': 12}
    assert len(decision['by_layout']) == 16
    native, offline = screen()
    selected = screen_decision(native, offline)
    confirmed = confirmation_decision(rows(panel()), panel(), 'B2')
    assert authorize_next('TEST', 'B2', screen=selected,
        confirmation=confirmed, development=decision)['requests'] == 40
    dev[4]['technical_failure'] = True
    failed = generalization_decision(dev, split, 'DEV', 'B2')
    assert not failed['passed'] and not failed['TEST_permission']
    with pytest.raises(PermissionError, match='PASSED_DEV'):
        authorize_next('TEST', 'B2', screen=selected,
            confirmation=confirmed, development=failed)
    wrong = copy.deepcopy(split)
    wrong['episodes'][0]['layout'] = 'TRAIN-T0'
    with pytest.raises(ValueError, match='EPISODE_IDENTITY'):
        generalization_decision(rows(panel('DEV')), wrong, 'DEV', 'B2')
