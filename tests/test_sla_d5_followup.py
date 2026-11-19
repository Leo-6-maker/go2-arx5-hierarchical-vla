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
