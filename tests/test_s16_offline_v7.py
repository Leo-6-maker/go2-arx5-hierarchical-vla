from collections import Counter
import json
from pathlib import Path
import pytest

from go2arx5_vla.sla_adapter.s16_offline_v7 import fixed_indices, prepare, visit_bank, STATS_SHA


def inputs():
    selection = json.loads((Path(__file__).parents[1] /
        'experiments/sla_go2_posttrain/s16_rp8_reference_v61.json').read_text())
    episodes = {}
    for row in selection['S16']:
        ep = episodes.setdefault(row['episode'], dict(id=row['episode'], split='TRAIN',
            layout=row['layout'], expert_stage_audit_only=['UNKNOWN'] * 61))
        ep['expert_stage_audit_only'][row['anchor']] = row['stage']
    return dict(episodes=list(episodes.values()), statistics={'sha256': STATS_SHA}), selection


def test_each_update_contains_all_fixed_windows_with_two_noise_slots():
    manifest, selection = inputs()
    indices = fixed_indices(manifest, selection)
    bank = visit_bank(indices)
    assert len(bank) == 16000
    assert set(Counter(bank).values()) == {1000}
    for start in range(0, len(bank), 32):
        assert Counter(bank[start:start+32]) == Counter({i: 2 for i in indices})
    plan = prepare(manifest, selection)
    assert plan['native_execution_admitted'] is False
    assert plan['training_registered'] is False
    assert plan['requires_RP8_for_offline_fitting'] is False


def test_statistics_and_fixed_selection_cannot_be_replaced_for_a_fit():
    manifest, selection = inputs()
    manifest['statistics']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='FROZEN_N20'):
        prepare(manifest, selection)
    manifest, selection = inputs()
    selection['S16'][-1] = selection['S16'][0]
    with pytest.raises(ValueError, match='DISTINCT'):
        prepare(manifest, selection)
