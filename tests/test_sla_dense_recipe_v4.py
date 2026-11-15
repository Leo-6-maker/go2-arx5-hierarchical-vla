import hashlib
import numpy as np
import pytest

from go2arx5_vla.sla_adapter.config_v1 import learning_rates as old_rates, lr_sha256
from go2arx5_vla.sla_adapter.dense_recipe_v4 import learning_rates, visit_map, microbatch_contract
from go2arx5_vla.sla_adapter.runtime_v1 import Engine


def test_legacy_schedule_binding_unchanged_and_dense_endpoint_uses_3k():
    legacy = Engine.__new__(Engine).learning_rate_values_for_spec({})
    assert legacy == old_rates() and len(legacy) == 500
    assert hashlib.sha256(''.join(v.hex()+'\n' for v in legacy).encode('ascii')).hexdigest() == lr_sha256()
    dense = learning_rates()
    assert len(dense) == 3000 and dense[0] == 5e-5/150
    assert dense[149] == 5e-5 and dense[-1] == 1e-5
    assert all(a <= b for a, b in zip(dense[:149], dense[1:150]))
    assert all(a >= b for a, b in zip(dense[149:-1], dense[150:]))
    assert dense[499] != legacy[-1]


def test_phase_balanced_pilot_bank_is_fixed_and_uses_both_independent_resets():
    class Dataset:
        manifest_sha = 'a'*64
        def catalog_rows(self):
            return [dict(split='TRAIN', task='PICK_HOLD', episode=ep, phase=phase, anchor=i)
                    for ep in ('r1', 'r2') for i, phase in enumerate(
                        ['PREGRASP']*25+['DESCEND']*15+['CLOSE']*10+['LIFT']*5+['HOLD']*6)]
    rows = Dataset().catalog_rows()
    visits, binding = visit_map(Dataset())
    assert visits.shape == (96000,) and visits.dtype == np.dtype('<i8')
    assert set(visits) == set(range(122))
    phases = {phase: sum(rows[i]['phase'] == phase for i in visits) for phase in set(r['phase'] for r in rows)}
    # Sample phases uniformly rather than letting the 25-anchor phase dominate.
    assert all(18000 < n < 20500 for n in phases.values())
    assert 46000 < sum(rows[i]['episode'] == 'r1' for i in visits) < 50000
    assert not binding['student_phase_input'] and not binding['formal_D300_comparison_arm']


def test_core_qualification_cannot_change_the_h200_candidate_microbatch():
    microbatch_contract(dict(operation='DENSE_CORE_MICRO1_QUAL4', microbatch=1, training_matmul_precision='highest'))
    microbatch_contract(dict(operation='DENSE_PICK_HOLD_3K', microbatch=8, training_matmul_precision='highest'))
    with pytest.raises(ValueError, match='MAIN_RECIPE_REMAINS_MICRO8'):
        microbatch_contract(dict(operation='DENSE_PICK_HOLD_3K', microbatch=1, training_matmul_precision='highest'))
