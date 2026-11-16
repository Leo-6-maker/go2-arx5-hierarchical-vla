import copy
import numpy as np
import pytest
from go2arx5_vla.sla_adapter import gripper_time_v1 as old
from go2arx5_vla.sla_adapter.gripper_time_shift7_v1 import SHIFT, SLOTS, VERSION, relocate, decode
from go2arx5_vla.sla_adapter.time_shift7_training_v1 import physical, RECIPE
from go2arx5_vla.sla_adapter.time_training_v1 import physical as old_physical
from go2arx5_vla.sla_adapter.codec_v1 import Quantiles
from go2arx5_vla.sla_adapter.provider_time_shift7_v1 import contract, SCHEMA
from go2arx5_vla.sla_adapter.provider_time_v1 import contract as old_contract, ORDER
from go2arx5_vla.sla_adapter.gripper_latent_v2 import VERSION as STATS_CODEC


def test_slot_intervention_preserves_counts_scalars_and_full_physical_decode():
    assert [len(g) for g in SLOTS] == [14, 13, 13]
    assert sorted(int(i) for g in SLOTS for i in g) == list(range(40))
    assert 27 + SHIFT == 34
    rng = np.random.default_rng(1727)
    state = np.array([0., 0., 0., 1., 0., 0., 0., 1., 0., .8])
    stats = {'actions': Quantiles([-1.]*6+[-9.21024036697585], [1.]*6+[9.21024036697596])}
    for _ in range(20):
        context = dict(width_m=float(rng.uniform(.002, .042)), source_native_s=3., query_native_s=3.05)
        original = rng.normal(0, .3, (1, 40, 32)); shifted = original.copy()
        shifted[0, :, 6] = relocate(original[0, :, 6])
        a, _ = old_physical(original, state, np.eye(4), stats, context)
        b, _ = physical(shifted, state, np.eye(4), stats, context)
        assert np.array_equal(a, b)
        assert np.array_equal(np.roll(shifted[0, :, 6], -SHIFT), original[0, :, 6])


def test_distinct_package_rejects_wrong_or_legacy_slot_map():
    m = dict(schema=SCHEMA, codec=VERSION, decode_order=ORDER, parameter_slot_shift=7,
        parameter_slots=[[int(i) for i in g] for g in SLOTS], step=500, cursor=16000,
        bindings=dict(parent_step=1000, parent_visits=32000, new_planned_updates=500,
            recipe=RECIPE, codec_version=VERSION, parameter_slot_shift=7,
            statistics_codec_version=STATS_CODEC, gripper_lambda=0),
        previous_reference_required=True, teacher_required=False, optimizer_required=False,
        training_cache_required=False, physical_execution_admitted=False)
    contract(m)
    with pytest.raises(ValueError): old_contract(m)
    wrong = copy.deepcopy(m); wrong['parameter_slots'] = [list(range(a,b)) for a,b in old.BLOCKS]
    with pytest.raises(ValueError): contract(wrong)
    with pytest.raises(ValueError): contract(dict(m, physical_execution_admitted=True))
