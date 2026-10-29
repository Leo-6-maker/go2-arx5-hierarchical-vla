from copy import deepcopy

import pytest

from .task_path_budget import REQUESTED, add_to_original


def baseline():
    return dict(dimensions={k:dict(cap=1000000000000000, projected=100)
        for k in REQUESTED}, old_costs='KEEP', final_reserve='KEEP')


def test_existing_work_and_protection_are_added_before_S1_without_credit():
    old = baseline()
    before = deepcopy(old)
    result = add_to_original(old)
    assert old == before
    assert result['total_new_Core_and_PC_CPU_s'] == 14400
    assert result['requested']['physics_ticks'] == 62012
    for k, value in result['dimensions'].items():
        assert value['projected_with_S1'] == 100+REQUESTED[k]
    assert not result['launch_allowed'] and not result['resources_reserved']


def test_S1_allowance_cannot_cover_one_byte_of_old_work_shortfall():
    old = baseline()
    old['dimensions']['Core_storage_bytes']['cap'] = 99
    result = add_to_original(old)
    assert result['shortfalls'] == {'Core_storage_bytes': 1}
    assert result['status'] == 'RESOURCE_FORECAST_SHORTFALL'


def test_explicit_new_transfer_grant_remains_separate_from_exhausted_old_pool():
    old = baseline()
    old['dimensions']['transfer_bytes']['cap'] = 100
    result = add_to_original(old)
    row = result['dimensions']['transfer_bytes']
    assert row['cap'] == 100 and row['original_scope_shortfall'] == 0
    assert row['separately_approved_S1_allowance'] == 8*1024**3
    assert row['projected_with_S1'] == row['combined_named_scope_ceiling']
    assert result['shortfalls'] == {}
    assert result['S1_allowance_spendable_by_old_work'] is False


def test_only_the_previously_adopted_PBS_count_can_have_no_numeric_cap():
    old = baseline()
    old['dimensions']['PBS_submissions'].update(cap=None, count_still_accounted=True)
    result = add_to_original(old)
    assert result['dimensions']['PBS_submissions']['projected_with_S1'] == 100
    old['dimensions']['physics_ticks']['cap'] = None
    with pytest.raises(PermissionError, match='ONLY_ALREADY_ADOPTED_PBS'):
        add_to_original(old)


def test_incomplete_resource_vector_never_defaults_to_zero_usage():
    old = baseline()
    del old['dimensions']['local_CPU_s']
    with pytest.raises(PermissionError, match='ALL_ORIGINAL_FOURTEEN'):
        add_to_original(old)
