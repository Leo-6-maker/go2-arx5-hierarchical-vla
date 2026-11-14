import numpy as np

from scripts.sla.audit_query_feasibility_v7 import geometry_detail, joint_rates


def test_joint_history_is_distinguished_from_the_first_h40_triple():
    q = np.zeros((42, 6))
    q[2:, 2] = .003
    seam = joint_rates(q, 2)['first_acceleration_exceedance']
    assert seam['joint'] == 3 and seam['location'] == 'REFERENCE_JOIN'
    assert seam['H40_knot_triple'] == [-2, -1, 0]
    assert np.isclose(seam['absolute_ratio'], 1.2)
    q[:] = 0
    q[10, 3] = .003
    interior = joint_rates(q, 2)['first_acceleration_exceedance']
    assert interior['joint'] == 4 and interior['location'] == 'H40_INTERIOR'
    assert interior['H40_knot_triple'] == [6, 7, 8]


def test_unconsumed_suffix_is_still_checked_at_original_limits():
    rows = np.tile([0., 0., .7, 1., 0., 0., 0., 1., 0., .5], (40, 1))
    actual = np.eye(4)
    actual[2, 3] = .7
    assert geometry_detail(rows, actual)['passes']
    rows[-1, 0] = .01
    result = geometry_detail(rows, actual)
    failures = {r['metric']: r for r in result['failed']}
    translation = failures['maximum_discrete_translation_acceleration_m_s2']
    assert translation['limit'] == .25
    assert translation['first_difference_index'] == 38
    assert np.isclose(translation['maximum'], 4.)
    assert np.isclose(failures['maximum_translation_speed_m_s']['maximum'], .2)
