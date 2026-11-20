import pytest

from scripts.sla.goal_v2_hold4_after_students import require_unqualified_tgt


@pytest.mark.parametrize('state,attempts,failure,eligible', [
    ('TGT_RGB16_FAIL',1,'ValueError: SCALE20K_CORE_REPLAY_STRICT_OUTPUT_OR_GUARD_MISMATCH',True),
    ('TGT_RGB16_REQUIRES_RECONCILIATION',0,'PermissionError: TGT_RGB16_RESOURCE_REJECT:host',True),
    ('TGT_RGB16_FAIL',1,'Unresolved worker exit',False),
    ('TGT_RGB16_PASS',16,None,False),
])
def test_only_reconciled_unqualified_target_releases_lower_priority_collection(state, attempts, failure, eligible):
    report = dict(status='TGT5K_STRICT_RGB16_UNQUALIFIED',native_started=False,
                  rgb16_status=state,rgb16_attempts=attempts,rgb16_failure=failure)
    if eligible:
        require_unqualified_tgt(report)
    else:
        with pytest.raises(ValueError,match='TGT_PRIORITY'):
            require_unqualified_tgt(report)
    report['native_started'] = True
    with pytest.raises(ValueError,match='TGT_PRIORITY'):
        require_unqualified_tgt(report)
