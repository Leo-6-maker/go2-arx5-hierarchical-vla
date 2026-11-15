import copy
import pytest
from scripts.wm02.sla_pd4_student_queue_v71 import require_training_summary


def terminal():
    result=dict(exit_code=0,state='BOTH_FIXED500_AND_FRESH_RGB_COMPLETE_OFFLINE_ONLY',
        completed_arms=[{},{}],partial_arms=[],actual_optimizer_updates=1000,actual_training_visits=32000)
    matching=dict(initial87_bit_exact=True,committed_updates_per_arm=500,
        actual_visit_sequences_equal=True,state_dropout_sequences_equal=True,physical_requests=0)
    return result,matching


def test_zero_physics_is_valid_for_completed_offline_training():
    require_training_summary(*terminal())


@pytest.mark.parametrize('field,value',[
    ('initial87_bit_exact',False),('committed_updates_per_arm',499),
    ('actual_visit_sequences_equal',False),('state_dropout_sequences_equal',False)])
def test_evaluation_never_starts_with_an_unmatched_training_endpoint(field,value):
    result,matching=terminal();matching[field]=value
    with pytest.raises(PermissionError,match='BOTH_REAL500'):require_training_summary(result,matching)


def test_partial_training_is_not_misreported_as_two_complete_arms():
    result,matching=terminal();result['partial_arms']=[dict(step=499,visits=15968)]
    result['actual_optimizer_updates']=999
    with pytest.raises(PermissionError,match='BOTH_REAL500'):require_training_summary(result,matching)
