import pytest
from scripts.wm02.sla_dev48_cold_successor_v61 import continuation_reason


def test_only_known_finite_window_exit_allows_an_unstarted_suffix():
    assert continuation_reason(dict(status='FOLLOWUP_STOPPED_RECONCILE_NO_REPLAY',error='TASK_LEARNING_GRASP_N20_FOLLOWUP_NO_LATE_NEW_GROUP'))=='FINITE_WINDOW_ONLY'
    assert continuation_reason(dict(status='COMPLETE_FIXED_TRAIN8_AND_NEW_DEV48_NOT_GUARANTEED_SUCCESS'))=='ALREADY_COMPLETE'
    for result in [dict(status='ACTUAL_TECHNICAL_OR_SAFETY_STOP_RETAINED'),
                   dict(status='FOLLOWUP_STOPPED_RECONCILE_NO_REPLAY',error='UNKNOWN_DISPATCH'),
                   dict(status='FOLLOWUP_STOPPED_RECONCILE_NO_REPLAY',error='N20_GROUP_TERMINAL_UNKNOWN_NO_RESTART')]:
        with pytest.raises(ValueError,match='RECONCILIATION'):continuation_reason(result)


def test_only_a_proven_undispatched_lock_refusal_can_be_waited(monkeypatch):
    from scripts.wm02 import sla_dev48_cold_successor_v61 as m
    def busy(_):raise BlockingIOError('owner busy')
    registered=dict(input=dict(path='fixture'))
    monkeypatch.setattr(m,'never_dispatched',lambda *_:True)
    with pytest.raises(BlockingIOError):m.capacity_dispatch(registered,'group','journal',busy)
    monkeypatch.setattr(m,'never_dispatched',lambda *_:False)
    with pytest.raises(RuntimeError,match='NEVER_REPLAY'):m.capacity_dispatch(registered,'group','journal',busy)
