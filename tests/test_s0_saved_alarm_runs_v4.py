from scripts.wm02.s0_saved_alarm_runs_v4 import alarm_runs


def test_disjoint_warning_cannot_be_reported_as_continuous_lead():
    states=[dict(step=k,time_s=k*.002,candidate_alarm=k in (10,11,12,28,29,30))for k in range(10,31)]
    runs=alarm_runs(states)
    assert len(runs)==2 and runs[0]['samples']==3 and runs[1]['samples']==3
    assert abs(.058-runs[0]['first_time_s']-.038)<1e-12
    assert abs(.058-runs[-1]['first_time_s']-.002)<1e-12


def test_missing_ticks_are_not_silently_joined_as_continuous_alarm():
    runs=alarm_runs([dict(step=2,time_s=.004,candidate_alarm=True),dict(step=4,time_s=.008,candidate_alarm=True)])
    assert len(runs)==2
