from scripts.wm02.close_goal_clearance_audit_v1 import fence,summarize_case


def test_alarm_at_contact_timestamp_is_not_a_prevention_claim():
    r=summarize_case([dict(step=99,candidate_alarm=True)],qualified=False,first_STOP_tick=100)
    assert r['lead_to_first_recorded_contact_s']==0
    assert not r['sampled_alarm_strictly_before_recorded_contact']
    assert not r['sampled_alarm_at_least10ms_before_contact']
    assert not r['monitor_control_latency_validated']


def test_fence_keeps_successful_episode_false_positive_in_denominator():
    assert not fence(.002,-.006,False)
    assert not fence(.002,-.001,True)
    assert fence(.002,-.006,True)
    r=summarize_case([dict(step=100,candidate_alarm=True)],qualified=True)
    assert r['successful_episode_would_be_blocked']
    assert r['original_outcome_unchanged']
