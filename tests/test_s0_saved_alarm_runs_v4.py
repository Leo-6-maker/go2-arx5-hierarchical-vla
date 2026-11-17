import json
from scripts.wm02.s0_saved_alarm_runs_v4 import alarm_runs,analyze


def test_disjoint_warning_cannot_be_reported_as_continuous_lead():
    states=[dict(step=k,time_s=k*.002,candidate_alarm=k in (10,11,12,28,29,30))for k in range(10,31)]
    runs=alarm_runs(states)
    assert len(runs)==2 and runs[0]['samples']==3 and runs[1]['samples']==3
    assert abs(.058-runs[0]['first_time_s']-.038)<1e-12
    assert abs(.058-runs[-1]['first_time_s']-.002)<1e-12


def test_missing_ticks_are_not_silently_joined_as_continuous_alarm():
    runs=alarm_runs([dict(step=2,time_s=.004,candidate_alarm=True),dict(step=4,time_s=.008,candidate_alarm=True)])
    assert len(runs)==2


def test_export_keeps_isolated_families_without_private_site_metadata():
    states=[dict(step=i,time_s=i*.002,candidate_alarm=False,pads=[dict(pad='task_right_pad',clearance_m=.01)])for i in (1,2)]
    a=dict(window=dict(id='fixture',original_qualified=True),states=states,original_first_illegal_contact=None,
        geometry_sha256='a'*64,original_result=dict(sha256='b'*64),journal=dict(sha256='c'*64),
        held_reference_profile=dict(first_alarm=None,lead_to_first_recorded_contact_s=None))
    bundle=dict(safety=[dict(result_sha256='d'*64,artifacts=[dict(sha256='e'*64,value=a)]),dict(result_sha256='f'*64)],
        quarantine=dict(F1=dict(site_path='/private/site/F1'),F2=dict(site_path='/private/site/F2')))
    result=analyze(bundle)
    assert result['quarantine']==['F1','F2']
    assert '/private/site'not in json.dumps(result)
