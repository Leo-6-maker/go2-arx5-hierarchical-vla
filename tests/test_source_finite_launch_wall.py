import pytest
from scripts.pi0294.multitask.night_source_stage import LAUNCH


@pytest.mark.parametrize('finite,conditions,expected',[(True,1,650),(False,1,1250),(False,4,1250)])
def test_actual_launcher_finite_clock_matches_one600s_child_plus_audit(finite,conditions,expected):
    code="finite="+LAUNCH.split("finite=",1)[1].split("assert lease['deadline_epoch']",1)[0]
    ns=dict(m={'sealed_inputs':{'SOURCE_V7_FINITE_JOB.json':'p'}if finite else{}},planned=[{}]*conditions)
    exec(compile(code,'actual_launcher_wall_profile','exec'),ns)
    assert ns['runtime_upper']==expected
    assert ns['runtime_upper']+50<900 if finite else ns['runtime_upper']+50==1300
    assert "'--property=RuntimeMaxSec='+str(runtime_upper)" in LAUNCH


def test_finite_job_cannot_accidentally_start_a_multiple_condition_chain():
    code="finite="+LAUNCH.split("finite=",1)[1].split("assert lease['deadline_epoch']",1)[0]
    with pytest.raises(AssertionError):
        exec(compile(code,'actual_launcher_multi_reject','exec'),dict(m={'sealed_inputs':{'SOURCE_V7_FINITE_JOB.json':'p'}},planned=[{},{}]))
