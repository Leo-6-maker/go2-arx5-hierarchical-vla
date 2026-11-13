"""Cold construction can finish; missing readiness never grants a clock."""
from types import SimpleNamespace
import pytest
from go2arx5_vla.world_model import target_sequential as native


def fixture(tmp_path,*,feedback=True,missing=None):
    markers={'controller':'go2_x5_ocs2 ready; waiting for\nMRT node is ready.',
             'solver':'MPC node is ready.','planner':'Trajectory JSON input:'}
    counts=dict(odom=11,arm=11)if feedback else dict(odom=11,arm=0)
    a=SimpleNamespace(output=tmp_path,counts=counts,reference_phase=lambda:True,calls=[],clock_grants=0)
    def spin(seconds,predicate):
        a.calls.append(seconds)
        if seconds>=45:
            for name,text in markers.items():
                if name!=missing:(tmp_path/(name+'.log')).write_text(text)
        return bool(predicate())
    a.spin=spin
    return a


def test_cold_compile_finishing_after30s_retains_all_readiness_requirements(tmp_path):
    a=fixture(tmp_path)
    assert not a.spin(30,lambda:native.native_startup_readiness(a.output,True))
    native.await_native_startup(a)
    assert a.calls==[30,180]and a.clock_grants==0


@pytest.mark.parametrize('missing',['controller','solver','planner','feedback'])
def test_cold_wait_never_substitutes_for_missing_markers_or_feedback(tmp_path,missing):
    a=fixture(tmp_path,feedback=missing!='feedback',missing=missing)
    with pytest.raises(RuntimeError,match='FEEDBACK_UNAVAILABLE'):
        native.await_native_startup(a)
    assert a.clock_grants==0 and a.calls==[180]
