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


def test_new_independent_expert_can_finish_slow_compile_without_a_clock_grant(tmp_path):
    a=fixture(tmp_path)
    a.spec=dict(phase='collect-target-task-learning-grasp-v1',
        task_learning_role='CORELAB_EXPERT_TRAIN4_SEMANTICS_NOT_BC_V3',
        corelab_world_binding_v3={'sha256':'existing'},wall_limit_s=3600,native_startup_read_wait_s=600)
    spin=a.spin
    a.spin=lambda seconds,predicate:spin(seconds,predicate)if seconds>=200 else False
    native.await_native_startup(a)
    assert a.calls==[600] and a.clock_grants==0
    a.spec['task_learning_role']='SLA_TARGET500_NATIVE_EVALUATION_V1'
    with pytest.raises(PermissionError,match='INDEPENDENT_EXPERT'):
        native.await_native_startup(a)
