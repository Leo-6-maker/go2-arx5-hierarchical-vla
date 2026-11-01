from copy import deepcopy
import pytest
from scripts.pi0294.multitask.phase_reconciliation import event_value, summarize_case


def state(time=5., grasp=False):
    return dict(time_s=time, native_grasp=grasp, object_pose=[[0]*4, [0]*4, [0,0,1.,0]],
        object_supported_by_shelf=False, native_libero_goal=False)


def fixture(success=False, safe=True, goal=False):
    c=dict(id='TRAIN_NOT_ADMITTED_DEV', success=success, valid_model_condition=True,
        hard_native_safety=not safe, gate_rejected=False)
    a=dict(task_success=success, complete_episode=safe, original_native_safety_pass=safe,
        native_LIBERO_goal_final=goal, additional_support_release_withdraw_hold_pass=success,
        integrity_problems=[], completed_intervals=1, terminal_error=None if safe else 'RuntimeError: FINGER_VELOCITY')
    rows=[dict(kind='observation',value=dict(index=0,actual_state=state())),
        dict(kind='submitted_interval',value={}),dict(kind='native_tick',value=dict(actual_state=state(6.,True))),
        dict(kind='completed_interval',value={})]
    if not safe:rows.append(dict(kind='terminal_failure',error=a['terminal_error'],actual_step=500,
        actual_state=state(7.)))
    return rows,c,a


def test_original_direct_failure_and_known_wrapped_failure_produce_same_causal_score():
    rows,c,a=fixture(safe=False);original=summarize_case(rows,c,a)
    direct=rows[-1];rows[-1]=dict(kind='terminal_failure',value={k:v for k,v in direct.items()if k!='kind'})
    assert summarize_case(rows,c,a)==original
    assert original['classification']=='NATIVE_SAFETY_STOP'
    assert original['first_native_failure']['task_s']==2.


def test_four_real_outcome_classes_keep_one_of_four_and_six_unrun():
    configurations=[{},dict(goal=True),dict(success=True,goal=True),dict(safe=False)]
    cases=[summarize_case(*fixture(**v))for v in configurations]
    assert sum(c['success']for c in cases)==1 and sum(c['valid']for c in cases)==4
    assert 10-len(cases)==6
    assert [c['classification']for c in cases]==['FULL_DURATION_TASK_GOAL_NOT_REACHED',
        'FINAL_GEOMETRY_TRUE_BUT_SUPPORT_RELEASE_WITHDRAW_HOLD_FAILED',
        'FULL_STUDENT_TASK_SUCCESS','NATIVE_SAFETY_STOP']


@pytest.mark.parametrize('record',[
    {},dict(kind='unknown',value={}),dict(kind='observation'),
    dict(kind='terminal_failure',error='x'),dict(kind='native_tick',value=[]),
    dict(kind='native_tick',value=dict(actual_state=dict(time_s=float('nan')))),
    dict(kind='terminal_failure',value=dict(error='',actual_state=state()))])
def test_missing_corrupt_and_unknown_event_shapes_fail_closed(record):
    with pytest.raises(ValueError):event_value(record)


@pytest.mark.parametrize('key', ['complete_episode','original_native_safety_pass',
    'native_LIBERO_goal_final','additional_support_release_withdraw_hold_pass'])
def test_success_cannot_be_inferred_from_partial_goal(key):
    rows,c,a=fixture(success=True,goal=True);a[key]=False
    with pytest.raises(ValueError):summarize_case(rows,c,a)


def test_wrong_boolean_and_missing_executed_interval_rejected():
    rows,c,a=fixture();a['task_success']=0
    with pytest.raises(ValueError):summarize_case(rows,c,a)
    rows,c,a=fixture();rows.pop()
    with pytest.raises(ValueError):summarize_case(rows,c,a)
