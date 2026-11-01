from copy import deepcopy
import pytest
from scripts.pi0294.multitask.source_cold_custody import is_cold_terminal
from scripts.pi0294.multitask import night_source_stage


def fixture():return dict(status='FINITE_SOURCE_CHAIN_TERMINAL',physics_integrations=0,new_model_queries_updates_PBS=0,
    results=[dict(OS_exit_status='NOT_STARTED_READBACK_SIGKILL_AT_HARD_CPU15',raw_terminal_exists=False,qualified_source_pilot=False)])


def test_original_empty_cold_and_explicit_zero_start_failure_keep_original_terminal_bytes():
    old=dict(cold_never_started=True,results=[]);new=fixture();before=deepcopy(new)
    assert is_cold_terminal(old,launch_intent_exists=False,detached_owner_exists=False)
    assert is_cold_terminal(new,launch_intent_exists=False,detached_owner_exists=False,
        stop=dict(new_physics=0,model_queries=0,source_episode_attempts=0))
    assert new==before


@pytest.mark.parametrize('field',['launch_intent_exists','detached_owner_exists'])
def test_an_actual_launch_cannot_be_misclassified_as_cold(field):
    options=dict(launch_intent_exists=False,detached_owner_exists=False);options[field]=True
    with pytest.raises(AssertionError):is_cold_terminal(dict(cold_never_started=True,results=[]),**options)


def test_failure_status_alone_or_nonzero_physics_never_grants_cold_custody():
    for new in(fixture(),dict(results=[dict(OS_exit_status=1)]),dict(results=[])):
        assert not is_cold_terminal(new,launch_intent_exists=False,detached_owner_exists=False)
    new=fixture();new['physics_integrations']=1
    assert not is_cold_terminal(new,launch_intent_exists=False,detached_owner_exists=False,
        stop=dict(new_physics=0,model_queries=0,source_episode_attempts=0))


def test_future_native_readback_has_finite_cpu_time_for_measured_history_without_relaxing_checks():
    assert 'RLIMIT_CPU,(90,90)'in night_source_stage.READBACK
    assert 'scope.validate_account'in night_source_stage.READBACK
    assert "lease['status']=='ACTIVE'"in night_source_stage.READBACK
    compile(night_source_stage.PROGRAM,'new-cold-stage','exec')
