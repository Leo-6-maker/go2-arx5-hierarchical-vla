"""Synthetic fixtures: F0 is separate from reader exit1, neither is erased."""
from copy import deepcopy
import pytest
from . import b_f0_read_failure_closeout as unit
from . import b_recovery2_completion_contract as core
from .test_b_precuda_retry_completion import packet


@pytest.fixture
def failed(packet,monkeypatch):
    e,v,probe,p=deepcopy(packet);monkeypatch.setattr(core.b,'PROBE_SHA',probe)
    p['OS_exit'].update(returncode=1,observed_epoch=104)
    p['preserved_stop_exists']=True
    p['current_process']['observed_epoch']=105
    p['current_wrapper_process']=dict(pid=unit.original.WRAPPER_PID,present_same_instance=False,proc_start_ticks=None,physical_platform='VANDA_A40',observed_epoch=105)
    identity=dict(run_uuid=unit.original.RUN,job_id=unit.original.JOB,source_sha=unit.original.SOURCE,request_sha256=e['request_sha256'])
    error=dict(event='ERROR',outcome='READ_FAILED',error_type='RuntimeError',observed_epoch=102,
        intent=dict(identity=identity,effect='READ_ONLY',phase='READ_ONLY_838268923C8968CCD5E7',timeout_s=40,automatic_write_retry=False,started_epoch=101))
    p.update(failed_read_event=error,all_failed_read_events=[error],
        current_rpc=dict(status='CONFIRMED',intent=dict(identity=identity,effect='HEARTBEAT')),
        preserved_stop=dict(epoch=103,error_type='RuntimeError',message='VANDA_LOCAL_CONTROL_PROGRAM_FAILED:1',automatic_mutation_retry=False))
    v['effects']['finished_epoch']=100;v['process_final']['effects']['finished_epoch']=100
    return e,v,p


def call(failed):
    e,v,p=failed;return unit.owner(p,e,v,now_epoch=106)


def test_financial_failure_path_never_turns_original_owner_success_or_science_true(failed):
    e,v,p=failed;r=call(failed)
    assert r['accounting_close_permitted'] is True
    assert r['checkpoint_reuse_or_new_science_admitted'] is False
    assert r['original_successful_owner_predicate_satisfied'] is False
    with pytest.raises(PermissionError):unit.original.owner(p,e,now_epoch=106)


def test_auditor_summary_read_time_never_replaces_immutable_training_finish(failed):
    e,v,p=failed;v['effects']['finished_epoch']=106
    assert v['process_final']['effects']['finished_epoch']==100
    assert call(failed)['accounting_close_permitted'] is True
    v['process_final']['effects']['finished_epoch']=106
    with pytest.raises(PermissionError):call(failed)


@pytest.mark.parametrize('fault',['live_owner','live_wrapper','stale','other_PID','missing_stop','write_error','unknown_RPC',
    'extra_error','before_training','changed_grant','native_F1','missing_4096','wrong_error_identity','restarted_owner','timeout_inferred_exit'])
def test_missing_true_F0_full_state_or_exact_known_failed_reader_cannot_close(failed,fault):
    e,v,p=failed
    if fault=='live_owner':p['current_process']['present_same_instance']=True
    elif fault=='live_wrapper':p['current_wrapper_process']['present_same_instance']=True
    elif fault=='stale':p['current_process']['observed_epoch']=-20
    elif fault=='other_PID':p['owner_process']['pid']+=1
    elif fault=='missing_stop':p['preserved_stop_exists']=False
    elif fault=='write_error':p['failed_read_event']['intent']['effect']='MUTATION'
    elif fault=='unknown_RPC':p['current_rpc']['status']='UNKNOWN'
    elif fault=='extra_error':p['all_failed_read_events'].append(deepcopy(p['failed_read_event']))
    elif fault=='before_training':p['failed_read_event']['intent']['started_epoch']=99
    elif fault=='changed_grant':e['request']['sealed_inputs'][unit.original.authority.PLAN]='0'*64
    elif fault=='native_F1':v['raw_native']['Exit_status']=1
    elif fault=='missing_4096':v['checkpoints'].pop('4096')
    elif fault=='wrong_error_identity':p['failed_read_event']['intent']['identity']['job_id']='OTHER'
    elif fault=='restarted_owner':p['owner_process']['proc_start_ticks']='OTHER'
    elif fault=='timeout_inferred_exit':p['OS_exit']['foreground_wait_completed']=False
    with pytest.raises(PermissionError):call(failed)
