"""CPU finality composition; synthetic fixtures cannot replace actual hashes."""
from copy import deepcopy
from unittest.mock import patch
import hashlib,json
import pytest
from . import b_retry2970_finality as c
from .test_b_retry2970_closeout import reserved_readonly

def raw(value):return (json.dumps(value,indent=2)+'\n').encode()

def fixture():
    e,v,digest,h,d,a=reserved_readonly();epoch=1792756412.
    e['request'].setdefault('run_manifest_sha256','a'*64)
    v['native_proof'].update(schema='VAP1_NATIVE_A40_SOURCE_TERMINAL_V1',observed_epoch=epoch,fresh_native_scheduler_check=False,
        process_final_sha256='b'*64,wrapper_final_sha256='c'*64,conservative_GPU_s=8481)
    v.update(financial_settlement_ready=False,CPU_audit_actual_s=214.,
        streaming_witnesses=dict(all_unique_inodes_independently_rehashed=True,all_paths_rechecked=True))
    v['raw_native'].update(Rerunable=False,run_count=1)
    history=dict(native_job=deepcopy(v['raw_native']),observed_epoch=epoch,receipts={
        'PBS_PROCESS_FINAL.json':dict(sha256='b'*64),'PBS_WRAPPER_FINAL.json':dict(sha256='c'*64),
        'A40_NATIVE_EFFECT_LEDGER.json':dict(sha256='d'*64)})
    rb=dict(schema=c.SCHEMA,observed_epoch=1792765000.,native_scheduler_query_used=False,
        request_sha256=c.close.failed.REQUEST_SHA,run_manifest_sha256=e['request']['run_manifest_sha256'],
        process_final_sha256='b'*64,wrapper_final_sha256='c'*64,native_effect_ledger_sha256='d'*64,
        effects=deepcopy(v['effects']),old_owner_state=dict(MainPID='0',ActiveState='failed',ExecMainStatus='1'),
        audit_actor_handle_absent=True)
    done=dict(status='COMPLETE_INDEPENDENT_SAVED_STATE_AUDIT_NO_SETTLEMENT',returncode=0,
        result_sha256=hashlib.sha256(raw(v)).hexdigest(),program_sha256=c.AUDIT_PROGRAM,CPU_actual_s=214.5)
    return e,v,digest,h,d,a,history,rb,done

def compose(e,v,digest,h,d,a,history,rb,done,*,now=1792765001.):
    saved,historical=raw(v),raw(history);done=deepcopy(done);done['result_sha256']=hashlib.sha256(saved).hexdigest();completed=raw(done)
    with patch.object(c,'AUDIT_SHA',hashlib.sha256(saved).hexdigest()),patch.object(c,'HISTORY_SHA',hashlib.sha256(historical).hexdigest()),patch.object(c,'COMPLETED_SHA',hashlib.sha256(completed).hexdigest()),patch.object(c.close.failed.prior.b,'PROBE_SHA',digest):
        return c.evidence(saved_bytes=saved,completion_bytes=completed,historical_native_bytes=historical,
            readback=rb,expected=e,readonly_allocation=a,budget_before=(h,d),now_epoch=now)

def test_old_real_F_timestamp_is_unchanged_and_fresh_metadata_is_explicitly_separate():
    values=fixture();before=deepcopy(values);result=compose(*values)
    assert values==before
    assert result['native_proof']['observed_epoch']==1792756412.
    assert result['historical_native_observed_epoch']==1792756412.
    proof=result['accounting_native_proof']
    assert proof['schema']==c.ACCOUNTING_SCHEMA and proof['schema']!=result['native_proof']['schema']
    assert proof['observed_epoch']==1792765000. and proof['original_PBS_observed_epoch']==1792756412.
    assert proof['fresh_native_scheduler_check'] is False and proof['fresh_native_terminal_artifact_check'] is True
    assert proof['new_training_admission'] is False and proof['observation_basis']==c.BASIS
    assert result['financial_settlement_ready'] is True and result['effects']['optimizer_updates_committed']==922
    assert result['CPU_audit_actual_s']==214.5 and result['native_proof']['conservative_CPU_s']==11023

@pytest.mark.parametrize('fault',['prior_R','rerunnable','secondrun','different_native','audit_not_final',
    'missing_rehash','wrong_program','audit_failed','stale_files','future_files','wrong_schema','fake_fresh_PBS',
    'request','manifest','process_final','wrapper','effect_file','changed_effects','live_owner','live_auditor',
    'CPU_dropped','hold_released','unknown','hardclose'])
def test_timeouts_or_changed_native_files_cannot_become_a_final_F_or_release_resources(fault):
    e,v,digest,h,d,a,history,rb,done=fixture();now=1792765001.
    if fault=='prior_R':history['native_job']['job_state']='R';v['raw_native']['job_state']='R'
    elif fault=='rerunnable':history['native_job']['Rerunable']=v['raw_native']['Rerunable']=True
    elif fault=='secondrun':history['native_job']['run_count']=v['raw_native']['run_count']=2
    elif fault=='different_native':history['native_job']['resources_used']={'cput':'99:00:00'}
    elif fault=='audit_not_final':v['financial_settlement_ready']=True
    elif fault=='missing_rehash':v['streaming_witnesses']['all_unique_inodes_independently_rehashed']=False
    elif fault=='wrong_program':done['program_sha256']='0'*64
    elif fault=='audit_failed':done['returncode']=1
    elif fault=='stale_files':rb['observed_epoch']=1792764800.
    elif fault=='future_files':rb['observed_epoch']=1792765002.
    elif fault=='wrong_schema':rb['schema']='TIMEOUT_MEANS_FINISHED'
    elif fault=='fake_fresh_PBS':rb['native_scheduler_query_used']=True
    elif fault=='request':rb['request_sha256']='0'*64
    elif fault=='manifest':rb['run_manifest_sha256']='0'*64
    elif fault=='process_final':rb['process_final_sha256']='0'*64
    elif fault=='wrapper':rb['wrapper_final_sha256']='0'*64
    elif fault=='effect_file':rb['native_effect_ledger_sha256']='0'*64
    elif fault=='changed_effects':rb['effects']['optimizer_updates_committed']=0
    elif fault=='live_owner':rb['old_owner_state']['MainPID']='1'
    elif fault=='live_auditor':rb['audit_actor_handle_absent']=False
    elif fault=='CPU_dropped':a['prior_all_audit_and_admin_CPU_upper_s']=0
    elif fault=='hold_released':d['CPU_reserved_s']=0
    elif fault=='unknown':h['unknown_effects']=[{}]
    else:now=1792952996
    with pytest.raises(PermissionError):compose(e,v,digest,h,d,a,history,rb,done,now=now)

def test_semantically_similar_unqualified_history_bytes_do_not_pass_exact_actual_binding():
    e,v,digest,h,d,a,history,rb,done=fixture()
    with pytest.raises(PermissionError):
        c.evidence(saved_bytes=raw(v),completion_bytes=raw(done),historical_native_bytes=raw(history),readback=rb,
            expected=e,readonly_allocation=a,budget_before=(h,d),now_epoch=1792765001.)

@pytest.mark.parametrize('fault',['rename_only','another_job','another_request','another_source',
    'missing_history','missing_completed_audit','restamped_PBS','stale_files','running_owner',
    'changed_effects','refunded_CPU','refunded_GPU','success','training_admission'])
def test_mature_accounting_boundary_rejects_relabelled_or_changed_proofs(fault):
    e,v,digest,h,d,a,history,rb,done=fixture()
    rb.update(process_final_sha256='ccb665e0b2041930f3b46b8dcb3c9245cfe34a48cdd9ba60bcd8eb1cd53e1bec',
        wrapper_final_sha256='899da277bacb96b476775bca19811282b237a4ca9fdf6fc46492a847d0e77916',
        native_effect_ledger_sha256='d96731acc5e9ea9a70e77438a8911db1a368aaf800f3892ed4933c8ac4cd71a6')
    v['native_proof'].update(process_final_sha256=rb['process_final_sha256'],wrapper_final_sha256=rb['wrapper_final_sha256'],
        new_resident_allocated_upper_bytes=70000000000,Exit_status=1)
    proof=c.accounting_proof(v,rb,1792756412.9734638)
    row=dict(job_id=c.close.failed.JOB,request_id=c.close.failed.REQUEST,
        request_sha256=c.close.failed.REQUEST_SHA,source_sha=c.close.failed.SOURCE,
        run_manifest_sha256=e['request']['run_manifest_sha256'])
    # These are synthetic CPU records. They verify the native boundary's exact
    # opt-in guards; only retained byte hashes and fresh native reads can supply
    # the actual values in production.
    c.validate_accounting_proof(proof,row=row,effects=v['effects'],now_epoch=1792765001.)
    if fault=='rename_only':proof=dict(v['native_proof'],schema=c.ACCOUNTING_SCHEMA,observed_epoch=1792765000.)
    elif fault=='another_job':row['job_id']='other.stdct-mgmt-02'
    elif fault=='another_request':row['request_id']='other'
    elif fault=='another_source':row['source_sha']='f'*40
    elif fault=='missing_history':proof.pop('original_PBS_readback_sha256')
    elif fault=='missing_completed_audit':proof.pop('full_state_audit_completed_sha256')
    elif fault=='restamped_PBS':proof['original_PBS_observed_epoch']=rb['observed_epoch']
    elif fault=='stale_files':proof['observed_epoch']=proof['terminal_artifact_readback']['observed_epoch']=1792764000.
    elif fault=='running_owner':proof['terminal_artifact_readback']['old_owner_state']['MainPID']='123'
    elif fault=='changed_effects':proof['terminal_artifact_readback']['effects']['optimizer_updates_committed']=0
    elif fault=='refunded_CPU':proof['conservative_CPU_s']=8323
    elif fault=='refunded_GPU':proof['conservative_GPU_s']=0
    elif fault=='success':proof['Exit_status']=0
    else:proof['new_training_admission']=True
    with pytest.raises(PermissionError):
        c.validate_accounting_proof(proof,row=row,effects=v['effects'],now_epoch=1792765001.)

def test_existing_accounting_gate_still_rejects_stale_original_terminal_proof():
    from .test_a40_accounting import SourceAccountingTests,terminal
    from . import a40_accounting
    f,h,d=SourceAccountingTests().bound();native,effects=terminal(f)
    native['observed_epoch']=f['now']-121
    with pytest.raises(PermissionError):
        a40_accounting.settle(h,d,request_id=json.loads(f['request_bytes'])['request_id'],
            native_proof=native,effects=effects,now_epoch=f['now'])
