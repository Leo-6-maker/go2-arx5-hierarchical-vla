"""CPU row-validation fixtures; never reserve, bind, settle or submit DEV."""
from copy import deepcopy

import pytest

from scripts.pi0294.pv1 import evaluation_accounting as a
from scripts.pi0294.pv1 import evaluation_platform as e
from scripts.pi0294.dec1 import alternate_accounting as legacy
from scripts.pi0294.dec1.test_alternate_accounting import fixture as old_fixture,NOW


def row(candidate='A_CONTROL',conditions=24):
    return dict(native_platform=e.PLATFORM,platform_evaluation_version=e.VERSION,
        scientific_admission_version=e.VERSION,authority_release=e.AUTHORITY,
        campaign_run_uuid='20261018T182956Z-localpc-pv1',owner_user='e1192548',
        child_run_uuid='fixture-vanda-evaluation',created_epoch=e.T0+100000,
        stage='B0_DEV' if candidate=='B0_CORR' else 'P4_DEV',evaluation_candidate=candidate,
        project='CFP04-CF-007',job_name='pi0294_pv1_a40_dev',checkpoint_manifest_sha256=e.ENDPOINTS[candidate],
        parent_checkpoint_manifest_sha256=e.PARENT,statistics_sha256=e.R0,physical_GPU_kind='NVIDIA_A40',
        physical_GPU_count=1,legacy_GPU_seconds_field_is_projection=True,reserved_H200_s=28800,
        accounting_pad_s=60,CPU_upper_s=80000,CPU_envelope_profile=a.BOUNDED_CPU_PROFILE,
        declared_process_CPU_limit_s=79760,query_limit=80*conditions+17,
        optimizer_update_limit=0,reserved_optimizer_updates=0,original_T0_epoch=e.T0,
        hard_close_epoch=e.HARD_CLOSE,charges_DEC1_subcaps=False,new_robot_physics=0,
        remaining_conditions=conditions,noise_bank=294101,additional_output_upper_bytes=a.output_upper(conditions),
        request_id='12345678-1234-4234-8234-123456789abc',submission_nonce='12345678-1234-4234-8234-123456789abd',
        request_sha256='1'*64,run_manifest_sha256='2'*64,source_sha='3'*40,cold_input_audit_sha256='4'*64,
        planned_panel_sha256='5'*64,prior_outcomes_sha256='6'*64,checkpoint_reuse_proof_sha256='7'*64,
        prequeue_future_admission_sha256='8'*64,status='SUBMITTED_PENDING_NATIVE',job_id='1999999.stdct-mgmt-02',
        reserved_neural_queries=80*conditions+17)


def global_fixture(r):
    return dict(schema='PI0294_PV1_HOPPER_CAMPAIGN_LEDGER_V1',authority_release=e.AUTHORITY,
        campaign_run_uuid='20261018T182956Z-localpc-pv1',PBS_submissions=[r],
        reserved=dict(H200_s=28860,neural_queries=r['query_limit'],optimizer_updates=0),
        CPU_reserved_s=80000,storage_reserved_bytes=r['additional_output_upper_bytes'])


@pytest.mark.parametrize('candidate',list(e.ENDPOINTS))
@pytest.mark.parametrize('conditions',[10,24])
def test_existing_models_original_noise_panel_upper_is_consistency_only(candidate,conditions):
    r=row(candidate,conditions);v=global_fixture(r);before=deepcopy(v)
    assert legacy.validate_items(v)==before and v==before
    assert r['additional_output_upper_bytes']==conditions*1024**3
    assert r['reserved_optimizer_updates']==0 and r['charges_DEC1_subcaps'] is False


@pytest.mark.parametrize('key,value',[
    ('platform_training_version','old-train'),('native_platform',None),('platform_evaluation_version',None),
    ('charges_DEC1_subcaps',True),('stage','P2_PRIMARY'),('evaluation_candidate','REPLICA_A_CONTROL'),
    ('checkpoint_manifest_sha256','f'*64),('statistics_sha256','f'*64),('optimizer_update_limit',1),
    ('reserved_optimizer_updates',True),('reserved_neural_queries',0),('noise_bank',123),
    ('CPU_upper_s',54000),('declared_process_CPU_limit_s',80000),('additional_output_upper_bytes',1),
    ('remaining_conditions',0),('remaining_conditions',25),('remaining_conditions',True),
    ('job_id','123.hopper-m-02'),('status','NATIVE_QSUB_REJECTED_NO_JOB'),
    ('prior_outcomes_sha256',''),('prequeue_future_admission_sha256',''),
    ('request_id','not-a-uuid'),('original_T0_epoch',e.T0+1),('created_epoch',e.STOP_NEW),
    ('child_run_uuid','20261018T182956Z-localpc-pv1')])
def test_eval_row_does_not_hide_as_training_or_compatibility_or_refund_unknown(key,value):
    r=row();v=global_fixture(r);r[key]=value
    with pytest.raises(PermissionError):legacy.validate_items(v)


@pytest.mark.parametrize('fault',['GPU','CPU','output','query','two_active','cap_proposal','duplicate_nonce'])
def test_same_global_escrows_and_cap_preserved(fault):
    r=row();v=global_fixture(r)
    if fault=='GPU':v['reserved']['H200_s']-=1
    if fault=='CPU':v['CPU_reserved_s']-=1
    if fault=='output':v['storage_reserved_bytes']-=1
    if fault=='query':v['reserved']['neural_queries']-=1
    if fault in ('two_active','duplicate_nonce'):
        other=dict(status='SUBMITTED_PENDING_NATIVE' if fault=='two_active' else 'NATIVE_TERMINAL',
            job_id='1.hopper-m-02',submission_nonce=r['submission_nonce'] if fault=='duplicate_nonce' else 'other')
        v['PBS_submissions'].insert(0,other)
    if fault=='cap_proposal':
        v['global_PBS_limit']=34
        v['PBS_submissions']=[dict(status='NATIVE_TERMINAL',job_id=str(i)+'.hopper-m-02') for i in range(24)]+[r]
    with pytest.raises(PermissionError):legacy.validate_items(v)


def test_unknown_qsub_keeps_full_escrow_without_invented_job_or_retry():
    r=row();r.update(job_id=None,status='QSUB_INTENT_OUTCOME_UNKNOWN_NO_RETRY');v=global_fixture(r)
    legacy.validate_items(v)
    assert r['job_id'] is None and v['reserved']['H200_s']==28860
    r['status']='SUBMITTED_PENDING_NATIVE'
    with pytest.raises(PermissionError):legacy.validate_items(v)


def test_terminal_failed_costs_are_preserved_without_creating_optimizer_effects():
    r=row();r.update(status='NATIVE_TERMINAL',native_exit_status=1,terminal_native_sha256='9'*64,
        terminal_effect_summary_sha256='a'*64,conservative_H200_s=200,conservative_CPU_s=160,
        query_attempts=17,reserved_neural_queries=0,committed_optimizer_updates=0)
    v=global_fixture(r);v.update(CPU_reserved_s=0,storage_reserved_bytes=0)
    v['reserved']=dict(H200_s=0,neural_queries=0,optimizer_updates=0)
    legacy.validate_items(v);assert r['conservative_H200_s']==200
    r['committed_optimizer_updates']=1
    with pytest.raises(PermissionError):legacy.validate_items(v)


def test_old_compatibility_row_retains_its_own_contract_next_to_eval():
    v,request,native=old_fixture();v=legacy.reserve(v,request,now_epoch=NOW,fresh_native=native)
    old=v['PBS_submissions'][-1]
    old.update(status='NATIVE_TERMINAL',job_id='1469999.stdct-mgmt-02',terminal_native_sha256='b'*64,terminal_effect_summary_sha256='c'*64,
        conservative_H200_s=120,conservative_CPU_s=87,query_attempts=17,reserved_neural_queries=0)
    r=row();v['PBS_submissions'].append(r)
    v['reserved']=dict(H200_s=28860,neural_queries=r['query_limit'],optimizer_updates=0)
    v.update(CPU_reserved_s=80000,storage_reserved_bytes=r['additional_output_upper_bytes'])
    before=deepcopy(old);legacy.validate_items(v);assert old==before
    old['status']='SUBMITTED_PENDING_NATIVE';old['reserved_neural_queries']=17
    with pytest.raises(PermissionError,match='ONE_REGISTERED_GPU_CHAIN'):legacy.validate_items(v)
