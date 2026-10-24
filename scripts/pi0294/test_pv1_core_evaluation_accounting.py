"""CPU lifecycle only. Captured budgets are real; process handles are fixtures."""
import base64
from copy import deepcopy
import json
from pathlib import Path
import uuid

import pytest

from scripts.pi0294.pv1 import evaluation_core_accounting as a
from scripts.pi0294.pv1 import evaluation_platform as e,evaluation_stage as s
from scripts.pi0294.pv1.neural_ledger import check

NOW=1792834700
BOOT='12345678-1234-4234-8234-123456789abc'


def row():
    return dict(native_platform='CORELAB_RTX4080',platform_evaluation_version=a.VERSION,
        authority_release=a.AUTHORITY,campaign_run_uuid=a.scope.CAMPAIGN,
        evaluation_execution_stage=s.VERSION,evaluation_stage_plan_sha256=s.PLAN_SHA,
        evaluation_stage_card_sha256=s.CARD_SHA,owner_user='liuyu-nfs',owner_uid=2009,
        hostname='corelab-4080s',physical_GPU_count=1,physical_GPU_kind='RTX_4080_SUPER',
        stage='P4_DEV',noise_bank=294101,remaining_conditions=24,evaluation_candidate='A_CONTROL',
        checkpoint_manifest_sha256=e.ENDPOINTS['A_CONTROL'],statistics_sha256=e.R0,
        parent_checkpoint_manifest_sha256=e.PARENT,optimizer_update_limit=0,
        reserved_optimizer_updates=0,charges_DEC1_subcaps=False,new_robot_physics=0,
        walltime_s=28800,accounting_pad_s=60,reserved_GPU_s=a.GPU_UPPER,CPU_upper_s=a.CPU_UPPER,
        declared_process_CPU_limit_s=79760,active_cpu_limit=4,requested_memory_bytes=48*1024**3,
        minimum_GPU_memory_bytes=30*1024**3,query_limit=1937,additional_output_upper_bytes=a.output_upper(24)+a.MODEL_BYTES,
        original_T0_epoch=e.T0,hard_close_epoch=e.HARD_CLOSE,PBS_submissions_consumed=0,
        legacy_GPU_seconds_field_is_projection=True,request_id='22345678-1234-4234-8234-123456789abc',
        submission_nonce='32345678-1234-4234-8234-123456789abc',child_run_uuid='fixture-core-evaluation',
        source_sha='1'*40,request_sha256='2'*64,run_manifest_sha256='3'*64,
        planned_panel_sha256='4'*64,prior_outcomes_sha256='5'*64,cold_input_audit_sha256='6'*64,
        prequeue_future_admission_sha256='7'*64,created_epoch=NOW-5,
        GPU_uuid='GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad',status=a.STATES[0],
        native_handle=None,reserved_neural_queries=1937)


def process(r):
    return dict(schema='PV1_CORE_GPU_NATIVE_PROCESS_STARTED_V1',hostname=r['hostname'],uid=2009,
        GPU_uuid=r['GPU_uuid'],request_id=r['request_id'],request_sha256=r['request_sha256'],
        submission_nonce=r['submission_nonce'],run_manifest_sha256=r['run_manifest_sha256'],
        source_sha=r['source_sha'],process_state='RUNNING',observed_epoch=NOW,started_epoch=NOW-2,
        boot_id=BOOT,pid=12345,proc_start=10001)


def test_unbound_process_intent_never_invents_a_PBS_handle():
    r=row();assert a.row(r)==r and r['native_handle'] is None
    assert a.process_handle(BOOT,12345,10001)=='coregpu:'+BOOT+':12345:10001'
    with pytest.raises(PermissionError):a.process_handle(BOOT,True,10001)


@pytest.mark.parametrize('key,value',[
    ('project','CFP04-CF-007'),('submit_count',26),('PBS_JOBID','1480000.stdct-mgmt-02'),
    ('job_name','pi0294_pv1_a40_dev'),('platform_training_version','old'),('training_block',{}),
    ('native_platform','VANDA_A40'),('hostname','corelab-amd-server'),('owner_uid',True),
    ('physical_GPU_count',2),('physical_GPU_kind','NVIDIA_A40'),('active_cpu_limit',16),
    ('requested_memory_bytes',32*1024**3),('minimum_GPU_memory_bytes',16*1024**3),
    ('checkpoint_manifest_sha256','f'*64),('noise_bank',294102),('remaining_conditions',8),
    ('charges_DEC1_subcaps',True),('optimizer_update_limit',1),('created_epoch',e.STOP_NEW),
    ('GPU_uuid','0'),('native_handle','1480000.stdct-mgmt-02'),('reserved_neural_queries',0),
    ('CPU_upper_s',80001),('request_id','not-a-uuid')])
def test_wrong_platform_device_stage_or_extra_cap_is_refused(key,value):
    r=row();r[key]=value
    with pytest.raises(PermissionError):a.row(r)


@pytest.fixture
def captured_global():
    p=Path(__file__).resolve().parents[4]/'work/private/science-next/wdr1-target-identity/v4-approved-s2-scope-adoption-01/INDEPENDENT_GLOBAL_READBACK.private.json'
    if not p.is_file():pytest.skip('private full actual global bytes absent')
    g=json.loads(base64.b64decode(json.loads(p.read_bytes())['global_after_base64']))
    check(g);assert len(g['PBS_submissions'])==25
    return g


def bound(g):
    r=row();reserved=a.reserve(g,r)
    return a.bind(reserved,request_id=r['request_id'],process=process(r),now_epoch=NOW)


def debit(v,kind,count,effect):
    r=next(iter(v[a.FIELD].values()))
    return a.charge(v,request_id=r['request_id'],native_handle=r['native_handle'],kind=kind,count=count,effect_id=effect)


def final(v):
    r=next(iter(v[a.FIELD].values()))
    n=dict(schema='PV1_CORE_GPU_NATIVE_PROCESS_TERMINAL_V1',request_id=r['request_id'],
        native_handle=r['native_handle'],hostname=r['hostname'],uid=2009,GPU_uuid=r['GPU_uuid'],
        boot_id=r['boot_id'],pid=r['pid'],proc_start=r['proc_start'],process_state='TERMINAL',
        request_sha256=r['request_sha256'],submission_nonce=r['submission_nonce'],
        recorded_process_terminal=True,terminal_inventory_stable=True,exit_evidence_kind='SYSTEMD_EXEC_MAIN_EXIT',
        native_exit_status=1,finished_epoch=NOW+8,observed_epoch=NOW+9,
        conservative_GPU_s=70,conservative_CPU_s=20,new_resident_allocated_upper_bytes=1024)
    f=dict(schema='PV1_CORE_GPU_NATIVE_DEV_EFFECT_SUMMARY_V1',request_id=r['request_id'],
        native_handle=r['native_handle'],request_sha256=r['request_sha256'],source_sha=r['source_sha'],
        run_manifest_sha256=r['run_manifest_sha256'],evaluation_candidate=r['evaluation_candidate'],
        checkpoint_manifest_sha256=r['checkpoint_manifest_sha256'],statistics_sha256=e.R0,
        optimizer_updates_committed=0,new_physics=0,new_episodes=0,effect_journal_complete=True,
        query_attempts=3,model_load_attempts=1,unknown_query_intents=[])
    return n,f


def test_actual25_costs_preserved_pure_lifecycle_and_no_PBS_consumed(captured_global):
    g=deepcopy(captured_global);v=bound(g)
    assert g==captured_global and v['used']['PBS_submissions']==25
    v=debit(v,'model_loads',1,'fixture/load');v=debit(v,'generation_queries',1,'fixture/q0')
    v=debit(v,'CPU_s',7,'fixture/cpu');n,f=final(v)
    out=a.settle(v,request_id=row()['request_id'],native_proof=n,effects=f,now_epoch=NOW+10)
    assert out['PBS_submissions']==g['PBS_submissions'] and out['training_runs']==g['training_runs']
    assert out['optimizer_updates']==11714 and out['generation_queries']==g['generation_queries']+3
    assert out['model_loads']==g['model_loads']+1 and out['CPU_s']==g['CPU_s']+20
    assert out['used']['H200_s']==g['used']['H200_s']+70
    assert out['reserved']==g['reserved'] and out['CPU_reserved_s']==g['CPU_reserved_s']
    assert out['storage_reserved_bytes']==g['storage_reserved_bytes']
    assert out['new_storage_current_bytes']==g['new_storage_current_bytes']+1024
    with pytest.raises(PermissionError):a.settle(out,request_id=row()['request_id'],native_proof=n,effects=f,now_epoch=NOW+10)


@pytest.mark.parametrize('fault',['duplicate','second_Core_GPU','concurrent_PBS','drop_GPU','drop_CPU','drop_output','drop_queries'])
def test_real_global_fences_cross_platform_dispatch_and_escrow_loss(captured_global,fault):
    v=bound(captured_global);r=row()
    if fault=='duplicate':
        with pytest.raises(PermissionError):a.reserve(v,r)
        return
    if fault=='second_Core_GPU':
        r.update(request_id=str(uuid.uuid4()),submission_nonce=str(uuid.uuid4()),evaluation_candidate='B_COVERAGE',
            checkpoint_manifest_sha256=e.ENDPOINTS['B_COVERAGE'])
        with pytest.raises(PermissionError):a.reserve(v,r)
        return
    if fault=='concurrent_PBS':
        v['PBS_submissions'][0]['status']='SUBMITTED_PENDING_NATIVE'
    if fault=='drop_GPU':v['reserved']['H200_s']-=1
    if fault=='drop_CPU':v['CPU_reserved_s']-=1
    if fault=='drop_output':v['storage_reserved_bytes']-=1
    if fault=='drop_queries':v['reserved']['neural_queries']-=1
    with pytest.raises(PermissionError):check(v)


@pytest.mark.parametrize('key,value',[
    ('process_state','MISSING'),('recorded_process_terminal',False),('terminal_inventory_stable',False),
    ('exit_evidence_kind','OBSERVATION_TIMEOUT'),('boot_id','42345678-1234-4234-8234-123456789abc'),
    ('native_exit_status',True),('conservative_GPU_s',69),('conservative_CPU_s',80001)])
def test_pid_absence_timeout_other_boot_or_cost_overrun_keeps_all_holds(captured_global,key,value):
    v=bound(captured_global);before=deepcopy(v);n,f=final(v);n[key]=value
    with pytest.raises(PermissionError):a.settle(v,request_id=row()['request_id'],native_proof=n,effects=f,now_epoch=NOW+10)
    assert v==before


def test_unknown_journals_kept_and_cpu_closeout_does_not_enable_new_effects(captured_global):
    v=debit(bound(captured_global),'model_loads',1,'load');v=debit(v,'generation_queries',1,'q0')
    v['unknown_effects']=['OLD_UNKNOWN_PRESERVED'];v=debit(v,'CPU_s',5,'cpu')
    n,f=final(v);f['unknown_query_intents']=[1]
    done=a.settle(v,request_id=row()['request_id'],native_proof=n,effects=f,now_epoch=NOW+10)
    assert done['unknown_effects'][0]=='OLD_UNKNOWN_PRESERVED'
    assert done['unknown_effects'][1]['unknown_query_intents']==[1]
    assert done['reserved']['neural_queries']==1
    with pytest.raises(PermissionError):debit(done,'generation_queries',1,'q_retry')
