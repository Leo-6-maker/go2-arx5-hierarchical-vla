"""Synthetic CPU terminal fixtures; never actual native/endpoint evidence."""
from copy import deepcopy
import uuid
import pytest
from scripts.pi0294.pv1 import mix_accounting as a,successor_cd as s,mix_stage_handoff as h


def fixture():
    resources=s.whole_package();marker=dict(s.scope(1000.),grant_sha256=s.GRANT_SHA,
        before_sha256=dict(GLOBAL='a'*64,LOCAL='b'*64),authorized_storage_caps=dict(s.NEW_STORAGE),
        old_clock_and_history_preserved=True,scope_only_not_GPU_or_physics_admission=True)
    package=dict(schema='V5_CD_EXISTING_LEDGER_PACKAGE_ESCROW_V1',scope_grant_sha256=s.GRANT_SHA,
        order=list(s.ORDER),created_epoch=1000.,original_resources=resources,
        remaining_resources=deepcopy(resources),completed_stages=[],engineering_replacements_used=0)
    g={s.FIELD:marker,a.PACKAGE:package,'PBS_submissions':[],
        'used':dict(H200_s=0.,PBS_submissions=0,optimizer_updates=0,neural_queries=0),
        'reserved':dict(H200_s=resources['H200_s'],optimizer_updates=2048,neural_queries=resources['neural_queries']),
        'CPU_s':0.,'CPU_reserved_s':resources['Hopper_CPU_s'],
        'storage_reserved_bytes':resources['Hopper_storage_bytes'],'new_storage_current_bytes':0,
        'new_storage_peak_bytes':0,'optimizer_updates':0,'generation_queries':0,'model_loads':0}
    g=a.prepay_training_control(g,arm='C50',now_epoch=1001.)
    l={s.FIELD:deepcopy(marker),a.PACKAGE:deepcopy(g[a.PACKAGE]),'PBS_submissions':[],
        'GPU_reserved_s':resources['H200_s'],'GPU_allocated_s':0.,
        'CPU_reserved_s':dict(Hopper=g['CPU_reserved_s']),'CPU_s':dict(Hopper=g['CPU_s']),
        'optimizer_updates':0,'neural_queries':0,'unknown_effects':[],'pending_updates':[],'pending_queries':[],
        'old_history_fixture':dict(failure='PRESERVED',paid_updates=805,old_T0=1792348196)}
    r=dict(platform_training_version=a.VERSION,native_platform='VANDA_A40',arm='C50',stage='V5_MATCHED_TRAIN1024',
        parent_checkpoint_manifest_sha256=a.PARENT,statistics_sha256=a.STATS,optimizer_update_limit=1024,
        query_limit=29,CPU_upper_s=54000,reserved_H200_s=22500,accounting_pad_s=60,physical_GPU_count=1,
        physical_GPU_kind='NVIDIA_A40',project='CFP04-CF-007',job_name='pi0294_dec1_a40_train',
        additional_output_upper_bytes=70000000000,source_sha='c'*40,request_id=str(uuid.uuid4()),
        submission_nonce=str(uuid.uuid4()),request_sha256='d'*64,run_manifest_sha256='e'*64,
        cold_input_audit_sha256='f'*64,status='QSUB_INTENT_OUTCOME_UNKNOWN_NO_RETRY',job_id=None,
        T_CD=1000.,deadline_epoch=marker['deadline_epoch'],scope_grant_sha256=s.GRANT_SHA,run_uuid='fixture-c50')
    g=a.register_request(g,r,now_epoch=1002.);g=a.bind(g,r['request_id'],'99.stdct-mgmt-02',now_epoch=1003.)
    native=dict(job_id='99.stdct-mgmt-02',job_state='F',Exit_status=0,request_id=r['request_id'],
        submission_nonce=r['submission_nonce'],conservative_GPU_s=8000.,conservative_CPU_s=9000.)
    effects=dict(schema='V5_CD_ACTUAL_NATIVE_EFFECT_SUMMARY_V1',request_id=r['request_id'],request_sha256=r['request_sha256'],
        job_id=native['job_id'],arm='C50',optimizer_updates_committed=1024,query_attempts=29,model_load_attempts=3,
        pending_updates=[],unknown_query_intents=[])
    audit=dict(status='PASS_INDEPENDENT_V5_CD_1024_COMPLETE_STATE',arm='C50',checkpoint_manifest_sha256='1'*64,
        checkpoint_root='/fixture/only/checkpoint-1024',member_count=15,actual_consumed_cursor=32768,
        optimizer_steps=[1024],scheduler_last_epoch=1024,rng_keys=['cpu','numpy','python'],
        parent_checkpoint_manifest_sha256=a.PARENT,statistics_sha256=a.STATS,probe_result_sha256='2'*64)
    g=a.settle_training(g,r['request_id'],native=native,effects=effects,audit=audit,now_epoch=1100.)
    settled=a.raw(dict(native=native,effects=effects,audit=audit))
    lease=a.raw(dict(status='CLOSED',run_uuid=r['run_uuid'],terminal_receipt_sha256=a.sha(settled),closed_epoch=1101.))
    closed=a.raw(dict(lease_sha256=a.sha(lease),job_id=native['job_id'],native_exit_status=0,complete_endpoint=True))
    captured=dict(settled_bytes=settled,lease_bytes=lease,closed_bytes=closed,
        owner_exit_bytes=a.raw(dict(returncode=0,endpoint_status=audit['status'],observed_epoch=1102.)),
        OS_exit_bytes=a.raw(dict(returncode=0,source_sha=r['source_sha'],foreground_wait_completed=True,finished_epoch=1103.)))
    return g,l,captured


def test_complete_closed_state_joins_all_actual_bytes_not_only_saved_model():
    g,l,c=fixture();v=h.closed_endpoint(global_bytes=a.raw(g),arm='C50',**c)
    assert v['global_step']==1024 and not v['model_or_physics_admission']


@pytest.mark.parametrize('which,field,value',[
    ('OS_exit_bytes','returncode',1),('owner_exit_bytes','returncode',1),
    ('OS_exit_bytes','returncode',False),('OS_exit_bytes','source_sha','wrong'),
    ('OS_exit_bytes','finished_epoch',1099.),('lease_bytes','status','ACTIVE'),
    ('lease_bytes','run_uuid','oldB'),('closed_bytes','complete_endpoint',False),
    ('closed_bytes','lease_sha256','0'*64),('closed_bytes','native_exit_status',1)])
def test_failed_owner_stale_or_prepared_close_not_an_endpoint(which,field,value):
    import json
    g,l,c=fixture();v=json.loads(c[which]);v[field]=value;c[which]=a.raw(v)
    with pytest.raises(PermissionError):h.closed_endpoint(global_bytes=a.raw(g),arm='C50',**c)


def test_partial_saved_state_or_other_arm_never_advances():
    g,l,c=fixture()
    with pytest.raises(PermissionError):h.closed_endpoint(global_bytes=a.raw(g),arm='D25',**c)
    g['PBS_submissions'][0]['committed_optimizer_updates']=1023
    with pytest.raises(PermissionError):h.closed_endpoint(global_bytes=a.raw(g),arm='C50',**c)


def test_existing_LOCAL_projection_credits_only_paid_training_once():
    g,l,c=fixture();before=deepcopy(l);gb=a.raw(g)
    proposal=h.project_training_to_LOCAL(local_bytes=a.raw(l),global_bytes=gb,terminal_capture=c,arm='C50');v=proposal['proposed_LOCAL']
    assert l==before and a.raw(g)==gb and v['old_history_fixture']==l['old_history_fixture']
    assert v['GPU_allocated_s']==8000 and v['GPU_reserved_s']==l['GPU_reserved_s']-22560
    assert v['CPU_s']['Hopper']==l['CPU_s']['Hopper']+9000
    assert v['CPU_reserved_s']['Hopper']==l['CPU_reserved_s']['Hopper']-54000
    assert v['optimizer_updates']==1024 and v['neural_queries']==29 and len(v['PBS_submissions'])==1
    assert v[a.PACKAGE]==g[a.PACKAGE] and not proposal['native_IO_performed']
    with pytest.raises(PermissionError):h.project_training_to_LOCAL(local_bytes=a.raw(v),global_bytes=gb,terminal_capture=c,arm='C50')


def test_cannot_drop_the_other_arm_or_GEN_resource_pool():
    g,l,c=fixture();l[a.PACKAGE]['remaining_resources']['Core_CPU_s']-=1200
    with pytest.raises(PermissionError):h.project_training_to_LOCAL(local_bytes=a.raw(l),global_bytes=a.raw(g),terminal_capture=c,arm='C50')
