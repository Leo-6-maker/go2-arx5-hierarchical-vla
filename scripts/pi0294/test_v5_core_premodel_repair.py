"""Exact captured zero-model failure; pure proposals, never native IO."""
from copy import deepcopy
import base64,gzip,json,os,time,uuid
from pathlib import Path
import pytest
from scripts.pi0294.pv1 import mix_core_premodel_repair as r,mix_core_accounting as c,mix_accounting as a


@pytest.fixture
def captured():
    base=Path(os.environ.get('PI0294_PRIVATE_WORKSPACE','/home/liuyu/ISSAC-GR00T'))/'work/private/v5-execution-20261025'
    path=base/'C50_PREMODEL_REPAIR_ACTUAL_FULL_BUDGET_CPU_PREPARED.private.json'
    if not path.is_file():pytest.skip('private actual native failure/closure/metadata bytes unavailable')
    v=json.loads(path.read_bytes());return dict(global_bytes=base64.b64decode(v['GLOBAL_base64']),local_bytes=base64.b64decode(v['LOCAL_base64']),closure_capture_bytes=a.raw(v['closure_capture']),metadata_capture_bytes=a.raw(v['metadata_capture']),replacement_request_id=str(uuid.uuid4()),replacement_submission_nonce=str(uuid.uuid4()),now_epoch=time.time())


def test_exact_actual_fees_and_gross_IO_retained_while_only_unused_compute_is_reescrowed(captured):
    original=deepcopy(captured);out=r.propose(**captured);g,l=json.loads(captured['global_bytes']),json.loads(captured['local_bytes']);v,w=out['proposed_global'],out['proposed_LOCAL'];assert captured==original
    for key in('used','CPU_s','new_storage_current_bytes','new_storage_peak_bytes','PBS_submissions',c.FIELD):assert v[key]==g[key]
    for key in('CPU_s','GPU_allocated_s','network_transfer_accounted_bytes','physics_ticks','episode_starts','engine_starts','Core_native_GPU_current_storage_bytes'):assert w[key]==l[key]
    assert out['full_remaining_proof']['shortfalls']=={}and v[a.PACKAGE]['completed_stages']==['TRAIN:C50:1024']
    assert out['remaining_compute_reescrow']['H200_s']==28860 and out['remaining_compute_reescrow']['Core_CPU_s']==173900
    for key in('Hopper_storage_bytes','Core_storage_bytes','transfer_bytes'):assert out['remaining_compute_reescrow'][key]==0
    assert r.validated(v)==r.validated(w)
    with pytest.raises(PermissionError):r.propose(**dict(captured,global_bytes=a.raw(v),local_bytes=a.raw(w)))


@pytest.mark.parametrize('case',range(17))
def test_wrong_effects_closure_identity_metadata_and_paid_bound_cannot_get_credit(captured,case):
    v=deepcopy(captured);g=json.loads(v['global_bytes']);cap=json.loads(v['closure_capture_bytes']);meta=json.loads(v['metadata_capture_bytes'])
    if case==0:cap['exact_cause']['all_other_registry_bindings_match']=False
    if case==1:cap['exact_cause']['old_Pump_expected']=cap['exact_cause']['actual_CD_expected']
    if case==2:cap['units']['owner']['MainPID']='123'
    if case==3:cap['units']['GPU']['ExecMainStatus']='0'
    if case==4:cap['status']='invented_pass'
    if case==5:cap['request_id']=str(uuid.uuid4())
    if case in(6,7,8,9,10,11):
        key={6:'OWNER_RESULT',7:'PHYSICAL_LEDGER',8:'DUAL_FINANCE',9:'GPU_LEASE',10:'NATIVE_FINAL',11:'physical_LEASE'}[case]
        p=json.loads(base64.b64decode(cap['files'][key]['base64']))
        if case==6:p['effects']['model_load_attempts']=1
        if case==7:p['shadow_integrations']=1
        if case==8:p['transfer_charged_upper_bytes']-=1
        if case==9:p['status']='ACTIVE'
        if case==10:p['phase']='C50-MODEL_LOAD'
        if case==11:p['terminal_native_sha256']='0'*64
        cap['files'][key]['base64']=base64.b64encode(a.raw(p)).decode()
    if case==12:meta['unmodified_model_input_inodes_excluded']=2
    if case==13:meta['unique_logical_metadata_bytes']=128*1024**2
    if case==14:meta['unique_allocated_metadata_bytes']=128*1024**2
    if case==15:meta['root']='/wrong'
    if case==16:v['replacement_request_id']=r.FAILED
    v.update(closure_capture_bytes=a.raw(cap),metadata_capture_bytes=a.raw(meta))
    with pytest.raises((PermissionError,KeyError,ValueError)):r.propose(**v)


def test_paid_IO_cannot_be_used_by_D25_or_another_request(captured):
    out=r.propose(**captured);g=out['proposed_global'];m=out['marker'];row=dict(g[c.FIELD][r.FAILED],request_id=m['replacement_request_id'],submission_nonce=m['replacement_submission_nonce'],child_run_uuid=m['replacement_run_uuid'],prepaid_panel_IO_credit_sha256=a.sha(a.raw(m)))
    assert r.output_credit(g,row)==c.remaining_stage()['Hopper_storage_bytes']
    for change in({'evaluation_candidate':'D25'},{'request_id':str(uuid.uuid4())},{'prepaid_panel_IO_credit_sha256':'0'*64},{'child_run_uuid':'different_root'}):
        with pytest.raises(PermissionError):r.output_credit(g,dict(row,**change))
