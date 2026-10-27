"""Captured original C50 proof; future PBS/closed Core values are CPU fixtures."""
import base64,gzip,json,os
from pathlib import Path
from copy import deepcopy
import pytest
from scripts.pi0294.generalization import minimal_execution as e,minimal_package as p,minimal_adoption as a,minimal_measurement as measurement
from scripts.pi0294.pv1 import mix_accounting as old,mix_core_accounting as core

@pytest.fixture
def actual():
    path=os.environ.get('ARX5_FIRST512_MEASUREMENT_CAPTURE')
    if not path:pytest.skip('explicit captured canonical GLOBAL + actual valid C50 condition required')
    v=json.loads(Path(path).read_bytes());return json.loads(gzip.decompress(base64.b64decode(v['GLOBAL_gzip_base64']))),base64.b64decode(v['measurement_capture_base64'])

def request(g,cap):
    m=a.validate(g);v=e.native_resources()
    return dict(schema='ARX5_MINIMAL_FIRST512_NATIVE_REQUEST_V1',scope=e.scope.SCOPE,
        platform_training_version=e.VERSION,native_platform='VANDA_A40',profile=e.PROFILE,
        project='CFP04-CF-007',job_name='pi0294_dec1_a40_train',
        raw_parent_files_identity=e.training.RAW_PARENT_FILES_ID,statistics_sha256=e.R0,
        panel_sha256=e.PANEL_SHA,full_cursor_sha256=e.CURSOR_SHA,scheduler_horizon=1024,
        formal_terminal_steps=512,optimizer_update_limit=1028,query_limit=147,model_load_limit=11,
        expected_native_start_step=0,physical_GPU_count=1,physical_GPU_kind='NVIDIA_A40',active_cpu_limit=8,
        walltime_s=59400,accounting_pad_s=120,reserved_H200_s=59520,CPU_upper_s=v['Hopper_CPU_s'],
        additional_output_upper_bytes=v['Hopper_storage_bytes'],scope_grant_sha256=a.GRANT_SHA,
        conditional1024_authorized=False,new_physics=0,new_constructors=0,session_plan=e.native_sessions(),
        reserved_session_plan_sha256=a.sha(a.raw(p.sessions())),
        native_session_plan_version='MATCHED15_AND_ACTUAL_OFFICIAL_EXPORT_WITHIN147_V1',
        request_id='12345678-1234-4234-8234-123456789abc',submission_nonce='12345678-1234-4234-8234-123456789abd',
        source_sha='f'*40,run_manifest_sha256='a'*64,cold_input_audit_sha256='b'*64,
        raw_parent_audit_plan_sha256='c'*64,frozen_input_packet_sha256='d'*64,Core_measurement_capture_sha256=a.sha(cap),
        T_CD=m['T_CD'],deadline_epoch=m['deadline_epoch'],stop_new_training_epoch=m['stop_new_training_epoch'])

def closed_CPU_fixture(g):
    v=deepcopy(g)
    # Explicit future fixture. This is NOT a real Core close/lease/OS witness.
    # Retain the captured request/process/counters and all original holds.
    for row in v[core.FIELD].values():
        if row['status']!='NATIVE_TERMINAL':
            row.update(status='NATIVE_TERMINAL',native_exit_status=0,reserved_neural_queries=0,unknown_query_intents=[])
    return v


def test_actual_valid_failure_is_measurement_success_without_task_success_gate(actual):
    g,cap=actual;proof=measurement.validate(g,cap)
    assert proof['valid_conditions']==1 and proof['success_rate_is_not_a_gate']
    assert not proof['usable_as_training_input']

@pytest.mark.parametrize('change',['adjudication','causal','native600','checkpoint','hash'])
def test_prepared_flags_or_engineering_do_not_replace_actual_measurement(actual,change):
    g,cap=deepcopy(actual);v=json.loads(cap)
    name={'adjudication':'adjudication','causal':'causal_chain','native600':'model_ready','checkpoint':'model_ready','hash':'result'}[change]
    if change=='hash':v['files'][name]['sha256']='a'*64
    else:
        doc=json.loads(base64.b64decode(v['files'][name]['base64']))
        if change=='adjudication':doc['status']='ENGINEERING_INCOMPLETE'
        elif change=='causal':doc['executed_ACKs']=0
        elif change=='native600':doc['native_pre_model_qualification']['heartbeat_elapsed_s']=599
        else:doc['identity']['checkpoint_manifest_sha256']='a'*64
        raw=a.raw(doc);v['files'][name]=dict(bytes=len(raw),sha256=a.sha(raw),base64=base64.b64encode(raw).decode())
    with pytest.raises((PermissionError,ValueError)):measurement.validate(g,a.raw(v))


def test_live_sealed_Core_owner_is_not_silently_given_foreign_PBS(actual):
    g,cap=actual;r=request(g,cap)
    with pytest.raises(PermissionError,match='COLD_SERIAL_BOUNDARY'):
        e.register(a.raw(g),request_bytes=a.raw(r),Core_measurement_capture_bytes=cap,now_epoch=1793131100.)


def test_first_main_reuses_original_reservation_and_once_control_only(actual):
    g,cap=actual;g=closed_CPU_fixture(g);r=request(g,cap);out=e.register(a.raw(g),request_bytes=a.raw(r),Core_measurement_capture_bytes=cap,now_epoch=1793131100.);v=out['proposed_global']
    assert v['reserved']==g['reserved'] and v['CPU_reserved_s']==g['CPU_reserved_s']-3600
    assert v['CPU_s']==g['CPU_s']+3600 and v['used']['PBS_submissions']==g['used']['PBS_submissions']+1
    assert v['used']['H200_s']==g['used']['H200_s']and v[old.PACKAGE]==g[old.PACKAGE]
    assert v['PBS_submissions'][:-1]==g['PBS_submissions']and v['unknown_effects']==g['unknown_effects']and v['pending_updates']==g['pending_updates']
    assert e.validate(v)['control_paid_once']==e.CONTROL and len(e.rows(v))==1
    with pytest.raises(PermissionError):e.register(a.raw(v),request_bytes=a.raw(r),Core_measurement_capture_bytes=cap,now_epoch=1793131101.)
    bound=e.bind(v,r['request_id'],'12345.stdct-mgmt-02',now_epoch=1793131102.)
    assert e.rows(bound)[0]['status']==e.STATES[1]
    with pytest.raises(PermissionError):e.bind(bound,r['request_id'],'12345.stdct-mgmt-02',now_epoch=1793131103.)

@pytest.mark.parametrize('change',['unknown','pending','row','clock','request','new-cap','new-1024'])
def test_unknown_or_arbitrary_recipe_or_budget_not_registered(actual,change):
    g,cap=actual;g=closed_CPU_fixture(g);r=request(g,cap)
    if change=='unknown':g['unknown_effects']=[{'preserved':True}]
    elif change=='pending':g['pending_updates']=[{'preserved':True}]
    elif change=='row':r['platform_training_version']='V5_CD_NATIVE'
    elif change=='clock':r['T_CD']+=1
    elif change=='request':r['request_id']='old-not-uuid'
    elif change=='new-cap':r['query_limit']+=1
    else:r['conditional1024_authorized']=True
    before=deepcopy(g)
    with pytest.raises(PermissionError):e.register(a.raw(g),request_bytes=a.raw(r),Core_measurement_capture_bytes=cap,now_epoch=1793131100.)
    assert g==before


def test_control_LOCAL_projection_is_additive_idempotent_and_not_a_native_write(actual):
    g,cap=actual;g=closed_CPU_fixture(g);r=request(g,cap);v=e.register(a.raw(g),request_bytes=a.raw(r),Core_measurement_capture_bytes=cap,now_epoch=1793131100.)['proposed_global']
    local=dict(CPU_s=dict(local_PC=100.),network_transfer_accounted_bytes=123,
               CPU_reserved_s=dict(local_PC=999.),other_history={'must':'stay'})
    before=deepcopy(local);out=e.budget_LOCAL(v,local)
    assert local==before and out['CPU_s']['local_PC']==730.
    assert out['network_transfer_accounted_bytes']==123+e.CONTROL['transfer_bytes']
    assert out['CPU_reserved_s']==before['CPU_reserved_s']and out['other_history']==before['other_history']
    assert e.budget_LOCAL(v,out)==out
    assert out['ARX5_minimal_first512_control_projection']['canonical_LOCAL_write_proven']is False


def test_registered_row_cannot_rewrite_original_request_bytes(actual):
    g,cap=actual;g=closed_CPU_fixture(g);r=request(g,cap);v=e.register(a.raw(g),request_bytes=a.raw(r),Core_measurement_capture_bytes=cap,now_epoch=1793131100.)['proposed_global']
    v['PBS_submissions'][-1]['submission_nonce']='12345678-1234-4234-8234-123456789abe'
    with pytest.raises(PermissionError,match='ACTUAL_REQUEST_BYTES'):e.validate(v)


@pytest.mark.parametrize('change',[None,'chunk','slots','wall','GPU','memory','legacy'])
def test_original_job_reader_has_one_explicit_new_capacity_profile(actual,change):
    g,cap=actual;r=request(g,cap)
    limits=dict(select='1:ncpus=8:mem=64gb:ngpus=1:gpu_model=A40',ncpus=8,mem='64gb',ngpus=1,walltime='16:30:00')
    if change=='chunk':limits['select']+='+'+limits['select']
    elif change=='slots':limits['select']=limits['select'].replace('ncpus=8','ncpus=36');limits['ncpus']=36
    elif change=='wall':limits['walltime']='16:31:00'
    elif change=='GPU':limits['select']=limits['select'].replace('A40','H200')
    elif change=='memory':limits['mem']='128gb'
    elif change=='legacy':r['platform_training_version']='OLD_C_D_TRAINING'
    if change:
        with pytest.raises(PermissionError):e.select_profile(limits,r)
    else:assert e.select_profile(limits,r)==(8,64*1024**3)


def test_native_query_plan_keeps_all_matched_scientific_probes_and_original_escrow():
    before=p.sessions();after=e.native_sessions()
    assert [r['query_limit']for r in before]==[41,17,17,36,36]
    assert [r['query_limit']for r in after]==[39,17,17,37,37]
    assert sum(r['query_limit']for r in after)==sum(r['query_limit']for r in before)==147
    for x,y in zip(before,after):
        assert {k:v for k,v in x.items()if k!='query_limit'}=={k:v for k,v in y.items()if k!='query_limit'}


def test_full_original_neural_entry_and_all_fourteen_future_dimensions_are_conserved(actual):
    from scripts.pi0294.pv1.neural_ledger import check
    from scripts.pi0294.generalization.minimal_scope import first512_forecast
    g,cap=actual;g=closed_CPU_fixture(g)
    root=Path(os.environ['ARX5_FIRST512_MEASUREMENT_CAPTURE']).parent
    p=root/'CORE_RETRY12_CONTINUATION_INDEPENDENT_DUAL_READBACK.private.json'
    if not p.exists():pytest.skip('explicit captured canonical LOCAL witness required')
    row=json.loads(p.read_bytes())['local'];lb=gzip.decompress(base64.b64decode(row['LOCAL_gzip_base64']))
    assert a.sha(lb)==row['LOCAL_sha256']
    before=first512_forecast(a.raw(g),lb);r=request(g,cap)
    out=e.register(a.raw(g),request_bytes=a.raw(r),Core_measurement_capture_bytes=cap,now_epoch=1793131100.)['proposed_global']
    assert check(out,allow_unknown=True)==out
    after=first512_forecast(a.raw(out),lb)
    assert before['cap_changes_required']==after['cap_changes_required']=={}
    assert len(before['dimensions'])==14
    assert {k:v['projected']for k,v in before['dimensions'].items()}=={k:v['projected']for k,v in after['dimensions'].items()}
    # A request-shaped foreign row without its original adopted typed index
    # cannot be excluded from the old source-arm validator.
    bad=deepcopy(out);bad.pop(e.FIELD)
    with pytest.raises(PermissionError):check(bad,allow_unknown=True)
