"""CPU fixtures exercise actual original parsers, no native GPU qualification."""
from copy import deepcopy
import pytest
from scripts.pi0294.test_v5_core_stage_accounting import paid, record, started
from scripts.pi0294.test_pv1_core_evaluation_identity import DEVICE, GPU
from scripts.pi0294.pv1 import mix_core_identity as n, mix_core_accounting as c
from scripts.pi0294.pv1 import mix_accounting as a, successor_cd as s
from scripts.pi0294.pv1 import evaluation_core_identity as original


def request():
    g, _, capture = paid(); row = record(g, capture)
    r = {k:v for k,v in row.items() if k not in ('status','native_handle','reserved_neural_queries',
        'request_sha256','reserved_optimizer_updates','optimizer_update_limit','reserved_GPU_s',
        'additional_output_upper_bytes','query_limit','owner_uid','hostname','physical_GPU_count',
        'physical_GPU_kind','PBS_submissions_consumed','accounting_pad_s','charges_DEC1_subcaps',
        'legacy_GPU_seconds_field_is_projection','new_robot_physics','child_run_uuid')}
    r.update(schema='V5_CD_CORE_NATIVE_GPU_REQUEST_V1', task=s.TASK, run_uuid='v5-c50-neural-fixture',
        allocation_number=1, native_unit='v5-c50-neural-fixture-neural.service', optimizer_updates=0,
        Teacher_takeover=False, normalizer_refit=False, evaluation_data_for_training=False,
        LOCKBOX_opened=False, numerical_mode='DETERMINISTIC_CUDA_MATH_SDP_V1', GPU_uuid=GPU)
    m = dict(schema='PI0294_PV1_NEURAL_CHILD_RUN_MANIFEST_V1', scope=c.ROLE,
        platform_evaluation_version=c.VERSION, owner='EXISTING_LOCAL_PC', owner_user='liuyu-nfs',
        owner_uid=2009, run_uuid=r['run_uuid'], code=dict(vla_commit=r['source_sha']),
        authority=dict(protocol_ref=s.TASK_SOURCE, scope_grant_sha256=s.GRANT_SHA),
        execution=dict(profile=c.PROFILE, host='corelab-4080s', uid=2009),
        limits=dict(T_CD=r['T_CD'], operation_deadline_epoch=r['deadline_epoch']))
    return m, r


def allocation():
    m, r = request(); p = started(dict(r, request_sha256='f'*64))
    p['kernel_limits'] = original.kernel_limits('400000 100000',str(48*1024**3),'64')
    p['native_unit_properties'] = dict(MainPID=str(p['pid']), ActiveState='active', RuntimeMaxUSec='8h', ControlGroup='/fixture')
    p['native_GPU_raw'] = dict(devices=dict(stdout=DEVICE, returncode=0), processes=dict(stdout='', returncode=0))
    p['device'] = original.parse_GPU_rows(DEVICE, '', GPU_uuid=GPU)
    v = dict(status=n.STATUS, platform_evaluation_version=c.VERSION, run_uuid=r['run_uuid'],
        manifest_sha256=r['run_manifest_sha256'], request_sha256='f'*64, job_id=p['native_handle'],
        native_handle=p['native_handle'], model_loaded=False, new_optimizer_updates=0,
        native_process=p, native_process_sha256=c.digest(p), hostname='corelab-4080s',
        allocation_start_epoch=p['started_epoch'], allocation_end_epoch=p['started_epoch']+28800)
    for name in ('source_sha','request_id','submission_nonce','stage','evaluation_candidate',
            'checkpoint_manifest_sha256','statistics_sha256','native_platform','allocation_number'):
        v[name] = r[name]
    return m, r, v


def check(m,r,v):
    return original.sealed_identity(m,r,v,run_uuid=r['run_uuid'],
        manifest_sha256=r['run_manifest_sha256'],request_sha256='f'*64)


def test_explicit_CD_request_and_actual_original_device_parsers_join():
    m,r,v = allocation()
    assert n.request_identity(m,r,r['run_manifest_sha256']) == r
    assert original.cold_profile(m) and check(m,r,v) == v
    assert n.require_device(r,v,visible=GPU,count=1,name='NVIDIA GeForce RTX 4080 SUPER',memory_bytes=32760*1024**2)


@pytest.mark.parametrize('field,value', [('evaluation_candidate','A_CONTROL'),('checkpoint_step',4096),
    ('stage','P4_DEV'),('statistics_sha256','x'),('allocation_number',2),
    ('Teacher_takeover',True),('optimizer_updates',False),('shared_GPU_grant_sha256','x'),
    ('startup_recovery_sha256','x'),('queue_dispatch_mode','x'),('PBS_JOBID','99.stdct-mgmt-02'),
    ('request_sha256','a'*64),('original_T0_epoch',1792348196),('deadline_epoch',2000.)])
def test_changed_science_legacy_or_partial_identity_is_rejected_before_native_reads(field,value):
    m,r = request();r[field]=value
    with pytest.raises(PermissionError): n.request_identity(m,r,r['run_manifest_sha256'])


@pytest.mark.parametrize('field,value', [('native_process_sha256','0'*64),('job_id','99.stdct-mgmt-02'),
    ('status','PASS_PV1_OWN_NATIVE_CORELAB_GPU_ALLOCATION'),('new_optimizer_updates',False),
    ('allocation_end_epoch',999999.),('checkpoint_manifest_sha256','0'*64)])
def test_old_pass_prepared_or_modified_native_receipt_never_qualifies(field,value):
    m,r,v=allocation();v[field]=value
    with pytest.raises(PermissionError):check(m,r,v)


def test_raw_foreign_GPU_use_and_kernel_limits_still_block_CD():
    for change in ('foreign', 'short-memory', 'wrong-pid', 'failed-read'):
        m,r,v=allocation();p=v['native_process']
        if change=='foreign':p['native_GPU_raw']['processes']['stdout']=GPU+', 999, independent-work, 5000\n'
        elif change=='short-memory':p['kernel_limits']['memory_max_bytes']=16*1024**3
        elif change=='wrong-pid':p['native_unit_properties']['MainPID']='999'
        else:p['native_GPU_raw']['devices']['returncode']=1
        v['native_process_sha256']=c.digest(p)
        with pytest.raises(PermissionError):check(m,r,v)


def test_real_new_window_does_not_reopen_old_AB_or_reset_C50_clock():
    _,r=request();n.check_clock(r,started_epoch=1203.,now_epoch=1205.)
    for start,now in [(999.,1205.),(1203.,1202.),(r['deadline_epoch']-28860,1205.),
                      (1203.,r['deadline_epoch']),(float('nan'),1205.),(True,1205.)]:
        with pytest.raises(PermissionError):n.check_clock(r,started_epoch=start,now_epoch=now)
    m,r=request();m['scope']='PI0294_PV1_SHARED_POLICY_CORELAB_EVALUATION_V1'
    with pytest.raises(PermissionError):original.cold_profile(m)


def test_device_gate_reuses_existing_neural_entry_without_loading_torch():
    from types import SimpleNamespace
    from unittest.mock import patch
    from scripts.pi0294.pv1.neural import require_native_device
    _,r,v=allocation()
    fake=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:1,
        get_device_name=lambda i:'NVIDIA GeForce RTX 4080 SUPER',
        get_device_properties=lambda i:SimpleNamespace(total_memory=32760*1024**2)))
    with patch.dict('os.environ',CUDA_VISIBLE_DEVICES=GPU): require_native_device(r,v,fake)
    fake.cuda.get_device_properties=lambda i:SimpleNamespace(total_memory=16*1024**3)
    with patch.dict('os.environ',CUDA_VISIBLE_DEVICES=GPU):
        with pytest.raises(PermissionError):require_native_device(r,v,fake)


def test_real_terminal_bytes_are_required_even_with_a_saved_inference_view():
    import base64
    from scripts.pi0294.pv1.mix_core_context import captured_endpoint
    from scripts.pi0294.pv1.mix_stage_handoff import closed_endpoint
    g,_,capture=paid();proof=closed_endpoint(global_bytes=a.raw(g),arm='C50',**capture)
    value=dict(schema='V5_CD_REAL_TRAINING_TERMINAL_BYTES_V1',arm='C50',
        global_base64=base64.b64encode(a.raw(g)).decode(),handoff=proof,
        terminal_capture_base64={k:base64.b64encode(v).decode()for k,v in capture.items()})
    assert captured_endpoint(value)==proof
    value['handoff']['checkpoint_manifest_sha256']='0'*64
    with pytest.raises(PermissionError):captured_endpoint(value)
    value['handoff']=proof;value['terminal_capture_base64'].pop('OS_exit_bytes')
    with pytest.raises(PermissionError):captured_endpoint(value)


def test_new_scope_uses_existing_wire_protocol_and_finite_control_without_old_authority():
    import re
    from scripts.pi0294.pv1.evaluation_platform import fixture_job_pattern,file_RPC_profile
    from scripts.pi0294.pv1.control_budget import validate, PROFILE
    from scripts.pi0294.pv1.mix_evaluation import stage
    from scripts.pi0294.pv1.successor_core_context import EVAL_ROLE
    m,r=request();m['execution'].update(control_budget_profile=PROFILE,sealed_inputs={
        'V5_CD_SCOPE_BINDING.json':'a'*64,'plans/V5_CD_PAIRED_EVALUATION_PLAN.json':'b'*64})
    physical=deepcopy(m);physical.update(scope=EVAL_ROLE,candidate='C50',evaluation_peer_platform=c.VERSION)
    physical.pop('platform_evaluation_version');physical['execution']['profile']='4080-simulation'
    for role,v in [('core',physical),('hopper',m)]:
        pattern=fixture_job_pattern(v,role)
        assert re.fullmatch(pattern,'coregpu:22222222-2222-4222-8222-222222222222:2000:12345')
        assert not re.fullmatch(pattern,'99.stdct-mgmt-02')
        assert file_RPC_profile(v,role)==('4080-simulation'if role=='core'else c.PROFILE)
        budget=validate(v,stage('C50'))
        assert budget['poll_s']==.25 and budget['max_calls']==150000 and budget['CPU_limit_s']==7200
        v['authority']['protocol_ref']='old'
        with pytest.raises(PermissionError):validate(v,stage('C50'))


def test_service_reuses_original_whole_panel_and_rejects_repeating_valid_failures():
    from scripts.pi0294.test_pv1_evaluation_plan import groups
    from scripts.pi0294.pv1.mix_evaluation import candidate_plan
    from scripts.pi0294.pv1.neural import evaluation_labels
    _,r=request();p=candidate_plan(groups(),'C50')
    assert evaluation_labels(r,p,[])==('C50-DEV-N294101',)
    spec=p['panels']['C50-DEV-N294101'];card=spec['cards'][0]
    prior=[dict(model='C50',condition_id=card['condition_id'],attempt_id=card['id'],
        status='VALID_TASK_FAIL',failure_stage='S1_PRE_EXECUTION_REJECT',pipeline_trustworthy=True,live_native_failures=0)]
    with pytest.raises(PermissionError):evaluation_labels(r,p,prior)
    p['panels']['C50-DEV-N294102']=deepcopy(spec)
    with pytest.raises(PermissionError):evaluation_labels(r,p,[])
