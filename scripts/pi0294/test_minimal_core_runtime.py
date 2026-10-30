"""Pure512 identities and real temporary onceIO; GPU facts are CPU fixtures."""
from copy import deepcopy
import json
from types import SimpleNamespace
import pytest

from scripts.pi0294.generalization import minimal_core_accounting as c,minimal_core_delegation as d
from scripts.pi0294.pv1 import mix_core_delegation as protocol,mix_core_identity as identity
from scripts.pi0294.pv1 import evaluation_core_identity as hardware,mix_core_delegation_store as store
from scripts.pi0294.pv1.mix_core_accounting import digest as native_digest
from scripts.pi0294.test_pv1_core_evaluation_identity import DEVICE,GPU


def fixture():
    r=dict(c.request_fields(),request_id='12345678-1234-4234-8234-123456789abc',
        submission_nonce='12345678-1234-4234-8234-123456789abd',run_uuid='arx5-e1-fixed512-n294101-01',
        native_unit='arx5-e1-fixed512-n294101-01-neural.service',source_sha='a'*40,
        cold_input_audit_sha256='b'*64,planned_panel_sha256='c'*64,training_handoff_sha256=c.digest(c.endpoint_reference()),
        GPU_uuid=GPU,T_CD=1000.,deadline_epoch=c.campaign.scope(1000.)['deadline_epoch'],created_epoch=1201.)
    m=dict(schema='PI0294_PV1_NEURAL_CHILD_RUN_MANIFEST_V1',scope=c.ROLE,platform_evaluation_version=c.VERSION,
        owner='EXISTING_LOCAL_PC',owner_user='liuyu-nfs',owner_uid=2009,run_uuid=r['run_uuid'],code=dict(vla_commit=r['source_sha']),
        authority=dict(protocol_ref=c.campaign.TASK_SOURCE,scope_grant_sha256=c.GRANT_SHA),
        execution=dict(profile=c.PROFILE,host='corelab-4080s',uid=2009,global_accounting_mode=protocol.MODE,
                       sealed_inputs={c.GRANT_FILE:c.GRANT_SHA}),
        limits=dict(T_CD=r['T_CD'],operation_deadline_epoch=r['deadline_epoch']))
    r['run_manifest_sha256']=c.digest(m);rs=c.digest(r);boot='22222222-2222-4222-8222-222222222222';pid=2000;start=12345
    from scripts.pi0294.pv1.evaluation_core_accounting import process_handle
    handle=process_handle(boot,pid,start)
    p=dict(schema='PV1_CORE_GPU_NATIVE_PROCESS_STARTED_V1',hostname='corelab-4080s',uid=2009,GPU_uuid=GPU,
        request_id=r['request_id'],request_sha256=rs,submission_nonce=r['submission_nonce'],run_manifest_sha256=c.digest(m),
        source_sha=r['source_sha'],process_state='RUNNING',native_handle=handle,native_model_permission=False,
        physics_authorized=False,started_epoch=1203.,observed_epoch=1205.,boot_id=boot,pid=pid,proc_start=start,
        kernel_limits=hardware.kernel_limits('400000 100000',str(48*1024**3),'64'),
        native_unit_properties=dict(MainPID=str(pid),ActiveState='active',RuntimeMaxUSec='8h',ControlGroup='/fixture'),
        native_GPU_raw=dict(devices=dict(stdout=DEVICE,returncode=0),processes=dict(stdout='',returncode=0)),
        device=hardware.parse_GPU_rows(DEVICE,'',GPU_uuid=GPU))
    allocation=dict(status=identity.STATUS,platform_evaluation_version=c.VERSION,run_uuid=r['run_uuid'],
        manifest_sha256=c.digest(m),request_sha256=rs,job_id=handle,native_handle=handle,model_loaded=False,
        new_optimizer_updates=0,native_process=p,native_process_sha256=native_digest(p),hostname=p['hostname'],
        allocation_start_epoch=p['started_epoch'],allocation_end_epoch=p['started_epoch']+28800)
    for name in ('source_sha','request_id','submission_nonce','stage','evaluation_candidate',
                 'checkpoint_manifest_sha256','statistics_sha256','native_platform','allocation_number'):
        allocation[name]=r[name]
    row=dict(r,request=deepcopy(r),request_sha256=rs,child_run_uuid=r['run_uuid'],hostname=p['hostname'],owner_uid=2009,
        reserved_GPU_s=28860,query_limit=2097,reserved_neural_queries=2097,
        additional_output_upper_bytes=c.HOLD['Hopper_storage_bytes'],status=c.STATES[1],native_handle=handle,
        boot_id=boot,pid=pid,proc_start=start,process_started_epoch=1203.,control_payment=deepcopy(c.CONTROL))
    src={k:'e'*64 for k in protocol._INPUTS};src.update(manifest_bytes=c.digest(m),request_bytes=rs)
    admission=dict(schema='ARX5_MINIMAL_CORE_FIXED512_FULL_RESOURCE_RECOMPUTATION_V1',request_id=r['request_id'],
        request_sha256=rs,GLOBAL_sha256=src['global_bytes'],LOCAL_sha256=src['local_bytes'],
        snapshot_sha256=src['snapshot_bytes'],shortfalls={},all_remaining_costs_fit=True,
        original_matrix_and_final_protected=True,model_permission=False,physics_authorized=False,allocation=allocation)
    b=dict(mode=protocol.MODE,request=r,manifest=m,issued_row=row,admission=admission,source_inputs=src,
        issued_epoch=1206.,execution_not_after_epoch=allocation['allocation_end_epoch']-2700,
        accounting_not_after_epoch=allocation['allocation_end_epoch'],
        limits=dict(generation_queries=2097,model_loads=2,CPU_s=80000,GPU_s=28860,
                    output_bytes=c.HOLD['Hopper_storage_bytes'],optimizer_updates=0),original_GLOBAL_revision=1)
    grant=c.raw(dict(schema=d.GRANT,body=b,body_sha256=c.digest(b)))
    return m,r,p,allocation,row,grant


def test_raw512_reuses_actual_original_native_identity_device_parsers():
    m,r,p,a,row,grant=fixture()
    assert identity.request_identity(m,r,c.digest(m))==r
    assert hardware.cold_profile(m)
    assert identity.sealed_identity(m,r,a,run_uuid=r['run_uuid'],manifest_sha256=c.digest(m),request_sha256=c.digest(r),same_bound_process=row)==a
    assert protocol._body(grant)['issued_row']==row


def shared_fixture():
    from scripts.pi0294.generalization import minimal_core_sharing as sharing
    m,r,*_=fixture();r.update(request_id=sharing.RID,submission_nonce=sharing.NONCE,
        run_uuid=sharing.RUN,native_unit=sharing.RUN+'-neural.service',
        shared_GPU_grant_sha256=sharing.GRANT_SHA)
    m['run_uuid']=r['run_uuid'];m['execution']['sealed_inputs'][sharing.PLAN]=sharing.GRANT_SHA
    r['run_manifest_sha256']=c.digest(m)
    return m,r


def test_exact_direct_GPU0_shared_device_keeps_foreign_rows_and_original_total_memory_gate():
    from scripts.pi0294.generalization import minimal_core_sharing as sharing
    m,r=shared_fixture();identity.request_identity(m,r,c.digest(m))
    processes=GPU+', 4101246, /foreign/python, 5770\n'
    device=hardware.parse_GPU_rows(DEVICE,processes,GPU_uuid=GPU,shared_request=r)
    assert device['human_authorized_shared_GPU_grant_sha256']==sharing.GRANT_SHA
    with pytest.raises(PermissionError):hardware.parse_GPU_rows(DEVICE,processes,GPU_uuid=GPU)
    low=DEVICE.replace('32760','16376')
    with pytest.raises(PermissionError):hardware.parse_GPU_rows(low,processes,GPU_uuid=GPU,shared_request=r)
    with pytest.raises(PermissionError):hardware.parse_GPU_rows(DEVICE,processes.replace('5770','unknown'),GPU_uuid=GPU,shared_request=r)


@pytest.mark.parametrize('key,value',[('request_id','12345678-1234-4234-8234-123456789abc'),
    ('GPU_uuid','GPU-845348f1-1511-dc19-0877-cc97658e3edd'),('evaluation_candidate','D25'),
    ('optimizer_updates',1),('shared_GPU_grant_sha256','0'*64)])
def test_direct_share_cannot_authorize_another_request_device_or_scientific_change(key,value):
    m,r=shared_fixture();r[key]=value
    with pytest.raises(PermissionError):identity.request_identity(m,r,c.digest(m))


def test_direct_share_requires_real_manifest_seal_and_all_original_cold_owner_seams():
    from scripts.pi0294.generalization import minimal_core_sharing as sharing,minimal_core_management as manager
    m,r=shared_fixture();del m['execution']['sealed_inputs'][sharing.PLAN]
    with pytest.raises(PermissionError):identity.request_identity(m,r,c.digest(m))
    assert "shared_request=r if 'shared_GPU_grant_sha256'in r else None"in manager.PROGRAMS['prestart']


@pytest.mark.parametrize('key,value',[('checkpoint_step',1024),('optimizer_updates',False),('evaluation_candidate','D25'),
    ('statistics_sha256','0'*64),('checkpoint_manifest_sha256','0'*64),('shared_GPU_grant_sha256','a'*64),
    ('parent_checkpoint_manifest_sha256','a'*64),('native_unit','foreign-neural.service')])
def test_no_alias_to_CORR1024_or_extra_training_or_another_scope(key,value):
    m,r,*_=fixture();r[key]=value
    with pytest.raises(PermissionError):identity.request_identity(m,r,c.digest(m))


def test_real_neural_and_transport_routers_select512_and_preserve_device_gate(tmp_path,monkeypatch):
    from scripts.pi0294.pv1 import neural,evaluation_core_runtime,evaluation_platform as platform
    m,r,p,a,_,_=fixture();path=tmp_path/'REQUEST.json';path.write_bytes(c.raw(r));sentinel=object()
    monkeypatch.setattr(evaluation_core_runtime,'run',lambda root,request_path:sentinel)
    assert neural.run(tmp_path,path) is sentinel
    fake=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:1,get_device_name=lambda i:'NVIDIA GeForce RTX 4080 SUPER',
        get_device_properties=lambda i:SimpleNamespace(total_memory=32760*1024**2)))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',GPU);neural.require_native_device(r,a,fake)
    fake.cuda.get_device_properties=lambda i:SimpleNamespace(total_memory=16*1024**3)
    with pytest.raises(PermissionError):neural.require_native_device(r,a,fake)
    assert platform.file_RPC_profile(m,'hopper')==c.PROFILE
    assert platform.fixture_job_pattern(m,'hopper').startswith('coregpu:')
    physical=dict(m,scope=c.PHYSICAL_ROLE,candidate='E1',execution=dict(profile='4080-simulation'))
    assert platform.fixture_job_pattern(physical,'core').startswith('coregpu:')


def setup_store(tmp_path):
    m,r,p,a,row,grant=fixture()
    activation=dict(epoch=1206.,**{k:'f'*64 for k in ['GLOBAL_sha256','LOCAL_sha256','GLOBAL_intent_sha256',
        'GLOBAL_receipt_sha256','LOCAL_intent_sha256','LOCAL_receipt_sha256']})
    # Pure CPU activation fixture; NOT real two-site adoption or native GO.
    state=protocol._initial(grant,activation)
    root=tmp_path/'raw512-core';(root/'evidence').mkdir(parents=True)
    (root/'evidence'/store.GRANT).write_bytes(grant);(root/'evidence'/store.STATE).write_bytes(c.raw(state))
    return root,grant,p


def debit(root,grant,p,kind,sequence=None):
    state=store.read_confirmed(root,expected_grant_sha256=c.sha(grant))['state'];p=deepcopy(p);p['observed_epoch']=1207.
    token=f"{p['pid']}:{p['proc_start']}:{state['sequence'] if sequence is None else sequence}:DEBIT"
    return store.consume_once(root,expected_grant_sha256=c.sha(grant),effect_id=token,kind=kind,count=1,
        native_facts_bytes=c.raw(p),now_epoch=1207.)


def test_original_once_store_records512_model_and_query_before_effect_no_replay(tmp_path):
    root,grant,p=setup_store(tmp_path);debit(root,grant,p,'model_loads');v=debit(root,grant,p,'generation_queries')
    assert v['state']['row']['charged_query_attempts']==1 and v['state']['row']['checkpoint_step']==512
    assert len(list((root/'evidence').glob('DELEGATED-EFFECT-*.RECEIPT.json')))==2
    assert not v['canonical_GLOBAL_is_fresh'] and not v['native_dispatch_admitted']
    with pytest.raises(FileExistsError):debit(root,grant,p,'model_loads',sequence=0)


def test_original_lost_reply_reconciliation_keeps512_debit_and_never_resends(tmp_path,monkeypatch):
    root,grant,p=setup_store(tmp_path);original=store.once.commit
    def lost(*args,**kwargs):original(*args,**kwargs);raise TimeoutError('CPU_TEST_REPLY_LOST')
    monkeypatch.setattr(store.once,'commit',lost)
    with pytest.raises(TimeoutError):debit(root,grant,p,'model_loads')
    assert store.read_confirmed(root,expected_grant_sha256=c.sha(grant))['state']['sequence']==1
    with pytest.raises(FileExistsError):debit(root,grant,p,'model_loads',sequence=0)


@pytest.mark.parametrize('role',['core','hopper'])
def test_actual_fixed512_production_control_and_owner_only_contracts(tmp_path,role):
    from scripts.pi0294.pv1 import control_budget,control_heartbeat,mix_evaluation
    m,r,*_=fixture()
    m['scope']=c.PHYSICAL_ROLE if role=='core'else c.ROLE
    m['execution'].update(control_budget_profile=control_budget.PROFILE,owner_heartbeat_policy=control_heartbeat.POLICY,
        production_RPC_control=mix_evaluation.CORE_CONTROL)
    stage=dict(task=c.plan.STAGE,scope_grant_sha256=c.GRANT_SHA,authority_release=c.campaign.TASK_SOURCE,
        control_budget_profile=control_budget.PROFILE,owner_heartbeat_policy=control_heartbeat.POLICY)
    (tmp_path/'PV1_STAGE_BINDING.json').write_bytes(c.raw(stage))
    assert control_budget.validate(m,stage)['max_calls']==150000
    assert control_heartbeat.owner_only(tmp_path,m)
    m['authority']['scope_grant_sha256']=c.campaign.GRANT_SHA
    with pytest.raises(PermissionError):control_budget.validate(m,stage)
    with pytest.raises(PermissionError):control_heartbeat.owner_only(tmp_path,m)


@pytest.mark.parametrize('outside',[False,True])
def test_existing_nested_canonical_local_resident_directory(tmp_path,monkeypatch,outside):
    from scripts.pi0294.generalization import minimal_core_context as context
    from scripts.pi0294.pv1.mix_core_context import LOCAL_BINDING
    m,r,p,a,row,_=fixture();root=tmp_path/'campaign/neural-children'/r['run_uuid'];root.mkdir(parents=True)
    local=(tmp_path/'other'if outside else root.parent.parent/'control/original-resident')/'LOCAL_RESOURCE_LEDGER.private.json'
    local.parent.mkdir(parents=True);binding=c.raw(dict(canonical_LOCAL_ledger=str(local)))
    (root/LOCAL_BINDING).parent.mkdir(parents=True);(root/LOCAL_BINDING).write_bytes(binding)
    m['authority']['request_id']=r['request_id'];m['execution']['sealed_inputs'][LOCAL_BINDING]=c.sha(binding)
    mb=c.raw(m);r['run_manifest_sha256']=c.sha(mb);row.update(r,request=deepcopy(r),request_sha256=c.digest(r))
    (root/'RUN_MANIFEST.json').write_bytes(mb);local.write_bytes(c.raw(dict(unknown_effects=[],pending_updates=[],pending_queries=[],**{c.FIELD:{r['request_id']:row}})))
    monkeypatch.setattr(context.time,'time',lambda:1207.)
    if outside:
        with pytest.raises(PermissionError,match='ORIGINAL_CANONICAL_LOCAL'):context.ledger_binding(root)
    else:
        marker,path=context.ledger_binding(root)
        assert path==local and marker['T_CD']==r['T_CD']and marker['deadline_epoch']==r['deadline_epoch']
