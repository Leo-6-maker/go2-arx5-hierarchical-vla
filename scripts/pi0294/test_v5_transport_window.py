"""C/D transport uses its actually adopted window; old lease bytes stay old."""
from copy import deepcopy
import json
import pytest
from scripts.pi0294.pv1 import transport_fixture as f,mix_core_context as ctx,mix_core_accounting as c,successor_cd as s,mix_accounting as a
from scripts.pi0294.test_v5_core_stage_accounting import closed


def setup(tmp_path):
    root=tmp_path/'campaign/children/new';(root/'plans').mkdir(parents=True)
    g,l,_=closed();ledger=tmp_path/'LOCAL.json';ledger.write_bytes(a.raw(l))
    binding=root/ctx.LOCAL_BINDING;binding.write_bytes(a.raw(dict(canonical_LOCAL_ledger=str(ledger))))
    (root/'V5_CD_SCOPE_BINDING.json').write_bytes(a.raw(l[s.FIELD]))
    manifest=dict(scope=c.ROLE,platform_evaluation_version=c.VERSION,execution=dict(sealed_inputs={ctx.LOCAL_BINDING:a.sha(binding.read_bytes())}))
    registry=dict(execution_scope=s.TASK,execution_authority_release=s.TASK_SOURCE,scope_grant_sha256=s.GRANT_SHA,T_CD=l[s.FIELD]['T_CD'],execution_deadline_epoch=l[s.FIELD]['deadline_epoch'])
    old=root.parent.parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json';old.write_bytes(a.raw(dict(run_uuid='campaign',stop_new_physics_epoch=1)))
    return root,manifest,registry,old.read_bytes()


def test_actual_new_window_keeps_original_protocol_and_old_parent_clock(tmp_path):
    root,m,r,old=setup(tmp_path);v=f.fixture_window(root,m,r)
    assert v['stop_new_physics_epoch']==r['execution_deadline_epoch']and v['run_uuid']=='campaign'
    assert (root.parent.parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json').read_bytes()==old
    assert f.fixture_window(root,dict(scope='old-PV1'),{})['stop_new_physics_epoch']==1


@pytest.mark.parametrize('field,value',[('execution_scope','old'),('execution_authority_release','old'),('scope_grant_sha256','0'*64),('T_CD',999.),('execution_deadline_epoch',1)])
def test_old_ACK_or_adapted_clock_is_not_a_CD_execution_binding(tmp_path,field,value):
    root,m,r,_=setup(tmp_path);r[field]=value
    with pytest.raises(PermissionError):f.fixture_window(root,m,r)


def test_exact_scope_native_status_does_not_alias_old_core_or_hopper():
    from scripts.pi0294.pv1 import transport_fixture as t,mix_core_identity as cd,evaluation_core_identity as old,evaluation_platform as v4
    assert t.allocation_status({'scope':cd.c.ROLE},core_typed=True,typed=False)==cd.STATUS
    assert t.allocation_status({'platform_evaluation_version':cd.c.VERSION},core_typed=True,typed=False)==cd.STATUS
    assert t.allocation_status({'scope':old.c.ROLE},core_typed=True,typed=False)==old.STATUS
    assert t.allocation_status({},core_typed=False,typed=True)==v4.STATUS
    assert t.allocation_status({},core_typed=False,typed=False)=='PASS_PV1_OWN_NATIVE_H200_ALLOCATION'
    assert cd.STATUS!=old.STATUS


@pytest.mark.parametrize('wrong_legacy_status',[False,True])
def test_whole_Pump_with_actual_saved_CD_admission_and_recorded_plan_CPU_fixture(tmp_path,monkeypatch,wrong_legacy_status):
    import base64,json,os
    from pathlib import Path
    from types import SimpleNamespace
    from scripts.pi0294.pv1 import transport_fixture as t
    from scripts.pi0294.pv1 import mix_core_identity as cd,evaluation_core_identity as old
    b=Path(os.environ.get('PI0294_PRIVATE_WORKSPACE','/home/liuyu/ISSAC-GR00T'))/'work/private/v5-execution-20261025'
    paths=sorted(b.glob('C50_CORE_EVAL_FAILED_PREMODEL_EXACT_DUAL_CLOSURE_READONLY*'))
    if not paths:pytest.skip('private actual admission/recorded plan unavailable')
    cap=json.loads(paths[-1].read_bytes()).get('value')
    if not cap:pytest.skip('actual complete independent capture unavailable')
    packet=json.loads((b/'C50_CORE_EVAL_PAIRED_PEERS_COLD2_PREPARED.private.json').read_bytes());row=packet['assembly']['rows']['hopper'];m=row['manifest'];root=tmp_path/m['run_uuid'];root.mkdir()
    def put(name,data):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);p.chmod(0o400)
    put('RUN_MANIFEST.json',base64.b64decode(row['manifest_bytes_base64']))
    for n,v in row['files'].items():put(n,base64.b64decode(v))
    allocation=json.loads(base64.b64decode(cap['files']['ADMISSION']['base64']));registry=json.loads(base64.b64decode(cap['files']['FIXTURE_REGISTRY']['base64']));label=registry['label']
    if wrong_legacy_status:allocation['status']=old.STATUS
    put('plans/'+label+'/REGISTRY.json',(json.dumps(registry,indent=2)+'\n').encode());put('pbs/alloc-01/ALLOCATION_ADMISSION.json',(json.dumps(allocation,indent=2)+'\n').encode());put('pbs/alloc-01/REQUEST.json',(json.dumps(packet['assembly']['request'],indent=2)+'\n').encode())
    checked=[]
    monkeypatch.setattr(old,'current_handle',lambda:allocation['job_id']);monkeypatch.setattr(t.os,'uname',lambda:SimpleNamespace(nodename=allocation['hostname']))
    # CPU test explicitly replaces actual kernel validation. Its full strict
    # real process checks are separately exercised by identity regressions.
    monkeypatch.setattr(old,'current_identity',lambda *args:checked.append('typed_current_identity_called'))
    if wrong_legacy_status:
        with pytest.raises(PermissionError,match='EXACT_CURRENT_ADMISSION_AND_REGISTRY'):t.Pump(root,label,allocation,lambda:None)
    else:
        pump=t.Pump(root,label,allocation,lambda:None);assert pump.next==0 and (pump.folder/'COMPUTE_STARTED.json').is_file();pump.close()
    assert checked==['typed_current_identity_called']
