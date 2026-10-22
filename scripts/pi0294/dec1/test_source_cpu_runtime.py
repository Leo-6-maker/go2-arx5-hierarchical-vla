"""Text-only draft contracts: fake filesystem, zero native execution evidence."""
import sys
_AUDIT={'active':True,'blocked':[]}
def _deny(event,args):
    if _AUDIT['active'] and (event.startswith(('subprocess.','socket.')) or event in (
            'os.system','os.fork','os.forkpty','os.posix_spawn','os.spawn','pty.spawn')):
        _AUDIT['blocked'].append(event);raise PermissionError('SOURCE_RUNTIME_FIXTURE_EXTERNAL_EFFECT:'+event)
sys.addaudithook(_deny)
from copy import deepcopy
import os
from pathlib import Path
import pytest
from scripts.pi0294.dec1 import source_cpu_runtime as unit
from scripts.pi0294.dec1.test_source_cpu_intake import packet,encode,sha,call as seal_packet
from scripts.pi0294.dec1.test_source_commissioning_plan import population
assert not _AUDIT['blocked'],'PROJECT_IMPORT_ATTEMPTED_EXTERNAL_EFFECT'
_AUDIT['active']=False

@pytest.fixture(autouse=True)
def no_external_effects():
    n=len(_AUDIT['blocked']);_AUDIT['active']=True
    try:yield
    finally:
        _AUDIT['active']=False
        assert len(_AUDIT['blocked'])==n


@pytest.fixture
def registered(packet,tmp_path,monkeypatch):
    p=packet;seal_packet(p);m=p['manifest'];campaign=p['adoption']['campaign_run_uuid']
    parent=tmp_path/campaign;root=parent/'children'/'cpu-source-fixture'
    infra=tmp_path/'approved-infra';registry=infra/'registry.json'
    code=root/'capsules'/m['code']['vla_commit']/'code';source={**p['source_files']}
    for n in (*unit.REQUIRED_RUNTIME_FILES,'scripts/pi0294/dec1/source_commissioning_plan.py'):
        source.setdefault(n,('SEALED_SYNTHETIC_SOURCE_NOT_IMPORTED:'+n).encode())
    def put(path,raw):
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():path.chmod(0o600)
        path.write_bytes(raw);path.chmod(0o400)
    def jput(path,v):put(path,encode(v))
    for n,b in source.items():put(code/n,b)
    monkeypatch.setattr(unit,'__file__',str(code/'scripts/pi0294/dec1/source_cpu_runtime.py'))
    monkeypatch.setattr(unit.intake,'__file__',str(code/'scripts/pi0294/dec1/source_cpu_intake.py'))
    monkeypatch.setattr(unit.intake.commissioning,'__file__',str(code/'scripts/pi0294/dec1/source_commissioning_plan.py'))
    entry=b'CPU_FIXTURE_ENTRY_NEVER_EXECUTED';doctor=b'CPU_FIXTURE_DOCTOR_NEVER_EXECUTED'
    put(infra/'enter.sh',entry);put(infra/'doctor.py',doctor)
    monkeypatch.setattr(unit,'ENTRY_SHA256',sha(entry));monkeypatch.setattr(unit,'DOCTOR_SHA256',sha(doctor))
    cpus=sorted(os.sched_getaffinity(0));m['limits'].update(cpus=cpus[:1],active_cores=p['budget']['concurrent_CoreLab_cores'])
    role=unit.ROLE+'__'+root.name;lease=infra/'leases'/'source-child.json';parent_lease=infra/'leases'/'parent.json'
    pb=dict(run_uuid=campaign,run_root=str(parent),uid=os.getuid(),role='CPU_FIXTURE_PARENT_CONTROL',
        registry=str(registry),lease=str(parent_lease))
    binding=dict(pb,run_uuid=root.name,run_root=str(root),registered_parent=str(root.parent),role=role,lease=str(lease))
    jput(parent/'OUTPUT_BINDING.private.json',pb);jput(root/'OUTPUT_BINDING.private.json',binding)
    inputs=dict(runtime_python='CPU_FIXTURE_NOT_EXECUTED',asset_root=str(tmp_path/'assets'))
    jput(root/'INPUT_BINDINGS.private.json',inputs);jput(parent/'INPUT_BINDINGS.private.json',inputs)
    original=dict(run_uuid=campaign,authority_release=unit.intake.PARENT_AUTHORITY,T0_epoch=p['parent']['T0_epoch'],
        stop_new_physics_epoch=p['parent']['T0_epoch']+160*3600,hard_closeout_epoch=unit.intake.HARD_CLOSE)
    jput(parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json',original)
    jput(parent_lease,dict(status='ACTIVE_CONTROL_PLANE_ONLY',owner='EXISTING_LOCAL_PC',run_uuid=campaign,
        authority_release=unit.intake.PARENT_AUTHORITY,deadline_epoch=unit.intake.HARD_CLOSE))
    jput(lease,dict(status='ACTIVE',owner='EXISTING_LOCAL_PC',run_uuid=root.name,
        authority_release=unit.intake.PARENT_AUTHORITY,deadline_epoch=m['limits']['operation_deadline_epoch']))
    rows=[dict(role=pb['role'],run_uuid=campaign,path=str(parent),executor_node=os.uname().nodename),
        dict(role=role,scope=unit.ROLE,run_uuid=root.name,path=str(root),executor_node=os.uname().nodename)]
    jput(registry,dict(entries=rows))
    assets=dict(absolute_files={})
    for i in range(26):
        f=tmp_path/'assets'/str(i);b=('CPU_FIXTURE_ASSET_'+str(i)).encode();put(f,b);assets['absolute_files'][str(f)]=sha(b)
    monkeypatch.setattr(unit,'ASSET_CANONICAL_SHA256',sha(unit._canonical(assets)))
    members={n:sha(b) for n,b in source.items()}
    seal=dict(source_sha=m['code']['vla_commit'],files=members);seal_raw=encode(seal)
    put(code.parent/'SOURCE_SEAL.json',seal_raw)
    ci=dict(schema=unit.CI_SCHEMA,source_sha=m['code']['vla_commit'],CI_conclusion='success',CI_run_id=1,CI_job_id=2,
        full_readback_file_sha256={n:members[n] for n in unit.REQUIRED_RUNTIME_FILES})
    ci_raw=encode(ci);p['stage']['runtime_CI_receipt_sha256']=sha(ci_raw)
    projection=dict(schema=unit.PROJECTION_SCHEMA,authority_release=unit.intake.AUTHORITY,campaign_run_uuid=campaign,
        parent_projection_origin='EXISTING_LOCAL_PC',native_safety_unknown_effects=[],unknown_effects=[],
        parent_snapshot_sha256=p['budget']['parent_snapshot_sha256'],DEC1_snapshot_sha256=p['budget']['DEC1_snapshot_sha256'],
        adoption_sha256=p['budget']['adoption_sha256'],other_registered_CoreLab_cores=1)
    for k in ('observed_epoch','valid_until_epoch','DEC1_used','DEC1_reserved',
            'parent_remaining_after_spent_reserved_and_final','site_remaining',
            'source_commissioning_used','source_commissioning_reserved','target_component_reserved'):
        projection[k]=deepcopy(p['budget'][k])
    projection_raw=encode(projection);p['budget']['native_projection_receipt_sha256']=sha(projection_raw)
    pin_raw=b'CPU_FIXTURE_PIN_REPORT_NOT_NATIVE';monkeypatch.setattr(unit.intake,'PIN_REPORT_SHA256',sha(pin_raw))
    p['stage']['original_pin_report_sha256']=sha(pin_raw)
    seal_packet(p)
    for key,path in unit.intake.PATHS.items():put(root/path,p['population_bytes'] if key=='population' else encode(p[key]))
    put(root/unit.PARENT_SNAPSHOT,encode(p['parent']));put(root/unit.DEC1_SNAPSHOT,encode(p['dec1']))
    put(parent/unit.NATIVE_ADOPTION,encode(p['adoption']))
    extra={unit.CI:ci_raw,unit.PROJECTION:projection_raw,unit.ASSETS:encode(assets),
        unit.PARENT_SNAPSHOT:encode(p['parent']),unit.DEC1_SNAPSHOT:encode(p['dec1']),
        'plans/SOURCE_ORIGINAL_PIN_IDENTITY.json':pin_raw}
    for n,b in extra.items():put(root/n,b)
    m.update(owner='EXISTING_LOCAL_PC',run_uuid=root.name)
    m['code'].update(capsule_relative=str(code.relative_to(root)),source_seal_relative=str((code.parent/'SOURCE_SEAL.json').relative_to(root)),
        source_seal_sha256=sha(seal_raw))
    m['execution'].update(host=os.uname().nodename,uid=os.getuid(),private_path_role=role,
        output_binding_sha256=sha(encode(binding)),input_binding_sha256=sha(encode(inputs)),
        entry_sha256=sha(entry),doctor_sha256=sha(doctor))
    m['execution']['sealed_inputs'].update({n:sha(b) for n,b in extra.items()})
    request=p['budget']['request_id'];current=deepcopy(p['dec1'])
    current['schema']='DEC1_NONADDITIVE_RESOURCE_CLASSIFICATION_V1'
    row=dict(schema=unit.RESERVATION_SCHEMA,request_id=request,authority_release=unit.intake.AUTHORITY,role=unit.ROLE,
        campaign_run_uuid=campaign,child_run_uuid=root.name,child_root=str(root),
        component_plan_sha256=unit.intake.PLAN_SHA256,budget_sha256=m['execution']['sealed_inputs'][unit.intake.PATHS['budget']],
        parent_projection_sha256=p['budget']['parent_snapshot_sha256'],pre_reservation_DEC1_sha256=p['budget']['DEC1_snapshot_sha256'],
        native_projection_receipt_sha256=sha(projection_raw),status='RESERVED',unknown_effects=[],reserved_epoch=p['now_epoch'],
        deadline_epoch=m['limits']['operation_deadline_epoch'],requested=deepcopy(p['budget']['requested']),remaining=deepcopy(p['budget']['requested']))
    current.update(source_cpu_reservations={request:row},source_cpu_reserved=deepcopy(row['remaining']),
        source_cpu_current_classification=deepcopy(p['budget']['DEC1_used']))
    child=dict(schema='PI0294_PV1_CORE_POLICY_LEDGER_V1',scope=unit.ROLE,run_uuid=root.name,
        authority_release=unit.intake.PARENT_AUTHORITY,authority_amendments=[unit.intake.AUTHORITY],source_budget_request_id=request,
        Core_CPU_s=0,CPU_reserved_s=p['budget']['requested']['CoreLab_CPU_s'],main_starts=0,engine_starts=0,
        live_and_recovery_integrations=0,shadow_integrations=0,reservations={},unknown_effects=[])
    jput(root/'RESOURCE_LEDGER.json',child)
    def refresh():
        jput(root/'RUN_MANIFEST.json',m);jput(code.parent/'WORKER_RUN_MANIFEST.json',m)
        row['child_manifest_sha256']=sha(encode(m));jput(parent/unit.CURRENT_DEC1,current)
    refresh()
    monkeypatch.setenv('PV1_RUN_MANIFEST',str(code.parent/'WORKER_RUN_MANIFEST.json'))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');monkeypatch.delenv('PBS_JOBID',raising=False)
    monkeypatch.setenv('MUJOCO_GL','egl');monkeypatch.setenv('LIBGL_ALWAYS_SOFTWARE','1')
    monkeypatch.setenv('__EGL_VENDOR_LIBRARY_FILENAMES','/usr/share/glvnd/egl_vendor.d/50_mesa.json')
    monkeypatch.delenv('MUJOCO_EGL_DEVICE_ID',raising=False)
    monkeypatch.setattr(unit.time,'time',lambda:p['now_epoch'])
    # This suite tests the narrow DEC1 add-on contract. Root separately runs
    # genuine shared-scope tests; do not duplicate that framework here.
    shared_calls=[]
    def shared(actual_root,*,role,closeout=False):
        assert actual_root==root and role==unit.ROLE
        shared_calls.append((actual_root,role,closeout));return m,inputs
    monkeypatch.setattr(unit,'validate_shared',shared)
    return dict(root=root,parent=parent,m=m,p=p,current=current,row=row,child=child,lease=lease,shared_calls=shared_calls,
        put=put,jput=jput,refresh=refresh,code=code,ci=ci,projection=projection)


def test_calls_shared_scope_and_requires_distinct_once_native_reservation(registered):
    h=registered
    assert not (h['parent']/'PV1_RESOURCE_LEDGER.json').exists()  # LOCAL PC snapshot only.
    assert unit.validate(h['root'])==h['m']
    assert h['shared_calls']==[(h['root'],unit.ROLE,False)]
    h['current'].pop('source_cpu_reservations');h['refresh']()
    with pytest.raises(PermissionError,match='ONCE_CHILD_RESERVATION_REQUIRED'):unit.validate(h['root'])


def test_own_full_native_reservation_is_not_added_twice_and_old_STOPs_stay_local(registered,monkeypatch):
    h=registered;cap=h['current']['CPU_s']+h['row']['remaining']['CoreLab_CPU_s']
    monkeypatch.setitem(unit.intake.CAPS,'CoreLab_CPU_s',cap)
    h['jput'](h['parent']/'children'/'old-closed-engineering'/'STOP.json',dict(reason='HISTORICAL_NOT_ZERO_EFFECT'))
    assert unit.validate(h['root'])==h['m']


@pytest.mark.parametrize('fault',['own_stop','reservation_total','reservation_aggregate',
    'lost_cost','native_unknown','affinity','worker_four','concurrency','source','actual_adoption','CI'])
def test_real_bytes_identity_and_current_reservation_contracts_reject(registered,fault):
    h=registered
    if fault=='own_stop':h['jput'](h['root']/'STOP.json',dict(reason='OWN_CHILD_STOP'))
    elif fault=='reservation_total':h['row']['requested']['physics_ticks']+=1;h['refresh']()
    elif fault=='reservation_aggregate':h['current']['source_cpu_reserved']['physics_ticks']-=1;h['refresh']()
    elif fault=='lost_cost':h['current']['CPU_s']=0;h['refresh']()
    elif fault=='native_unknown':h['current']['unknown_effects']=['UNKNOWN_NATIVE_EFFECT'];h['refresh']()
    elif fault=='affinity':h['m']['limits']['cpus']=[max(os.sched_getaffinity(0))+1];h['refresh']()
    elif fault=='worker_four':h['m']['limits']['cpus']=[0,1,2,3];h['refresh']()
    elif fault=='concurrency':h['m']['limits']['active_cores']=5;h['refresh']()
    elif fault=='source':h['put'](h['code']/next(iter(unit.intake.ORIGINAL_13)),b'CHANGED')
    elif fault=='actual_adoption':
        path=h['parent']/unit.NATIVE_ADOPTION;h['put'](path,path.read_bytes()+b' ')
    elif fault=='CI':h['put'](h['root']/unit.CI,b'CHANGED')
    with pytest.raises(PermissionError):unit.validate(h['root'])


def test_entry_identity_options_cannot_be_injected(registered):
    with pytest.raises(TypeError):unit.validate(registered['root'],now=registered['p']['now_epoch'])
    with pytest.raises(PermissionError,match='FIXED_PROFILE'):unit.validate(registered['root'],'hopper-neural')


def test_expired_budget_is_metadata_closeout_only_and_unknown_is_preserved(registered,monkeypatch):
    h=registered
    monkeypatch.setattr(unit.time,'time',lambda:unit.intake.HARD_CLOSE+10)
    h['current']['unknown_effects']=['ACTUAL_EFFECT_UNRESOLVED']
    h['row'].update(status='UNKNOWN',unknown_effects=['ACTUAL_EFFECT_UNRESOLVED']);h['refresh']()
    h['jput'](h['root']/'STOP.json',dict(reason='PRESERVED_UNKNOWN'))
    with pytest.raises(PermissionError):unit.validate(h['root'])
    assert unit.validate(h['root'],closeout=True)==h['m']
    assert h['row']['remaining']==h['row']['requested']
    assert h['current']['unknown_effects']==['ACTUAL_EFFECT_UNRESOLVED']


def test_late_native_reservation_cannot_be_backdated_by_closeout(registered):
    h=registered;h['row']['reserved_epoch']=h['row']['deadline_epoch'];h['refresh']()
    with pytest.raises(PermissionError,match='RESERVED_INTERVAL'):unit.validate(h['root'],closeout=True)


def test_actual_cost_overrun_remains_visible_for_closeout_only(registered):
    h=registered;cost=unit.intake.CAPS['CoreLab_CPU_s']+1
    h['current']['CPU_s']=cost
    h['current']['source_cpu_current_classification']['CoreLab_CPU_s']=cost;h['refresh']()
    with pytest.raises(PermissionError,match='COSTS_PRESERVED_AND_BOUNDED'):unit.validate(h['root'])
    assert unit.validate(h['root'],closeout=True)==h['m']
    assert h['current']['CPU_s']==cost
