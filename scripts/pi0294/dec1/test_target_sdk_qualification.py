"""Temp-file entry guards and real finite subprocesses; no installed SDK."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import pytest
from . import target_sdk_qualification as unit
from .test_target_sdk_resources import packet, reserve
from .test_target_sdk_budget import bundle


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(unit.contract.common.encode(value))


@pytest.fixture
def owned(packet,tmp_path):
    proposed=reserve(packet);row=proposed['reservation'];root=tmp_path/unit.contract.CAMPAIGN/'children'/row['child_run_uuid']
    root.mkdir(parents=True);parent=root.parent.parent
    role=unit.contract.ROLE+'__'+root.name
    m=dict(execution=dict(private_path_role=role),code=dict(vla_commit=row['source_sha']));q=deepcopy(packet['q']);q['deadline_epoch']=time.time()+30
    write(root/'RUN_MANIFEST.json',m);write(root/'REQUEST.json',q)
    row['manifest_sha256']=unit.filehash(root/'RUN_MANIFEST.json');row['request_sha256']=unit.filehash(root/'REQUEST.json')
    write(root/unit.contract.PROJECTION,dict(CPU_fixture=True))
    row['projection_sha256']=unit.filehash(root/unit.contract.PROJECTION);row['deadline_epoch']=q['deadline_epoch']
    p=parent/'lease.json';write(p,dict(status='ACTIVE_CONTROL_PLANE_ONLY',authority_release=unit.contract.common.PARENT_AUTHORITY))
    write(parent/'OUTPUT_BINDING.private.json',dict(lease=str(p)))
    lease=root/'lease.json';write(lease,dict(status='ACTIVE',scope=unit.contract.ROLE,run_uuid=root.name,
        deadline_epoch=q['deadline_epoch'],authority_release=unit.contract.common.PARENT_AUTHORITY))
    registry=tmp_path/'registry.json';write(registry,dict(entries=[dict(role=role,run_uuid=root.name,path=str(root))]))
    binding=dict(role=role,lease=str(lease),registry=str(registry))
    write(parent/'authority/DEC1/DEC1_RESOURCE_LEDGER.json',proposed['proposed_Core'])
    # The row object above is shared with the proposed ledger in this fixture.
    ledger=proposed['proposed_Core'];ledger[unit.budget.ROWS][row['request_id']]=deepcopy(row)
    write(parent/'authority/DEC1/DEC1_RESOURCE_LEDGER.json',ledger)
    write(root/'LOCAL_RESERVATION_CONFIRMED.private.json',dict(status='ACTUAL_LOCAL_PARENT_RESERVATION_CONFIRMED',
        request_id=row['request_id'],manifest_sha256=row['manifest_sha256'],reservation=row))
    return root,m,q,binding


def test_same_owned_guard_and_expired_stop_cleanup_semantics(owned):
    root,m,q,b=owned;guard=unit.resource_guard(root,m,q,b);guard()
    (root/'STOP.json').write_text('{}')
    with pytest.raises(PermissionError):guard()
    guard(closeout=True)


@pytest.mark.parametrize('fault',['lease_closed','other_run','other_scope','other_deadline','duplicate_registry',
    'old_role','wrong_manifest','unknown_write','dropped_hold','wrong_LOCAL','parent_inactive'])
def test_guard_rejects_actual_lease_registry_or_either_reservation_change(owned,fault):
    root,m,q,b=owned;guard=unit.resource_guard(root,m,q,b);guard()
    if fault in ('lease_closed','other_run','other_scope','other_deadline'):
        p=Path(b['lease']);v=unit.read(p)
        key,value={'lease_closed':('status','CLOSED'),'other_run':('run_uuid','old'),
            'other_scope':('scope','OLD_COUNTER'),'other_deadline':('deadline_epoch',1)}[fault]
        v[key]=value;write(p,v)
    elif fault=='duplicate_registry':
        p=Path(b['registry']);v=unit.read(p);v['entries'].append(v['entries'][0]);write(p,v)
    elif fault=='old_role':b['role']='OLD_EXPIRED_R4'
    elif fault=='wrong_manifest':write(root/'RUN_MANIFEST.json',dict(execution=dict(private_path_role='changed')))
    elif fault in ('unknown_write','dropped_hold'):
        p=root.parent.parent/'authority/DEC1/DEC1_RESOURCE_LEDGER.json';v=unit.read(p)
        if fault=='unknown_write':v['unknown_effects']=['UNKNOWN']
        else:v[unit.budget.ROWS][q['request_id']]['remaining']['CoreLab_CPU_s']=0
        write(p,v)
    elif fault=='wrong_LOCAL':
        p=root/'LOCAL_RESERVATION_CONFIRMED.private.json';v=unit.read(p);v['request_id']='other';write(p,v)
    elif fault=='parent_inactive':write(root.parent.parent/'lease.json',dict(status='CLOSED'))
    with pytest.raises(PermissionError):guard()


def test_unregistered_root_refuses_before_any_compiler_or_SDK(tmp_path,monkeypatch):
    attempts=[]
    monkeypatch.setattr(unit.subprocess,'Popen',lambda *a,**kw:attempts.append(a))
    with pytest.raises(FileNotFoundError):unit.run(tmp_path)
    assert attempts==[] and not (tmp_path/'evidence').exists()


def test_finite_build_phase_real_exit_and_durable_no_replay(tmp_path):
    root=tmp_path;(root/'evidence/sdk-build').mkdir(parents=True)
    marker=root/'evidence/sdk-build/own.txt'
    argv=[sys.executable,'-c','from pathlib import Path;Path("own.txt").write_text("CPU fixture")']
    unit.phase(root,'COMPILE',argv,guard=lambda:None,deadline=time.time()+5)
    assert marker.read_text()=='CPU fixture'
    assert unit.read(root/'evidence/SDK_COMPILE_RETURNED.json')['returncode']==0
    with pytest.raises(FileExistsError):unit.phase(root,'COMPILE',argv,guard=lambda:None,deadline=time.time()+5)


def test_failed_real_phase_never_runs_a_second_time(tmp_path):
    (tmp_path/'evidence/sdk-build').mkdir(parents=True)
    args=[sys.executable,'-c','raise SystemExit(7)']
    with pytest.raises(PermissionError):unit.phase(tmp_path,'LINK',args,guard=lambda:None,deadline=time.time()+5)
    assert (tmp_path/'evidence/SDK_LINK_INTENT.json').is_file()
    assert not (tmp_path/'evidence/SDK_LINK_RETURNED.json').exists()
    with pytest.raises(FileExistsError):unit.phase(tmp_path,'LINK',args,guard=lambda:None,deadline=time.time()+5)


def test_expired_phase_refuses_before_intent_or_process(tmp_path,monkeypatch):
    (tmp_path/'evidence/sdk-build').mkdir(parents=True);attempts=[]
    monkeypatch.setattr(unit.subprocess,'Popen',lambda *a,**kw:attempts.append(a))
    with pytest.raises(PermissionError):unit.phase(tmp_path,'COMPILE',['irrelevant'],guard=lambda:None,deadline=time.time()-1)
    assert attempts==[] and not (tmp_path/'evidence/SDK_COMPILE_INTENT.json').exists()


def test_whole_original_forecast_pack_roundtrip_and_unknown_input_refused(packet):
    raw=unit.contract.forecast_pack(packet['args'])
    assert unit.contract.unpack_forecast(raw)==packet['args']
    bad=deepcopy(packet['args']);bad['unsealed_grant']=b'not authority'
    with pytest.raises(PermissionError):unit.contract.forecast_pack(bad)


@pytest.fixture
def full_entry(packet,tmp_path,monkeypatch):
    """An explicitly synthetic current-entry filesystem, no compiler/SDK."""
    root=tmp_path/unit.contract.CAMPAIGN/'children'/packet['q']['child_run_uuid'];root.mkdir(parents=True)
    code=root/'capsule/code';source=code/'scripts/pi0294/dec1';source.mkdir(parents=True)
    pins={}
    for name in unit.REQUIRED_CODE:
        path=source/name;path.write_bytes(('CPU entry fixture '+name).encode());path.chmod(0o400)
        pins['scripts/pi0294/dec1/'+name]=unit.filehash(path)
    monkeypatch.setattr(unit,'__file__',str(source/'target_sdk_qualification.py'))
    monkeypatch.setattr(unit,'time',SimpleNamespace(time=lambda:packet['args']['now_epoch']))
    sdk=tmp_path/'original-sdk';build=sdk/'build/ocs2_mobile_manipulator_ros';core=sdk/'build/ocs2_core';build.mkdir(parents=True);core.mkdir()
    mobile=build/'libmobile.a';mobile.write_bytes(b'CPU original mobile archive fixture')
    core=core/'libcore.a';core.write_bytes(b'CPU original core archive fixture')
    flags=tmp_path/'flags.make';flags.write_bytes(b'CXX_DEFINES = -DCPU_FIXTURE\nCXX_INCLUDES = -I/cpu-fixture\nCXX_FLAGS = -std=gnu++17\n')
    link=tmp_path/'link.txt';link.write_bytes(('/usr/bin/c++ '+unit.build_plan.NODE_OBJECT+
        ' -o mobile_manipulator_mpc_node libmobile.a ../ocs2_core/libcore.a\n').encode())
    dependencies=[];dependency_pins={}
    for name in unit.build_plan.DEPENDENCY_SHARED_SHA256:
        path=sdk.parent/'dependency-root/usr/lib/x86_64-linux-gnu'/name
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(('CPU dependency '+name).encode())
        dependencies.append(path);dependency_pins[name]=unit.filehash(path)
    monkeypatch.setattr(unit.build_plan,'DEPENDENCY_SHARED_SHA256',dependency_pins)
    monkeypatch.setattr(unit.build_plan,'FLAGS_SHA256',unit.filehash(flags));monkeypatch.setattr(unit.build_plan,'LINK_SHA256',unit.filehash(link))
    task=tmp_path/'task.info';task.write_bytes(b'CPU original task fixture')
    clock=tmp_path/'clock.py';clock.write_bytes(b'CPU original frozen clock fixture')
    package=tmp_path/'packages/robot_fixture';(package/'meshes').mkdir(parents=True)
    mesh=package/'meshes/link.glb';mesh.write_bytes(b'CPU_MESH_FIXTURE_NOT_NATIVE_GEOMETRY')
    urdf=tmp_path/'model.urdf';urdf.write_bytes(b'<robot name="CPU"><link name="base"><visual><geometry><mesh filename="package://robot_fixture/meshes/link.glb"/></geometry></visual></link></robot>')
    entry=tmp_path/'enter.sh';entry.write_bytes(b'CPU standard entry fixture')
    doctor=tmp_path/'doctor.py';doctor.write_bytes(b'CPU standard doctor fixture')
    for key,path in (('MOBILE_SDK_SHA256',mobile),('CORE_SDK_SHA256',core),('FROZEN_TASK_SHA256',task),
                     ('FROZEN_CLOCK_SHA',clock),('ENTRY_SHA',entry),('DOCTOR_SHA',doctor)):
        monkeypatch.setattr(unit,key,unit.filehash(path))
    role=unit.contract.ROLE+'__'+root.name;parent=root.parent.parent
    lease=root/'lease.json';registry=tmp_path/'registry.json';parent_lease=parent/'lease.json'
    write(parent_lease,dict(status='ACTIVE_CONTROL_PLANE_ONLY',authority_release=unit.contract.common.PARENT_AUTHORITY))
    write(parent/'OUTPUT_BINDING.private.json',dict(lease=str(parent_lease)))
    write(registry,dict(entries=[dict(role=role,run_uuid=root.name,path=str(root))]))
    write(lease,dict(status='ACTIVE',scope=unit.contract.ROLE,run_uuid=root.name,deadline_epoch=packet['q']['deadline_epoch'],authority_release=unit.contract.common.PARENT_AUTHORITY))
    binding=dict(run_root=str(root),uid=2009,role=role,lease=str(lease),registry=str(registry))
    write(root/'OUTPUT_BINDING.private.json',binding)
    inputs=dict(runtime_python=sys.executable,sdk_root=str(sdk),flags_file=str(flags),link_file=str(link),
        original_task=str(task),urdf=str(urdf),urdf_sha256=unit.filehash(urdf),mobile_archive=str(mobile),
        core_archive=str(core),frozen_clock=str(clock),frames=['-','-','-'],
        sdk_package_roots={'robot_fixture':str(package)},sdk_package_mesh_sha256={str(mesh):unit.filehash(mesh)},
        original_readonly_files={str(p):unit.filehash(p) for p in (mobile,core,flags,link,urdf,mesh,*dependencies)})
    write(root/'INPUT_BINDINGS.private.json',inputs)
    write(root/'capsule/SOURCE_SEAL.json',dict(source_sha='a'*40,files=pins))
    write(root/'plans/PUBLICATION_AND_CI.private.json',dict(source='a'*40,CI_conclusion='success',all_12_steps_success=True,full_readback_file_sha256=pins))
    for name,data in (('DEC1_NATIVE_ADOPTION_AND_BUDGET.json',b'{"CPU_fixture":true}\n'),('VAP1_ADOPTION_AND_DELTA_BUDGET.json',packet['VAP1'])):
        p=root/'plans'/name;p.write_bytes(data)
        dest=parent/'authority'/('DEC1' if name.startswith('DEC1') else 'VAP1')/name
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    (root/'REQUEST.json').write_bytes(unit.contract.common.encode(packet['q']))
    (root/unit.contract.PROJECTION).write_bytes(unit.contract.common.encode(packet['x']))
    (root/unit.contract.FORECAST).write_bytes(unit.contract.forecast_pack(packet['args']))
    write(root/'plans/SDK_LOCAL_NAMESPACES.private.json',packet['space'])
    m=dict(schema='PI0294_PV1_CPU_CHILD_RUN_MANIFEST_V1',scope=unit.contract.ROLE,run_uuid=root.name,owner='EXISTING_LOCAL_PC',
        authority=dict(protocol_ref=unit.contract.common.PARENT_AUTHORITY,amendments=[unit.contract.common.AUTHORITY,unit.contract.common.VAP1_AUTHORITY]),
        code=dict(vla_commit='a'*40,capsule_relative='capsule/code',source_seal_relative='capsule/SOURCE_SEAL.json',source_seal_sha256=unit.filehash(root/'capsule/SOURCE_SEAL.json')),
        limits=dict(engine_starts=2,physics_ticks=0,policy_episodes=0,PV1_T0_epoch=unit.contract.T0,PV1_hard_close_epoch=unit.contract.HARD_CLOSE,
            operation_deadline_epoch=packet['q']['deadline_epoch'],cpus=[1,2],memory_bytes=8*1024**3),
        execution=dict(profile='4080-cpu-data',host='corelab-4080s',uid=2009,private_path_role=role,
            sdk_initialization_mode='ORIGINAL_SDK_INITIALIZE',entry_path=str(entry),entry_sha256=unit.ENTRY_SHA,
            doctor_path=str(doctor),doctor_sha256=unit.DOCTOR_SHA,input_binding_sha256=unit.filehash(root/'INPUT_BINDINGS.private.json'),
            output_binding_sha256=unit.filehash(root/'OUTPUT_BINDING.private.json'),sealed_inputs={}))
    for name in ('REQUEST.json',unit.contract.PROJECTION,unit.contract.FORECAST,'plans/SDK_LOCAL_NAMESPACES.private.json'):
        p=root/name;p.chmod(0o400);m['execution']['sealed_inputs'][name]=unit.filehash(p)
    write(root/'RUN_MANIFEST.json',m);write(root/'capsule/WORKER_RUN_MANIFEST.json',m)
    proposal=unit.contract.reserve(packet['parent'],packet['Core'],(root/'REQUEST.json').read_bytes(),
        (root/'RUN_MANIFEST.json').read_bytes(),(root/unit.contract.PROJECTION).read_bytes(),packet['VAP1'],
        forecast_inputs=packet['args'],namespace_proof=packet['space'],now_epoch=packet['args']['now_epoch'])
    write(parent/'authority/DEC1/DEC1_RESOURCE_LEDGER.json',proposal['proposed_Core'])
    parent_after=unit.contract.common.encode(proposal['proposed_parent'])
    write(root/'LOCAL_RESERVATION_CONFIRMED.private.json',dict(status='ACTUAL_LOCAL_PARENT_RESERVATION_CONFIRMED',
        request_id=packet['q']['request_id'],manifest_sha256=unit.filehash(root/'RUN_MANIFEST.json'),reservation=proposal['reservation'],
        parent_after_base64=__import__('base64').b64encode(parent_after).decode(),parent_after_sha256=unit.contract.common.sha(parent_after)))
    for name in ('sdk-build','sdk-cache','sdk-client'):(root/'evidence'/name).mkdir(parents=True)
    monkeypatch.setattr(unit,'os',SimpleNamespace(uname=lambda:SimpleNamespace(nodename='corelab-4080s'),getuid=lambda:2009,
        sched_getaffinity=lambda _:set([1,2]),environ=dict(CUDA_VISIBLE_DEVICES='',GR00T_WBC_PYTHON=sys.executable,
            PV1_RUN_MANIFEST=str(root/'capsule/WORKER_RUN_MANIFEST.json'))))
    return root


def test_complete_temp_entry_recomputes_original_forecast_and_build_plan_without_native_dispatch(full_entry):
    m,q,inputs,plan,guard=unit.validate(full_entry);guard()
    assert plan['vendor_sources_compiled']==[] and plan['native_action'] is False
    assert plan['compile_argv'][-4:]==['-c',str(full_entry/'capsule/code/scripts/pi0294/dec1/target_sdk_client.cpp'),
        '-o',str(full_entry/'evidence/sdk-build/target_sdk_client.o')]
    assert not (full_entry/'evidence/SDK_QUALIFICATION_INTENT.json').exists()


@pytest.mark.parametrize('fault',['wrong_host','SDK_changed','code_changed','old_clock','cache_not_empty','unsealed_forecast'])
def test_complete_entry_faults_refuse_before_compilation_or_constructor(full_entry,fault):
    root=full_entry
    if fault=='wrong_host':unit.os.uname=lambda:SimpleNamespace(nodename='unregistered-host')
    elif fault=='SDK_changed':Path(unit.read(root/'INPUT_BINDINGS.private.json')['mobile_archive']).write_bytes(b'changed SDK')
    elif fault=='code_changed':
        p=root/'capsule/code/scripts/pi0294/dec1/target_sdk_client.cpp';p.chmod(0o600);p.write_bytes(b'changed own source')
    elif fault=='old_clock':unit.time.time=lambda:unit.contract.STOP_NEW+1
    elif fault=='cache_not_empty':(root/'evidence/sdk-cache/old.cache').write_bytes(b'old')
    elif fault=='unsealed_forecast':
        p=root/unit.contract.FORECAST;p.chmod(0o600);p.write_bytes(b'{}')
    with pytest.raises((PermissionError,KeyError)):unit.validate(root)
    assert not (root/'evidence/SDK_QUALIFICATION_INTENT.json').exists()


def test_real_transitive_RUNPATH_failure_and_RPATH_resolution_before_any_SDK(tmp_path):
    """Three trivial C objects, no installed SDK, controller, model or physics."""
    import shutil,subprocess
    compiler=shutil.which('cc')
    if compiler is None:pytest.skip('CPU C compiler unavailable')
    base=tmp_path/'base';middle=tmp_path/'middle';base.mkdir();middle.mkdir()
    (base/'base.c').write_text('int fixture_base(void){return 0;}\n')
    (middle/'middle.c').write_text('extern int fixture_base(void);int fixture_middle(void){return fixture_base();}\n')
    main=tmp_path/'main.c';main.write_text('extern int fixture_middle(void);int main(void){return fixture_middle();}\n')
    def build(argv):
        p=subprocess.run([compiler,*map(str,argv)],capture_output=True,timeout=20)
        assert p.returncode==0,p.stderr.decode()
    build(['-shared','-fPIC',base/'base.c','-Wl,-soname,libfixture_base.so','-o',base/'libfixture_base.so'])
    build(['-shared','-fPIC',middle/'middle.c','-L'+str(base),'-lfixture_base','-Wl,-soname,libfixture_middle.so','-o',middle/'libfixture_middle.so'])
    for transitive in (False,True):
        root=tmp_path/('RPATH' if transitive else 'RUNPATH');(root/'evidence/sdk-build').mkdir(parents=True)
        binary=root/'evidence/sdk-build/client'
        args=[main,'-L'+str(middle),'-lfixture_middle','-Wl,-rpath-link,'+str(base),'-Wl,-rpath,'+str(middle)+':'+str(base),'-o',binary]
        if transitive:args.append('-Wl,--disable-new-dtags')
        build(args)
        if transitive:
            unit.runtime_link(root,str(binary),guard=lambda:None,deadline=time.time()+10)
            assert (root/'evidence/SDK_RUNTIME_LINK_RETURNED.json').is_file()
        else:
            with pytest.raises(PermissionError):unit.runtime_link(root,str(binary),guard=lambda:None,deadline=time.time()+10)
            assert 'not found' in (root/'evidence/SDK_RUNTIME_LINK.log').read_text()
        with pytest.raises(FileExistsError):unit.runtime_link(root,str(binary),guard=lambda:None,deadline=time.time()+10)
