"""Real file/finite-process CPU tests; no native runtime or fake admission."""
import base64
from copy import deepcopy
import os
from pathlib import Path
import signal
import sys
import time
import pytest

from . import target_native_io_entry as unit
from . import target_native_io_graph as graph
from . import target_native_io_ros_environment as ros_environment
from .test_target_native_io_resources import proposal,reserve,manifest
from .test_target_native_io_budget import context
from .test_target_sdk_budget import bundle,NOW
from .test_target_sdk_resources import packet
from .test_target_sdk_closed_costs import closed


def write(path,v):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(unit.budget.common.encode(v))


@pytest.fixture
def owned(proposal,tmp_path,monkeypatch):
    """All files are isolated CPU doubles; no fixture is a native proof."""
    v=deepcopy(proposal);b=unit.budget;c=b.common;m=manifest(v);q=v['q']
    # ROS origins/loader startup have their own isolated tests. These existing
    # fixtures exercise every standard entry and paired resource rejection.
    monkeypatch.setattr(ros_environment,'validate_binding',lambda inputs:None)
    root=tmp_path/b.CAMPAIGN/'children'/q['child_run_uuid'];root.mkdir(parents=True)
    parent=root.parent.parent;code=root/'code/capsule';code.mkdir(parents=True)
    monkeypatch.setattr(unit,'__file__',str(code/'scripts/pi0294/dec1/target_native_io_entry.py'))
    monkeypatch.setattr(unit.time,'time',lambda:NOW+1)
    monkeypatch.setattr(unit.os,'getuid',lambda:2009)
    monkeypatch.setattr(unit.os,'uname',lambda:type('Host',(),dict(nodename='corelab-4080s'))())
    asset=tmp_path/'CPU_ONLY_ASSET';asset.write_bytes(b'isolated asset fixture')
    clock=tmp_path/'CPU_ONLY_CLOCK';clock.write_bytes(b'isolated clock fixture')
    entry=tmp_path/'CPU_ONLY_ENTRY';entry.write_bytes(b'isolated entry fixture')
    doctor=tmp_path/'CPU_ONLY_DOCTOR';doctor.write_bytes(b'isolated doctor fixture')
    monkeypatch.setattr(unit,'SIM_SHA',unit.filehash(asset));monkeypatch.setattr(unit,'FROZEN_CLOCK',unit.filehash(clock))
    monkeypatch.setattr(unit.contract,'FROZEN_CLOCK',unit.filehash(clock))
    monkeypatch.setattr(unit,'ENTRY_SHA',unit.filehash(entry));monkeypatch.setattr(unit,'DOCTOR_SHA',unit.filehash(doctor))
    include=tmp_path/'include';(include/'mujoco').mkdir(parents=True)
    (include/'mujoco/mujoco.h').write_text('#define mjVERSION_HEADER 327\n')
    headers={'mujoco/mujoco.h':unit.filehash(include/'mujoco/mujoco.h')}
    originals={str(p):unit.filehash(p) for p in (asset,clock,include/'mujoco/mujoco.h')}
    monkeypatch.setattr(unit,'NATIVE_PINS',{k:unit.filehash(asset) for k in unit.NATIVE_PINS})
    monkeypatch.setattr(unit,'NATIVE_RELATIVE_PINS',{})
    cache=root/'original-math-cache';cache.mkdir();(cache/'CPU_FIXTURE.so').write_bytes(b'not a library')
    (cache/'CPU_FIXTURE.so').chmod(0o400)
    task=root/'cache-only-task.info';task.write_bytes(b'isolated cache-only task fixture');task.chmod(0o400)
    monkeypatch.setattr(unit,'CACHE_ONLY_TASK',unit.filehash(task))
    inputs=dict(runtime_python=sys.executable,frozen_clock=str(clock),sim_binary=str(asset),compiler=str(asset),
        xvfb_run=str(asset),solver_binary=str(asset),controller_binary=str(asset),original_task_observer=str(asset),
        original_task=str(asset),urdf=str(asset),compiled_model=str(asset),original_readonly_files=originals,
        mujoco_include=str(include),mujoco_headers=headers,ROS_DOMAIN_ID=169,native_root=str(tmp_path),
        codegen_directory=str(cache),codegen_members={'CPU_FIXTURE.so':unit.filehash(cache/'CPU_FIXTURE.so')})
    inputs['runtime_cache_only_task']=str(task)
    write(root/'INPUT_BINDINGS.private.json',inputs)
    pop=b'{"isolated_population_fixture":true}\n';geometry={'CPU_fixture_only':True}
    monkeypatch.setattr(c,'plan',lambda raw:geometry if raw==pop else c.need(False,'BAD_CPU_POPULATION'))
    (root/'plans').mkdir();(root/'plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json').write_bytes(pop)
    write(root/'plans/ZERO_IO_GEOMETRY.json',geometry);write(root/unit.NAMESPACE,v['space'])
    write(root/'REQUEST.json',q);write(root/b.PROJECTION,v['x'])
    (root/b.FORECAST).write_bytes(b.forecast_pack(v['inputs'],DEC1_bytes=c.encode(v['d']),SDK_closures=v['closures']))
    for folder,name,raw in (('DEC1','DEC1_NATIVE_ADOPTION_AND_BUDGET.json',b'{}\n'),
            ('VAP1','VAP1_ADOPTION_AND_DELTA_BUDGET.json',v['VAP1'])):
        write(parent/'authority'/folder/name,{})
        (parent/'authority'/folder/name).write_bytes(raw);(root/'plans'/name).write_bytes(raw)
    source_files={}
    pub=Path(__file__).resolve().parents[3]
    for name in unit.REQUIRED_CODE:
        rel='scripts/pi0294/dec1/'+name;raw=('isolated source fixture '+name).encode()
        p=code/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);p.chmod(0o400)
        source_files[rel]=c.sha(raw)
    for rel,pin in c.FROZEN_SOURCES.items():
        # These are the actual published frozen geometry sources, never native
        # assets. The fixture does not replace their accepted hash contract.
        raw=(pub/rel).read_bytes();assert c.sha(raw)==pin
        p=code/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);p.chmod(0o400);source_files[rel]=pin
    seal=dict(source_sha=m['code']['vla_commit'],files=source_files)
    write(root/'SOURCE_SEAL.json',seal)
    write(root/'plans/PUBLICATION_AND_CI.private.json',dict(source=seal['source_sha'],CI_conclusion='success',
        all_12_steps_success=True,full_readback_file_sha256=source_files))
    role=b.ROLE+'__'+root.name;lease=root/'lease.json';registry=tmp_path/'registry.json'
    binding=dict(run_root=str(root),uid=2009,role=role,lease=str(lease),registry=str(registry))
    write(root/'OUTPUT_BINDING.private.json',binding)
    write(lease,dict(status='ACTIVE',scope=b.ROLE,run_uuid=root.name,deadline_epoch=q['deadline_epoch'],
        authority_release=c.PARENT_AUTHORITY))
    write(registry,dict(entries=[dict(role=role,run_uuid=root.name,path=str(root))]))
    write(parent/'parentlease.json',dict(status='ACTIVE_CONTROL_PLANE_ONLY',authority_release=c.PARENT_AUTHORITY))
    write(parent/'OUTPUT_BINDING.private.json',dict(lease=str(parent/'parentlease.json')))
    m['code'].update(capsule_relative='code/capsule',source_seal_relative='SOURCE_SEAL.json',source_seal_sha256=unit.filehash(root/'SOURCE_SEAL.json'))
    m['limits'].update(cpus=[min(os.sched_getaffinity(0))],memory_bytes=8*1024**3)
    m['execution'].update(host='corelab-4080s',uid=2009,private_path_role=role,entry_path=str(entry),entry_sha256=unit.ENTRY_SHA,
        doctor_path=str(doctor),doctor_sha256=unit.DOCTOR_SHA,input_binding_sha256=unit.filehash(root/'INPUT_BINDINGS.private.json'),
        output_binding_sha256=unit.filehash(root/'OUTPUT_BINDING.private.json'),frozen_task_clock_sha256=unit.FROZEN_CLOCK)
    m['execution']['sealed_inputs']['original-math-cache/CPU_FIXTURE.so']=unit.filehash(cache/'CPU_FIXTURE.so')
    m['execution']['sealed_inputs']['cache-only-task.info']=unit.filehash(task)
    for p in root.rglob('*'):
        if p.is_file() and (p.parent==root/'plans' or p.name in ('REQUEST.json','INPUT_BINDINGS.private.json')):
            p.chmod(0o400);m['execution']['sealed_inputs'][str(p.relative_to(root))]=unit.filehash(p)
    write(root/'RUN_MANIFEST.json',m);write(code.parent/'WORKER_RUN_MANIFEST.json',m)
    r=reserve(v,m=m);row=r['reservation']
    write(parent/'authority/DEC1/DEC1_RESOURCE_LEDGER.json',r['proposed_Core'])
    write(root/'LOCAL_RESERVATION_CONFIRMED.private.json',dict(status='ACTUAL_LOCAL_PARENT_RESERVATION_CONFIRMED',
        request_id=q['request_id'],manifest_sha256=row['manifest_sha256'],reservation=row,
        parent_after_base64=base64.b64encode(c.encode(r['proposed_parent'])).decode(),parent_after_sha256=c.sha(c.encode(r['proposed_parent']))))
    for name in ('graph','build'):(root/'evidence'/name).mkdir(parents=True)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');monkeypatch.delenv('LD_PRELOAD',raising=False)
    monkeypatch.delenv('LD_AUDIT',raising=False);monkeypatch.delenv('PBS_JOBID',raising=False)
    monkeypatch.setenv('GR00T_WBC_PYTHON',sys.executable);monkeypatch.setenv('PV1_RUN_MANIFEST',str(code.parent/'WORKER_RUN_MANIFEST.json'))
    return root,m,q,binding


def test_complete_files_and_real_resource_recomputation_before_native_import(owned):
    root,m,q,b=owned
    before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    actual=unit.validate(root)
    assert actual[:2]==(m,q);actual[3]()
    assert before=={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('fault',['lease_closed','missing_LOCAL','changed_Core','wrong_source','extra_header',
    'asset_changed','old_role','bad_clock','old_entry','dirty_graph','expired','unsealed_request','wrong_profile','unknown_Core'])
def test_current_entry_rejects_before_native_dispatch(owned,monkeypatch,fault):
    root,m,q,b=owned
    if fault=='lease_closed':v=unit.read(b['lease']);v['status']='CLOSED';write(Path(b['lease']),v)
    elif fault=='missing_LOCAL':(root/'LOCAL_RESERVATION_CONFIRMED.private.json').unlink()
    elif fault in ('changed_Core','unknown_Core'):
        p=root.parent.parent/'authority/DEC1/DEC1_RESOURCE_LEDGER.json';v=unit.read(p)
        if fault=='changed_Core':v[unit.budget.ROWS][q['request_id']]['remaining']['constructors']=0
        else:v['unknown_effects']=['CPU-fixture-unknown']
        write(p,v)
    elif fault=='wrong_source':p=root/'code/capsule/scripts/pi0294/dec1/target_zero_quota.py';p.chmod(0o600);p.write_text('changed')
    elif fault=='extra_header':Path(unit.read(root/'INPUT_BINDINGS.private.json')['mujoco_include'],'unsealed.h').touch()
    elif fault=='asset_changed':Path(unit.read(root/'INPUT_BINDINGS.private.json')['sim_binary']).write_text('changed')
    elif fault=='old_role':m['scope']='OLD_R4';write(root/'RUN_MANIFEST.json',m)
    elif fault=='bad_clock':monkeypatch.setattr(unit,'FROZEN_CLOCK','f'*64)
    elif fault=='old_entry':Path(m['execution']['entry_path']).write_text('different')
    elif fault=='dirty_graph':(root/'evidence/graph/old-intent').touch()
    elif fault=='expired':monkeypatch.setattr(unit.time,'time',lambda:q['deadline_epoch']+1)
    elif fault=='unsealed_request':p=root/'REQUEST.json';p.chmod(0o600);p.write_text('{}')
    elif fault=='wrong_profile':monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    called=[];monkeypatch.setattr(graph,'run_graph',lambda *a:called.append(a))
    with pytest.raises((PermissionError,KeyError,FileNotFoundError)):unit.run(root)
    assert called==[] and not(root/'evidence/NATIVE_IO_OWNER_INTENT.json').exists()


def test_real_finite_group_teardown_and_no_restart_after_lost_started_publication(tmp_path,monkeypatch):
    groups=graph.OwnedGroups(tmp_path,lambda:None);original=graph.put
    def put(path,value):
        if path.name.endswith('_STARTED.json'):raise OSError('CPU fixture lost durable return')
        return original(path,value)
    monkeypatch.setattr(graph,'put',put)
    with pytest.raises(OSError):groups.launch('one',[sys.executable,'-c','import time;time.sleep(30)'],cwd=tmp_path,env=os.environ.copy())
    assert len(groups.children)==1
    groups.close();groups.close()
    assert groups.children[0][1].poll() is not None
    with pytest.raises(PermissionError):groups.launch('two',[sys.executable,'-c','pass'],cwd=tmp_path,env=os.environ.copy())
    assert len(groups.children)==1 and (tmp_path/'one_INTENT.json').exists() and not(tmp_path/'two_INTENT.json').exists()
