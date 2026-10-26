"""Isolated CPU regressions of the new adapters and unchanged legacy fences."""
import ast
import base64
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace,ModuleType

import pytest
from scripts.pi0294.pv1 import mix_vanda_eval_identity as identity,mix_vanda_eval_context as context
from scripts.pi0294.pv1 import mix_vanda_reference as reference,mix_evaluation as plans,mix_accounting as a
from scripts.pi0294.pv1 import control_budget,control_heartbeat,evaluation_platform,evaluation_plan
from scripts.pi0294.pv1 import transport_fixture,mix_vanda_eval_output as output
from scripts.pi0294.test_v5_vanda_reference import fixture_plan


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(a.raw(value));path.chmod(0o400)


def typed_manifest(arm='C50'):
    return dict(scope=identity.NEURAL_ROLE,platform_evaluation_version=identity.VERSION,candidate=arm,
        authority=dict(protocol_ref=identity.s.TASK_SOURCE,scope_grant_sha256=identity.s.GRANT_SHA),
        execution=dict(profile=identity.NEURAL_PROFILE,control_budget_profile=control_budget.PROFILE,
                       owner_heartbeat_policy=control_heartbeat.POLICY))


def test_original_control_policy_is_explicit_and_RPC_cannot_renew_owner(tmp_path):
    m=typed_manifest();stage=plans.stage('C50',colocated=True);put(tmp_path/'PV1_STAGE_BINDING.json',stage)
    m['owner']='EXISTING_LOCAL_PC'
    assert control_budget.validate(m,stage)['max_calls']==150000
    assert control_heartbeat.owner_only(tmp_path,m) is True
    assert plans.stage('C50')!=stage  # legacy Core input builder stays unchanged
    wrong=deepcopy(m);wrong['execution']['profile']='4080-neural-evaluation'
    with pytest.raises(PermissionError):evaluation_platform.file_RPC_profile(wrong,'hopper')
    wrong=deepcopy(m);wrong['authority']['protocol_ref']='OLD'
    with pytest.raises(PermissionError):control_heartbeat.owner_only(tmp_path,wrong)


def test_original_wire_keeps_PBS_handle_and_new_typed_status_separate_from_Core():
    m=typed_manifest()
    assert evaluation_platform.file_RPC_profile(m,'hopper')==identity.NEURAL_PROFILE
    assert transport_fixture.allocation_status(m,core_typed=False,typed=False)==identity.STATUS
    assert evaluation_platform.fixture_job_pattern(m,'hopper')==r'[0-9]+\.stdct-mgmt-02'
    m.update(scope=identity.PHYSICAL_ROLE,evaluation_peer_platform=identity.VERSION)
    m.pop('platform_evaluation_version');m['execution']['profile']=identity.PHYSICAL_PROFILE
    assert evaluation_platform.fixture_job_pattern(m,'core')==r'[0-9]+\.stdct-mgmt-02'


def test_original_loader_accepts_only_nine_sealed_input_links(tmp_path):
    source=tmp_path/'payload';source.mkdir();plan=fixture_plan(source)
    root=tmp_path/'neural';root.mkdir();view=root/'model_views/C50';view.parent.mkdir()
    reference.create_view(view,plan,reference.audit_reference(plan))
    m=typed_manifest();p=root/'plans/VANDA_READONLY_INFERENCE_REFERENCE.private.json';put(p,plan)
    m['execution']['sealed_inputs']={str(p.relative_to(root)):a.sha(p.read_bytes())};put(root/'RUN_MANIFEST.json',m)
    name=next(iter(plan['files']));pin=plan['files'][name]['sha256']
    before=reference.metadata(plan['files'][name]['path'])
    assert reference.inference_member(root,view,name,pin)==view/name
    assert reference.metadata(plan['files'][name]['path'])==before
    for changed in ('optimizer.pt','../source','/absolute'):
        with pytest.raises(PermissionError):reference.inference_member(root,view,changed,pin)
    with pytest.raises(PermissionError):reference.inference_member(root,view,name,'0'*64)
    # An extra output cannot be hidden under the excluded model-input view.
    assert output.reference_view(root,plan)==view.parent
    (view/'hidden-output').write_bytes(b'x')
    with pytest.raises(PermissionError):output.reference_view(root,plan)


def test_new_output_partition_preserves_every_raw_file_on_overflow(tmp_path):
    parent=tmp_path/'campaign';neural=parent/'neural-children/panel';physical=parent/'children/panel';owner=parent/'control/owner'
    for p in (neural,physical,owner):p.mkdir(parents=True)
    source=tmp_path/'payload';source.mkdir();plan=fixture_plan(source)
    view=neural/'model_views/C50';view.parent.mkdir();reference.create_view(view,plan,reference.audit_reference(plan))
    (physical/'episodes').mkdir();(owner/'resident').mkdir()
    assert output.enforce(neural,physical,owner,plan)['raw_deleted_or_truncated'] is False
    episode=physical/'episodes/condition';episode.mkdir();p=episode/'raw.bin'
    with p.open('wb') as f:f.truncate(768*1024**2+1)
    with pytest.raises(PermissionError):output.enforce(neural,physical,owner,plan)
    assert p.exists() and p.stat().st_size==768*1024**2+1


def test_source_render_reuses_original_counter_and_never_calls_integrator():
    from scripts.pi0294.pv1 import mix_vanda_eval_render as render
    tree=ast.parse(Path(render.__file__).read_bytes())
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    calls=[ast.unparse(n.func) for n in ast.walk(run) if isinstance(n,ast.Call)]
    assert 'core.qualify' in calls and 'ledger.charge' not in calls
    assert not any(n in calls for n in ('mujoco.mj_step','engine.step','engine.integrate'))
    statements=[ast.unparse(n) for n in run.body]
    assert next(i for i,x in enumerate(statements) if 'context.validate(root)' in x)<statements.index('import mujoco')


def test_no_native_import_at_new_cold_context_and_failed_entry_cannot_load_mujoco(tmp_path,monkeypatch):
    from scripts.pi0294.pv1 import mix_vanda_eval_render as render
    for module in (context,identity):
        tree=ast.parse(Path(module.__file__).read_bytes())
        top=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
        assert all(not any(x in ast.unparse(n) for x in ('import mujoco','import torch','import OpenGL')) for n in top)
    def deny(*args,**kwargs):raise PermissionError('ACTUAL_ENTRY_DENIED_FIXTURE')
    monkeypatch.setattr(render.context,'validate',deny)
    with pytest.raises(PermissionError,match='ENTRY_DENIED'):render.run(tmp_path)
    assert list(tmp_path.iterdir())==[]


def test_original_metrics_default_and_explicit_existing_control_upper():
    from scripts.pi0294.rpc_metrics import SessionMetrics
    old=SessionMetrics();old.operations=40000
    with pytest.raises(PermissionError):old.reply({},lambda x:None)
    new=SessionMetrics(maximum=150000);new.operations=40000;new.reply({},lambda x:None)
    assert new.operations==40001
    with pytest.raises(PermissionError):SessionMetrics(maximum=150001)


def test_counter_wrapper_keeps_two_original_constructors_and_all_native_state(tmp_path,monkeypatch):
    """Native packages/engine are explicit CPU fakes; original counter is real."""
    import numpy as np
    monkeypatch.setattr(np,'__version__','1.26.4')
    from scripts.pi0294.pv1 import mix_vanda_eval_render as render,capture
    import scripts.pi0294 as source_package
    import scripts.pi0294.pv1 as pv1_package
    from scripts.pi0294 import s1_preflight
    count=[]
    class Model:
        @staticmethod
        def from_xml_string(*args,**kwargs):return Model()
    mujoco=ModuleType('mujoco');mujoco.__version__='3.2.7';mujoco.MjModel=Model
    monkeypatch.setitem(sys.modules,'mujoco',mujoco)
    class Engine:
        def __init__(self):
            self.model=mujoco.MjModel.from_xml_string('original-fixture');self.steps=0;self.data=SimpleNamespace(time=0.);self.xml='same';self.closed=False
        def snapshot(self):return dict(time=0.)
        def render(self,state):
            image=np.tile(np.arange(256,dtype=np.uint8)[None,:,None],(256,1,3))
            return dict(body_rgb=image,wrist_rgb=image.copy()),dict(body_rgb={},wrist_rgb={})
        def close(self):self.closed=True
    original_factory=lambda *args,**kwargs:Engine()
    scene=SimpleNamespace(make_engine=original_factory)
    monkeypatch.setattr(source_package,'scene',scene,raising=False);monkeypatch.setitem(sys.modules,'scripts.pi0294.scene',scene)
    monkeypatch.setattr(s1_preflight,'fingerprint',lambda e:'original-zero-time-fixture')
    def original_counter(root,label):
        class Meter:
            def __init__(self):self.root=root;self.plan=dict(cards=[dict(task='T1')]);self.inputs=dict(asset_root='fixture')
            def start(self):count.append('real-original-counter-call')
        capture.qualify_counter(Meter())
        return json.loads((root/'evidence/COUNTER_NATIVE_QUALIFICATION.json').read_bytes())
    core=SimpleNamespace(qualify=original_counter);monkeypatch.setattr(pv1_package,'core',core,raising=False)
    monkeypatch.setitem(sys.modules,'scripts.pi0294.pv1.core',core)
    gl=ModuleType('OpenGL.GL');gl.GL_VENDOR=1;gl.GL_RENDERER=2;gl.glGetString=lambda k:b'Mesa fixture' if k==1 else b'llvmpipe fixture'
    ogl=ModuleType('OpenGL');ogl.GL=gl;monkeypatch.setitem(sys.modules,'OpenGL',ogl);monkeypatch.setitem(sys.modules,'OpenGL.GL',gl)
    monkeypatch.setattr(render.context,'validate',lambda root:(dict(scope=identity.PHYSICAL_ROLE,candidate='C50',code=dict(vla_commit='a'*40)),{},{}))
    monkeypatch.setenv('PBS_JOBID','1234567.stdct-mgmt-02')
    (tmp_path/'evidence').mkdir()
    out=render.run(tmp_path)
    assert count==['real-original-counter-call']*2
    assert out['constructors']==2 and out['physics_ticks']==out['episode_starts']==out['model_loads']==out['queries']==0
    assert out['original_counter_qualification']['model_constructor_attempts']==2
    assert len(list((tmp_path/'evidence/COLOCATED_ZERO_INTEGRATION_SOURCE').glob('*.png')))==4
    assert scene.make_engine is original_factory and mujoco.MjModel is Model


def test_clean_container_preserves_registered_worker_manifest_and_rejects_another_role():
    from scripts.pi0294.pv1.mix_vanda_container import physics_command
    args=dict(code_root='/sealed/capsule/code',container='/original.sif',readonly=['/input'],writable=['/output'],
        module='scripts.pi0294.pv1.mix_vanda_eval_render',worker_manifest='/sealed/capsule/WORKER_RUN_MANIFEST.json')
    command=physics_command({'PBS_JOBID':'1234567.stdct-mgmt-02'},**args)
    assert 'PV1_RUN_MANIFEST=/sealed/capsule/WORKER_RUN_MANIFEST.json' in command
    assert 'PBS_JOBID=1234567.stdct-mgmt-02' in command and 'CUDA_VISIBLE_DEVICES=' in command
    args['worker_manifest']='/other/WORKER_RUN_MANIFEST.json'
    with pytest.raises(PermissionError):physics_command({'PBS_JOBID':'1234567.stdct-mgmt-02'},**args)
