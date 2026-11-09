"""Actual CPU MuJoCo FK plus hermetic renderer/capture seam doubles.

Pixels and software-GL identity below are explicit test doubles, not native
renderer qualification. There is no integration or policy/model query.
"""
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.pi0292.relay_protocol import digest as document_hash, sealed, unseal
from scripts.pi0294.task_state import ActualState
from go2arx5_vla.world_model import fusion_source_camera_capture as c
from go2arx5_vla.world_model import fusion_source_camera_rig as rig_module
from go2arx5_vla.world_model import fusion_paused_observation as paused
from go2arx5_vla.world_model.fusion_observation import TargetShadowBinding, camera_records, shadow_inference_payload
from go2arx5_vla.world_model.storage import digest
from tests.world_model.test_fusion_source_camera_rig import fixture as rig_fixture, FRAME, WRIST_Q

QUERY = '12345678-1234-4234-8234-123456789abc'


def scene(tmp_path):
    import mujoco
    bodies = []
    for i in range(20):
        if i == 0:
            body, joint = 'x5_link6', 'j0'
            extra = ('<site name="x5_ee" pos="0 0 .1"/>'
                '<camera name="wrist_rgb" pos=".1124 .1 .055" quat="' + ' '.join(map(str, WRIST_Q)) + '" fovy="75"/>')
        else:
            body = 'joint_body' + str(i)
            joint = ('task_left_slide' if i == 18 else 'task_right_slide' if i == 19 else 'j' + str(i))
            extra = ''
        geom = ('task_left_pad' if i == 18 else 'task_right_pad' if i == 19 else 'geom' + str(i))
        bodies.append(f'<body name="{body}" pos="0 0 .1"><joint name="{joint}" type="hinge" axis="0 0 1"/>'
                      f'<geom name="{geom}" size=".01" mass=".1" contype="0" conaffinity="0"/>{extra}</body>')
    joints = [('task_left_slide' if i == 18 else 'task_right_slide' if i == 19 else 'j' + str(i)) for i in range(20)]
    xml = ('<mujoco><option timestep=".002"/><visual><global offwidth="256" offheight="256"/></visual>'
        '<worldbody><light pos="0 0 3"/><geom type="plane" size="2 2 .1" rgba=".3 .3 .3 1"/>'
        '<body name="base_link" pos="0 0 .4"><freejoint/><geom size=".05" mass="1" contype="0" conaffinity="0"/>'
        '<camera name="body_rgb" pos=".25 -.24 .28" fovy="75"/>' + ''.join(bodies) + '</body>'
        '<body name="task_cube" pos=".4 0 .1"><freejoint/><geom name="task_cube_geom" size=".03" mass=".1" rgba="1 0 0 1"/></body>'
        '</worldbody><actuator>' + ''.join(f'<motor joint="{j}"/>' for j in joints) + '</actuator></mujoco>')
    model = mujoco.MjModel.from_xml_string(xml)
    assert (model.nq, model.nv, model.nu) == (34, 32, 20)
    path = tmp_path / 'scene.mjb'
    mujoco.mj_saveModel(model, str(path), None)
    data = mujoco.MjData(model)
    data.qpos[7] = .1
    data.time = 3703 * .002
    mujoco.mj_forward(model, data)
    packet = dict(step=3703, time_s=data.time, qpos=data.qpos.copy(), qvel=data.qvel.copy(), ctrl=data.ctrl.copy(),
                  contacts=[], contact_time_s=data.time - .002)
    raw = struct.pack('<8sQdIII', b'GRTSTATE', 3703, data.time, 34, 32, 20)
    raw += np.r_[data.qpos, data.qvel, data.ctrl].astype('<f8').tobytes() + struct.pack('<I', 0)
    return path, model, data, packet, raw


def recipe(monkeypatch):
    audit, _ = rig_fixture()
    raw = json.dumps(audit, allow_nan=False).encode()
    pin = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(c, 'SOURCE_IDENTITY_SHA256', pin)
    monkeypatch.setattr(rig_module, 'SOURCE_IDENTITY_SHA256', pin)
    return c.SourceCameraCapture(raw, FRAME)


def forbid_step(monkeypatch):
    import mujoco
    def blocked(*args, **kwargs):
        raise AssertionError('no physical integration permitted')
    monkeypatch.setattr(mujoco, 'mj_step', blocked)
    monkeypatch.setattr(mujoco, 'mj_step1', blocked)
    monkeypatch.setattr(mujoco, 'mj_step2', blocked)


def cpu_environment(monkeypatch):
    for key, value in dict(CUDA_VISIBLE_DEVICES='', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
            LIBGL_ALWAYS_SOFTWARE='1', __EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json').items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv('MUJOCO_EGL_DEVICE_ID', raising=False)


def test_actual_copy_world_reparent_fk_preserves_native_arrays_packet_and_wrist(tmp_path, monkeypatch):
    import mujoco
    path, model, data, packet, raw = scene(tmp_path)
    original = c._arrays(model)
    state = SimpleNamespace(model=model, data=data)
    native = camera_records(state, c._calibration(model, data, packet['time_s']), packet['time_s'])
    r = recipe(monkeypatch)
    rig = rig_module.SourceCameraRig(r.source_identity_bytes, FRAME, native,
        expected_native_camera_records_sha256=rig_module.record_hash(native))
    copied = mujoco.MjModel.from_binary_path(str(path)); copied_data = mujoco.MjData(copied)
    forbid_step(monkeypatch)
    c.apply_source_rig_to_render_copy(model, copied, copied_data, packet, rig)
    cid = int(copied.camera('body_rgb').id)
    assert int(copied.cam_bodyid[cid]) == 0 and copied.cam_fovy[cid] == 55
    assert model.cam_bodyid[cid] != 0 and model.cam_fovy[cid] == 75
    for name, value in original.items():
        np.testing.assert_array_equal(value, getattr(model, name))
    wrist = int(model.camera('wrist_rgb').id)
    np.testing.assert_allclose(data.cam_xpos[wrist], copied_data.cam_xpos[wrist], atol=1e-12)
    np.testing.assert_allclose(data.cam_xmat[wrist], copied_data.cam_xmat[wrist], atol=1e-12)
    for key in ('qpos', 'qvel', 'ctrl'):
        np.testing.assert_array_equal(packet[key], getattr(data, key))
    assert data.time == packet['time_s'] and path.read_bytes() and raw.startswith(b'GRTSTATE')
    with pytest.raises(ValueError, match='INDEPENDENT_RENDER_MODEL'):
        c.apply_source_rig_to_render_copy(model, model, data, packet, rig)


def test_episode_cache_matches_fresh_capture_across_packets_and_closes_contexts(tmp_path, monkeypatch):
    """Actual MuJoCo FK, deterministic renderer double; not pixel qualification."""
    import mujoco
    path,model,data,first,_=scene(tmp_path)
    r=recipe(monkeypatch)
    renderers=[]
    class Renderer:
        def __init__(self,model,*,height,width):
            self.model=model;self.closed=False;renderers.append(self)
        def update_scene(self,data,*,camera):
            self.data,self.camera=data,camera
        def render(self):
            cid=int(self.model.camera(self.camera).id)
            value=int(self.model.cam_fovy[cid]+self.data.qpos[7]*100+self.data.time)%256
            return np.full((256,256,3),value,dtype=np.uint8)
        def close(self):self.closed=True
    monkeypatch.setattr(mujoco,'Renderer',Renderer)
    forbid_step(monkeypatch)
    session=c.SourceCameraSession(path,r)
    retained={}
    packets=[first,dict(first,step=4703,time_s=9.406,qpos=first['qpos'].copy())]
    packets[1]['qpos'][7]=.3
    for i,packet in enumerate(packets):
        files=[]
        for cached in (False,True):
            folder=tmp_path/f'{i}-{cached}';(folder/'images').mkdir(parents=True)
            recorder=c.SourceCameraRecorder(path,folder,r,session=session if cached else None)
            recorder.save(0,packet)
            if cached:
                with pytest.raises(ValueError,match='EXCLUSIVE_CAMERA_SESSION'):
                    session.acquire(path,folder,r)
            c._packet_data(recorder.camera_adapter.camera_state.data,packet)
            files.append({p.relative_to(folder).as_posix():p.read_bytes()for p in folder.rglob('*')if p.is_file()})
            recorder.close()
            assert not session.busy
        assert files[0]==files[1]
        retained[i]=files[1]
    assert retained[0]['images/000000-wrist_rgb.png']!=retained[1]['images/000000-wrist_rgb.png']
    assert (tmp_path/'0-True/images/000000-wrist_rgb.png').read_bytes()==retained[0]['images/000000-wrist_rgb.png']
    assert len(renderers)==6 and sum(not x.closed for x in renderers)==2
    session.close();session.close()
    assert all(x.closed for x in renderers)
    with pytest.raises(ValueError,match='EXCLUSIVE_CAMERA_SESSION'):session.acquire(path,tmp_path,r)


def test_camera_cache_still_rejects_changed_non_camera_model_array(tmp_path,monkeypatch):
    import mujoco
    path,model,data,packet,_=scene(tmp_path);r=recipe(monkeypatch)
    class Renderer:
        def __init__(self,*args,**kwargs):pass
        def update_scene(self,*args,**kwargs):pass
        def render(self):return np.zeros((256,256,3),dtype=np.uint8)
        def close(self):pass
    monkeypatch.setattr(mujoco,'Renderer',Renderer)
    session=c.SourceCameraSession(path,r)
    try:
        folder=tmp_path/'one';(folder/'images').mkdir(parents=True)
        recorder=c.SourceCameraRecorder(path,folder,r,session=session);recorder.save(0,packet);recorder.close()
        session.model.body_mass[1]+=1
        folder=tmp_path/'two';(folder/'images').mkdir(parents=True)
        recorder=c.SourceCameraRecorder(path,folder,r,session=session)
        try:
            with pytest.raises(ValueError,match='FRESH_NATIVE_BINARY_RENDER_COPY_body_mass'):
                recorder.save(0,packet)
        finally:recorder.close()
    finally:session.close()


def adapter_fixture(tmp_path, monkeypatch):
    import mujoco
    cpu_environment(monkeypatch)
    class Renderer:
        def __init__(self, model, *, height, width):
            assert (height, width) == (256, 256)
            self.model = model
        def update_scene(self, data, *, camera): self.camera = camera
        def render(self):
            cid = int(self.model.camera(self.camera).id)
            return np.full((256, 256, 3), int(self.model.cam_fovy[cid]), dtype=np.uint8)
        def close(self): pass
    monkeypatch.setattr(mujoco, 'Renderer', Renderer)
    monkeypatch.setattr(paused, 'software_renderer_identity', lambda: dict(
        GL_VENDOR='Mesa SYNTHETIC TEST DOUBLE', GL_RENDERER='llvmpipe SYNTHETIC TEST DOUBLE', backend='egl_mesa_cpu'))
    root = tmp_path / 'run'; root.mkdir(parents=True)
    path, model, data, packet, raw = scene(root)
    native = root / 'native-binary'; native.write_bytes(b'fixture, never executed')
    output = root / 'native-01'; output.mkdir()
    spec = dict(registered_root=str(root), model_path=str(path), native_binary=str(native),
                stage_identity_sha256='a'*64, max_ticks=5003)
    (root / 'EXECUTION_SPEC.json').write_text(json.dumps(spec))
    (output / 'state.bin').write_bytes(raw)
    pause = dict(step=3703, time_s=packet['time_s'], running=False, clock_paused=True,
        ack_sequence=1, motiontime=3900, leg_history_steps=120, gait_indices=.3)
    (output / 'pause_ack.json').write_text(json.dumps(pause))
    (output / 'step_quota').write_text('3703\n')
    (output / 'quota.binding').write_text('\n'.join(('GRTQUOTA2', output.name, 'a'*64, output.name, '5003', '5', '32'))+'\n')
    class Adapter:
        def guard(self): pass
        def observe(self): return packet
        def capture_loaded_identity(self): return dict(sim=dict(pid=12345, executable_sha256=digest(native)))
        def spin(self, seconds, predicate):
            pause['ack_sequence'] += 1
            (output / 'pause_ack.json').write_text(json.dumps(pause))
            return predicate()
    adapter = Adapter(); adapter.spec = spec; adapter.output = output; adapter.actual_state = ActualState(model)
    binding = TargetShadowBinding('ACTUAL_TARGET_WORLD', digest(path), digest(root / 'EXECUTION_SPEC.json'),
        FRAME, rig_module.FRAME_MAP_SHA256)
    task = dict(schema=paused.TASK_SCHEMA, task_id='KNOWN_CALIBRATION', scene_sha256=digest(path),
        language='Move the open gripper a little toward the red cube, then hold without touching it.',
        score_contract_sha256='b'*64, language_provenance='NEW_TARGET_SHORT_SKILL_PROMPT')
    return adapter, binding, task, raw


def test_opt_in_paused_capture_reuses_contexts_and_closes_after_binding_failure(tmp_path, monkeypatch):
    """Exercise the real capture/session/finally wiring, with a renderer double."""
    import mujoco
    from go2arx5_vla.world_model import native_collection_performance_v1 as performance
    from go2arx5_vla.world_model.task_learning_grasp_v1 import CONSTRUCTION_ROLE
    adapter,binding,task,_=adapter_fixture(tmp_path,monkeypatch)
    adapter.spec.update(phase='collect-target-task-learning-grasp-v1',task_learning_role=CONSTRUCTION_ROLE,
        source_sha='c'*40,collection_performance=dict(version=performance.VERSION,mode='camera_cache'))
    spec_path=Path(adapter.spec['registered_root'])/'EXECUTION_SPEC.json'
    spec_path.write_text(json.dumps(adapter.spec))
    binding=TargetShadowBinding('ACTUAL_TARGET_WORLD',digest(adapter.spec['model_path']),digest(spec_path),FRAME,rig_module.FRAME_MAP_SHA256)
    adapter.performance=performance.PerformanceTrace()
    original=mujoco.Renderer;renderers=[]
    class Renderer(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.closed=False;renderers.append(self)
        def close(self):self.closed=True
    monkeypatch.setattr(mujoco,'Renderer',Renderer)
    r=recipe(monkeypatch);forbid_step(monkeypatch)
    first=paused.capture_paused_observation(adapter,binding=binding,task=task,
        task_sha256=document_hash(task),query_uuid=QUERY,camera_recipe=r)
    second=paused.capture_paused_observation(adapter,binding=binding,task=task,
        task_sha256=document_hash(task),query_uuid='22345678-1234-4234-8234-123456789abc',camera_recipe=r)
    assert len(renderers)==2 and not any(x.closed for x in renderers)
    a,b=unseal(first),unseal(second)
    assert a['query_uuid']!=b['query_uuid']
    assert {k:v for k,v in a['files'].items()if k!='BOUND_SHADOW_ABI.private.json'}=={
        k:v for k,v in b['files'].items()if k!='BOUND_SHADOW_ABI.private.json'}
    adapter._source_camera_session.model.body_mass[1]+=1
    with pytest.raises(ValueError,match='FRESH_NATIVE_BINARY_RENDER_COPY_body_mass'):
        paused.capture_paused_observation(adapter,binding=binding,task=task,
            task_sha256=document_hash(task),query_uuid='32345678-1234-4234-8234-123456789abc',camera_recipe=r)
    assert not adapter._source_camera_session.busy
    failed=adapter.output/'PAUSED_OBSERVATIONS/32345678-1234-4234-8234-123456789abc/CAPTURE_FAILED.private.json'
    assert failed.exists()
    performance.close_resources(adapter)
    assert all(x.closed for x in renderers)
    report=json.loads((adapter.output/'COLLECTION_PERFORMANCE.private.json').read_text())
    assert report['spans']['camera.capture_total']['count']==3
    assert report['spans']['camera.capture_total']['failures']==1 and report['active_spans']==0


def test_cpu_capture_retains_original_views_and_explicit_new_sensor_same_packet(tmp_path, monkeypatch):
    adapter, binding, task, raw = adapter_fixture(tmp_path, monkeypatch)
    original_arrays = c._arrays(adapter.actual_state.model)
    model_bytes = Path(adapter.spec['model_path']).read_bytes()
    r = recipe(monkeypatch)
    forbid_step(monkeypatch)
    record = paused.capture_paused_observation(adapter, binding=binding, task=task,
        task_sha256=document_hash(task), query_uuid=QUERY, camera_recipe=r)
    body = unseal(record); shadow = unseal(body['original_shadow_ABI'])
    payload = paused.paused_inference_payload(record)
    target = shadow['target_observation']; sensor = target['sensor_adapter']
    assert sensor['rig'] == 'SIM_EXTERNAL_WORLD_SOURCE_RIG'
    assert sensor['status'] == 'ACTUAL_TARGET_PACKET_CPU_RENDERED'
    assert set(target['cameras']) == {rig_module.OVERLAY_CAMERA, 'wrist_rgb'}
    assert sensor['native_target_camera_records']['body_rgb']['fovy_deg'] == 75
    assert target['cameras'][rig_module.OVERLAY_CAMERA]['fovy_deg'] == 55
    assert sensor['effects']['integrations'] == 0 and sensor['source_images_supplied'] is False
    folder = adapter.output / 'PAUSED_OBSERVATIONS' / QUERY
    assert (folder / 'images/000000-wrist_rgb.png').read_bytes() == (folder / 'native-camera/images/000000-wrist_rgb.png').read_bytes()
    assert digest(folder / 'images/000000-body_rgb.png') != digest(folder / 'native-camera/images/000000-body_rgb.png')
    assert payload['images'][0]['sha256'] == digest(folder / 'images/000000-body_rgb.png')
    assert body['renderer']['backend'] == 'egl_mesa_cpu'
    assert all(digest(folder / name) == pin for name, pin in body['files'].items())
    assert (adapter.output / 'state.bin').read_bytes() == raw
    assert Path(adapter.spec['model_path']).read_bytes() == model_bytes
    for name, array in original_arrays.items():
        np.testing.assert_array_equal(array, getattr(adapter.actual_state.model, name))
    assert paused.verify_still_paused(adapter, record)['native_step'] == 3703


@pytest.mark.parametrize('fault', ['rig', 'pin', 'frame', 'grant', 'camera_name', 'wrist_hash', 'files'])
def test_source_sensor_capture_sidecar_failures_rejected(tmp_path, monkeypatch, fault):
    adapter, binding, task, _ = adapter_fixture(tmp_path, monkeypatch)
    r = recipe(monkeypatch); forbid_step(monkeypatch)
    record = paused.capture_paused_observation(adapter, binding=binding, task=task,
        task_sha256=document_hash(task), query_uuid=QUERY, camera_recipe=r)
    body = unseal(record); shadow = unseal(body['original_shadow_ABI'])
    sensor = shadow['target_observation']['sensor_adapter']
    if fault == 'rig': sensor['rig'] = 'ORIGINAL_TARGET_NATIVE_DUAL_CAMERA_V1'
    if fault == 'pin': sensor['source_identity_audit_sha256'] = '0'*64
    if fault == 'frame': sensor['frame_map_sha256'] = '0'*64
    if fault == 'grant': sensor['native_execution_authorized'] = True
    if fault == 'camera_name': shadow['target_observation']['cameras'][rig_module.OVERLAY_CAMERA]['name'] = 'body_rgb'
    if fault == 'wrist_hash': sensor['native_image_sha256']['wrist_rgb'] = '0'*64
    if fault == 'files': body['files']['native-camera/images/000000-calibration.json'] = '0'*64
    body['original_shadow_ABI'] = sealed(shadow)
    with pytest.raises(ValueError): paused.paused_inference_payload(sealed(body))


def test_noncamera_render_copy_mutation_and_untyped_capture_are_rejected(tmp_path, monkeypatch):
    import mujoco
    path, model, data, packet, _ = scene(tmp_path)
    copied = mujoco.MjModel.from_binary_path(str(path))
    copied.body_mass[1] += .1
    with pytest.raises(ValueError, match='NONDECLARED_ARRAY_body_mass'):
        c._same_model_except_external(model, copied)
    adapter, binding, task, _ = adapter_fixture(tmp_path / 'capture', monkeypatch)
    with pytest.raises(ValueError, match='TYPED_SOURCE_CAMERA_RECIPE'):
        paused.capture_paused_observation(adapter, binding=binding, task=task,
            task_sha256=document_hash(task), query_uuid=QUERY, camera_recipe=object())


@pytest.mark.parametrize('fault', ['physics_array', 'physics_option', 'copy_time', 'copy_qpos', 'wrist_pixels', 'external_pixels'])
def test_capture_rejects_camera_adapter_with_changed_physics_or_packet(tmp_path, monkeypatch, fault):
    adapter, binding, task, raw = adapter_fixture(tmp_path, monkeypatch)
    r = recipe(monkeypatch); forbid_step(monkeypatch)
    original = c.CapturedSourceCameraAdapter.validate
    def mutate(self, **kwargs):
        if fault == 'physics_array': self.camera_state.model.body_mass[1] += .1
        if fault == 'physics_option': self.camera_state.model.opt.timestep = .01
        if fault == 'copy_time': self.camera_state.data.time += .002
        if fault == 'copy_qpos': self.camera_state.data.qpos[0] += .1
        if fault == 'wrist_pixels': self.native_image_bytes['wrist_rgb'] += b'changed'
        if fault == 'external_pixels': kwargs['image_bytes']['body_rgb'] = self.native_image_bytes['body_rgb']
        return original(self, **kwargs)
    monkeypatch.setattr(c.CapturedSourceCameraAdapter, 'validate', mutate)
    with pytest.raises(ValueError):
        paused.capture_paused_observation(adapter, binding=binding, task=task,
            task_sha256=document_hash(task), query_uuid=QUERY, camera_recipe=r)
    assert (adapter.output / 'state.bin').read_bytes() == raw
    assert adapter.actual_state.model.opt.timestep == .002
    assert (adapter.output / 'PAUSED_OBSERVATIONS' / QUERY / 'CAPTURE_FAILED.private.json').exists()


def test_import_does_not_load_mujoco_torch_ros_or_opengl():
    code = "import sys;import go2arx5_vla.world_model.fusion_source_camera_capture;assert not any(x in sys.modules for x in ('mujoco','torch','rclpy','OpenGL'))"
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_original_native_body_png_cannot_replace_new_external_render_before_binding(tmp_path, monkeypatch):
    adapter, binding, task, _ = adapter_fixture(tmp_path, monkeypatch)
    r = recipe(monkeypatch); forbid_step(monkeypatch)
    save = c.SourceCameraRecorder.save
    def replace_after_render(self, index, packet):
        result = save(self, index, packet)
        (self.folder / 'images' / f'{index:06d}-body_rgb.png').write_bytes(
            self.camera_adapter.native_image_bytes['body_rgb'])
        return result
    monkeypatch.setattr(c.SourceCameraRecorder, 'save', replace_after_render)
    with pytest.raises(ValueError, match='RENDERED_EXTERNAL_PNG_UNCHANGED'):
        paused.capture_paused_observation(adapter, binding=binding, task=task,
            task_sha256=document_hash(task), query_uuid=QUERY, camera_recipe=r)
