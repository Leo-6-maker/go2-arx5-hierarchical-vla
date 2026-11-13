"""Immutable ownership, slow storage and real FK; pixels here are doubles."""
import json
from pathlib import Path
import struct
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from go2arx5_vla.world_model.native_frame_writer_v3 import FrozenFrame,FrameWriter,PIXEL_BYTES
from go2arx5_vla.world_model import native_body_memory_v3 as memory


def frozen(query='frame1'):
    pixels=np.zeros((256,256,3),np.uint8)
    return FrozenFrame(json.dumps(dict(query_uuid=query)).encode(),b'state',pixels.tobytes(),
                       pixels.tobytes(),pixels.tobytes(),b'{}',b'{}')


def test_both_consumers_keep_immutable_owned_bytes_after_original_buffer_reuse():
    a=np.full((256,256,3),17,np.uint8)
    frame=FrozenFrame(b'{"query_uuid":"one"}',b'state',a.tobytes(),a.tobytes(),a.tobytes(),b'{}',b'{}')
    policy=frame.pixels('body');a[:]=99
    assert np.all(policy==17)
    with pytest.raises(ValueError):policy.setflags(write=True)
    with pytest.raises(ValueError):policy[0,0,0]=3
    assert frame.nbytes>=3*PIXEL_BYTES
    with pytest.raises(ValueError,match='IMMUTABLE'):
        FrozenFrame(bytearray(frame.header),frame.native_state,frame.native_body,frame.native_wrist,
                    frame.body,frame.native_calibration,frame.calibration)


def test_slow_writer_counts_inflight_frame_and_bytes_without_blocking_handoff():
    entered,release=threading.Event(),threading.Event();threads=[]
    def publish(frame):
        threads.append(threading.get_ident());entered.set();assert release.wait(5)
        return dict(query_uuid=json.loads(frame.header)['query_uuid'],committed=True)
    first=frozen();writer=FrameWriter(publish,max_frames=1,max_bytes=first.nbytes)
    try:
        writer.submit(first);assert entered.wait(2)
        assert writer.report()['pending_payload_bytes']==first.nbytes
        with pytest.raises(BufferError):writer.submit(frozen('frame2'))
        with pytest.raises(TimeoutError):writer.wait_room(first.nbytes,timeout=.01)
        assert not writer.committed and all(x!=threading.get_ident()for x in threads)
    finally:release.set();writer.close()
    assert writer.report()['committed']==writer.report()['accepted']==1
    assert writer.report()['pending_payload_bytes']==0


def test_writer_failure_is_retained_and_unknown_or_duplicate_frame_never_replayed():
    def fail(frame):raise OSError('injected storage failure')
    writer=FrameWriter(fail);writer.submit(frozen())
    with pytest.raises(RuntimeError,match='PUBLICATION_FAILED'):writer.drain()
    with pytest.raises(RuntimeError,match='PUBLICATION_FAILED'):writer.close()
    assert writer.report()['committed']==0 and writer.report()['accepted']==1
    writer=FrameWriter(lambda f:dict(query_uuid=json.loads(f.header)['query_uuid'],committed=True))
    try:
        writer.submit(frozen());writer.drain()
        with pytest.raises(FileExistsError):writer.submit(frozen())
    finally:writer.close()


def test_delayed_archive_uses_original_frame_and_full_existing_ABI(tmp_path,monkeypatch):
    import mujoco
    from tests.world_model.test_fusion_source_camera_capture import scene,forbid_step
    from tests.world_model.test_fusion_source_camera_rig import FRAME
    from go2arx5_vla.world_model.native_body_camera_capture_v1 import BodyBaseCameraSession,BodyBaseCameraRecorder
    from go2arx5_vla.world_model.fusion_observation import TargetShadowBinding
    from go2arx5_vla.world_model.storage import digest
    from scripts.pi0292.relay_protocol import digest as document_hash,unseal
    path,model,_,packet,raw=scene(tmp_path)
    cube=model.geom('task_cube_geom').id;model.geom_type[cube]=mujoco.mjtGeom.mjGEOM_BOX
    model.geom_size[cube]=[.03]*3;mujoco.mj_saveModel(model,str(path),None)
    class Renderer:
        def __init__(self,model,**kwargs):
            self.model=model;self._scene=mujoco.MjvScene(model,maxgeom=100)
        @property
        def scene(self):return self._scene
        def update_scene(self,data,*,camera):self.camera=camera;self.value=int(round(data.qpos[7]*100))
        def disable_segmentation_rendering(self):pass
        def render(self):return np.full((256,256,3),self.value+(40 if self.camera=='body_rgb'else 80),np.uint8)
        def close(self):pass
    monkeypatch.setattr(mujoco,'Renderer',Renderer);forbid_step(monkeypatch)
    session=BodyBaseCameraSession(path)
    native,body,ncal,cal=session.render_memory(packet)
    binding=TargetShadowBinding('MUJOCO_GO2_ARX5_WORLD',digest(path),'1'*64,FRAME,
                               __import__('go2arx5_vla.world_model.fusion_observation',fromlist=['transform_hash']).transform_hash(FRAME))
    query='12345678-1234-4234-8234-123456789abc'
    task=dict(schema='WM02_FUSION_PAUSED_TARGET_TASK_V2',task_id='memorytest',language='Grasp the red cube.',
        scene_sha256=digest(path),score_contract_sha256='3'*64,language_provenance='NEW_TARGET_SHORT_SKILL_PROMPT')
    ack=dict(step=packet['step'],time_s=packet['time_s'],motiontime=0,leg_history_steps=1,gait_indices=0,
             running=False,clock_paused=True,ack_sequence=1)
    header=dict(query_uuid=query,capture=dict(index=0,native_step=packet['step'],time_s=packet['time_s']),
        task=task,task_sha256=document_hash(task),world=dict(scene_sha256=digest(path),execution_spec_sha256='1'*64),
        renderer={},pause_before=ack,pause_after=dict(ack,ack_sequence=2),step_quota_sha256='4'*64,
        binding=dict(target_world_frame=binding.target_world_frame,scene_sha256=binding.scene_sha256,
          target_runtime_sha256=binding.target_runtime_sha256,T_source_from_target=FRAME.tolist(),
          expected_transform_sha256=binding.expected_transform_sha256),task_command='GRASP_RED_LIFT20MM_HOLD1S',
        capture_wall_s=0.,dense_anchor=False)
    frame=FrozenFrame(memory._json(header),raw,native['body_rgb'].tobytes(),native['wrist_rgb'].tobytes(),
        body.tobytes(),memory._json(ncal),memory._json(cal))
    reference=tmp_path/'reference';(reference/'images').mkdir(parents=True)
    recorder=BodyBaseCameraRecorder(path,reference);recorder.save(0,packet);recorder.close()
    capture=memory.MemoryCapture.__new__(memory.MemoryCapture);capture.root=tmp_path/'archive';capture.root.mkdir()
    capture.native_model=session.native.model;capture.body_model=session.render_copy.model
    capture.actual_state=capture.body_data=None;capture.binding=binding
    # Reuse the actor's renderer/data before the slow archive ever reads frame1.
    changed=dict(packet,qpos=packet['qpos'].copy());changed['qpos'][7]=.8
    session.render_memory(changed)
    result=capture.publish(frame)
    record=json.loads(Path(result['path']).read_text());body_record=unseal(record)
    assert body_record['native_step']==packet['step'] and result['committed']
    for prefix in ('images','native-camera/images'):
        for suffix in ('body_rgb.png','wrist_rgb.png','calibration.json'):
            name=prefix+'/000000-'+suffix
            assert (capture.root/query/name).read_bytes()==(reference/name).read_bytes()
    assert (capture.root/query/'STATE.private.bin').read_bytes()==raw
    assert not (capture.root/query/'PAUSED_OBSERVATION.private.json.pending').exists()
    with pytest.raises(FileExistsError):capture.publish(frame)
    session.close()


def test_dense_windows_keep_cross_replan_nominal_commands_and_incomplete_tail():
    from go2arx5_vla.world_model.task_learning_grasp_v1 import B,E
    records=[dict(query_uuid=str(i),native_step=s,dense_anchor=(s-B)%250==0)for i,s in enumerate(range(B,E+1,50))]
    rows=[dict(native_start_step=B+25*i,native_end_step=B+25*(i+1),solver_gripper_confirmed=True,
               Teacher_decision_sha256=str(i//40),policy_EEF10=[i]*10,installed_consumed_EEF10=[-i]*10)for i in range(600)]
    result=memory.nominal_windows(records,rows)
    assert len(records)==311 and len(result)==63 and sum(r['complete']for r in result)==57
    cross=result[6]
    assert cross['nominal_policy_EEF10_target_world']==[[i]*10 for i in range(60,100)]
    assert len(set(cross['Teacher_decision_sha256']))==2
    assert not result[-1]['complete'] and len(result[-1]['missing_ticks'])==40
    rows[0]['solver_gripper_confirmed']=False
    with pytest.raises(ValueError,match='ACTUALLY_CONSUMED'):memory.nominal_windows(records,rows)


def test_old_or_student_capsules_cannot_enable_memory_or_dense_transport():
    from tests.test_body_camera_session_v3 import spec
    assert memory.configuration({})is None
    value=dict(spec(),body_camera_transport_v3=dict(version=memory.VERSION),
        expert_dense_observation_v3=dict(version=memory.DENSE_VERSION,frame_period_ticks=50,anchor_period_ticks=250))
    assert memory.configuration(value)
    for changed in (dict(task_learning_role='SLA_TARGET500_NATIVE_EVALUATION_V1'),
                    dict(expert_dense_observation_v3=dict(version=memory.DENSE_VERSION,frame_period_ticks=100,anchor_period_ticks=250))):
        with pytest.raises(ValueError):memory.configuration(dict(value,**changed))
