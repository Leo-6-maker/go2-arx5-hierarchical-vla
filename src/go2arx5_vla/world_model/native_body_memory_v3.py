"""Paused CPU-expert pixel snapshots and asynchronous original-ABI archives.

Only the main thread touches renderers. The writer owns immutable copies and
its own FK data, reusing the existing state/camera binding and validation.
An archive commit is not a qualified demonstration or an inference grant.
"""
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
import uuid

import numpy as np

from .native_frame_writer_v3 import FrozenFrame, FrameWriter
from .native_collection_performance_v1 import measure
from .storage import digest, write_json

VERSION = 'IMMUTABLE_BODY_MEMORY_ASYNC_EXPERT_V3'
DENSE_VERSION = 'NATIVE_RGB_STATE10HZ_ANCHOR2HZ_V3'


def configuration(spec):
    config = spec.get('body_camera_transport_v3')
    if config is None:
        if spec.get('expert_dense_observation_v3') is not None:
            raise ValueError('DENSE_RECORDING_REQUIRES_IMMUTABLE_BODY_TRANSPORT')
        return None
    from .native_body_camera_capture_v1 import session_configuration
    if config != dict(version=VERSION) or session_configuration(spec) is None:
        raise ValueError('MEMORY_BODY_TRANSPORT_FRESH_CPU_EXPERT_ONLY')
    dense = spec.get('expert_dense_observation_v3')
    if dense is not None and dense != dict(version=DENSE_VERSION, frame_period_ticks=50, anchor_period_ticks=250):
        raise ValueError('DENSE_RECORDING_FIXED_NATIVE10HZ_AND2HZ_ONLY')
    return config


def _json(value):
    return json.dumps(value, allow_nan=False).encode('utf-8')


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _atomic_record(path, value):
    temporary = path.with_name(path.name + '.pending')
    if path.exists():raise FileExistsError('FRAME_MANIFEST_NEVER_REPLAY')
    write_json(temporary, value)
    os.rename(temporary, path)


def frame_files(frame):
    """Pure byte encoding, also audited against retained saved captures."""
    from PIL import Image
    encoded = {}
    for key in ('native_body','native_wrist','body'):
        stream = io.BytesIO();Image.fromarray(frame.pixels(key)).save(stream,format='PNG')
        encoded[key] = stream.getvalue()
    return {'STATE.private.bin':frame.native_state,
        'images/000000-body_rgb.png':encoded['body'],
        'images/000000-wrist_rgb.png':encoded['native_wrist'],
        'images/000000-calibration.json':frame.calibration,
        'native-camera/images/000000-body_rgb.png':encoded['native_body'],
        'native-camera/images/000000-wrist_rgb.png':encoded['native_wrist'],
        'native-camera/images/000000-calibration.json':frame.native_calibration}


class MemoryCapture:
    def __init__(self, adapter, binding):
        from .fusion_paused_observation import _runtime
        from .native_body_camera_capture_v1 import BodyBaseCameraSession
        self.adapter = adapter
        self.root = adapter.output/'PAUSED_OBSERVATIONS'
        self.root.mkdir(exist_ok=True)
        self.binding = binding
        self.world = _runtime(adapter, binding)
        self.identity = self._identity()
        self.session = BodyBaseCameraSession(adapter.spec['model_path'])
        adapter._body_camera_session = self.session
        self.native_model = self.body_model = self.actual_state = self.body_data = None
        self.writer = FrameWriter(self.publish)
        self.tickets = {}
        self.last = None
        self.capture_requests = 0
        self.finalized = False

    def _identity(self):
        spec = self.adapter.spec
        paths = [Path(spec['model_path']), Path(spec['registered_root'])/'EXECUTION_SPEC.json',
                 self.adapter.output/'quota.binding']
        result=[]
        for path in paths:
            stat=path.stat()
            result.append((str(path),stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns))
        return result

    def check(self):
        self.writer.reap()

    def capture(self, *, binding, task, task_sha256):
        from .fusion_paused_observation import (_pause, _after_pause, _cpu_renderer_environment, _task)
        from scripts.wm02.capture_fusion_observation import software_renderer_identity
        from .task_learning_grasp_v1 import B
        self.capture_requests += 1
        _cpu_renderer_environment()
        if (self.finalized or self._identity() != self.identity
                or binding.scene_sha256 != self.binding.scene_sha256
                or binding.target_runtime_sha256 != self.binding.target_runtime_sha256
                or binding.expected_transform_sha256 != self.binding.expected_transform_sha256
                or not np.array_equal(binding.T_source_from_target, self.binding.T_source_from_target)):
            raise ValueError('IMMUTABLE_CAPTURE_RUNTIME_OR_BINDING_CHANGED')
        task = _task(task, task_sha256, binding.scene_sha256)
        first = _pause(self.adapter)
        step = first['packet']['step'];state_sha = _sha(first['raw']);quota_sha = _sha(first['quota'])
        if self.last is not None and self.last['native_step'] == step:
            if (self.last['state_sha256'] != state_sha or self.last['task_sha256'] != task_sha256
                    or self.last['quota_sha256'] != quota_sha):
                raise ValueError('SAME_CURRENT_PAUSE_HAS_DIFFERENT_FRAME_INPUTS')
            return dict(self.last)
        if len(self.tickets) >= 1024 or (self.last is not None and step <= self.last['native_step']):
            raise ValueError('FINITE_MONOTONE_NEW_FRAME_SEQUENCE_REQUIRED')
        # _pause has proved the exact current quota/ACK before any storage wait.
        with measure(self.adapter, 'camera.writer_room_at_verified_pause'):
            self.writer.wait_room(guard=self.adapter.guard)
        began = time.monotonic()
        with measure(self.adapter, 'camera.immutable_memory_render'):
            native, body, native_cal, cal = self.session.render_memory(first['packet'],
                performance=getattr(self.adapter, 'performance', None))
        self.native_model = self.session.native.model
        self.body_model = self.session.render_copy.model
        renderer = software_renderer_identity()
        if not ('mesa' in renderer.get('GL_VENDOR', '').lower() and any(
                x in renderer.get('GL_RENDERER', '').lower() for x in ('llvmpipe', 'softpipe', 'software rasterizer'))):
            raise ValueError('ACTUAL_MEMORY_CAPTURE_CPU_RENDERER_REQUIRED')
        last = _after_pause(self.adapter, first)
        query = str(uuid.uuid4())
        header = dict(query_uuid=query, capture=dict(index=0, native_step=step, time_s=first['packet']['time_s']),
            task=task, task_sha256=task_sha256, world=self.world, renderer=renderer,
            pause_before=first['pause'], pause_after=last['pause'], step_quota_sha256=quota_sha,
            binding=dict(target_world_frame=binding.target_world_frame, scene_sha256=binding.scene_sha256,
                target_runtime_sha256=binding.target_runtime_sha256,
                T_source_from_target=binding.T_source_from_target.tolist(),
                expected_transform_sha256=binding.expected_transform_sha256),
            task_command=self.adapter.spec['task_command'], capture_wall_s=time.monotonic()-began,
            dense_anchor=bool(self.adapter.spec.get('expert_dense_observation_v3') and (step-B)%250 == 0))
        frame = FrozenFrame(_json(header), first['raw'], native['body_rgb'].tobytes(),
            native['wrist_rgb'].tobytes(), body.tobytes(), _json(native_cal), _json(cal))
        ticket = dict(schema=VERSION, query_uuid=query, native_step=step, native_time_s=first['packet']['time_s'],
            path=str(self.root/query/'PAUSED_OBSERVATION.private.json'), sha256=None, state_sha256=state_sha,
            quota_sha256=quota_sha, task_sha256=task_sha256, task_command=self.adapter.spec['task_command'],
            memory_snapshot_sha256=frame.sha256, archive_state='PENDING_NOT_READABLE',
            GR00T_queries=0, actual_cube_truth_is_student_input=False)
        with measure(self.adapter, 'camera.immutable_frame_handoff'):
            self.writer.submit(frame)
        self.tickets[query] = ticket;self.last = ticket
        return dict(ticket)

    def publish(self, frame):
        """Writer thread: PNG, FK binding, checksums, flush and atomic discovery."""
        import mujoco
        from scripts.pi0294.task_state import ActualState, parse_state
        from scripts.pi0292.relay_protocol import sealed
        from .fusion_observation import TargetShadowBinding, shadow_inference_payload
        from .fusion_source_camera_capture import _load_packet
        from .native_body_camera_capture_v1 import CapturedBodyBaseCameraAdapter
        from .fusion_paused_observation import SCHEMA, INPUT_MODE, paused_inference_payload
        h = json.loads(frame.header);query = h['query_uuid'];folder = self.root/query
        folder.mkdir(exist_ok=False);(folder/'images').mkdir();(folder/'native-camera').mkdir()
        (folder/'native-camera/images').mkdir()
        files = frame_files(frame)
        encoded=dict(body=files['images/000000-body_rgb.png'],
                     native_body=files['native-camera/images/000000-body_rgb.png'],
                     native_wrist=files['images/000000-wrist_rgb.png'])
        if self.actual_state is None:
            self.actual_state = ActualState(self.native_model)
            self.body_data = mujoco.MjData(self.body_model)
        packet = parse_state(frame.native_state)
        _load_packet(self.body_model, self.body_data, packet)
        camera = CapturedBodyBaseCameraAdapter(SimpleNamespace(model=self.body_model,data=self.body_data),
            frame.native_calibration, dict(body_rgb=encoded['native_body'],wrist_rgb=encoded['native_wrist']),
            self.binding.scene_sha256, _sha(encoded['body']))
        shadow = TargetShadowBinding(**h['binding']).bind(actual_state=self.actual_state,
            native_state_bytes=frame.native_state,expected_state_sha256=_sha(frame.native_state),
            capture=h['capture'],calibration_bytes=frame.calibration,
            expected_calibration_sha256=_sha(frame.calibration),
            image_bytes=dict(body_rgb=encoded['body'],wrist_rgb=encoded['native_wrist']),
            expected_image_sha256=dict(body_rgb=_sha(encoded['body']),wrist_rgb=_sha(encoded['native_wrist'])),
            language=h['task']['language'],snapshot_uuid=query,camera_adapter=camera)
        shadow_inference_payload(shadow)
        for name, raw in files.items():
            with (folder/name).open('xb') as stream:
                stream.write(raw);stream.flush();os.fsync(stream.fileno())
        write_json(folder/'BOUND_SHADOW_ABI.private.json',shadow)
        pins = {name:digest(folder/name) for name in (*files,'BOUND_SHADOW_ABI.private.json')}
        if any(pins[name] != _sha(raw) for name,raw in files.items()):
            raise ValueError('ASYNC_FRAME_DISK_BYTES_DIFFER_FROM_IMMUTABLE_SNAPSHOT')
        record = sealed(dict(schema=SCHEMA,input_mode=INPUT_MODE,query_uuid=query,snapshot_uuid=query,
            capture=h['capture'],native_step=h['capture']['native_step'],native_time_s=h['capture']['time_s'],
            task=h['task'],task_sha256=h['task_sha256'],world=h['world'],renderer=h['renderer'],
            pause_before=h['pause_before'],pause_after=h['pause_after'],step_quota_sha256=h['step_quota_sha256'],
            files=pins,original_shadow_ABI=shadow,capture_wall_s=h['capture_wall_s'],
            transport=dict(version=VERSION,immutable_snapshot_sha256=frame.sha256,
                original_state_at_capture=True, PNG_encoding_on_control_thread=False),
            provenance=dict(actual_current_owned_world=True,simulation_paused_during_capture=True,
                native_clock_rebased=False,task_language_bound_to_declared_scene=True,camera_equivalence=False,
                target_task_success=False,native_execution_authorized=False,source_outcomes_supplied=False,
                VLA_inferences=0,physics_integrations=0,first_draw_replay_is_sequential_feedback=False,
                online_closed_loop='REQUIRES_EACH_NEW_BRANCH_OBSERVATION_QUERY_AND_ACTUAL_PREFIX_ACK')))
        paused_inference_payload(record)
        path = folder/'PAUSED_OBSERVATION.private.json'
        _atomic_record(path,record)
        return dict(path=str(path),sha256=digest(path),query_uuid=query,committed=True,
            native_step=h['capture']['native_step'],native_time_s=h['capture']['time_s'],
            state_sha256=pins['STATE.private.bin'],task_command=h['task_command'],
            memory_snapshot_sha256=frame.sha256,dense_anchor=h['dense_anchor'],
            pause_revalidated_at_capture=True,GR00T_queries=0,actual_cube_truth_is_student_input=False,
            files_verified=len(pins))

    def finish(self, captures, *, observed_task_success, confirmed):
        self.writer.drain(guard=self.adapter.guard)
        if len(self.writer.committed) != len(self.tickets):
            raise ValueError('ALL_REQUESTED_MEMORY_FRAMES_MUST_BE_COMMITTED')
        resolved = [self.writer.committed[t['query_uuid']] for t in captures]
        records = sorted(self.writer.committed.values(),key=lambda r:r['native_step'])
        dense = self.adapter.spec.get('expert_dense_observation_v3') is not None
        if dense:
            from .task_learning_grasp_v1 import B,E
            if [r['native_step']for r in records] != list(range(B,E+1,50)):
                raise ValueError('NO_MISSING_DENSE10HZ_FRAME_OR_CHANGED_TASK_DURATION')
        labels = nominal_windows(records,confirmed) if dense else []
        manifest = dict(schema=VERSION,frames_committed=True,recording_complete=True,records=records,
            frames=len(records),anchors=sum(r['dense_anchor']for r in records),
            nominal_windows=labels,complete_consumption_joined_nominal_windows=sum(r['complete']for r in labels),
            frame_period_ticks=50 if dense else None,anchor_period_ticks=250 if dense else None,
            original_dt_s=.002,observed_task_success=observed_task_success,
            outer_native_shutdown_qualified=False,BC_eligible=False,supervised_windows_admitted=0,
            source_sha=self.adapter.spec['source_sha'],writer=self.writer.report(),
            capture_requests=self.capture_requests,renderer_threads='ACTOR_MAIN_THREAD_ONLY',
            frames_per_window_changed=False,model_calls=0,physics_integrations_by_capture=0)
        _atomic_record(self.adapter.output/'BODY_MEMORY_FRAME_ARCHIVE.private.json',manifest)
        self.finalized = True
        return resolved,dict(path=str(self.adapter.output/'BODY_MEMORY_FRAME_ARCHIVE.private.json'),
                             sha256=digest(self.adapter.output/'BODY_MEMORY_FRAME_ARCHIVE.private.json'))

    def close(self):
        try:self.writer.close()
        finally:
            from scripts.pi0292.durable import write
            write(self.adapter.output/'BODY_MEMORY_WRITER_ACTUAL.private.json',
                  dict(self.writer.report(),episode_archive_published=self.finalized))


def capture_input(adapter, *, binding, task, task_sha256):
    if configuration(adapter.spec) is None:raise ValueError('EXPLICIT_BODY_MEMORY_TRANSPORT_REQUIRED')
    if not hasattr(adapter,'_body_memory_capture'):
        adapter._body_memory_capture = MemoryCapture(adapter,binding)
    return adapter._body_memory_capture.capture(binding=binding,task=task,task_sha256=task_sha256)


def observe_dense(adapter, sample, capture):
    if not adapter.spec.get('expert_dense_observation_v3'):return
    from .task_learning_grasp_v1 import B,E
    step = sample['native_step']
    if B <= step <= E and (step-B)%50 == 0:
        capture()


def nominal_windows(records, confirmed):
    """Join all2Hz anchors to actually consumed50ms nominal commands.

    Preserve both original policy nominal EEF10 and its installed reference;
    no achieved TCP or unexecuted suffix becomes a label. These target-world
    rows still require the separately versioned dataset/frame/codec admission.
    """
    by_step = {}
    for row in confirmed:
        step = row['native_start_step']
        if (step in by_step or row['native_end_step'] != step+25
                or row.get('solver_gripper_confirmed') is not True
                or not row.get('Teacher_decision_sha256')):
            raise ValueError('UNIQUE_ACTUALLY_CONSUMED_EXPERT_NOMINAL_LABELS_REQUIRED')
        by_step[step] = row
    result = []
    for frame in records:
        if not frame['dense_anchor']:continue
        ticks = [frame['native_step']+25*i for i in range(40)]
        missing = [step for step in ticks if step not in by_step]
        if missing:
            result.append(dict(query_uuid=frame['query_uuid'],native_step=frame['native_step'],
                complete=False,reason='NO_FULL_EXECUTED_FUTURE_H40_TAIL_NOT_PADDED',missing_ticks=missing))
            continue
        rows = [by_step[step]for step in ticks]
        result.append(dict(query_uuid=frame['query_uuid'],native_step=frame['native_step'],complete=True,
            nominal_policy_EEF10_target_world=[r['policy_EEF10']for r in rows],
            installed_consumed_EEF10_reference=[r['installed_consumed_EEF10']for r in rows],
            Teacher_decision_sha256=[r['Teacher_decision_sha256']for r in rows],
            native_start_steps=ticks,period_s=.05,unexecuted_suffix_included=False,
            achieved_TCP_used_as_label=False,frame='MUJOCO_GO2_ARX5_WORLD_AND_DECLARED_REFERENCE_TRANSPORT'))
    return result
