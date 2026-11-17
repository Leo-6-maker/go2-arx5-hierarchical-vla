"""Bounded live paused RGB shards, separate planned and consumed H40 evidence."""
import json
from pathlib import Path
from queue import Queue, Full
from threading import Thread
import numpy as np
from .native_body_camera_capture_v1 import BodyBaseCameraSession
from .storage import digest
from scripts.pi0292.durable import write


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k:plain(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [plain(v) for v in value]
    return value


class LiveCamera(BodyBaseCameraSession):
    def acquire(self, model_path, folder):
        # The immutable scene is verified when opening this session and at the
        # sealed transfer boundary. Do not hash its entire bytes every query.
        if self.busy or self.closed or Path(model_path) != self.model_path:
            raise PermissionError('EXCLUSIVE_IMMUTABLE_LIVE_CAMERA_SESSION')
        if self.native is None:
            from scripts.pi0294.task_native_worker import FrameRecorder
            from .native_camera_transfer_v1 import FixedBodyRenderer
            self.native = FrameRecorder(self.model_path, folder)
            self.native.reset_each_packet = True
            self.render_copy = FixedBodyRenderer(self.model_path)
        else:
            self.native.folder = folder
        self.busy = True
        return self.native, self.render_copy


class LiveWriter:
    def __init__(self, adapter, frame, *, camera_factory=LiveCamera):
        self.adapter = adapter; self.frame = frame
        self.root = adapter.output/'LIVE_DATA'; self.root.mkdir()
        self.camera = camera_factory(adapter.spec['model_path'])
        self.queue = Queue(4); self.rows = []; self.shards = []; self.error = None
        self.thread = Thread(target=self._worker, daemon=True); self.thread.start()

    def _worker(self):
        buffered = []
        try:
            while True:
                value = self.queue.get()
                if value is None:
                    if buffered: self._flush(buffered)
                    return
                buffered.append(value)
                if len(buffered) == 16:
                    self._flush(buffered); buffered = []
        except BaseException as error:
            self.error = repr(error)

    def enqueue(self, value):
        while True:
            if self.error: raise RuntimeError(self.error)
            try:
                self.queue.put(value, timeout=.1); return
            except Full: pass

    def _flush(self, values):
        path = self.root/f'frames-{len(self.shards):03d}.npz'
        with path.open('xb') as handle:
            np.savez(handle, rgb=np.stack([v[0] for v in values]),
                state_eef10=np.stack([v[1] for v in values]),
                native_step=np.array([v[2] for v in values], dtype=np.int64))
        self.shards.append(dict(path=str(path), sha256=digest(path), count=len(values),
            first_query=sum(s['count'] for s in self.shards)))

    def capture(self, sample, program, evidence, boundary_evidence, teacher):
        from .fusion_paused_observation import _pause, _after_pause
        from scripts.pi0291_arx5_data.native_dataset import pose10
        before = _pause(self.adapter)
        if before['packet']['step'] != sample['native_step']:
            raise ValueError('LIVE_RGB_AND_EXPERT_QUERY_SAME_NATIVE_CLOCK')
        images, body, native_cal, calibration = self.camera.render_memory(before['packet'])
        _after_pause(self.adapter, before)
        state = pose10(np.asarray(self.adapter.spec['policy_from_world']) @ sample['tcp'], sample['gripper_fraction'])
        rgb = np.stack([body, images['wrist_rgb']]).copy()
        if rgb.shape != (2,256,256,3) or rgb.dtype != np.uint8:
            raise ValueError('CURRENT_NATIVE_BODY_WRIST_RGB_UINT8_REQUIRED')
        self.enqueue((rgb, state.copy(), sample['native_step']))
        if self.error: raise RuntimeError(self.error)
        row = dict(query=len(self.rows), native_step=sample['native_step'], native_time_s=sample['time_s'],
            actual_state=sample, state_eef10=state,
            nominal_world_H40=program.decoded_H40.copy(), installed_reference_H40=program.reference_path.copy(),
            raw_intent=dict(kind='FEEDBACK_GEOMETRIC_GOAL_NOT_UNFILTERED_H40', goal=evidence['privileged_goal'],
                orientation=teacher.rotation.copy(), closing_start_step=teacher.close_start),
            phase=teacher.stage, planning_epoch=len(self.rows), recent_reference=boundary_evidence,
            teacher_memory_audit_only=dict(velocity=teacher.velocity.copy(), inner_velocity=teacher.inner_velocity.copy(),
                angular_velocity=teacher.angular_velocity.copy(), inner_angular_velocity=teacher.inner_angular_velocity.copy(),
                close_start=teacher.close_start, last_committed_step=teacher.last_committed_step),
            supervision_mask=[True]*40, consumed_mask=[False]*40, intervals=[],
            full_EEF_H40_planner_check=True, rgb_provenance='CURRENT_OWNED_PAUSED_NATIVE_PACKET',
            camera_calibration=calibration, native_camera_calibration=native_cal,
            episode_fixed_T_policy_from_mount=self.frame,
            object_phase_memory_in_student_inputs=False)
        self.rows.append(row)
        return row

    def close(self):
        self.camera.close()
        if self.thread.is_alive() and not self.error: self.enqueue(None)
        self.thread.join()
        # Even on a native STOP, retain the final measured clock and distinguish
        # partial intervals from confirmed 25-tick consumption.
        from .native_live_control_v3 import live_path
        from scripts.pi0294.task_state import parse_state
        packet = getattr(self.adapter,'native_packet_parser',parse_state)(live_path(self.adapter,'state.bin').read_bytes())
        for row in self.rows:
            row['actual_elapsed_prefix_ticks'] = max(0, min(250, packet['step']-row['native_step']))
            row['partial_interval_ticks'] = row['actual_elapsed_prefix_ticks'] % 25
        write(self.root/'QUERIES.private.json', plain(self.rows))
        manifest = dict(schema='MFREE_LIVE_FINAL_PLAN_CAPTURE_V1', queries=len(self.rows),
            frame_shards=self.shards, writer_error=self.error, labels=dict(path=str(self.root/'QUERIES.private.json'),
                sha256=digest(self.root/'QUERIES.private.json')), final_native_step=packet['step'],
            frozen_scene_sha256=self.camera.model_sha, language=self.adapter.spec['language'],
            data_contract=self.adapter.spec['collection_contract'],
            planned_suffix_is_executed=False, success_admission_requires_native_result=True)
        write(self.root/'MANIFEST.private.json', manifest)
        if self.error: raise RuntimeError(self.error)
        return dict(path=str(self.root/'MANIFEST.private.json'), sha256=digest(self.root/'MANIFEST.private.json'))
