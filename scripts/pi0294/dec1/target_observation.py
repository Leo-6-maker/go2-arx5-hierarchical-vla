"""Bind existing target measured-state/camera records to the original query ABI.

Use only inside a genuinely admitted cold owner with its frozen ActualState
model and original FrameRecorder output. This module performs no rendering,
integration, file read/write, authority adoption or native safety prediction.
Hash consistency and CPU fixtures are not proof that capture actually occurred.
"""
from dataclasses import dataclass
import base64
import hashlib
import json
import numpy as np
from scripts.pi0292.eval_contract import FRAME,TCP
from scripts.pi0292.relay_protocol import PROTOCOL,sealed,validate_observation,uuid_string
from scripts.pi0294.task_state import ActualState,parse_state
from go2arx5_vla.adapters.execution_progress import ActualFeedback
from go2arx5_vla.trajectory.so3 import matrix_to_quat,rotation_matrix_to_rot6d_rows
from .mapped_ack import _hash,_int,_finite
from .target_time_transaction import native_source_age


def need(ok,why):
    if not ok:raise ValueError('DEC1_TARGET_OBSERVATION_'+why)


def rigid(value):
    need(not any(isinstance(v,(bool,np.bool_)) for v in np.asarray(value,dtype=object).flat),'NUMERIC_RIGID_TRANSFORM_NOT_BOOLEANS')
    a=np.asarray(value,dtype=float)
    need(a.shape==(4,4) and np.isfinite(a).all() and np.array_equal(a[3],[0,0,0,1])
         and np.allclose(a[:3,:3].T@a[:3,:3],np.eye(3),rtol=0,atol=1e-10)
         and abs(np.linalg.det(a[:3,:3])-1)<=1e-10,'EXPLICIT_RIGID_WORLD_TRANSFORM')
    return a.copy()


def transform_hash(value):return hashlib.sha256(rigid(value).astype('<f8').tobytes()).hexdigest()


def object_bytes(raw):
    need(type(raw) is bytes and len(raw)<=2*1024**2,'BOUNDED_ORIGINAL_CALIBRATION_BYTES')
    def pairs(rows):
        d={}
        for k,v in rows:
            need(k not in d,'NO_DUPLICATE_CALIBRATION_FIELDS');d[k]=v
        return d
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:need(False,'FINITE_CALIBRATION'))


@dataclass(frozen=True)
class BoundTargetObservation:
    packet: dict
    actual_feedback: ActualFeedback
    raw_actual_state10: tuple
    mapped_actual_state10: tuple
    native_state_sha256: str
    calibration_sha256: str
    target_runtime_sha256: str
    frame_transform_sha256: str
    native_step: int
    raw_native_time_s: float
    raw_MPC_source_time_s: float
    target_task_native_qualified: bool=False
    physics_advance_authorized: bool=False


class FixedTargetObservationBinding:
    """Owner-pinned rigid frame mapping; never fit to a task or desired action.

    The same explicit T_target_world_from_policy_world must be pinned in the
    progress adapter. Source query FRAME/TCP remain original; all feedback fed
    to the target mapper is measured in the target's actual world coordinates.
    """
    def __init__(self,*,run_uuid,episode_uuid,target_runtime_sha256,target_world_frame,
                 T_target_world_from_policy_world,expected_transform_sha256):
        need(isinstance(run_uuid,str) and bool(run_uuid),'OWNER_RUN_REQUIRED');uuid_string(episode_uuid)
        need(isinstance(target_world_frame,str) and bool(target_world_frame),'EXPLICIT_TARGET_WORLD_FRAME')
        transform=rigid(T_target_world_from_policy_world)
        need(transform_hash(transform)==_hash(expected_transform_sha256),'OWNER_PINNED_RIGID_FRAME_BYTES')
        transform.setflags(write=False)
        self._transform=transform;self.run=run_uuid;self.episode=episode_uuid
        self.runtime_sha256=_hash(target_runtime_sha256);self.target_frame=target_world_frame
        self.transform_sha256=expected_transform_sha256

    def bind(self,*,actual_state,native_state_bytes,expected_state_sha256,native_step,
             MPC_source_time_s,predecessor_ack_id,language,snapshot_uuid,capture,
             calibration_bytes,body_png_bytes,wrist_png_bytes):
        need(isinstance(actual_state,ActualState),'EXISTING_FROZEN_NATIVE_ACTUAL_STATE_READER')
        step=_int(native_step)
        need(2500<=step<18500 and (step-2500)%200==0,'ORIGINAL_FRESH_QUERY_AFTER_WARMUP_PREFIX200')
        sequence=(step-2500)//200
        need((predecessor_ack_id is None) if sequence==0 else
             (isinstance(predecessor_ack_id,str) and bool(predecessor_ack_id)),'PREVIOUS_REAL_ADAPTER_ACK')
        need(type(native_state_bytes) is bytes and len(native_state_bytes)<=2*1024**2
             and hashlib.sha256(native_state_bytes).hexdigest()==_hash(expected_state_sha256),'EXACT_RETAINED_NATIVE_STATE_BYTES')
        packet=parse_state(native_state_bytes);time_s=_finite(packet['time_s']);source=_finite(MPC_source_time_s)
        need(packet['step']==step and abs(time_s-step/500)<=1e-8
             and 0<=native_source_age(time_s,source)<=.025,'UNMODIFIED_NATIVE_CLOCK_AND_ORIGINAL_25MS_AGE')
        for contact in packet['contacts']:
            need(all(np.isfinite(np.asarray(contact[k])).all() for k in
                 ('distance_m','wrench','normal','frame','point')),'FINITE_ACTUAL_CONTACT_PACKET')
        # Original ActualState uses mj_forward only on its read-only copy of
        # measured qpos/qvel/ctrl. It does not integrate or install a reference.
        measured=actual_state.update(packet)
        need(measured['native_step']==step and measured['time_s']==time_s
             and measured['safety_valid'] is True and measured['gripper_state_valid'] is True
             and measured['native_abort'] is False,'CURRENT_MEASURED_NATIVE_FEEDBACK_NOT_DESIRED_OR_RETIMED')
        tcp=rigid(measured['tcp']);grip=_finite(measured['gripper_fraction'])
        need(0<=grip<=1,'MEASURED_MECHANICAL_GRIPPER_NO_CLIPPING')
        policy_tcp=np.linalg.solve(self._transform,tcp)
        policy_row=np.r_[policy_tcp[:3,3],rotation_matrix_to_rot6d_rows(policy_tcp[:3,:3]),grip]
        target_row=np.r_[tcp[:3,3],rotation_matrix_to_rot6d_rows(tcp[:3,:3]),grip]
        need(type(capture) is dict and set(capture)=={'index','native_step','time_s'}
             and type(capture['index']) is int and capture['index']==(step-2500)//25
             and type(capture['native_step']) is int and capture['native_step']==step
             and type(capture['time_s']) in (int,float) and capture['time_s']==time_s,
             'EXISTING_FRAME_RECORDER_SAME_ACTUAL_NATIVE_PACKET')
        calibration=object_bytes(calibration_bytes)
        need(type(calibration) is dict and set(calibration)=={'body_rgb','wrist_rgb'},'ORIGINAL_TARGET_TWO_CAMERA_CALIBRATIONS')
        for c in calibration.values():
            need(type(c) is dict and set(c)=={'timestamp_s','T_world_optical','K'}
                 and type(c['timestamp_s']) in (int,float) and c['timestamp_s']==time_s,
                 'ORIGINAL_CAMERA_CLOCK_NOT_QUERY_RELATIVE_OR_REFRESHED')
            rigid(c['T_world_optical']);K=np.asarray(c['K'],dtype=float)
            need(K.shape==(3,3) and np.isfinite(K).all() and np.array_equal(K[2],[0,0,1])
                 and K[0,0]>0 and K[1,1]>0,'ACTUAL_FINITE_CAMERA_INTRINSICS')
        images=[]
        for name,raw in (('external_rgb',body_png_bytes),('wrist_rgb',wrist_png_bytes)):
            need(type(raw) is bytes and len(raw)<=4*1024**2,'ORIGINAL_BOUNDED_CAPTURED_PNG')
            images.append(dict(camera=name,png_base64=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest()))
        payload=dict(snapshot_uuid=snapshot_uuid,language=language,state_eef10=policy_row.tolist(),
                     images=images,frame=FRAME,tcp=TCP)
        validate_observation(payload)
        observation=sealed(dict(protocol=PROTOCOL,kind='observation',run_uuid=self.run,episode_uuid=self.episode,
            sequence=sequence,tick=step-2500,payload=payload))
        feedback=ActualFeedback(time_s,tcp[:3,3],matrix_to_quat(tcp[:3,:3]),grip,self.target_frame,TCP,predecessor_ack_id)
        return BoundTargetObservation(observation,feedback,tuple(policy_row),tuple(target_row),expected_state_sha256,
            hashlib.sha256(calibration_bytes).hexdigest(),self.runtime_sha256,self.transform_sha256,step,time_s,source)
