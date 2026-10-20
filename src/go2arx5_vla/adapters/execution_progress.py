"""DEC1 task-blind feedback anchoring, not an executor or safety replacement.

The caller supplies the unchanged executor's raw/full/prefix validators and
must stop that executor on any exception. Proposals do not mean execution:
an explicit execution ACK and a subsequent fresh observation are required.
"""
from dataclasses import dataclass, replace
import copy
import hashlib
import json
from typing import Callable, Literal

import numpy as np

from ..contracts.trajectory import GripperCommandV1, TrajectoryChunkV1, WBCPreviewV1
from ..trajectory.preview import _arc_map, se3_arc_length_preview
from ..trajectory.so3 import matrix_to_quat, quat_slerp, rot6d_rows_to_matrix
from .groot_to_wbc import ReferenceAdapterConfig
from .groot_to_wbc_v2 import GR00TToWBCAdapterV2


@dataclass(frozen=True)
class ProgressConfig:
    reference_period_s: float = .05
    local_forward_points: float = 2.
    initial_forward_points: float = 8.
    feedback_max_age_s: float = .075
    maximum_feedback_gap_s: float = .075
    new_chunk_max_age_s: float = .4
    stationary_arc_epsilon_m: float = 1e-8
    stationary_position_tolerance_m: float = .001
    stationary_rotation_tolerance_rad: float = .01
    gripper_metric_m_per_unit: float = .01
    phase_snap_epsilon: float = 1e-7
    projection_grid_points: int = 17
    projection_refinements: int = 8

    def __post_init__(self):
        for name,value in vars(self).items():
            if type(value) not in (int,float) or not np.isfinite(value) or value<=0:
                raise ValueError('DEC1_FINITE_POSITIVE_CONFIG:'+name)
        if (type(self.projection_grid_points) is not int or self.projection_grid_points<3
                or type(self.projection_refinements) is not int
                or self.projection_grid_points>65 or self.projection_refinements>12
                or not 1<=self.local_forward_points<=self.initial_forward_points<=8
                or self.reference_period_s != .05
                or self.new_chunk_max_age_s>8*self.reference_period_s):
            raise ValueError('DEC1_BOUNDED_SOURCE_ONLY_PROJECTION_CONFIG')


def _readonly(value):
    out=np.asarray(value,dtype=np.float64).copy();out.setflags(write=False)
    return out


def _angle(a,b):
    """Stable SO(3) geodesic, including quaternion sign aliases and tiny angles."""
    b=np.asarray(b);a=np.asarray(a)
    aligned=np.where((np.sum(a*b,axis=-1)<0)[...,None],-b,b)
    return 4*np.arctan2(np.linalg.norm(a-aligned,axis=-1),np.linalg.norm(a+aligned,axis=-1))


@dataclass(frozen=True)
class ActualFeedback:
    timestamp: float
    eef_position: np.ndarray
    eef_orientation: np.ndarray
    gripper: float
    source_frame: str
    tcp_semantic_id: str
    predecessor_ack_id: str | None = None

    def __post_init__(self):
        p=_readonly(self.eef_position);q=np.asarray(self.eef_orientation,dtype=float)
        if (p.shape!=(3,) or q.shape!=(4,) or not np.isfinite(p).all()
                or not np.isfinite(q).all() or not np.isfinite(self.timestamp)
                or not np.isfinite(self.gripper) or np.linalg.norm(q)<1e-10):
            raise ValueError('DEC1_FINITE_ACTUAL_TCP_GRIPPER_FEEDBACK')
        object.__setattr__(self,'eef_position',p)
        object.__setattr__(self,'eef_orientation',_readonly(q/np.linalg.norm(q)))


def canonical_chunk_from_row_eef10(*,raw_eef10,chunk_id,observation_timestamp,
                                 point_timestamps,source_frame,tcp_semantic_id,validate_raw_eef10):
    """Explicit PI0294 wire boundary; never feed row rot6d to from_rot6d.

    The unchanged native full-H40 checker runs on the original decoded array
    before quaternion conversion. The caller still binds its native safety
    chain and clocks; this conversion is not execution authorization.
    """
    value=_readonly(raw_eef10);times=np.asarray(point_timestamps,dtype=float)
    if (value.shape!=(40,10) or not np.isfinite(value).all() or times.shape!=(40,)
            or not np.isfinite(times).all() or not callable(validate_raw_eef10)):
        raise ValueError('DEC1_EXPLICIT_FINITE_NATIVE_ROW_EEF10_FULL_H40')
    if validate_raw_eef10(value) is not None:
        raise PermissionError('DEC1_NATIVE_VALIDATOR_MUST_RAISE_OR_RETURN_NONE')
    return TrajectoryChunkV1(chunk_id=chunk_id,source='GR00T_NATIVE_ROW_EEF10',
        observation_timestamp=observation_timestamp,point_timestamps=times,
        eef_position=value[:,:3],eef_orientation=matrix_to_quat(rot6d_rows_to_matrix(value[:,3:9])),
        source_frame=source_frame,tcp_semantic_id=tcp_semantic_id,
        validity_start=float(times[0]),validity_end=float(times[-1]),gripper=value[:,9])


def _copy_chunk(chunk):
    result=copy.deepcopy(chunk)
    for name in ('point_timestamps','eef_position','eef_orientation','gripper','eef_rot6d'):
        value=getattr(chunk,name)
        object.__setattr__(result,name,None if value is None else _readonly(value))
    return result


def _digest(chunk):
    h=hashlib.sha256(json.dumps([chunk.chunk_id,chunk.source,chunk.source_frame,chunk.tcp_semantic_id,
        chunk.observation_timestamp,chunk.validity_start,chunk.validity_end],allow_nan=False).encode())
    for array in (chunk.point_timestamps,chunk.eef_position,chunk.eef_orientation,chunk.gripper):
        h.update(np.asarray(array,dtype='<f8').tobytes())
    return h.hexdigest()


def _proposal_digest(proposal):
    h=hashlib.sha256(_digest(proposal.mapped_full).encode())
    h.update(json.dumps([proposal.command_id,proposal.controller_timestamp,proposal.feedback_timestamp,
        proposal.phase,proposal.arc_m,proposal.gripper.timestamp,proposal.gripper.normalized_command,
        proposal.gripper.stale,proposal.gripper.raw_source_convention],allow_nan=False).encode())
    # Include the directly dispatched preview as well as its full forecast.
    for p in (proposal.preview, *proposal.mapped_previews):
        h.update(json.dumps([p.anchor_timestamp,p.source_frame,p.tcp_semantic_id],allow_nan=False).encode())
        for value in (p.eef_position,p.eef_rot6d,p.offsets_m,p.endpoint_clamped):
            h.update(np.asarray(value,dtype='<f8').tobytes())
    return h.hexdigest()


def _pose(chunk,phase):
    phase=float(np.clip(phase,0,chunk.point_count-1));i=min(int(phase),chunk.point_count-2);f=phase-i
    return (1-f)*chunk.eef_position[i]+f*chunk.eef_position[i+1],quat_slerp(
        chunk.eef_orientation[i],chunk.eef_orientation[i+1],f)


def _project(chunk,feedback,lower,upper,config,stationary,rotation_weight):
    """Finite deterministic local SE(3) search; no global nearest-branch jump.

    Nested NumPy grids make the approximation and evaluation bound explicit.
    Gripper distance only disambiguates zero-SE(3)-length segments. Constant
    stationary segments are advanced separately by their original local clock.
    """
    best=(float('inf'),lower)
    for i in range(int(np.floor(lower)),min(int(np.floor(upper))+1,39)):
        lo=max(0.,lower-i);hi=min(1.,upper-i)
        if hi<lo:continue
        def score(f):
            p=(1-f[:,None])*chunk.eef_position[i]+f[:,None]*chunk.eef_position[i+1]
            q=quat_slerp(np.broadcast_to(chunk.eef_orientation[i],(len(f),4)),
                np.broadcast_to(chunk.eef_orientation[i+1],(len(f),4)),f)
            d=np.sum((p-feedback.eef_position)**2,axis=1)+(rotation_weight*_angle(feedback.eef_orientation,q))**2
            if stationary[i]:
                g=(1-f)*chunk.gripper[i]+f*chunk.gripper[i+1]
                d+=(config.gripper_metric_m_per_unit*(g-feedback.gripper))**2
            return d
        for _ in range(config.projection_refinements):
            grid=np.linspace(lo,hi,config.projection_grid_points);values=score(grid)
            k=int(np.argmin(values));candidate=(float(values[k]),float(i+grid[k]))
            if candidate<best:best=candidate
            lo,hi=float(grid[max(0,k-1)]),float(grid[min(len(grid)-1,k+1)])
    phase=best[1];nearest=round(phase)
    return float(nearest if abs(phase-nearest)<=config.phase_snap_epsilon else phase)


@dataclass(frozen=True)
class ReferenceProposal:
    command_id: str
    controller_timestamp: float
    feedback_timestamp: float
    phase: float
    arc_m: float
    preview: WBCPreviewV1
    gripper: GripperCommandV1
    mapped_full: TrajectoryChunkV1
    mapped_previews: tuple[WBCPreviewV1,...]

    @property
    def prefix(self):
        return self.mapped_previews[:8]


class ExecutionProgressAdapter:
    """Common V2 frame/preview path; only the phase estimator differs by mode.

    Validators must raise on rejection and return None only on acceptance. CPU fixtures can
    test these calls but cannot qualify a source/target native safety chain.
    No object state, language, task ID, contact or Teacher phase is consumed.
    """
    def __init__(self,reference: ReferenceAdapterConfig,config: ProgressConfig,*,
                 mode: Literal['I_TIME','I_PROGRESS'],episode_id: str,
                 validate_raw: Callable,validate_mapped_full: Callable,validate_prefix: Callable):
        if mode not in ('I_TIME','I_PROGRESS') or not episode_id:
            raise ValueError('DEC1_EXPLICIT_INTERFACE_AND_EPISODE')
        if not all(callable(v) for v in (validate_raw,validate_mapped_full,validate_prefix)):
            raise ValueError('DEC1_RAW_FULL_AND_PREFIX_NATIVE_VALIDATORS_REQUIRED')
        offsets=np.asarray(reference.preview_offsets_m)
        if (reference.stale_timeout_s is None or not np.isfinite(reference.stale_timeout_s)
                or reference.stale_timeout_s<=0 or not np.isfinite(reference.lambda_m_per_rad)
                or reference.lambda_m_per_rad<=0 or offsets.shape!=(9,) or not np.isfinite(offsets).all()
                or np.any(offsets<0) or np.any(np.diff(offsets)<0)):
            raise ValueError('DEC1_EXPLICIT_STALE_LIMIT_AND_ORIGINAL_NINE_POINT_PREVIEW')
        if not all((reference.expected_source_frame,reference.wbc_frame,reference.tcp_semantic_id)):
            raise ValueError('DEC1_EXPLICIT_FRAME_AND_TCP_IDENTITIES')
        if reference.frame_transform is not None:
            tf=np.asarray(reference.frame_transform,dtype=float)
            if (tf.shape!=(4,4) or not np.isfinite(tf).all() or not np.allclose(tf[3],[0,0,0,1],atol=1e-10)
                    or not np.allclose(tf[:3,:3].T@tf[:3,:3],np.eye(3),atol=1e-10)
                    or not np.isclose(np.linalg.det(tf[:3,:3]),1.,atol=1e-10)):
                raise ValueError('DEC1_ONLY_EXPLICIT_RIGID_FRAME_TRANSFORM')
        self.reference=replace(reference,preview_offsets_m=_readonly(offsets),frame_transform=
            None if reference.frame_transform is None else _readonly(reference.frame_transform))
        self.config=config;self.mode=mode;self.episode_id=episode_id
        self._validators=(validate_raw,validate_mapped_full,validate_prefix)
        self._base=GR00TToWBCAdapterV2(self.reference)
        self._chunk=None;self._phase=None;self._pending=None;self._pending_digest=None
        self._last_feedback=None;self._last_ack=None;self._last_ack_end=None
        self._sequence=0;self._seen={};self.stopped_reason=None

    def _halt(self,reason):
        self.stopped_reason=reason
        raise PermissionError(reason)

    def _validate(self,index,value):
        if self._validators[index](value) is not None:
            self._halt('DEC1_NATIVE_VALIDATOR_MUST_RAISE_OR_RETURN_NONE')

    def _open(self):
        if self.stopped_reason:self._halt('DEC1_LATCHED_STOP:'+self.stopped_reason)
        if self._pending is not None:self._halt('DEC1_UNACKNOWLEDGED_REFERENCE_NO_NEW_PROPOSAL_OR_CHUNK')

    def _feedback(self,feedback,now):
        if (not isinstance(feedback,ActualFeedback) or not np.isfinite(now)
                or not 0<=now-feedback.timestamp<=self.config.feedback_max_age_s
                or feedback.source_frame!=self.reference.wbc_frame
                or feedback.tcp_semantic_id!=self.reference.tcp_semantic_id
                or feedback.predecessor_ack_id!=self._last_ack):
            self._halt('DEC1_STALE_FRAME_TCP_OR_PREDECESSOR_FEEDBACK')
        if self._last_feedback is not None and (feedback.timestamp<=self._last_feedback
                or feedback.timestamp-self._last_feedback>self.config.maximum_feedback_gap_s+1e-10
                or feedback.timestamp<self._last_ack_end-1e-10):
            self._halt('DEC1_FEEDBACK_MUST_FOLLOW_PREVIOUS_EXECUTED_ACTION')

    def load_chunk(self,chunk: TrajectoryChunkV1,feedback: ActualFeedback,*,controller_timestamp: float):
        self._open();self._feedback(feedback,controller_timestamp)
        try:
            times=np.asarray(chunk.point_timestamps)
            if (chunk.point_count!=40 or chunk.gripper is None or not np.isfinite(chunk.observation_timestamp)
                    or chunk.source_frame!=self.reference.expected_source_frame
                    or chunk.tcp_semantic_id!=self.reference.tcp_semantic_id
                    or np.shape(chunk.eef_position)!=(40,3) or np.shape(chunk.eef_orientation)!=(40,4)
                    or np.shape(chunk.gripper) not in ((40,),(40,1))
                    or not np.isfinite(chunk.validity_start) or not np.isfinite(chunk.validity_end)
                    or not np.allclose(np.diff(times),self.config.reference_period_s,rtol=0,atol=1e-10)
                    or not 0<=controller_timestamp-chunk.observation_timestamp<=self.config.new_chunk_max_age_s
                    or chunk.point_timestamps[0]<chunk.observation_timestamp-1e-10
                    or chunk.point_timestamps[0]-chunk.observation_timestamp>self.config.reference_period_s+1e-10
                    or chunk.validity_end<controller_timestamp or chunk.validity_end<times[-1]
                    or chunk.validity_start>times[0]
                    or self._last_ack_end is not None and chunk.observation_timestamp<self._last_ack_end-1e-10
                    or any(not np.isfinite(np.asarray(v)).all() for v in
                        (chunk.eef_position,chunk.eef_orientation,chunk.gripper))
                    or not np.allclose(np.linalg.norm(chunk.eef_orientation,axis=-1),1.,rtol=0,atol=1e-8)):
                self._halt('DEC1_FULL_H40_FIXED_20HZ_FINITE_FRESH_CHUNK_REQUIRED')
            raw=_copy_chunk(chunk);identity=_digest(raw)
            if chunk.chunk_id in self._seen:self._halt('DEC1_CHUNK_ID_NOT_REUSABLE_OR_MUTABLE')
            # The complete original chunk, including its unexecuted tail, goes
            # through the native raw gate before any frame/phase mapping.
            self._validate(0,raw)
            prepared=_copy_chunk(self._base._prepare_chunk(raw))
            object.__setattr__(prepared,'gripper',_readonly(prepared.gripper).reshape(-1))
            segments,cumulative=_arc_map(prepared.eef_position,prepared.eef_orientation,self.reference.lambda_m_per_rad)
            self._chunk=prepared;self._segments=segments;self._cumulative=cumulative
            self._stationary=segments<=self.config.stationary_arc_epsilon_m
            self._phase=None;self._seen[chunk.chunk_id]=identity
        except Exception as error:
            self.stopped_reason='DEC1_RAW_CHUNK_REJECTED:'+type(error).__name__
            raise

    def _preview(self,phase,now):
        chunk=self._chunk;end=39
        # Spatial preview alone drops the duration of repeated waypoints.
        # Both modes share the same original-pose dwell barrier: lookahead
        # ends at the next stationary segment until its phase has elapsed.
        for index in range(min(int(np.floor(phase)),38),39):
            if self._stationary[index] and phase<index+1:
                end=max(index,1);break
        view=replace(chunk,point_timestamps=chunk.point_timestamps[:end+1],
            eef_position=chunk.eef_position[:end+1],eef_orientation=chunk.eef_orientation[:end+1],
            gripper=chunk.gripper[:end+1])
        virtual=float(np.interp(phase,np.arange(40),chunk.point_timestamps))
        p=se3_arc_length_preview(view,anchor_timestamp=virtual,offsets_m=self.reference.preview_offsets_m,
            lambda_m_per_rad=self.reference.lambda_m_per_rad)
        p=replace(p,anchor_timestamp=now,provenance=dict(adapter='ExecutionProgressAdapter',
            reference_adapter_version='FRAME_CORRECTED_REFERENCE_V2',interface_mode=self.mode,
            phase=phase,virtual_path_anchor_timestamp=virtual,real_controller_timestamp=now,
            dwell_semantics='SHARED_ORIGINAL_STATIONARY_SEGMENT_ENDPOINT_V1',dwell_preview_endpoint_index=end,
            stale=False,raw_chunk_identity=self._seen[chunk.chunk_id]))
        for v in (p.offsets_m,p.eef_position,p.eef_rot6d,p.endpoint_clamped):v.setflags(write=False)
        return p

    def propose(self,feedback: ActualFeedback,*,controller_timestamp: float):
        self._open();self._feedback(feedback,controller_timestamp)
        if self._chunk is None:self._halt('DEC1_NO_ACCEPTED_CHUNK')
        chunk=self._chunk;now=float(controller_timestamp)
        if not 0<=now-chunk.observation_timestamp<=self.reference.stale_timeout_s or now>chunk.validity_end:
            self._halt('DEC1_STALE_OR_EXPIRED_INSTALLED_CHUNK')
        try:
            if self.mode=='I_TIME':
                phase=float(np.interp(now,chunk.point_timestamps,np.arange(40)))
            else:
                lo=0. if self._phase is None else self._phase
                hi=min(39.,lo+(self.config.initial_forward_points if self._phase is None else self.config.local_forward_points))
                phase=_project(chunk,feedback,lo,hi,self.config,self._stationary,self.reference.lambda_m_per_rad)
                i=min(int(lo),38);p,q=_pose(chunk,lo)
                near=(np.linalg.norm(feedback.eef_position-p)<=self.config.stationary_position_tolerance_m
                    and _angle(feedback.eef_orientation,q)<=self.config.stationary_rotation_tolerance_rad)
                if self._stationary[i] and near:
                    nominal=float(np.interp(now,chunk.point_timestamps,np.arange(40)))
                    elapsed=nominal if self._phase is None else (feedback.timestamp-self._last_feedback)/self.config.reference_period_s
                    phase=(min(hi,nominal,i+1.) if self._phase is None else
                        max(phase,min(hi,lo+elapsed,i+1.)))
            previous=0 if self._phase is None else int(self._phase)
            changes=[i for i in range(previous+1,int(phase)+1) if chunk.gripper[i]!=chunk.gripper[i-1]]
            if self._phase is None and changes:
                self._halt('DEC1_LATE_NEW_CHUNK_WOULD_SKIP_UNISSUED_GRIPPER_EVENT')
            if len(changes)>1:
                if self.mode=='I_TIME':self._halt('DEC1_CLOCK_GAP_WOULD_SKIP_PREDICTED_GRIPPER_EVENT')
                phase=changes[1]-2*self.config.phase_snap_epsilon
            previews=tuple(self._preview(min(39.,phase+i),now+i*self.config.reference_period_s) for i in range(40))
            phases=np.minimum(39.,phase+np.arange(40));indices=np.floor(phases).astype(int)
            full=TrajectoryChunkV1(chunk_id=chunk.chunk_id,source=chunk.source,observation_timestamp=chunk.observation_timestamp,
                point_timestamps=now+np.arange(40)*self.config.reference_period_s,
                eef_position=np.array([p.eef_position[0] for p in previews]),
                eef_orientation=np.array([p.eef_orientation[0] for p in previews]),
                source_frame=chunk.source_frame,tcp_semantic_id=chunk.tcp_semantic_id,
                validity_start=now,validity_end=now+39*self.config.reference_period_s,gripper=chunk.gripper[indices])
            full=_copy_chunk(full)
            grip=GripperCommandV1(timestamp=now,normalized_command=float(chunk.gripper[int(phase)]),stale=False,
                raw_source_convention='canonical_normalized',provenance=dict(chunk_id=chunk.chunk_id,phase=phase))
            self._sequence+=1
            command_id=f'{self.episode_id}:{self._sequence}:{self._seen[chunk.chunk_id]}'
            proposal=ReferenceProposal(command_id,now,feedback.timestamp,phase,
                float(np.interp(phase,np.arange(40),self._cumulative)),previews[0],grip,full,previews)
            self._validate(1,proposal)
            self._validate(2,proposal)
            self._pending=proposal;self._pending_digest=_proposal_digest(proposal)
            return proposal
        except Exception as error:
            self.stopped_reason='DEC1_MAPPED_REFERENCE_REJECTED:'+type(error).__name__
            raise

    def acknowledge(self,command_id: str,*,execution_end_timestamp: float):
        if self.stopped_reason:self._halt('DEC1_LATCHED_STOP:'+self.stopped_reason)
        if command_id==self._last_ack and execution_end_timestamp==self._last_ack_end:return
        p=self._pending
        if (p is None or command_id!=p.command_id or not np.isfinite(execution_end_timestamp)
                or not 0<execution_end_timestamp-p.controller_timestamp<=self.config.maximum_feedback_gap_s+1e-10
                or _proposal_digest(p)!=self._pending_digest):
            self._halt('DEC1_UNKNOWN_OR_MUTATED_EXECUTION_ACK')
        self._phase=p.phase;self._last_feedback=p.feedback_timestamp
        self._last_ack=command_id;self._last_ack_end=float(execution_end_timestamp)
        self._pending=None;self._pending_digest=None

    def state(self):
        return dict(episode_id=self.episode_id,mode=self.mode,phase=self._phase,
            pending_command_id=None if self._pending is None else self._pending.command_id,
            predecessor_ack_id=self._last_ack,execution_end_timestamp=self._last_ack_end,
            last_feedback_timestamp=self._last_feedback,stopped_reason=self.stopped_reason,
            chunk_id=None if self._chunk is None else self._chunk.chunk_id)
