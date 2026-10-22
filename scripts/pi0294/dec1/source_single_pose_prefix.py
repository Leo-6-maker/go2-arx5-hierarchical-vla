"""Cold single-pose DEC1 source path: one clone and 200 shadow ticks/prefix.

Native qualification is pending. This module does not
qualify preview9; the separate SourcePrefixShadow NOT_SUPPORTED stays intact.
Only actual TCP/gripper/time enters the progress adapter. Native safety and
observed-state parity retain the original source primitives and thresholds.
The owner supplies original authority/lease admission and durable accounting.
"""
from copy import deepcopy
import hashlib
import json
import uuid

import numpy as np
from scripts.pi0292.eval_contract import FRAME,TCP,gate_action,matrix10
from scripts.pi0294 import s1_preflight as native
from go2arx5_vla.adapters.execution_progress import (
    ProgressConfig,ActualFeedback,canonical_chunk_from_row_eef10,_proposal_digest,_digest,
)
from go2arx5_vla.adapters.execution_progress_binding import NativeStopReceipt
from go2arx5_vla.trajectory.so3 import matrix_to_quat,quat_to_matrix,rotation_matrix_to_rot6d_rows

VERSION='DEC1_SINGLE_POSE_PROGRESS_PREFIX8_V1'
TICKS_PER_REFERENCE=25
PREFIX_REFERENCES=8
PREFIX_TICKS=200
PARITY_TOLERANCE=1e-10  # Same as original ProtectedTeacher.compare_previous.


class PrefixFailure(PermissionError):
    pass


def _row10(snapshot):
    tcp=np.asarray(snapshot['tcp'],dtype=float)
    grip=float(snapshot['gripper_fraction']);now=float(snapshot['time_s'])
    if tcp.shape!=(4,4) or not np.isfinite(tcp).all() or not np.isfinite([grip,now]).all():
        raise PrefixFailure('SOURCE_ACTUAL_TCP_GRIPPER_CLOCK_REQUIRED')
    return np.r_[tcp[:3,3],rotation_matrix_to_rot6d_rows(tcp[:3,:3]),grip]


def _mapped10(proposal):
    c=proposal.mapped_full
    if c.source_frame!=FRAME or c.tcp_semantic_id!=TCP:
        raise PrefixFailure('SOURCE_MAPPED_FRAME_TCP_REQUIRED')
    return np.column_stack((c.eef_position,rotation_matrix_to_rot6d_rows(quat_to_matrix(c.eef_orientation)),c.gripper))


def _action(proposal):
    rows=_mapped10(proposal)
    target=matrix10(rows[:1])[0]
    opening=float(proposal.gripper.normalized_command)
    if rows.shape!=(40,10) or rows[0,9]!=opening:
        raise PrefixFailure('SOURCE_SINGLE_POSE_GRIPPER_NOT_MAPPED_FIRST_REFERENCE')
    return target,opening


class SinglePosePrefixExecutor:
    """One owner, one source engine, one progress adapter; failure is terminal.

    reserve(kind, upper_ticks, clone_attempts) -> reservation_id
    charge(kind, known_ticks, unknown, context) records each native step attempt.
    finish(kind, reservation_id, known_ticks, unknown, status) retains liabilities.
    record(event) is the existing owner's durable evidence/intent recorder.
    stop(reason) must invoke the actual stop and return NativeStopReceipt.
    No retry, ledger writer, timer reset, model, scene or controller is provided.
    """
    def __init__(self,engine,adapter,*,executor_id,guard,reserve,charge,finish,record,stop):
        reference=adapter.reference
        if (reference.expected_source_frame!=FRAME or reference.wbc_frame!=FRAME
                or reference.tcp_semantic_id!=TCP or reference.frame_transform is not None
                or reference.lambda_m_per_rad!=.15 or not np.array_equal(reference.preview_offsets_m,
                    [.01,.025,.05,.09,.15,.225,.325,.425,.5])
                or adapter.config!=ProgressConfig() or adapter.state()['pending_command_id'] is not None
                or not executor_id or not all(callable(f) for f in (guard,reserve,charge,finish,record,stop))):
            raise PrefixFailure('EXACT_IDLE_SOURCE_FRAME_TCP_FIXED_CONFIG_AND_OWNER_CALLBACKS')
        self.engine=engine;self.adapter=adapter;self.executor_id=executor_id
        self.episode_id=adapter.episode_id;self.guard=guard;self.reserve=reserve
        self.charge=charge;self.finish=finish;self.record=record;self.stop=stop
        self.stopped=False;self.stop_confirmed=False;self.stop_reason=None;self.stop_calls=0
        self.pending=None;self.counts={'shadow':0,'live':0};self.unknown={'shadow':False,'live':False}
        self.clone_attempts=0

    def _event(self,kind,**values):
        self.record(dict(kind=kind,episode_id=self.episode_id,executor_id=self.executor_id,**values))

    def _open(self):
        if self.stopped:raise PrefixFailure('SOURCE_STOP_LATCHED:'+self.stop_reason)
        try:self.guard()
        except BaseException as error:self._halt(error);raise

    def _halt(self,error):
        if not self.stopped:
            self.stopped=True;self.stop_reason=type(error).__name__+':'+str(error);self.stop_calls+=1
            try:
                ack=self.stop(self.stop_reason)
                if (not isinstance(ack,NativeStopReceipt) or ack.episode_id!=self.episode_id
                        or ack.executor_id!=self.executor_id or not ack.receipt_id or ack.stopped is not True):
                    raise PrefixFailure('SOURCE_ACTUAL_STOP_ACK_MISSING') from error
                self._event('SOURCE_STOP_ACK',receipt_id=ack.receipt_id,reason=self.stop_reason)
                self.stop_confirmed=True
            except BaseException as cleanup_error:
                # Stop intent is already latched; an interrupted durable/native
                # callback must not permit a second stop or release liabilities.
                self.unknown['live']=True
                if not isinstance(error,Exception) and cleanup_error is not error:
                    if error.__cause__ is None:raise error from cleanup_error
                    raise error
                raise

    def _gate(self,rows,snapshot,identifier):
        result=gate_action(rows,_row10(snapshot),snapshot['time_s'],identifier)
        self._event('SOURCE_FULL_H40_GATE',reference_id=identifier,decision=result)
        if not isinstance(result,dict) or result.get('decision') not in ('ACCEPT','WARN'):
            raise PrefixFailure('SOURCE_ORIGINAL_FULL_H40_REJECT')

    def _bind(self,adapter,box,chunk_digest):
        def raw(chunk):
            if _digest(chunk)!=chunk_digest:raise PrefixFailure('SOURCE_RAW_CANONICAL_BINDING_CHANGED')
        def full(proposal):self._gate(_mapped10(proposal),box['actual'],proposal.command_id)
        def prefix_format(proposal):
            # No nested shadow call here: all eight feedback-reanchored single
            # poses are checked in ONE 200-tick native clone below.
            _action(proposal)
        adapter._validators=(raw,full,prefix_format)

    def _feedback(self,snapshot,adapter):
        tcp=np.asarray(snapshot['tcp'])
        _row10(snapshot)
        return ActualFeedback(snapshot['time_s'],tcp[:3,3],matrix_to_quat(tcp[:3,:3]),
            snapshot['gripper_fraction'],FRAME,TCP,adapter.state()['predecessor_ack_id'])

    def _proposal(self,adapter,engine,box,raw,identifier):
        box['actual']=engine.snapshot()
        # Recheck the original complete raw H40 BEFORE this reference's mapping.
        self._gate(raw,box['actual'],identifier)
        return adapter.propose(self._feedback(box['actual'],adapter),
            controller_timestamp=box['actual']['time_s'])

    def _tick(self,kind,engine,target,opening,context):
        self.guard();before=engine.steps;error=None
        self._event('SOURCE_NATIVE_TICK_INTENT',domain=kind,native_step_before=before,**context)
        try:engine.step(target,opening)
        except BaseException as caught:error=caught  # Reconcile even an interrupted native step.
        try:
            delta=engine.steps-before
            if type(delta) is not int or delta!=1:
                self.unknown[kind]=True
                known=delta if type(delta) is int and delta>0 else 0
                self.counts[kind]+=known
                self.charge(kind,known,True,dict(context,native_delta=delta,error='COUNTER_UNCERTAIN'))
                if error is not None and not isinstance(error,Exception):raise error
                raise PrefixFailure('SOURCE_NATIVE_TICK_ACCOUNTING_UNCERTAIN') from error
            self.counts[kind]+=1
            self.charge(kind,1,False,dict(context,native_step_after=engine.steps))
            engine.snapshot()  # Same original post-step record_tick observation.
            state=np.asarray(native.observed_state(engine)).copy()
            if state.ndim!=1 or not state.size or not np.isfinite(state).all():
                raise PrefixFailure('SOURCE_NONFINITE_OR_MISSING_NATIVE_OBSERVED_STATE')
        except BaseException:
            # An already charged step remains charged even if post-state or the
            # durable charge callback failed; never claim a zero-effect retry.
            self.unknown[kind]=True
            raise
        if engine.safety:raise PrefixFailure('SOURCE_ORIGINAL_NATIVE_S1_REJECT') from error
        if error is not None:raise error
        return state

    def _reserve(self,kind,clones):
        self._event('SOURCE_PREFIX_RESERVE_INTENT',domain=kind,upper_ticks=PREFIX_TICKS,clone_attempts=clones)
        try:identifier=self.reserve(kind,PREFIX_TICKS,clones)
        except BaseException:
            self.unknown[kind]=True;raise
        if not isinstance(identifier,str) or not identifier:
            self.unknown[kind]=True;raise PrefixFailure('SOURCE_RESERVATION_ACK_REQUIRED')
        return identifier

    def prepare(self,raw_eef10,*,chunk_id,observation_timestamp,point_timestamps):
        self._open()
        if self.pending is not None:
            error=PrefixFailure('SOURCE_PREFIX_ALREADY_RESERVED_NO_RETRY');self._halt(error);raise error
        shadow=None;reservation=None;before=None;start_count=self.counts['shadow'];status='ENGINEERING_MISSING';primary_error=None
        try:
            raw=np.asarray(raw_eef10,dtype=float).copy()
            actual=self.engine.snapshot()
            before=native.fingerprint(self.engine)
            chunk=canonical_chunk_from_row_eef10(raw_eef10=raw,chunk_id=chunk_id,
                observation_timestamp=observation_timestamp,point_timestamps=point_timestamps,
                source_frame=FRAME,tcp_semantic_id=TCP,
                validate_raw_eef10=lambda value:self._gate(value,actual,chunk_id))
            reservation=self._reserve('shadow',1);self.clone_attempts+=1
            shadow=native.clone(self.engine)  # The only clone for this prefix.
            if native.fingerprint(self.engine)!=before:
                self.unknown['live']=True;raise PrefixFailure('SOURCE_CLONE_CHANGED_LIVE_OR_RNG')
            trial=deepcopy(self.adapter);box={'actual':shadow.snapshot()}
            self._bind(trial,box,_digest(chunk))
            trial.load_chunk(chunk,self._feedback(box['actual'],trial),controller_timestamp=box['actual']['time_s'])
            records=[];poses=[];openings=[]
            for reference in range(PREFIX_REFERENCES):
                proposal=self._proposal(trial,shadow,box,raw,chunk_id)
                target,opening=_action(proposal);expected=[]
                record=dict(command_id=proposal.command_id,digest=_proposal_digest(proposal),phase=proposal.phase,
                    predecessor_ack_id=trial.state()['predecessor_ack_id'],target=target.copy(),opening=opening)
                for tick in range(TICKS_PER_REFERENCE):
                    expected.append(self._tick('shadow',shadow,target,opening,dict(reference=reference,tick=tick)))
                end=shadow.snapshot()['time_s']
                trial.acknowledge(proposal.command_id,execution_end_timestamp=end)
                record.update(execution_end_timestamp=end,expected_states=np.asarray(expected))
                records.append(record);poses.append(target);openings.append(opening)
            if native.fingerprint(self.engine)!=before:
                self.unknown['live']=True;raise PrefixFailure('SOURCE_SHADOW_CHANGED_LIVE_OR_RNG')
            nonce=str(uuid.uuid4());poses=np.asarray(poses);openings=np.asarray(openings)
            forecast=dict(version=VERSION,decision='PASS',nonce=nonce,state_sha256=before,
                action_sha256=hashlib.sha256(poses.tobytes()+openings.tobytes()).hexdigest(),shadow_integrations=PREFIX_TICKS)
            self.pending=dict(nonce=nonce,raw=raw,chunk=chunk,records=records,poses=poses,openings=openings,
                forecast=forecast,adapter_state=deepcopy(self.adapter.state()),adapter_sequence=self.adapter._sequence,
                native_step=self.engine.steps)
            status='PASS'
            self._event('SOURCE_PREFIX_SHADOW_COMPLETE',nonce=nonce,shadow_ticks=PREFIX_TICKS,
                clone_attempts=1,proposal_digests=[r['digest'] for r in records],phases=[r['phase'] for r in records])
            return dict(status='PASS_SHADOW_ONLY_NATIVE_QUALIFICATION_PENDING',nonce=nonce,
                shadow_ticks=PREFIX_TICKS,clone_attempts=1)
        except BaseException as error:
            primary_error=error
            status='ENGINEERING_MISSING'  # An interrupted completion event is not PASS.
            if isinstance(error,PrefixFailure) and str(error)=='SOURCE_ORIGINAL_NATIVE_S1_REJECT':status='S1_REJECT'
            elif isinstance(error,PrefixFailure) and str(error)=='SOURCE_ORIGINAL_FULL_H40_REJECT':status='REFERENCE_REJECT'
            try:self._halt(error)
            except BaseException as halt_error:
                if not isinstance(halt_error,Exception):primary_error=halt_error
                raise
            raise
        finally:
            cleanup_error=None
            try:
                if before is not None and native.fingerprint(self.engine)!=before:
                    self.unknown['live']=True
                    raise PrefixFailure('SOURCE_SHADOW_CHANGED_LIVE_OR_RNG')
            except BaseException as error:
                self.unknown['live']=True;cleanup_error=error
            try:
                if shadow is not None:shadow.close()
            except BaseException as error:
                self.unknown['shadow']=True;cleanup_error=cleanup_error or error
            try:
                if reservation is not None:
                    self.finish('shadow',reservation,self.counts['shadow']-start_count,
                        self.unknown['shadow'],'ENGINEERING_MISSING' if cleanup_error else status)
            except BaseException as error:
                self.unknown['shadow']=True;cleanup_error=cleanup_error or error
            if cleanup_error is not None:
                try:self._halt(cleanup_error)
                except BaseException as halt_error:
                    if primary_error is not None and not isinstance(primary_error,Exception):
                        if primary_error.__cause__ is None:raise primary_error from halt_error
                        raise primary_error
                    raise
                if primary_error is not None and not isinstance(primary_error,Exception):
                    if primary_error.__cause__ is None:raise primary_error from cleanup_error
                    raise primary_error
                raise cleanup_error

    def execute_prefix(self,nonce):
        self._open();reservation=None;start_count=self.counts['live'];status='ENGINEERING_MISSING'
        try:
            pending=self.pending
            if (pending is None or nonce!=pending['nonce'] or self.engine.steps!=pending['native_step']
                    or self.adapter.state()!=pending['adapter_state'] or self.adapter._sequence!=pending['adapter_sequence']):
                raise PrefixFailure('SOURCE_EXACT_PENDING_PREFIX_AND_UNCHANGED_ACTUAL_START_REQUIRED')
            native.bind_install(self.engine,pending['poses'],pending['openings'],pending['forecast'],nonce)
            reservation=self._reserve('live',0)
            box={'actual':self.engine.snapshot()};self._bind(self.adapter,box,_digest(pending['chunk']))
            self.adapter.load_chunk(pending['chunk'],self._feedback(box['actual'],self.adapter),controller_timestamp=box['actual']['time_s'])
            for reference,expected in enumerate(pending['records']):
                proposal=self._proposal(self.adapter,self.engine,box,pending['raw'],pending['chunk'].chunk_id)
                target,opening=_action(proposal)
                if (_proposal_digest(proposal)!=expected['digest'] or proposal.phase!=expected['phase']
                        or proposal.command_id!=expected['command_id']
                        or self.adapter.state()['predecessor_ack_id']!=expected['predecessor_ack_id']
                        or not np.array_equal(target,expected['target']) or opening!=expected['opening']):
                    raise PrefixFailure('SOURCE_LIVE_PROPOSAL_DIFFERS_FROM_200TICK_SHADOW')
                for tick in range(TICKS_PER_REFERENCE):
                    state=self._tick('live',self.engine,target,opening,dict(reference=reference,tick=tick))
                    if state.shape!=expected['expected_states'][tick].shape or not np.isfinite(state).all() or np.max(np.abs(state-expected['expected_states'][tick]))>PARITY_TOLERANCE:
                        raise PrefixFailure('S1_ACTUAL_LIVE_SHADOW_PARITY_MISMATCH')
                end=self.engine.snapshot()['time_s']
                if end!=expected['execution_end_timestamp']:raise PrefixFailure('SOURCE_ACTUAL_ACK_CLOCK_DIFFERS_FROM_SHADOW')
                self.adapter.acknowledge(proposal.command_id,execution_end_timestamp=end)
                self._event('SOURCE_REFERENCE_ACK',nonce=nonce,reference=reference,command_id=proposal.command_id,
                    execution_end_timestamp=end,actual_ticks=TICKS_PER_REFERENCE)
            if self.counts['live']-start_count!=PREFIX_TICKS:raise PrefixFailure('SOURCE_EXACT_PREFIX200_ACK_REQUIRED')
            self.pending=None;status='PASS'
            self._event('SOURCE_PREFIX200_ACK',nonce=nonce,actual_ticks=PREFIX_TICKS,final_command_id=proposal.command_id)
            return dict(status='PASS_SINGLE_POSE_PREFIX200',actual_ticks=PREFIX_TICKS,nonce=nonce)
        except BaseException as error:
            status='ENGINEERING_MISSING'
            if isinstance(error,PrefixFailure) and str(error) in ('SOURCE_ORIGINAL_NATIVE_S1_REJECT','S1_ACTUAL_LIVE_SHADOW_PARITY_MISMATCH'):
                status='NATIVE_SAFETY_ERROR'
            self._halt(error);raise
        finally:
            if reservation is not None:
                try:self.finish('live',reservation,self.counts['live']-start_count,self.unknown['live'],status)
                except BaseException as error:
                    self.unknown['live']=True;self._halt(error);raise

    def state(self):
        return dict(stopped=self.stopped,stop_confirmed=self.stop_confirmed,stop_calls=self.stop_calls,
            stop_reason=self.stop_reason,known_ticks=dict(self.counts),unknown=dict(self.unknown),
            clone_attempts=self.clone_attempts,pending_nonce=None if self.pending is None else self.pending['nonce'],
            native_qualification='NATIVE_COMMISSIONING_NOT_RUN',ABI='SINGLE_4x4_POSE_PLUS_GRIPPER_NOT_PREVIEW9')
