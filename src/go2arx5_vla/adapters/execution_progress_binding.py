"""Cold DEC1 native-call wiring, not a safety rule or a physical executor.

Owner supplies unchanged executor callbacks, frozen tick rate and actual
feedback. Only dispatch_once may execute; no retry occurs after any error.
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np

from .execution_progress import (ActualFeedback, ExecutionProgressAdapter,
    ProgressConfig, canonical_chunk_from_row_eef10, _digest)


@dataclass(frozen=True)
class NativeReceipt:
    episode_id: str
    executor_id: str
    command_id: str
    receipt_id: str
    ticks: int
    execution_end_timestamp: float | None = None


@dataclass(frozen=True)
class NativeShadowReceipt(NativeReceipt):
    decision: str = 'PASS'


class NativeShadowRejected(PermissionError):
    """Known native S1 rejection with identity-bound actual partial ticks."""
    def __init__(self, receipt):
        super().__init__('DEC1_KNOWN_NATIVE_S1_REJECT')
        self.receipt=receipt


class NativeShadowUnsupported(PermissionError):
    """The declared preview ABI has no qualified native shadow implementation."""


@dataclass(frozen=True)
class NativeStopReceipt:
    episode_id: str
    executor_id: str
    receipt_id: str
    stopped: bool


class BoundProgressExecutor:
    """One episode and executor; native callers retain admission/stop ownership.

    Validators raise on rejection. Raw/full validators return None only on
    acceptance. Prefix shadow and dispatch return NativeReceipt. dispatch
    installs a single paired preview/gripper and advances one reference tick
    interval; the outer original loop invokes step eight times per chunk.
    """
    def __init__(self, reference, *, mode, episode_id, executor_id,
                 ticks_per_reference, validate_raw_eef10: Callable,
                 validate_mapped_full: Callable, prefix_shadow: Callable,
                 dispatch_once: Callable, actual_feedback: Callable,
                 stop: Callable, record: Callable):
        callbacks=(validate_raw_eef10,validate_mapped_full,prefix_shadow,
                   dispatch_once,actual_feedback,stop,record)
        if (not episode_id or not executor_id or type(ticks_per_reference) is not int
                or ticks_per_reference<=0 or not all(callable(f) for f in callbacks)):
            raise ValueError('DEC1_EXPLICIT_NATIVE_CALLBACKS_AND_FROZEN_TICK_RATE')
        if reference.lambda_m_per_rad!=.15 or not np.array_equal(reference.preview_offsets_m,
                [.01,.025,.05,.09,.15,.225,.325,.425,.5]):
            raise ValueError('DEC1_FROZEN_SHARED_PREVIEW_PARAMETERS')
        self.episode_id=episode_id;self.executor_id=executor_id
        self.ticks_per_reference=ticks_per_reference
        self.raw_gate=validate_raw_eef10;self.shadow=prefix_shadow
        self.dispatch=dispatch_once;self.feedback=actual_feedback;self.stop=stop;self.record=record
        self.stopped=False;self.stop_reason=None;self.stop_calls=0
        self._raw_digest=None;self._receipts=set();self.events=[]
        self.shadow_ticks=0;self.live_ticks=0;self.unknown_dispatches=0;self.unknown_shadows=0
        self.stop_confirmed=False
        self.known_shadow_rejections=0;self.unsupported_shadows=0;self.last_shadow_decision=None
        self._adapter=ExecutionProgressAdapter(reference,ProgressConfig(),mode=mode,
            episode_id=episode_id,validate_raw=self._validated_raw,
            validate_mapped_full=validate_mapped_full,validate_prefix=self._prefix)

    def _record(self,event):
        event=dict(event,episode_id=self.episode_id,executor_id=self.executor_id)
        self.record(event)
        self.events.append(event)

    def _open(self):
        if self.stopped:raise PermissionError('DEC1_EXECUTOR_STOP_LATCHED:'+self.stop_reason)

    def _fail(self,error):
        if not self.stopped:
            self.stopped=True;self.stop_reason=type(error).__name__+':'+str(error)
            self.stop_calls+=1
            ack=self.stop(self.stop_reason)
            if (not isinstance(ack,NativeStopReceipt) or ack.episode_id!=self.episode_id
                    or ack.executor_id!=self.executor_id or not ack.receipt_id or ack.stopped is not True):
                raise PermissionError('DEC1_NATIVE_STOP_NOT_ACKNOWLEDGED') from error
            self.stop_confirmed=True
            self._record(dict(kind='STOP_ACK',receipt_id=ack.receipt_id,reason=self.stop_reason))

    def _validated_raw(self,chunk):
        if _digest(chunk)!=self._raw_digest:
            raise PermissionError('DEC1_RAW_NATIVE_GATE_NOT_BOUND_TO_THIS_FULL_CHUNK')

    def _receipt(self,value,command_id):
        if (not isinstance(value,NativeReceipt) or value.episode_id!=self.episode_id
                or value.executor_id!=self.executor_id or value.command_id!=command_id
                or not value.receipt_id or value.receipt_id in self._receipts
                or type(value.ticks) is not int or value.ticks<0):
            raise PermissionError('DEC1_NATIVE_RECEIPT_IDENTITY_TICKS_OR_REUSE')
        self._receipts.add(value.receipt_id)
        return value

    def _prefix(self,proposal):
        # Native shadow applies its existing safety checks to all eight mapped
        # previews, and reports actual integrations for original accounting.
        self._record(dict(kind='PREFIX_SHADOW_INTENT',command_id=proposal.command_id))
        self.unknown_shadows+=1
        try:
            value=self.shadow(proposal)
        except NativeShadowRejected as error:
            value=error.receipt
            if not isinstance(value,NativeShadowReceipt) or value.decision!='REJECT':
                raise PermissionError('DEC1_REJECT_EXCEPTION_NEEDS_KNOWN_NATIVE_RECEIPT') from error
        receipt=self._receipt(value,proposal.command_id)
        self.unknown_shadows-=1
        self.shadow_ticks+=receipt.ticks
        decision=getattr(receipt,'decision','PASS')
        self.last_shadow_decision=decision
        self._record(dict(kind='MAPPED_PREFIX_SHADOW',receipt_id=receipt.receipt_id,
            command_id=proposal.command_id,ticks=receipt.ticks,decision=decision))
        if decision=='REJECT':
            self.known_shadow_rejections+=1
            raise NativeShadowRejected(receipt)
        if decision=='NOT_SUPPORTED' and receipt.ticks==0:
            self.unsupported_shadows+=1
            raise NativeShadowUnsupported('DEC1_NATIVE_PREVIEW_SHADOW_ABI_NOT_SUPPORTED')
        if decision!='PASS':
            raise PermissionError('DEC1_INVALID_NATIVE_SHADOW_DECISION')
        if receipt.ticks!=8*self.ticks_per_reference:
            raise PermissionError('DEC1_NATIVE_PREFIX_SHADOW_TICK_COUNT')

    def load_raw(self, *, raw_eef10, chunk_id, observation_timestamp,
                 point_timestamps, source_frame, tcp_semantic_id, controller_timestamp):
        self._open()
        try:
            chunk=canonical_chunk_from_row_eef10(raw_eef10=raw_eef10,chunk_id=chunk_id,
                observation_timestamp=observation_timestamp,point_timestamps=point_timestamps,
                source_frame=source_frame,tcp_semantic_id=tcp_semantic_id,
                validate_raw_eef10=self.raw_gate)
            self._raw_digest=_digest(chunk)
            actual=self.feedback()
            self._adapter.load_chunk(chunk,actual,controller_timestamp=controller_timestamp)
            self._record(dict(kind='RAW_H40_ACCEPTED',chunk_id=chunk_id,
                raw_chunk_identity=self._raw_digest))
        except Exception as error:
            self._fail(error);raise

    def step(self, *, controller_timestamp):
        self._open()
        try:
            before=self.feedback()
            proposal=self._adapter.propose(before,controller_timestamp=controller_timestamp)
            self._record(dict(kind='EXECUTION_INTENT',command_id=proposal.command_id))
            self.unknown_dispatches+=1
            ack=self._receipt(self.dispatch(proposal),proposal.command_id)
            self.unknown_dispatches-=1
            # Account committed native ticks before checking the ACK/feedback;
            # a later rejection cannot make an already executed action free.
            self.live_ticks+=ack.ticks
            self._record(dict(kind='EXECUTION_ACK',receipt_id=ack.receipt_id,
                command_id=ack.command_id,ticks=ack.ticks))
            if ack.ticks!=self.ticks_per_reference:
                raise PermissionError('DEC1_NATIVE_EXECUTED_TICK_COUNT')
            self._adapter.acknowledge(ack.command_id,
                execution_end_timestamp=ack.execution_end_timestamp)
            after=self.feedback()
            # Same unchanged freshness/predecessor/order gate used by propose.
            self._adapter._feedback(after,ack.execution_end_timestamp)
            return proposal,ack,after
        except Exception as error:
            self._fail(error);raise

    def state(self):
        return dict(adapter=self._adapter.state(),stopped=self.stopped,
            stop_reason=self.stop_reason,stop_calls=self.stop_calls,
            live_ticks=self.live_ticks,shadow_ticks=self.shadow_ticks,
            unknown_dispatches=self.unknown_dispatches,unknown_shadows=self.unknown_shadows,
            stop_confirmed=self.stop_confirmed,known_shadow_rejections=self.known_shadow_rejections,
            unsupported_shadows=self.unsupported_shadows,last_shadow_decision=self.last_shadow_decision,
            native_qualification='NATIVE_COMMISSIONING_NOT_RUN')
