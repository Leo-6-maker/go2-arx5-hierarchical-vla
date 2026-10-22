"""Cold synchronous-source live-step stop barrier; CPU verified, native pending.

No engine state/locks are placed in engine.__dict__. Original frozen numerical
step/snapshot functions are supplied by an already admitted cold owner. The
same original step is forwarded verbatim for private S1 clones. This is not an
asynchronous WBC stop API or a sandbox against direct foreign mj_step calls.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import weakref
from scripts.pi0292.durable import read,write
from go2arx5_vla.adapters.execution_progress_binding import NativeStopReceipt

SCHEMA='DEC1_SOURCE_SYNCHRONOUS_LIVE_STOP_V1'
_INSTALL_LOCK=threading.Lock()
_OWNERS=weakref.WeakKeyDictionary()  # External only; never pickled by frozen S1.


class SourceStopError(PermissionError):
    pass


class SourceSyncStopBarrier:
    """Construct ONCE around a new, unescaped source engine at native tick0.

    Existing single owner/lease/source-seal admission MUST precede construction.
    The owner retains its lock for the whole episode and supplies exact frozen
    BaseEngine.step and BaseEngine.snapshot functions (not bound methods).
    No background integrator, saved raw-step alias, direct mj_step or foreign
    mutation is permitted by that source ABI. No model/clone/step is created here.
    """
    def __init__(self,engine,*,episode_id,executor_id,receipt_path,source_step_sha256,
                 expected_step,expected_snapshot,fingerprint,writer=write,reader=read):
        if (not isinstance(episode_id,str) or not episode_id or not isinstance(executor_id,str) or not executor_id
                or not isinstance(source_step_sha256,str) or re.fullmatch('[0-9a-f]{64}',source_step_sha256) is None
                or not all(callable(f) for f in (writer,reader,fingerprint))):
            raise SourceStopError('EXACT_OWNER_IDENTITIES_FROZEN_SOURCE_AND_DURABLE_IO_REQUIRED')
        self._pid=os.getpid();self._thread=threading.get_ident();self._lock=threading.Lock()
        self._engine=weakref.ref(engine);self._base=type(engine)
        self._step=expected_step;self._snapshot=expected_snapshot
        self._path=Path(receipt_path);self._source_sha=source_step_sha256
        self.episode_id=episode_id;self.executor_id=executor_id
        self._write=writer;self._read=reader
        self._fingerprint=fingerprint
        self._latched=False;self._inflight=False;self._stop_attempted=False;self._acknowledged=False
        self._unknown=False;self._fault=None;self._last_attempt=None;self._observed_steps=0
        if (self._base.step is not expected_step or self._base.snapshot is not expected_snapshot
                or 'step' in engine.__dict__ or 'snapshot' in engine.__dict__
                or getattr(engine,'mode',None)!='normal' or hasattr(engine,'finger_driver')
                or type(engine.steps) is not int or engine.steps!=0 or float(engine.data.time)!=0.
                or float(engine.model.opt.timestep)!=.002
                or not self._path.parent.is_dir() or self._path.exists() or self._path.is_symlink()):
            raise SourceStopError('EXACT_FRESH_FROZEN_SYNCHRONOUS_ENGINE_AND_NEW_RECEIPT_PATH_REQUIRED')
        original_step=expected_step;live=self._engine;barrier=self
        def forward_step(instance,target,opening):
            if instance is live():return barrier._advance(target,opening)
            # Frozen clone() makes the same type with copied MjData and no new
            # instance fields. It remains a private original engine, not live.
            return original_step(instance,target,opening)
        self._forward=forward_step
        guarded=type('DEC1SyncLive_'+self._base.__name__,(self._base,),
            {'__module__':__name__,'step':forward_step})
        self._guarded=guarded
        with _INSTALL_LOCK:
            if engine in _OWNERS:raise SourceStopError('NO_SECOND_OWNER_OR_BARRIER_ON_THIS_ENGINE')
            if type(engine) is not self._base or self._base.step is not expected_step:
                raise SourceStopError('COLD_INSTALLATION_CONFLICT')
            before_keys=tuple(engine.__dict__)
            install_sha=self._fingerprint(engine)
            if not isinstance(install_sha,str) or re.fullmatch('[0-9a-f]{64}',install_sha) is None:
                raise SourceStopError('ORIGINAL_COLD_INSTALL_FINGERPRINT_REQUIRED')
            try:
                engine.__class__=guarded
                if tuple(engine.__dict__)!=before_keys or type(engine).step is not forward_step:
                    raise SourceStopError('COLD_INSTALLATION_CHANGED_NATIVE_INSTANCE_FIELDS')
                if self._fingerprint(engine)!=install_sha:
                    raise SourceStopError('COLD_INSTALLATION_CHANGED_ORIGINAL_NATIVE_STATE_OR_RNG')
                _OWNERS[engine]=self
            except BaseException:
                # No native call and no escaped engine exists at this point.
                engine.__class__=self._base
                raise

    def _violation(self,message):
        self._latched=True;self._unknown=True
        if self._fault is None:self._fault=message
        return SourceStopError(message)

    def _owner(self):
        if os.getpid()!=self._pid or threading.get_ident()!=self._thread:
            raise self._violation('FOREIGN_PROCESS_OR_THREAD_NO_NATIVE_STOP_ACK')

    def _binding(self):
        engine=self._engine()
        if (engine is None or type(engine) is not self._guarded or type(engine).step is not self._forward
                or self._base.step is not self._step or self._base.snapshot is not self._snapshot
                or 'step' in engine.__dict__ or 'snapshot' in engine.__dict__):
            raise self._violation('OWNED_ENGINE_OR_FROZEN_FUNCTION_BINDING_CHANGED')
        return engine

    def _advance(self,target,opening):
        self._owner()
        if self._latched:raise SourceStopError('SOURCE_LIVE_STEP_STOP_LATCHED')
        if not self._lock.acquire(blocking=False):raise self._violation('REENTRANT_OR_INFLIGHT_NATIVE_STEP')
        error=None;result=None;fault=None
        try:
            engine=self._binding()
            if self._inflight or self._latched:raise self._violation('REENTRANT_OR_STOPPED_NATIVE_STEP')
            if type(engine.steps) is not int or engine.steps!=self._observed_steps:
                raise self._violation('UNOWNED_NATIVE_COUNTER_CHANGE')
            before=engine.steps
            try:before_time=float(engine.data.time)
            except BaseException as caught:
                raise self._violation('OWNED_NATIVE_CLOCK_READBACK_UNKNOWN') from caught
            if not math.isfinite(before_time) or abs(before_time-before*.002)>1e-8:
                raise self._violation('UNOWNED_NATIVE_CLOCK_CHANGE')
            self._inflight=True
            try:result=self._step(engine,target,opening)
            except BaseException as caught:error=caught
            finally:
                after=engine.steps;after_time=float(engine.data.time)
                delta=after-before if type(after) is int else None
                self._last_attempt=dict(tick_before=before,tick_after=after,native_delta=delta,
                    time_before_s=before_time,time_after_s=after_time,
                    native_error=None if error is None else type(error).__name__+':'+str(error))
                if type(after) is int:self._observed_steps=after
                if (delta!=1 or not math.isfinite(after_time) or not math.isfinite(before_time)
                        or abs(after_time-after*.002)>1e-8 or abs(after_time-before_time-.002)>1e-8):
                    fault=self._violation('NATIVE_COUNTER_OR_CLOCK_EFFECT_UNKNOWN')
                self._inflight=False
                if error is not None:self._latched=True
                if self._fault is not None and fault is None:fault=SourceStopError(self._fault)
        finally:
            # An exception reading a real native counter/clock must never leave
            # an apparent idle success: keep unknown + latch and clear only the
            # local active-frame bit after this native Python frame unwinds.
            if self._inflight:
                self._unknown=True;self._latched=True;self._fault=self._fault or 'NATIVE_FRAME_READBACK_UNKNOWN'
                self._inflight=False
            self._lock.release()
        if fault is not None:raise fault from error
        if error is not None:raise error
        return result

    def _stopped_readback(self):
        engine=self._binding()
        steps=engine.steps
        if type(steps) is not int or steps!=self._observed_steps:
            raise self._violation('STOP_COUNTER_DIFFERS_FROM_LAST_OWNED_NATIVE_EFFECT')
        # Original snapshot() calls mj_forward and may update integration
        # warm-start state. STOP must observe the owned native clock directly,
        # without invoking a state-mutating observation or repairing state.
        clock=float(engine.data.time)
        if (engine.steps!=steps or not math.isfinite(clock)
                or float(engine.data.time)!=clock or abs(clock-steps*.002)>1e-8):
            raise self._violation('STOP_NATIVE_COUNTER_OR_CLOCK_NOT_QUIESCENT')
        # Stop is not a new task/safety-success gate. A known native safety
        # exception may leave an unsafe TCP; stopped still means no more owned
        # integration, not a safe or successful task outcome. Only clock/counter
        # quiescence is required here, with no object/teacher/oracle inputs.
        return dict(native_steps=steps,sim_time_s=clock)

    def stop(self,reason):
        """Once only; latch first. Unknown IO is never retried or ACKed."""
        self._latched=True
        if self._stop_attempted:raise SourceStopError('STOP_ALREADY_ATTEMPTED_NO_RETRY_OR_REACK')
        self._stop_attempted=True
        try:
            self._owner()
            if not isinstance(reason,str) or not reason:raise self._violation('STOP_REASON_REQUIRED')
            if not self._lock.acquire(blocking=False):raise self._violation('STOP_WHILE_NATIVE_FRAME_INFLIGHT_NO_ACK')
            try:
                if self._inflight or self._unknown or self._fault:
                    raise SourceStopError('STOP_EFFECTS_OR_NATIVE_FRAME_UNKNOWN_NO_ACK')
                state_sha=self._fingerprint(self._binding())
                if not isinstance(state_sha,str) or re.fullmatch('[0-9a-f]{64}',state_sha) is None:
                    raise self._violation('ORIGINAL_NATIVE_FINGERPRINT_REQUIRED')
                before=self._stopped_readback()
                if self._path.exists() or self._path.is_symlink():
                    raise SourceStopError('PRESERVE_EXISTING_STOP_RECEIPT_NO_REPLAY')
                document=dict(schema=SCHEMA,status='LATCHED_READBACK_RECORD_ACK_REQUIRES_OWNER_RETURN',
                    episode_id=self.episode_id,executor_id=self.executor_id,
                    owner_pid=self._pid,owner_thread_id=self._thread,reason=reason,
                    stopped=True,owned_live_dispatch_latched=True,native_frame_inflight=False,
                    backend='EXPLICIT_SINGLE_OWNER_SYNCHRONOUS_ENGINE_STEP',
                    source_step_sha256=self._source_sha,native_state_sha256=state_sha,last_attempt=self._last_attempt,
                    stop_mechanism='LATCHED_OWNED_LIVE_STEP_WITH_NATIVE_COUNTER_CLOCK_READBACK',
                    **before)
                expected=(json.dumps(document,indent=2,allow_nan=False)+'\n').encode()
                self._write(self._path,document)  # Original fsync+atomic replace+directory fsync.
                actual=self._read(self._path);raw=self._path.read_bytes()
                if actual!=document or raw!=expected:
                    raise SourceStopError('STOP_ORIGINAL_DURABLE_BYTES_READBACK_MISMATCH')
                after=self._stopped_readback()
                if after!=before:raise self._violation('STOP_STATE_CHANGED_DURING_DURABLE_READBACK')
                if self._fingerprint(self._binding())!=state_sha:
                    raise self._violation('STOP_READBACK_CHANGED_ORIGINAL_NATIVE_STATE_OR_RNG')
                receipt_sha=hashlib.sha256(raw).hexdigest()
                receipt=NativeStopReceipt(self.episode_id,self.executor_id,
                    self._path.name+'#sha256='+receipt_sha,True)
                self._acknowledged=True
                return receipt
            finally:self._lock.release()
        except BaseException:
            self._unknown=True
            raise

    def state(self):
        return dict(latched=self._latched,inflight=self._inflight,stop_attempted=self._stop_attempted,
            stop_acknowledged=self._acknowledged,unknown=self._unknown,fault=self._fault,
            last_observed_native_steps=self._observed_steps,last_attempt=self._last_attempt,
            native_qualification='NOT_ASSERTED_BY_THIS_COMPONENT_CLASS')

    # No live-object detach/unlatch method is supplied: restoring its original
    # class while it is still owned would permit a post-ACK owned integration.
    # No global engine class was patched; normal owner teardown drops the live
    # object, and weak registration vanishes. Clones retain original forwarding.
