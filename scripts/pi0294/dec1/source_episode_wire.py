"""Cold source wiring. Existing episode owns every live Engine.step/record_tick.

No admission, scientific runner, ledger writer or native stop implementation.
Owner must bind original authority, durable callbacks and protocol-aware ACK.
Native qualification remains pending.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import re
import numpy as np

from scripts.pi0294.dec1.source_single_pose_prefix import (
    SinglePosePrefixExecutor,PrefixFailure,PREFIX_TICKS,TICKS_PER_REFERENCE,
    PARITY_TOLERANCE,_action,_digest,_proposal_digest,native,
)
from go2arx5_vla.task_data.codec import pose10


@dataclass(frozen=True)
class NativeMappedPrefixAck:
    """Returned only after the owner reads back its actual mapped-execution ACK.

    This type is a binding check, not a producer, validator or native proof by
    itself. The admitted owner must verify the persisted physics/protocol bytes.
    """
    episode_id: str
    executor_id: str
    nonce: str
    raw_h40_sha256: str
    mapped_prefix_sha256: str
    tick_before: int
    tick_after: int
    receipt_id: str
    physics_receipt_sha256: str
    acknowledgement_sha256: str


class OuterLoopSinglePosePrefixExecutor(SinglePosePrefixExecutor):
    """Same one-clone shadow; live is split around the EXISTING outer tick.

    Callback ownership is unchanged: charge('live',...) is durable evidence,
    not a second ledger debit. The existing core live meter remains sole debit.
    native.clone remains behind the existing core engine-start counter wrapper.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.outer=None
        self.maximum_parity=None

    def execute_prefix(self,nonce):
        error=PrefixFailure('OUTER_EPISODE_OWNS_NATIVE_STEPS_USE_TICK_HOOKS')
        self._halt(error)
        raise error

    def _close_live(self,status):
        active=self.outer
        if active is None or active['finished']:return
        active['finished']=True  # A failed/interrupted durable finish is not retried.
        active['finish_confirmed']=False
        try:
            self.finish('live',active['reservation'],self.counts['live']-active['start_count'],
                self.unknown['live'],status)
            active['finish_confirmed']=True
        except BaseException:
            self.unknown['live']=True
            raise

    def _fail(self,error):
        if self.outer is not None and self.outer['inflight'] is not None:
            self.unknown['live']=True
        try:self._halt(error)
        finally:self._close_live('NATIVE_SAFETY_ERROR' if str(error) in (
            'SOURCE_ORIGINAL_NATIVE_S1_REJECT','S1_ACTUAL_LIVE_SHADOW_PARITY_MISMATCH')
            else 'ENGINEERING_MISSING')
        raise error

    def begin_prefix(self,nonce):
        try:
            self._open();pending=self.pending
            if (self.outer is not None or pending is None or nonce!=pending['nonce']
                    or self.engine.steps!=pending['native_step']
                    or self.adapter.state()!=pending['adapter_state']
                    or self.adapter._sequence!=pending['adapter_sequence']):
                raise PrefixFailure('SOURCE_EXACT_PENDING_PREFIX_AND_UNCHANGED_ACTUAL_START_REQUIRED')
            self._event('SOURCE_MAPPED_PREFIX_INSTALL_INTENT',nonce=nonce,
                raw_h40_sha256=hashlib.sha256(pending['raw'].tobytes()).hexdigest(),
                action_sha256=pending['forecast']['action_sha256'],
                proposal_digests=[r['digest'] for r in pending['records']],
                phases=[r['phase'] for r in pending['records']],
                mapped_poses=pending['poses'].tolist(),mapped_grippers=pending['openings'].tolist(),
                native_step_before=self.engine.steps)
            native.bind_install(self.engine,pending['poses'],pending['openings'],pending['forecast'],nonce)
            self._event('SOURCE_MAPPED_PREFIX_BIND_CONFIRMED',nonce=nonce)
            reservation=self._reserve('live',0)
            self.outer=dict(reservation=reservation,start_count=self.counts['live'],
                start_step=self.engine.steps,index=0,inflight=None,proposal=None,finished=False,
                actual_acks=[],initial_adapter_ack_id=self.adapter.state()['predecessor_ack_id'])
            box={'actual':self.engine.snapshot()}
            self.outer['box']=box
            self._bind(self.adapter,box,_digest(pending['chunk']))
            self.adapter.load_chunk(pending['chunk'],self._feedback(box['actual'],self.adapter),
                controller_timestamp=box['actual']['time_s'])
        except BaseException as error:self._fail(error)

    def reference_action(self,reference):
        """Called once per 20 Hz reference, before supervisor action logging."""
        try:
            self._open();a=self.outer
            if (a is None or a['finished'] or a['inflight'] is not None
                    or type(reference) is not int or reference not in range(8)
                    or a['index']!=reference*TICKS_PER_REFERENCE or a['proposal'] is not None
                    or self.engine.steps!=a['start_step']+a['index']):
                raise PrefixFailure('SOURCE_REFERENCE_ORDER_OR_NATIVE_CLOCK_CHANGED')
            expected=self.pending['records'][reference]
            proposal=self._proposal(self.adapter,self.engine,a['box'],self.pending['raw'],self.pending['chunk'].chunk_id)
            target,opening=_action(proposal)
            if (_proposal_digest(proposal)!=expected['digest'] or proposal.phase!=expected['phase']
                    or proposal.command_id!=expected['command_id']
                    or self.adapter.state()['predecessor_ack_id']!=expected['predecessor_ack_id']
                    or not np.array_equal(target,expected['target']) or opening!=expected['opening']):
                raise PrefixFailure('SOURCE_LIVE_PROPOSAL_DIFFERS_FROM_200TICK_SHADOW')
            a['proposal']=proposal
            return target.copy(),opening
        except BaseException as error:self._fail(error)

    def before_native_tick(self,engine,target,opening):
        try:
            self._open();a=self.outer
            if (engine is not self.engine or a is None or a['finished'] or a['inflight'] is not None
                    or a['index']>=PREFIX_TICKS or a['proposal'] is None
                    or engine.steps!=a['start_step']+a['index']):
                raise PrefixFailure('SOURCE_DUPLICATE_SKIPPED_OR_FOREIGN_OUTER_TICK')
            actual_target,actual_opening=_action(a['proposal'])
            if not np.array_equal(target,actual_target) or float(opening)!=actual_opening:
                raise PrefixFailure('SOURCE_OUTER_DISPATCH_ACTION_DIFFERS_FROM_BOUND_PROPOSAL')
            context=dict(reference=a['index']//25,tick=a['index']%25)
            self._event('SOURCE_NATIVE_TICK_INTENT',domain='live',native_step_before=engine.steps,**context)
            a['inflight']=dict(before=engine.steps,context=context)
        except BaseException as error:self._fail(error)

    def after_native_tick(self,engine,native_error=None,record_error=None):
        """Must run even when Engine.step OR original record_tick raises."""
        try:
            a=self.outer
            if engine is not self.engine or a is None or a['finished'] or a['inflight'] is None:
                raise PrefixFailure('SOURCE_AFTER_TICK_WITHOUT_SINGLE_MATCHED_INTENT')
            attempt=a['inflight'];a['inflight']=None
            delta=engine.steps-attempt['before'];known=delta if type(delta) is int and delta>0 else 0
            uncertain=type(delta) is not int or delta!=1
            self.counts['live']+=known
            self.unknown['live']|=uncertain
            # Existing core's outer meter debits actual engine.steps separately.
            try:self.charge('live',known,uncertain,dict(attempt['context'],native_delta=delta))
            except BaseException:
                self.unknown['live']=True;raise
            if uncertain:
                if native_error is not None and not isinstance(native_error,Exception):raise native_error
                raise PrefixFailure('SOURCE_NATIVE_TICK_ACCOUNTING_UNCERTAIN') from native_error
            a['index']+=1
            if record_error is not None:
                self.unknown['live']=True
                raise record_error
            try:
                # The original episode record_tick already did the matching
                # post-step snapshot that shadow _tick performs. Do not repeat.
                observed=np.asarray(native.observed_state(engine))
                expected=self.pending['records'][attempt['context']['reference']]['expected_states'][attempt['context']['tick']]
                if observed.ndim!=1 or not observed.size or not np.isfinite(observed).all():
                    raise PrefixFailure('SOURCE_NONFINITE_OR_MISSING_NATIVE_OBSERVED_STATE')
            except BaseException:
                self.unknown['live']=True;raise
            if engine.safety:raise PrefixFailure('SOURCE_ORIGINAL_NATIVE_S1_REJECT') from native_error
            if native_error is not None:raise native_error
            if observed.shape!=expected.shape:
                raise PrefixFailure('S1_ACTUAL_LIVE_SHADOW_PARITY_MISMATCH')
            difference=float(np.max(np.abs(observed-expected)))
            self.maximum_parity=difference if self.maximum_parity is None else max(self.maximum_parity,difference)
            if difference>PARITY_TOLERANCE:
                raise PrefixFailure('S1_ACTUAL_LIVE_SHADOW_PARITY_MISMATCH')
            if a['index']%25==0:
                record=self.pending['records'][attempt['context']['reference']]
                end=engine.snapshot()['time_s']
                if end!=record['execution_end_timestamp']:
                    raise PrefixFailure('SOURCE_ACTUAL_ACK_CLOCK_DIFFERS_FROM_SHADOW')
                predecessor=self.adapter.state()['predecessor_ack_id']
                actual_proposal=a['proposal']
                target,opening=_action(actual_proposal)
                self.adapter.acknowledge(actual_proposal.command_id,execution_end_timestamp=end)
                actual_ack=dict(reference=attempt['context']['reference'],command_id=actual_proposal.command_id,
                    proposal_sha256=_proposal_digest(actual_proposal),phase=float(actual_proposal.phase),
                    predecessor_ack_id=predecessor,ack_id=self.adapter.state()['predecessor_ack_id'],
                    tick_before=self.engine.steps-25,tick_after=self.engine.steps,
                    execution_end_timestamp=end,target=target.tolist(),opening=opening)
                # These are ACTUAL successful acknowledge results, not copied
                # predicted shadow records. Owner persists the event unchanged.
                self._event('SOURCE_ACTUAL_REFERENCE_ACK',nonce=self.pending['nonce'],**actual_ack)
                a['actual_acks'].append(actual_ack)
                self._event('SOURCE_REFERENCE_ACK',nonce=self.pending['nonce'],reference=attempt['context']['reference'],
                    command_id=a['proposal'].command_id,execution_end_timestamp=end,actual_ticks=25)
                a['proposal']=None
        except BaseException as error:self._fail(error)

    def complete_prefix(self,engine,rows):
        try:
            self._open();a=self.outer
            if (engine is not self.engine or a is None or a['finished'] or a['index']!=PREFIX_TICKS
                    or a['inflight'] is not None or a['proposal'] is not None or len(rows)!=PREFIX_TICKS
                    or len(a['actual_acks'])!=8
                    or engine.steps!=a['start_step']+PREFIX_TICKS
                    or self.counts['live']-a['start_count']!=PREFIX_TICKS):
                raise PrefixFailure('SOURCE_EXACT_PREFIX200_ACK_REQUIRED')
            proof=dict(nonce=self.pending['nonce'],raw_chunk_id=self.pending['chunk'].chunk_id,
                episode_id=self.episode_id,executor_id=self.executor_id,
                raw_h40_sha256=hashlib.sha256(self.pending['raw'].tobytes()).hexdigest(),
                action_sha256=self.pending['forecast']['action_sha256'],
                proposal_digests=[r['digest'] for r in self.pending['records']],
                phases=[r['phase'] for r in self.pending['records']],actual_ticks=PREFIX_TICKS,
                tick_before=a['start_step'],tick_after=engine.steps,
                initial_adapter_ack_id=a['initial_adapter_ack_id'],
                s1=deepcopy(self.pending['forecast']),actual_reference_acks=deepcopy(a['actual_acks']),
                maximum_live_shadow_difference=self.maximum_parity,
                ABI='SINGLE_4x4_POSE_PLUS_GRIPPER_NOT_PREVIEW9')
            self._event('SOURCE_PREFIX200_ACK',**{key:value for key,value in proof.items()
                if key not in ('episode_id','executor_id')})
            self._close_live('PASS')
            self.pending=None;self.outer=None
            return proof
        except BaseException as error:self._fail(error)

    def abort_outer(self,error):
        """Owner calls on episode failure; no hidden physics or retry."""
        if self.outer is not None and self.outer['inflight'] is not None:
            # A terminated callback boundary cannot certify the unreturned step.
            self.unknown['live']=True
        self._fail(error)

    def state(self):
        state=super().state()
        state.update(outer_tick_index=None if self.outer is None else self.outer['index'],
            live_reservation_closed=None if self.outer is None else self.outer.get('finish_confirmed',False),
            live_finish_attempted=None if self.outer is None else self.outer['finished'],
            maximum_live_shadow_difference=self.maximum_parity)
        return state


class SourceInterfaceTeacher:
    """Wrap an existing Teacher/RelayTeacher, never make model or teacher calls twice.

    executor must be a cold owner-supplied OuterLoopSinglePosePrefixExecutor.
    acknowledge_source(engine, rows, proof) is REQUIRED, including for RelayTeacher:
    its old prefix_sha256 identifies raw submitted8 and cannot silently become the
    mapped executed8 identity. Owner supplies its qualified protocol-aware binding.
    persist(status,error,state) preserves existing S1-compatible result + audits.
    This class deliberately has no ledger/model/renderer/engine constructors.
    """
    def __init__(self,teacher,executor,*,acknowledge_source,persist):
        if not callable(acknowledge_source) or not callable(persist):
            raise TypeError('EXISTING_OWNER_ACK_AND_RESULT_PERSISTENCE_CALLBACKS_REQUIRED')
        self.teacher,self.executor=teacher,executor
        self.acknowledge_source,self.persist=acknowledge_source,persist
        self.plan_count=0;self.nonce=None
        self.protocol_ack_unknown=False
        self.finish_unknown=False

    def __getattr__(self,name):return getattr(self.teacher,name)
    def observe(self,state):return self.teacher.observe(state)

    def plan(self,state):
        try:
            self.executor._open()
            original=self.teacher.plan(state)  # Exactly one original fresh query.
            rows=np.stack([pose10(p,g) for p,g in zip(original['poses'],original['opening'])])
            chunk_id=original.get('reference_uuid',self.executor.episode_id+':ref'+str(self.plan_count))
            prepared=self.executor.prepare(rows,chunk_id=chunk_id,observation_timestamp=state['time_s'],
                point_timestamps=state['time_s']+np.asarray(original['offset_s']))
            self.nonce=prepared['nonce'];self.executor.begin_prefix(self.nonce);self.plan_count+=1
            return original  # references.json remains the actual original raw H40.
        except Exception as error:
            self.executor.abort_outer(error)

    def reference_action(self,reference):return self.executor.reference_action(reference)

    def before_native_tick(self,engine,target,opening):
        try:
            if hasattr(self.teacher,'guard_prefix'):self.teacher.guard_prefix()  # Original wire TTL.
            self.executor.before_native_tick(engine,target,opening)
        except BaseException as error:self.executor.abort_outer(error)

    def after_native_tick(self,engine,native_error=None,record_error=None):
        return self.executor.after_native_tick(engine,native_error,record_error)

    def acknowledge(self,engine,rows):
        try:
            proof=self.executor.complete_prefix(engine,rows)
            # No unconditional old RelayTeacher.acknowledge: retain raw/mapped truth.
            self.protocol_ack_unknown=True
            ack=self.acknowledge_source(engine,rows,deepcopy(proof))
            if (not isinstance(ack,NativeMappedPrefixAck)
                    or ack.episode_id!=self.executor.episode_id or ack.executor_id!=self.executor.executor_id
                    or ack.nonce!=proof['nonce'] or ack.raw_h40_sha256!=proof['raw_h40_sha256']
                    or ack.mapped_prefix_sha256!=proof['action_sha256']
                    or type(ack.tick_before) is not int or type(ack.tick_after) is not int
                    or ack.tick_before!=proof['tick_before'] or ack.tick_after!=proof['tick_after']
                    or not isinstance(ack.receipt_id,str) or not ack.receipt_id
                    or any(not isinstance(s,str) or re.fullmatch('[0-9a-f]{64}',s) is None
                        for s in (ack.physics_receipt_sha256,ack.acknowledgement_sha256))):
                raise PrefixFailure('SOURCE_ACTUAL_RAW_MAPPED_PROTOCOL_ACK_BINDING_REQUIRED')
            self.executor._event('SOURCE_PROTOCOL_ACK_READBACK',nonce=ack.nonce,receipt_id=ack.receipt_id,
                acknowledgement_sha256=ack.acknowledgement_sha256,physics_receipt_sha256=ack.physics_receipt_sha256)
            self.protocol_ack_unknown=False
        except BaseException as error:self.executor.abort_outer(error)

    def finish(self,status,error):
        if self.finish_unknown:raise PrefixFailure('SOURCE_FINISH_OUTCOME_UNKNOWN_NO_RETRY')
        finish_error=None;interrupted=None
        try:
            if not self.executor.stopped and (self.executor.outer is not None or self.executor.pending is not None):
                self.executor.abort_outer(PrefixFailure(error or 'SOURCE_EPISODE_ENDED_WITH_PENDING_PREFIX'))
        except BaseException as caught:
            if not isinstance(caught,Exception):
                interrupted=caught;self.finish_unknown=True
            # episode's original finally must continue saving raw evidence. The
            # native failure remains latched; this is never successful STOP proof.
            finish_error=type(caught).__name__+':'+str(caught)
        finally:
            state=self.executor.state();state['finish_error']=finish_error
            state['protocol_ack_unknown']=self.protocol_ack_unknown
            state['finish_unknown']=self.finish_unknown
            effective='FAILED_PRESERVED' if finish_error or status=='VALID_COMPLETE' and (
                state['stopped'] or self.protocol_ack_unknown or any(state['unknown'].values())) else status
            effective_error=error or finish_error or ('SOURCE_UNRESOLVED_EXECUTION_OR_ACK' if effective!=status else None)
            try:self.persist(effective,effective_error,state)
            except BaseException as caught:
                # Persist may have committed before interruption. Keep its
                # outcome unknown and never call it again to obtain a new ACK.
                self.finish_unknown=True;self.executor.unknown['live']=True
                effective='FAILED_PRESERVED';effective_error=type(caught).__name__+':'+str(caught)
                self.executor._halt(caught)
                raise
            finally:
                if hasattr(self.teacher,'finish'):
                    try:self.teacher.finish(effective,effective_error)
                    except BaseException as caught:
                        self.finish_unknown=True;self.executor.unknown['live']=True
                        self.executor._halt(caught)
                        raise
        if interrupted is not None:raise interrupted
