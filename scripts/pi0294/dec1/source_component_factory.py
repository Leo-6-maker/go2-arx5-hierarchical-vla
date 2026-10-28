"""Cold make_teacher/persist seam for the EXISTING pv1.core live owner.

Preparation only: no runner, admission, model, scheduler or physics entry point.
Install only in a new already admitted component child. The original owner
retains its outer live ledger context, engine/clone counting, episode function,
Mesa rendering, safety, original scoring, module restoration and CPU closeout.
"""
from copy import deepcopy
import math
from pathlib import Path

from scripts.pi0292.durable import read, write
from scripts.pi0294.scene_cards import digest
from scripts.pi0294.s1_preflight import VERSION as ORIGINAL_S1_VERSION
from scripts.pi0294.dec1.source_component_ack import (
    SourceComponentAcknowledger, POLICY, PURPOSE, _bytes,
)
from scripts.pi0294.dec1.source_component_evidence import SourceComponentEvidence
from scripts.pi0294.dec1.source_component_meter import SourceComponentMeters, SourceComponentUnknown
from scripts.pi0294.dec1.source_episode_wire import SourceInterfaceTeacher, OuterLoopSinglePosePrefixExecutor
from scripts.pi0294.dec1.source_sync_stop import SourceSyncStopBarrier, SourceStopError
from go2arx5_vla.adapters.execution_progress_binding import NativeStopReceipt
from go2arx5_vla.adapters.execution_progress import ExecutionProgressAdapter, ProgressConfig
from go2arx5_vla.adapters.groot_to_wbc import ReferenceAdapterConfig
from scripts.pi0292.eval_contract import FRAME, TCP


def source_component_adapter(mode, episode_id):
    """Both modes use the original fixed config and its .4s new-chunk limit.

    No request can choose a validator. The original single-pose executor binds
    full raw/mapped H40 and prefix gates before loading or proposing a chunk.
    Until then, all three adapter entries reject.
    """
    def unbound(_):
        raise PermissionError('SOURCE_COMPONENT_NATIVE_PREFIX_GATES_NOT_BOUND')
    config = ProgressConfig()
    return ExecutionProgressAdapter(ReferenceAdapterConfig(FRAME, FRAME, TCP,
        stale_timeout_s=config.new_chunk_max_age_s), config, mode=mode,
        episode_id=episode_id, validate_raw=unbound,
        validate_mapped_full=unbound, validate_prefix=unbound)


class SourceComponentFactory:
    """One-use callable with original make_teacher(card, initial, engine) ABI.

    adapter is the exact idle, fixed owner-selected mode/config used by both
    source executor and component ACK. No stale timeout/config is invented here.
    guard is the ORIGINAL pv1.core physics/CPU guard. ledger is the original
    admitted child ledger. outer_live_meter is its existing steps() meter.
    original_make_teacher is captured BEFORE installing this callable.
    expected_step/snapshot/fingerprint are exact frozen native source primitives.
    Actual native engine creation/clone counting stays outside this class.
    """
    def __init__(self, *, folder, ledger, outer_live_meter, original_make_teacher,
                 adapter, guard, plan_bytes, expected_plan_sha256, population_bytes,
                 plan_owner_release, runtime_release, condition_id, episode_id,
                 executor_id, source_step_sha256, expected_step, expected_snapshot,
                 fingerprint, event_maximum_bytes):
        if not all(callable(f) for f in (original_make_teacher, guard, expected_step, expected_snapshot, fingerprint)):
            raise TypeError('ORIGINAL_OWNER_FACTORY_GUARD_AND_SOURCE_PRIMITIVES_REQUIRED')
        self.folder = Path(folder)
        self.ledger, self.outer = ledger, outer_live_meter
        self.original, self.adapter, self.guard = original_make_teacher, adapter, guard
        self.episode_id, self.executor_id = episode_id, executor_id
        self.ack_options = dict(plan_bytes=plan_bytes, expected_plan_sha256=expected_plan_sha256,
            population_bytes=population_bytes, owner_release=plan_owner_release,
            runtime_release=runtime_release, condition_id=condition_id, episode_id=episode_id,
            executor_id=executor_id, adapter=adapter, guard=guard)
        self.stop_options = dict(source_step_sha256=source_step_sha256,
            expected_step=expected_step, expected_snapshot=expected_snapshot, fingerprint=fingerprint)
        self.maximum = event_maximum_bytes
        self.engine = self.evidence = self.meters = self.barrier = self.ack = self.executor = self.interface = None
        self.called = self.persist_attempted = self.persist_confirmed = self.closed = False
        self.unknown = False
        self.actual_stop = None
        self.stop_return_unknown = False
        self.factory_error = None

    def _mark_unknown(self):
        self.unknown = True
        self.outer['component_unknown'] = True

    def record(self, event):
        # Saving actual effects/STOP must retain the original CLOSEOUT guard;
        # it must not require permission to start another native integration.
        # Original physics guard still runs before every proposed/live/shadow
        # integration and at component ACK/Teacher plan entry.
        try:
            self.ledger.guard(closeout=True)
            if self.evidence is None:
                raise SourceComponentUnknown('SOURCE_EVENT_SINK_NOT_CONSTRUCTED')
            return self.evidence.record(event)
        except BaseException:
            self._mark_unknown()
            raise

    def stop_once(self, reason):
        # Latch/readback the real engine even if the physics guard just expired.
        # A prior attempted stop is never repeated and never served from cache.
        if self.barrier is None or self.barrier.state()['stop_attempted'] or self.stop_return_unknown:
            self._mark_unknown()
            raise SourceStopError('ACTUAL_COMPONENT_STOP_ALREADY_ATTEMPTED_OR_UNKNOWN_NO_REACK')
        try:
            receipt = self.barrier.stop(reason)
            if (not isinstance(receipt, NativeStopReceipt) or receipt.episode_id != self.episode_id
                    or receipt.executor_id != self.executor_id or receipt.stopped is not True
                    or not receipt.receipt_id):
                raise SourceStopError('ACTUAL_COMPONENT_STOP_RETURN_BINDING_REQUIRED')
            self.actual_stop = receipt
            return receipt
        except BaseException:
            self.stop_return_unknown = True
            self._mark_unknown()
            raise

    def _owner_stop_event(self, kind, reason):
        receipt = self.stop_once(reason)
        self.record(dict(kind=kind, episode_id=self.episode_id, executor_id=self.executor_id,
            receipt_id=receipt.receipt_id, reason=reason, actual_native_steps=self.engine.steps))
        return receipt

    def __call__(self, card, initial, engine):
        if self.called:
            self._mark_unknown()
            raise SourceComponentUnknown('ONE_COMPONENT_FACTORY_CALL_NO_RETRY')
        self.called = True
        self.engine = engine  # Actual engine retained even if later initialization fails.
        try:
            self.barrier = SourceSyncStopBarrier(engine, episode_id=self.episode_id,
                executor_id=self.executor_id, receipt_path=self.folder / 'SOURCE_COMPONENT_STOP.json',
                **self.stop_options)
            self.guard()
            if 'relay_request_relative' in card or digest(card) != digest(engine.card):
                raise ValueError('ORIGINAL_SOURCE_COMPONENT_CARD_WITHOUT_NEURAL_RELAY_REQUIRED')
            self.evidence = SourceComponentEvidence(self.folder / 'SOURCE_COMPONENT_EVENTS.jsonl',
                episode_id=self.episode_id, executor_id=self.executor_id, maximum_bytes=self.maximum)
            self.meters = SourceComponentMeters(self.ledger, episode_id=self.episode_id,
                outer_live_meter=self.outer, record=self.record)
            ack_directory = self.folder / 'source_component_ack'
            ack_directory.mkdir()
            self.ack = SourceComponentAcknowledger(engine, ack_directory,
                read_events=self.evidence.read_events, **self.ack_options)
            original_teacher = self.original(card, initial, engine)  # ONE original Teacher construction.
            bound_teacher = self.ack.wrap_teacher(original_teacher)
            self.executor = OuterLoopSinglePosePrefixExecutor(engine, self.adapter,
                executor_id=self.executor_id, guard=self.guard, reserve=self.meters.reserve,
                charge=self.meters.charge, finish=self.meters.finish, record=self.record,
                stop=self.stop_once)
            self.interface = SourceInterfaceTeacher(bound_teacher, self.executor,
                acknowledge_source=self.ack.acknowledge, persist=self.persist)
            return self.interface
        except BaseException as error:
            self.factory_error = type(error).__name__ + ':' + str(error)
            self._mark_unknown()  # Failed construction is never a successful component.
            if self.barrier is not None and not self.barrier.state()['stop_attempted']:
                try:
                    self._owner_stop_event('SOURCE_FACTORY_STOP_ACK', 'SOURCE_COMPONENT_FACTORY_FAILURE')
                except BaseException as cleanup_error:
                    if not isinstance(error, Exception):
                        if error.__cause__ is None:raise error from cleanup_error
                        raise error
                    raise
            raise

    def _terminal_stop(self, status):
        if self.barrier is None:
            self._mark_unknown()
            raise SourceComponentUnknown('SOURCE_LIVE_BARRIER_NOT_CONSTRUCTED')
        state = self.barrier.state()
        if not state['stop_attempted']:
            normal = (status == 'VALID_COMPLETE' and self.executor is not None
                and self.executor.outer is None and self.executor.pending is None
                and type(self.engine.steps) is int and self.engine.steps == 16000
                and math.isfinite(float(self.engine.data.time))
                and abs(float(self.engine.data.time) - 32.) <= 1e-8)
            if normal:
                self._owner_stop_event('SOURCE_NORMAL_STOP_ACK', 'SOURCE_COMPONENT_NORMAL_32S_COMPLETE')
            else:
                if status == 'VALID_COMPLETE':
                    self._mark_unknown()
                self._owner_stop_event('SOURCE_TERMINAL_STOP_ACK', 'SOURCE_COMPONENT_TERMINAL:' + status)
        # Existing executor failure STOP is checked, never called a second time.
        state = self.barrier.state()
        if (self.actual_stop is None or not state['stop_acknowledged'] or state['unknown']
                or self.stop_return_unknown):
            self._mark_unknown()
            raise SourceComponentUnknown('ACTUAL_RETURNED_COMPONENT_STOP_ACK_REQUIRED')

    def _write_result_once(self, path, document):
        self.ledger.guard(closeout=True)
        if path.exists() or path.is_symlink():
            raise FileExistsError('PRESERVE_FIRST_COMPONENT_S1_OUTCOME_NO_RETRY')
        write(path, document)
        if read(path) != document or path.read_bytes() != _bytes(document):
            raise SourceComponentUnknown('COMPLETE_COMPONENT_S1_DURABLE_READBACK_REQUIRED')

    def persist(self, status, error, state):
        """SourceInterfaceTeacher.finish callback; exactly once, before close()."""
        if self.persist_attempted:
            self._mark_unknown()
            raise SourceComponentUnknown('COMPONENT_S1_PERSIST_ALREADY_ATTEMPTED_NO_RETRY')
        self.persist_attempted = True
        try:
            stop_error = event_error = interrupted = None
            try:
                self._terminal_stop(status)
            except BaseException as caught:
                self._mark_unknown()
                stop_error = type(caught).__name__ + ':' + str(caught)
                if not isinstance(caught, Exception):
                    interrupted = caught
            try:
                self.ledger.guard(closeout=True)
                if self.evidence is None:
                    raise SourceComponentUnknown('COMPONENT_NATIVE_EVENTS_UNAVAILABLE')
                events = self.evidence.read_events()
            except BaseException as caught:
                self._mark_unknown()
                event_error = type(caught).__name__ + ':' + str(caught)
                events = []  # Explicit unavailable evidence, never fabricated ACKs.
                if not isinstance(caught, Exception) and interrupted is None:
                    interrupted = caught
            stopping = [e for e in events if e.get('kind') in (
                'SOURCE_STOP_ACK', 'SOURCE_NORMAL_STOP_ACK', 'SOURCE_TERMINAL_STOP_ACK', 'SOURCE_FACTORY_STOP_ACK')]
            if (self.actual_stop is None or len(stopping) != 1
                    or stopping[0].get('receipt_id') != self.actual_stop.receipt_id
                    or stopping[0].get('episode_id') != self.episode_id
                    or stopping[0].get('executor_id') != self.executor_id):
                self._mark_unknown()
                stop_error = stop_error or 'ACTUAL_DURABLE_COMPONENT_TERMINAL_STOP_ACK_REQUIRED'
            self._propagate_unknown(state)
            finished = [e for e in events if e.get('kind') == 'COMPONENT_PREFIX_FINISHED']
            s1_rejects = [e for e in finished if e['domain'] == 'shadow' and e['status'] == 'S1_REJECT'
                and e['reservation_retained'] is False]
            gates = [e for e in events if e.get('kind') == 'SOURCE_FULL_H40_GATE']
            rejects = [e for e in gates if isinstance(e.get('decision'), dict)
                and e['decision'].get('decision') == 'REJECT']
            raw_ids = {self.episode_id + ':ref' + str(i) for i in range(80)}
            raw_rejects = [e for e in rejects if e.get('reference_id') in raw_ids]
            mapped_rejects = [e for e in rejects if e.get('reference_id') not in raw_ids]
            complete = [e for e in events if e.get('kind') == 'SOURCE_PREFIX200_ACK']
            known = {kind:sum(v['known'] for v in self.meters.entries.values() if v['kind'] == kind)
                for kind in ('shadow', 'live')} if self.meters is not None else dict(shadow=0, live=0)
            maximum = None if self.executor is None else self.executor.maximum_parity
            if maximum is not None and (not math.isfinite(maximum) or maximum < 0):
                self._mark_unknown()
            # Preserve old summary keys; never let an unknown closeout become a
            # scientific pre-execution rejection through the legacy classifier.
            result = dict(version=ORIGINAL_S1_VERSION, execution_path='DEC1_SOURCE_SINGLE_POSE_MAPPED_PREFIX8',
                status='FAILED_PRESERVED' if self.unknown else status, error=error,
                policy=POLICY, purpose=PURPOSE, interface_mode=self.adapter.mode,
                episode_id=self.episode_id, executor_id=self.executor_id,
                component_plan_sha256=self.ack_options['expected_plan_sha256'],
                plan_owner_release=self.ack_options['owner_release'],
                runtime_release=self.ack_options['runtime_release'], engineering_unknown=self.unknown,
                shadow_invocations=0 if self.executor is None else self.executor.clone_attempts,
                shadow_integrations=known['shadow'], live_integrations_lower_bound=known['live'],
                native_live_integrations=None if self.engine is None else self.engine.steps,
                pre_execution_rejections=0 if self.unknown else len(s1_rejects),
                known_partial_S1_rejections=len(s1_rejects), full_raw_gate_rejections=len(raw_rejects),
                mapped_full_gate_rejections=len(mapped_rejects), full_H40_gate_results=gates,
                live_native_failures=None if self.engine is None else int(bool(self.engine.safety)),
                maximum_live_shadow_difference=maximum,
                actual_completed_prefixes=complete, actual_25tick_ACKs=[e for e in events
                    if e.get('kind') == 'SOURCE_ACTUAL_REFERENCE_ACK'],
                shadow_and_live_finish_records=finished, execution_state=deepcopy(state),
                component_meter_unknown=self.meters is None or self.meters.unknown,
                component_ack_state=None if self.ack is None else self.ack.state(),
                events_readback_valid=event_error is None, events_readback_error=event_error,
                terminal_stop_error=stop_error,
                actual_stop_event=stopping[0] if len(stopping) == 1 else None,
                actual_stop_state=None if self.barrier is None else self.barrier.state(),
                contexts=[], negatives=None, not_formal_training_data=True,
                privileged_simulation_state=True, raw_actions_modified=False, actions_modified=True,
                student_neural_ACK_qualification=False, selected='NONE', deployment='NONE')
            if rejects and not self.unknown:
                self._write_result_once(self.folder / 'MODEL_REJECT.json', dict(policy=POLICY,
                    purpose=PURPOSE, original_gate=rejects[0]['decision'],
                    source_gate_scope='FULL_RAW_H40' if rejects[0] in raw_rejects else 'MAPPED_FULL_H40',
                    reference_id=rejects[0]['reference_id'], student_neural_ACK_qualification=False))
            self._write_result_once(self.folder / 'S1_RESULT.json', result)
            self.persist_confirmed = True
            if interrupted is not None:
                raise interrupted
        except BaseException as caught:
            self._mark_unknown()
            if interrupted is not None and caught is not interrupted:
                if interrupted.__cause__ is None:raise interrupted from caught
                raise interrupted
            raise

    def _propagate_unknown(self, state=None):
        state = state if state is not None else ({} if self.executor is None else self.executor.state())
        if (self.unknown or self.factory_error or self.evidence is None or self.evidence.unknown
                or self.meters is None or self.meters.unknown or self.ack is None or self.ack.state()['unknown']
                or any(state.get('unknown', {}).values()) or state.get('protocol_ack_unknown')
                or state.get('finish_unknown') or state.get('finish_error')
                or self.interface is not None and (self.interface.protocol_ack_unknown or self.interface.finish_unknown)
                or self.barrier is None or self.barrier.state()['unknown'] or self.stop_return_unknown):
            self._mark_unknown()

    def persist_if_unfinished(self, status, error):
        """Original owner finally fallback when teacher was not returned.

        Never retries an attempted write or STOP. Do not use this as a second
        successful S1_RESULT after an uncertain first persist.
        """
        if not self.persist_attempted:
            self.persist(status, error, {} if self.executor is None else self.executor.state())

    def close_evidence(self):
        if self.closed:
            self._mark_unknown()
            raise SourceComponentUnknown('COMPONENT_EVIDENCE_CLOSE_ALREADY_ATTEMPTED')
        self.closed = True
        try:
            if self.evidence is not None:
                self.evidence.close()
        except BaseException:
            self._mark_unknown()
            raise

    def reconcile_outer(self):
        """Inside original live context, AFTER persist + evidence close attempts."""
        self._propagate_unknown()
        if not self.persist_confirmed or not self.closed or self.engine is None or self.meters is None:
            self._mark_unknown()
        steps = None if self.engine is None else self.engine.steps
        if type(steps) is int and 0 <= steps <= 16000:
            self.outer['actual'] = steps  # Preserve actual lower bound even on unknown closeout.
        else:
            self._mark_unknown()
        if self.meters is None:
            raise SourceComponentUnknown('KEEP_ORIGINAL_LIVE_BOUND_WITHOUT_COMPONENT_CLOSEOUT')
        # Meter propagates component_unknown into the existing Ledger.steps
        # exception path. A known partial scientific rejection exits normally.
        actual=self.meters.reconcile_outer(steps)
        self._outer_reconciled=True
        return actual

    def terminal_anchor(self,*,manifest_sha256):
        from .source_component_anchor import capture_terminal_anchor
        return capture_terminal_anchor(self,manifest_sha256=manifest_sha256)
