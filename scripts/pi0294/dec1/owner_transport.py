"""Cold-owner RPC journal; uncertain writes never enter read-only retry paths.

This module has no SSH, scheduler, model or automatic retry implementation.
The registered owner injects a bounded transport and exact native readback.
It does not grant a run, renew a lease, or change the 300-second watchdog.
"""
from __future__ import annotations

import math
from pathlib import Path
import re
import subprocess
import time
import uuid

from scripts.pi0292.durable import OwnerLock, read, write

TIMEOUTS = {'READ_ONLY': 40, 'HEARTBEAT': 30, 'MUTATION': 40}
IDENTITY_KEYS = {'run_uuid', 'job_id', 'source_sha', 'request_sha256'}


class ReadOnlyFailure(ConnectionError):
    """Only this error may enter the owner's bounded observation retry path."""


class UnknownMutation(RuntimeError):
    """Latched across processes; resolve by exact readback, never replay."""


class JournalFailure(RuntimeError):
    """Control persistence failed; stop, even when the native call was a read."""


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _identity(value):
    if set(value) != IDENTITY_KEYS or not all(isinstance(v, str) and v for v in value.values()):
        raise ValueError('EXACT_REGISTERED_RUN_JOB_SOURCE_REQUEST_REQUIRED')
    if not all(re.fullmatch('[0-9a-f]{'+str(n)+'}', value[k]) for k, n in
               [('source_sha', 40), ('request_sha256', 64)]):
        raise ValueError('EXACT_SOURCE_AND_REQUEST_DIGESTS')
    return dict(value)


def checked_heartbeat(value, intent, *, now):
    """A readback must join the exact publication and retain native guards."""
    if not _finite(now) or not isinstance(value, dict):
        raise ValueError('FINITE_POST_READ_CLOCK_AND_HEARTBEAT_RECEIPT')
    if value.get('schema')=='V5_CD_ACTUAL_BOUND_NATIVE_HEARTBEAT_RECEIPT_V1':
        from scripts.pi0294.pv1.mix_owner import validate_heartbeat_receipt
        return validate_heartbeat_receipt(value,intent,now=now)
    if value.get('schema')=='GEN1_ACTUAL_BOUND_NATIVE_HEARTBEAT_RECEIPT_V1':
        from scripts.pi0294.generalization.amd_training_owner import validate_heartbeat_receipt
        return validate_heartbeat_receipt(value,intent,now=now)
    if value.get('schema')=='ARX5_MINIMAL_ACTUAL_BOUND_NATIVE_HEARTBEAT_RECEIPT_V1':
        from scripts.pi0294.generalization.minimal_native_owner import validate_heartbeat_receipt
        return validate_heartbeat_receipt(value,intent,now=now)
    beat = value.get('heartbeat', {})
    view_status=value.get('campaign_view_status')
    if view_status=='PASS_VANDA_ONLY_CAMPAIGN_VIEW':
        from .vanda_campaign import validate_heartbeat_receipt
        validate_heartbeat_receipt(value,intent,now_epoch=now)
    if (value.get('lease_status') != 'ACTIVE' or value.get('run_STOP') is not False
            or value.get('parent_STOP') is not False
            or view_status not in ('PASS_TWO_PLATFORM_CAMPAIGN_VIEW','PASS_VANDA_ONLY_CAMPAIGN_VIEW')
            or value.get('unknown_effects') != [] or value.get('pending_updates') != []
            or beat.get('native_job_state') not in ('Q', 'R')
            or beat.get('rpc_intent_id') != intent['operation_id']
            or any(beat.get(k) != v for k, v in intent['identity'].items())):
        raise PermissionError('EXACT_HEARTBEAT_IDENTITY_AND_ORIGINAL_NATIVE_GUARDS')
    observed = beat.get('observed_epoch')
    view_epoch = value.get('campaign_view_observed_epoch')
    if (not _finite(observed) or not _finite(view_epoch)
            or not intent['started_epoch'] <= observed <= now
            or not 0 <= now - observed <= 300
            or not 0 <= now - view_epoch < 120):
        raise PermissionError('ORIGINAL_HEARTBEAT_AND_CAMPAIGN_VIEW_FRESHNESS')
    return value


class Journal:
    """One cold registered owner's durable phase records and unresolved latch.

    ``invoke(intent, timeout_s)`` MUST enforce its supplied timeout; callers
    cannot substitute unbounded legacy RPCs. The elapsed check also rejects
    late success. ``readback`` is purely read-only and receives the same intent.
    The owner retains its outer allocation lock; this lock prevents concurrent
    journal callers from publishing a second heartbeat during reconciliation.
    """
    def __init__(self, root, identity, *, clock=time.time, monotonic=time.monotonic):
        self.root = Path(root)
        if not self.root.is_dir() or self.root.is_symlink():
            raise ValueError('EXISTING_REGISTERED_CONTROL_JOURNAL_DIRECTORY')
        self.identity = _identity(identity)
        self.clock, self.monotonic = clock, monotonic
        self.state_path = self.root / 'RPC_EFFECT_STATE.json'
        if self.state_path.is_symlink():
            raise ValueError('NO_JOURNAL_SYMLINK')

    def _save(self, path, value):
        try:
            write(path, value)
        except BaseException as error:
            raise JournalFailure('DURABLE_CONTROL_JOURNAL_FAILED') from error

    def _state(self):
        if not self.state_path.exists():
            return None
        state = read(self.state_path)
        if state['intent']['identity'] != self.identity:
            raise PermissionError('JOURNAL_BELONGS_TO_OTHER_RUN')
        return state

    def _event(self, intent, event, **values):
        path = self.root / (intent['operation_id'] + '-' + event + '.json')
        if path.exists():
            raise JournalFailure('PRESERVE_EXISTING_RPC_EVENT')
        self._save(path, dict(intent=intent, event=event, observed_epoch=self.clock(), **values))

    def call(self, phase, effect, invoke, *, readback=None):
        if (effect not in TIMEOUTS or not isinstance(phase, str)
                or re.fullmatch('[A-Z][A-Z0-9_]{0,63}', phase) is None
                or readback is not None and effect != 'HEARTBEAT'):
            raise ValueError('EXPLICIT_BOUNDED_PHASE_AND_EFFECT_CLASS')
        lock = OwnerLock(self.root / 'RPC_OWNER.lock')
        try:
            state = self._state()
            if effect != 'READ_ONLY' and state and state['status'] != 'CONFIRMED':
                raise UnknownMutation('UNRESOLVED_PRIOR_WRITE_NO_NEW_PUBLICATION')
            now = self.clock()
            if not _finite(now):
                raise ValueError('FINITE_REAL_CONTROL_CLOCK')
            intent = dict(operation_id=uuid.uuid4().hex, identity=self.identity,
                          phase=phase, effect=effect, started_epoch=now,
                          timeout_s=TIMEOUTS[effect], automatic_write_retry=False)
            # Latch before dispatch; a crash even before INTENT is conservative.
            if effect != 'READ_ONLY':
                self._save(self.state_path, dict(status='PENDING', intent=intent))
            self._event(intent, 'INTENT')
            began = self.monotonic()
            try:
                result = invoke(intent, TIMEOUTS[effect])
                elapsed = self.monotonic() - began
                if not 0 <= elapsed <= TIMEOUTS[effect]:
                    raise TimeoutError('CONTROL_CALL_EXCEEDED_BOUND')
                if effect == 'HEARTBEAT':
                    checked_heartbeat(result, intent, now=self.clock())
            except BaseException as error:
                self._event(intent, 'ERROR', error_type=type(error).__name__,
                            elapsed_s=self.monotonic() - began,
                            outcome='READ_FAILED' if effect == 'READ_ONLY' else 'UNKNOWN_WRITE')
                if effect == 'READ_ONLY':
                    if not isinstance(error, (TimeoutError, ConnectionError, subprocess.TimeoutExpired)):
                        raise
                    raise ReadOnlyFailure('READ_ONLY_'+phase+'_FAILED') from error
                self._save(self.state_path, dict(status='UNKNOWN', intent=intent))
                # Interruptions preserve the latch and never trigger extra I/O.
                if not isinstance(error, Exception):
                    raise
                if effect == 'HEARTBEAT' and readback is not None:
                    return self._resolve(intent, readback)
                raise UnknownMutation('UNKNOWN_'+phase+'_NO_AUTOMATIC_RETRY') from error
            self._event(intent, 'CONFIRMED', elapsed_s=elapsed)
            if effect != 'READ_ONLY':
                self._save(self.state_path, dict(status='CONFIRMED', intent=intent))
            return result
        finally:
            lock.close()

    def _resolve(self, intent, readback):
        attempt = uuid.uuid4().hex
        path = self.root / (intent['operation_id'] + '-READBACK-' + attempt + '.json')
        began = self.monotonic()
        try:
            value = readback(intent, TIMEOUTS['READ_ONLY'])
            elapsed = self.monotonic() - began
            if not 0 <= elapsed <= TIMEOUTS['READ_ONLY']:
                raise TimeoutError('READBACK_EXCEEDED_BOUND')
            checked_heartbeat(value, intent, now=self.clock())
        except BaseException as error:
            self._save(path, dict(intent=intent, status='UNKNOWN_PRESERVED',
                                 error_type=type(error).__name__, elapsed_s=self.monotonic()-began))
            if not isinstance(error, Exception):
                raise
            raise UnknownMutation('HEARTBEAT_READBACK_NOT_EXACT_NO_REPLAY') from error
        self._save(path, dict(intent=intent, status='CONFIRMED_BY_READ_ONLY_READBACK',
                             heartbeat=value['heartbeat'], elapsed_s=elapsed))
        self._save(self.state_path, dict(status='CONFIRMED', intent=intent))
        return value

    def reconcile_heartbeat(self, readback):
        """Resume only readback after owner recovery; never invoke the write."""
        lock = OwnerLock(self.root / 'RPC_OWNER.lock')
        try:
            state = self._state()
            if not state or state['status'] not in ('PENDING', 'UNKNOWN'):
                raise ValueError('NO_UNRESOLVED_HEARTBEAT')
            if state['intent']['effect'] != 'HEARTBEAT':
                raise UnknownMutation('OTHER_WRITE_REQUIRES_ITS_OWN_EXACT_RECONCILIATION')
            return self._resolve(state['intent'], readback)
        finally:
            lock.close()
