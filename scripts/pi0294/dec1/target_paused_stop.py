"""Cold target quota/STOP IO for the existing paused native task worker.

This is an owned file boundary, not a controller, runner, prefix predictor or
admission. A new admitted owner must route ALL its quota writes through this
object. Original raw/mapped/native safety and install gates still precede each
quota call. Never attach it to an already escaped/running worker. No old R4
lease, pilot card, cutoff, or ledger is imported or reopened here.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import threading

from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0294.task_state import parse_state
from go2arx5_vla.adapters.execution_progress_binding import NativeStopReceipt

FROZEN_CLOCK = '82004c3171cc8cd2a0486f391595ac39424bad96882d12c5936c4eca55fe3b08'


class TargetStopError(PermissionError):
    pass


def need(ok, why):
    if not ok:
        raise TargetStopError('DEC1_TARGET_STOP_' + why)


def pause_document(raw):
    def pairs(rows):
        result = {}
        for k, v in rows:
            need(k not in result, 'DUPLICATE_NATIVE_PAUSE_FIELD')
            result[k] = v
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: need(False, 'NONFINITE_NATIVE_PAUSE_FIELD'))


def frozen_clock(path):
    """Load the actual frozen member, never silently use management HEAD."""
    path = Path(path)
    need(not path.is_symlink() and path.is_file(), 'EXPLICIT_FROZEN_CLOCK_MEMBER')
    raw = path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == FROZEN_CLOCK, 'FROZEN_CLOCK_BYTES_CHANGED')
    # This exact hash is a known, pure historical Python member. Executing its
    # definitions does not construct a model, import native libraries or run it.
    namespace = {'__name__': 'dec1_exact_frozen_task_clock'}
    exec(compile(raw, str(path), 'exec'), namespace)
    return namespace


class OwnedTargetQuota:
    """One new owner, one thread, no reopen and no quota after STOP/unknown.

    guard(closeout=...) is the CURRENT admitted owner's original resource and
    identity guard. spin is that worker's original bounded ROS/process-health
    loop. STOP only acknowledges an already reached native quota pause; a
    running/unknown boundary is left UNKNOWN for the original process teardown.
    It never claims to interrupt an in-flight native integration synchronously.
    """
    def __init__(self, folder, *, clock_path, episode_id, executor_id,
                 run_uuid, manifest_sha256, guard, spin, maximum_native_step):
        self.clock = frozen_clock(clock_path)
        need(all(isinstance(v, str) and v for v in (episode_id, executor_id, run_uuid)), 'EXPLICIT_OWNER_IDENTITIES')
        need(isinstance(manifest_sha256, str) and len(manifest_sha256) == 64
             and all(c in '0123456789abcdef' for c in manifest_sha256), 'MANIFEST_HASH')
        need(callable(guard) and callable(spin), 'EXISTING_OWNER_GUARD_AND_SPIN')
        need(type(maximum_native_step) is int and 0 < maximum_native_step <= 20000, 'ORIGINAL_QUOTA_CAP')
        self.root = Path(folder)
        need(self.root.is_dir() and not self.root.is_symlink(), 'REGISTERED_OUTPUT_DIRECTORY')
        self.guard, self.spin = guard, spin
        self.episode_id, self.executor_id = episode_id, executor_id
        self.identity = dict(run_uuid=run_uuid, manifest_sha256=manifest_sha256,
                             episode_id=episode_id, executor_id=executor_id)
        self.pid, self.thread = os.getpid(), threading.get_ident()
        self.maximum, self.last, self.sequence = maximum_native_step, None, 0
        self.latched = self.unknown = self.stop_attempted = self.acknowledged = False
        self.lock = None
        guard(closeout=False)
        for name in ('TARGET_QUOTA_OWNER.json', 'TARGET_QUOTA_OWNER.lock', 'TARGET_STOP_INTENT.json',
                     'TARGET_STOP_ACK.json', 'step_quota', 'state.bin', 'pause_ack.json'):
            need(not (self.root / name).exists() and not (self.root / name).is_symlink(), 'NEW_UNESCAPED_OWNER_ONLY')
        self.lock = OwnerLock(self.root / 'TARGET_QUOTA_OWNER.lock')
        try:
            need(not (self.root / 'TARGET_QUOTA_OWNER.json').exists(), 'NO_SECOND_OWNER')
            write(self.root / 'TARGET_QUOTA_OWNER.json', dict(self.identity,
                pid=self.pid, thread=self.thread, task_clock_sha256=FROZEN_CLOCK,
                maximum_native_step=self.maximum, native_qualification=False))
        except BaseException:
            self.latched = self.unknown = True
            self.lock.close()
            raise

    def _owner(self):
        need(os.getpid() == self.pid and threading.get_ident() == self.thread, 'SAME_OWNER_PROCESS_AND_THREAD')

    def _bytes(self, name, limit):
        path = self.root / name
        need(not path.is_symlink() and path.is_file() and path.stat().st_size <= limit, 'BOUNDED_NATIVE_FILE_' + name)
        raw = path.read_bytes()
        need(len(raw) <= limit, 'BOUNDED_NATIVE_READ_' + name)
        return raw

    def quota(self, step):
        """Original worker quota write; this method supplies NO safety verdict."""
        need(not self.latched and not self.unknown, 'LATCHED_NO_MORE_QUOTA')
        try:
            self._owner()
            self.guard(closeout=False)
            need(type(step) is int and 0 < step <= self.maximum
                 and (self.last is None or step > self.last), 'ORIGINAL_MONOTONIC_NATIVE_QUOTA')
            intent = self.root / ('TARGET_QUOTA_%06d_INTENT.json' % self.sequence)
            need(not intent.exists(), 'NO_QUOTA_INTENT_REPLAY')
            write(intent, dict(self.identity, native_step=step, no_automatic_retry=True))
            path = self.root / 'step_quota'
            temporary = self.root / 'step_quota.pending'
            with temporary.open('xb') as handle:
                handle.write((str(step) + '\n').encode())
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            need(self._bytes('step_quota', 32) == (str(step) + '\n').encode(), 'QUOTA_RETURN_READBACK')
            self.last = step
            self.sequence += 1
        except BaseException:
            self.latched = self.unknown = True
            raise

    def _boundary(self):
        raw = self._bytes('state.bin', 2 * 1024**2)
        packet = parse_state(raw)
        ack = pause_document(self._bytes('pause_ack.json', 128 * 1024))
        step = packet['step']
        need(type(step) is int and step == self.last and type(packet['time_s']) in (int, float)
             and math.isfinite(packet['time_s']) and abs(packet['time_s'] - step * .002) <= 1e-8,
             'ACTUAL_LAST_QUOTA_STATE_AND_CLOCK')
        need(type(ack['step']) is int and ack['step'] == step
             and ack['clock_paused'] is True and ack['running'] is False
             and type(ack['ack_sequence']) is int and ack['ack_sequence'] >= 0,
             'ACTUAL_NATIVE_QUOTA_PAUSE_NOT_RUNNING')
        need(type(ack['time_s']) in (int, float) and math.isfinite(ack['time_s'])
             and abs(ack['time_s'] - step * .002) <= 1e-8
             and all(type(ack[k]) is int and ack[k] >= 0 for k in ('motiontime', 'leg_history_steps'))
             and type(ack['gait_indices']) in (int, float) and math.isfinite(ack['gait_indices']),
             'ORIGINAL_FINITE_NATIVE_PAUSE_COUNTERS')
        need(self._bytes('step_quota', 32) == (str(step) + '\n').encode(), 'OWNED_QUOTA_UNCHANGED')
        return raw, ack, step

    def stop(self, reason):
        """Latch first; require fresh native pause ACK and unchanged FULL state.

        No mj_forward, cancel/reference publication, integration, quota increase
        or process restart occurs. Unsafe task state may still be stopped; this
        is deliberately not a successful-task or future-prefix safety receipt.
        Any failed/lost IO stays latched UNKNOWN and cannot be called again.
        """
        self.latched = True
        need(not self.stop_attempted, 'STOP_ALREADY_ATTEMPTED_NO_REACK')
        self.stop_attempted = True
        try:
            self._owner()
            need(not self.unknown, 'PRIOR_EFFECT_UNKNOWN')
            need(isinstance(reason, str) and bool(reason), 'REASON_REQUIRED')
            self.guard(closeout=True)
            before_raw, before, step = self._boundary()
            write(self.root / 'TARGET_STOP_INTENT.json', dict(self.identity,
                reason=reason, native_step=step, state_sha256=hashlib.sha256(before_raw).hexdigest(),
                task_clock_sha256=FROZEN_CLOCK, no_automatic_retry=True))
            need(self.spin(2, lambda: pause_document(self._bytes('pause_ack.json', 128 * 1024))['ack_sequence'] > before['ack_sequence']),
                 'FRESH_NATIVE_PAUSE_ACK_MISSING')
            after_raw, after, after_step = self._boundary()
            self.clock['require_pause'](before, after, step)
            need(after_step == step and after_raw == before_raw, 'FULL_NATIVE_STATE_CHANGED_WHILE_STOPPING')
            document = dict(self.identity, schema='DEC1_TARGET_OWNED_QUOTA_STOP_ACK_V1', reason=reason,
                native_step=step, sim_time_s=step * .002, task_clock_sha256=FROZEN_CLOCK,
                native_state_sha256=hashlib.sha256(after_raw).hexdigest(),
                pause_before=before, pause_after=after,
                mechanism='PERMANENT_OWNER_LATCH_AT_FROZEN_NATIVE_QUOTA_PAUSE',
                target_task_native_qualified=False, native_prefix_safety_qualified=False)
            path = self.root / 'TARGET_STOP_ACK.json'
            write(path, document)
            raw = path.read_bytes()
            need(read(path) == document and raw == (json.dumps(document, indent=2, allow_nan=False) + '\n').encode(), 'DURABLE_STOP_READBACK')
            final_raw, final, final_step = self._boundary()
            need(final_raw == after_raw and final_step == step
                 and all(final[k] == after[k] for k in self.clock['FROZEN_FIELDS'])
                 and final['ack_sequence'] >= after['ack_sequence'], 'NATIVE_STATE_CHANGED_DURING_DURABLE_STOP_READBACK')
            self.acknowledged = True
            return NativeStopReceipt(self.episode_id, self.executor_id,
                path.name + '#sha256=' + hashlib.sha256(raw).hexdigest(), True)
        except BaseException:
            self.unknown = True
            raise

    def state(self):
        return dict(latched=self.latched, unknown=self.unknown, stop_attempted=self.stop_attempted,
                    stop_acknowledged=self.acknowledged, last_native_quota=self.last,
                    native_qualification='NOT_ASSERTED_BY_THIS_IO_COMPONENT')

    def close(self):
        # No unlatch or reopen. The original native process cleanup is the
        # admitted owner's responsibility, including every UNKNOWN STOP exit.
        self.latched = True
        if self.lock is not None:
            self.lock.close()
