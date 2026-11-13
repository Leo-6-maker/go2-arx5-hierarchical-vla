"""Bounded immutable callback payloads; existing control evidence stays in RAM.

Only fresh CPU-expert construction may opt in. Successful close includes every
accepted record, flush and fsync. Failed/overflowed writes never become a
complete journal, and backpressure waits precede a new native clock grant.
"""
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import threading
import time

VERSION = 'CORELAB_BOUNDED_ASYNC_ROS_JOURNAL_V3'
MAX_RECORDS = 1024
MAX_BYTES = 32 * 1024**2


def configuration(spec):
    value = spec.get('collection_async_journal_v3')
    if value is None:
        return None
    from .native_collection_performance_v1 import configuration as performance
    if (value != dict(version=VERSION) or performance(spec) is None):
        raise ValueError('ASYNC_JOURNAL_FRESH_PERFORMANCE_EXPERT_ONLY')
    return value


class BoundedJournal:
    def __init__(self, path, *, max_records=MAX_RECORDS, max_bytes=MAX_BYTES,
                 open_file=None):
        if max_records < 1 or max_bytes < 1:
            raise ValueError('POSITIVE_JOURNAL_CAPACITY_REQUIRED')
        self.path = Path(path)
        self.max_records, self.max_bytes = max_records, max_bytes
        self.condition = threading.Condition()
        self.queue = deque()
        self.pending_records = self.pending_bytes = 0
        self.peak_records = self.peak_bytes = 0
        self.accepted = self.written = self.written_bytes = 0
        self.write_wall_s = self.maximum_write_wall_s = 0.
        self.closing = self.finished = False
        self.error = None
        self.hasher = hashlib.sha256()
        # Exclusive open precedes the worker; an old/unknown file is retained.
        self.file = (open_file or (lambda p: p.open('xb', buffering=0)))(self.path)
        self.thread = threading.Thread(target=self._run, name='native-ros-journal', daemon=True)
        self.thread.start()

    def check(self):
        with self.condition:
            if self.error is not None:
                raise RuntimeError('ASYNC_JOURNAL_WRITE_FAILED') from self.error

    def append(self, payload):
        if type(payload) is not bytes or not payload.endswith(b'\n') or payload.count(b'\n') != 1:
            raise ValueError('ONE_IMMUTABLE_COMMITTED_JSON_LINE_REQUIRED')
        with self.condition:
            self.check()
            if self.closing:
                raise RuntimeError('CLOSED_ASYNC_JOURNAL_NO_APPEND')
            if self.pending_records == self.max_records or self.pending_bytes + len(payload) > self.max_bytes:
                raise BufferError('ASYNC_JOURNAL_FULL_NO_RECORD_DROP_OR_NEW_CLOCK')
            self.queue.append(payload)
            self.accepted += 1
            self.pending_records += 1
            self.pending_bytes += len(payload)
            self.peak_records = max(self.peak_records, self.pending_records)
            self.peak_bytes = max(self.peak_bytes, self.pending_bytes)
            self.condition.notify_all()

    def backpressured(self):
        with self.condition:
            self.check()
            return (self.pending_records * 4 >= self.max_records * 3
                    or self.pending_bytes * 4 >= self.max_bytes * 3)

    def wait_low_water(self, timeout, guard=lambda: None):
        deadline = time.monotonic() + timeout
        with self.condition:
            while (self.pending_records * 2 > self.max_records
                   or self.pending_bytes * 2 > self.max_bytes):
                self.check()
                guard()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('ASYNC_JOURNAL_SAFE_PAUSE_BACKPRESSURE_TIMEOUT')
                self.condition.wait(min(.05, remaining))
            self.check()

    def _run(self):
        try:
            while True:
                with self.condition:
                    while not self.queue and not self.closing:
                        self.condition.wait()
                    if not self.queue:
                        break
                    payload = self.queue.popleft()
                start = time.perf_counter()
                view = memoryview(payload)
                while view:
                    count = self.file.write(view)
                    if not count:
                        raise OSError('ASYNC_JOURNAL_SHORT_WRITE')
                    view = view[count:]
                elapsed = time.perf_counter() - start
                self.hasher.update(payload)
                with self.condition:
                    self.written += 1
                    self.written_bytes += len(payload)
                    self.pending_records -= 1
                    self.pending_bytes -= len(payload)
                    self.write_wall_s += elapsed
                    self.maximum_write_wall_s = max(self.maximum_write_wall_s, elapsed)
                    self.condition.notify_all()
            self.file.flush()
            os.fsync(self.file.fileno())
            self.finished = True
        except BaseException as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()
        finally:
            try:
                self.file.close()
            except BaseException as error:
                with self.condition:
                    self.finished = False
                    self.error = error
                    self.condition.notify_all()

    def close(self, timeout=30):
        with self.condition:
            self.closing = True
            self.condition.notify_all()
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise TimeoutError('ASYNC_JOURNAL_FINITE_CLOSE_INCOMPLETE')
        self.check()
        if not self.finished or self.accepted != self.written or self.pending_records:
            raise RuntimeError('ASYNC_JOURNAL_INCOMPLETE_NO_COMMIT')
        return self.report()

    def report(self):
        with self.condition:
            committed = (self.finished and not self.thread.is_alive() and self.error is None
                         and self.accepted == self.written and self.pending_records == 0)
            return dict(schema=VERSION, committed=committed,
                        accepted_records=self.accepted, written_records=self.written,
                        written_bytes=self.written_bytes, pending_records=self.pending_records,
                        pending_bytes=self.pending_bytes, peak_records=self.peak_records,
                        peak_payload_bytes=self.peak_bytes, max_records=self.max_records,
                        max_payload_bytes=self.max_bytes, includes_inflight_payload=True,
                        sha256=self.hasher.hexdigest() if committed else None,
                        write_wall_s=self.write_wall_s, maximum_write_wall_s=self.maximum_write_wall_s,
                        failure=None if self.error is None else repr(self.error))


def before_clock_grant(adapter):
    writer = getattr(adapter, 'ros_journal', None)
    if writer is None or not writer.backpressured():
        return
    packet = adapter.observe()
    ack = json.loads((adapter.output/'pause_ack.json').read_bytes())
    if (ack.get('step') != packet['step'] or ack.get('running') is not False
            or ack.get('clock_paused') is not True
            or abs(ack.get('time_s', float('inf')) - packet['time_s']) > 1e-8):
        raise RuntimeError('ASYNC_JOURNAL_BACKPRESSURE_REQUIRES_EXACT_SAFE_PAUSE')
    from .native_collection_performance_v1 import measure
    with measure(adapter, 'ros.async_backpressure_at_verified_pause'):
        writer.wait_low_water(5, adapter.guard)


def close_journal(adapter):
    writer = getattr(adapter, 'ros_journal', None)
    if writer is not None:
        from scripts.pi0292.durable import write
        try:
            writer.close()
            if writer.accepted != len(adapter.ros_evidence):
                raise RuntimeError('ASYNC_JOURNAL_CALLBACK_COUNT_MISMATCH_NO_COMMIT')
        finally:
            report = writer.report()
            report['observed_callback_records'] = len(adapter.ros_evidence)
            if writer.accepted != len(adapter.ros_evidence):
                report.update(committed=False, sha256=None,
                              failure='ASYNC_JOURNAL_CALLBACK_COUNT_MISMATCH_NO_COMMIT')
            write(adapter.output/'ASYNC_ROS_JOURNAL_ACTUAL.private.json', report)
