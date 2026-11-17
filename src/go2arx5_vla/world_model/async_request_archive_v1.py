"""Bounded post-ACK archive thread; existing Journal/owner, no scheduler.

Reserve BEFORE capture/dispatch. A reservation without a COMPLETE archive is
UNKNOWN after restart, including reservations whose model cost may be zero.
No automatic adoption, archive retry, model replay or deletion of live files.
The caller must stop mutating a request subtree before submit().
"""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from queue import Queue
import stat
from threading import Condition, Thread
import time

from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0294.e1.journal import Journal, MutationOutcomeUnknown, digest as journal_digest
from .hybrid_ipc_storage_v31 import archive_request


def manifest(source, limit):
    source = Path(source)
    if source.is_symlink() or not source.is_dir():
        raise ValueError('OWNED_REQUEST_DIRECTORY_REQUIRED')
    rows, size = [], 0
    for path in sorted(source.rglob('*')):
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError('REGULAR_ARCHIVE_FILES_ONLY')
        size += path.stat().st_size
        if size > limit:
            raise MemoryError('REQUEST_ARCHIVE_RESERVATION_BYTES_EXCEEDED')
        with path.open('rb') as handle:
            raw = handle.read(limit + 1)
        if len(raw) != path.stat().st_size or len(raw) > limit:
            raise ValueError('SOURCE_CHANGED_DURING_SEAL')
        rows.append(dict(file=str(path.relative_to(source)), bytes=len(raw),
                         sha256=hashlib.sha256(raw).hexdigest()))
    if sum(row['bytes'] for row in rows) > limit:
        raise MemoryError('REQUEST_ARCHIVE_RESERVATION_BYTES_EXCEEDED')
    return rows


def unknown_reservations(journal):
    unknown = []
    for path in journal.directory.glob('RESERVE-*.json'):
        message = path.stem[len('RESERVE-'):]
        reservation = read(path)
        archive = journal.directory / ('ARCHIVE-' + message + '.json')
        if (reservation['run_uuid'] != journal.run or reservation['status'] != 'COMPLETE'
                or not archive.exists()):
            unknown.append(message)
            continue
        record = read(archive)
        if (record['run_uuid'] != journal.run or record['status'] != 'COMPLETE'
                or journal_digest(reservation['result']) != reservation['result_sha256']):
            unknown.append(message)
            continue
        try:
            result = record['result']
            path = Path(result['path'])
            if (journal_digest(result) != record['result_sha256']
                    or hashlib.sha256(path.read_bytes()).hexdigest() != result['sha256']):
                raise ValueError('COMPLETED_ARCHIVE_RECEIPT_CHANGED')
            sealed = read(path)
            if not sealed['complete']:
                raise ValueError('INCOMPLETE_ARCHIVE')
            for item in sealed['files']:
                file = path.parent/item['file']
                if file.stat().st_size != item['bytes'] or hashlib.sha256(file.read_bytes()).hexdigest() != item['sha256']:
                    raise ValueError('COMPLETED_ARCHIVE_BYTES_CHANGED')
        except (OSError, ValueError, KeyError):
            unknown.append(message)
    return sorted(unknown)


@dataclass(frozen=True)
class Reservation:
    message_id: str
    max_bytes: int


class AsyncRequestArchive:
    def __init__(self, journal, *, max_pending=4, max_bytes=8*1024**2, archive=archive_request):
        if not isinstance(journal, Journal) or type(max_pending) is not int or not 1 <= max_pending <= 4:
            raise ValueError('EXISTING_JOURNAL_AND_AT_MOST_FOUR_PENDING_REQUIRED')
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('BOUNDED_ARCHIVE_BYTES_REQUIRED')
        self.journal, self.max_pending, self.max_bytes = journal, max_pending, max_bytes
        self._owner = OwnerLock(journal.directory / 'archive-queue-owner.lock')
        try:
            unknown = unknown_reservations(journal)
            if unknown:
                raise MutationOutcomeUnknown('UNARCHIVED_UNKNOWN_NO_BLIND_REPLAY:' + ','.join(unknown))
        except BaseException:
            self._owner.close()
            raise
        self._archive = archive
        self._condition = Condition()
        self._work = Queue(maxsize=max_pending)
        self._pending = {}
        self._failure = None
        self._closed = False
        self.completed = {}
        self.peak_pending = self.peak_reserved_bytes = 0
        self._thread = Thread(target=self._worker, name='sla-request-archive', daemon=True)
        self._thread.start()

    def require_dispatch(self, reservation=None):
        with self._condition:
            if self._failure is not None:
                raise MutationOutcomeUnknown('ARCHIVE_FAILED_STOP_NEW_DISPATCH') from self._failure
            if self._closed:
                raise RuntimeError('ARCHIVE_QUEUE_CLOSED')
            if reservation is not None and self._pending.get(reservation.message_id) != (reservation, 'RESERVED'):
                raise ValueError('ACTIVE_UNSUBMITTED_RESERVATION_REQUIRED')

    @property
    def pending_count(self):
        with self._condition:
            return len(self._pending)

    def reserve(self, message_id, payload, *, max_bytes, timeout_s, stopped=lambda: False):
        """Durable upper bound before any request/model effect; no dispatch here."""
        if type(max_bytes) is not int or not 0 < max_bytes <= self.max_bytes or timeout_s <= 0:
            raise ValueError('FINITE_RESERVATION_REQUIRED')
        self.journal.path('RESERVE-' + message_id)
        end = time.monotonic() + timeout_s
        with self._condition:
            while True:
                self.require_dispatch()
                if stopped() or time.monotonic() >= end:
                    raise TimeoutError('ARCHIVE_BACKPRESSURE_STOP_OR_TIMEOUT_BEFORE_DISPATCH')
                if message_id in self._pending or self.journal.path('RESERVE-' + message_id).exists():
                    raise MutationOutcomeUnknown('RESERVATION_ID_ALREADY_USED_NO_REDISPATCH')
                size = sum(t.max_bytes for t, _ in self._pending.values())
                if len(self._pending) < self.max_pending and size + max_bytes <= self.max_bytes:
                    break
                self._condition.wait(min(.02, end - time.monotonic()))
            self.journal.once('RESERVE-' + message_id, dict(request=payload, max_bytes=max_bytes),
                              lambda: dict(reserved=True, automatic_retry=False, actual_generations='NOT_ASSERTED'))
            ticket = Reservation(message_id, max_bytes)
            self._pending[message_id] = (ticket, 'RESERVED')
            self.peak_pending = max(self.peak_pending, len(self._pending))
            self.peak_reserved_bytes = max(self.peak_reserved_bytes, size + max_bytes)
            return ticket

    def submit(self, ticket, source, destination, *, expected_pins=None):
        # Allow evidence from an ALREADY dispatched request to drain even when
        # another archive failed. require_dispatch() forbids further inference.
        with self._condition:
            if self._closed or self._pending.get(ticket.message_id) != (ticket, 'RESERVED'):
                raise ValueError('ONE_SUBMISSION_PER_OWNED_RESERVATION')
            self._pending[ticket.message_id] = (ticket, 'SEALING')
        try:
            rows = manifest(source, ticket.max_bytes)
            pins = {r['file']: r['sha256'] for r in rows}
            if any(pins.get(name) != sha for name, sha in (expected_pins or {}).items()):
                raise ValueError('ISSUED_INPUT_ACTION_OR_COMPILED_BYTES_CHANGED')
            payload = dict(source=str(source), destination=str(destination), files=rows)
            with self._condition:
                self._pending[ticket.message_id] = (ticket, 'QUEUED')
            self._work.put_nowait((ticket, payload))
        except BaseException as error:
            self._fail(error, ticket.message_id)
            raise

    def _fail(self, error, message_id):
        with self._condition:
            if self._failure is None:
                self._failure = error
            self._condition.notify_all()
        # Journal reservations/intents already survive if this disk is broken.
        try:
            write(self.journal.directory / ('FAILED-' + message_id + '.json'),
                  dict(state='UNKNOWN', reason=repr(error), automatic_retry=False))
        except Exception:
            pass

    def _worker(self):
        while True:
            job = self._work.get()
            if job is None:
                return
            ticket, payload = job
            try:
                def persist():
                    if manifest(payload['source'], ticket.max_bytes) != payload['files']:
                        raise ValueError('SEALED_REQUEST_CHANGED_BEFORE_ARCHIVE')
                    result = self._archive(payload['source'], payload['destination'])
                    archived = read(result['path'])
                    if (not archived['complete'] or archived['files'] != payload['files']
                            or manifest(payload['source'], ticket.max_bytes) != payload['files']):
                        raise ValueError('SEALED_REQUEST_ARCHIVE_MISMATCH')
                    destination = Path(payload['destination'])
                    directories = [p for p in destination.rglob('*') if p.is_dir()]
                    for directory in [*sorted(directories, key=lambda p: len(p.parts), reverse=True),
                                      destination, destination.parent]:
                        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                        try:
                            os.fsync(fd)
                        finally:
                            os.close(fd)
                    return result
                start = time.monotonic()
                result = self.journal.once('ARCHIVE-' + ticket.message_id, payload, persist)
                with self._condition:
                    self.completed[ticket.message_id] = dict(result, archive_wall_s=time.monotonic()-start)
            except BaseException as error:
                self._fail(error, ticket.message_id)
            finally:
                with self._condition:
                    self._pending.pop(ticket.message_id)
                    self._condition.notify_all()

    def close(self, *, timeout_s):
        """Finite drain. A hung disk requires the existing service deadline."""
        end = time.monotonic() + timeout_s
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            while any(stage == 'QUEUED' for _, stage in self._pending.values()):
                remaining = end - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('ARCHIVE_DRAIN_TIMEOUT_UNKNOWN_NO_REPLAY')
                self._condition.wait(min(.05, remaining))
        self._work.put_nowait(None)
        self._thread.join(max(0., end-time.monotonic()))
        if self._thread.is_alive():
            raise TimeoutError('ARCHIVE_THREAD_UNKNOWN')
        self._owner.close()
        if self._failure is not None or unknown_reservations(self.journal):
            raise MutationOutcomeUnknown('UNARCHIVED_UNKNOWN_NO_BLIND_REPLAY') from self._failure
        return dict(archived=len(self.completed), peak_pending=self.peak_pending,
                    peak_reserved_bytes=self.peak_reserved_bytes, max_pending=self.max_pending,
                    max_reserved_bytes=self.max_bytes)
