"""Finite bounded handoff of immutable frames to one archive consumer.

No renderer, native world or policy lives in this executor. Payload ownership
is retained until publication completes; the calling thread handles pressure
only at its verified paused boundary. Readers discover only committed frames.
"""
from collections import deque
from concurrent.futures import Future, TimeoutError as FutureTimeout
from dataclasses import dataclass
import hashlib
import json
from queue import SimpleQueue
import threading
import time

MAX_FRAMES = 16
MAX_BYTES = 16 * 1024**2
RESERVATION_BYTES = 4 * 1024**2
PIXEL_BYTES = 256 * 256 * 3


@dataclass(frozen=True)
class FrozenFrame:
    header: bytes
    native_state: bytes
    native_body: bytes
    native_wrist: bytes
    body: bytes
    native_calibration: bytes
    calibration: bytes

    def __post_init__(self):
        if any(type(getattr(self, key)) is not bytes for key in self.__dataclass_fields__):
            raise ValueError('FRAME_REQUIRES_OWNED_IMMUTABLE_BYTES')
        if any(len(value) != PIXEL_BYTES for value in (self.native_body, self.native_wrist, self.body)):
            raise ValueError('THREE_ORIGINAL_RGB256_ARRAYS_REQUIRED')
        if self.nbytes > RESERVATION_BYTES:
            raise ValueError('FRAME_EXCEEDS_FINITE_PAYLOAD_RESERVATION')
        json.loads(self.header)

    @property
    def nbytes(self):
        return sum(len(getattr(self, key)) for key in self.__dataclass_fields__)

    @property
    def sha256(self):
        h = hashlib.sha256()
        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            h.update(len(value).to_bytes(8, 'little'));h.update(value)
        return h.hexdigest()

    def pixels(self, name):
        """A view backed by bytes; neither consumer can re-enable writes."""
        import numpy as np
        return np.frombuffer(getattr(self, name), dtype=np.uint8).reshape(256, 256, 3)


class FrameWriter:
    def __init__(self, publish, *, max_frames=MAX_FRAMES, max_bytes=MAX_BYTES):
        if max_frames < 1 or max_bytes < 1:
            raise ValueError('POSITIVE_FRAME_WRITER_CAPACITY_REQUIRED')
        self.publish = publish
        self.max_frames, self.max_bytes = max_frames, max_bytes
        self.tasks = SimpleQueue()
        self.pending = deque()
        self.pending_bytes = self.peak_frames = self.peak_bytes = 0
        self.accepted = 0
        self.committed = {}
        self.failure = None
        self.closed = False
        self.publish_wall_s = 0.
        self.thread = threading.Thread(target=self._work, name='native-frame-archive', daemon=True)
        self.thread.start()

    def _publish(self, frame):
        began = time.perf_counter()
        value = self.publish(frame)
        return value, time.perf_counter() - began

    def _work(self):
        while True:
            item = self.tasks.get()
            if item is None:return
            future, frame = item
            if future.set_running_or_notify_cancel():
                try:future.set_result(self._publish(frame))
                except BaseException as error:future.set_exception(error)
            del item, future, frame

    def reap(self):
        if self.failure is not None:
            raise RuntimeError('ASYNC_FRAME_PUBLICATION_FAILED') from self.failure
        while self.pending and self.pending[0][1].done():
            key, future, size = self.pending[0]
            try:
                value, wall = future.result()
            except BaseException as error:
                self.failure = error
                raise RuntimeError('ASYNC_FRAME_PUBLICATION_FAILED') from error
            if value.get('query_uuid') != key or value.get('committed') is not True:
                self.failure = ValueError('FRAME_PUBLISHER_RETURNED_NO_COMMIT')
                raise RuntimeError('ASYNC_FRAME_PUBLICATION_FAILED') from self.failure
            self.pending.popleft();self.pending_bytes -= size
            self.committed[key] = value;self.publish_wall_s += wall

    def wait_room(self, size=RESERVATION_BYTES, *, timeout=5, guard=lambda: None):
        if size > self.max_bytes:
            raise ValueError('FRAME_RESERVATION_EXCEEDS_CAPACITY')
        until = time.monotonic() + timeout
        while True:
            self.reap();guard()
            if len(self.pending) < self.max_frames and self.pending_bytes + size <= self.max_bytes:
                return
            remaining = until - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('ASYNC_FRAME_SAFE_PAUSE_BACKPRESSURE_TIMEOUT')
            try:self.pending[0][1].result(timeout=min(.05, remaining))
            except FutureTimeout:pass
            except BaseException:
                self.reap()

    def submit(self, frame):
        self.reap()
        if self.closed or not isinstance(frame, FrozenFrame):
            raise ValueError('OPEN_WRITER_AND_FROZEN_FRAME_REQUIRED')
        key = json.loads(frame.header)['query_uuid']
        if key in self.committed or any(k == key for k, _, _ in self.pending):
            raise FileExistsError('FRAME_IDENTITY_NEVER_REPLAY')
        if len(self.pending) >= self.max_frames or self.pending_bytes + frame.nbytes > self.max_bytes:
            raise BufferError('FRAME_CAPACITY_REFUSAL_NO_DROP_OR_NEW_CLOCK')
        future = Future()
        self.pending.append((key, future, frame.nbytes));self.pending_bytes += frame.nbytes
        self.accepted += 1
        self.peak_frames = max(self.peak_frames, len(self.pending))
        self.peak_bytes = max(self.peak_bytes, self.pending_bytes)
        self.tasks.put((future, frame))

    def drain(self, *, timeout=30, guard=lambda: None):
        until = time.monotonic() + timeout
        while self.pending:
            self.reap();guard()
            if not self.pending:break
            remaining = until - time.monotonic()
            if remaining <= 0:raise TimeoutError('ASYNC_FRAME_FINITE_DRAIN_INCOMPLETE')
            try:self.pending[0][1].result(timeout=min(.05, remaining))
            except FutureTimeout:pass
            except BaseException:self.reap()
        self.reap()

    def close(self):
        if self.closed:return
        try:self.drain()
        finally:
            self.closed = True
            # A failed native endpoint never publishes a complete episode.
            for _, future, _ in self.pending:future.cancel()
            self.tasks.put(None)
            self.thread.join(timeout=1)

    def report(self):
        return dict(accepted=self.accepted, committed=len(self.committed),
                    pending_frames=len(self.pending), pending_payload_bytes=self.pending_bytes,
                    peak_frames=self.peak_frames, peak_payload_bytes=self.peak_bytes,
                    max_frames=self.max_frames, max_payload_bytes=self.max_bytes,
                    includes_inflight_payload=True, immutable_owned_payloads=True,
                    reusable_slots=False, publish_wall_s=self.publish_wall_s,
                    failure=None if self.failure is None else repr(self.failure))
