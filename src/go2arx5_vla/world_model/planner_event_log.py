"""Incremental reads of the original receiver's append-only JSONL log.

Only newline-committed records are visible. Returned records are internal
read-only evidence; callers may change the returned list, but not its records.
No reference, clock, controller or raw log is changed here.
"""
import json
import os
from pathlib import Path


class PlannerEventLog:
    def __init__(self, path):
        self.path = Path(path)
        self._reset()

    def _reset(self):
        self._identity = None
        self._offset = 0
        self._size = 0
        self._mtime_ns = None
        self._events = []

    def read(self):
        try:
            with self.path.open('rb') as handle:
                before = os.fstat(handle.fileno())
                identity = (before.st_dev, before.st_ino)
                if (identity != self._identity or before.st_size < self._size
                        or (before.st_size == self._size
                            and before.st_mtime_ns != self._mtime_ns)):
                    self._reset()
                handle.seek(self._offset)
                raw = handle.read()
                committed = raw[:raw.rfind(b'\n')+1]
                # Decode transactionally: a malformed complete line must keep
                # failing, rather than disappear behind an advanced cursor.
                fresh = [json.loads(line) for line in committed.splitlines() if line.strip()]
                after = os.fstat(handle.fileno())
                self._events.extend(fresh)
                self._offset += len(committed)
                self._identity = identity
                self._size = after.st_size
                self._mtime_ns = after.st_mtime_ns
        except FileNotFoundError:
            self._reset()
            return []
        return list(self._events)
