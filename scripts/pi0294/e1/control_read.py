"""Bounded read-only recovery for NFS visibility or transient file-table pressure."""
import errno
import time

from scripts.pi0292.durable import read as original_read


def read(path):
    for attempt in range(6):
        try:return original_read(path)
        except OSError as exc:
            if exc.errno not in (errno.ENOENT,errno.ESTALE,errno.ENFILE,errno.EMFILE) or attempt==5:raise
            time.sleep(.1)
