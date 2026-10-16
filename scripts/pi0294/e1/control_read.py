"""Bounded read-only recovery for NFS rename visibility; never retries mutations."""
import errno
import time

from scripts.pi0292.durable import read as original_read


def read(path):
    for attempt in range(6):
        try:return original_read(path)
        except OSError as exc:
            if exc.errno not in (errno.ENOENT,errno.ESTALE) or attempt==5:raise
            time.sleep(.1)
