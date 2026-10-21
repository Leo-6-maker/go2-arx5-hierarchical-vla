"""Bounded retries for read-only observations, never effect-bearing RPCs."""
import time


def observe(read, *, sleep=time.sleep, delays=(15, 45)):
    """Retry known transient transports; callers must supply a pure read."""
    for attempt in range(len(delays) + 1):
        try:
            return read()
        except (RuntimeError, OSError, TimeoutError) as error:
            transient = any(marker in str(error) for marker in (
                'Temporary failure in name resolution', 'Could not resolve hostname',
                'Connection timed out', 'Connection reset', 'Broken pipe', 'timed out'))
            if not transient or attempt == len(delays):
                raise
            sleep(delays[attempt])
