"""Bound only this delivery process to the original owner deadline."""
import json
import os
from pathlib import Path
import signal
import time

DEADLINE = 1791851837.0
EXTERNAL_OUTPUT_UPPER_BYTES = 256 * 1024**2


def atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending-' + str(os.getpid()))
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def install_deadline(root, stage):
    def expired(signum, frame):
        atomic_write(Path(root)/'evidence'/('DELIVERY_TIMEOUT_' + stage + '.json'),
            dict(status='STOP_ORIGINAL_DEADLINE', stage=stage, deadline_epoch=DEADLINE,
                 observed_epoch=time.time(), remote_operation_result='UNKNOWN_IF_IN_FLIGHT',
                 recovery='Inspect atomic TRANSFER_PROGRESS and exact remote prefix; do not blindly reupload',
                 other_processes_signalled=0))
        os._exit(124)
    signal.signal(signal.SIGALRM, expired)
    remaining = DEADLINE - time.time()
    assert remaining > 0, 'Original deadline expired; no reset or extension'
    signal.setitimer(signal.ITIMER_REAL, remaining)
