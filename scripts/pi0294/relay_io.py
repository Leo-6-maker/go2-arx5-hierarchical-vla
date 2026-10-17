"""Bounded direct file reads and metadata-only RPC phase timings."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

MAX_FILE_BYTES=2*1024**2


def read_available(path):
    # Do not cache a negative is_file() result before opening a new UUID packet.
    try:fd=os.open(Path(path),os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    except FileNotFoundError:return None
    try:
        import stat
        if not stat.S_ISREG(os.fstat(fd).st_mode):raise ValueError('REGULAR_RELAY_PACKET_REQUIRED')
        with os.fdopen(fd,'rb',closefd=False) as stream:raw=stream.read(MAX_FILE_BYTES+1)
        if len(raw)>MAX_FILE_BYTES:raise ValueError('BOUNDED_RELAY_PACKET_REQUIRED')
        return json.loads(raw)
    finally:os.close(fd)


class PhaseTrace:
    """stderr only: no observation, action, model output, path or credential text."""
    def __init__(self,stream,*,clock=time.monotonic):
        self.stream,self.clock=stream,clock;self.sequence=0

    @contextmanager
    def phase(self,name):
        if name not in ['guard','invoke','accounting','emit']:raise ValueError('EXACT_RPC_PHASE')
        start=self.clock();self.sequence+=1;sequence=self.sequence
        self.stream.write(json.dumps(dict(event='FILE_RPC_PHASE_BEGIN',phase=name,sequence=sequence))+'\n');self.stream.flush()
        status='FAILED_OR_UNKNOWN'
        try:yield;status='RETURNED'
        finally:
            self.stream.write(json.dumps(dict(event='FILE_RPC_PHASE_END',phase=name,sequence=sequence,
                elapsed_s=self.clock()-start,status=status))+'\n');self.stream.flush()
