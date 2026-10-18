"""Bounded direct file reads and metadata-only RPC phase timings."""
from contextlib import contextmanager
import json
import hashlib
import os
from pathlib import Path
import re
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
    def __init__(self,stream,*,clock=time.monotonic,wall_clock=time.time):
        self.stream,self.clock,self.wall_clock=stream,clock,wall_clock;self.sequence=0;self.context={}

    def received(self,encoded_bytes):
        if type(encoded_bytes) is not int or not 0<encoded_bytes<=MAX_FILE_BYTES:
            raise ValueError('BOUNDED_RPC_FRAME_SIZE_ONLY')
        self.stream.write(json.dumps(dict(event='FILE_RPC_FRAME_RECEIVED',epoch=self.wall_clock(),
            monotonic=self.clock(),encoded_bytes=encoded_bytes,**self.context))+'\n');self.stream.flush()

    def bind_rpc(self,packet):
        nonce=packet.get('rpc_id') if isinstance(packet,dict) else None
        args=packet.get('args') if isinstance(packet,dict) else None
        operation=args.get('operation') if isinstance(args,dict) else None
        if (isinstance(nonce,str) and 1<=len(nonce)<=64
                and operation in ['observation','ack','install','query','response']):
            identity=dict(rpc_id=nonce) if re.fullmatch(r'[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}',nonce) else dict(rpc_id_sha256=hashlib.sha256(nonce.encode()).hexdigest())
            self.context=dict(identity,operation=operation)
        else:self.context={}

    @contextmanager
    def phase(self,name):
        if name not in ['guard','invoke','accounting','emit']:raise ValueError('EXACT_RPC_PHASE')
        start=self.clock();self.sequence+=1;sequence=self.sequence
        self.stream.write(json.dumps(dict(event='FILE_RPC_PHASE_BEGIN',phase=name,sequence=sequence,
            epoch=self.wall_clock(),monotonic=start,**self.context))+'\n');self.stream.flush()
        status='FAILED_OR_UNKNOWN'
        try:yield;status='RETURNED'
        finally:
            end=self.clock()
            self.stream.write(json.dumps(dict(event='FILE_RPC_PHASE_END',phase=name,sequence=sequence,
                epoch=self.wall_clock(),monotonic=end,elapsed_s=end-start,status=status,**self.context))+'\n');self.stream.flush()
