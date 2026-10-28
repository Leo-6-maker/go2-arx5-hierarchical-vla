"""A partial pipe write is transport progress, never a new RPC or retry."""
import os
import select
import time


class FrameReader:
    """Read newline frames without mixing fd readiness with buffered stdio."""
    def __init__(self,fd,maximum_bytes):
        if type(fd) is not int or type(maximum_bytes) is not int or maximum_bytes<=0:
            raise ValueError('VALID_FILE_RPC_FRAME_READER_REQUIRED')
        self.fd,self.maximum_bytes,self.buffer=fd,maximum_bytes,b''

    def read(self,timeout):
        if type(timeout) not in (int,float) or timeout<0:
            raise ValueError('NONNEGATIVE_FILE_RPC_READ_TIMEOUT_REQUIRED')
        deadline=time.monotonic()+timeout
        while b'\n' not in self.buffer:
            if len(self.buffer)>self.maximum_bytes:
                raise ValueError('BOUNDED_ONE_JSON_LINE_REQUIRED')
            if not select.select([self.fd],[],[],max(0,deadline-time.monotonic()))[0]:return None
            chunk=os.read(self.fd,min(256*1024,self.maximum_bytes+1-len(self.buffer)))
            if not chunk:
                line,self.buffer=self.buffer,b''
                return line
            self.buffer+=chunk
        line,self.buffer=self.buffer.split(b'\n',1);line+=b'\n'
        if len(line)>self.maximum_bytes:raise ValueError('BOUNDED_ONE_JSON_LINE_REQUIRED')
        return line


def write_frame(fd,data,deadline):
    if not isinstance(data,bytes) or not data:raise ValueError('ONE_NONEMPTY_ENCODED_FRAME_REQUIRED')
    previous=os.get_blocking(fd);os.set_blocking(fd,False);sent=0
    try:
        while sent<len(data):
            remaining=deadline-time.monotonic()
            if remaining<=0 or not select.select([],[fd],[],max(0,remaining))[1]:
                raise TimeoutError('ORIGINAL_FILE_RPC_WRITE_TIMEOUT_UNKNOWN_EFFECT_NO_RETRY')
            try:count=os.write(fd,memoryview(data)[sent:])
            except BlockingIOError:continue
            if count<=0:raise ConnectionError('FILE_RPC_WRITE_CLOSED_UNKNOWN_EFFECT_NO_RETRY')
            sent+=count
    finally:os.set_blocking(fd,previous)
