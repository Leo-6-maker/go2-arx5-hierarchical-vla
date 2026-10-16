"""A partial pipe write is transport progress, never a new RPC or retry."""
import os
import select
import time


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
