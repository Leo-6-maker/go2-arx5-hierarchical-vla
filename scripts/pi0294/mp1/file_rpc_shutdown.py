"""Drain abandoned stdout during finite close; never recover or replay an RPC."""
import hashlib
import os
import select
import subprocess
import time


def close_channel(process,*,timeout=30,max_bytes=2*1024**2):
    if type(timeout) not in [int,float] or not 0<timeout<=30 or type(max_bytes) is not int or not 0<max_bytes<=2*1024**2:
        raise ValueError('FINITE_ORIGINAL_CLOSE_AND_PACKET_BOUND')
    if process.stdin and not process.stdin.closed:process.stdin.close()
    end=time.monotonic()+timeout;count=0;digest=hashlib.sha256();status='UNKNOWN_PROCESS_TERMINAL'
    stream=process.stdout;fd=None;blocking=None;eof=stream is None or stream.closed
    try:
        if not eof:
            fd=stream.fileno();blocking=os.get_blocking(fd);os.set_blocking(fd,False)
        while True:
            remaining=end-time.monotonic()
            if remaining<=0:break
            if eof:
                try:process.wait(timeout=remaining);status='PROCESS_TERMINAL';break
                except subprocess.TimeoutExpired:break
            if not select.select([stream],[],[],remaining)[0]:break
            try:chunk=os.read(fd,min(256*1024,max_bytes-count+1))
            except BlockingIOError:continue
            if not chunk:eof=True;continue
            count+=len(chunk);digest.update(chunk)
            if count>max_bytes:status='ABANDONED_OUTPUT_EXCEEDS_PACKET_BOUND';break
        code=process.poll()
        if code is not None and status=='UNKNOWN_PROCESS_TERMINAL':status='PROCESS_TERMINAL'
        return dict(status=status,returncode=code,abandoned_stdout_bytes=count,
            abandoned_stdout_sha256=digest.hexdigest(),discarded_output_is_rpc_ack=False,
            effects_retried=False,process_signalled=False,close_watchdog_s=timeout)
    finally:
        if fd is not None:os.set_blocking(fd,blocking)
