"""Private local receipts before a single wire attempt, never a retry policy."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time


def append(path,value):
    return append_records(path,[value])


def append_records(path,values):
    if not isinstance(values,list) or not 1<=len(values)<=2:
        raise ValueError('ONE_RECEIPT_OR_WRITTEN_AND_TERMINAL_PAIR')
    raw=''.join(json.dumps(value,allow_nan=False)+'\n' for value in values).encode()
    fd=os.open(path,os.O_CREAT|os.O_APPEND|os.O_WRONLY|os.O_NOFOLLOW|os.O_NONBLOCK,0o600)
    try:
        info=os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_mode&0o077
                or info.st_size+len(raw)>128*1024**2):
            raise PermissionError('PRIVATE_BOUNDED_REGULAR_RPC_JOURNAL_REQUIRED')
        if os.write(fd,raw)!=len(raw):raise IOError('INCOMPLETE_RPC_JOURNAL_RECEIPT')
        before=time.monotonic();epoch=time.time()
        os.fsync(fd)
        after=time.monotonic();ended_epoch=time.time()
        return dict(fsync_started_monotonic=before,fsync_ended_monotonic=after,
            fsync_started_epoch=epoch,fsync_ended_epoch=ended_epoch,fsync_elapsed_s=after-before)
    finally:os.close(fd)


class Attempt:
    def __init__(self,session,args,raw,nonce,deadline):
        if (session.role not in ['core','hopper'] or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',session.label)
                or args.get('operation') not in ['observation','ack','install','query','response']):
            raise PermissionError('EXACT_PRIVATE_RPC_RECEIPT_SCOPE')
        base=Path(session.base)
        if base.is_symlink() or not base.is_dir():raise PermissionError('REAL_OWNED_RPC_JOURNAL_PARENT')
        self.session=session;self.path=base/('file-rpc-'+session.label+'-'+session.role+'.attempts.private.jsonl')
        self.identity=dict(rpc_id=nonce,role=session.role,label=session.label,operation=args['operation'],
            request_sha256=hashlib.sha256(raw).hexdigest(),request_bytes=len(raw),
            deadline_monotonic=deadline,call_watchdog_s=20,effect_retried=False)
        self.phase='REQUEST_INTENT_BEFORE_WIRE';self.received=False;self.persistence=None;self.pending_written=None
        self.record(self.phase)

    def event(self,phase,**fields):
        return dict(self.identity,event=phase,monotonic=time.monotonic(),epoch=time.time(),
            previous_receipt_persistence=self.persistence,**fields)

    def record(self,phase,**fields):
        records=[] if self.pending_written is None else [self.pending_written]
        self.pending_written=None
        records.append(self.event(phase,**fields))
        persistence=append_records(self.path,records)
        self.phase=phase
        self.persistence=dict(event=phase,batch_events=[r['event'] for r in records],**persistence)

    def __enter__(self):return self

    def written(self):
        if self.pending_written is not None or self.phase!='REQUEST_INTENT_BEFORE_WIRE':
            raise PermissionError('ONE_LOCAL_WRITE_PROGRESS_EVENT_PER_ATTEMPT')
        # The durable pre-wire intent already represents an unknown remote effect.
        # Persist progress with response/failure, never block reply draining for it.
        self.phase='LOCAL_WRITE_RETURNED_NOT_REMOTE_ACK';self.pending_written=self.event(self.phase)

    def response(self,line,value):
        self.received=True
        self.record('RESPONSE_PARSED',response_sha256=hashlib.sha256(line).hexdigest(),
            response_bytes=len(line),response_ok=value.get('ok') is True)

    def __exit__(self,kind,error,traceback):
        if error is not None:
            self.session.failed=True
            try:
                self.record('FAILED_NO_RETRY',last_completed_phase=self.phase,error_type=kind.__name__,
                    original_error=str(error),process_returncode=self.session.process.poll(),
                    response_received=self.received,unknown_remote_effect=not self.received)
            except Exception as logging_error:
                if hasattr(error,'add_note'):error.add_note('RPC failure receipt also failed: '+type(logging_error).__name__)
        return False
