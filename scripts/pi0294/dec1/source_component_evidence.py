"""One admitted source owner's durable append evidence; no native admission.

Only the exact original callbacks supplied by a cold CPU worker use this file.
Each append is fsynced and read back before return. A failed write/readback is
latched, including an interruption after a successful persisted write. It is
never retried, truncated or reconstructed. No model or physics is invoked.
"""
import hashlib
import json
import os
from pathlib import Path
import threading
from go2arx5_vla.task_data.io import jsonable

SCHEMA='DEC1_SOURCE_COMPONENT_EVENT_CHAIN_V1'
ZERO='0'*64


def canonical(value):
    return json.dumps(jsonable(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def verify_event_bytes(raw,*,episode_id,executor_id,maximum_bytes,expected_offset,
                       expected_sequence,expected_previous):
    """Original strict readback loop, with EXTERNALLY confirmed final anchors.

    Neither parsing the supplied bytes nor hashing their last line supplies
    those anchors. Caller must retain the live owner anchors or sealed terminal
    readback of them. No IO/native/state producer is performed here.
    """
    if (type(raw) is not bytes or type(maximum_bytes) is not int or maximum_bytes<1
            or type(expected_offset) is not int or expected_offset<0
            or type(expected_sequence) is not int or expected_sequence<0
            or not isinstance(expected_previous,str) or len(expected_previous)!=64
            or any(c not in '0123456789abcdef' for c in expected_previous)):
        raise ValueError('EXACT_OWNER_CONFIRMED_EVENT_ANCHORS_REQUIRED')
    if len(raw)!=expected_offset or len(raw)>maximum_bytes or raw and not raw.endswith(b'\n'):
        raise ValueError('SOURCE_EVENT_CHAIN_TRUNCATED_OR_EXTRA_BYTES')
    events=[];previous=ZERO
    for sequence,line in enumerate(raw.splitlines()):
        record=json.loads(line);body={k:v for k,v in record.items() if k!='sha256'}
        if (set(record)!={'schema','episode_id','executor_id','sequence','previous_sha256','event','sha256'}
                or record['schema']!=SCHEMA or record['episode_id']!=episode_id
                or record['executor_id']!=executor_id or type(record['sequence']) is not int
                or record['sequence']!=sequence or record['previous_sha256']!=previous
                or type(record['event']) is not dict
                or hashlib.sha256(canonical(body)).hexdigest()!=record['sha256']
                or canonical(record)!=line):
            raise ValueError('SOURCE_EVENT_CHAIN_IDENTITY_SEQUENCE_OR_BYTES_CHANGED')
        events.append(record['event']);previous=record['sha256']
    if len(events)!=expected_sequence or previous!=expected_previous:
        raise ValueError('SOURCE_EVENT_CHAIN_NOT_LAST_CONFIRMED_OWNER_APPEND')
    return events


class SourceComponentEvidence:
    def __init__(self,path,*,episode_id,executor_id,maximum_bytes):
        self.path=Path(path)
        if (not isinstance(episode_id,str) or not episode_id or not isinstance(executor_id,str) or not executor_id
                or type(maximum_bytes) is not int or maximum_bytes<1 or maximum_bytes>512*1024**2
                or not self.path.parent.is_dir() or self.path.exists() or self.path.is_symlink()):
            raise PermissionError('FRESH_OWNER_REGISTERED_EVENT_FILE_AND_BOUNDED_OUTPUT_REQUIRED')
        self.episode_id=episode_id;self.executor_id=executor_id;self.maximum=maximum_bytes
        self.pid=os.getpid();self.thread=threading.get_ident();self.unknown=False;self.closed=False
        self.offset=0;self.sequence=0;self.previous=ZERO
        self.fd=os.open(self.path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        stat=os.fstat(self.fd);self.identity=(stat.st_dev,stat.st_ino)
        try:
            os.fsync(self.fd)
            directory=os.open(self.path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(directory)
            finally:os.close(directory)
        except BaseException:
            self.unknown=True;os.close(self.fd);self.closed=True;raise

    def _owner(self):
        if os.getpid()!=self.pid or threading.get_ident()!=self.thread:
            self.unknown=True;raise PermissionError('SOURCE_EVIDENCE_SINGLE_PROCESS_THREAD_OWNER')

    def _open_read(self):
        fd=os.open(self.path,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            stat=os.fstat(fd)
            if (stat.st_dev,stat.st_ino)!=self.identity or stat.st_nlink!=1:
                raise PermissionError('SOURCE_EVENT_FILE_IDENTITY_CHANGED')
            return os.fdopen(fd,'rb')
        except BaseException:os.close(fd);raise

    def record(self,event):
        self._owner()
        if self.closed or self.unknown:raise PermissionError('SOURCE_EVIDENCE_CLOSED_OR_WRITE_OUTCOME_UNKNOWN_NO_RETRY')
        try:
            if type(event) is not dict or not isinstance(event.get('kind'),str) or not event['kind']:
                raise ValueError('SOURCE_TYPED_EVENT_REQUIRED')
            for key,value in (('episode_id',self.episode_id),('executor_id',self.executor_id)):
                if key in event and event[key]!=value:raise PermissionError('FOREIGN_SOURCE_EVENT_IDENTITY')
            body=dict(schema=SCHEMA,episode_id=self.episode_id,executor_id=self.executor_id,
                sequence=self.sequence,previous_sha256=self.previous,event=json.loads(canonical(event)))
            digest=hashlib.sha256(canonical(body)).hexdigest();frame=canonical(dict(body,sha256=digest))+b'\n'
            stat=os.fstat(self.fd)
            if stat.st_size!=self.offset or (stat.st_dev,stat.st_ino)!=self.identity or stat.st_nlink!=1:
                raise PermissionError('SOURCE_EVENT_FILE_CHANGED_OUTSIDE_OWNER')
            if self.offset+len(frame)>self.maximum:raise PermissionError('SOURCE_REGISTERED_EVENT_OUTPUT_BOUND')
            wrote=0
            while wrote<len(frame):
                count=os.write(self.fd,frame[wrote:])
                if count<=0:raise OSError('SOURCE_EVENT_APPEND_NO_PROGRESS')
                wrote+=count
            os.fsync(self.fd)
            with self._open_read() as h:
                if os.fstat(h.fileno()).st_size!=self.offset+len(frame):raise ValueError('SOURCE_EVENT_APPEND_SIZE_UNKNOWN')
                h.seek(self.offset)
                if h.read()!=frame:raise ValueError('SOURCE_EVENT_APPEND_BYTES_READBACK_UNKNOWN')
            self.offset+=len(frame);self.sequence+=1;self.previous=digest
            return digest
        except BaseException:
            self.unknown=True;raise

    def read_events(self):
        """Independently reread all durable bytes; list hash is not file hash."""
        self._owner()
        if self.unknown:raise PermissionError('SOURCE_EVENT_OUTCOME_UNKNOWN_NO_RECEIPT')
        try:
            with self._open_read() as h:raw=h.read(self.maximum+1)
            return verify_event_bytes(raw,episode_id=self.episode_id,executor_id=self.executor_id,
                maximum_bytes=self.maximum,expected_offset=self.offset,
                expected_sequence=self.sequence,expected_previous=self.previous)
        except BaseException:self.unknown=True;raise

    def close(self):
        self._owner()
        if self.closed:raise PermissionError('SOURCE_EVENT_CLOSE_ALREADY_ATTEMPTED_NO_RETRY')
        self.closed=True
        try:os.close(self.fd)
        except BaseException:self.unknown=True;raise
