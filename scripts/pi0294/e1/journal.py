"""One-shot mutation intents survive client death; completed reads are repeatable."""
import hashlib
import json
from pathlib import Path
import time

from scripts.pi0292.durable import OwnerLock, read, write


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class MutationOutcomeUnknown(RuntimeError):pass


class Journal:
    def __init__(self,directory,run):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.run=run

    def path(self,message_id):
        if not isinstance(message_id,str) or not message_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in message_id):
            raise ValueError('SAFE_MESSAGE_ID_REQUIRED')
        return self.directory/(message_id+'.json')

    def completed(self,message_id,payload):
        p=self.path(message_id)
        if not p.exists():return None
        record=read(p)
        if record['run_uuid']!=self.run or record['payload_sha256']!=digest(payload):
            raise ValueError('MUTATION_ID_OR_PAYLOAD_CHANGED')
        if record['status']!='COMPLETE':raise MutationOutcomeUnknown('PRESERVED_INTENT_NO_BLIND_RESEND')
        if digest(record['result'])!=record['result_sha256']:raise ValueError('COMPLETED_RESPONSE_CHANGED')
        return record['result']

    def once(self,message_id,payload,operation):
        p=self.path(message_id);lock=OwnerLock(p.with_suffix('.lock'))
        try:
            if p.exists():return self.completed(message_id,payload)
            record=dict(run_uuid=self.run,status='INTENT',message_id=message_id,
                        payload_sha256=digest(payload),created_epoch=time.time())
            write(p,record)
            result=operation()
            record.update(status='COMPLETE',result=result,result_sha256=digest(result),completed_epoch=time.time())
            write(p,record)
            return result
        finally:lock.close()


def process_is_same(record,proc_root='/proc'):
    p=Path(proc_root)/str(record['pid'])/'stat'
    if not p.exists():return False
    # Linux comm can contain spaces; fields after the final ')' are stable.
    fields=p.read_text().rsplit(')',1)[1].split()
    return fields[0]!='Z' and fields[19]==str(record['proc_start'])
