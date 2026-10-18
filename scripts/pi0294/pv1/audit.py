"""Read-side independent audit of terminal PV1 attempts, not new physics."""
import argparse
import json
from pathlib import Path
import resource
import time

from scripts.pi0292.durable import read,write
from scripts.pi0294.e1.audit import episode
from scripts.pi0294.native_dataset import sha
from .core_ledger import Ledger
from .runtime import validate_runtime


def terminal_attempt(root,identifier):
    root=Path(root)
    if Path(identifier).name!=identifier or identifier in ('.','..'):
        raise PermissionError('PV1_EXACT_SINGLE_TERMINAL_ATTEMPT')
    folder=root/'episodes'/identifier;decision=read(folder/'PV1_ADJUDICATION.json')
    if decision['attempt_id']!=identifier:
        raise PermissionError('PV1_TERMINAL_ADJUDICATION_IDENTITY')
    worker=read(root/'worker_processes'/(identifier+'.json'))
    p=Path('/proc',str(worker['pid']),'stat')
    if p.exists():
        fields=p.read_text().split()
        if fields[2]!='Z' and fields[21]==str(worker['proc_start']):
            raise PermissionError('PV1_NO_AUDIT_OR_VIDEO_WHILE_ATTEMPT_IS_STILL_ACTIVE')
    return {n:sha(folder/n) for n in ('RESULT.json','PV1_ADJUDICATION.json','FILE_MANIFEST.json')}


def run(root,identifier):
    root=Path(root);validate_runtime(root,'4080-simulation',closeout=True)
    ledger=Ledger(root);ledger.guard(closeout=True);before=terminal_attempt(root,identifier)
    started=time.process_time();children=resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        value=episode(root,identifier,adjudication_name='PV1_ADJUDICATION.json')
        if terminal_attempt(root,identifier)!=before:
            raise PermissionError('PV1_ORIGINAL_ATTEMPT_CHANGED_DURING_READ_SIDE_AUDIT')
        result=dict(value,new_integrations=0,new_native_constructors=0,new_model_queries=0)
        target=root/'evidence/PV1_AUDIT_RECEIPTS'/(identifier+'.json');target.parent.mkdir(exist_ok=True)
        if target.exists():
            if read(target)!=result:raise PermissionError('PV1_PRESERVE_PRIOR_AUDIT_RECEIPT')
        else:write(target,result)
        return result
    finally:
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        ledger.cpu(time.process_time()-started+after.ru_utime+after.ru_stime-children.ru_utime-children.ru_stime)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--attempt',required=True)
    args=parser.parse_args();print(json.dumps(run(args.root,args.attempt)),flush=True)
