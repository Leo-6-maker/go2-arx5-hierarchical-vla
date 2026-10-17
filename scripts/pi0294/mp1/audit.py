"""Reuse the independent raw evidence audit without relabeling old E1 records."""
import argparse
import json
import os
from pathlib import Path
import resource
import time

from scripts.pi0292.durable import read,write
from scripts.pi0294.e1.audit import episode
from .science import Ledger,validate_runtime


def run(root):
    root=Path(root);validate_runtime(root,'4080-simulation');ledger=Ledger(root,'core');ledger.guard(closeout=True)
    if read(ledger.path)['reservations']:raise PermissionError('NO_AUDIT_WITH_UNSETTLED_PHYSICS')
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]));started=time.process_time();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        rows=[]
        for folder in sorted((root/'episodes').iterdir()):
            ledger.guard(closeout=True)
            if (folder/'MP1_ADJUDICATION.json').exists():
                rows.append(episode(root,folder.name,adjudication_name='MP1_ADJUDICATION.json'))
        good={'PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT','PASS_ZERO_INTEGRATION_ENGINEERING_EVIDENCE'}
        value=dict(status='PASS_ALL_DERIVED_MAIN_EVIDENCE' if rows and all(r['status'] in good for r in rows) else 'PRESERVED_AUDIT_GAPS',
            episodes=len(rows),rows=rows,all_audits_PASS_does_not_mean_all_attempts_have_task_results=True)
        write(root/'evidence/MAIN_EVIDENCE_AUDIT_INDEX.private.json',value);return value
    finally:
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        ledger.cpu(time.process_time()-started+after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();v=run(a.root)
    print(json.dumps(dict(status=v['status'],episodes=v['episodes'])),flush=True)
