"""Locked quality attempts and per-host GPU reservations within the one GO."""
import json,os,time
from collections import Counter
from pathlib import Path
from contract import ROOT,guard
COUNTS={}


def publication(root):
    assert Path(root).resolve()==ROOT.resolve();return guard()


def charge(root,kind,identity):
    import fcntl
    root=Path(root);publication(root);assert kind in ('query','head')
    limits=json.loads((root/'design/RESOURCE_RESERVATIONS.json').read_text())
    tier=root/'design/RECORDING_TIER.json'
    p=Path(json.loads(tier.read_text())['record_root'])/'evidence/quality_budget.jsonl' if tier.exists() else root/'runtime/quality_budget.jsonl'
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        # ponytail: one model lease per host, so count the durable prefix once.
        if p not in COUNTS:
            f.seek(0);COUNTS[p]=Counter(json.loads(x)['kind'] for x in f if x.strip())
        count=COUNTS[p][kind]
        assert count<limits[kind],('QUALITY_CAP',kind,count)
        f.seek(0,2);f.write(json.dumps(dict(kind=kind,ordinal=count+1,identity=identity,time_ns=time.time_ns()))+'\n');f.flush();os.fsync(f.fileno())
        COUNTS[p][kind]+=1


def process(root,event,start=None):
    root=Path(root);publication(root);assert event in ('start','end');now=time.time()
    row=dict(event=event,pid=os.getpid(),epoch=now,start_epoch=now if start is None else start,device='OWN_LEGAL_H200' if os.environ.get('PBS_JOBID') else 'CORE_GPU0')
    p=root/'runtime/GPU_PROCESSES.jsonl'
    with p.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    return row['start_epoch']


def seconds(intervals):
    total=0.;end=float('-inf')
    for start,stop in sorted(intervals):
        assert stop>=start;total+=max(0.,stop-max(start,end));end=max(end,stop)
    return total


def time_allowed(root,reserve=0):
    root=Path(root);limits=json.loads((root/'design/RESOURCE_RESERVATIONS.json').read_text())
    path=root/'runtime/GPU_PROCESSES.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    ended={(r['pid'],r['start_epoch']) for r in rows if r['event']=='end'}
    intervals=[(r['start_epoch'],r['epoch']) for r in rows if r['event']=='end']
    intervals.extend((r['epoch'],time.time()) for r in rows if r['event']=='start' and (r['pid'],r['start_epoch']) not in ended)
    assert seconds(intervals)+reserve<limits['GPU_seconds']


if __name__=='__main__':
    assert seconds([(0,4),(2,7),(9,10)])==8
    assert 32640+128==32768 and 130560+512==131072 and 57600*2==32*3600
    print('PASS_DISJOINT_HOST_CAPS_AND_SAME_DEVICE_INTERVAL_UNION; neural/physics=0')
