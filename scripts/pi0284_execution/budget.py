"""One locked append-only ledger; disjoint host reservations sum to the published cap."""
import fcntl,json,os,time
from pathlib import Path

RELEASE='fc9647b2183e5bb9d3943f4a57565ed9751e0739'
CANONICAL=6053658526

def publication(root):
    v=json.loads((Path(root)/'design/PUBLICATION_RECEIPT.json').read_text())
    assert v['release']==RELEASE and v['canonical']==CANONICAL and v['launch_allowed']
    assert v['status']=='PUBLISHED_EXACT_FILES_AND_CANONICAL_READ_BACK'
    return v

def charge(root,kind,identity):
    root=Path(root);publication(root)
    card=json.loads((root/'design/RUNTIME_CARD.json').read_text())
    assert kind in ('query','head')
    path=root/'evidence/quality_budget.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.seek(0)
        count=sum(json.loads(x)['kind']==kind for x in f if x.strip())
        assert count<card['host_caps'][kind],('PI0284_HOST_CAP',kind,count)
        f.seek(0,2);f.write(json.dumps(dict(kind=kind,identity=identity,time_ns=time.time_ns()))+'\n');f.flush();os.fsync(f.fileno())

def process(root,event,start=None):
    root=Path(root);publication(root);now=time.time()
    row=dict(event=event,pid=os.getpid(),epoch=now,start_epoch=now if start is None else start)
    if event=='end':row['elapsed_s']=now-start
    path=root/'evidence/model_processes.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    return row['start_epoch']

def time_allowed(root,reserve=0):
    root=Path(root);card=json.loads((root/'design/RUNTIME_CARD.json').read_text())
    p=root/'evidence/model_processes.jsonl'
    rows=[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
    ended={v['pid'] for v in rows if v['event']=='end'}
    used=sum(v['elapsed_s'] for v in rows if v['event']=='end')+sum(time.time()-v['epoch'] for v in rows if v['event']=='start' and v['pid'] not in ended)
    assert used+reserve<card['host_caps']['GPU_seconds'],'PI0284_GPU_RESERVATION'

def cpu_check(folder):
    """Exercise exhausted counters and legal K without loading any neural module."""
    root=Path(folder);root.mkdir(parents=True,exist_ok=True)
    (root/'design').mkdir(exist_ok=True)
    (root/'design/PUBLICATION_RECEIPT.json').write_text(json.dumps(dict(release=RELEASE,canonical=CANONICAL,launch_allowed=True,status='PUBLISHED_EXACT_FILES_AND_CANONICAL_READ_BACK')))
    (root/'design/RUNTIME_CARD.json').write_text(json.dumps(dict(host_caps=dict(query=1,head=1,GPU_seconds=1))))
    for k in ('query','head'):
        charge(root,k,'CPU_CHECK')
        try:charge(root,k,'CPU_CHECK_EXHAUSTED')
        except AssertionError:pass
        else:raise AssertionError('Counter guard missing')
    assert sum([6272,128])==6400 and sum([100352,2048])==102400
    assert sum([14400,28800])==12*3600
    print('PASS_CPU_PUBLICATION_AND_EXHAUSTED_COUNTERS; neural/optimizer/physics=0')

if __name__=='__main__':
    import sys
    cpu_check(sys.argv[1])
