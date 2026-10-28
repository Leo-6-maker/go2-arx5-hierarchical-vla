"""PI0283 finite caps; user explicitly removed the inherited wall clock."""
import json,time,os,fcntl
from pathlib import Path
RELEASE='508e3db7b63d3c47758794f3ffbad87b577e6e9d'


def load(root):
    root=Path(root);v=json.loads((root/'design/RUNTIME_AUTHORITY.json').read_text())
    assert v['release']==RELEASE and v['canonical']==6048886543
    assert v['formal_cap']==3072 and v['all_updates_cap']==3136
    r=json.loads((root/'design/CLOCK_RESOLUTION.json').read_text())
    assert r['resolution']=='EXPLICIT_USER_REMOVAL_OF_WALL_CLOCK_GATES' and r['deadline_epoch'] is None
    return v


def allowed(root,kind,reserve_s=0):
    root=Path(root);v=load(root)
    p=root/'runtime/TRAINING_ALLOCATION.json'
    if p.exists() and kind=='update':
        lease=json.loads(p.read_text());assert time.time()-lease['started_epoch']+reserve_s<v['training_GPU_reservation_s'],'HOPPER_GPU_RESERVATION_EXHAUSTED'
    if str(root).startswith('/home/') and kind in ('query','episode'):
        p=root/'source/evidence/model_processes.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
        ended={r['pid'] for r in rows if r['event']=='end'}
        seconds=sum(r['elapsed_s'] for r in rows if r['event']=='end')+sum(time.time()-r['started_epoch'] for r in rows if r['event']=='start' and r['pid'] not in ended)
        assert seconds+reserve_s<v['evaluation_GPU_reservation_s'],'CORELAB_GPU_RESERVATION_EXHAUSTED'
    return v


def model_process(root,event,started=None):
    root=Path(root)
    if not str(root).startswith('/home/'):return None
    start=time.time() if event=='start' else started
    row=dict(event=event,pid=os.getpid(),started_epoch=start)
    if event=='end':row['elapsed_s']=time.time()-start
    p=root/'source/evidence/model_processes.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    return start
