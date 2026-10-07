"""Owner reservations shared by finite workers; actual event journals stay in each lane."""
import fcntl,hashlib,json,os,time
from pathlib import Path

ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0281')
RELEASE='360b412f55be8aec1568054d7d039ce7de2bc597'
CAPS=dict(policy_requests=640,policy_heads=2560,physical_starts=36,technical_replacements=4,
          wbc_controls=20000,wbc_actor_forwards=80000,integrations=150000,dispatches=100000,RGB_pairs=20000)

def guard():
    assert json.loads((ROOT.parent/'PI0279/runtime/SCIENTIFIC_PAUSE.json').read_text())['launch_allowed'] is False
    seal=json.loads((ROOT/'design/IMPLEMENTATION.json').read_text())
    assert seal['release']==RELEASE and seal['published_commit'] and seal['canonical']==6040516315
    for name,digest in seal['files'].items():
        assert hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest()==digest,name
    return seal

def reserve(lane,uid,counts):
    guard();assert lane in ('SOURCE','WBC') and uid and counts
    assert all(k in CAPS and isinstance(v,int) and 0<=v<=CAPS[k] for k,v in counts.items())
    path=ROOT/'evidence/RESERVATIONS.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.seek(0)
        rows=[json.loads(line) for line in f if line.strip()]
        prior=[r for r in rows if r['lane']==lane and r['uid']==uid]
        if prior:
            assert len(prior)==1 and prior[0]['counts']==counts
            return prior[0]
        total={k:sum(r['counts'].get(k,0) for r in rows)+counts.get(k,0) for k in CAPS}
        assert all(total[k]<=CAPS[k] for k in CAPS),total
        row=dict(lane=lane,uid=uid,counts=counts,time_ns=time.time_ns(),pid=os.getpid())
        f.seek(0,2);f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
        return row

def actual(lane,uid,counts,status):
    path=ROOT/'evidence/ACTUAL.jsonl'
    reservations=[json.loads(x) for x in (ROOT/'evidence/RESERVATIONS.jsonl').read_text().splitlines()]
    bound=next(r['counts'] for r in reservations if r['lane']==lane and r['uid']==uid)
    assert all(isinstance(v,int) and 0<=v<=bound.get(k,0) for k,v in counts.items())
    with path.open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);rows=[json.loads(x) for x in f if x.strip()]
        prior=[r for r in rows if r['lane']==lane and r['uid']==uid]
        row=dict(lane=lane,uid=uid,counts=counts,status=status)
        if prior:assert prior==[row];return
        f.seek(0,2);f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())

def self_check():
    import tempfile
    global ROOT,guard
    old,old_guard=ROOT,guard
    with tempfile.TemporaryDirectory() as d:
        ROOT=Path(d);guard=lambda:None
        trial=dict(policy_requests=60,policy_heads=240,physical_starts=1)
        reserve('SOURCE','fixture',trial);reserve('SOURCE','fixture',trial)
        assert len((ROOT/'evidence/RESERVATIONS.jsonl').read_text().splitlines())==1
        try:reserve('SOURCE','over_cap',dict(policy_requests=641))
        except AssertionError:pass
        else:raise AssertionError('cap not enforced')
        try:actual('SOURCE','fixture',dict(policy_requests=61),'FAIL')
        except AssertionError:pass
        else:raise AssertionError('actual exceeded reservation')
        actual('SOURCE','fixture',dict(policy_requests=3),'PASS')
        actual('SOURCE','fixture',dict(policy_requests=3),'PASS')
        assert len((ROOT/'evidence/ACTUAL.jsonl').read_text().splitlines())==1
    ROOT,guard=old,old_guard

if __name__=='__main__':self_check();print('BUDGET_ARITHMETIC_CHECK_PASS')
