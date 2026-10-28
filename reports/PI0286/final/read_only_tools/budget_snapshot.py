"""Finite read-only ledger prefix snapshot; no live/runtime edits or model calls."""
import sys,json,time,concurrent.futures
from pathlib import Path
sys.path.insert(0,'scripts/pi0276_execution');from coordinator import remote
SCRIPT=r'''
import os,json,time,hashlib,collections
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='4'
r=Path(ROOT);now=time.time();seen={};out={}
def snap(p):
 with p.open('rb') as f:n=os.fstat(f.fileno()).st_size;b=f.read(n)
 complete=b.rsplit(b'\n',1)[0]+b'\n' if b and not b.endswith(b'\n') else b
 rows=[json.loads(x) for x in complete.splitlines() if x.strip()]
 return rows,dict(path=str(p),realpath=str(p.resolve()),captured_bytes=len(b),complete_prefix_bytes=len(complete),discarded_incomplete_tail_bytes=len(b)-len(complete),prefix_sha256=hashlib.sha256(complete).hexdigest(),lines=len(rows),snapshot_epoch=now)
for rel in ('runtime/quality_budget.jsonl','source/evidence/quality_budget.jsonl','source/evidence/budget.jsonl'):
 p=r/rel
 if not p.exists():continue
 st=p.stat();key=(st.st_dev,st.st_ino)
 if key in seen:out[rel]=dict(alias_of=seen[key],not_counted_twice=True);continue
 seen[key]=rel;rows,identity=snap(p);counts=collections.Counter(x['kind'] for x in rows);mx={k:max(x['ordinal'] for x in rows if x['kind']==k) for k in counts}
 out[rel]=dict(identity=identity,counts=dict(counts),max_ordinal=mx,contiguous_ordinals={k:sorted(x['ordinal'] for x in rows if x['kind']==k)==list(range(1,counts[k]+1)) for k in counts},last_wall_ns=max((x.get('wall_ns',x.get('time_ns',0)) for x in rows),default=None),raw_start_slots=[x['slot'] for x in rows if x['kind']=='starts'])
p=r/'runtime/GPU_PROCESSES.jsonl';rows,identity=snap(p);ends={(x['pid'],x['start_epoch']):x['epoch'] for x in rows if x['event']=='end'};intervals=[];active=[]
for x in rows:
 if x['event']!='start':continue
 k=(x['pid'],x['start_epoch']);stop=ends.get(k,now);assert stop>=x['start_epoch'];intervals.append([x['start_epoch'],stop])
 if k not in ends:active.append(x)
def union(xs):
 total=0.;end=float('-inf')
 for a,b in sorted(xs):total+=max(0.,b-max(a,end));end=max(end,b)
 return total
closed=[(x['start_epoch'],x['epoch']) for x in rows if x['event']=='end'];out['GPU_process_intervals']=dict(identity=identity,rows=rows,closed_interval_union_seconds=union(closed),including_live_to_snapshot_union_seconds=union(intervals),interval_count=len(intervals),active_starts=active,interpretation='Reservation/service interval union, not measured GPU-compute utilization. Never sum nested model child phases again within same device lease.')
p=r/'learning/completed_updates.jsonl'
if p.exists():
 rows,identity=snap(p);out['completed_updates']=dict(identity=identity,counts={'/'.join(k):v for k,v in collections.Counter((x['arm'],x['phase'],x['work_kind']) for x in rows).items()},rows=len(rows))
out['resource_caps']=json.loads((r/'design/RESOURCE_RESERVATIONS.json').read_text());print(json.dumps(dict(host=HOST,status='READ_ONLY_LIVE_LEDGER_COMPLETE_PREFIX_SNAPSHOT_NOT_FINAL',snapshot_epoch=now,ledgers=out,model=0,physics=0,optimizer=0,codec=0)))
'''
def f(h):
 root='/home/liuyu-nfs/Projects/go2-arx5/PI0286' if h=='core' else '/scratch/e1192548/Projects/go2-arx5/experiments/PI0286'
 v=json.loads(remote(h,'ROOT='+repr(root)+'\nHOST='+repr(h)+'\n'+SCRIPT));p=Path('reports/PI0286/final/PI0286_BUDGET_SNAPSHOT_'+h.upper()+'.json');p.write_text(json.dumps(v,indent=2)+'\n');return v
if __name__=='__main__':
 vals=list(concurrent.futures.ThreadPoolExecutor(max_workers=2).map(f,('core','hop')))
 print(json.dumps([dict(host=x['host'],snapshot=x['snapshot_epoch'],quality=x['ledgers'].get('runtime/quality_budget.jsonl',{}).get('counts'),raw=x['ledgers'].get('source/evidence/budget.jsonl',{}).get('counts'),GPU_union=x['ledgers']['GPU_process_intervals']['including_live_to_snapshot_union_seconds'],active=x['ledgers']['GPU_process_intervals']['active_starts'],updates=x['ledgers'].get('completed_updates',{}).get('counts')) for x in vals]))
