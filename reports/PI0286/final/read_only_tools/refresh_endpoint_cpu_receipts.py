"""Refresh comparison from CLOSED small CPU/source receipts only; no array/model read."""
import json,sys,hashlib,time,concurrent.futures
from pathlib import Path
sys.path.insert(0,'scripts/pi0276_execution');from coordinator import remote
OUT=Path('reports/PI0286/final');POINTS=['C0_REPLAY1024-1024','C1_CORRECT1024-1024']
CORE=r'''
import json,hashlib,time
from pathlib import Path
r=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0286');out={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for point in POINTS:
 panels={}
 for name,lo,n in [('TRAIN',285100,8),('DEV',285200,16),('CONFIRM_C',285500,32)]:
  source=r/'source/evidence'/f'{point}_{name}.json';cpu=r/'audit'/f'{point}_{name}_CPU.json'
  if not source.exists() or not cpu.exists():
   panels[name]=dict(status='UNOBSERVED_INDEPENDENT_CPU_CLOSED',source_exists=source.exists(),cpu_exists=cpu.exists());continue
  v=json.loads(source.read_text());c=json.loads(cpu.read_text())
  assert c['status']=='PASS_INDEPENDENT_CPU_CONTACT_SCORE_CLOCK_INSTALLATION_AND_PAIRED_INITIAL_INPUTS'
  assert c['source_sha256']==sha(source) and len(v['rows'])==len(c['rows'])==n
  assert sorted(x['layout'] for x in v['rows'])==list(range(lo,lo+n))
  assert [(x['task'],x['layout'],x['persistent_success']) for x in c['rows']]==[(x['task'],x['layout'],x['persistent_success']) for x in v['rows']]
  assert all(x['task']=='T1' and x['point']==point and x['split']==name for x in v['rows'])
  count=sum(bool(x['persistent_success']) for x in c['rows'])
  panels[name]=dict(status='ACTUAL_CLOSED_INDEPENDENT_CPU_PASS',requested_denominator=n,success=count,source_path=str(source),source_sha256=sha(source),CPU_path=str(cpu),CPU_sha256=sha(cpu),scorer_and_actual_input_install_checks=c['status'],statuses={s:sum(x['status']==s for x in c['rows']) for s in set(x['status'] for x in c['rows'])},failure_layouts=[x['layout'] for x in c['rows'] if not x['persistent_success']],CPU_receipt=c)
 out[point]=panels
print(json.dumps(dict(epoch=time.time(),panels=out)))
'''
HOP=r'''
import json,hashlib,time
from pathlib import Path
r=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0286');out={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for point in POINTS:
 arm=point.rsplit('-',1)[0];out[point]={}
 for kind,p in [('canary',r/'learning'/arm/'CANARY_COMPARISON.json'),('training',r/'evidence'/f'{point}_TRAINING_CPU.json')]:
  if p.exists():out[point][kind]=dict(path=str(p),sha256=sha(p),receipt=json.loads(p.read_text()))
print(json.dumps(dict(epoch=time.time(),arms=out)))
'''
def fetch(host,code):return json.loads(remote(host,'POINTS='+repr(POINTS)+'\n'+code))
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  a=pool.submit(fetch,'core',CORE);b=pool.submit(fetch,'hop',HOP);core=a.result();hop=b.result()
 for host,v in [('CORE',core),('HOP',hop)]:
  d=OUT/'endpoint_cpu_receipts';d.mkdir(exist_ok=True);(d/(host+'_SMALL_RECEIPTS.json')).write_text(json.dumps(v,indent=2)+'\n')
 p=OUT/'TWO_ARM_ENDPOINT_COMPARISON.json';v=json.loads(p.read_text());v['receipt_refresh_epoch']=time.time()
 for arm in v['arms']:
  point=arm['point'];h=hop['arms'][point]
  if 'canary' in h:
   arm['canary_status']=h['canary']['receipt']['status'];arm['canary_receipt_identity']={k:h['canary'][k] for k in ('path','sha256')}
  if 'training' in h:
   t=h['training'];assert t['receipt']['status']=='PASS_ACTUAL_NATIVE_TRAINER_INPUT_PLAN_NOISE_LOSS_MODES_SCOPE_AND_COMPLETE_LOCAL_STATE';arm['training_integrity_status']=t['receipt']['status'];arm['training_CPU_path']=t['path'];arm['training_CPU_sha256']=t['sha256'];arm['formal_updates']=1024
  for name,c in core['panels'][point].items():
   if c['status']=='ACTUAL_CLOSED_INDEPENDENT_CPU_PASS':arm['panels'][name].update({k:x for k,x in c.items() if k!='CPU_receipt'})
   else:arm['panels'][name]['source_exists']=c['source_exists'];arm['panels'][name]['CPU_exists']=c['cpu_exists']
 p.write_text(json.dumps(v,indent=2)+'\n')
 print(json.dumps({a['arm']:{n:{'status':x['status'],'success':x['success']} for n,x in a['panels'].items()} for a in v['arms']}))
