"""Read retained source ledgers and layout manifests before new confirmation exposure."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote
CODE=r"""
import hashlib,json,os,re,time
from pathlib import Path
root=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0286');base=root.parent
rx=re.compile(r'(?<!\d)(2855(?:[0-2]\d|3[01]))(?!\d)')
roots={base,Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5')};tiers=[]
for p in base.glob('PI*/design/RECORDING_TIER.json'):
 v=json.loads(p.read_text());tiers.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 if v.get('record_root'):roots.add(Path(v['record_root']))
checked=[];hits=[];seen=set();dirs=[];errors=[]
for tree in sorted(roots):
 if not tree.exists():continue
 def error(e):errors.append(str(e))
 for parent,ds,fs in os.walk(tree,onerror=error,followlinks=False):
  d=Path(parent)
  if d==root or 'PI0286' in d.parts:ds[:]=[];continue
  ds[:]=[n for n in ds if n not in {'.git','__pycache__','site-packages','node_modules','venv','hf_home','hub','models','learning','action_target','policy_observation','transfers','delivery','pristine','cache'} and not n.startswith('checkpoint-')]
  if rx.search(str(d)):hits.append(dict(path=str(d),kind='SOURCE_DIRECTORY_ID',ids=rx.findall(str(d))))
  for n in fs:
   p=d/n
   if p in seen or p.suffix not in ('.json','.jsonl','.csv','.tsv'):continue
   low=n.lower()
   if not (n in ('START.json','EPISODE_CARD.json','RESULT.json','initial.json') or any(s in low for s in ('budget','ledger','exposure','layout','condition','episode','panel'))):continue
   seen.add(p)
   try:
    h=hashlib.sha256();found=set();size=0
    with p.open('rb') as f:
     for line in f:
      size+=len(line);h.update(line);found.update(rx.findall(line.decode('utf8',errors='replace')))
    checked.append(dict(path=str(p),bytes=size,sha256=h.hexdigest()))
    if found:hits.append(dict(path=str(p),kind='RETAINED_LEDGER_OR_LAYOUT_CONTENT',ids=sorted(found)))
   except OSError as e:errors.append(str(e))
out=dict(status='PASS_ALL_RETAINED_PROJECT_LEDGER_AND_SOURCE_START_SCAN' if not hits and not errors else 'BLOCKED_NEW_CONFIRM_SET',epoch=time.time(),new_ids=[285500,285531],roots=[str(p) for p in sorted(roots)],recording_tiers=tiers,checked=checked,hits=hits,errors=errors,new_model_queries=0,new_render=0,new_physics=0,scope='ALL_RETAINED_PROJECT_SOURCE_DIRECTORIES_AND_LAYOUT_EXPOSURE_LEDGERS; metadata-only proposed future IDs are not executed exposure')
p=root/'design/EXPOSURE.json';p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
"""
if __name__=='__main__':
 v=json.loads(remote('core',CODE));p=Path('reports/PI0286/adoption/EXPOSURE.json');p.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({k:v[k] for k in ('status','epoch','hits','errors')}));print('files',len(v['checked']))
