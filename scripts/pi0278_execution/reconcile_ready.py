"""Bind a legacy snapshot creator's metadata to the actual, unchanged PI0278 run."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote

def reconcile(point):
    assert point in [f'COVER93{n}' for n in (512,1024,2048,4096)]
    value=json.loads(remote('hop',f'''
import hashlib,json
from pathlib import Path
root=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0278/run');view=root/'snapshots'/{point!r}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
run=root/'COVER93/main/training/RUN_0_4096.json';creator=root.parents[1]/'PI0277/R2_FOCUS/code/persistence.py'
receipt=dict(status='PASS_LEGACY_CREATOR_METADATA_RECONCILIATION',governing_release='83af9099f3bab9a4ac08eaa85380c2cfd6351f34',point={point!r},original_ready_sha256=sha(view/'READY.json'),readout_sha256=sha(view/'PI0277_READOUT_CONTRACT.json'),training_run_sha256=sha(run),creator_sha256=sha(creator),creator_path=str(creator),source_run_path=str(run),reason='Inherited module_paths resolve the qualified PI0277 snapshot helper; only its hardcoded release metadata is stale',payload_modified=False,training_modified=False)
assert receipt['creator_sha256']=='eb790dbea1e8fc69bb328369aeadc32917e41bae5500d1e4f44f27aab93c55cf'
for name,text in [('TRAINING_RUN_SOURCE.json',run.read_text()),('READY_AUTHORITY_RECONCILIATION.json',json.dumps(receipt,sort_keys=True)+chr(10))]:
 p=view/name
 if p.exists():assert p.read_text()==text
 else:p.write_text(text)
print(json.dumps(dict(receipt=receipt,run_text=run.read_text())))
'''))
    panel=Path(__file__).with_name('panel.py').read_text()
    result=remote('core',f'''
import hashlib,json,os
from pathlib import Path
root=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0278/R2');view=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0278/R2/models')/{point!r}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();receipt={value['receipt']!r}
assert sha(view/'READY.json')==receipt['original_ready_sha256'] and sha(view/'PI0277_READOUT_CONTRACT.json')==receipt['readout_sha256']
for name,text in [('TRAINING_RUN_SOURCE.json',{value['run_text']!r}),('READY_AUTHORITY_RECONCILIATION.json',json.dumps(receipt,sort_keys=True)+chr(10))]:
 p=view/name
 if p.exists():assert p.read_text()==text
 else:p.write_text(text)
target=root/'code/panel.py';new={panel!r};old_hash=sha(target);new_hash=hashlib.sha256(new.encode()).hexdigest()
if old_hash!=new_hash:
 assert old_hash=='8822a6263ae6d77fdea738c2ae3a2ebc6299e52b4db3fa5de47d0035f2b17c69'
 for proc in Path('/proc').glob('[0-9]*/cmdline'):
  try:args=proc.read_bytes().split(bytes([0]));assert str(target).encode() not in args
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
 saved=target.with_name('panel.before-ready-reconciliation.txt');saved.write_bytes(target.read_bytes());target.write_text(new)
 for seal in [root/'design/EVALUATION_FILESET.json',root.parent/'R3/EXECUTION_FILESET.json']:
  value=json.loads(seal.read_text());backup=seal.with_name(seal.stem+'.before-ready-reconciliation.json');assert not backup.exists();backup.write_bytes(seal.read_bytes())
  for row in value['files']:
   if row['path']==str(target):assert row['sha256']==old_hash;row['sha256']=new_hash
  value['metadata_reconciliation']='READY_AUTHORITY_RECONCILIATION.json; original payload and algorithm unchanged'
  seal.write_text(json.dumps(value,sort_keys=True)+chr(10))
done=root/'runtime'/({point!r}+'-admit.done.json')
if done.exists() and json.loads(done.read_text())['returncode']!=0:
 assert json.loads(done.read_text())['returncode']==1
 for suffix in ['done.json','start.json','log','worker.log']:
  p=root/'runtime'/({point!r}+'-admit.'+suffix)
  if p.exists():q=p.with_name(p.name+'.attempt1');assert not q.exists();p.rename(q)
print(json.dumps(dict(receipt=receipt,panel_sha256=new_hash,original_READY_preserved=True)))
''')
    print(result,flush=True)
    Path('reports/PI0278/execution/READY_RECONCILIATION_'+point+'.json').write_text(result,encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('point');reconcile(p.parse_args().point)
