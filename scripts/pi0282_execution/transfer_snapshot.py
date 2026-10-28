"""Reuse the verified ordered partial-readback relay; no local model files."""
import argparse,hashlib,inspect,json,shlex,subprocess,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'pi0276_execution'))
from coordinator import remote
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'pi0277_execution/r2'))
import importlib.util
_spec=importlib.util.spec_from_file_location("pi0277_byte_relay",Path(__file__).resolve().parent.parent/"pi0277_execution/r2/transfer.py")
_relay=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_relay)
prefix_hash=_relay.prefix_hash;OPTIONS=_relay.OPTIONS
def copy(point,source,destination):
    ready=json.loads(remote('hop',f'from pathlib import Path\nprint(Path({(source+"/READY.json")!r}).read_text())'))
    assert ready['point']==point and ready['status']=='LOCAL_IMMUTABLE_MODEL_ONLY_SNAPSHOT'
    for row in ready['files']:
        name=row['path'];assert Path(name).name==name
        dst=destination+'/'+name;src=source+'/'+name
        present=json.loads(remote('core',f'import json\nfrom pathlib import Path\np=Path({dst!r});print(json.dumps(p.is_file() and p.stat().st_size=={row["bytes"]}))'))
        if present:continue  # Full-file readback below closes any reused member.
        proof=json.loads(remote('core',f'''import fcntl,hashlib,json
from pathlib import Path
{inspect.getsource(prefix_hash)}
p=Path({dst!r});p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial')
with p.with_suffix(p.suffix+'.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 n=tmp.stat().st_size if tmp.exists() else 0;assert n<={row['bytes']}
 print(json.dumps(dict(bytes=n,sha256=prefix_hash(tmp,n) if n else hashlib.sha256().hexdigest())))
'''))
        offset=proof['bytes']
        source_prefix=remote('hop',f'import hashlib\nfrom pathlib import Path\n{inspect.getsource(prefix_hash)}\nprint(prefix_hash(Path({src!r}),{offset}))').strip() if offset else hashlib.sha256().hexdigest()
        assert source_prefix==proof['sha256'],'Existing partial differs from fixed source; do not append'
        print('VERIFIED_PARTIAL_RESUME',name,offset,flush=True)
        code=f'''import fcntl,hashlib,os,sys
from pathlib import Path
p=Path({dst!r});p.parent.mkdir(parents=True,exist_ok=True)
with p.with_suffix(p.suffix+'.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 h=hashlib.sha256();n={offset};tmp=p.with_suffix(p.suffix+'.partial')
 if n:
  assert tmp.stat().st_size==n
  with tmp.open('rb') as prior:
   for block in iter(lambda:prior.read(8<<20),b''):h.update(block)
  assert h.hexdigest()=={source_prefix!r}
 else:assert not tmp.exists() or tmp.stat().st_size==0
 with tmp.open('ab') as f:
  while True:
   block=sys.stdin.buffer.read(8<<20)
   if not block:break
   n+=len(block);assert n<={row['bytes']};h.update(block);f.write(block)
  f.flush();os.fsync(f.fileno())
 assert n=={row['bytes']} and h.hexdigest()=={row['sha256']!r}
 tmp.replace(p);p.chmod(0o444)
 print('TRANSFERRED',p.name,n,flush=True)
'''
        inner='python3 -c '+shlex.quote(code)
        outer=shlex.join(['ssh',*OPTIONS,'corelab-4080s',inner])
        producer=subprocess.Popen(['ssh',*OPTIONS,'nus-hopper','tail','-c','+'+str(offset+1),src],stdout=subprocess.PIPE)
        consumer=subprocess.Popen(['ssh',*OPTIONS,'corelab-amd-server',outer],stdin=producer.stdout)
        producer.stdout.close();rc=consumer.wait();sr=producer.wait();assert rc==sr==0,(name,rc,sr)
    script=f'''import hashlib,json,os
from pathlib import Path
root=Path({destination!r});ready={ready!r}
for r in ready['files']:
 p=root/r['path'];h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 assert p.stat().st_size==r['bytes'] and h.hexdigest()==r['sha256'],p
p=root/'READY.json';tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(ready,sort_keys=True)+chr(10));tmp.replace(p)
print(json.dumps(dict(status='PASS_DIRECT_LOCAL_SNAPSHOT_READBACK',point={point!r},model_only=True,archive='DEFERRED_USER_PRIORITY')))
'''
    print(remote('core',script),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('point',choices=('ACTION512','ACTION1024','ACTION2048'));a=p.parse_args()
    step=a.point.removeprefix('ACTION')
    copy(a.point,'/scratch/e1192548/Projects/go2-arx5/experiments/PI0282/learning/ACTION2048/snapshots/step-'+step,
        '/home/liuyu-nfs/Projects/go2-arx5/PI0282/incoming/'+a.point)
