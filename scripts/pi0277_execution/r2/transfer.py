"""Copy only a ready model snapshot through the existing SSH routes, no local weights."""
import argparse,hashlib,json,shlex,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'pi0276_execution'))
from coordinator import remote
HOP='/scratch/e1192548/Projects/go2-arx5/experiments/PI0277/R2'
RAW='/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0277/R2'
OPTIONS=['-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6']

def copy(point):
    assert point in [f'{a}{n}' for a in ('GRIP','FOCUS') for n in (512,1024,2048,4096)]
    source=HOP+'/snapshots/'+point
    ready=json.loads(remote('hop',f'from pathlib import Path\nprint(Path({(source+"/READY.json")!r}).read_text())'))
    assert ready['point']==point and ready['status']=='LOCAL_IMMUTABLE_MODEL_ONLY_SNAPSHOT'
    destination=RAW+'/models/'+point
    for row in ready['files']:
        name=row['path'];assert Path(name).name==name
        dst=destination+'/'+name;src=source+'/'+name
        present=json.loads(remote('core',f'import json\nfrom pathlib import Path\np=Path({dst!r});print(json.dumps(p.is_file() and p.stat().st_size=={row["bytes"]}))'))
        if present:continue  # Full-file readback below closes any reused member.
        code=f'''import fcntl,hashlib,os,sys
from pathlib import Path
p=Path({dst!r});p.parent.mkdir(parents=True,exist_ok=True)
with p.with_suffix(p.suffix+'.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 h=hashlib.sha256();n=0;tmp=p.with_suffix(p.suffix+'.partial')
 with tmp.open('wb') as f:
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
        producer=subprocess.Popen(['ssh',*OPTIONS,'nus-hopper','cat',src],stdout=subprocess.PIPE)
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
p=root/'READY.json';tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(ready,sort_keys=True)+'\\n');tmp.replace(p)
print(json.dumps(dict(status='PASS_DIRECT_LOCAL_SNAPSHOT_READBACK',point={point!r},model_only=True,archive='DEFERRED_USER_PRIORITY')))
'''
    print(remote('core',script),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('point');copy(p.parse_args().point)
