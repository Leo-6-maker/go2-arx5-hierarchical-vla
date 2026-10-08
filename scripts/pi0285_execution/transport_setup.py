"""Scoped direct data delivery/model readback; private keys stay on Corelab."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote


def setup():
    hop='/scratch/e1192548/Projects/go2-arx5/experiments/PI0285'
    core='/home/liuyu-nfs/Projects/go2-arx5/PI0285'
    keys=json.loads(remote('core',f'''import json,os,subprocess
from pathlib import Path
r=Path({core!r});out={{}}
for name,comment in [('hopper_model_pull','PI0285_MODEL_READONLY'),('hopper_data_push','PI0285_DATA_WRITEONLY')]:
 p=r/'runtime'/name
 if not p.exists():subprocess.run(['ssh-keygen','-t','ed25519','-N','','-C',comment,'-f',str(p)],capture_output=True,check=True)
 assert p.stat().st_mode&0o777==0o600
 out[name]=p.with_suffix('.pub').read_text().strip()
print(json.dumps(out))
'''))
    identity=json.loads(remote('hop',f'''import json,os,fcntl,hashlib
from pathlib import Path
r=Path({hop!r});assert not (r/'runtime/TRAINING_ALLOCATION.json').exists(),'No credential change around an active lease'
for name in ('data','delivery'):(r/name).mkdir(exist_ok=True)
p=Path.home()/'.ssh/authorized_keys';assert p.is_file() and not p.is_symlink()
lines=[]
for name,flags,folder in [('hopper_model_pull','-ro','delivery'),('hopper_data_push','-wo -no-del','data')]:
 key={keys!r}[name]
 lines.append('restrict,command="/usr/bin/python3 '+str(r/'runtime/rrsync')+' '+flags+' '+str(r/folder)+'" '+key)
with p.open('a+') as f:
 fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);prior=f.read();added=[]
 for line in lines:
  key=line.split('ssh-ed25519 ',1)[1].split()[0]
  present=[v for v in prior.splitlines() if key in v]
  assert not present or present==[line],'Existing key must have exactly this scoped grant'
  if not present:f.seek(0,2);f.write(('' if prior.endswith('\\n') else '\\n')+line+'\\n');prior+='\\n'+line+'\\n';added.append(line.split()[-1])
 f.flush();os.fsync(f.fileno())
public=Path('/etc/ssh/ssh_host_ed25519_key.pub').read_text().strip().split()[:2]
print(json.dumps(dict(host_key=' '.join(public),added=added,scopes=['READ_ONLY_OWN_DELIVERY','WRITE_ONLY_NO_DELETE_OWN_DATA'],helper_sha256=hashlib.sha256((r/'runtime/rrsync').read_bytes()).hexdigest())))
'''))
    result=json.loads(remote('core',f'''import json,hashlib,shlex,subprocess,time
from pathlib import Path
r=Path({core!r});hosts=r/'runtime/hopper_known_hosts'
payload='hopper.nus.edu.sg '+{identity['host_key']!r}+'\\n'
if hosts.exists():assert hosts.read_text()==payload,'Authenticated host identity changed'
else:hosts.write_text(payload)
def ssh(key):return shlex.join(['ssh','-i',str(r/'runtime'/key),'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(hosts),'-o','ConnectTimeout=15'])
listed=subprocess.run(['rsync','--list-only','-e',ssh('hopper_model_pull'),'e1192548@hopper.nus.edu.sg:./'],capture_output=True,text=True);assert listed.returncode==0,(listed.stdout,listed.stderr)
probe=r/'runtime/TRANSPORT_PREFLIGHT.json';probe.write_text(json.dumps(dict(status='OWN_DATA_WRITEONLY_TRANSPORT_PROBE_NOT_DATA_ADMISSION',created=time.time()))+'\\n')
sent=subprocess.run(['rsync','-a','-e',ssh('hopper_data_push'),str(probe),'e1192548@hopper.nus.edu.sg:TRANSPORT_PREFLIGHT.json'],capture_output=True,text=True);assert sent.returncode==0,sent.stderr
fingerprints={{n:subprocess.check_output(['ssh-keygen','-lf',str(r/'runtime'/n)+'.pub'],text=True).strip() for n in {list(keys)!r}}}
v=dict(status='PASS_SCOPED_DIRECT_MODEL_LIST_AND_OWN_DATA_PUSH',scopes={identity['scopes']!r},fingerprints=fingerprints,private_keys_on_PC=False,private_key_locations=[str(r/'runtime'/n) for n in fingerprints],authenticated_host_key_sha256=hashlib.sha256(payload.encode()).hexdigest(),probe_sha256=hashlib.sha256(probe.read_bytes()).hexdigest(),HF_critical_path=False,new_model_queries=0,new_physics=0)
(r/'runtime/TRANSPORT_SETUP.json').write_text(json.dumps(v,indent=2)+'\\n');print(json.dumps(v))
'''))
    readback=remote('hop',f'''import hashlib,json
from pathlib import Path
p=Path({hop!r})/'data/TRANSPORT_PREFLIGHT.json'
assert hashlib.sha256(p.read_bytes()).hexdigest()=={result['probe_sha256']!r}
print(json.dumps(dict(readback_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),status='PASS_WRITE_PROBE_REMOTE_READBACK')))
''')
    result['remote_readback']=json.loads(readback)
    path=Path('reports/PI0285/preparation/TRANSPORT_SETUP.json');path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('private_key_locations','fingerprints')}))


if __name__=='__main__':setup()
