"""Grant only read-only access to PI0286 model delivery; key stays on Core."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote


def setup():
    core='/home/liuyu-nfs/Projects/go2-arx5/PI0286'
    hop='/scratch/e1192548/Projects/go2-arx5/experiments/PI0286'
    public=json.loads(remote('core',f'''import json,subprocess
from pathlib import Path
r=Path({core!r});p=r/'runtime/hopper_model_pull';p.parent.mkdir(exist_ok=True)
if not p.exists():subprocess.run(['ssh-keygen','-t','ed25519','-N','','-C','PI0286_MODEL_READONLY','-f',str(p)],capture_output=True,check=True)
assert not p.is_symlink() and p.stat().st_mode&0o777==0o600
print(json.dumps(dict(key=p.with_suffix('.pub').read_text().strip())))
'''))
    identity=json.loads(remote('hop',f'''import json,fcntl,hashlib,os
from pathlib import Path
r=Path({hop!r});(r/'delivery').mkdir(exist_ok=True);(r/'runtime').mkdir(exist_ok=True)
old=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0285/runtime/rrsync');helper=r/'runtime/rrsync'
assert old.is_file() and not old.is_symlink()
if helper.exists():assert helper.read_bytes()==old.read_bytes()
else:helper.write_bytes(old.read_bytes());helper.chmod(0o700)
p=Path.home()/'.ssh/authorized_keys';assert p.is_file() and not p.is_symlink()
key={public['key']!r};token=key.split()[1]
line='restrict,command="/usr/bin/python3 '+str(helper)+' -ro '+str(r/'delivery')+'" '+key
with p.open('a+') as f:
 fcntl.flock(f,fcntl.LOCK_EX);f.seek(0);prior=f.read()
 present=[v for v in prior.splitlines() if token in v]
 assert not present or present==[line],'Existing key scope mismatch'
 added=not present
 if added:f.seek(0,2);f.write(('' if not prior or prior.endswith(chr(10)) else chr(10))+line+chr(10))
 f.flush();os.fsync(f.fileno())
host=' '.join(Path('/etc/ssh/ssh_host_ed25519_key.pub').read_text().split()[:2])
print(json.dumps(dict(host_key=host,grant_added=added,helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest())))
'''))
    result=json.loads(remote('core',f'''import hashlib,json,shlex,subprocess
from pathlib import Path
r=Path({core!r});hosts=r/'runtime/hopper_known_hosts';payload='hopper.nus.edu.sg '+{identity['host_key']!r}+chr(10)
if hosts.exists():assert hosts.read_text()==payload,'Authenticated host identity changed'
else:hosts.write_text(payload)
ssh=shlex.join(['ssh','-i',str(r/'runtime/hopper_model_pull'),'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(hosts),'-o','ConnectTimeout=15'])
v=subprocess.run(['rsync','--list-only','-e',ssh,'e1192548@hopper.nus.edu.sg:./'],capture_output=True,text=True)
assert v.returncode==0,v.stderr
out=dict(status='PASS_PI0286_OWN_DELIVERY_READ_ONLY_LIST',scope='READ_ONLY_OWN_PI0286_DELIVERY',grant_added={identity['grant_added']!r},helper_sha256={identity['helper_sha256']!r},authenticated_host_key_sha256=hashlib.sha256(payload.encode()).hexdigest(),private_keys_on_PC=False,data_write_grant_added=False,model_queries=0,physics=0)
(r/'runtime/TRANSPORT_SETUP.json').write_text(json.dumps(out,indent=2)+chr(10));print(json.dumps(out))
'''))
    path=Path('reports/PI0286/preparation/TRANSPORT_SETUP.json');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    return result


if __name__=='__main__':print(json.dumps(setup()))
