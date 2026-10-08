"""Own PI0285 paths and immutable publication contract."""
import hashlib,json,os
from pathlib import Path

ROOT=Path(os.environ.get('PI0285_ROOT','/home/liuyu-nfs/Projects/go2-arx5/PI0285'))
RELEASE='7e840324c71aef47b401bd861b22710774334c97'
CANONICAL=6058329368
INSTRUCTIONS={
 'T1':'Pick up the red cube, place it inside the blue square, release it, and move the gripper away.',
 'T2':'Stack the red cube on the green cube, release it, and move the gripper away.'}
SPLITS={'PILOT':(285000,285007),'TRAIN':(285100,285163),'DEV':(285200,285215),'CONFIRM_A':(285300,285331),'CONFIRM_B':(285400,285431)}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def guard():
    v=json.loads((ROOT/'design/PUBLICATION_READBACK.json').read_text())
    assert v['release']==RELEASE and v['canonical']==CANONICAL and v['status'].startswith('PASS_')
    seal=json.loads((ROOT/'design/CODE_SEAL.json').read_text())
    for name,h in seal['files'].items():assert sha(ROOT/'code'/name)==h,name
    return json.loads((ROOT/'design/PLAN_AND_LIMITS.json').read_text())
