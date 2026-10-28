"""Read back owned numeric evidence; never copy private models or WBC logs."""
import base64
import hashlib
import json
from pathlib import Path
import sys

WORKTREE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(WORKTREE/'scripts/pi0287_target_sim'))
from owner import remote
ROOT='/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0289'

def main():
    payload=json.loads(remote(f'''from pathlib import Path
import base64,hashlib,json
r=Path({ROOT!r});files=[r/'evidence/BUDGET.json']
files += [r/'evidence'/n for n in ['SOURCE_POSTCHECK.json','GRIPPER_SELECTION.json','HF_ARCHIVE_RECEIPT.json'] if (r/'evidence'/n).exists()]
for v in [2,3,4]:
 files += [p for p in (r/f'startup-v{{v}}').glob('*.json')]
for e in json.loads((r/'evidence/BUDGET.json').read_text())['episodes']:
 p=Path(e['out'])
 files += [q for q in p.glob('*.json')]
 files += [q for q in (p/'cpu').glob('*') if q.suffix in ['.json','.jsonl']]
 files += [p/n for n in ['control.jsonl','initialization.jsonl','planner.jsonl'] if (p/n).exists()]
print(json.dumps({{str(p.relative_to(r)):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),data=base64.b64encode(p.read_bytes()).decode()) for p in files}}))
'''))
    out=WORKTREE/'reports/PI0289/execution';receipt=[]
    for name,item in payload.items():
        target=out/name;data=base64.b64decode(item['data'])
        assert hashlib.sha256(data).hexdigest()==item['sha256']
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        assert hashlib.sha256(target.read_bytes()).hexdigest()==item['sha256']
        receipt.append(dict(path=name,bytes=len(data),sha256=item['sha256']))
    (out/'READBACK.json').write_text(json.dumps(dict(status='PASS',files=receipt,scope='owned evidence only; full physics/ROS retained remotely and inventoried'),indent=2)+'\n')
    print('READBACK_PASS',len(receipt),sum(x['bytes'] for x in receipt))

if __name__=='__main__':main()
