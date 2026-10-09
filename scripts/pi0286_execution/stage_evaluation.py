"""Stage only the newly isolated evaluator; never change archived PI0285 code."""
import base64,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote


def stage():
    here=Path(__file__).resolve().parent
    names=('policy.py','model_worker.py','evaluate.py','qualification.py','diagnostics.py','transport.py')
    payload={n:base64.b64encode((here/n).read_bytes()).decode() for n in names}
    code=f'''import base64,hashlib,json,os
from pathlib import Path
r=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0286');assert (r/'design/PUBLICATION_READBACK.json').is_file()
active=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:b=p.read_bytes()
 except (OSError,PermissionError):continue
 if str(r/'code/').encode() in b:active.append(int(p.parent.name))
assert not active,('Never change a healthy new evaluator',active)
sealpath=r/'design/CODE_SEAL.json';seal=json.loads(sealpath.read_text());updated={{}}
for name,value in {payload!r}.items():
 p=r/'code'/name;p.write_bytes(base64.b64decode(value));updated[name]=hashlib.sha256(p.read_bytes()).hexdigest()
seal['files'].update(updated);sealpath.write_text(json.dumps(seal,indent=2)+'\\n')
out=dict(status='PI0286_ISOLATED_EVALUATOR_STAGED_NO_MODEL_OR_PHYSICS_LAUNCH',files=updated,archived_PI0285_modified=False,model_queries=0,physics=0)
(r/'evidence/EVALUATOR_STAGE.json').write_text(json.dumps(out,indent=2)+'\\n');print(json.dumps(out))
'''
    result=json.loads(remote('core',code));path=Path('reports/PI0286/preparation/EVALUATOR_STAGE.json');path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':print(json.dumps(stage()))
