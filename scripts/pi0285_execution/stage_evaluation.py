"""Update future evaluator only after the original bank CPU process closes."""
import base64,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote


def stage():
    here=Path(__file__).resolve().parent
    names=('budget.py','policy.py','model_worker.py','audit_source.py','qualification.py','diagnostics.py','transport.py','evaluate.py')
    files={n:base64.b64encode((here/n).read_bytes()).decode() for n in names}
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    value=json.loads(remote('core',f'''import base64,hashlib,json,subprocess,time
from pathlib import Path
r=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0285');admission=r/'runtime/R2_ADMISSION_COMPLETE.json'
assert json.loads(admission.read_text())['status']=='PASS_SUPERVISED_TRAIN_AND_INPUTS_WITH_DISCLOSED_DEV_EXTRAPOLATION'
# A completed receipt alone does not prove the process has exited.
active=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:b=p.read_bytes()
 except (OSError,PermissionError):continue
 if str(r/'code/').encode() in b or str(r/'runtime/admit.py').encode() in b:active.append(dict(pid=int(p.parent.name),argv=b.decode(errors='replace').replace('\\0',' ')))
assert not active,('Keep healthy owner/code intact',active)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old=r/'design/CODE_SEAL.json';preserved=r/'design/CODE_SEAL_BANK.json'
if not preserved.exists():preserved.write_bytes(old.read_bytes())
seal=json.loads(old.read_text());updated={{}}
for name,value in {files!r}.items():
 p=r/'code'/name;p.write_bytes(base64.b64decode(value));updated[name]=sha(p)
seal['files'].update(updated);seal.update(published_commit={commit!r},bank_seal_sha256=sha(preserved),reason='CLOSED_ACTUAL_BANK_CPU_TO_FIXED_NATIVE_EVALUATION_ONLY')
old.write_text(json.dumps(seal,indent=2)+'\\n')
proof=dict(status='PASS_CLOSED_BANK_CPU_FUTURE_EVALUATOR_STAGE_NOT_MODEL_EVALUATION',published_commit={commit!r},bank_admission_sha256=sha(admission),preserved_bank_code_seal_sha256=sha(preserved),updated_files=updated,new_model_queries=0,new_physics=0)
(r/'evidence/EVALUATOR_STAGE.json').write_text(json.dumps(proof,indent=2)+'\\n');print(json.dumps(proof))
'''))
    p=Path('reports/PI0285/preparation/EVALUATOR_STAGE.json');p.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8');print(json.dumps(value))


if __name__=='__main__':stage()
