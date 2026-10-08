"""Adopt explicit R2 at dead-process boundaries, preserving the original failed seal."""
import base64,hashlib,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote


def stage():
    here=Path(__file__).resolve().parent
    names=('r2.py','authority.py','budget.py','launch.py','train.py','training_audit.py','policy.py','diagnostics.py','qualification.py','source.py','evaluate.py','transport.py','model_worker.py','audit_source.py')
    code={n:base64.b64encode((here/n).read_bytes()).decode() for n in names}
    docs={'design/R2_PROTOCOL.md':Path('docs/protocols/PI0285_R2_1000_SCOPED_ADMISSION_AND_POLICY_DELIVERY_V1.md').read_bytes(),'design/R2_DEADLINE_SCOPE_AND_LIMITS.json':Path('reports/PI0285/design/R2_DEADLINE_SCOPE_AND_LIMITS.json').read_bytes()}
    proof=dict(status='PASS_FIXED_RELEASE_FILES_AND_CANONICAL_READBACK',release='97773a036ebaffe59285bd2ecbee0c155f12822b',canonical=6065031954,files={n:hashlib.sha256(b).hexdigest() for n,b in docs.items()},parent_release='7e840324c71aef47b401bd861b22710774334c97',same_goal_cumulative_caps=True)
    document_payload={n:base64.b64encode(b).decode() for n,b in docs.items()}
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    result=json.loads(remote('core',f'''import base64,hashlib,json,os,subprocess,time
from pathlib import Path
r=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0285');active=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:b=p.read_bytes()
 except (OSError,PermissionError):continue
 if str(r/'code/').encode() in b or str(r/'runtime/native_input_check').encode() in b or str(r/'runtime/admit.py').encode() in b:active.append(int(p.parent.name))
assert not active,('Adopt healthy processes before changing their code',active)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old=r/'design/CODE_SEAL.json';bank=r/'design/CODE_SEAL_BANK.json'
if not bank.exists():bank.write_bytes(old.read_bytes())
strict=r/'runtime/original-codec-checker.py'
if not strict.exists():strict.write_bytes((r/'code/codec_check.py').read_bytes())
assert sha(strict)==json.loads(bank.read_text())['files']['codec_check.py']
for n,b in {document_payload!r}.items():(r/n).write_bytes(base64.b64decode(b))
(r/'design/R2_PUBLICATION_READBACK.json').write_text(json.dumps({proof!r},indent=2)+'\\n')
seal=json.loads(old.read_text())
for n,b in {code!r}.items():
 (r/'code'/n).write_bytes(base64.b64decode(b));seal['files'][n]=sha(r/'code'/n)
seal.update(published_commit={commit!r},R2_release='97773a036ebaffe59285bd2ecbee0c155f12822b',R2_canonical=6065031954,preserved_bank_seal_sha256=sha(bank),reason='EXPLICIT_R2_SCOPED_ADMISSION_PHASE_DEADLINES_AND_FIXED_EARLY_FEEDBACK')
old.write_text(json.dumps(seal,indent=2)+'\\n')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',OPENBLAS_CORETYPE='Prescott',PI0285_ROOT=str(r),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
env['PYTHONPATH']=':'.join([str(r/'code'),str(r.parent/'PI0283/code'),str(r.parent/'PI0280/official'),str(r/'project-src'),str(r/'project-src/src')])
with (r/'runtime/R2_SCOPED_ADMISSION_CPU.log').open('ab') as log:
 started=time.time();q=subprocess.run(['/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/venv/bin/python','-u',str(r/'code/r2.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
assert q.returncode==0,('Inspect retained actual R2 CPU log; do not repeat full data pipeline',q.returncode)
v=dict(status='PASS_REAL_R2_SCOPED_SOURCE_ADMISSION_OLD_FAIL_PRESERVED',published_commit={commit!r},publication={proof!r},scoped_receipt_sha256=sha(r/'data/R2_SCOPED_DATA_ADMISSION.json'),receipt=json.loads((r/'data/R2_SCOPED_DATA_ADMISSION.json').read_text()),elapsed_s=time.time()-started,new_model_queries=0,new_optimizer_updates=0,new_physics=0)
(r/'evidence/R2_ADOPTION.json').write_text(json.dumps(v,indent=2)+'\\n');print(json.dumps(v))
'''))
    # Publication files arrive before the Hopper CPU entry imports its R2 readers.
    hop=remote('hop',f'''import base64,json
from pathlib import Path
r=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0285')
assert not (r/'runtime/JOB_SUBMISSION.json').exists() and not (r/'runtime/TRAINING_ALLOCATION.json').exists(),'Adopt accepted healthy work'
for n,b in {document_payload!r}.items():(r/n).write_bytes(base64.b64decode(b))
(r/'design/R2_PUBLICATION_READBACK.json').write_text(json.dumps({proof!r},indent=2)+'\\n')
print('R2_AUTHORITY_FILES_READY_NO_GPU_SUBMISSION')
''')
    out=Path('reports/PI0285/execution/R2_ADOPTION.json');out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='receipt'}));print(hop)


if __name__=='__main__':stage()
