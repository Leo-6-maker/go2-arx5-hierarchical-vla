"""Mirror terminal journals only, then run the existing CPU training audit."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote

def audit():
    value=json.loads(remote('hop',r'''
import json,hashlib
from pathlib import Path
root=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0278/run');t=root/'COVER93/main/training'
complete=json.loads((t/'SEGMENT_COMPLETE.json').read_text());assert complete['status']=='PASS_EXECUTED_SEGMENT' and (root/'evidence/TRAINING_COMPLETE.json').is_file()
names=['SEGMENT_COMPLETE.json','sample_audit.jsonl','step_rng.jsonl','S4_audit.jsonl','gripper_objective.jsonl','full_loss.jsonl','box_objective.jsonl','update_gradient_norms.jsonl','TIMINGS.jsonl','SAVE_TIMINGS.jsonl','PARAMETER_SCOPE.json','CURSOR.json']
files={('COVER93/main/training/'+name):(t/name).read_text() for name in names}
for name in ['design/TRAIN_PLAN.json','design/INPUT_BINDING.json']:files[name]=(root/name).read_text()
print(json.dumps(dict(files=files,source_root=str(root),completed=complete)))
'''))
    result=remote('core',f'''
import json,subprocess,os,hashlib
from pathlib import Path
base=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0278');root=base/'audit-training';files={value['files']!r}
for name,text in files.items():
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_text()==text
 else:p.write_text(text)
p=root/'design/ELIGIBLE_MEMBERSHIP.json';source=base/'design/ELIGIBLE_MEMBERSHIP.json'
if p.exists():assert p.read_bytes()==source.read_bytes()
else:p.write_bytes(source.read_bytes())
(root/'evidence').mkdir(exist_ok=True)
receipt=dict(source_root={value['source_root']!r},completed={value['completed']!r},metadata_only=True,model_copied=False,files=[dict(path=n,sha256=hashlib.sha256(s.encode()).hexdigest(),bytes=len(s.encode())) for n,s in files.items()])
(root/'evidence/TERMINAL_JOURNAL_MIRROR.json').write_text(json.dumps(receipt,sort_keys=True)+chr(10))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
p=subprocess.run(['/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/venv/bin/python',str(base/'code/audit_training.py'),'--root',str(root)],env=env,capture_output=True,text=True);print(p.stdout+p.stderr);assert p.returncode==0
''')
    print(result,flush=True)
    for name in ['COVER93_ACTUAL_TRAINING_AUDIT.json','TERMINAL_JOURNAL_MIRROR.json']:
        text=remote('core',f"from pathlib import Path\nprint(Path('/home/liuyu-nfs/Projects/go2-arx5/PI0278/audit-training/evidence/{name}').read_text())")
        Path('reports/PI0278/execution',name).write_text(text,encoding='utf-8')

if __name__=='__main__':audit()
