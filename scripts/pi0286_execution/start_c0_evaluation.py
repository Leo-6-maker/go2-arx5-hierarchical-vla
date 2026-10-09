"""Owner-authorized finite C0 entry, gated by actual Hopper1024 audit; adopt once."""
import json,sys
from pathlib import Path
from read_endpoint_ready import read
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi0276_execution'))
from coordinator import remote


def start():
    ready=read('C0_REPLAY1024-1024');assert ready.get('endpoint_actual_ready_CPU_PASS'),'Actual1024 READY plus independent training CPU required'
    code=r'''
import json,os,subprocess,time,hashlib
from pathlib import Path
R=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0286');point='C0_REPLAY1024-1024';ready=READY
cmd=['/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/venv/bin/python','-u',str(R/'code/evaluate.py'),'--point',point,'--mode','endpoint']
record=R/'runtime/C0_ENDPOINT_EVALUATION_DISPATCH.json';closed=R/'runtime'/(point+'_endpoint_EVALUATION_CLOSED.json')
active=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
  if str(R/'code/evaluate.py') in argv and '--point' in argv and argv[argv.index('--point')+1]==point:active.append(int(p.name))
 except (OSError,UnicodeError):pass
if active:
 assert len(active)==1;print(json.dumps(dict(status='ADOPT_ACTUAL_EXISTING_C0_ENDPOINT_EVALUATOR',pid=active[0],point=point)))
elif record.exists():
 assert closed.exists(),'Recorded endpoint process ended without final closure; retain and owner-reconcile original work'
 v=json.loads(closed.read_text());print(json.dumps(dict(status='EXISTING_C0_ENDPOINT_CLOSURE_RETAINED_NO_RELAUNCH',closure=v)))
else:
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PI0286_ROOT=str(R),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_CORETYPE='Prescott',PYTHONPATH=':'.join([str(R/'code'),str(R.parent/'PI0283/code'),str(R.parent/'PI0280/official'),str(R/'project-src'),str(R/'project-src/src')]))
 check="from contract import guard; from r2 import phase_allowed; guard(); phase_allowed('model'); print('PASS_ACTUAL_CURRENT_SEAL_AND_MODEL_TIME_GUARD')"
 p=subprocess.run([cmd[0],'-c',check],env=env,capture_output=True,text=True);assert p.returncode==0,p.stderr
 gpu=subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);assert not gpu.strip(),'Owner must adopt live Core GPU0 model service'
 with (R/'runtime/C0_ENDPOINT_OWNER.log').open('ab') as f:child=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 v=dict(status='STARTED_ONCE_ACTUAL_1024_TRAINING_CPU_GATED_FINITE_C0_ENDPOINT',pid=child.pid,point=point,arguments=cmd,started_epoch=time.time(),Hopper_ready_CPU_identity=ready,evaluator_sha256=hashlib.sha256((R/'code/evaluate.py').read_bytes()).hexdigest(),Core_seal_sha256=hashlib.sha256((R/'design/CODE_SEAL.json').read_bytes()).hexdigest(),confirmation_started=False,selected='NONE',deployment='NONE')
 tmp=record.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+chr(10));tmp.replace(record);print(json.dumps(v))
'''.replace('READY',repr(ready),1)
    result=json.loads(remote('core',code))
    out=Path(__file__).resolve().parents[2]/'reports/PI0286/preparation/C0_ENDPOINT_EVALUATION_DISPATCH.json'
    out.write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':print(json.dumps(start()))
