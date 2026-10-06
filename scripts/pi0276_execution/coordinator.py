"""Finite two-host continuation through the PC's existing SSH identities; no monitor."""
import argparse,base64,hashlib,inspect,json,os,shlex,subprocess,sys,time
from pathlib import Path

RELEASE='57fd8a9c53ce241c05a8d5b351034bed7ba56b14'
CORE='/home/liuyu-nfs/Projects/go2-arx5/PI0276'
HOP='/scratch/e1192548/Projects/go2-arx5/experiments/PI0276'
CODE=Path(__file__).resolve().parent


def hopper_required():
    assert os.environ.get('PBS_JOBID'),'Model computation requires the sealed Hopper PBS allocation'


def remote(host,script):
    options=['-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6']
    args=['ssh',*options]
    args+=['corelab-amd-server','ssh',*options,'corelab-4080s','python3'] if host=='core' else ['nus-hopper','python3']
    result=subprocess.run(args,input=script,text=True,capture_output=True)
    if result.returncode:
        error=ConnectionError if result.returncode==255 else RuntimeError
        raise error(result.stderr[-6000:]+result.stdout[-6000:])
    return result.stdout


def read(host,path):
    while True:
        try:
            out=remote(host,f"import json\nfrom pathlib import Path\np=Path({path!r})\nprint(p.read_text() if p.exists() else 'null')\n")
            return json.loads(out)
        except ConnectionError as error:
            print('SSH unavailable; read-only wait '+host+': '+str(error)[-500:],flush=True);time.sleep(30)


def phase_status(start,done,target,arguments,marker,proc=Path('/proc')):
    if marker and Path(marker).is_file():
        assert json.loads(Path(marker).read_text())['status']=='PASS_PUBLIC_FIXED_REVISION_FILES_AND_ALL_MEMBERS'
        return {'status':'COMPLETE','receipt':marker}
    if done.exists():
        value=json.loads(done.read_text());assert value['target']==target and value['arguments']==list(arguments)
        return dict(value,status='COMPLETE' if value['returncode']==0 else 'FAILED')
    value=json.loads(start.read_text()) if start.exists() else {}
    if value:assert value['target']==target and value['arguments']==list(arguments)
    pids=[]
    for p in proc.iterdir():
        if not p.name.isdigit():continue
        try:
            if (p/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z':continue
            tokens=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\0') if v]
            if target in tokens and tokens[tokens.index(target)+1:]==list(arguments):pids.append(int(p.name))
            elif int(p.name)==value.get('pid') and tokens==value.get('wrapper_argv'):pids.append(int(p.name))
        except (OSError,UnicodeError):continue
    if pids:return {'status':'RUNNING','pids':pids,'adopted':not bool(value)}
    return {'status':'FAILED' if value else 'NEW','reason':'Recorded process ended without completion receipt' if value else None}


def call(host,script,*args,native=False):
    root=CORE if host=='core' else HOP
    if host=='core':
        python=CORE+'/venv/bin/python' if native else '/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/venv/bin/python'
        command=[python,'-u',root+'/code/'+script,*args]
    else:
        shell='source '+shlex.quote(HOP.rsplit('/',1)[0]+'/PI0273/code/environment.sh')+'; export PYTHONPATH='+shlex.quote(HOP+'/code')+':"$PYTHONPATH"; "$venv/bin/python" -u '+shlex.quote(root+'/code/'+script)+' '+shlex.join(args)
        command=['/bin/bash','-lc',shell]
    target=root+'/code/'+script
    phase=script.replace('.py','')+'-'+hashlib.sha256(json.dumps(args).encode()).hexdigest()[:12]
    marker=root+'/evidence/RESULT_ARCHIVE_'+args[0]+'.json' if script=='archive.py' else None
    worker=f"""import json,os,subprocess
from pathlib import Path
root=Path({root!r});done=root/'runtime'/{(phase+'.done.json')!r}
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_CORETYPE='Prescott')
if {script!r} not in ('evaluate.py',):env['CUDA_VISIBLE_DEVICES']=''
with (root/'runtime'/('continuation-'+{script!r}+'.log')).open('ab') as f:
 result=subprocess.run({command!r},env=env,stdout=f,stderr=subprocess.STDOUT)
tmp=done.with_suffix('.tmp');tmp.write_text(json.dumps({{'returncode':result.returncode,'target':{target!r},'arguments':{list(args)!r}}})+'\\n');tmp.replace(done)
"""
    program=f"""import fcntl,json,subprocess,sys
from pathlib import Path
{inspect.getsource(phase_status)}
root=Path({root!r});runtime=root/'runtime';runtime.mkdir(parents=True,exist_ok=True)
start=runtime/{(phase+'.start.json')!r};done=runtime/{(phase+'.done.json')!r}
with (runtime/{(phase+'.lock')!r}).open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 value=phase_status(start,done,{target!r},{list(args)!r},{marker!r})
 if value['status']=='NEW':
  with (runtime/{(phase+'.worker.log')!r}).open('ab') as log:
   wrapper_argv=[sys.executable,'-u','-c',{worker!r}]
   child=subprocess.Popen(wrapper_argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  tmp=start.with_suffix('.tmp');tmp.write_text(json.dumps({{'pid':child.pid,'target':{target!r},'arguments':{list(args)!r},'wrapper_argv':wrapper_argv}})+'\\n');tmp.replace(start)
  value={{'status':'RUNNING','pids':[child.pid],'adopted':False}}
 elif value['status']=='RUNNING' and value.get('adopted'):
  tmp=start.with_suffix('.tmp');tmp.write_text(json.dumps({{'pid':value['pids'][0],'target':{target!r},'arguments':{list(args)!r},'adopted':True}})+'\\n');tmp.replace(start)
print(json.dumps(value))
"""
    # ponytail: one finite phase per existing command; no service or background monitor.
    announced=False
    while True:
        try:value=json.loads(remote(host,program))
        except ConnectionError as error:
            print('SSH unavailable; retaining exact phase '+phase+': '+str(error)[-500:],flush=True);time.sleep(30);continue
        if value['status']=='COMPLETE':print('COMPLETE '+script+' '+shlex.join(args),flush=True);return
        if value['status']=='FAILED':raise RuntimeError('Actual remote phase failed: '+json.dumps(value))
        if not announced:print('ADOPTED/RUNNING '+script+' '+shlex.join(args)+' '+json.dumps(value),flush=True);announced=True
        time.sleep(30)


def stage_hopper():
    # Only small code/authority/receipt files cross the PC; task payload comes from pinned public HF.
    files={p.name:base64.b64encode(p.read_bytes()).decode() for p in CODE.iterdir() if p.suffix in ('.py','.pbs')}
    payload=remote('core',f"""import json,base64
from pathlib import Path
r=Path({CORE!r});paths=list((r/'authority').glob('*'))+[r/'evidence/RESULT_ARCHIVE_training_data.json']
print(json.dumps({{str(p.relative_to(r)):base64.b64encode(p.read_bytes()).decode() for p in paths if p.is_file()}}))
""")
    payload=json.loads(payload);payload.update({'code/'+n:v for n,v in files.items()})
    remote('hop',f"""import json,base64
from pathlib import Path
r=Path({HOP!r});assert not (r/'evidence/TRAINING_JOB.json').exists(),'Do not restage a launched runtime'
for name,data in {payload!r}.items():
 p=r/name;assert p.resolve().is_relative_to(r.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(data))
(r/'logs').mkdir(exist_ok=True);(r/'runtime').mkdir(exist_ok=True)
print('STAGED_SMALL_TASK_RUNTIME')
""")


def source_data():
    while not read('core',CORE+'/evidence/SOURCE_TO_DATA_COMPLETE.json'):
        state=json.loads(remote('core',f"""import json,os
from pathlib import Path
r=Path({CORE!r});p=r/'runtime/owner.json';v=json.loads(p.read_text()) if p.exists() else {{}}
pid=v.get('pid',-1);proc=Path('/proc')/str(pid)/'cmdline'
alive=pid>0 and proc.exists() and {CORE!r}.encode() in proc.read_bytes()
print(json.dumps({{'active':alive,'owner':v}}))
"""))
        if not state['active']:
            if not read('core',CORE+'/evidence/SOURCE.json'):call('core','run.py','source','--release',RELEASE,'--resume',native=True)
            call('core','finish_source.py',native=True)
        else:time.sleep(30)
    if not read('core',CORE+'/evidence/RESULT_ARCHIVE_training_data.json'):call('core','archive.py','training_data')


def training():
    job=read('hop',HOP+'/evidence/TRAINING_JOB.json')
    if not job:
        stage_hopper();call('hop','provision.py')
        for step in ('cpu','binding','seal'):call('hop','admit_hopper.py',step)
        job=json.loads(remote('hop',f"""import json,subprocess,fcntl
from pathlib import Path
r=Path({HOP!r})
with (r/'runtime/submit.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=r/'evidence/TRAINING_JOB.json';assert not p.exists()
 result=subprocess.run(['/cm/shared/apps/pbspro/current/bin/qsub',str(r/'code/training.pbs')],capture_output=True,text=True,check=True)
 value={{'job_id':result.stdout.strip(),'release':{RELEASE!r}}};p.write_text(json.dumps(value)+'\\n');print(json.dumps(value))
"""))
    assert job['release']==RELEASE
    while True:
        out=remote('hop',f"""import json,subprocess
from pathlib import Path
r=Path({HOP!r});q=subprocess.run(['/cm/shared/apps/pbspro/current/bin/qstat','-f',{job['job_id']!r}],capture_output=True,text=True)
print(json.dumps({{'returncode':q.returncode,'qstat':q.stdout+q.stderr,'complete':(r/'evidence/TASK/checkpoint-8192.full_readback.json').exists() and (r/'evidence/TASK/checkpoint-4096.full_readback.json').exists()}}))
""")
        state=json.loads(out)
        if state['complete']:break
        if state['returncode']:raise RuntimeError('Actual training/archive job ended without complete receipts: '+state['qstat'])
        time.sleep(30)
    paths=['evidence/MAIN_TRAINING_AUDIT.json','design/TASK_READOUT_R1.json','EXECUTION_FILESET.json','TASK/canary_compare.json']
    paths += [f'evidence/TASK/checkpoint-{s}.{kind}.json' for s in (4096,8192) for kind in ('archive_receipt','full_readback')]
    payload=json.loads(remote('hop',f"""import base64,json
from pathlib import Path
r=Path({HOP!r});print(json.dumps({{n:base64.b64encode((r/n).read_bytes()).decode() for n in {paths!r}}}))
"""))
    remote('core',f"""import base64
from pathlib import Path
r=Path({CORE!r})
for n,data in {payload!r}.items():
 p=r/('hopper' if n=='EXECUTION_FILESET.json' else '')/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(data))
""")


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('all','audit','setup','native','source','bridge','dataset','admit','canary','train','evaluate','replay','archive','closeout'))
    p.add_argument('--release',required=True);p.add_argument('--resume',action='store_true');args=p.parse_args();assert args.release==RELEASE
    import msvcrt
    root=Path('D:/Users/GR00T project/Projects/go2-arx5/PI0276');root.mkdir(exist_ok=True)
    with (root/'coordinator.lock').open('a+b') as lock:
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        if args.stage in ('all','source','admit'):source_data()
        if args.stage in ('all','canary','train'):training()
        if args.stage in ('all','evaluate'):
            if not read('core',CORE+'/evidence/FINAL_POLICY_ADMISSION.json'):call('core','materialize.py')
            if not read('core',CORE+'/evidence/PHYSICAL_EVALUATION_COMPLETE.json'):call('core','evaluate.py',native=True)
        if args.stage in ('all','replay','archive','closeout'):
            for script in ('model_replay.py','report.py','video.py','closeout.py'):call('core',script)
            for part in ('source','predictions','summary'):
                if not read('core',CORE+'/evidence/RESULT_ARCHIVE_'+part+'.json'):call('core','archive.py',part)
            call('core','closeout.py','--verify')
        if args.stage in ('audit','setup','native','bridge','dataset'):call('core','run.py',args.stage,'--release',RELEASE,'--resume',native=True)
        print('FINITE_CONTINUATION_RETURNED; final GitHub publication remains owned by the active goal',flush=True)
