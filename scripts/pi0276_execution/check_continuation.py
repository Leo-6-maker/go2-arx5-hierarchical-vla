"""CPU-only finite-phase adoption check; no remote calls or model imports."""
import ast,contextlib,io,json,tempfile
from pathlib import Path
import coordinator
from coordinator import phase_status

with tempfile.TemporaryDirectory() as folder:
    root=Path(folder);proc=root/'proc';proc.mkdir();start=root/'start.json';done=root/'done.json';marker=root/'receipt.json'
    target='/experiment/code/archive.py';args=['training_data']
    def status():return phase_status(start,done,target,args,str(marker),proc)
    assert status()['status']=='NEW'
    p=proc/'41';p.mkdir();(p/'stat').write_text('41 (python worker) S 0')
    (p/'cmdline').write_bytes(b'python\0'+target.encode()+b'\0source\0')
    assert status()['status']=='NEW'
    (p/'cmdline').write_bytes(b'python\0'+target.encode()+b'\0training_data\0')
    assert status()=={'status':'RUNNING','pids':[41],'adopted':True}
    start.write_text(json.dumps({'pid':41,'target':target,'arguments':args,'adopted':True}))
    assert status()['status']=='RUNNING'
    (p/'cmdline').write_bytes(b'python\0unrelated.py\0')
    assert status()['status']=='FAILED'  # Reused PID cannot authorize a restart or false RUNNING.
    start.write_text(json.dumps({'pid':41,'target':target,'arguments':args,'wrapper_argv':['python','-c','worker']}))
    (p/'cmdline').write_bytes(b'python\0-c\0worker\0')
    assert status()['status']=='RUNNING'
    (p/'stat').write_text('41 (python worker) Z 0')
    assert status()['status']=='FAILED'
    done.write_text(json.dumps({'returncode':1,'target':target,'arguments':args}))
    assert status()['status']=='FAILED'
    done.write_text(json.dumps({'returncode':0,'target':target,'arguments':args}))
    assert status()['status']=='COMPLETE'
    done.unlink();marker.write_text('{"status":"PASS_PUBLIC_FIXED_REVISION_FILES_AND_ALL_MEMBERS"}')
    assert status()['status']=='COMPLETE'
def dry_remote(host,program):
    tree=ast.parse(program)
    workers=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('import json,os,subprocess\n')]
    assert len(workers)==1
    worker=ast.parse(workers[0])
    compile(tree,'remote_phase','exec');compile(worker,'detached_worker','exec')
    assert any(isinstance(n,ast.keyword) and n.arg=='start_new_session' and n.value.value is True for n in ast.walk(tree))
    return '{"status":"COMPLETE"}'
attempts=[]
def reconnect_read(host,program):
    ast.parse(program);attempts.append(program)
    if len(attempts)==1:raise ConnectionError('CPU simulated disconnect')
    return '{"resumed":true}'
coordinator.remote=reconnect_read
original_sleep=coordinator.time.sleep;coordinator.time.sleep=lambda seconds:None
with contextlib.redirect_stdout(io.StringIO()):assert coordinator.read('core','/receipt.json')=={'resumed':True}
assert len(attempts)==2 and attempts[0]==attempts[1]
coordinator.time.sleep=original_sleep
coordinator.remote=dry_remote
with contextlib.redirect_stdout(io.StringIO()):
    coordinator.call('core','archive.py','training_data')
    coordinator.call('core','evaluate.py',native=True)
    coordinator.call('hop','admit_hopper.py','cpu')
def dry_training_remote(host,program):
    ast.parse(program)
    if "'qsub'" in program or '/qsub' in program:return json.dumps({'job_id':'CPU_DRY_RUN','release':coordinator.RELEASE})
    if '/qstat' in program:return json.dumps({'complete':True,'returncode':0})
    return '{}'
coordinator.remote=dry_training_remote
coordinator.read=lambda *args:None
coordinator.stage_hopper=lambda:None
coordinator.call=lambda *args,**kwargs:None
coordinator.training()  # Only compile the actual submission/copy RPCs; never contact a host.
print('PASS_CPU_FINITE_PHASE_ADOPTION_NO_MODEL_QUERY')
