"""First-valid fixed offline inputs and the original source panel, one GPU policy."""
import argparse,fcntl,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from prepare_core import write,sha
from diagnostic_runner import reply

PYTHON='/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/venv/bin/python'
SOURCE_PYTHON='/home/liuyu-nfs/Projects/go2-arx5/PI0276/venv/bin/python'

def environment(root,gpu):
    card=json.loads((root/'design/SOURCE_CARD.json').read_text())
    return dict(os.environ,CUDA_VISIBLE_DEVICES='0' if gpu else '',CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HOME=card['hf_home'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',NO_ALBUMENTATIONS_UPDATE='1',PYTHONPATH=card['native_pythonpath'])

def diagnostic(root,point):
    from score import metrics,gate
    folder=root/'endpoint_diagnostics'/point;folder.mkdir(parents=True,exist_ok=True)
    target=folder/'RESULT.json'
    if target.exists():return json.loads(target.read_text())
    plan=json.loads((root/'design/PRE_SCORE_INPUTS.json').read_text())['rows'];records=[]
    with (folder/'worker.log').open('ab') as log:
        server=subprocess.Popen([PYTHON,'-u',str(root/'code/endpoint_worker.py'),'--root',str(root),'--point',point],env=environment(root,True),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
        try:
            reply(server,'ready')
            for row in plan:
                with np.load(row['target']) as data:commands=data['commands'].copy()
                for packet in (0,1):
                    path=folder/f"{row['index']:02d}-packet{packet}.npz"
                    request=dict(input=row['input'],output=str(path),noise=str(Path(row['input']).parent/f'noise-{packet}.npy'))
                    if not path.exists():server.stdin.write(json.dumps(request)+'\n');server.stdin.flush();reply(server,'output')
                    with np.load(path) as data:
                        assert str(data['source_sha256'])==sha(row['input']) and np.array_equal(data['noise_uint16'],np.load(request['noise']))
                        pred=data['decoded'].copy();state=data['actual_state'].copy();raw=data['raw'][0,:,:10].copy()
                    error='';prefix=full=None;decision=dict(decision='REJECT')
                    try:prefix=metrics(pred[:8],commands[:8]);full=metrics(pred,commands);decision=gate(pred,state)
                    except (ValueError,AssertionError) as exc:error=str(exc)
                    records.append(dict(index=row['index'],uid=row['uid'],split=row['split'],phase=row['phase'],packet=packet,path=str(path),sha256=sha(path),target=row['target'],target_sha256=sha(row['target']),prefix=prefix,fullH40=full,gate=decision,error=error,raw_OOB_prefix=((raw[:8]<-1)|(raw[:8]>1)).sum(0).tolist(),raw_OOB_H40=((raw<-1)|(raw>1)).sum(0).tolist()))
                    write(folder/'PROGRESS.json',dict(records=records))
            if point.endswith('2048'):
                pairs=json.loads((root/'design/NATURAL_PAIRS.json').read_text())['pairs'];natural=[]
                for i,pair in enumerate(pairs):
                    for packet in (0,1):
                        for member,side in enumerate(('a','b')):
                            uid=pair[side]['uid'];source=Path(pair[side]['input']);path=folder/f'natural-pair{i}-packet{packet}-member{member}.npz'
                            request=dict(input=str(source),output=str(path),noise=str(root/'natural'/f'pair-{i}-noise-{packet}.npy'))
                            if not path.exists():server.stdin.write(json.dumps(request)+'\n');server.stdin.flush();reply(server,'output')
                            natural.append(dict(pair=i,packet=packet,member=member,uid=uid,path=str(path),sha256=sha(path)))
                write(folder/'NATURAL_RESULTS.json',dict(records=natural))
        finally:server.stdin.close();assert server.wait(timeout=120)==0
    result=dict(status='FIXED_DIAG48_TWO_PACKETS_COMPLETE_NOT_TASK_QUALIFICATION',point=point,records=records,rows=96)
    write(target,result);return result

def source(root,point,panel):
    name=point+'-'+panel;runtime=root/'runtime';done=runtime/(name+'-source.done.json');start=runtime/(name+'-source.start.json')
    if done.exists():assert json.loads(done.read_text())['returncode']==0;return
    assert not start.exists(),'Adopt existing physical execution, never quality retry'
    with (runtime/(name+'-source.log')).open('ab') as log:
        began=time.time();worker=subprocess.Popen([SOURCE_PYTHON,'-u',str(root/'code/source_entry.py'),'--root',str(root),'--point',point,'--panel',panel],env=environment(root,True),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        write(start,dict(pid=worker.pid,started_epoch=began,point=point,panel=panel));returncode=worker.wait()
    write(done,dict(returncode=returncode,started_epoch=began,ended_epoch=time.time(),point=point,panel=panel));assert returncode==0
    code=json.loads((root/'design/RUNTIME_CARD.json').read_text())['independent_CPU_audit']
    program="import importlib.util;from pathlib import Path;s=importlib.util.spec_from_file_location('independent_fixed_task',"+repr(code)+");m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.ROOT=Path("+repr(str(root))+");m.audit("+repr(point)+","+repr(panel)+")"
    with (runtime/(name+'-independent-CPU.log')).open('ab') as log:subprocess.run([PYTHON,'-c',program],env=environment(root,False),stdout=log,stderr=subprocess.STDOUT,check=True)

def run(root,point,panel):
    root=Path(root)
    with (root/'runtime/EVALUATION_LEASE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if panel=='TRAIN':diagnostic(root,point)
        source(root,point,panel)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--point',choices=('CTRL1024','CTRL2048','LR5X1024','LR5X2048'),required=True);p.add_argument('--panel',choices=('TRAIN','DEV','CONFIRM'),default='TRAIN');a=p.parse_args();run(a.root,a.point,a.panel)
