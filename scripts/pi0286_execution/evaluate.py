"""Finite PI0286 learner collection, endpoint panels or one frozen confirmation."""
import argparse,json,os,subprocess,time
from pathlib import Path
from contract import ROOT,guard,sha,write
from policy import PARENT,ENDPOINTS,tasks


def call(script,arguments,gpu=False,renderer=False):
    import hashlib
    python=ROOT.parent/('PI0276/venv/bin/python' if renderer else 'PI0250/runtime/venv/bin/python')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0' if gpu else '',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_CORETYPE='Prescott',PI0286_ROOT=str(ROOT),PI0286_RECORD_ROOT=str(ROOT/'source'),HF_HOME=str(ROOT.parent/'PI0250/runtime/hf_home'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    env['PYTHONPATH']=':'.join([str(ROOT/'code'),str(ROOT.parent/'PI0283/code'),str(ROOT.parent/'PI0280/official'),str(ROOT/'project-src'),str(ROOT/'project-src/src')])
    name=script.replace('.py','')+'-'+hashlib.sha256(json.dumps(arguments).encode()).hexdigest()[:12]
    with (ROOT/'runtime'/(name+'.EVAL.log')).open('ab') as log:
        started=time.time();result=subprocess.run([str(python),'-u',str(ROOT/'code'/script),*arguments],env=env,stdout=log,stderr=subprocess.STDOUT)
    write(ROOT/'runtime'/(name+'.EVAL_COMPLETE.json'),dict(returncode=result.returncode,script=script,arguments=arguments,started_epoch=started,ended_epoch=time.time(),GPU_visible=env['CUDA_VISIBLE_DEVICES']))
    assert result.returncode==0,('Retain original attempted work; inspect the actual engineering log',name)


def run(point,mode):
    import fcntl
    guard();assert os.environ['CUDA_VISIBLE_DEVICES']==''
    assert (mode=='learner' and point==PARENT) or (mode in ('endpoint','confirm') and point in ENDPOINTS)
    with (ROOT/'runtime/EVALUATION_OWNER.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(ROOT/'runtime/EVALUATION_PROCESS.json',dict(pid=os.getpid(),point=point,mode=mode,started_epoch=time.time()))
        status='ENGINEERING_PENDING_ORIGINAL_ATTEMPTS_PRESERVED'
        try:
            if mode=='learner':
                call('policy.py',['--point',point,'--panel','LEARNER_TRAIN16'],gpu=True,renderer=True)
                call('audit_source.py',['--split',point+'_LEARNER_TRAIN16'])
                status='FIXED16_TRAIN_LEARNER_STATES_AND_INDEPENDENT_CPU_CLOSED_NOT_R3_RESCORE'
            elif mode=='endpoint':
                call('transport.py',['--point',point])
                call('policy.py',['--point',point,'--panel','TRAIN'],gpu=True,renderer=True)
                call('audit_source.py',['--split',point+'_TRAIN'])
                call('qualification.py',['--point',point])
                route=json.loads((ROOT/'qualification'/(point+'_ROUTE.json')).read_text())
                if route['next_action']=='RUN_FIXED_DEV':
                    call('policy.py',['--point',point,'--panel','DEV'],gpu=True,renderer=True)
                    call('audit_source.py',['--split',point+'_DEV'])
                if not (ROOT/'design/DIAG96_PER_TASK.json').exists():call('diagnostics.py',['prepare'])
                call('diagnostics.py',['run','--point',point],gpu=True)
                call('diagnostics.py',['cpu','--point',point])
                call('qualification.py',['--point',point])
                status=json.loads((ROOT/'qualification'/(point+'_ROUTE.json')).read_text())['next_action']
            else:
                from qualification import confirmation_gate
                confirmation_gate(point)
                call('policy.py',['--point',point,'--panel','CONFIRM_C'],gpu=True,renderer=True)
                call('audit_source.py',['--split',point+'_CONFIRM_C'])
                call('qualification.py',['--point',point])
                status=json.loads((ROOT/'qualification'/(point+'_ROUTE.json')).read_text())['next_action']
            return dict(status=status,point=point,mode=mode,selected='NONE',deployment='NONE')
        finally:write(ROOT/'runtime'/f'{point}_{mode}_EVALUATION_CLOSED.json',dict(status=status,ended_epoch=time.time(),point=point,mode=mode,selected='NONE',deployment='NONE'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',required=True);p.add_argument('--mode',choices=('learner','endpoint','confirm'),required=True);a=p.parse_args();print(json.dumps(run(a.point,a.mode)))
