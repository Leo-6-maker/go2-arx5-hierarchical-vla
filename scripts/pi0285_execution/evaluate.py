"""One finite source evaluation endpoint; CPU readback before each owner decision."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from contract import ROOT,guard,sha,write
from policy import tasks


def phases(point,action):
    tasks(point,'TRAIN')
    if action=='START':return ['DIAG','TRAIN']
    if action=='RUN_FIXED_DEV':return ['DEV']
    if action=='FREEZE_FIRST_CANDIDATE_FOR_UNIQUE_CONFIRM_A':return ['CONFIRM_A']
    if action=='FREEZE_FIRST_CANDIDATE_FOR_UNIQUE_CONFIRM_B':return ['CONFIRM_B']
    assert action in ('CONTINUE_SAME_PHASE_4096','COLLECT_FIXED_CORRECTIONS','STOP_VALID_QUALITY_FAIL','STOP_CONFIRMATION_VALID_QUALITY_FAIL_NO_SECOND_CANDIDATE','PROTECT_PLACE_V1_BEFORE_SHARED','PROTECT_SHARED_TWO_TASK_POLICY')
    return []


def call(script,arguments,gpu=False,renderer=False):
    python=ROOT.parent/('PI0276/venv/bin/python' if renderer else 'PI0250/runtime/venv/bin/python')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0' if gpu else '',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_CORETYPE='Prescott',PI0285_ROOT=str(ROOT),PI0285_RECORD_ROOT=json.loads((ROOT/'design/RECORDING_TIER.json').read_text())['record_root'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    env['PYTHONPATH']=':'.join([str(ROOT/'code'),str(ROOT.parent/'PI0283/code'),str(ROOT.parent/'PI0280/official'),str(ROOT/'project-src'),str(ROOT/'project-src/src')])
    name=script.replace('.py','')+'-'+hashlib_name(arguments);path=ROOT/'runtime'/(name+'.EVAL_COMPLETE.json')
    with (ROOT/'runtime'/(name+'.EVAL.log')).open('ab') as log:
        started=time.time();result=subprocess.run([str(python),'-u',str(ROOT/'code'/script),*arguments],env=env,stdout=log,stderr=subprocess.STDOUT)
    write(path,dict(returncode=result.returncode,script=script,arguments=arguments,started_epoch=started,ended_epoch=time.time(),GPU_visible=env['CUDA_VISIBLE_DEVICES']))
    assert result.returncode==0,('Original first-valid results retained; inspect actual engineering failure',str(path))


def hashlib_name(value):
    import hashlib
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()[:12]


def run(point):
    import fcntl
    guard();tasks(point,'TRAIN');assert os.environ['CUDA_VISIBLE_DEVICES']==''
    with (ROOT/'runtime/EVALUATION_OWNER.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(ROOT/'runtime/EVALUATION_PROCESS.json',dict(pid=os.getpid(),point=point,started_epoch=time.time(),role='FINITE_ENDPOINT_FIRST_VALID_SOURCE_AND_CPU'))
        status='ENGINEERING_PENDING_ORIGINAL_RESULTS_PRESERVED'
        try:
            # Actual audited Trainer -> immutable model view; no optimizer copied to Corelab.
            call('transport.py',['--point',point])
            if not (ROOT/'design/DIAG96_PER_TASK.json').exists():call('diagnostics.py',['prepare'])
            call('diagnostics.py',['run','--point',point],gpu=True)
            call('diagnostics.py',['cpu','--point',point])
            call('policy.py',['--point',point,'--panel','TRAIN'],gpu=True,renderer=True)
            call('audit_source.py',['--split',point+'_TRAIN'])
            while True:
                call('qualification.py',['--point',point])
                route=json.loads((ROOT/'qualification'/(point+'_ROUTE.json')).read_text());action=route['next_action']
                following=phases(point,action)
                if not following:status=action;return route
                for panel in following:
                    call('policy.py',['--point',point,'--panel',panel],gpu=True,renderer=True)
                    call('audit_source.py',['--split',point+'_'+panel])
        finally:
            write(ROOT/'runtime'/(point+'_EVALUATION_CLOSED.json'),dict(status=status,ended_epoch=time.time(),point=point,selected='NONE',deployment='NONE'))


def self_check():
    assert phases('PLACE4096-2048','START')==['DIAG','TRAIN']
    assert phases('PLACE4096-2048','RUN_FIXED_DEV')==['DEV']
    assert phases('PLACE4096-2048','FREEZE_FIRST_CANDIDATE_FOR_UNIQUE_CONFIRM_A')==['CONFIRM_A']
    assert phases('SHARED4096-4096','FREEZE_FIRST_CANDIDATE_FOR_UNIQUE_CONFIRM_B')==['CONFIRM_B']
    assert not phases('PLACE4096-4096','COLLECT_FIXED_CORRECTIONS')
    assert not phases('PLACE4096-2048','STOP_CONFIRMATION_VALID_QUALITY_FAIL_NO_SECOND_CANDIDATE')
    assert not phases('PLACE4096-2048','PROTECT_PLACE_V1_BEFORE_SHARED')
    print('PASS_FINITE_ENDPOINT_REAL_CPU_ROUTES_FIRST_CANDIDATE_AND_NO_QUALITY_RETRY; model/physics=0')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point');p.add_argument('--self-check',action='store_true');a=p.parse_args()
    self_check() if a.self_check else print(json.dumps(run(a.point)))
