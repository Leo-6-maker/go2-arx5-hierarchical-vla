"""Finite owner feedback for one frozen snapshot; unchanged source evaluators."""
import argparse,fcntl,hashlib,json,os,resource,subprocess,time
from pathlib import Path
ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0283')
PYTHON='/home/liuyu-nfs/Projects/go2-arx5/PI0250/runtime/venv/bin/python'
BASE='/var/cache/users/liuyu-nfs/go2-arx5-vla-stage10/assets/GR00T-N1.7-3B'


def write(path,v):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(path)


def run(point,panel,resume_technical=False):
    assert point in ('MORE1024','MORE2048','EVENT1024') and panel in ('TRAIN','DEV','CONFIRM')
    runtime=ROOT/'runtime';done=runtime/(point+'-'+panel+'-feedback.done.json')
    if done.exists():assert json.loads(done.read_text())['returncode']==0;return
    with (runtime/'EVALUATION_LEASE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start=runtime/(point+'-'+panel+'-feedback.start.json');began=time.time()
        if start.exists():
            assert resume_technical,'ADOPT_EXISTING_FEEDBACK_NO_DUPLICATE'
            prior=json.loads(start.read_text())
            assert not Path('/proc/'+str(prior['pid'])).exists(),'HEALTHY_FEEDBACK_NO_RESTART'
            write(runtime/(point+'-'+panel+'-feedback.technical-resume.json'),dict(pid=os.getpid(),prior=prior,started_epoch=began,reason='MISSING_ORIGINAL_CUBLAS_WORKSPACE_ENVIRONMENT',quality_retry=False))
        else:write(start,dict(pid=os.getpid(),point=point,panel=panel,started_epoch=began))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONPATH=str(ROOT/'code')+':/home/liuyu-nfs/Projects/go2-arx5/PI0280/official:/home/liuyu-nfs/Projects/go2-arx5/PI0276/project-src',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        def call(script,args,gpu=False):
            e=dict(env,CUDA_VISIBLE_DEVICES='0' if gpu else '')
            with (runtime/(point+'-'+panel+'-'+script+'.log')).open('ab') as log:
                subprocess.run([PYTHON,'-u',str(ROOT/'code'/script),*args],env=e,stdout=log,stderr=subprocess.STDOUT,check=True)
        if panel=='TRAIN':
            if not (ROOT/'transfers'/point/'DIRECT_READBACK.json').exists():call('transport.py',['--root',str(ROOT),'--point',point])
            nvme=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0283/models')/point;nvme.mkdir(parents=True,exist_ok=True)
            folder=ROOT/'models'/point;folder.parent.mkdir(exist_ok=True)
            if not folder.exists():folder.symlink_to(nvme,target_is_directory=True)
            assert folder.resolve()==nvme.resolve()
            if not (folder/'MATERIALIZATION.json').exists():call('model_view.py',['--root',str(ROOT),'--point',point,'--base',BASE,'--export',str(ROOT/'transfers'/point)])
            else:assert json.loads((folder/'MATERIALIZATION.json').read_text())['status']=='PASS_EXACT_TRANSFER_MASTER_READBACK_NATIVE_BF16_VIEW'
            # Only the consumed transfer duplicate is retired; Hopper full Trainer stays protected.
            delta=ROOT/'transfers'/point/'exact_changed_tensors.safetensors'
            material=json.loads((folder/'MATERIALIZATION.json').read_text());assert material['status']=='PASS_EXACT_TRANSFER_MASTER_READBACK_NATIVE_BF16_VIEW'
            if delta.exists():
                export=json.loads((delta.parent/'EXPORT.json').read_text());assert delta.stat().st_size==export['exact_delta_bytes']
                write(delta.parent/'DUPLICATE_RETIREMENT.json',dict(path=str(delta),class_='EXACT_TRANSFER_COPY_AFTER_FULL_MATERIALIZATION',bytes=delta.stat().st_size,sha256=export['exact_delta_sha256'],protected_full_trainer=export['checkpoint'],status='PREVIEW'))
                delta.unlink();v=json.loads((delta.parent/'DUPLICATE_RETIREMENT.json').read_text());v['status']='DELETED_VERIFIED_DUPLICATE';write(delta.parent/'DUPLICATE_RETIREMENT.json',v)
            call('evaluate.py',['diag','--point',point],True)
            call('evaluate.py',['cpu','--point',point,'--root',str(ROOT)])
        if panel!='TRAIN':
            choice=json.loads((ROOT/'design/ONE_DEV_CANDIDATE.json').read_text());assert choice['point']==point
        call('source_launch.py',[point,panel],True)
        with (runtime/(point+'-'+panel+'-independent-CPU.log')).open('ab') as log:
            subprocess.run([PYTHON,str(runtime/'closed_loop_audit.py'),point,'--panel',panel],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        write(done,dict(returncode=0,point=point,panel=panel,started_epoch=began,ended_epoch=time.time(),elapsed_s=time.time()-began,CPU_children_s=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,source_candidate='NONE',selected='NONE',deployment='NONE'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('point');p.add_argument('--panel',default='TRAIN');p.add_argument('--resume-technical',action='store_true');a=p.parse_args();run(a.point,a.panel,a.resume_technical)
