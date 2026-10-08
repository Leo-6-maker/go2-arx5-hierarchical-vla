"""One scheduler lease, actual full-resume canary, then one fixed training segment."""
import argparse,fcntl,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
from contract import ROOT,guard,sha,write
from budget import process,time_allowed,seconds


def lease(root):
    job=os.environ['PBS_JOBID'];assert os.environ.get('CUDA_VISIBLE_DEVICES')
    raw=subprocess.check_output(['/cm/shared/apps/pbspro/current/bin/qstat','-f',job],text=True)
    wall=re.search(r'Resource_List.walltime\s*=\s*(\d+):(\d+):(\d+)',raw)
    elapsed=re.search(r'resources_used.walltime\s*=\s*(\d+):(\d+):(\d+)',raw)
    assert wall and re.search(r'job_state\s*=\s*R\b',raw)
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total','--format=csv,noheader'],text=True)
    assert 'H200' in gpu
    duration=lambda m:sum(int(v)*w for v,w in zip(m.groups(),(3600,60,1)))
    left=duration(wall)-(duration(elapsed) if elapsed else 0)
    limits=json.loads((root/'design/RESOURCE_RESERVATIONS.json').read_text())
    ledger=root/'runtime/GPU_PROCESSES.jsonl';rows=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
    ended={(r['pid'],r['start_epoch']) for r in rows if r['event']=='end'}
    assert all((r['pid'],r['start_epoch']) in ended for r in rows if r['event']=='start'),'Adopt actual unclosed allocation before another lease'
    used=seconds([(r['start_epoch'],r['epoch']) for r in rows if r['event']=='end'])
    left=min(left,limits['GPU_seconds']-used);assert left>600
    return dict(kind='PBS_H200',PBS_JOBID=job,pid=os.getpid(),started_epoch=time.time(),deadline_epoch=time.time()+left,scheduler_CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],scheduler_record=raw,actual_GPU=gpu,remaining_own_GPU_seconds=limits['GPU_seconds']-used,selected='NONE',deployment='NONE')


def run(a):
    root=Path(a.root);assert root.resolve()==ROOT.resolve();guard()
    sys.path.append(str(root.parent/'PI0283/code'))
    from training_launch import call,bind,retire_consumed_canary
    from canary import compare
    from authority import allowed,stage_card
    with (root/'runtime/TRAINING_ALLOCATION.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        allocation=lease(root);path=root/'runtime/TRAINING_ALLOCATION.json'
        if path.exists():
            old=json.loads(path.read_text());closed=root/'runtime'/('ALLOCATION_CLOSED_'+old['PBS_JOBID']+'.json')
            assert old['PBS_JOBID']!=allocation['PBS_JOBID'] and closed.exists(),'Adopt current allocation; no duplicate segment'
            assert json.loads(closed.read_text())['ended_epoch']<=allocation['started_epoch']
        write(path,allocation);write(root/'runtime'/('ALLOCATION_'+allocation['PBS_JOBID']+'.json'),allocation)
        allowed(root,'update',120);card=stage_card(root,a.card);assert card['stage']==a.arm
        common=['--root',str(root),'--base',a.base,'--data-root',str(root/'data'),'--card',a.card,'--arm',a.arm]
        began=process(root,'start');state='ENGINEERING_ATTEMPT_RETAINED'
        try:
            proof=root/'learning'/a.arm/'CANARY_COMPARISON.json'
            if not proof.exists():
                assert a.resume==0,'Missing original carry proof is not permission to repeat formal updates'
                for phase,resume,stop in [('continuous',0,8),('split',0,4),('split',4,8)]:
                    call(root,f'{a.arm}-{phase}-{resume}-{stop}','train.py',common+['--phase',phase,'--resume',str(resume),'--stop',str(stop)])
                for phase in ('continuous','split'):
                    point=a.arm+'-CANARY-'+phase
                    bind(root,point,root/'learning'/a.arm/phase/'checkpoint-8')
                    call(root,point,'model_worker.py',['--point',point,'--canary',str(root/'learning'/a.arm/phase)])
                compare(root,a.arm)
            assert json.loads(proof.read_text())['status']=='PASS_FULL_RESUME_CANARY'
            retire_consumed_canary(root,a.arm)
            call(root,f'{a.arm}-main-{a.resume}-{a.stop}','train.py',common+['--phase','main','--resume',str(a.resume),'--stop',str(a.stop)])
            output=root/'learning'/a.arm/'main';checkpoints=sorted(output.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
            assert checkpoints;last=int(checkpoints[-1].name.split('-')[-1])
            assert json.loads((checkpoints[-1]/'PI0282_RECOVERY.json').read_text())['cursor']==last*8
            state='TRAINING_COMPLETE_CPU_FINALIZATION_PENDING' if last==a.stop else 'SAFE_STOP_FULL_EXACT_RECOVERY'
            if last==a.stop:
                # CPU readback is part of this one lease; it adds no forward/gradient query.
                command=[sys.executable,'-u',str(root/'code/training_audit.py'),'--arm',a.arm,'--step',str(last)]
                with (root/'runtime'/f'{a.arm}-{last}_training_CPU.log').open('ab') as log:
                    subprocess.run(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,check=True)
                audits=root/'delivery/audits';audits.mkdir(exist_ok=True)
                shutil.copyfile(root/'evidence'/f'{a.arm}-{last}_TRAINING_CPU.json',audits/f'{a.arm}-{last}.json')
                state='FIXED_SEGMENT_COMPLETE'
            write(root/'runtime'/f'{a.arm}_SEGMENT_{a.resume}_{a.stop}.json',dict(status=state,last_completed_step=last,checkpoint=str(checkpoints[-1]),PBS_JOBID=allocation['PBS_JOBID'],card_sha256=sha(a.card),selected='NONE',deployment='NONE'))
        finally:
            process(root,'end',began)
            write(root/'runtime'/('ALLOCATION_CLOSED_'+allocation['PBS_JOBID']+'.json'),dict(status=state,ended_epoch=time.time(),PBS_JOBID=allocation['PBS_JOBID'],arm=a.arm,selected='NONE',deployment='NONE'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--base',required=True);p.add_argument('--card',required=True);p.add_argument('--arm',required=True);p.add_argument('--resume',type=int,default=0);p.add_argument('--stop',type=int,required=True);run(p.parse_args())
