"""PI0286 isolated copy: One scheduler lease, actual full-resume canary, then one fixed training segment."""
import argparse,fcntl,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
from contract import ROOT,guard,sha,write
from budget import process,time_allowed,seconds
from r2 import UPDATE_STOP,ARM_START_STOP


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
    return dict(kind='PBS_H200',PBS_JOBID=job,pid=os.getpid(),started_epoch=time.time(),deadline_epoch=min(time.time()+left,UPDATE_STOP+120),scheduler_CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],scheduler_record=raw,actual_GPU=gpu,remaining_own_GPU_seconds=limits['GPU_seconds']-used,selected='NONE',deployment='NONE')


def run_arm(root,arm,base,card_path,stop=1024):
    from training_launch import call,bind,retire_consumed_canary
    from canary import compare
    from authority import allowed,stage_card
    allowed(root,'update',120);card=stage_card(root,card_path);assert card['stage']==arm
    assert time.time()<ARM_START_STOP
    common=['--root',str(root),'--base',base,'--data-root',str(root/'data'),'--card',str(card_path),'--arm',arm]
    proof=root/'learning'/arm/'CANARY_COMPARISON.json'
    assert not proof.exists(),'New PI0286 arm requires its real new carry proof; adopt existing work instead'
    for phase,resume,end in [('continuous',0,8),('split',0,4),('split',4,8)]:
        call(root,f'{arm}-{phase}-{resume}-{end}','train.py',common+['--phase',phase,'--resume',str(resume),'--stop',str(end)])
    for phase in ('continuous','split'):
        point=arm+'-CANARY-'+phase;bind(root,point,root/'learning'/arm/phase/'checkpoint-8')
        call(root,point,'model_worker.py',['--point',point,'--canary',str(root/'learning'/arm/phase)])
    compare(root,arm);assert json.loads(proof.read_text())['status']=='PASS_FULL_RESUME_CANARY'
    retire_consumed_canary(root,arm)
    call(root,f'{arm}-main-0-{stop}','train.py',common+['--phase','main','--resume','0','--stop',str(stop)])
    output=root/'learning'/arm/'main';checkpoints=sorted(output.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
    assert checkpoints;last=int(checkpoints[-1].name.split('-')[-1])
    assert json.loads((checkpoints[-1]/'PI0282_RECOVERY.json').read_text())['cursor']==last*8
    state='SAFE_STOP_FULL_EXACT_RECOVERY'
    if last==stop:
        with (root/'runtime'/f'{arm}-{last}_training_CPU.log').open('ab') as log:
            subprocess.run([sys.executable,'-u',str(root/'code/training_audit.py'),'--arm',arm,'--step',str(last)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,check=True)
        audits=root/'delivery/audits';audits.mkdir(exist_ok=True);shutil.copyfile(root/'evidence'/f'{arm}-{last}_TRAINING_CPU.json',audits/f'{arm}-{last}.json')
        state='FIXED_SEGMENT_COMPLETE'
    write(root/'runtime'/f'{arm}_SEGMENT_0_{stop}.json',dict(status=state,last_completed_step=last,checkpoint=str(checkpoints[-1]),PBS_JOBID=os.environ['PBS_JOBID'],card_sha256=sha(card_path),selected='NONE',deployment='NONE'))
    return state


def run(a):
    root=Path(a.root);assert root.resolve()==ROOT.resolve();guard();assert time.time()<ARM_START_STOP
    sys.path.append(str(root.parent/'PI0283/code'))
    with (root/'runtime/TRAINING_ALLOCATION.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=root/'runtime/TRAINING_ALLOCATION.json';assert not path.exists(),'Adopt the one accepted lease; never create a second driver'
        allocation=lease(root);write(path,allocation);write(root/'runtime'/('ALLOCATION_'+allocation['PBS_JOBID']+'.json'),allocation)
        began=process(root,'start');state='ENGINEERING_ATTEMPT_RETAINED'
        try:
            state=run_arm(root,'C0_REPLAY1024',a.base,root/'design/C0_REPLAY1024_CARD.json')
            if state!='FIXED_SEGMENT_COMPLETE':return
            card=root/'design/C1_CORRECT1024_CARD.json'
            while time.time()<ARM_START_STOP:
                if (root/'runtime/STOP_NEW_ARMS.json').exists() or (root/'qualification/CONFIRM_C_CANDIDATE.json').exists():
                    state='C1_NOT_STARTED_OWNER_CANDIDATE_OR_STOP';break
                if card.exists() and json.loads(card.read_text()).get('launch_allowed'):
                    state=run_arm(root,'C1_CORRECT1024',a.base,card);break
                write(root/'runtime/C1_WAITING.json',dict(status='C0_COMPLETE_WAITING_ACTUAL_C1_ADMISSION_SAME_LEASE',epoch=time.time(),last_arm_start_epoch=ARM_START_STOP,model_queries=0,optimizer_updates=0))
                time.sleep(min(30,max(0,ARM_START_STOP-time.time())))
            else:state='C1_NOT_ADMITTED_BEFORE_LAST_ARM_START'
        finally:
            process(root,'end',began);write(root/'runtime'/('ALLOCATION_CLOSED_'+allocation['PBS_JOBID']+'.json'),dict(status=state,ended_epoch=time.time(),PBS_JOBID=allocation['PBS_JOBID'],selected='NONE',deployment='NONE'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--base',required=True);p.add_argument('--card');p.add_argument('--arm');p.add_argument('--stop',type=int,default=1024);a=p.parse_args();run(a)
