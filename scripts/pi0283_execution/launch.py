"""One legal lease: exact carry proof, MORE, then owner-routed EVENT only."""
import argparse,fcntl,json,os,time
from pathlib import Path
from training_launch import call,bind,write,retire_consumed_canary


def run(root,base):
    root=Path(root);runtime=root/'runtime'
    assert os.environ.get('PBS_JOBID') and os.environ.get('CUDA_VISIBLE_DEVICES')
    from authority import allowed
    allowed(root,'update')
    with (runtime/'TRAINING_ALLOCATION.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        record=runtime/'TRAINING_ALLOCATION.json';assert not record.exists(),'ADOPT_HEALTHY_ALLOCATION_NO_DUPLICATE'
        started=time.time();write(record,dict(PBS_JOBID=os.environ['PBS_JOBID'],pid=os.getpid(),started_epoch=started,scheduler_CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES']))
        common=['--root',str(root),'--base',base,'--data-root',str(root/'data'),'--microbatch','4']
        more=common+['--arm','MORE','--continuation-card',str(root/'design/MORE_CARD.json')]
        for phase,resume,stop in [('continuous',0,8),('split',0,4),('split',4,8)]:
            call(root,f'MORE-{phase}-{resume}-{stop}','train.py',more+['--phase',phase,'--resume',str(resume),'--stop',str(stop)])
        for phase in ('continuous','split'):
            point='MORE-CANARY-'+phase;bind(root,point,root/'learning/MORE'/phase/'checkpoint-8')
            call(root,point,'inference.py',['--root',str(root),'--point',point,'--canary','--arm','MORE','--phase',phase])
        call(root,'MORE-canary-compare','canary.py',['--root',str(root),'--arm','MORE'])
        retire_consumed_canary(root,'MORE')
        call(root,'MORE-formal-2048','train.py',more+['--phase','main','--stop','2048'])
        parent=root/'learning/MORE/main/checkpoint-2048'
        assert json.loads((parent/'PI0282_RECOVERY.json').read_text())['cursor']==16384
        write(runtime/'MORE_COMPLETE.json',dict(status='MORE2048_FULL_TRAINER_COMPLETE',checkpoint=str(parent),PBS_JOBID=os.environ['PBS_JOBID'],selected='NONE',deployment='NONE'))
        # Finite wait inside this allocation; no daemon, second queue or blind afterany.
        from authority import load
        end=started+load(root)['training_GPU_reservation_s']-60
        decision=root/'design/EVENT_DECISION.json'
        while not decision.exists() and time.time()<end:time.sleep(10)
        event=False
        if decision.exists():
            route=json.loads(decision.read_text());event=route['launch_allowed']
            if event:
                assert route['MORE_complete'] and not route['qualified_candidate']
                allowed(root,'update',reserve_s=route['measured_training_reserve_s'])
                call(root,'EVENT-formal-1024','train.py',common+['--arm','EVENT','--continuation-card',str(root/'design/EVENT_CARD.json'),'--phase','main','--stop','1024'])
        write(runtime/'TRAINING_COMPLETE.json',dict(status='MORE_AND_ROUTED_EVENT_COMPLETE' if event else 'MORE_COMPLETE_EVENT_NOT_RUN',EVENT_executed=event,started_epoch=started,ended_epoch=time.time(),PBS_JOBID=os.environ['PBS_JOBID'],selected='NONE',deployment='NONE'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--base',required=True);a=p.parse_args();run(a.root,a.base)
