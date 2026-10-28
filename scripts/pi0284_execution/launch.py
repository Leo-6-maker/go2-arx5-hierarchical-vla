"""One finite lease, two independent carry proofs, then the frozen matched arms."""
import argparse,fcntl,json,os,sys,time
from pathlib import Path

def run(root,base):
    root=Path(root);sys.path.append(str(root.parent/'PI0283/code'))
    from authority import allowed
    from training_launch import call,bind,write,retire_consumed_canary
    allowed(root,'update');assert os.environ.get('PBS_JOBID') and os.environ.get('CUDA_VISIBLE_DEVICES')
    assert json.loads((root/'runtime/CPU_TRAINING_CHECK.json').read_text())['status']=='PASS_CPU_RUNTIME_CARDS_SAME_PARENT_SINGLE_LR_DIFFERENCE_PLAN_AND_FINITE_CAPS'
    assert len(json.loads((root/'design/IMPLEMENTATION.json').read_text())['published_commit'])==40
    with (root/'runtime/TRAINING_ALLOCATION.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=root/'runtime/TRAINING_ALLOCATION.json';assert not path.exists(),'ADOPT_RECORDED_LEASE_NO_DUPLICATE'
        write(path,dict(PBS_JOBID=os.environ['PBS_JOBID'],pid=os.getpid(),started_epoch=time.time(),scheduler_CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES']))
        common=['--root',str(root),'--base',base,'--data-root',str(root/'data'),'--microbatch','4']
        arms=('CTRL2048','LR5X2048')
        for arm in arms:
            args=common+['--arm',arm,'--continuation-card',str(root/'design'/(arm+'_CARD.json'))]
            for phase,resume,stop in [('continuous',0,8),('split',0,4),('split',4,8)]:
                call(root,f'{arm}-{phase}-{resume}-{stop}','train_arm.py',args+['--phase',phase,'--resume',str(resume),'--stop',str(stop)])
            for phase in ('continuous','split'):
                bind(root,arm+'-CANARY-'+phase,root/'learning'/arm/phase/'checkpoint-8')
                call(root,arm+'-prediction-'+phase,'carry_check.py',['--root',str(root),'--arm',arm,'--phase',phase])
            call(root,arm+'-compare','carry_check.py',['--root',str(root),'--arm',arm,'--phase','compare'])
            retire_consumed_canary(root,arm)
        rows=[]
        for arm in arms:
            rows.append([{k:v[k] for k in ('step','micro','cursor','visits','inputs','native_noise_sha256','native_time_sha256','native_time')} for v in map(json.loads,(root/'learning'/arm/'continuous/consumption.jsonl').read_text().splitlines())])
        assert rows[0]==rows[1],'CROSS_ARM_ACTUAL_INPUT_OR_NATIVE_RNG_MISMATCH'
        write(root/'runtime/PAIRED_CANARY.json',dict(status='PASS_BOTH_REAL_CARRY_PROOFS_AND_CROSS_ARM_INPUT_NOISE_TIME',discarded_updates=32))
        data_receipt=root/'runtime/DUPLICATE_AND_EXPOSURE_AUDIT.json'
        while not data_receipt.exists():
            allowed(root,'update',reserve_s=4800);time.sleep(10)
        assert json.loads(data_receipt.read_text())['status']=='PASS_SAVED_TRAIN_DATA_AND_COMMON_PLAN_AUDIT_NOT_MODEL_FIT'
        for arm in arms:
            args=common+['--arm',arm,'--continuation-card',str(root/'design'/(arm+'_CARD.json')),'--phase','main','--stop','2048']
            call(root,arm+'-formal-2048','train_arm.py',args)
            for step in (1024,2048):
                cp=root/'learning'/arm/'main'/f'checkpoint-{step}'
                assert json.loads((cp/'PI0282_RECOVERY.json').read_text())['cursor']==step*8
            write(root/'runtime'/(arm+'_COMPLETE.json'),dict(status='COMPLETE_BOTH_FULL_TRAINERS',arm=arm,formal_updates=2048,selected='NONE',deployment='NONE'))
        write(root/'runtime/TRAINING_COMPLETE.json',dict(status='BOTH_FROZEN_ARMS2048_COMPLETE_NOT_QUALITY_VERDICT',formal_updates=4096,canary_updates=32,selected='NONE',deployment='NONE'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--base',required=True);a=p.parse_args();run(a.root,a.base)
