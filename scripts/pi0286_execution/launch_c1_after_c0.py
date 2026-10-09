"""User-authorized C1 continuation after complete C0; no repeated C0 work."""
import argparse,base64,fcntl,json,os,shutil,subprocess,sys,time
from pathlib import Path


def run(base):
    root=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0286')
    authority=json.loads((root/'design/BOTH_ARMS_CONTINUATION_1400.json').read_text())
    assert authority['launch_allowed'] and authority['formal_cap']==2048
    old=authority['C0_job'];job=os.environ['PBS_JOBID'];assert job!=old
    q=subprocess.check_output(['/cm/shared/apps/pbspro/current/bin/qstat','-xf',old],text=True)
    assert 'job_state = F' in q and 'Exit_status = 0' in q
    complete=json.loads((root/'runtime/C0_REPLAY1024_SEGMENT_0_1024.json').read_text())
    assert complete['status']=='FIXED_SEGMENT_COMPLETE' and complete['last_completed_step']==1024
    audit=json.loads((root/'evidence/C0_REPLAY1024-1024_TRAINING_CPU.json').read_text())
    assert audit['status']=='PASS_ACTUAL_NATIVE_TRAINER_INPUT_PLAN_NOISE_LOSS_MODES_SCOPE_AND_COMPLETE_LOCAL_STATE'
    sys.path.insert(0,str(root/'code'));sys.path.append(str(root.parent/'PI0283/code'))
    history=root/'runtime/continuation-c1/history';history.mkdir(parents=True,exist_ok=True)
    with (root/'runtime/TRAINING_ALLOCATION.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=root/'runtime/TRAINING_ALLOCATION.json'
        assert json.loads(path.read_text())['PBS_JOBID']==old
        assert (root/'runtime'/('ALLOCATION_CLOSED_'+old+'.json')).exists()
        shutil.copyfile(path,history/path.name);assert path.read_bytes()==(history/path.name).read_bytes();path.unlink()
        seal=root/'design/CODE_SEAL.json';shutil.copyfile(seal,history/'CODE_SEAL.json')
        source=root/'code/r2.py';shutil.copyfile(source,history/'r2.py')
        source.write_bytes(base64.b64decode(authority['r2_source_base64']))
        from contract import sha,write,guard
        value=json.loads(seal.read_text());value['files']['r2.py']=sha(source);write(seal,value)
        guard()
        from launch import lease,run_arm
        from budget import process
        allocation=lease(root);write(path,allocation);write(root/'runtime'/('ALLOCATION_'+job+'.json'),allocation)
        began=process(root,'start');state='ENGINEERING_ATTEMPT_RETAINED'
        try:
            assert not (root/'learning/C1_CORRECT1024/CANARY_COMPARISON.json').exists()
            state=run_arm(root,'C1_CORRECT1024',base,root/'design/C1_CORRECT1024_CARD.json')
        finally:
            process(root,'end',began)
            write(root/'runtime'/('ALLOCATION_CLOSED_'+job+'.json'),dict(status=state,ended_epoch=time.time(),PBS_JOBID=job,selected='NONE',deployment='NONE'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);run(p.parse_args().base)
