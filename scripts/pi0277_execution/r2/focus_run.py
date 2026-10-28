"""One serialized optional lineage, sixteen qualification updates, same formal budget."""
import argparse,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import cpu_inputs as data
from run import compare

def segment(root,phase,route,limit,resume=0):
    output=root/'FOCUS'/phase/'training';done=output/'SEGMENT_COMPLETE.json'
    if done.exists() and data.read(done)['actual_update']==limit:return output
    if datetime.now(timezone.utc).isoformat()>='2026-10-06T23:00:00+00:00':
        data.write(root/'evidence/NO_NEW_SEGMENT_CUTOFF.json',dict(phase=phase,limit=limit,resume=resume,status='STOP_NEW_TRAINING_AT_0700_SGT'));raise SystemExit(0)
    subprocess.run([sys.executable,'-u',str(root/'code/focus_train.py'),'--root',str(root),'--phase',phase,'--route',route,'--limit',str(limit),'--resume',str(resume)],check=True)
    assert data.read(done)['actual_update']==limit or (root/'runtime/STOP_TRAINING.json').exists()
    return output

def retire(root,phase):
    output=root/'FOCUS'/phase/'training'
    for cp in output.glob('checkpoint-*'):
        assert cp.parent==output and not cp.is_symlink()
        files=[]
        for p in cp.rglob('*'):
            assert not p.is_symlink()
            if p.is_file():files.append(dict(path=str(p.relative_to(cp)),bytes=p.stat().st_size,sha256=data.sha(p)))
        data.write(root/'evidence/retired'/f'{phase}-{cp.name}.json',dict(reason='COMPLETED_EXACT_FOCUS_QUALIFICATION_ONLY',files=files))
        for row in files:(cp/row['path']).unlink()
        for p in sorted((p for p in cp.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):p.rmdir()
        cp.rmdir()

def main(root):
    import fcntl
    assert os.environ.get('PBS_JOBID')
    with (root.parent/'runtime/owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        card=data.read(root/'FOCUS_AUTHORITY.json');assert card['status']=='ACTUAL_PRIMARY4096_DEVELOPMENT_FAIL' and card['launch_allowed']
        for row in data.read(root/'EXECUTION_FILESET.json')['files']:assert data.sha(row['path'])==row['sha256']
        route=data.read(root/'evidence/LOADER_SELECTION.json')['route'];assert route=='FAST_CACHE2'
        receipt=root/'FOCUS/canary_compare.json'
        if not receipt.exists():
            continuous=segment(root,'continuous',route,8)
            split=root/'FOCUS/split/training'
            if not (split/'checkpoint-4/trainer_state.json').exists():segment(root,'split',route,4)
            segment(root,'split',route,8,4)
            result=compare(root,continuous,split)
            data.write(receipt,dict(status='PASS_R2_EXACT_GRIP_RESUME',legacy_status_name=True,lineage='FOCUS',continuous8_vs4_restore4=True,route=route,**result))
            for phase in ('continuous','split'):retire(root,phase)
        segment(root,'main',route,4096)
        data.write(root/'evidence/HOPPER_FOCUS_SEGMENT_COMPLETE.json',dict(status='FOCUS_SEGMENT_EXECUTED',quality='PENDING_ACTUAL_SCREENS',archive='DEFERRED_USER_PRIORITY',selected_checkpoint='NONE',deployment='NONE'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
