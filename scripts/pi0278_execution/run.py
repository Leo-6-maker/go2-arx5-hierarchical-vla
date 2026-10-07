"""Finite canary, exact state check, and sole authorized formal run."""
import argparse,fcntl,importlib.util,os,subprocess,sys
from pathlib import Path
import cpu_inputs as data
import train

def segment(root,phase,limit,resume=0,arm='COVER93'):
    output=root/arm/phase/'training';done=output/'SEGMENT_COMPLETE.json'
    if done.exists() and data.read(done)['actual_update']==limit:return output
    subprocess.run([sys.executable,'-u',str(root/'code/entry.py'),'--root',str(root),'--phase',phase,'--limit',str(limit),'--resume',str(resume)],check=True)
    assert data.read(done)['actual_update']==limit or (root/'runtime/STOP_TRAINING.json').exists()
    return output

def compare(root,a,b):
    import torch
    cfg=data.read(root/'design/INPUT_BINDING.json');sys.path[:0]=cfg['module_paths']
    from pi0260_r3_compare_canary import model_delta,optimizer_delta
    spec=importlib.util.spec_from_file_location('qualified_logs',data.read(root/'RUN_CONFIG.json')['canary_comparison_source']);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    old.logs_equal(a,b,8)
    for name in ('GRADIENTS8.json','ISOLATED_GRIPPER_GRADIENT.json'):assert data.read(a/name)==data.read(b/name),name
    for name in ('OUTPUT_HASHES.jsonl','box_objective.jsonl'):assert train.lines(a/name)==train.lines(b/name),name
    x,y=a/'checkpoint-8',b/'checkpoint-8';md,_=model_delta(x,y);od,_=optimizer_delta(x,y);assert md==od==0
    for name in ('scheduler.pt','rng_state.pth'):assert train.fingerprint(torch.load(x/name,map_location='cpu',weights_only=False))==train.fingerprint(torch.load(y/name,map_location='cpu',weights_only=False)),name
    return dict(status='PASS_PI0278_REAL_EXACT_RESUME',continuous8_vs4_restore4=True,model_max_abs=md,optimizer_max_abs=od,outputs_gradients_rng_scheduler_inputs_losses='EXACT',completed_updates=16)

def retire_canary(root,phase,arm='COVER93'):
    output=root/arm/phase/'training'
    for cp in output.glob('checkpoint-*'):
        assert cp.resolve().is_relative_to(root.resolve()) and not cp.is_symlink()
        files=[p for p in cp.rglob('*') if p.is_file()];assert not any(p.is_symlink() for p in cp.rglob('*'))
        data.write(root/'evidence/retired'/f'{phase}-{cp.name}.json',dict(reason='PASSED_EXACT_QUALIFICATION_NO_MODEL_CANDIDATE',files=[dict(path=str(p.relative_to(cp)),bytes=p.stat().st_size,sha256=data.sha(p)) for p in files]))
        for p in files:p.unlink()
        for p in sorted((p for p in cp.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):p.rmdir()
        cp.rmdir()

def main(root):
    assert os.environ.get('PBS_JOBID')
    with (root.parent/'runtime/owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for r in data.read(root/'EXECUTION_FILESET.json')['files']:assert data.sha(r['path'])==r['sha256']
        receipt=root/'COVER93/canary_compare.json'
        if not receipt.exists():
            continuous=segment(root,'continuous',8);split=root/'COVER93/split/training'
            if not (split/'checkpoint-4/trainer_state.json').exists():segment(root,'split',4)
            segment(root,'split',8,4);data.write(receipt,compare(root,continuous,split))
        for phase in ('continuous','split'):retire_canary(root,phase)
        segment(root,'main',4096)
        data.write(root/'evidence/TRAINING_COMPLETE.json',dict(status='ACTUAL_COVER93_SEGMENT_ENDED',quality='PENDING_FIXED_LOCAL_SCREENS',HF='DEFERRED_USER_PRIORITY',selected_checkpoint='NONE',deployment='NONE'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
