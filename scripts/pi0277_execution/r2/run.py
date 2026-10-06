"""R2 finite GRIP-first owner: short input checks, exact canary, useful updates."""
import argparse,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
import cpu_inputs as data
RELEASE='1a20200ff066ef5a003650110cd20106f9a4bf64'

def call(root,name,*args):
    print('START',name,*args,flush=True)
    subprocess.run([sys.executable,'-u',str(root/'code'/name),*map(str,args)],check=True)

def prepare(root):
    parent=root.parent
    if not (parent/'evidence/LOCAL_BINDING.json').exists():
        subprocess.run([sys.executable,str(parent/'code/prepare.py')],check=True)
    import shutil
    for name in ('DONOR_LOCAL_READBACK.json','DONOR_PARAMETER_SCOPE.json','STORAGE_ADMISSION.json','design/TRAIN_PLAN.json','design/NOISE_CARD.json','design/GRIP_READOUT.json'):
        dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copyfile(parent/name,dest)
        assert dest.read_bytes()==(parent/name).read_bytes()
    binding=data.read(parent/'design/INPUT_BINDING.json');binding['module_paths']=[str(root/'code')]+binding['module_paths']
    data.write(root/'design/INPUT_BINDING.json',binding)
    # Preserve the scientific plan's V1 identity; R2 changes scheduling, not its bytes.
    rows={r['path']:r for r in data.read(parent/'EXECUTION_FILESET.json')['files']}
    for p in list((root/'code').glob('*.py'))+list((root/'code').glob('*.pbs'))+list((root/'design').glob('*.json'))+[root/'ADOPTION.json']:
        rows[str(p)]=dict(path=str(p),bytes=p.stat().st_size,sha256=data.sha(p))
    value=dict(release=RELEASE,files=sorted(rows.values(),key=lambda r:r['path']),qualified_math_changed=False)
    seal=root/'EXECUTION_FILESET.json'
    if seal.exists():assert data.read(seal)==value
    else:data.write(seal,value)
    data.write(root/'evidence/LOCAL_BINDING.json',dict(status='PASS_R2_LOCAL_SOURCE_BINDING',fileset_sha256=data.sha(seal),launch_authority_only=True))

def segment(root,phase,route,limit,resume=0):
    output=root/'GRIP'/phase/'training';done=output/'SEGMENT_COMPLETE.json'
    if done.exists() and data.read(done)['limit']==limit:return output
    from datetime import datetime,timezone
    if datetime.now(timezone.utc).isoformat()>='2026-10-06T23:00:00+00:00':
        data.write(root/'evidence/NO_NEW_SEGMENT_CUTOFF.json',dict(phase=phase,limit=limit,resume=resume,status='STOP_NEW_TRAINING_AT_0700_SGT'))
        raise SystemExit(0)
    call(root,'train.py','--root',root,'--arm','GRIP','--phase',phase,'--route',route,'--limit',limit,'--resume',resume)
    assert data.read(done)['limit']==limit
    return output

def compare(root,a,b):
    import torch
    from train import fingerprint,lines
    sys.path[:0]=data.read(root/'design/INPUT_BINDING.json')['module_paths']
    from pi0260_r3_compare_canary import model_delta,optimizer_delta
    spec=importlib.util.spec_from_file_location('pi0277_v1_qualify',root.parent/'code/qualify.py')
    previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
    previous.logs_equal(a,b,8)
    assert data.read(a/'GRADIENTS8.json')==data.read(b/'GRADIENTS8.json')
    assert lines(a/'OUTPUT_HASHES.jsonl')==lines(b/'OUTPUT_HASHES.jsonl')
    x,y=a/'checkpoint-8',b/'checkpoint-8'
    md,_=model_delta(x,y);od,_=optimizer_delta(x,y);assert md==od==0
    for name in ('scheduler.pt','rng_state.pth'):
        assert fingerprint(torch.load(x/name,map_location='cpu',weights_only=False))==fingerprint(torch.load(y/name,map_location='cpu',weights_only=False)),name
    assert data.read(a/'ISOLATED_GRIPPER_GRADIENT.json')==data.read(b/'ISOLATED_GRIPPER_GRADIENT.json')
    return dict(model_max_abs=md,optimizer_max_abs=od,outputs_gradients_rng_scheduler_inputs_losses='EXACT')

def retire(root,phase):
    # ponytail: only this run's compared qualification checkpoints, never model candidates.
    output=root/'GRIP'/phase/'training'
    for cp in sorted(output.glob('checkpoint-*')):
        assert cp.parent==output and not cp.is_symlink()
        rows=[]
        for p in cp.rglob('*'):
            assert not p.is_symlink()
            if p.is_file():rows.append(dict(path=str(p.relative_to(cp)),bytes=p.stat().st_size,sha256=data.sha(p)))
        receipt=root/'evidence/retired'/f'{phase}-{cp.name}.json'
        data.write(receipt,dict(reason='COMPLETED_EXACT_QUALIFICATION_ONLY',files=rows))
        for r in rows:(cp/r['path']).unlink()
        for p in sorted((p for p in cp.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):p.rmdir()
        cp.rmdir()

def main(root):
    import fcntl
    (root/'runtime').mkdir(parents=True,exist_ok=True)
    with (root.parent/'runtime/owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert os.environ.get('PBS_JOBID') and data.read(root/'ADOPTION.json')['release']==RELEASE
        prepare(root)
        selection=root/'evidence/LOADER_SELECTION.json'
        if not selection.exists():
            started=time.monotonic();rates={}
            for route in ('REF_W0','FAST_CACHE0','FAST_CACHE2'):
                path=root/'evidence/cpu_bench'/f'{route}-0.json'
                if not path.exists():call(root,'bench_cpu.py','--root',root,'--route',route,'--trial',0)
                rates[route]=data.read(path)['items_per_second']
                if time.monotonic()-started>=45*60:break
            route=max(rates,key=rates.get)
            data.write(selection,dict(route=route,CPU_rates=rates,H200_speed='UNOBSERVED',speed_is_launch_gate=False))
        route=data.read(selection)['route'];receipt=root/'GRIP/canary_compare.json'
        if not receipt.exists():
            reference=segment(root,'reference','REF_W0',8)
            continuous=reference
            if route!='REF_W0':
                continuous=segment(root,'continuous',route,8)
                data.write(root/'evidence/LOADER_EQUIVALENCE.json',compare(root,reference,continuous))
                retire(root,'reference')
            split=root/'GRIP/split/training'
            if not (split/'checkpoint-4/trainer_state.json').exists():segment(root,'split',route,4)
            segment(root,'split',route,8,4)
            result=compare(root,continuous,split)
            data.write(receipt,dict(status='PASS_R2_EXACT_GRIP_RESUME',route=route,continuous8_vs4_restore4=True,**result))
            for phase in ('reference','continuous','split'):retire(root,phase)
        segment(root,'main',route,4096)
        data.write(root/'evidence/HOPPER_GRIP_SEGMENT_COMPLETE.json',dict(status='GRIP_SEGMENT_EXECUTED',quality='PENDING_ACTUAL_SCREENS',archive='DEFERRED_USER_PRIORITY',selected_checkpoint='NONE',deployment_selected_checkpoint='NONE'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
