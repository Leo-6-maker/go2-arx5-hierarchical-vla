"""Authorized OLD/ROT contracts; reuse the qualified two-microbatch/S4 trainer."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import IterableDataset

ROOT=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0268')
R3=ROOT.with_name('PI0260')/'R3';OLD=ROOT.with_name('PI0264')
STAGE=Path('/scratch/e1192548/stage22_h200_runtime_deb920_20260902')
GROOT=STAGE/'src/isaac-gr00t'
sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT.with_name('PI0267')/'code'),str(OLD/'code'),str(R3/'execution'),
              '/scratch/e1192548/pi0244_fixed16_10k_v1/scripts','/scratch/e1192548/pi0244_fixed16_10k_v1/src',str(GROOT)]
import pi0264_train as paired
assert Path(paired.__file__).resolve()==ROOT.with_name('PI0267')/'code/pi0264_train.py'
from pi0260_r3_inputs import batch,identity
from pi0264_plan import NONLANGUAGE


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seed_for(unit):return int.from_bytes(hashlib.sha256(f'PI0267:TRAIN:26741:{unit}'.encode()).digest()[:8],'big')%2**31


def bind_checkpoint_metadata(output,checkpoint,stats,contract):
    # Native Trainer saves processor beside checkpoints; reuse its actual bytes.
    assert read(output/'processor/statistics.json')==read(stats),'SAVED_PROCESSOR_CONTRACT_MISMATCH'
    sources={name:output/'processor'/name for name in ('statistics.json','processor_config.json','embodiment_id.json')}
    sources.update({f'experiment_cfg/{name}':output/'experiment_cfg'/name for name in ('conf.yaml','config.yaml','final_model_config.json')})
    sources['wandb_config.json']=output/'wandb_config.json'
    sources['effective_statistics.json']=stats
    for name,source in sources.items():
        assert source.is_file() and not source.is_symlink()
        target=checkpoint/name;target.parent.mkdir(parents=True,exist_ok=True);content=source.read_bytes()
        if target.exists():assert target.read_bytes()==content
        else:
            temp=target.with_suffix(target.suffix+'.partial')
            with temp.open('wb') as stream:stream.write(content);stream.flush();os.fsync(stream.fileno())
            os.replace(temp,target)
    target=checkpoint/'PI0268_POLICY_CONTRACT.json'
    content=json.dumps(contract,sort_keys=True)+'\n'
    if target.exists():assert target.read_text()==content
    else:target.write_text(content)


def language(plan,u,c):
    colors=plan['source_family_colors'][u['family']]
    if u['task']=='M1':return plan['templates'][u['template']].format(color=colors[c])
    start,end,direction=('black','white','counterclockwise') if c==0 else ('white','black','clockwise')
    return f'Approach the {start} endpoint, then follow the open contour {direction} to the {end} endpoint as seen from above.'


class Pairs(IterableDataset):
    def __init__(self,processor,output,arm,resume):
        self.processor,self.output,self.arm=processor,output,arm;self.seed=26741;self.cursor=2*resume
        self.plan=read(ROOT/'PAIRING_PLAN.json');assert len(self.plan['units'])==4096

    def reset_seed(self,seed):
        self.cursor=2*(seed-self.seed);assert 0<=self.cursor<4096 and self.cursor%2==0

    @lru_cache(maxsize=24)
    def family(self,fid):
        folder=ROOT.with_name('PI0267')/'source/S2'/fid
        complete=read(folder/'COMPLETE.json');checks=read(folder/'codec_admission.json');assert checks['pass_admission']
        # Actual source packet identities checked once per process, not once per sample.
        for name in ('proposals.npz','policy_frames.npz','rows.json'):assert sha(folder/name)==complete['hashes'][name]
        z=np.load(folder/'proposals.npz');fr=np.load(folder/'policy_frames.npz')
        return z,fr,read(folder/'rows.json')

    def __iter__(self):
        for u in self.plan['units'][self.cursor//2:2048]:
            z,fr,rows=self.family(u['family'])
            for micro,c in enumerate(u['order']):
                i=u['R_indices'][c];row=rows[i];assert row['task']==u['task'] and row['command']==c
                assert row['anchor_index']==u['anchor'] and row['precodec_valid']
                if u['slot']=='COMMON_NOMINAL':assert row['nominal'] and i==u['N_indices'][c]
                else:assert row['variant']==u['recovery_variant']
                j=row['frame_index'];text=language(self.plan,u,c)
                packet=dict(body_rgb=fr['body_rgb'][j],wrist_rgb=fr['wrist_rgb'][j],state_v1=z['state_v1'][i],language=text)
                value=batch(self.processor,packet,z['raw_action'][i]);condition={k:v for k,v in value['inputs'].items() if k not in ('action','action_mask')}
                key=identity(condition);value['_pi0264_meta']=dict(unit=u['unit'],micro=micro,command=c,condition_sha256=key)
                with (self.output/'sample_audit.jsonl').open('a') as f:
                    f.write(json.dumps(dict(unit=u['unit'],micro=micro,arm=self.arm,task=u['task'],family=u['family'],slot=u['slot'],command=c,
                        sequence=u['family'],index=i,condition_sha256=key,original_unit=f"{u['task']}:{u['bank_index']}",visit=u['visit'],
                        color=self.plan['source_family_colors'][u['family']][c],template=u['template'],language=text,frame_key=row['frame_key'],
                        nonlanguage_sha256=identity({k:value['inputs'][k] for k in NONLANGUAGE}),
                        labels_masks_sha256=identity({k:value['inputs'][k] for k in ('action','action_mask')})),sort_keys=True)+'\n')
                yield value


def run(args):
    assert os.environ.get('PBS_JOBID') and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    assert torch.cuda.device_count()==1 and torch.cuda.get_device_properties(0).total_memory>=100*(1<<30)
    seal=read(ROOT/'EXECUTION_FILESET.json')
    for r in seal['files']:assert Path(r['path']).stat().st_size==r['bytes'] and sha(r['path'])==r['sha256']
    assert read(ROOT/'evidence/HOPPER_CPU_ADMISSION.json')['status']=='PASS_PI0268_HOPPER_ACTUAL_PROCESSOR_BOUNDARY'
    assert read(ROOT/'evidence/HOPPER_PHYSICAL_FIXTURES.json')['status']=='PASS_BOTH_CONTRACT_PHYSICAL_ROUNDTRIPS'
    for arm in ('OLD','ROT'):
        assert read(ROOT/'evidence'/('RUNTIME_BINDING_'+arm+'.json'))['status']=='PASS_EXACT_FROZEN_HELPER_IMPORTS_AND_CPU_HOOK_INSTALL'
    card=read(ROOT/'design/APPROVED_RUN_CARD.json')
    assert card['launch_allowed'] and card['release']=='e5ec6d52100f6bb2e32be5b9661c756b0adb61f4'
    stats=ROOT/'contracts'/args.arm/'source/dataset/statistics.json'
    assert sha(stats)==card['contracts'][args.arm]
    donor=read(ROOT/'DONOR_LOCAL_READBACK.json');assert donor['status']=='PASS_PI0267_FIXED_L2048_COMPLETE_TRAINER_BINDING'
    assert donor['revision']=='e4290e6ca387abd2de4337b5f12b9c72a5f85c99' and donor['tree_sha256']=='2f61d8ed8c8fe8444b7ef23164cb4df2bcacd8a73c93f19be77fcdf459ef6fae'
    output=ROOT/args.arm/args.phase/'training';output.mkdir(parents=True,exist_ok=True)
    if args.phase=='main':assert read(ROOT/args.arm/'canary_compare.json')['status']=='PASS_PI0268_REAL_TWO_MICROBATCH_EXACT_RESUME_CANARY'
    if args.resume:
        cp=output/f'checkpoint-{args.resume}';assert read(cp/'trainer_state.json')['global_step']==args.resume
        assert all((cp/n).is_file() for n in ('optimizer.pt','scheduler.pt','rng_state.pth'))
        for name in ('S4_audit.jsonl','sample_audit.jsonl','step_rng.jsonl'):
            assert len((output/name).read_text().splitlines())==2*args.resume
    else:assert not (output/'S4_audit.jsonl').exists()
    paired.ROOT=ROOT;paired.seed_for=seed_for;paired.Pairs=Pairs
    paired.r3.ROOT=ROOT/'contracts'/args.arm
    # Reuse r3's actual processor/loss construction; its old visit table is not consumed.
    paired.r3.Visits=lambda processor,*unused:SimpleNamespace(processor=processor)
    import pi0228_h200_large_adapt_v1 as old
    from stage40_b0_model_control import _bind_local_cosmos_processor
    assert _bind_local_cosmos_processor(STAGE,paired.COSMOS)['resolved_snapshot']==str(paired.COSMOS.resolve())
    config_args=argparse.Namespace(stage10_root=STAGE,base_model_path=Path(donor['path']),
        dataset_dir=ROOT.with_name('PI0227')/'pi0227_large_adapt_v1_inputs/train_dataset',
        modality_config=Path('/scratch/e1192548/pi0227_h200_prep_20260919/inputs/go2_arx5_modality.py'),groot_source=GROOT)
    save_steps=1024 if args.phase=='main' else (4 if args.phase.startswith('split') else 8)
    config=old.build_config(config_args,output,steps=2048,save_steps=save_steps,save_limit=2)
    config.training.learning_rate=1e-5;config.training.lr_scheduler_type='cosine';config.training.gradient_accumulation_steps=2
    config.training.max_steps=2048;config.training.save_steps=save_steps;config.training.save_total_limit=2
    config.training.resume_from_checkpoint=bool(args.resume);config.data.seed=config.training.seed=26741;config.training.remove_unused_columns=False
    assert config.training.warmup_steps==10 and config.training.max_grad_norm==1. and config.training.weight_decay==1e-5
    audit=paired.install(args,output,raw_head_cap=90000,raw_update_cap=4384)
    from gr00t.experiment import experiment
    base_trainer=experiment.Gr00tTrainer
    class SpaceCheckedTrainer(base_trainer):
        def training_step(self,model,inputs,*pos,**kw):
            if args.resume and not getattr(self,'_pi0267_resume_rng_synced',False):
                # HF restores before iter(DataLoader), which consumes one extra CPU base-seed draw.
                # This corpus uses zero workers and deterministic encoding; restore after iterator creation.
                assert self.args.dataloader_num_workers==0 and self.state.global_step==args.resume
                checkpoint=output/f'checkpoint-{args.resume}'
                self._load_rng_state(str(checkpoint))
                self._pi0267_resume_rng_synced=True
                (output/'RESUME_RNG_LIFECYCLE.json').write_text(json.dumps(dict(status='RESTORED_AFTER_DATALOADER_ITERATOR_BEFORE_FIRST_MICRO',
                    checkpoint=str(checkpoint),rng_sha256=sha(checkpoint/'rng_state.pth'),unit=args.resume+1,
                    inputs_unmodified=True,worker_count=0),sort_keys=True)+'\n')
            result=super().training_step(model,inputs,*pos,**kw)
            gradients=audit['first_update_euler_gradient_norms']
            target=output/'S4_first_update_gradients.json'
            if not args.resume and gradients and not target.exists():
                assert set(gradients)=={'1','2','3','4'} and all(float(v)>0 for v in gradients.values())
                with target.open('w') as stream:
                    stream.write(json.dumps(gradients,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())
            return result

        def _save_checkpoint(self,*pos,**kw):
            snapshot=read(ROOT/'STORAGE_ADMISSION.json');unique={}
            for p in ROOT.rglob('*'):
                if p.is_file() and not p.is_symlink():
                    s=p.stat();unique[s.st_dev,s.st_ino]=s.st_size
            used=sum(unique.values())
            live=int(subprocess.check_output(['/usr/bin/df','-B1','--output=avail',snapshot['quota_scope_path']],text=True).splitlines()[-1])
            conservative=snapshot['available_bytes_at_freeze']-max(0,used-snapshot['owned_bytes_at_freeze'])
            required=donor['complete_trainer_bytes']+10*2**30
            with (output/'checkpoint_capacity.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(unit=self.state.global_step,df_available_bytes=live,
                    conservative_owned_delta_headroom_bytes=conservative,required_bytes=required,quota_report=snapshot['quota_report']))+'\n')
            assert min(live,conservative)>=required,'CHECKPOINT_QUOTA_HEADROOM'
            result=super()._save_checkpoint(*pos,**kw)
            checkpoint=output/f'checkpoint-{self.state.global_step}'
            bind_checkpoint_metadata(output,checkpoint,stats,dict(arm=args.arm,statistics_sha256=sha(stats),release=card['release'],donor_revision=donor['revision'],plan_sha256=sha(ROOT/'PAIRING_PLAN.json')))
            return result
    experiment.Gr00tTrainer=SpaceCheckedTrainer
    old.run_official(config,GROOT)
    assert audit['forwards']==2*args.limit
    assert read(output/f'checkpoint-{args.limit}/trainer_state.json')['global_step']==args.limit
    if audit['first_update_euler_gradient_norms']:
        (output/'S4_first_update_gradients.json').write_text(json.dumps(audit['first_update_euler_gradient_norms'],sort_keys=True)+'\n')
    print(json.dumps(dict(status='PI0268_TRAINING_SEGMENT_COMPLETE',arm=args.arm,phase=args.phase,updates=args.limit)),flush=True)


def compare(arm):
    import pi0264_compare_canary as mature
    routes=read(ROOT/'CANARY_ROUTES.json') if (ROOT/'CANARY_ROUTES.json').exists() else {}
    split_phase=routes.get(arm,{}).get('split','split')
    mature.compare(arm,root=ROOT,max_steps=2048,split_phase=split_phase,status='PASS_PI0268_NUMERICAL_CANARY_PENDING_RNG_LIFECYCLE')
    def identical(a,b):
        if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
        if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
        if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(identical(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(identical(x,y) for x,y in zip(a,b))
        return type(a)==type(b) and a==b
    paths=[ROOT/arm/phase/'training/checkpoint-8/rng_state.pth' for phase in ('continuous',split_phase)]
    rng=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
    equal={k:identical(rng[0][k],rng[1][k]) for k in rng[0]};assert all(equal.values()),('RESTORED_RNG_STATE_MISMATCH',equal)
    target=ROOT/arm/'canary_compare.json';result=read(target)
    result.update(status='PASS_PI0268_REAL_TWO_MICROBATCH_EXACT_RESUME_CANARY',saved_RNG_states_equal=equal,split_phase=split_phase)
    target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('OLD','ROT'),required=True);p.add_argument('--phase',choices=('continuous','split','split_recovery1','main'))
    p.add_argument('--limit',type=int);p.add_argument('--resume',type=int,default=0);p.add_argument('--compare',action='store_true');args=p.parse_args()
    if args.compare:compare(args.arm)
    else:
        assert args.phase and 0<=args.resume<args.limit<=2048
        run(args)
