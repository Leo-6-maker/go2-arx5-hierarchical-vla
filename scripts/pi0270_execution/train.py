"""PI0270 AGG: qualified PREFIX recipe, sole learner-state data intervention."""
import argparse
import ctypes
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

ROOT=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0270')
CONTROL=ROOT.with_name('PI0269')
CONTROL_SAMPLE_SHA='1bc54a47f82bff833609f12c74e377ee8a22aaa0a9292021ce57cb9f44efa0c7'
CONTROL_RNG_SHA='2e09c0275b260df5aac088156a45766917daf7b3215e4ba4645943a7e7b7a083'
R3=ROOT.with_name('PI0260')/'R3';OLD=ROOT.with_name('PI0264')
STAGE=Path('/scratch/e1192548/stage22_h200_runtime_deb920_20260902')
GROOT=STAGE/'src/isaac-gr00t'
sys.path[:0]=[str(Path(__file__).resolve().parent),str(CONTROL/'code'),str(ROOT.with_name('PI0267')/'code'),str(OLD/'code'),str(R3/'execution'),
              '/scratch/e1192548/pi0244_fixed16_10k_v1/scripts','/scratch/e1192548/pi0244_fixed16_10k_v1/src',str(GROOT)]
import pi0264_train as paired
assert Path(paired.__file__).resolve()==ROOT.with_name('PI0267')/'code/pi0264_train.py'
from pi0260_r3_inputs import batch,identity
from pi0264_plan import NONLANGUAGE


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
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
    target=checkpoint/'PI0270_POLICY_CONTRACT.json'
    content=json.dumps(contract,sort_keys=True)+'\n'
    if target.exists():assert target.read_text()==content
    else:target.write_text(content)


def language(plan,u,c):
    colors=plan['source_family_colors'][u['family']]
    if u['task']=='M1':return plan['templates'][u['template']].format(color=colors[c])
    start,end,direction=('black','white','counterclockwise') if c==0 else ('white','black','clockwise')
    return f'Approach the {start} endpoint, then follow the open contour {direction} to the {end} endpoint as seen from above.'


def require_comparable(condition,reason,details):
    if condition:return
    path=ROOT/'evidence/CONTROL_NOT_COMPARABLE.json'
    if not path.exists():path.write_text(json.dumps(dict(status='CONTROL_NOT_COMPARABLE',reason=reason,details=details),indent=2,sort_keys=True)+'\n')
    raise RuntimeError(('CONTROL_NOT_COMPARABLE',reason,details))


class Pairs(IterableDataset):
    def __init__(self,processor,output,arm,resume):
        assert arm=='AGG'
        self.processor,self.output,self.arm=processor,output,arm;self.seed=26741;self.cursor=2*resume
        self.plan=read(ROOT/'PAIRING_PLAN.json');assert len(self.plan['units'])==4096
        assert sha(ROOT/'PAIRING_PLAN.json')=='fca9a29e69e4b63ec35ab995bacc60a4867e4b86a11e2d7a3c6397d4d316f5ef'
        rows=read(ROOT/'RECOVERY_MAPPING.json')['rows'];assert len(rows)==4096
        self.mapping={(r['unit'],r['micro']):r for r in rows};assert len(self.mapping)==4096
        self.control=read_lines(CONTROL/'PREFIX/main/training/sample_audit.jsonl')
        assert sha(CONTROL/'PREFIX/main/training/sample_audit.jsonl')==CONTROL_SAMPLE_SHA
        assert len(self.control)==8192

    def reset_seed(self,seed):
        self.cursor=2*(seed-self.seed);assert 0<=self.cursor<8192 and self.cursor%2==0

    @lru_cache(maxsize=24)
    def family(self,fid):
        folder=ROOT.with_name('PI0267')/'source/S2'/fid
        complete=read(folder/'COMPLETE.json');checks=read(folder/'codec_admission.json');assert checks['pass_admission']
        for name in ('proposals.npz','policy_frames.npz','rows.json'):assert sha(folder/name)==complete['hashes'][name]
        return np.load(folder/'proposals.npz'),np.load(folder/'policy_frames.npz'),read(folder/'rows.json')

    @lru_cache(maxsize=256)
    def recovery(self,uid,digest):
        folder=ROOT/'source/recovery';path=folder/uid/'sample.npz'
        assert path.resolve().is_relative_to(folder.resolve()) and not path.is_symlink() and sha(path)==digest
        with np.load(path,allow_pickle=False) as z:
            return {k:z[k].copy() for k in ('body_rgb','wrist_rgb','state_v1','raw_action')}

    def __iter__(self):
        for u in self.plan['units'][self.cursor//2:4096]:
            for micro,c in enumerate(u['order']):
                text=language(self.plan,u,c);i=u['R_indices'][c];replacement=None
                if u['slot']=='COMMON_NOMINAL':
                    z,fr,rows=self.family(u['family']);row=rows[i]
                    assert row['task']==u['task'] and row['command']==c and row['anchor_index']==u['anchor'] and row['precodec_valid']
                    assert row['nominal'] and i==u['N_indices'][c]
                    j=row['frame_index'];packet=dict(body_rgb=fr['body_rgb'][j],wrist_rgb=fr['wrist_rgb'][j],state_v1=z['state_v1'][i],language=text)
                    action=z['raw_action'][i];frame_key=row['frame_key']
                else:
                    assert u['slot']=='RECOVERY_INTERVENTION'
                    replacement=self.mapping[u['unit'],micro]
                    assert (replacement['family'],replacement['task'],replacement['command'],replacement['language'])==(u['family'],u['task'],c,text)
                    sample=self.recovery(replacement['source_uid'],replacement['sample_sha256'])
                    packet={k:sample[k] for k in ('body_rgb','wrist_rgb','state_v1')};packet['language']=text
                    action=sample['raw_action'];frame_key=None
                value=batch(self.processor,packet,action)
                condition={k:v for k,v in value['inputs'].items() if k not in ('action','action_mask')};key=identity(condition)
                labels=identity({k:value['inputs'][k] for k in ('action','action_mask')})
                nonlanguage=identity({k:value['inputs'][k] for k in NONLANGUAGE})
                old=self.control[2*(u['unit']-1)+micro]
                for name,expected in dict(unit=u['unit'],micro=micro,task=u['task'],family=u['family'],slot=u['slot'],command=c,template=u['template'],language=text).items():
                    require_comparable(old[name]==expected,'CONTROL_SCHEDULE_'+name,dict(unit=u['unit'],micro=micro))
                if replacement is None:
                    for name,actual in dict(condition_sha256=key,nonlanguage_sha256=nonlanguage,labels_masks_sha256=labels).items():
                        require_comparable(old[name]==actual,'COMMON_INPUT_'+name,dict(unit=u['unit'],micro=micro))
                else:
                    assert key==replacement['condition_sha256'] and labels==replacement['labels_masks_sha256'],'RECOVERY_PROCESSOR_INPUT_OR_LABEL_MISMATCH'
                value['_pi0264_meta']=dict(unit=u['unit'],micro=micro,command=c,condition_sha256=key)
                record=dict(unit=u['unit'],micro=micro,arm=self.arm,task=u['task'],family=u['family'],slot=u['slot'],command=c,
                    sequence=u['family'],index=i,condition_sha256=key,original_unit=f"{u['task']}:{u['bank_index']}",visit=u['visit'],
                    color=self.plan['source_family_colors'][u['family']][c],template=u['template'],language=text,frame_key=frame_key,
                    nonlanguage_sha256=nonlanguage,labels_masks_sha256=labels,original_index_is_provenance_only=replacement is not None,
                    actual_source_uid=replacement['source_uid'] if replacement else None,
                    source_sample_sha256=replacement['sample_sha256'] if replacement else None)
                with (self.output/'sample_audit.jsonl').open('a') as f:f.write(json.dumps(record,sort_keys=True)+'\n')
                yield value


def run(args):
    assert os.environ.get('PBS_JOBID') and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    # Match the frozen Corelab codec's CPU kernel; scientific dtypes stay unchanged.
    assert os.environ.get('OPENBLAS_CORETYPE')=='Prescott'
    library=next((Path(np.__file__).parent.parent/'numpy.libs').glob('*openblas*'))
    core=ctypes.CDLL(str(library)).openblas_get_corename64_;core.restype=ctypes.c_char_p
    assert core().decode()=='Prescott'
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    assert torch.cuda.device_count()==1 and torch.cuda.get_device_properties(0).total_memory>=100*(1<<30)
    seal=read(ROOT/'EXECUTION_FILESET.json')
    for r in seal['files']:assert Path(r['path']).stat().st_size==r['bytes'] and sha(r['path'])==r['sha256']
    assert read(ROOT/'evidence/HOPPER_CPU_ADMISSION.json')['status']=='PASS_PI0270_ACTUAL_FROZEN_MAPPING_AND_CONTROL_INPUTS'
    assert read(ROOT/'evidence/DATA_COVERAGE_ADMISSION.json')['status']=='PASS_ALL96_FIXED_NEW_SUPPORT_CELLS'
    assert read(ROOT/'evidence/NEWFIT_ADMISSION.json')['status']=='PASS_FIXED_NEWFIT192_AVAILABLE'
    assert read(CONTROL/'evidence/HOPPER_PHYSICAL_FIXTURES.json')['status']=='PASS_ROT_CONTRACT_PHYSICAL_AND_WEIGHTED_FIXTURES'
    for arm in ('AGG',):
        assert read(ROOT/'evidence'/('RUNTIME_BINDING_'+arm+'.json'))['status']=='PASS_EXACT_FROZEN_HELPER_IMPORTS_AND_CPU_HOOK_INSTALL'
    card=read(ROOT/'design/APPROVED_RUN_CARD.json')
    assert card['launch_allowed'] and card['release']=='7a6eea5ae38854bec7b80782213a92ca02fd0318'
    stats=ROOT/'contracts'/'ROT'/'source/dataset/statistics.json'
    assert sha(stats)==card['statistics_sha256']
    donor=read(ROOT/'DONOR_LOCAL_READBACK.json');assert donor['status']=='PASS_PI0269_FIXED_ROT2048_COMPLETE_TRAINER_BINDING'
    assert donor['revision']=='97d8bf027f07241e2206b05e07597e4719c891e4' and donor['tree_sha256']=='3db80d35d5e476fe5a90d540ff8959fb5edf3da43a988fde7f88ce9f54b9b147'
    output=ROOT/args.arm/args.phase/'training';output.mkdir(parents=True,exist_ok=True)
    encoder_receipt=dict(core=core().decode(),library=str(library),library_sha256=sha(library),
        numpy_version=np.__version__,scientific_dtypes_unchanged=True)
    encoder_path=output/'CPU_ENCODER_KERNEL.json';encoder_text=json.dumps(encoder_receipt,sort_keys=True)+'\n'
    if encoder_path.exists():assert encoder_path.read_text()==encoder_text
    else:encoder_path.write_text(encoder_text)
    if args.phase=='main':assert read(ROOT/args.arm/'canary_compare.json')['status']=='PASS_PI0270_REAL_TWO_MICROBATCH_EXACT_RESUME_CANARY'
    if args.resume:
        cp=output/f'checkpoint-{args.resume}';assert read(cp/'trainer_state.json')['global_step']==args.resume
        assert all((cp/n).is_file() for n in ('optimizer.pt','scheduler.pt','rng_state.pth'))
        for name in ('S4_audit.jsonl','sample_audit.jsonl','step_rng.jsonl'):
            assert len((output/name).read_text().splitlines())==2*args.resume
    else:assert not (output/'S4_audit.jsonl').exists()
    paired.ROOT=ROOT;paired.seed_for=seed_for;paired.Pairs=Pairs
    paired.r3.ROOT=ROOT/'contracts'/'ROT'
    # Reuse r3's actual processor/loss construction; its old visit table is not consumed.
    paired.r3.Visits=lambda processor,*unused:SimpleNamespace(processor=processor)
    import pi0228_h200_large_adapt_v1 as old
    from stage40_b0_model_control import _bind_local_cosmos_processor
    assert _bind_local_cosmos_processor(STAGE,paired.COSMOS)['resolved_snapshot']==str(paired.COSMOS.resolve())
    config_args=argparse.Namespace(stage10_root=STAGE,base_model_path=Path(donor['path']),
        dataset_dir=ROOT.with_name('PI0227')/'pi0227_large_adapt_v1_inputs/train_dataset',
        modality_config=Path('/scratch/e1192548/pi0227_h200_prep_20260919/inputs/go2_arx5_modality.py'),groot_source=GROOT)
    save_steps=2048 if args.phase=='main' else (4 if args.phase.startswith('split') else 8)
    config=old.build_config(config_args,output,steps=4096,save_steps=save_steps,save_limit=2)
    config.training.learning_rate=1e-5;config.training.lr_scheduler_type='cosine';config.training.gradient_accumulation_steps=2
    config.training.max_steps=4096;config.training.save_steps=save_steps;config.training.save_total_limit=2
    config.training.resume_from_checkpoint=bool(args.resume);config.data.seed=config.training.seed=26741;config.training.remove_unused_columns=False
    assert config.training.warmup_steps==10 and config.training.max_grad_norm==1. and config.training.weight_decay==1e-5
    audit=paired.install(args,output,raw_head_cap=80000,raw_update_cap=4368)
    from temporal import install as install_temporal
    install_temporal('PREFIX',output)  # Exact inherited reduction; recipe label stays PREFIX.
    from gr00t.experiment import experiment
    base_trainer=experiment.Gr00tTrainer
    assert sha(CONTROL/'PREFIX/main/training/step_rng.jsonl')==CONTROL_RNG_SHA
    control_rng=read_lines(CONTROL/'PREFIX/main/training/step_rng.jsonl');assert len(control_rng)==8192
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
            rng_path=output/'step_rng.jsonl';position=rng_path.stat().st_size if rng_path.exists() else 0
            result=super().training_step(model,inputs,*pos,**kw)
            with rng_path.open() as stream:
                stream.seek(position);actual=json.loads(stream.readline());assert not stream.readline()
            expected=control_rng[2*(actual['unit']-1)+actual['micro']]
            for name in ('unit','micro','command','seed','native_epsilon','native_tau','stock_initial_noise','mode_restored'):
                require_comparable(actual[name]==expected[name],'ACTUAL_RANDOM_PACKET_'+name,dict(unit=actual['unit'],micro=actual['micro']))
            assert np.isfinite(actual['loss']),'PI0270_NONFINITE_OPTIMIZATION_CORRECTNESS_STOP'
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
            bind_checkpoint_metadata(output,checkpoint,stats,dict(arm=args.arm,statistics_sha256=sha(stats),release=card['release'],donor_revision=donor['revision'],plan_sha256=sha(ROOT/'PAIRING_PLAN.json'),recovery_mapping_sha256=sha(ROOT/'RECOVERY_MAPPING.json'),control_sample_sha256=CONTROL_SAMPLE_SHA,control_rng_sha256=CONTROL_RNG_SHA,cpu_encoder_kernel=encoder_receipt))
            return result
    experiment.Gr00tTrainer=SpaceCheckedTrainer
    old.run_official(config,GROOT)
    assert audit['forwards']==2*args.limit
    assert read(output/f'checkpoint-{args.limit}/trainer_state.json')['global_step']==args.limit
    if audit['first_update_euler_gradient_norms']:
        (output/'S4_first_update_gradients.json').write_text(json.dumps(audit['first_update_euler_gradient_norms'],sort_keys=True)+'\n')
    print(json.dumps(dict(status='PI0270_TRAINING_SEGMENT_COMPLETE',arm=args.arm,phase=args.phase,updates=args.limit)),flush=True)


def compare(arm):
    import pi0264_compare_canary as mature
    routes=read(ROOT/'CANARY_ROUTES.json') if (ROOT/'CANARY_ROUTES.json').exists() else {}
    split_phase=routes.get(arm,{}).get('split','split')
    mature.compare(arm,root=ROOT,max_steps=4096,split_phase=split_phase,status='PASS_PI0270_NUMERICAL_CANARY_PENDING_RNG_LIFECYCLE')
    def identical(a,b):
        if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
        if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
        if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(identical(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(identical(x,y) for x,y in zip(a,b))
        return type(a)==type(b) and a==b
    paths=[ROOT/arm/phase/'training/checkpoint-8/rng_state.pth' for phase in ('continuous',split_phase)]
    rng=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
    equal={k:identical(rng[0][k],rng[1][k]) for k in rng[0]};assert all(equal.values()),('RESTORED_RNG_STATE_MISMATCH',equal)
    for name in ('temporal_components.jsonl','full_loss.jsonl','update_gradient_norms.jsonl'):
        values=[read_lines(ROOT/arm/phase/'training'/name) for phase in ('continuous',split_phase)]
        assert values[0]==values[1],('EXACT_CANARY_COMPONENTS_OR_GRADIENT_MISMATCH',name)
    target=ROOT/arm/'canary_compare.json';result=read(target)
    result.update(status='PASS_PI0270_REAL_TWO_MICROBATCH_EXACT_RESUME_CANARY',saved_RNG_states_equal=equal,split_phase=split_phase)
    target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('AGG',),required=True);p.add_argument('--phase',choices=('continuous','split','main'))
    p.add_argument('--limit',type=int);p.add_argument('--resume',type=int,default=0);p.add_argument('--compare',action='store_true');args=p.parse_args()
    if args.compare:compare(args.arm)
    else:
        assert args.phase and 0<=args.resume<args.limit<=4096
        run(args)
