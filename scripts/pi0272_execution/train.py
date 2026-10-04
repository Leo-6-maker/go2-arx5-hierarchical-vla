"""Matched PI0272 arms on the qualified Trainer; no optimizer inheritance."""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0272')
spec=importlib.util.spec_from_file_location('pi0270_qualified_train',ROOT.with_name('PI0270')/'code/train.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
np,torch,paired,read,sha,read_lines=old.np,old.torch,old.paired,old.read,old.sha,old.read_lines
sys.path.insert(0,str(Path(__file__).resolve().parent))
from prepare import seed_for,STATS_SHA,RELEASE
from torch.utils.data import IterableDataset


class Pairs(IterableDataset):
    def __init__(self,processor,output,arm,resume):
        self.processor,self.output,self.arm=processor,output,arm
        self.seed,self.cursor=27202,2*resume
        self.card=read(ROOT/'design/FITTING_CARD.json');self.rows={r['input_id']:r for r in self.card['rows']}
        assert len(self.rows)==296 and len(self.card['stream'])==2048
    def reset_seed(self,seed):
        self.cursor=2*(seed-self.seed);assert 0<=self.cursor<2048 and self.cursor%2==0
    def __iter__(self):
        for visit in self.card['stream'][self.cursor:]:
            row=self.rows[visit['input_id']];source=row['source_mapping'];path=Path(row['hopper_input'])
            assert path.resolve().is_relative_to(ROOT.with_name('PI0270')/'source/recovery') and sha(path)==row['input_sha256']
            with np.load(path,allow_pickle=False) as z:
                packet={k:z[k].copy() for k in ('body_rgb','wrist_rgb','state_v1')};packet['language']=row['language']
                value=old.batch(self.processor,packet,z['raw_action'].copy())
            condition={k:v for k,v in value['inputs'].items() if k not in ('action','action_mask')}
            labels={k:value['inputs'][k] for k in ('action','action_mask')}
            assert old.identity(condition)==row['condition_sha256'] and old.identity(labels)==row['labels_masks_sha256']
            assert visit['seed']==seed_for(visit['unit'])
            value['_pi0264_meta']=dict(unit=visit['unit'],micro=visit['micro'],command=source['command'],condition_sha256=row['condition_sha256'])
            record=dict(**visit,arm=self.arm,task=source['task'],command=source['command'],family=source['family'],
                bucket=source['bucket'],condition_sha256=row['condition_sha256'],labels_masks_sha256=row['labels_masks_sha256'],
                source_sample_sha256=sha(path),source_unit=source['unit'],source_micro=source['micro'])
            with (self.output/'sample_audit.jsonl').open('a') as f:f.write(json.dumps(record,sort_keys=True)+'\n')
            yield value


def hooks(args,output):
    paired.ROOT=ROOT;paired.Pairs=Pairs;paired.seed_for=seed_for
    paired.r3.ROOT=ROOT/'contracts/ROT'
    paired.r3.Visits=lambda processor,*unused:SimpleNamespace(processor=processor)
    if args.arm=='DIRECT':
        import pi0247_u_rollout_overlay as overlay
        from readout import install
        overlay.install=install
    elif not args.resume:
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
        stock=Gr00tN1d7ActionHead.get_action
        def capture_reference(self,backbone,condition,*a,**kw):
            assert 'action' not in condition
            result=stock(self,backbone,condition,*a,**kw)
            target=output/'S4_first_reference.npz'
            with target.open('xb') as f:np.savez(f,raw=result.action_pred.detach().float().cpu().numpy(),state=condition.state.detach().float().cpu().numpy())
            return result
        Gr00tN1d7ActionHead.get_action=capture_reference
    audit=paired.install(args,output,raw_head_cap=32768,raw_update_cap=2208)
    from temporal import install
    install('PREFIX',output)
    if args.arm=='S4FIT' and not args.resume:
        import pi0247_u_rollout_overlay as overlay
        physical=overlay._physical_losses
        def capture_generated(g,*a,**kw):
            target=output/'S4_first_generated.npz'
            if not target.exists():
                with target.open('xb') as f:np.savez(f,raw=g.detach().float().cpu().numpy())
            return physical(g,*a,**kw)
        overlay._physical_losses=capture_generated
    return audit


def run(args):
    resolution=read(ROOT/'evidence/PRECISION_CONTRACT_RESOLUTION.json')
    assert resolution['status']=='RESOLVED_BY_EXPLICIT_SCIENTIFIC_AUTHORITY' and resolution['launch_allowed']
    assert os.environ.get('PBS_JOBID') and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    assert os.environ.get('OPENBLAS_CORETYPE')=='Prescott'
    assert torch.cuda.device_count()==1 and torch.cuda.get_device_properties(0).total_memory>=100*2**30
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    for r in read(ROOT/'EXECUTION_FILESET.json')['files']:assert sha(r['path'])==r['sha256']
    assert read(ROOT/'evidence/HOPPER_CPU_ADMISSION.json')['status']=='PASS_PI0272_BOTH_FULL_2048_ACTUAL_PROCESSOR_STREAMS'
    assert read(ROOT/'evidence'/f'RUNTIME_BINDING_{args.arm}.json')['status']=='PASS_PI0272_ACTUAL_FROZEN_HELPERS_AND_HOOKS'
    stats=ROOT/'contracts/ROT/source/dataset/statistics.json';assert sha(stats)==STATS_SHA
    donor=read(ROOT/'DONOR_LOCAL_READBACK.json');assert donor['revision']=='f862abb26041c8e7d355470d6e9d48899f78b9ce'
    output=ROOT/args.arm/args.phase/'training';output.mkdir(parents=True,exist_ok=True)
    if args.phase=='main':
        for arm in ('S4FIT','DIRECT'):assert read(ROOT/arm/'canary_compare.json')['status']=='PASS_PI0272_REAL_EXACT_RESUME_CANARY'
    if args.resume:
        cp=output/f'checkpoint-{args.resume}';assert read(cp/'trainer_state.json')['global_step']==args.resume
        assert all((cp/n).is_file() for n in ('optimizer.pt','scheduler.pt','rng_state.pth','PI0272_READOUT_CONTRACT.json'))
        for n in ('sample_audit.jsonl','step_rng.jsonl','S4_audit.jsonl'):assert len(read_lines(output/n))==2*args.resume
    else:assert not (output/'S4_audit.jsonl').exists()
    import pi0228_h200_large_adapt_v1 as native
    from stage40_b0_model_control import _bind_local_cosmos_processor
    assert _bind_local_cosmos_processor(old.STAGE,paired.COSMOS)['resolved_snapshot']==str(paired.COSMOS.resolve())
    config_args=argparse.Namespace(stage10_root=old.STAGE,base_model_path=Path(donor['path']),
        dataset_dir=ROOT.with_name('PI0227')/'pi0227_large_adapt_v1_inputs/train_dataset',
        modality_config=Path('/scratch/e1192548/pi0227_h200_prep_20260919/inputs/go2_arx5_modality.py'),groot_source=old.GROOT)
    save=512 if args.phase=='main' else (4 if args.phase=='split' else 8)
    config=native.build_config(config_args,output,steps=1024,save_steps=save,save_limit=2)
    config.training.learning_rate=1e-5;config.training.lr_scheduler_type='cosine';config.training.gradient_accumulation_steps=2
    config.training.max_steps=1024;config.training.save_steps=save;config.training.save_total_limit=2
    config.training.resume_from_checkpoint=bool(args.resume);config.data.seed=config.training.seed=27202
    config.training.remove_unused_columns=False
    assert config.training.warmup_steps==10 and config.training.max_grad_norm==1. and config.training.weight_decay==1e-5
    audit=hooks(args,output)
    from gr00t.experiment import experiment
    class Trainer(experiment.Gr00tTrainer):
        def training_step(self,model,inputs,*pos,**kw):
            if not getattr(self,'scope_checked',False):
                scope={kind:[dict(name=n,shape=list(p.shape),dtype=str(p.dtype)) for n,p in model.named_parameters() if p.requires_grad==enabled]
                       for kind,enabled in (('trainable',True),('frozen',False))}
                assert len(scope['trainable'])==471 and len(scope['frozen'])==559
                donor_scope=read(ROOT/'DONOR_PARAMETER_SCOPE.json')
                for kind in scope:
                    actual=sorted((p['name'],p['shape']) for p in scope[kind])
                    assert actual==sorted((p['name'],p['shape']) for p in donor_scope[kind]),'INITIALIZER_SCOPE_NAME_SHAPE'
                    expected_dtype={'BF16':'torch.bfloat16','F32':'torch.float32'}
                    assert sorted((p['name'],p['dtype']) for p in scope[kind])==sorted((p['name'],expected_dtype[p['stored_dtype']]) for p in donor_scope[kind])
                optimized={id(p) for g in self.optimizer.param_groups for p in g['params']}
                assert optimized=={id(p) for p in model.parameters() if p.requires_grad}
                target=ROOT/'PARAMETER_SCOPE.json'
                if target.exists():assert read(target)==scope
                else:target.write_text(json.dumps(scope,sort_keys=True)+'\n')
                self.scope_checked=True
            if args.resume and not getattr(self,'rng_synced',False):
                assert self.args.dataloader_num_workers==0 and self.state.global_step==args.resume
                self._load_rng_state(str(output/f'checkpoint-{args.resume}'));self.rng_synced=True
            result=super().training_step(model,inputs,*pos,**kw)
            assert audit['native_mode_snapshot']==[m.training for m in model.action_head.modules()]
            row=read_lines(output/'step_rng.jsonl')[-1];assert np.isfinite(row['loss'])
            if args.arm=='DIRECT':
                other=ROOT/'S4FIT'/args.phase/'training/step_rng.jsonl'
                expected=read_lines(other)[2*(row['unit']-1)+row['micro']]
                for k in ('unit','micro','command','condition_sha256','seed','native_epsilon','native_tau','stock_initial_noise','mode_restored'):
                    assert row[k]==expected[k],('MATCHED_PACKET',k,row['unit'])
            gradients=audit['first_update_euler_gradient_norms'];target=output/'S4_first_update_gradients.json'
            if gradients and not target.exists():
                assert set(gradients)==({'1'} if args.arm=='DIRECT' else {'1','2','3','4'})
                assert all(np.isfinite(v) and v>0 for v in gradients.values())
                target.write_text(json.dumps(gradients,sort_keys=True)+'\n')
            return result
        def _save_checkpoint(self,*pos,**kw):
            import shutil
            snapshot=read(ROOT/'STORAGE_ADMISSION.json');unique={}
            for p in ROOT.rglob('*'):
                if p.is_file() and not p.is_symlink():
                    s=p.stat();unique[s.st_dev,s.st_ino]=s.st_size
            conservative=snapshot['available_bytes_at_freeze']-max(0,sum(unique.values())-snapshot['owned_bytes_at_freeze'])
            assert min(conservative,shutil.disk_usage(ROOT).free)>=donor['complete_trainer_bytes']+10*2**30
            result=super()._save_checkpoint(*pos,**kw);cp=output/f'checkpoint-{self.state.global_step}'
            contract=read(ROOT/'design'/f'{args.arm}_READOUT.json')
            old.bind_checkpoint_metadata(output,cp,stats,dict(release=RELEASE,mode=args.arm,donor_revision=donor['revision'],
                fitting_card_sha256=sha(ROOT/'design/FITTING_CARD.json'),legacy_metadata_filename_only=True))
            (cp/'PI0272_READOUT_CONTRACT.json').write_text(json.dumps(contract,sort_keys=True)+'\n')
            for name in ('EXECUTION_FILESET.json','PARAMETER_SCOPE.json','DONOR_PARAMETER_SCOPE.json','design/FITTING_CARD.json','STORAGE_ADMISSION.json'):
                target=cp/'pi0272_provenance'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
            return result
    experiment.Gr00tTrainer=Trainer
    native.run_official(config,old.GROOT)
    assert audit['forwards']==2*args.limit and read(output/f'checkpoint-{args.limit}/trainer_state.json')['global_step']==args.limit
    print('PI0272_SEGMENT_COMPLETE',args.arm,args.phase,args.limit,flush=True)


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b


def compare(arm):
    from pi0260_r3_compare_canary import model_delta,optimizer_delta
    a,b=[ROOT/arm/p/'training' for p in ('continuous','split')];x,y=[p/'checkpoint-8' for p in (a,b)]
    for cp in (x,y):assert (read(cp/'trainer_state.json')['global_step'],read(cp/'trainer_state.json')['max_steps'])==(8,1024)
    for name in ('sample_audit.jsonl','step_rng.jsonl','full_loss.jsonl','temporal_components.jsonl','update_gradient_norms.jsonl'):
        assert read_lines(a/name)==read_lines(b/name),name
    parity='PASS_PI0272_DIRECT_READOUT_PARITY' if arm=='DIRECT' else 'PASS_PI0247_U_S4_PARITY'
    for p in (a,b):assert read(p/'S4_parity.json')['status']==parity
    assert read(a/'S4_first_update_gradients.json')==read(b/'S4_first_update_gradients.json')
    md,mt=model_delta(x,y);od,ot=optimizer_delta(x,y);assert md==od==0
    for name in ('rng_state.pth','scheduler.pt'):
        assert equal(torch.load(x/name,map_location='cpu',weights_only=False),torch.load(y/name,map_location='cpu',weights_only=False)),name
    result=dict(status='PASS_PI0272_REAL_EXACT_RESUME_CANARY',arm=arm,units=8,microbatches=16,
        model_max_abs=md,optimizer_max_abs=od,model_tensors=mt,optimizer_tensors=ot,saved_rng_scheduler_exact=True)
    (ROOT/arm/'canary_compare.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('S4FIT','DIRECT'),required=True)
    p.add_argument('--phase',choices=('continuous','split','main'));p.add_argument('--limit',type=int);p.add_argument('--resume',type=int,default=0)
    p.add_argument('--compare',action='store_true');args=p.parse_args()
    if args.compare:compare(args.arm)
    else:
        assert args.phase and 0<=args.resume<args.limit<=1024
        run(args)
