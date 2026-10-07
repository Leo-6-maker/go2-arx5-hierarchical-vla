"""Typed PI0277 run binding around the source-bound, qualified S4 math."""
import argparse, ctypes, hashlib, importlib.util, json, os, sys, time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import cpu_inputs as data

RELEASE='83af9099f3bab9a4ac08eaa85380c2cfd6351f34'

@dataclass(frozen=True)
class Binding:
    root: Path
    arm: str
    phase: str
    route: str
    limit: int
    resume: int=0
    def __post_init__(self):
        assert self.arm=='COVER93' and self.route in ('REF_W0','FAST_CACHE0','FAST_CACHE2')
        assert self.phase in ('reference','continuous','split','main')
        assert 0<=self.resume<self.limit<=4096
        if self.phase in ('reference','continuous','split'):assert self.limit in (4,8)
        if self.phase=='main':assert self.limit==4096
    @property
    def output(self):
        return self.root/self.arm/self.phase/'training'

def seed_for(unit):return int.from_bytes(hashlib.sha256(f'PI0278:COVER93:27803:{unit}'.encode()).digest()[:8],'big')%2**31

LOG_CURSORS={}
def lines(path):
    path=Path(path);s=path.stat();identity=(s.st_dev,s.st_ino)
    offset,rows,previous=LOG_CURSORS.get(path,(0,[],identity))
    assert identity==previous and s.st_size>=offset
    with path.open('rb') as f:
        f.seek(offset);extra=f.read();assert not extra or extra.endswith(b'\n')
        rows.extend(json.loads(line) for line in extra.splitlines() if line)
        LOG_CURSORS[path]=(f.tell(),rows,identity)
    return rows

def append(path,row):
    with Path(path).open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n');f.flush()

def fingerprint(value):
    """Byte equality, including dtype/shape and non-tensor optimizer/RNG fields."""
    import numpy as np
    import torch
    if isinstance(value,torch.Tensor):
        x=value.detach().cpu().contiguous()
        return dict(dtype=str(x.dtype),shape=list(x.shape),sha256=hashlib.sha256(x.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
    if isinstance(value,np.ndarray):return dict(dtype=str(value.dtype),shape=list(value.shape),sha256=hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value,dict):return {str(k):fingerprint(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return dict(sequence=type(value).__name__,items=[fingerprint(v) for v in value])
    assert value is None or type(value) in (int,float,str,bool),type(value)
    return value

def run(b):
    import random, shutil
    import numpy as np
    import torch
    from transformers import TrainerCallback
    from gripper import generated_gripper_loss
    from bench_cpu import tree_rss
    root=b.root;budget_root=root.parent;family_root=budget_root.parent;output=b.output;output.mkdir(parents=True,exist_ok=True)
    settings=data.read(root/'RUN_CONFIG.json');caps=settings['caps'];assert settings['release']==RELEASE
    assert os.environ.get('PBS_JOBID') and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    assert os.environ.get('OPENBLAS_CORETYPE')=='Prescott'
    assert torch.cuda.device_count()==1 and torch.cuda.get_device_properties(0).total_memory>=100*2**30
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    for row in data.read(root/'EXECUTION_FILESET.json')['files']:assert data.sha(row['path'])==row['sha256'],row['path']
    cfg=data.read(root/'design/INPUT_BINDING.json');sys.path[:0]=cfg['module_paths']
    import pi0264_train as paired
    assert str(Path(paired.__file__).resolve())==str(family_root/'PI0267'/'code/pi0264_train.py')
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
    from gr00t.experiment import experiment
    import pi0247_u_rollout_overlay as overlay
    donor=data.read(root/'DONOR_LOCAL_READBACK.json');assert donor['revision']==settings['donor_revision'] and not donor['optimizer_inherited']
    stats=Path(cfg['old_root'])/'contracts/TASK/source/dataset/statistics.json';assert data.sha(stats)==data.STATS_SHA
    library=next((Path(np.__file__).parent.parent/'numpy.libs').glob('*openblas*'))
    fn=ctypes.CDLL(str(library)).openblas_get_corename64_;fn.restype=ctypes.c_char_p;assert fn().decode()=='Prescott'
    data.write(output/'CPU_ENCODER_KERNEL.json',dict(core='Prescott',library=str(library),library_sha256=data.sha(library),numpy_version=np.__version__))
    if b.phase=='main':
        assert data.read(root/'evidence/LOADER_SELECTION.json')['route']==b.route
        assert data.read(root/b.arm/'canary_compare.json')['status']=='PASS_PI0278_REAL_EXACT_RESUME'
    if b.resume:
        cp=output/f'checkpoint-{b.resume}';assert data.read(cp/'trainer_state.json')['global_step']==b.resume
        assert data.read(cp/'PI0277_RUN.json')['route']==b.route
        for name in ('sample_audit.jsonl','step_rng.jsonl','S4_audit.jsonl'):assert len(lines(output/name))==2*b.resume
        assert data.read(output/'CURSOR.json')['next_unconsumed_micro']==2*b.resume
    else:assert not (output/'S4_audit.jsonl').exists(),'Existing attempted run must be adopted, never silently restarted'
    ledger=data.ItemLedger(budget_root/'runtime/cpu_items.jsonl',torch.multiprocessing.get_context('spawn'))
    paired.ROOT=budget_root;paired.seed_for=seed_for
    paired.Pairs=lambda processor,*unused:data.Inputs(root,b.route,ledger,b.arm+':'+b.phase,processor)
    paired.r3.ROOT=Path(cfg['old_root'])/'contracts/TASK'
    paired.r3.Visits=lambda processor,*unused:SimpleNamespace(processor=processor)
    stock=Gr00tN1d7ActionHead.get_action
    def reference(self,backbone,condition,*a,**kw):
        result=stock(self,backbone,condition,*a,**kw)
        with (output/'S4_first_reference.npz').open('xb') as f:np.savez(f,raw=result.action_pred.detach().float().cpu().numpy(),state=condition.state.detach().float().cpu().numpy())
        return result
    if not b.resume:Gr00tN1d7ActionHead.get_action=reference
    audit=paired.install(b,output,raw_head_cap=caps['training_heads'],raw_update_cap=caps['all_completed_updates'])
    from contexts import reusable
    overlay._module_eval=reusable(overlay._module_eval)
    original_append=overlay._append_json
    def durable_accounting(path,row,count):
        original_append(path,row,count)
        if path==budget_root/'raw_completed_updates.jsonl':
            with path.open('a') as f:f.flush();os.fsync(f.fileno())
    overlay._append_json=durable_accounting
    from temporal import install
    install('PREFIX',output)
    captured={};physical=overlay._physical_losses;euler=overlay._euler_rollout
    def capture_euler(*a,**kw):
        result=euler(*a,**kw)
        if kw.get('checkpointed') and torch.is_grad_enabled():captured['states']=result[1]
        return result
    overlay._euler_rollout=capture_euler
    def gripper_observation(g,e,m,*a,**kw):
        captured.update(generated=g,label=e,mask=m)
        if not b.resume and not (output/'S4_first_generated.npz').exists():
            with (output/'S4_first_generated.npz').open('xb') as f:np.savez(f,raw=g.detach().float().cpu().numpy())
        append(output/'gripper_generated_error.jsonl',dict(forward=audit['forwards']+1,command_fraction_MAE=float((g[...,9].detach().float()-e[...,9].detach().float()).abs().mean())/2))
        return physical(g,e,m,*a,**kw)
    overlay._physical_losses=gripper_observation
    original_head=Gr00tN1d7ActionHead.forward
    def final_objective(self,*a,**kw):
        captured.clear();result=original_head(self,*a,**kw)
        g,e,m=(captured[k] for k in ('generated','label','mask'))
        penalty=generated_gripper_loss(g,e,m)
        # Keep the original pose scalar and add the authorized gripper term once.
        result['loss']=result['loss']+.1*penalty
        add_box(result,g,m,captured['states'],output,audit['forwards'],b.resume)
        error=(g[...,9].detach().float()-e[...,9].detach().float()).abs()/2
        append(output/'gripper_objective.jsonl',dict(forward=audit['forwards'],unweighted=float(penalty.detach()),
            weighted=float((.1*penalty).detach()) if b.arm=='COVER93' else 0.,prefix_MAE=float(error[:,:8].mean()),full_MAE=float(error.mean()),total=float(result['loss'].detach())))
        if b.arm=='COVER93' and b.phase in ('reference','continuous','split') and not b.resume and audit['forwards']==1:
            states=[s for s in captured['states'] if s.requires_grad]
            assert len(states)==4 and states[-1] is g
            isolated=torch.autograd.grad(.1*penalty,states,retain_graph=True)
            norms=[float(x.float().norm()) for x in isolated];assert all(np.isfinite(n) and n>0 for n in norms)
            assert not isolated[-1][...,:9].any() and not isolated[-1][...,10:].any()
            data.write(output/'ISOLATED_GRIPPER_GRADIENT.json',dict(status='PASS_REAL_GENERATED_GRIPPER_FOUR_STEP_REACH',state_norms=norms,
                final_channel9_norm=float(isolated[-1][...,9].float().norm()),additional_S4_forward=False,recompute_heads_counted=True,native_supervision_retained=True))
        if b.phase in ('reference','continuous','split') and audit['forwards'] in (15,16):
            append(output/'OUTPUT_HASHES.jsonl',dict(micro_index=audit['forwards']-1,generated=fingerprint(g)))
        captured.clear()
        return result
    Gr00tN1d7ActionHead.forward=final_objective
    import pi0228_h200_large_adapt_v1 as native
    from stage40_b0_model_control import _bind_local_cosmos_processor
    assert _bind_local_cosmos_processor(paired.STAGE,paired.COSMOS)['resolved_snapshot']==str(paired.COSMOS.resolve())
    config_args=argparse.Namespace(stage10_root=paired.STAGE,base_model_path=Path(donor['path']),
        dataset_dir=family_root/'PI0227'/'pi0227_large_adapt_v1_inputs/train_dataset',
        modality_config=Path('/scratch/e1192548/pi0227_h200_prep_20260919/inputs/go2_arx5_modality.py'),groot_source=paired.GROOT)
    save=1024 if b.phase=='main' else (4 if b.phase=='split' else 4096)
    config=native.build_config(config_args,output,steps=4096,save_steps=save,save_limit=2)
    config.training.learning_rate=1e-5;config.training.lr_scheduler_type='cosine';config.training.gradient_accumulation_steps=2
    config.training.max_steps=4096;config.training.save_steps=save;config.training.save_total_limit=2
    config.training.resume_from_checkpoint=bool(b.resume);config.data.seed=config.training.seed=27803
    config.training.remove_unused_columns=False;config.training.dataloader_num_workers=2 if b.route=='FAST_CACHE2' else 0
    config.training.dataloader_pin_memory=True
    assert config.training.warmup_steps==10 and config.training.max_grad_norm==1. and config.training.weight_decay==1e-5
    timings={};trainer_ref={}
    class Events(TrainerCallback):
        def on_step_end(self,_args,state,control,**kw):
            torch.cuda.synchronize();now=time.perf_counter()
            append(output/'TIMINGS.jsonl',dict(unit=state.global_step,completed_seconds=now-timings.pop('start'),
                loader_wait_seconds=timings.pop('loader',0.),h2d_seconds=timings.pop('h2d',0.),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),process_tree_RSS_bytes=tree_rss()))
            data.write(output/'CURSOR.json',dict(completed_update=state.global_step,next_unconsumed_micro=2*state.global_step,queues_are_discardable=True))
            for path in [output/n for n in ('sample_audit.jsonl','step_rng.jsonl','S4_audit.jsonl')]+[budget_root/'raw_training_heads.jsonl',budget_root/'raw_completed_updates.jsonl']:
                if path.exists():
                    with path.open('a') as f:f.flush();os.fsync(f.fileno())
            if b.phase=='main' and state.global_step in (512,1024,2048,4096):control.should_save=True
            from datetime import datetime,timezone
            if (root/'runtime/STOP_TRAINING.json').exists():
                control.should_save=True;control.should_training_stop=True
            return control
    base=experiment.Gr00tTrainer
    class Trainer(base):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);self.add_callback(Events());trainer_ref['value']=self
        def get_train_dataloader(self):
            self.args.ignore_data_skip=True
            inner=data.loader(self.train_dataset,start=2*b.resume,pin_memory=True)
            class TimedLoader:
                def __len__(_self):return len(inner)
                def __iter__(_self):
                    iterator=iter(inner)
                    for index in range(2*b.resume,8192):
                        if index%2==0:torch.cuda.synchronize();timings['start']=time.perf_counter()
                        start=time.perf_counter()
                        try:value=next(iterator)
                        except StopIteration:return
                        timings['loader']=timings.get('loader',0.)+time.perf_counter()-start
                        yield value
                @property
                def dataset(_self):return inner.dataset
            return TimedLoader()
        def _prepare_inputs(self,inputs):
            def tensors(v):
                if isinstance(v,torch.Tensor):yield v
                elif isinstance(v,Mapping):
                    for x in v.values():yield from tensors(x)
                elif isinstance(v,(list,tuple)):
                    for x in v:yield from tensors(x)
            cpu=[t for t in tensors(inputs) if t.device.type=='cpu']
            assert cpu and all(t.is_pinned() for t in cpu),'ACTUAL_H200_PIN_MEMORY'
            if not (output/'ACTUAL_PIN_MEMORY.json').exists():
                data.write(output/'ACTUAL_PIN_MEMORY.json',dict(status='PASS_ACTUAL_PINNED_CPU_TENSORS',batch_type=type(inputs).__name__,tensors=len(cpu),all_pinned=True))
            torch.cuda.synchronize();start=time.perf_counter();result=super()._prepare_inputs(inputs);torch.cuda.synchronize()
            timings['h2d']=timings.get('h2d',0.)+time.perf_counter()-start
            return result
        def training_step(self,model,inputs,*a,**kw):
            visit=inputs.pop('_pi0276_visit');assert visit['micro_index']==2*self.state.global_step+visit['micro']
            if visit['micro']==0:
                counted=budget_root/'raw_completed_updates.jsonl'
                assert (len(lines(counted)) if counted.exists() else 0)<caps['all_completed_updates'],'COMPLETED_UPDATE_CAP_BEFORE_NEW_OPTIMIZER_WORK'
                if b.phase!='main':
                    start=data.read(root/'ADOPTION.json')['completed_updates_at_adoption']
                    recent=lines(counted)[start:] if counted.exists() else []
                    assert sum(r['phase']!='main' for r in recent)<80,'R2_QUALIFICATION_CAP'
            append(output/'sample_audit.jsonl',visit)
            if not getattr(self,'scope_checked',False):
                scope={k:[dict(name=n,shape=list(p.shape),dtype=str(p.dtype)) for n,p in model.named_parameters() if p.requires_grad==enable] for k,enable in [('trainable',True),('frozen',False)]}
                expected=data.read(root/'DONOR_PARAMETER_SCOPE.json')
                assert len(scope['trainable'])==471 and len(scope['frozen'])==559
                for k in scope:
                    assert sorted((p['name'],p['shape'],p['dtype']) for p in scope[k])==sorted((p['name'],p['shape'],{'BF16':'torch.bfloat16','F32':'torch.float32'}[p['stored_dtype']]) for p in expected[k])
                assert {id(p) for g in self.optimizer.param_groups for p in g['params']}=={id(p) for p in model.parameters() if p.requires_grad}
                data.write(output/'PARAMETER_SCOPE.json',scope);self.scope_checked=True
                def native_grip(module,unused,result):
                    value=float(result['action_loss'][...,9].detach().float().sum()/result['action_mask'][...,9].sum().clamp_min(1))
                    append(output/'gripper_native_loss.jsonl',dict(unit=self.state.global_step+1,native_velocity_MSE=value))
                model.action_head.register_forward_hook(native_grip)
                def gradient_hook(module,args,result):
                    if torch.is_grad_enabled() and isinstance(result,torch.Tensor) and result.requires_grad:
                        def record(gradient):append(output/'gripper_output_gradients.jsonl',dict(unit=self.state.global_step+1,norm=float(gradient[...,9].detach().float().norm())))
                        result.register_hook(record)
                model.action_head.action_decoder.register_forward_hook(gradient_hook)
                self.health_name,self.health_weight=min(((n,p) for n,p in model.named_parameters() if p.requires_grad and n.startswith('action_head.action_decoder.')),key=lambda item:item[1].numel())
                self.health_start=self.health_weight.detach().clone()
                native_clip=self.accelerator.clip_grad_norm_
                def clip(*args,**kwargs):
                    if b.phase in ('reference','continuous','split') and self.state.global_step+1==8:
                        data.write(output/f'GRADIENTS{self.state.global_step+1}.json',{n:fingerprint(p.grad) for n,p in model.named_parameters() if p.requires_grad})
                    return native_clip(*args,**kwargs)
                self.accelerator.clip_grad_norm_=clip
            if b.resume and not getattr(self,'rng_synced',False):self._load_rng_state(str(output/f'checkpoint-{b.resume}'));self.rng_synced=True
            result=super().training_step(model,inputs,*a,**kw)
            assert audit['native_mode_snapshot']==[m.training for m in model.action_head.modules()]
            row=lines(output/'step_rng.jsonl')[-1];assert np.isfinite(row['loss'])
            gradients=audit['first_update_euler_gradient_norms']
            if gradients and not (output/'S4_first_update_gradients.json').exists():
                assert set(gradients)=={'1','2','3','4'} and all(np.isfinite(v) and v>0 for v in gradients.values())
                data.write(output/'S4_first_update_gradients.json',gradients)
            if b.phase=='main' and self.state.global_step==127 and visit['micro']==1:
                assert not torch.equal(self.health_weight.detach(),self.health_start)
                data.write(output/'HEALTH128.json',dict(status='PASS_CONTINUING_LEARNING',permitted_tensor_changed=self.health_name,reset=False))
            return result
        def _get_output_dir(self,*a,**kw):
            return str(self.checkpoint_staging) if hasattr(self,'checkpoint_staging') else super()._get_output_dir(*a,**kw)
        def _save_checkpoint(self,*a,**kw):
            step=self.state.global_step;start=time.perf_counter()
            assert shutil.disk_usage(root).free>=donor['complete_trainer_bytes']+10*2**30
            snapshot=data.read(root/'STORAGE_ADMISSION.json')
            inodes={(p.stat().st_dev,p.stat().st_ino):p.stat().st_size for p in budget_root.rglob('*') if p.is_file() and not p.is_symlink()}
            owned=sum(inodes.values())
            assert snapshot['available_bytes_at_freeze']-max(0,owned-snapshot['owned_bytes_at_freeze'])>=donor['complete_trainer_bytes']+10*2**30,'USER_QUOTA_CONSERVATIVE_HEADROOM'
            self.checkpoint_staging=output/f'.saving-{step}';self.checkpoint_staging.mkdir(exist_ok=False)
            try:result=super()._save_checkpoint(*a,**kw)
            finally:staging=self.checkpoint_staging;del self.checkpoint_staging
            cp=staging/f'checkpoint-{step}'
            spec=importlib.util.spec_from_file_location('pi0277_metadata_helper',family_root/'PI0270'/'code/train.py')
            helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
            helper.bind_checkpoint_metadata(output,cp,stats,dict(release=RELEASE,mode='S4',arm=b.arm,donor_revision=settings['donor_revision'],legacy_metadata_filename_only=True))
            data.write(cp/'PI0277_RUN.json',dict(arm=b.arm,phase=b.phase,route=b.route,limit=b.limit,scheduler_context=4096,release=RELEASE))
            data.write(cp/'PI0277_READOUT_CONTRACT.json',data.read(root/'design'/f'{b.arm}_READOUT.json'))
            for source in [root/'EXECUTION_FILESET.json',root/'ADOPTION.json',root/'design/TRAIN_PLAN.json',root/'design/INPUT_BINDING.json',root/'DONOR_LOCAL_READBACK.json',root/'DONOR_PARAMETER_SCOPE.json',output/'PARAMETER_SCOPE.json',output/'CURSOR.json']:
                dest=cp/'pi0277_provenance'/source.name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(source,dest)
            for p in cp.rglob('*'):
                if p.is_file():
                    with p.open('rb') as f:os.fsync(f.fileno())
            cp.replace(output/f'checkpoint-{step}');staging.rmdir();cp=output/f'checkpoint-{step}'
            directory=os.open(output,os.O_RDONLY)
            try:os.fsync(directory)
            finally:os.close(directory)
            if b.phase=='main':
                from persistence import snapshot,retain_two
                if step in (512,1024,2048,4096):snapshot(root,b.arm,step,cp)
                retain_two(root,b.arm,output)
            append(output/'SAVE_TIMINGS.jsonl',dict(step=step,seconds=time.perf_counter()-start,kind='COMPLETE_TRAINER'))
            return result
        def save_model(self,output_dir=None,*a,**kw):
            if output_dir is None and (output/f'checkpoint-{self.state.global_step}/PI0277_RUN.json').exists():return
            return super().save_model(output_dir,*a,**kw)
    experiment.Gr00tTrainer=Trainer
    binding=dict(root=str(root),arm=b.arm,phase=b.phase,route=b.route,limit=b.limit,resume=b.resume,release=RELEASE)
    data.write(output/f'RUN_{b.resume}_{b.limit}.json',binding)
    native.run_official(config,paired.GROOT)
    actual=trainer_ref['value'].state.global_step
    assert audit['forwards']==2*actual and (actual==b.limit or (root/'runtime/STOP_TRAINING.json').exists())
    data.write(output/'SEGMENT_COMPLETE.json',dict(status='PASS_EXECUTED_SEGMENT',**binding,actual_microbatches=audit['forwards'],actual_update=actual))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',required=True);p.add_argument('--phase',required=True)
    p.add_argument('--route',required=True);p.add_argument('--limit',type=int,required=True);p.add_argument('--resume',type=int,default=0)
    run(Binding(**vars(p.parse_args())))
