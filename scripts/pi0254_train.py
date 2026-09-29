"""PI0254 adapter over the qualified G objective, Trainer and durable lifecycle."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys

import pi0253_full_train as parent
import pi0251_ce_train as ce
import pi0254_step_rng as rng

ROOT=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0254')
AUTH='PI0254_DIAGNOSE_THEN_CONTROLLED_FINETUNE_GO_V1'
RELEASE='7e5a4aae69912ba8e593d50d313f288abc16a914'
STEPS=4608
SAVES=(1152,2304,3456,4608)
ARMS=('U576','R576')
ce.STEPS,ce.CHECKPOINTS=STEPS,SAVES
ce.REVISION,ce.TREE_SHA256=parent.REVISION,parent.TREE


def verify_card(path,arm):
    doc=ce._json(path)
    parent.require(arm in ARMS and doc['arm']==arm and doc['schema']=='PI0254_RUN_CARD_V1', 'PI0254 arm/schema')
    parent.require(ce._json(path.with_name('run_card.seal.json'))['sha256']==ce._sha(path),'PI0254 seal')
    parent.require(doc['authorization']==doc['execution_authorization']==AUTH and doc['authority_commit']==RELEASE,
                   'PI0254 authority')
    parent.require(doc['selected_checkpoint']==doc['deployment_selected_checkpoint']=='NONE' and doc['seed']==25402,
                   'PI0254 deployment/seed')
    parent.require(doc['stage']['steps']==STEPS and tuple(doc['stage']['checkpoint_steps'])==SAVES
                   and doc['cohort']['count']==576 and doc['schedule']['count']==STEPS
                   and doc['loss_coefficients']==parent.COEFF and doc['d1_runtime_id']=='S4','PI0254 recipe')
    for key in ('cohort','schedule'):
        parent.require(ce._sha(Path(doc[key]['path']))==doc[key]['sha256'],f'PI0254 {key}')
    for label,entry in doc['source']['bound_files'].items():
        parent.require(ce._sha(Path(entry['path']))==entry['sha256'],f'PI0254 input {label}')
    for name,digest in doc['source']['custom_execution_code_sha256'].items():
        parent.require(ce._sha(Path(doc['paths']['execution_driver_root'])/name)==digest,f'PI0254 code {name}')
    return doc


def paired_records(doc):
    if doc['arm']=='U576':return [None]*STEPS
    u=ce._json(ROOT/'U576/run_card.json')
    parent.require(u['initializer']==doc['initializer'] and u['resolved_config']==doc['resolved_config'],
                   'PI0254 common initializer/config')
    output=Path(u['paths']['output_root']);partial=output/'terminal_partial.json'
    if partial.exists():
        terminal=ce._json(partial)
        assert terminal['status']=='INVALID_OR_BUDGET_BLOCKED_PARTIAL' and not terminal['shared_contract_invalid']
        record_path=Path(terminal['actual_pair_identity_path'])
        assert record_path.resolve().is_relative_to(output.resolve()) and ce._sha(record_path)==terminal['actual_pair_identity_sha256']
        records=ce._jsonl(record_path)
    else:
        root=ce.helper._winning_training_root(u)
        resume=0 if root==output/'training' else int(ce._json(root.parent/'resume_record.json')['resume_after_step'])
        records=ce._rows_for_root(output,root,'pi0254_pair_identity.jsonl',resume)
    reference=ce._json(Path(u['schedule']['path']))['rows']
    current=ce._json(Path(doc['schedule']['path']))['rows']
    parent.require((len(records)==STEPS or partial.exists()) and all(r['status']=='REFERENCE_CAPTURED' and r['before_backward'] and r['optimizer_step']==i for i,r in enumerate(records,1)),
                   'PI0254 U actual draws incomplete')
    return rng.peer_for_visit(current,reference,records)


def install_head_counter(head_cls,audit,step_rng,arm):
    original_forward=head_cls.forward
    def forward(self,*args,**kwargs):
        active=step_rng.active_step_context()
        assert active is not None
        self._pi0254_count_step=active['optimizer_step']
        if not getattr(self,'_pi0254_head_count_hook',False):
            def count(module,inputs):
                active=step_rng.active_step_context()
                step=self._pi0254_count_step
                phase=audit.get('phase') or 'native_training'
                if active is None:assert phase=='backward_recompute'
                else:assert active['optimizer_step']==step
                ce._append(Path(ce.runtime_state['training_root'])/'pi0254_training_head_forwards.jsonl',
                           dict(optimizer_step=step,phase=phase,arm=arm),sync=step==1)
            self.model.register_forward_pre_hook(count);self._pi0254_head_count_hook=True
        return original_forward(self,*args,**kwargs)
    head_cls.forward=forward


def counter_cpu_check(qualified):
    import tempfile
    import torch
    from types import SimpleNamespace
    active={'optimizer_step':1};audit={};gradients={}
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__();self.model=torch.nn.Linear(1,1).double()
            with torch.no_grad():self.model.weight.fill_(.37);self.model.bias.fill_(.07)
        def forward(self,value):
            native=self.model(value)
            def contexts():
                return qualified._module_eval(self,audit,'rollout_forward'),qualified._module_eval(self,audit,'backward_recompute')
            def track(index,state):
                state.register_hook(lambda gradient:gradients.update({index:float(gradient.abs().sum())}))
            result,_,_=qualified._euler_rollout(value,lambda current,index:torch.tanh(self.model(current)),(),
                         steps=4,checkpointed=True,context_fn=contexts,state_hook=track)
            return native.square().sum()+result.square().sum()
    install_head_counter(Head,audit,SimpleNamespace(active_step_context=lambda:active),'CPU_FIXTURE')
    previous=ce.runtime_state.get('training_root')
    try:
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='cpu-counter-') as temporary:
            ce.runtime_state['training_root']=temporary;head=Head().train()
            loss=head(torch.tensor([[.61]],dtype=torch.float64,requires_grad=True));active=None
            loss.backward()
            rows=ce._jsonl(Path(temporary)/'pi0254_training_head_forwards.jsonl')
            assert [r['phase'] for r in rows]==['native_training']+['rollout_forward']*4+['backward_recompute']*4
            assert all(r['optimizer_step']==1 for r in rows)
            assert len(gradients)==4 and all(v>0 for v in gradients.values())
            assert all(m.training for m in head.modules()) and audit.get('phase') is None
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
            try:head.model(torch.ones(1,1,dtype=torch.float64))
            except AssertionError:pass
            else:raise AssertionError('missing context outside recomputation accepted')
    finally:
        if previous is None:ce.runtime_state.pop('training_root',None)
        else:ce.runtime_state['training_root']=previous
    return dict(status='PASS_ACTUAL_COUNTER_QUALIFIED_EULER_CPU',native_forwards=1,rollout_forwards=4,
                backward_recomputations=4,all_four_state_gradients=True,mode_restored=True,missing_context_rejected_outside_recompute=True)


original_overlay=ce._install_overlay_runtime
def overlay(module,doc,step_rng):
    original_overlay(module,doc,step_rng)
    peers=paired_records(doc)
    schedule=ce._json(Path(doc['schedule']['path']))['rows']
    original_guard=ce.runtime_state['before_forward_guard']
    def guard(active):
        result=original_guard(active);peer=peers[active['optimizer_step']-1]
        if peer is not None:step_rng.verify_identity(active,peer,('key','seed','model_input_sha256'))
        return result
    ce.runtime_state['before_forward_guard']=guard
    original_install=module.install
    def install(*args,**kwargs):
        audit=original_install(*args,**kwargs)
        original_append=module._append_json
        def append(path,row,count):
            original_append(path,row,count)
            root=Path(ce.runtime_state['training_root'])
            actual=ce._last_jsonl(root/'pi0254_pair_identity.jsonl')
            assert actual['optimizer_step']==count
            rng.verify_visit(actual,peers[count-1])
            plan=schedule[count-1]
            ce._append(root/'pi0254_source_visit_pair.jsonl',dict(optimizer_step=count,
                       source_episode_id=plan['source_episode_id'],visit=plan['visit'],
                       status='MATCHED_ACTUAL_DRAWS' if peers[count-1] is not None else 'REFERENCE_CAPTURED' if doc['arm']=='U576' else 'UNPAIRED_VISIT_CAPTURED',
                       before_backward=True,actual_identity=actual,peer_identity=peers[count-1]),sync=True)
        module._append_json=append
        install_head_counter(args[0] if args else kwargs['head_cls'],audit,step_rng,doc['arm'])
        return audit
    module.install=install


ce._install_overlay_runtime=overlay
ce._verify_card=verify_card
ce._validate_initializer=parent.validate_initializer
original_route=parent.original_route
def route(command,doc,path):
    result=original_route(command,doc,path)
    if result is not command:
        result=list(result);result[1]=str(Path(doc['paths']['execution_driver_root'])/'pi0254_train.py')
    return result
ce._route_archive_sidecar=route
original_preverify=ce._preverify_training
def preverify(doc,path):
    result=original_preverify(doc,path);root=result['root'];resume=result['resume_after_step']
    updates=ce._rows_for_root(Path(doc['paths']['output_root']),root,'pi0254_actual_updates.jsonl',resume)
    visits=ce._rows_for_root(Path(doc['paths']['output_root']),root,'pi0254_source_visit_pair.jsonl',resume)
    assert len(updates)==len(visits)==STEPS
    assert all(r['optimizer_completed'] and r['optimizer_step']==i for i,r in enumerate(updates,1))
    assert {r['optimizer_step'] for r in updates if r.get('parameter_update_norms')}=={1,64,256,1152,4608}
    peers=paired_records(doc)
    for i,row in enumerate(visits):
        assert row['optimizer_step']==i+1 and row['before_backward']
        rng.verify_visit(row['actual_identity'],peers[i])
    assert sum(r['status']=='MATCHED_ACTUAL_DRAWS' for r in visits)==sum(p is not None for p in peers)
    assert len(ce._jsonl(ROOT/'raw_optimizer_intents.jsonl'))<=9216
    result.update(actual_update_rows=updates,source_visit_rows=visits)
    return result
ce._preverify_training=preverify
original_integrity=ce._write_integrity
def integrity(doc,path,verified):
    original_integrity(doc,path,verified)
    target=Path(doc['paths']['output_root'])/'pi0254_training_integrity.json';receipt=ce._json(target)
    receipt.update(source_visit_pairing=True,actual_update_rows=len(verified['actual_update_rows']),
                   actual_matched_source_visits=sum(r['status']=='MATCHED_ACTUAL_DRAWS' for r in verified['source_visit_rows']),
                   paired_identity_status='U_SOURCE_VISIT_REFERENCE_CAPTURED' if doc['arm']=='U576' else 'PASS_AVAILABLE_SHARED_VISITS_ACTUAL_DRAWS',
                   raw_update_intents=len(ce._jsonl(ROOT/'raw_optimizer_intents.jsonl')))
    ce._replace(target,receipt)
    state_path=target.parent/'execution_state.json';state=ce._json(state_path)
    state['training_integrity_sha256']=ce._sha(target);ce._replace(state_path,state)
ce._write_integrity=integrity


def configure(core):
    core.STEPS,core.CHECKPOINTS,core.SEED=STEPS,SAVES,25402
    original=core.build_runtime_config
    def build(*args,**kwargs):
        pi0228,config,summary=original(*args,**kwargs)
        config.training.save_steps=1152;config.training.seed=25402
        config.data.seed=25402
        summary.update(save_steps=1152,save_step_set=list(SAVES),seed=25402)
        assert config.training.max_steps==4608 and config.training.warmup_steps==10
        return pi0228,config,summary
    core.build_runtime_config=build
    original_readback=core.verify_pt_hf_readback
    def readback(receipt,record,**kwargs):
        import requests
        from huggingface_hub import hf_hub_url
        result=original_readback(receipt,record,**kwargs)
        for entry in record['inventory']['entries']:
            remote=f"{kwargs['prefix']}/checkpoint-{record['checkpoint_step']}/{entry['path']}"
            digest=hashlib.sha256();size=0
            with requests.get(hf_hub_url(kwargs['repo_id'],filename=remote,revision=receipt['hf_manifest_commit_id']),
                              headers={'Authorization':'Bearer '+kwargs['token']},stream=True,timeout=(30,180)) as response:
                response.raise_for_status()
                for block in response.iter_content(8<<20):digest.update(block);size+=len(block)
            assert size==entry['bytes'] and digest.hexdigest()==entry['sha256'],remote
        result.update(full_content_stream_readback=True,full_content_files=len(record['inventory']['entries']))
        return result
    core.verify_pt_hf_readback=readback
    original_retire=core.retire_pt_checkpoint_after_verified_archive
    # Keep the current state locally. At the endpoint a linked complete state survives lifecycle retirement.
    def retain(archive_root,training_root,step,record,receipt_path,readback_path,**kwargs):
        read=ce._json(readback_path)
        assert read.get('full_content_stream_readback') and read['source_tree_sha256']==record['inventory']['tree_sha256']
        if step==1152:
            view=Path(archive_root)/'model-only-1152';view.mkdir(exist_ok=True)
            source=Path(training_root)/'checkpoint-1152'
            names=['config.json','model.safetensors.index.json','model-00001-of-00002.safetensors','model-00002-of-00002.safetensors']
            for name in names:
                if not (view/name).exists():os.link(source/name,view/name)
                assert ce._sha(source/name)==ce._sha(view/name)
            ce._write(view/'snapshot.json',dict(status='MODEL_ONLY_NOT_RESUMABLE',step=1152,
                      files=[dict(path=n,bytes=(view/n).stat().st_size,sha256=ce._sha(view/n)) for n in names],
                      revision=read['revision'],checkpoint_tree_sha256=read['source_tree_sha256']))
        if step==4608:
            retained=Path(archive_root)/'final_recovery/checkpoint-4608';retained.mkdir(parents=True,exist_ok=True)
            for entry in record['inventory']['entries']:
                source=Path(training_root)/'checkpoint-4608'/entry['path'];destination=retained/entry['path']
                assert not source.is_symlink() and ROOT.resolve() in source.resolve().parents
                destination.parent.mkdir(parents=True,exist_ok=True)
                if not destination.exists():os.link(source,destination)
                assert ce._sha(destination)==entry['sha256'] and destination.stat().st_size==entry['bytes']
            ce._write(retained.parent/'receipt.json',dict(status='LOCAL_COMPLETE_TRAINER_AND_HF',record=record,
                                                       hf_revision=read['revision'],full_content_readback_sha256=ce._sha(readback_path)))
        for prior in ((2304,) if step==3456 else (1152,3456,4608) if step==4608 else ()):
            old_receipt=Path(archive_root)/f'durable_receipts/checkpoint-{prior}.durable_receipt.json'
            old_readback=Path(archive_root)/f'hf_readbacks/checkpoint-{prior}.json'
            old_record=ce._json(old_receipt)['source_record']
            assert ce._json(old_readback).get('full_content_stream_readback')
            if not (Path(archive_root)/f'checkpoint_retirements/checkpoint-{prior}.deleted.json').exists():
                original_retire(archive_root,training_root,prior,old_record,old_receipt,old_readback,**kwargs)
    core.retire_pt_checkpoint_after_verified_archive=retain


def install(path,arm,resume):
    os.environ['PI0254_RUN_CARD']=str(path)
    doc,life=ce._install_adapters(path,arm,resume)
    life.SEED=ce.helper.SEED=25402
    life.AUTHORIZATION=AUTH;life.R1_CANONICAL_COMMENT=5882151489;life.R1_POINTER_COMMENT=0
    ce.helper.STEPS,ce.helper.CHECKPOINTS=STEPS,SAVES
    life._validate_storage=parent.capacity
    original_load=life.load_module
    def load(source,name):
        module=original_load(source,name)
        if source.name=='pi0241_fixed16_learning.py':module.SEED=25402
        if source.name=='pi0246_gate_pass_train.py':configure(module)
        if source.name=='pi0247_u_rollout_overlay.py' and not resume:
            original=module.install
            def inherited_install(*args,**kwargs):
                audit=original(*args,**kwargs);parity_source=Path(doc['source']['bound_files']['G_parity']['path'])
                parity=ce._json(parity_source)
                assert parity['status']=='PASS_PI0247_U_S4_PARITY' and parity['gate_decision_equal'] and parity['gate_issues_equal']
                audit.update(parity=parity,parity_inherited_sha256=ce._sha(parity_source),parity_nonselection_rollouts=0)
                ce._write(kwargs['parity_path'],parity);return audit
            module.install=inherited_install
        return module
    life.load_module=load
    original_chain=life._install_u_chain_check
    def chain(*args,**kwargs):
        original_chain(*args,**kwargs)
        import torch
        import pi0241_fixed16_learning as fixed_groups
        from transformers import TrainerCallback
        from gr00t.experiment import experiment as experiment_module
        class ActualUpdateAudit(TrainerCallback):
            def on_train_begin(self,args,state,control,**kw):
                self.frozen={name:(value,value._version) for name,value in kw['model'].named_parameters() if not value.requires_grad}
                self.before={}
                return control
            def on_pre_optimizer_step(self,args,state,control,**kw):
                step=int(state.global_step)+1
                ledger=ROOT/'raw_optimizer_intents.jsonl'
                previous=ce._jsonl(ledger) if ledger.exists() else []
                if len(previous)>=9216:raise RuntimeError('PI0254 total raw update cap exhausted')
                for name,(value,version) in self.frozen.items():
                    if value._version!=version:raise RuntimeError('PI0254 frozen parameter drift: '+name)
                trainable=[(name,value) for name,value in kw['model'].named_parameters() if value.requires_grad]
                assert all(fixed_groups._parameter_group(name) in ('projector','diffusion') for name,_ in trainable)
                norm2=0.
                for name,value in trainable:
                    assert torch.isfinite(value).all() and (value.grad is None or torch.isfinite(value.grad).all()),name
                    if value.grad is not None:norm2+=float(value.grad.detach().float().square().sum())
                if step in (1,64,256,1152,4608):self.before={name:value.detach().clone() for name,value in trainable}
                plan=ce._json(Path(doc['schedule']['path']))['rows'][step-1]
                event=dict(raw_ordinal=len(previous)+1,arm=arm,optimizer_step=step,source_episode_id=plan['source_episode_id'],
                           visit=plan['visit'],learning_rates=[group['lr'] for group in kw['optimizer'].param_groups],
                           post_clip_grad_norm=math.sqrt(norm2),pre_clip_grad_norm='USE_EXISTING_TRAINER_LOG_IF_AVAILABLE',
                           accounting='DURABLE_BEFORE_OPTIMIZER_CONSERVATIVE_IF_INTERRUPTED',nonfinite=False)
                ce._append(ledger,event,sync=True);self.event=event
                return control
            def on_optimizer_step(self,args,state,control,**kw):
                norms={};dtypes={}
                for name,value in kw['model'].named_parameters():
                    if value.requires_grad:
                        assert torch.isfinite(value).all(),name
                        if name in self.before:
                            group=fixed_groups._parameter_group(name)
                            norms[group]=norms.get(group,0.)+float((value.detach().float()-self.before[name].float()).square().sum())
                            dtypes.setdefault(group,set()).add(str(value.dtype))
                event={**self.event,'optimizer_completed':True}
                if self.before:event.update(parameter_update_norms={k:math.sqrt(v) for k,v in norms.items()},
                                              parameter_dtypes={k:sorted(v) for k,v in dtypes.items()})
                self.before={};ce._append(Path(ce.runtime_state['training_root'])/'pi0254_actual_updates.jsonl',event,sync=True)
                return control
        base=experiment_module.Gr00tTrainer
        class AuditTrainer(base):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs);self.add_callback(ActualUpdateAudit())
        experiment_module.Gr00tTrainer=AuditTrainer
    life._install_u_chain_check=chain
    return doc,life


def prepare(commit):
    source=ce._json(ROOT.parent/'PI0253/FULL_G/run_card.json')
    exe=Path(__file__).resolve().parent
    cohort=ROOT/'design/support576.json'
    raw=ce._json(ROOT/'source640/manifest.json');times={r['source_episode_id']:r['timestamp_record']['future_action_point_timestamps'] for r in raw['files']}
    support=ce._json(cohort)['rows']
    dt=ROOT/'actual_dt.json'
    ce._write(dt,dict(rows=[dict(fixed_index=r['fixed_index'],source_episode_id=r['source_episode_id'],
                    dt_s=[b-a for a,b in zip(times[r['source_episode_id']],times[r['source_episode_id']][1:])]) for r in support]))
    names=('pi0254_train.py','pi0254_step_rng.py','pi0254_train_h200.pbs','pi0251_ce_train.py','pi0251_step_rng.py','pi0252_speed_margin.py','pi0253_full_train.py')
    for arm in ARMS:
        out=ROOT/arm;out.mkdir(exist_ok=True);(out/'logs').mkdir(exist_ok=True)
        schedule=ROOT/f'design/{arm}_schedule.json';rows=ce._json(schedule)['rows']
        keys=ROOT/f'{arm}_step_keys.json';ce._write(keys,dict(schema=rng.SCHEMA,rows=rng.build_key_table(rows)))
        doc=copy.deepcopy(source)
        doc.update(schema='PI0254_RUN_CARD_V1',protocol='PI0254',authorization=AUTH,execution_authorization=AUTH,
                   authority_commit=RELEASE,canonical_issue_comment=5882151489,pointer_pr_comment=0,
                   family=f'PI0254_{arm}_SUPPORT576_V1',arm=arm,seed=25402,selected_checkpoint='NONE',deployment_selected_checkpoint='NONE',
                   integrity_schema='PI0254_TRAINING_INTEGRITY_V1',training_manifest_family='CONTROLLED_SUPPORT',
                   stage=dict(name='SUPPORT576',steps=STEPS,checkpoint_steps=list(SAVES),evaluation_steps=[1152,4608]))
        doc['paths'].update(output_root=str(out),output_base=str(ROOT),execution_driver_root=str(exe),execution_driver=str(exe/'pi0254_train.py'),pbs_script=str(exe/'pi0254_train_h200.pbs'))
        doc['cohort']=dict(path=str(cohort),sha256=ce._sha(cohort),count=576)
        doc['schedule']=dict(path=str(schedule),sha256=ce._sha(schedule),count=STEPS)
        doc['archive'].update(path_prefix=f'arms/PI0254/controlled-support-20260929/{arm}',checkpoint_steps=list(SAVES))
        src=doc['source'];src.update(audit_prefix='pi0254',paired_arms=[arm],reference_arm=arm,intervention_arm=None,
             step_rng_module='pi0254_step_rng.py',step_rng_module_name='pi0254_sample_visit_rng_runtime',
             dt_table_path=str(dt),dt_table_sha256=ce._sha(dt),step_key_table_path=str(keys),step_key_table_sha256=ce._sha(keys),
             execution_code_commit=commit,authorization=AUTH)
        src['bound_files'].update(dt_table=dict(path=str(dt),sha256=ce._sha(dt)),step_keys=dict(path=str(keys),sha256=ce._sha(keys)),
                                 raw_inputs=dict(path=str(ROOT/'source640/manifest.json'),sha256=ce._sha(ROOT/'source640/manifest.json')))
        src['custom_execution_code_sha256']={n:ce._sha(exe/n) for n in names}
        doc['optimizer'].update(steps=STEPS,checkpoint_steps=list(SAVES),optimizer_state='fresh identical AdamW/cosine4608')
        doc['resolved_config'].update(max_steps=STEPS,save_steps=1152,save_step_set=list(SAVES),seed=25402)
        doc['execution_contract']=dict(source_visit_pairing=True,shared_visits=3584,raw_update_cap=9216,
             submissions_per_arm_cap=2,new_parity_queries=0,serial_arms=list(ARMS),evaluation_steps=[1152,4608])
        path=out/'run_card.json';ce._write(path,doc);ce._write(out/'run_card.seal.json',dict(sha256=ce._sha(path)))
        print(json.dumps(dict(status='SEALED_NOT_SUBMITTED',arm=arm,sha256=ce._sha(path))),flush=True)


def cpu_check(path,arm):
    import torch
    doc,life=install(path,arm,0)
    parent.validate_initializer(doc)
    life._verify_code(doc,Path(doc['paths']['repo_code_root']),Path(doc['paths']['code_root']),Path(doc['paths']['training_runtime_root']))
    life._verify_runtime_bindings(doc['paths'],doc['source']['runtime_bindings'])
    paths=doc['paths']
    sys.path[:0]=[str(Path(paths['stage10_root'])/'runtime'),paths['training_runtime_root'],
                  str(Path(paths['code_root'])/'scripts'),str(Path(paths['code_root'])/'src'),paths['groot_source']]
    fixed=life.load_module(Path(paths['code_root'])/'scripts/pi0241_fixed16_learning.py','pi0254_cpu_fixed')
    fixed.MAX_STEPS,fixed.CHECKPOINT_STEPS=STEPS,SAVES
    fixed.bind_runtime(life._card_with_p_runtime_paths(doc))
    core=life.load_module(Path(paths['training_runtime_root'])/'pi0246_gate_pass_train.py','pi0254_cpu_core')
    qualified=life.load_module(Path(paths['repo_code_root'])/'scripts/pi0247_u_rollout_overlay.py','pi0254_cpu_counter_overlay')
    counter=counter_cpu_check(qualified)
    _,config,summary=core.build_runtime_config(paths,Path(paths['output_root'])/'training',fixed)
    assert summary==doc['resolved_config'],(summary,doc['resolved_config'])
    result=rng.cpu_check(ROOT/'design')
    rng.configure_schedule(ce._json(Path(doc['schedule']['path']))['rows'])
    inherited_rng=rng.inherited.cpu_self_check()
    assert inherited_rng['checkpoint_rng_preserved'] and inherited_rng['trainer_invoked_pre_forward_guard']
    from transformers import get_scheduler
    parameter=torch.nn.Parameter(torch.ones(1));optimizer=torch.optim.AdamW([parameter],lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=1e-5)
    scheduler=get_scheduler('cosine',optimizer,num_warmup_steps=10,num_training_steps=STEPS)
    lr={str(k):1e-5*scheduler.lr_lambdas[0](k) for k in (0,1,10,1152,2304,3456,4608)}
    assert lr['0']==lr['4608']==0. and lr['10']==1e-5
    for cursor in (1152,2304,3456):
        suffix=fixed.fixed_schedule_suffix(ce._json(Path(doc['schedule']['path']))['rows'],cursor)
        assert len(suffix)==STEPS-cursor and suffix[0]['optimizer_step']==cursor+1
    import tempfile
    from types import SimpleNamespace
    command=[sys.executable,str(Path(doc['paths']['repo_code_root'])/'scripts/pi0247_lu_train.py'),
             '--archive-sidecar','--run-card',str(path),'--arm',arm,'--step','4608','--archive-root',doc['paths']['output_root']]
    assert Path(route(command,doc,path)[1]).name=='pi0254_train.py'
    unrelated=[sys.executable,'unrelated.py'];assert route(unrelated,doc,path) is unrelated
    with tempfile.TemporaryDirectory(dir=ROOT,prefix='cpu-lifecycle-') as temporary:
        base=Path(temporary);training=base/'training';training.mkdir();retired=[]
        def retire(archive,train,step,*args,**kwargs):
            assert Path(train).resolve().is_relative_to(ROOT.resolve())
            shutil.rmtree(Path(train)/f'checkpoint-{step}');retired.append(step)
            ce._write(Path(archive)/f'checkpoint_retirements/checkpoint-{step}.deleted.json',dict(fixture=True))
        fixture=SimpleNamespace(build_runtime_config=lambda:None,verify_pt_hf_readback=lambda *a,**k:None,
                                retire_pt_checkpoint_after_verified_archive=retire)
        configure(fixture)
        for step in SAVES:
            checkpoint=training/f'checkpoint-{step}';checkpoint.mkdir()
            names=['config.json','model.safetensors.index.json','model-00001-of-00002.safetensors','model-00002-of-00002.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','training_args.bin']
            for name in names:(checkpoint/name).write_bytes(f'CPU fixture {step} {name}'.encode())
            record=dict(checkpoint_step=step,inventory=dict(tree_sha256=f'fixture-{step}',entries=[dict(path=n,bytes=(checkpoint/n).stat().st_size,sha256=ce._sha(checkpoint/n)) for n in names]))
            receipt=base/f'durable_receipts/checkpoint-{step}.durable_receipt.json';readback=base/f'hf_readbacks/checkpoint-{step}.json'
            ce._write(receipt,dict(source_record=record));ce._write(readback,dict(full_content_stream_readback=True,source_tree_sha256=f'fixture-{step}',revision=f'fixture-revision-{step}'))
            fixture.retire_pt_checkpoint_after_verified_archive(base,training,step,record,receipt,readback)
            if step<4608:assert checkpoint.exists() and (training/'checkpoint-1152').exists()
        assert retired==[2304,1152,3456,4608]
        assert len(list((base/'final_recovery/checkpoint-4608').iterdir()))==9
        assert len(list((base/'model-only-1152').glob('*.safetensors')))==2
        for cursor in (1152,2304,3456):
            scheduler.last_epoch=cursor;optimizer.param_groups[0]['lr']=lr[str(cursor)]
            checkpoint=base/f'resume-{cursor}.pt';torch.save(dict(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=torch.get_rng_state()),checkpoint)
            restored=torch.load(checkpoint,weights_only=False);optimizer.load_state_dict(restored['optimizer']);scheduler.load_state_dict(restored['scheduler'])
            assert scheduler.last_epoch==cursor and optimizer.param_groups[0]['lr']==lr[str(cursor)] and torch.equal(restored['rng'],torch.get_rng_state())
    output=Path(paths['output_root']);probe=output/'cpu_bounded_write.bin'
    assert not probe.exists()
    with probe.open('xb') as stream:stream.write(bytes(1<<20));stream.flush();os.fsync(stream.fileno())
    assert probe.stat().st_size==1<<20;probe.unlink()
    result.update(status='PASS_PI0254_ACTUAL_CPU_BINDINGS_CONFIG_RNG_AND_LIFECYCLE',arm=arm,
                  run_card_sha256=ce._sha(path),resolved_config=summary,learning_rate_at_fixed_cursors=lr,
                  bounded_write_bytes=1<<20,quota_headroom='UNKNOWN',checkpoint_cpu_rng_recompute=inherited_rng,
                  lifecycle_fixture='ACTUAL_ADAPTER_RETENTION_AND_ROUTE_SYNTHETIC_PAYLOAD_NO_HF',exact_optimizer_scheduler_rng_restore=True,
                  head_counter_cpu_recompute=counter,
                  new_model_queries=0,new_optimizer_updates=0)
    ce._write(output/'prequery_cpu_receipt.json',result);print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-card',type=Path);p.add_argument('--arm',choices=ARMS)
    p.add_argument('--prepare',action='store_true');p.add_argument('--execution-commit')
    p.add_argument('--cpu-check',action='store_true')
    for mode in ('train','finalize','hf-preflight','archive-sidecar'):p.add_argument('--'+mode,action='store_true')
    p.add_argument('--step',type=int);p.add_argument('--archive-root',type=Path);p.add_argument('--resume-after-step',type=int,default=0)
    args=p.parse_args()
    if args.prepare:return prepare(args.execution_commit)
    if args.cpu_check:return cpu_check(args.run_card,args.arm)
    if args.train:
        baseline=ce._json(ROOT/'P1_comparison.json')
        assert baseline['status']=='P1_VALID_BASELINES_COMPLETE' and baseline['new_queries_valid']==320
        registry=ROOT/'submissions.jsonl';prior=ce._jsonl(registry) if registry.exists() else []
        attempts=[r for r in prior if r['arm']==args.arm]
        assert len(attempts)<2,'PI0254 submission limit'
        assert not any(r['job_id']==os.environ.get('PBS_JOBID') for r in prior),'PI0254 duplicate invocation'
        consumed=ce._jsonl(ROOT/'raw_optimizer_intents.jsonl') if (ROOT/'raw_optimizer_intents.jsonl').exists() else []
        assert args.resume_after_step or not any(r['arm']==args.arm for r in consumed),'PI0254 must resume consumed state exactly'
        assert len(consumed)+STEPS-args.resume_after_step<=9216,'PI0254 endpoint exceeds remaining raw cap'
        ce._append(registry,dict(arm=args.arm,job_id=os.environ.get('PBS_JOBID'),resume_after_step=args.resume_after_step,
                   run_card_sha256=ce._sha(args.run_card),p1_sha256=ce._sha(ROOT/'P1_comparison.json')),sync=True)
    _,life=install(args.run_card,args.arm,args.resume_after_step)
    return ce._dispatch_lifecycle(args,life)


if __name__=='__main__':main()
