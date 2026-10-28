"""Source-bound carry semantics, separate phase cursor and fixed native RNG."""
import json,hashlib
from pathlib import Path


def validate(root,path):
    from learning import file_digest
    root=Path(root);v=json.loads(Path(path).read_text())
    assert v['authority_release']=='508e3db7b63d3c47758794f3ffbad87b577e6e9d' and v['launch_allowed'] is True
    assert v['stage'] in ('MORE','EVENT')
    assert file_digest(v['plan_path'])==v['plan_sha256']
    export=json.loads(Path(v['parent_export']).read_text())
    assert export['full_state_sha256']==v['parent_full_master_sha256']
    parent=Path(v['parent']);assert json.loads((parent/'PI0282_RECOVERY.json').read_text())['cursor']==v['parent_phase_step']*8
    for name,item in v['parent_files'].items():
        assert (parent/name).stat().st_size==item['bytes'],name
        if not name.endswith(('.safetensors','.pt')):assert file_digest(parent/name)==item['sha256'],name
    if v['stage']=='EVENT':
        decision=json.loads((root/'design/EVENT_DECISION.json').read_text())
        assert decision['launch_allowed'] and decision['MORE_complete'] and not decision['qualified_candidate']
        assert decision['TRAIN_fit_count']<18 and (decision['release_bad_count']>=3 or decision['approach_bad_count']>=3)
    return v


def restore_model(model,v):
    from continuation import restore_model as exact
    exact(model,v)


def restore_optimizer(optimizer,v,output):
    import torch
    from learning import tensor_tree
    saved=torch.load(Path(v['parent'])/'optimizer.pt',map_location='cpu',weights_only=False)
    expected=tensor_tree(saved['state']);optimizer.load_state_dict(saved)
    assert tensor_tree(optimizer.state_dict()['state'])==expected
    assert all(int(s['step'])==v['parent_global_update'] for s in optimizer.state.values())
    for i,g in enumerate(optimizer.param_groups):g['lr']=g['initial_lr']=v['peak_LRs'][i]
    Path(output).write_text(json.dumps(dict(status='PASS_EXACT_ADAM_MOMENTS_GLOBAL_STEPS_CARRY',parent=v['parent'],parent_global_update=v['parent_global_update'],phase_updates=v['updates'],warmup=v['warmup'],peak_LRs=v['peak_LRs'],only_scheduler_reset=True),indent=2)+'\n')


def install_audit(trainer,model):
    """Hooks observe the existing forward; no extra forward or gradient query."""
    import torch
    from learning import Journal
    journal=Journal(Path(trainer.args.output_dir)/'native_loss_modes_LR.jsonl')
    modules=dict(model.named_modules());observed={};handles=[]
    prefixes=('backbone','backbone.model','action_head.model','action_head.state_encoder','action_head.action_encoder','action_head.action_decoder','action_head.vlln','action_head.vl_self_attention','action_head.position')
    selected={n:m for n,m in modules.items() if n in prefixes or any(n.startswith(p) and '.' not in n[len(p)+1:] for p in prefixes[5:])}
    for name,module in selected.items():
        def pre(m,inputs,n=name):observed[n]=dict(training=m.training,trainable_parameters=any(p.requires_grad for p in m.parameters()))
        handles.append(module.register_forward_pre_hook(pre))
    def record(module,inputs,result):
        if not torch.is_grad_enabled():return
        loss=result.get('loss') if hasattr(result,'get') else None
        if loss is None:return
        native=float(loss.detach());mask=result.get('action_mask');element=result.get('action_loss')
        reduction=None
        if mask is not None and element is not None:
            expected=element.sum()/(mask.sum()+1e-6)
            assert torch.equal(loss.detach(),expected.detach()),'NATIVE_MASKED_MEAN_REDUCTION_CHANGED'
            reduction=dict(active_mask_sum=float(mask.sum()),element_sum=float(element.detach().sum()))
        trainer.actual_native_loss=native
        journal.append(dict(phase_step=trainer.state.global_step+1,cursor=trainer.cursor,native_loss=native,expected_returned_loss=native/trainer.accumulation,actual_LRs=[g['lr'] for g in trainer.optimizer.param_groups],modules=dict(observed),position_parameters={n:dict(requires_grad=p.requires_grad,dtype=str(p.dtype)) for n,p in model.named_parameters() if 'position' in n or 'pos_embed' in n},native_reduction=reduction));observed.clear()
    handles.append(model.action_head.register_forward_hook(record))
    trainer.mode_journal=journal
    return handles
