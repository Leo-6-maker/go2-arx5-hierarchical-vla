"""Exact packed-row scope; stock architecture and input autograd retained."""
import hashlib
import torch
ROW=10
PREFIXES=tuple('action_head.'+x+'.' for x in ('state_encoder','action_encoder','action_decoder'))


def digest(tensor):
    t=tensor.detach().reshape(-1).contiguous().cpu().view(torch.uint8).numpy()
    return hashlib.sha256(t.tobytes()).hexdigest()


def is_projector(name):return name.startswith(PREFIXES)


def mask(gradient):
    gradient[:ROW].zero_();gradient[ROW+1:].zero_()
    return gradient


def configure(model,arm):
    assert arm in ('ACTION2048','INTERFACE2048')
    model.requires_grad_(False)
    model.backbone.tune_llm=False;model.backbone.tune_visual=False
    model.config.tune_llm=False;model.config.tune_visual=False
    head=model.action_head
    head.tune_projector=True
    head.tune_diffusion_model=arm=='ACTION2048'
    head.tune_vlln=False
    head.state_dropout_prob=0.
    model.config.state_dropout_prob=0.
    projectors=[];dit=[];source_rows={};packed=0;allowed=0
    for name,p in model.named_parameters():
        if is_projector(name):
            assert p.shape[0]==32
            source_rows[name]=dict(dtype=str(p.dtype),shape=list(p.shape),row10_sha256=digest(p[ROW]))
            packed+=p.numel();allowed+=p[ROW].numel()
            p.data=p.data.float();p.requires_grad_(True);p.register_hook(mask);projectors.append(p)
        elif arm=='ACTION2048' and name.startswith('action_head.model.'):
            p.data=p.data.float();p.requires_grad_(True);dit.append(p)
    assert len(projectors)==14 and packed==325783680 and allowed==10180740
    # Native head.forward invokes set_frozen_modules_to_eval_mode; no no_grad
    # surrounds the frozen DiT, so encoder input gradients remain connected.
    model.config.tune_projector=True;model.config.tune_diffusion_model=head.tune_diffusion_model;model.config.tune_vlln=False
    groups=[dict(params=projectors,lr=1e-4,weight_decay=0.,group_name='packed_projectors')]
    if dit:groups.append(dict(params=dit,lr=1e-5,weight_decay=1e-5,group_name='shared_DiT'))
    return groups,dict(arm=arm,source_rows=source_rows,packed_scalars=packed,allowed_row_scalars=allowed,
       DiT_scalars=sum(p.numel() for p in dit),optimizer_moment_bytes=8*(packed+sum(p.numel() for p in dit)),
       master_dtype='float32',projector_decay=0.,frozen_DiT_input_autograd='NATIVE_FORWARD_NO_NO_GRAD')


def frozen_hashes(model):
    result={}
    for n,p in model.named_parameters():
        if is_projector(n):
            result[n]=[digest(p[:ROW]),digest(p[ROW+1:])]
        elif not p.requires_grad:result[n]=digest(p)
    return result


def assert_frozen(model,before,optimizer):
    assert frozen_hashes(model)==before,'FROZEN_PARAMETERS_CHANGED'
    for n,p in model.named_parameters():
        if is_projector(n):
            for key in ('exp_avg','exp_avg_sq'):
                value=optimizer.state.get(p,{}).get(key)
                if value is not None:
                    assert not torch.count_nonzero(value[:ROW]) and not torch.count_nonzero(value[ROW+1:]),(n,key)


def gradient_proof(model):
    module_norms={k:0. for k in ('state_encoder','action_encoder','action_decoder')};dit=0.
    for name,p in model.named_parameters():
        if p.grad is None:continue
        assert torch.isfinite(p.grad).all(),name
        if is_projector(name):
            assert not torch.count_nonzero(p.grad[:ROW]) and not torch.count_nonzero(p.grad[ROW+1:]),name
            key=name.split('.')[1];module_norms[key]+=float(p.grad[ROW].float().norm())
        elif name.startswith('action_head.model.'):dit+=float(p.grad.float().norm())
    assert all(v>0 for v in module_norms.values()),module_norms
    return dict(row_module_gradient_norms=module_norms,DiT_gradient_norm=dit)
