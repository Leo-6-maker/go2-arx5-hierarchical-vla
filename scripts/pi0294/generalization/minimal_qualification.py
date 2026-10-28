"""Observations around the original Trainer's two warmup optimizer calls.

No optimizer, loss, model loader, query dispatch or launcher is implemented.
The first original LambdaLR call uses LR0, so it cannot prove weight change.
All four qualification optimizer calls across both arms remain actual effects.
"""
from .minimal_adapter import LEAVES, ARMS, frozen_fingerprint, parameter_state_fingerprint
from . import minimal_study as study

STEPS = 2


def draw_plan():
    """Keep17 total:15 matched raw draws, then trained/restored same-noise pair.

    This is the new minimal-adaptation scope's cold qualification plan. The
    historical C/D/GEN seventeen-before-update qualification is unchanged.
    All calls still use the original fixed_draw and pre-effect query debit.
    """
    return [dict(index=i,phase='ZERO_STEP_MATCHED',seed=29400+i,optimizer_step=0)
            for i in range(15)]+[
        dict(index=15,phase='QUALIFIED2_SAVED_REFERENCE',seed=29400,optimizer_step=2),
        dict(index=16,phase='QUALIFIED2_FULL_RESTORE',seed=29400,optimizer_step=2)]


def gradients(contract):
    """Call after the original final-microbatch backward, before optimizer."""
    import torch
    contract.validate()
    target = {p+'.'+key for p in LEAVES for key in ('target_W','target_b')}
    rows = {}
    for name,p in contract.model.named_parameters():
        if not p.requires_grad:
            study.need(p.grad is None, 'FROZEN_PARAMETER_RECEIVED_GRADIENT:' + name)
            continue
        g = p.grad
        if name in target:
            study.need(g is not None, 'TARGET_PARAMETER_HAS_NO_AUTOGRAD_PATH:' + name)
        if g is None:
            rows[name] = dict(present=False); continue
        study.need(bool(torch.isfinite(g).all()), 'NONFINITE_QUALIFICATION_GRADIENT:' + name)
        rows[name] = dict(present=True, finite=True, maximum_absolute=float(g.detach().abs().max()),
                          dtype=str(g.dtype), shape=list(g.shape))
    for group in ('state_encoder','action_encoder','action_decoder'):
        names = [n for n in target if n.startswith('action_head.'+group+'.')]
        study.need(any(rows[n]['maximum_absolute'] > 0 for n in names),
                   'NO_NONZERO_TARGET_COMPONENT_GRADIENT:' + group)
    return rows


def after_optimizer(contract, *, initial, frozen, gradient_rows, step, cursor):
    import torch
    contract.validate()
    study.need(step == STEPS and cursor == STEPS*32 and gradient_rows,
               'QUALIFICATION_EXACT_TWO_ORIGINAL_CALLS_AND64_VISITS')
    study.need(frozen_fingerprint(contract.model) == frozen, 'QUALIFICATION_CHANGED_FROZEN_TENSOR')
    for name,p in contract.model.named_parameters():
        if p.requires_grad:
            study.need(bool(torch.isfinite(p).all()), 'QUALIFICATION_NONFINITE_UPDATED_PARAMETER:' + name)
    actual = parameter_state_fingerprint(contract.model)
    study.need(set(actual) == set(initial), 'QUALIFICATION_CHANGED_STATE_LAYOUT')
    changed = [n for n in contract.trainable_names if actual['trainable:'+n] != initial['trainable:'+n]]
    for group in ('state_encoder','action_encoder','action_decoder'):
        study.need(any(n.startswith('action_head.'+group+'.') for n in changed),
                   'QUALIFICATION_NO_REAL_TARGET_COMPONENT_UPDATE:' + group)
    if contract.arm == ARMS[1]:
        study.need(any(n.startswith(('action_head.model.','action_head.vlln.',
                                    'action_head.vl_self_attention.')) for n in changed),
                   'QUALIFICATION_NO_REAL_SHARED_UPDATE')
    return dict(status='ACTUAL_ORIGINAL_TRAINER_TWO_WARMUP_CALL_PARAMETER_UPDATE_CHECK',
                optimizer_calls=STEPS, cursor=cursor, arm=contract.arm,
                frozen_tensors_unchanged=True, gradient_rows=gradient_rows,
                changed_trainable_parameters=changed,
                formal_training_start_requires_fresh_raw_parent=True,
                qualification_is_development_checkpoint=False,
                native_model_qualification_complete=False,
                remaining='Full framework optimizer/scheduler/RNG/sampler restore and output equivalence')
