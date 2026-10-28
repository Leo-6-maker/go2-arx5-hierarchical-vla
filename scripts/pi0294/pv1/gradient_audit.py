"""Equivalent gradient evidence with one device-to-host read, not new loss math."""


def _group(name):
    return 'diffusion' if name.startswith('action_head.model.') else 'projector'


def legacy(named_parameters, torch):
    """The existing callback, retained as the native equivalence reference."""
    values=dict(projector=0.,diffusion=0.);checked=reads=0
    for name,p in named_parameters:
        if p.grad is None:continue
        if not p.requires_grad:
            raise ValueError('PV1_NONFINITE_OR_FROZEN_PARAMETER_GRADIENT')
        checked+=1;reads+=1
        if not bool(torch.isfinite(p.grad).all()):
            raise ValueError('PV1_NONFINITE_OR_FROZEN_PARAMETER_GRADIENT')
        if name.startswith('action_head.'):
            values[_group(name)]+=float(p.grad.float().square().sum());reads+=1
    return dict(norm_squared=values,gradient_tensors_checked=checked,host_reads=reads,mode='LEGACY_PER_TENSOR')


def aggregated(named_parameters, torch):
    """Preserve every per-tensor reduction and the Python accumulation order.

    Only scalar *transport* is batched. Float32 square/sum happens exactly as
    before; converting those scalars to float64 is lossless. Python adds the
    returned values in original parameter order, avoiding a new reduction tree.
    No gradients are modified, no random operation or optimizer is invoked.
    """
    flags=[];sums=[];groups=[];device=None
    for name,p in named_parameters:
        if p.grad is None:continue
        if not p.requires_grad:
            raise ValueError('PV1_NONFINITE_OR_FROZEN_PARAMETER_GRADIENT')
        if device is None:device=p.grad.device
        elif p.grad.device!=device:
            raise ValueError('PV1_SINGLE_DEVICE_GRADIENT_AUDIT_REQUIRED')
        flags.append(torch.isfinite(p.grad).all())
        if name.startswith('action_head.'):
            sums.append(p.grad.float().square().sum());groups.append(_group(name))
    values=dict(projector=0.,diffusion=0.)
    if flags:
        parts=[torch.stack(flags).to(dtype=torch.float64)]
        if sums:parts.append(torch.stack(sums).to(dtype=torch.float64))
        host=torch.cat(parts).cpu().tolist()
        if not all(value==1. for value in host[:len(flags)]):
            raise ValueError('PV1_NONFINITE_OR_FROZEN_PARAMETER_GRADIENT')
        for group,value in zip(groups,host[len(flags):]):values[group]+=value
    return dict(norm_squared=values,gradient_tensors_checked=len(flags),host_reads=int(bool(flags)),
        mode='BATCHED_SCALAR_TRANSPORT_ORIGINAL_REDUCTIONS')


def equivalent(reference, candidate):
    return (reference['norm_squared']==candidate['norm_squared']
        and reference['gradient_tensors_checked']==candidate['gradient_tensors_checked'])
