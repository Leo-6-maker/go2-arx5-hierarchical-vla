"""M4-B causal S-only attachment prior; not a native grasp detector or safety rule."""
import torch


def attachment_gate(state,reference,clock,context,slices):
    """Declared conservative geometric hypothesis, never Teacher/native event inputs.

    The command is the *consumed* 50ms point, not the first/last H40 row.
    FIT anchor fractions are ~.433 when held and ~1 when open. The broad
    [.2,.6] band is frozen before training, not selected on DEV outcomes.
    """
    if context.shape[-1]!=5:raise ValueError('S cube context only; no target padding')
    point=torch.round(clock[:,1]/.05).to(torch.long)
    if torch.any((point<0)|(point>7)):raise ValueError('Legal consumed first8 clock required')
    command=reference[torch.arange(len(state),device=state.device),point,9]
    grip=state[:,slices['gripper_fraction']].squeeze(-1)
    offset=state[:,slices['object_position']]-state[:,slices['tcp_position']]
    # Rotation-invariant center bound, relative to the observed cube edge.
    near=offset.norm(dim=-1)<=context[:,4]*.5
    return (grip>=.2)&(grip<=.6)&(command<=.2)&near


def conditional_rigid_residual(state,independent,reference,clock,context,slices):
    """Execution predicts TCP; attached interaction predicts slip around SE3 carry.

    Identical parameters, inputs and residual scales. Only interpretation of
    the object pose delta changes under the declared causal attachment gate.
    Opening/uncertain modes keep the independent learned interaction head.
    """
    gate=attachment_gate(state,reference,clock,context,slices)
    rt=state[:,slices['tcp_rotation']].reshape(-1,3,3)
    rn=independent[:,slices['tcp_rotation']].reshape(-1,3,3)
    change=rn@rt.transpose(-1,-2)
    old_object=state[:,slices['object_position']]
    carried=independent[:,slices['tcp_position']]+(change@(old_object-state[:,slices['tcp_position']])[...,None]).squeeze(-1)
    slip=independent[:,slices['object_position']]-old_object
    # Existing object rotation residual is composed after the TCP rotation.
    rotation=change@independent[:,slices['object_rotation']].reshape(-1,3,3)
    parts=[]
    for name,sl in slices.items():
        value=independent[:,sl]
        if name=='object_position':value=torch.where(gate[:,None],carried+slip,value)
        elif name=='object_rotation':value=torch.where(gate[:,None],rotation.flatten(-2),value)
        parts.append(value)
    return torch.cat(parts,-1)
