"""Bind inherited differentiable S4 physical losses to the frozen unclamped R2 codec."""
import torch
from pi0245_physical_delta import decode_xyz as STOCK_DECODE_XYZ


def bounds(statistics):
    import numpy as np
    params=statistics['new_embodiment']['relative_action']['eef_9d']
    lo=np.asarray(params['q01'],dtype=np.float32)
    hi=lo+np.maximum(np.asarray(params['q99'],dtype=np.float32)-lo,1e-6)
    assert lo.shape==hi.shape==(40,9)
    return torch.from_numpy(lo),torch.from_numpy(hi)


def decode_xyz(normalized,state_eef,scale,offset):
    # Existing meanstd branch is exactly the unclamped affine inverse when scale > 0.
    return STOCK_DECODE_XYZ(normalized,state_eef,scale,offset,mode='meanstd')


def decode_relative_rotation(normalized,state_rot6d,minimum,maximum):
    from pi0248_rotation_cpu import row_rot6d_to_matrix
    if normalized.ndim!=3 or normalized.shape[1:]!=(40,6) or state_rot6d.shape!=(normalized.shape[0],6):
        raise ValueError('R3 rotation tensor shape')
    lo=minimum.to(normalized.device,torch.float32)
    hi=maximum.to(normalized.device,torch.float32)
    assert lo.shape==hi.shape==(40,6) and torch.all(hi>lo)
    relative=(normalized.float()+1)*.5*(hi-lo)+lo
    rotation,n1,n2=row_rot6d_to_matrix(relative)
    anchor,s1,s2=row_rot6d_to_matrix(state_rot6d.float())
    return anchor[:,None]@rotation,dict(relative_row1_norm=n1,relative_row2_orth_norm=n2,
                                       state_row1_norm=s1,state_row2_orth_norm=s2,relative_physical_rot6d=relative)


def install():
    import pi0245_physical_delta as delta
    import pi0246_physical_position as position
    import pi0247_u_rollout_overlay as rollout
    import pi0248_rotation_cpu as rotation
    delta.decode_xyz=decode_xyz
    position.decode_xyz=decode_xyz
    rollout.decode_xyz=decode_xyz
    rollout.CLAMP_NORMALIZED_STATE=False
    rotation.decode_relative_rotation=decode_relative_rotation
