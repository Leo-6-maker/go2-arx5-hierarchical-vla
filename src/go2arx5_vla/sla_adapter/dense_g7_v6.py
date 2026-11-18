"""Pointwise target G7 and predeclared full-generated grip-speed objective.

No source statistics/width retargeting, block averaging, clipping or retiming.
The unchanged native H40 compiler remains the execution authority.
"""
import numpy as np

VERSION='DENSE_REFERENCE_G7_V6'
DT=.05
GRIP_MARGIN=.038
GRIP_LIMIT=.040


def previous_command_context(reference):
    """Validate a recorded/confirmed clock; never synthesize a past command."""
    if set(reference)!={'width_m','source_native_s','query_native_s'}:raise ValueError('EXACT_PREVIOUS_ACTUAL_COMMAND_CONTEXT_REQUIRED')
    try:width,source,query=[float(reference[k])for k in('width_m','source_native_s','query_native_s')]
    except (TypeError,ValueError)as error:raise ValueError('REAL_PREVIOUS_COMMAND_VALUES_REQUIRED')from error
    if not np.isfinite([width,source,query]).all()or not .002-1e-10<=width<=.042+1e-10 or abs(query-source-DT)>1e-8:
        raise ValueError('REAL_50MS_PREVIOUS_EMITTED_WIDTH_CLOCK_REQUIRED')
    return width


def grip_speed_loss(normalized,q01,q99,previous_width,*,xp,sigmoid):
    if normalized.shape[-2:]!=(40,32):raise ValueError('FULL_GENERATED_TARGET_H40_REQUIRED')
    latent=(normalized[...,6]+1)*.5*(q99[6]-q01[6]+1e-6)+q01[6]
    width=.002+.040*sigmoid(latent)
    previous=xp.asarray(previous_width,dtype=width.dtype)
    if previous.shape!=width.shape[:-1]:raise ValueError('ONE_ACTUAL_PREVIOUS_WIDTH_PER_GENERATED_H40_REQUIRED')
    speed=xp.abs(xp.diff(xp.concatenate([previous[...,None],width],axis=-1),axis=-1))/DT
    penalty=xp.maximum((speed-GRIP_MARGIN)/GRIP_LIMIT,0)**2
    return xp.mean(xp.max(penalty,axis=-1)),dict(max_speed=xp.max(speed),first_seam_speed=xp.mean(speed[...,0]),interior_max_speed=xp.max(speed[...,1:]))


def full_objective(normalized,q01,q99,previous_width,*,xp,sigmoid):
    from .full_h40_v6 import loss as XYZ_loss
    xyz,parts=XYZ_loss(normalized,q01,q99,xp=xp)
    grip,detail=grip_speed_loss(normalized,q01,q99,previous_width,xp=xp,sigmoid=sigmoid)
    return .1*xyz+.1*grip,dict(XYZ_dyn=xyz,gripper_speed=grip,XYZ_parts=parts,gripper=detail)


def label_margin_report(nominal_h40,reference):
    action=np.asarray(nominal_h40);previous=previous_command_context(reference)
    if action.shape!=(40,10)or not np.isfinite(action).all()or((action[:,9]<0)|(action[:,9]>1)).any():raise ValueError('ACTUAL_NOMINAL_H40_LABEL_REQUIRED')
    width=.002+.040*action[:,9];speed=abs(np.diff(np.r_[previous,width]))/DT
    return dict(first_seam_speed_m_s=float(speed[0]),maximum_speed_m_s=float(speed.max()),
        training_margin_exceeded=bool((speed>GRIP_MARGIN+1e-10).any()),original_execution_speed_exceeded=bool((speed>GRIP_LIMIT+1e-10).any()),
        labels_not_retimed=True,representation=VERSION)
