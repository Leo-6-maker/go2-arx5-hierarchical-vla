"""Learned robot-time event parameters, shared by TRAIN and deployment.

This is a representation change, not a filter applied to legacy width H40.
No privileged phase or expert close_start is accepted by the decoder.
"""
import numpy as np
from .gripper_latent_v2 import EPSILON,label_latent,decode_latent

VERSION='SLA_REFERENCE_CONTINUED_GRIP_TIME_V1'
DT=.05
LOW_M=.002
HIGH_M=.042
TRAVEL_M=.040
VMAX=.040
SLACK_MAX_S=.5
DELAY_MAX_S=4.
BLOCKS=((0,14),(14,27),(27,40)) # goal, duration slack, delayed onset


def quintic(u):
    # Event definition includes its pre-onset and post-completion constant parts.
    u=np.clip(np.asarray(u,dtype=np.float64),0.,1.)
    return u**3*(10.+u*(-15.+6.*u))


def inverse_quintic(progress):
    if not np.isfinite(progress)or not 0<=progress<=1:raise ValueError('FINITE_UNIT_EVENT_PROGRESS')
    lo,hi=0.,1.
    for _ in range(64):
        mid=(lo+hi)/2
        if float(quintic(mid))<progress:lo=mid
        else:hi=mid
    return (lo+hi)/2


def reference_width(context):
    w=float(context['width_m']);source=float(context['source_native_s']);query=float(context['query_native_s'])
    if(not np.isfinite([w,source,query]).all()or abs(query-source-DT)>1e-8
            or w<LOW_M-1e-10 or w>HIGH_M+1e-10):
        raise ValueError('PREVIOUS_ACTUALLY_EMITTED_REFERENCE_AND_EXACT_50MS_CLOCK_REQUIRED')
    return w


def decode(normalized40,context,low,high):
    values=np.asarray(normalized40,dtype=np.float64)
    if values.shape!=(40,)or not np.isfinite(values).all()or not np.isfinite([low,high]).all()or high<=low:
        raise ValueError('FINITE_TIME_PARAMETER_CONTAINER_AND_ORIGINAL_QUANTILES_REQUIRED')
    previous=reference_width(context)
    normalized=np.asarray([values[a:b].mean()for a,b in BLOCKS])
    parameter_latents=(normalized+1)/2*(high-low+1e-6)+low
    probabilities=decode_latent(parameter_latents)
    goal=LOW_M+TRAVEL_M*probabilities[0]
    start=LOW_M if goal>=previous else HIGH_M
    amplitude=goal-start
    duration=1.875*abs(amplitude)/VMAX+SLACK_MAX_S*probabilities[1]
    delay=DELAY_MAX_S*probabilities[2]
    if abs(goal-previous)<=1e-15:
        width=np.full(40,previous);phase=1.;used_delay=0.
    else:
        if duration<=0:raise ValueError('POSITIVE_NONTRIVIAL_EVENT_DURATION_REQUIRED')
        # Only sub-1e-10m input/float roundoff can extend an endpoint. No model
        # prediction, width trajectory, speed, or duration is clipped here.
        progress=float(np.clip((previous-start)/amplitude,0.,1.))
        phase=inverse_quintic(progress)
        used_delay=delay if progress<=1e-12 else 0.
        time=phase*duration+DT*(1+np.arange(40))-used_delay
        width=start+amplitude*quintic(time/duration)
    if not np.isfinite(width).all():raise ValueError('FINITE_ANALYTICAL_EVENT_REQUIRED')
    return width,dict(version=VERSION,goal_width_m=float(goal),event_start_endpoint_m=start,
        event_total_duration_s=float(duration),learned_delay_s=float(delay),used_delay_s=float(used_delay),
        reference_phase=float(phase),previous_reference_width_m=previous,
        analytical_speed_bound_m_s=0. if not duration else 1.875*abs(amplitude)/duration,
        raw_H40_width_postprocessing=False,teacher_phase_input=False)


def label_parameters(width40,context,low,high):
    """Extract only from the submitted expert H40 and previous emitted command."""
    width=np.asarray(width40,dtype=np.float64)
    if width.shape!=(40,)or not np.isfinite(width).all()or np.any(width<LOW_M-1e-10)or np.any(width>HIGH_M+1e-10):
        raise ValueError('ORIGINAL_FINITE_EXPERT_WIDTH_H40_REQUIRED')
    # Preserve the original declared TRAIN label epsilon at the two endpoints.
    # This changes labels only; the decoder never clips model predictions.
    target=LOW_M+TRAVEL_M*np.clip((width-LOW_M)/TRAVEL_M,EPSILON,1-EPSILON)
    previous=reference_width(context);difference=np.diff(target)
    moving=bool(np.ptp(target)>1e-10)
    if moving:
        closing=target[-1]<target[0]
        if(closing and np.any(difference>1e-10))or(not closing and np.any(difference< -1e-10)):
            raise ValueError('SINGLE_EXPERT_EVENT_H40_REQUIRED_NO_SILENT_FIT')
        probability=EPSILON if closing else 1-EPSILON
        goal=LOW_M+TRAVEL_M*probability
        start=LOW_M if goal>=previous else HIGH_M
        slack=2.-1.875*abs(goal-start)/VMAX # Expert's submitted total event is2s.
        delay=DT # First event point is query time, previous source is query-DT.
    else:
        probability=float((target[-1]-LOW_M)/TRAVEL_M)
        slack=.125;delay=DELAY_MAX_S # Explicit hold target, not forced closure.
    parameters=np.asarray([probability,slack/SLACK_MAX_S,delay/DELAY_MAX_S])
    latent=label_latent(parameters)
    normalized=(latent-low)/(high-low+1e-6)*2-1
    result=np.empty(40,dtype=np.float32)
    for value,(a,b)in zip(normalized,BLOCKS):result[a:b]=value
    rebuilt,detail=decode(result,context,low,high)
    return result,dict(moving=moving,maximum_width_reconstruction_error_m=float(np.max(abs(rebuilt-width))),
                       **detail)
