"""Same edge-hinge geometry, with a finite deceleration before the endpoint.

PORT07's constant-speed hinge reference stopped in one 20Hz interval. Its
first grasp failure coincided with that stop, after reaching the native goal.
This version changes the source Teacher time law, never grasp/safety scoring.
"""
import math
import numpy as np
from .cabinet_edge_hinge import cards as previous_cards
from .object_relations import DoorTurnReference,axis_rotation,inverse,pose

VERSION='ROBOCASA_ARX5_PORT08_EDGE_HINGE_BRAKED_TURN'
PROFILE='COSINE_FINAL_0_8S_DECELERATION_V1'


def cards():
    result=previous_cards();right=result['right']
    right.update(id='V7_ROBOCASA_RIGHT_PORT08_EDGE_HINGE_BRAKED_TURN',port_revision=VERSION,
        turn_time_profile=PROFILE)
    return result


def brake(reference,hinge_world,axis_hinge=(0,0,1)):
    """Retain the original safe anchor/arc and peak speed, add <=0.45s.

    Acceleration to the existing peak is unchanged at the start; the last
    0.8 seconds smoothly reduce angular speed to zero. A slightly negative
    safe native soft-limit anchor is retained; all future points must stay
    inside the original declared limits, as checked by the caller.
    """
    start,end=map(float,reference.hinge_angles_rad[[0,-1]])
    delta=end-start;base=float(reference.times_s[-1]);ramp=.8
    if base<ramp or abs(delta)<1e-8:raise ValueError('BRAKED_TURN_REQUIRES_FINITE_FULL_ARC')
    ticks=math.ceil((base+ramp/2)/.05)
    if ticks>640:raise ValueError('BRAKED_TURN_EXCEEDS_ORIGINAL32S')
    times=np.arange(ticks+1)*.05;duration=float(times[-1]);peak=delta/(duration-ramp/2)
    if abs(peak)>abs(delta)/base+1e-12:raise ValueError('BRAKE_CANNOT_INCREASE_ORIGINAL_REFERENCE_SPEED')
    distance=peak*times;mask=times>duration-ramp
    u=times[mask]-(duration-ramp)
    distance[mask]=peak*(duration-ramp+u/2+ramp/(2*math.pi)*np.sin(math.pi*u/ramp))
    angles=start+distance;angles[0]=start;angles[-1]=end
    h=pose(hinge_world);anchor=pose(reference.tcp_world[0])
    future=np.stack([h@axis_rotation(axis_hinge,float(a-start))@inverse(h)@anchor for a in angles])
    if not np.allclose(future[[0,-1]],reference.tcp_world[[0,-1]],rtol=0,atol=1e-10):raise ValueError('BRAKE_RETAINS_MEASURED_ARC_ENDPOINTS')
    return DoorTurnReference(times,future,angles)
