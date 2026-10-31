"""Separate right-cabinet development layout after full-body sweep diagnosis.

The measured grasp calibration is source Teacher geometry, never a policy
input. Geometry, forces, safety, score and the32s task are unchanged. Static
forward/contact checks are not native dynamic or camera qualification.
"""
from .cabinet_port05 import cards as previous_cards

VERSION='ROBOCASA_ARX5_PORT06_LAYOUT40_RETRACT8'
GRASP_OBJECT_TO_TCP=[[0.0008048159934325564, -0.9999854830281223, -0.005327851746279447, 0.0005307630193694401], [0.7040823569633692, -0.0032167562928536657, 0.7101110385649975, -0.009052363890440451], [-0.7101178683036526, -0.004322755136021947, 0.7040695469226818, 0.005237507732047453], [0.0, 0.0, 0.0, 1.0]]


def cards():
    result=previous_cards();right=result['right']
    right.update(id='V7_ROBOCASA_RIGHT_PORT06_LAYOUT40_RETRACT8',port_revision=VERSION)
    for pose in(right['cabinet']['cabinet_world'],right['door']['hinge_world']):pose[0][3]+=.04
    right['source_base_card']['goal_xy'][0]+=.04
    right['door']['object_to_tcp_grasp']=[list(row)for row in GRASP_OBJECT_TO_TCP]
    return result
