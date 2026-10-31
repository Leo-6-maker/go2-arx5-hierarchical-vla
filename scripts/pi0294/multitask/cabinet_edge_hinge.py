"""Declared external edge hinge source variant, not exact RoboCasa geometry.

The original inside pivot sweeps the slab into its housing. Keep all collision
geometry active and place the pivot at the outer panel edge. Original fixtures
and historical results remain available through their unchanged cards.
"""
import math
import xml.etree.ElementTree as ET
import numpy as np
from .robocasa_cabinet_assets import numbers
from .cabinet_layout_port06 import cards as previous_cards

VERSION='ROBOCASA_ARX5_PORT07_EDGE_HINGE_LAYOUT60'
MECHANICS='OUTER_PANEL_EDGE_AXIS_EXACT_QUARTER_TURN_V1'


def geometry(xml,*,size,orientation):
    root=ET.fromstring(xml);joint=root.find(".//joint[@name='doorhinge']")
    panel=root.find(".//geom[@name='door']")
    old=np.fromstring(joint.get('pos'),sep=' ')
    side=1 if orientation=='right'else-1
    expected=np.array([side*(size[0]/2-.015),-size[1]/2,0.])
    if not np.allclose(old,expected,atol=1e-15,rtol=0):raise ValueError('EDGE_VARIANT_REQUIRES_ORIGINAL_PINNED_PIVOT')
    new=old.copy();new[0]=side*(size[0]/2-.003/2)
    joint.set('pos',numbers(new));panel.set('euler',numbers([math.pi/2,0.,0.]))
    return ET.tostring(root,encoding='unicode'),dict(mechanical_variant=MECHANICS,
        original_hinge_axis_local=old.tolist(),hinge_axis_local=new.tolist(),
        exact_upstream_mechanics=False,collision_exclusions_or_mask_changes=False)


def cards():
    result=previous_cards();right=result['right']
    right.update(id='V7_ROBOCASA_RIGHT_PORT07_EDGE_HINGE_LAYOUT60',port_revision=VERSION)
    right['cabinet']['mechanical_variant']=MECHANICS
    for frame in(right['cabinet']['cabinet_world'],right['door']['hinge_world']):frame[0][3]+=.02
    right['source_base_card']['goal_xy'][0]+=.02
    # Previous right pivot was 13.5mm inside this panel's outer edge.
    world=np.array(right['cabinet']['cabinet_world']);hinge=np.array(right['door']['hinge_world'])
    hinge[:3,3]+=world[:3,:3]@np.array([.0135,0.,0.]);right['door']['hinge_world']=hinge.tolist()
    return result
