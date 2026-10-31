"""Passive pinned cabinet around the unchanged source ARX5; no constructor."""
from copy import deepcopy
import xml.etree.ElementTree as ET

import numpy as np

from .contact_scene import body,numbers
from .object_relations import pose
from .robocasa_cabinet_assets import single_cabinet_geometry,bind_minimum_textures

VERSION='ROBOCASA_SINGLE_CABINET_ARX5_SOURCE_PORT_V1'


def overlay(source_xml,card,*,definitions,texture_paths):
    if card.get('source_port')!=VERSION or card.get('task')!='DOOR':
        raise ValueError('EXPLICIT_PINNED_SINGLE_CABINET_SOURCE_PORT')
    spec=card['cabinet'];door_spec=card['door']
    fixture,meta=single_cabinet_geometry(definitions,size=spec['size_m'],orientation=spec['orientation'])
    mechanical_meta={}
    if spec.get('mechanical_variant')is not None:
        from .cabinet_edge_hinge import MECHANICS,geometry
        if spec['mechanical_variant']!=MECHANICS:raise ValueError('EXPLICIT_DECLARED_CABINET_MECHANICAL_VARIANT')
        fixture,mechanical_meta=geometry(fixture,size=spec['size_m'],orientation=spec['orientation'])
    fixture,texture_meta=bind_minimum_textures(fixture,texture_paths);fixture=ET.fromstring(fixture)
    cabinet_world=pose(spec['cabinet_world'])
    joint=fixture.find(".//joint[@name='doorhinge']");hinge_local=np.fromstring(joint.get('pos'),sep=' ')
    hinge_world=cabinet_world.copy();hinge_world[:3,3]+=cabinet_world[:3,:3]@hinge_local
    direction=1 if spec['orientation']=='right'else-1
    panel_height=float(spec['size_m'][2])-.003;handle_length=min(.24,panel_height-.08)
    if (door_spec['initial_angle_rad']!=0 or door_spec['angle_limits_rad']!=meta['joint_range']
            or door_spec['target_angle_rad']!=direction*.95*1.57
            or door_spec['handle_radius_m']!=.013
            or abs(door_spec['handle_length_m']-handle_length)>1e-12
            or not np.allclose(pose(door_spec['hinge_world']),hinge_world,atol=1e-10,rtol=0)):
        raise ValueError('ACTUAL_PASSIVE_HINGE_FRAME_RANGE_AND_HANDLE_DIMENSIONS')
    root=ET.fromstring(source_xml);world,assets=root.find('worldbody'),root.find('asset')
    compiler=root.find('compiler')
    if (root.get('model')not in('PI0291_FIXED_BASE_ARX5_TASK_SIM','PI0294_FIXED_BASE_ARX5_MULTITASK')
            or world is None or assets is None or compiler is None or compiler.get('angle')!='radian'
            or compiler.get('inertiafromgeom','auto')not in('auto','false')):
        raise ValueError('UNCHANGED_ORIGINAL_RADIAN_SOURCE_SCENE')
    robot=world.find("body[@name='base_link']");cube=world.find("body[@name='cube']")
    if robot is None or robot.find('freejoint')is not None or cube is None:
        raise ValueError('ORIGINAL_FIXED_BASE_SOURCE_ROBOT_AND_CUBE')
    if any(x.get('name','').startswith('rc_')for x in root.iter()):
        raise ValueError('FRESH_PASSIVE_CABINET_NAMESPACE')
    protected={k:ET.tostring(root.find(k))if root.find(k)is not None else None
        for k in('actuator','contact','equality','option','compiler')}
    original_robot=ET.tostring(robot)
    defaults=root.find('default')
    if defaults is None:defaults=ET.SubElement(root,'default')
    original_default_members=[ET.tostring(x)for x in defaults]
    def prefixed(element):
        result=deepcopy(element)
        for item in result.iter():
            for key in('name','class','childclass','material','mesh','texture'):
                if key in item.attrib:item.set(key,'rc_'+item.attrib[key])
        return result
    # Only named fixture classes are appended. Original global defaults and
    # robot/control/solver nodes are retained; native contact/inertia remain
    # subject to the original source owner's model and safety qualification.
    for item in fixture.findall('default/default'):defaults.append(prefixed(item))
    for item in fixture.findall('asset/*'):assets.append(prefixed(item))
    world.remove(cube)
    for marker in list(world.findall('geom')):
        if marker.get('name','').startswith('goal_marker_'):
            if marker.get('contype')!='0'or marker.get('conaffinity')!='0':
                raise ValueError('ONLY_NONCONTACT_GOAL_MARKER_REMOVAL')
            world.remove(marker)
    cabinet=body(world,'rc_cabinet_frame',cabinet_world)
    for item in fixture.find("worldbody/body/body[@name='object']"):cabinet.append(prefixed(item))
    handle=cabinet.find(".//body[@name='rc_door_handle']")
    grasp=handle.find("geom[@name='rc_handle']")
    # The original handle body's origin sits at the panel attachment. The
    # measured grasp frame is at its actual cylinder center, without moving
    # the original geom, hinge or door body.
    ET.SubElement(handle,'body',name='rc_handle_grasp_frame',pos=grasp.get('pos','0 0 0'))
    ET.SubElement(cabinet,'body',name='rc_hinge_frame',pos=numbers(hinge_local))
    obstacles=[x.get('name')for x in cabinet.iter('geom')if x.get('class')=='rc_cab'and x.get('name')!='rc_handle']
    if (ET.tostring(robot)!=original_robot or len(obstacles)!=8
            or [ET.tostring(x)for x in list(defaults)[:len(original_default_members)]]!=original_default_members
            or any(v!=(ET.tostring(root.find(k))if root.find(k)is not None else None)for k,v in protected.items())):
        raise ValueError('SOURCE_ROBOT_DEFAULTS_CONTROL_CONTACT_OR_INTEGRATOR_CHANGED')
    root.set('model',VERSION)
    return ET.tostring(root,encoding='unicode'),dict(version=VERSION,object_body='rc_handle_grasp_frame',
        native_handle_body='rc_door_handle',grasp_geom='rc_handle',passive_joint='rc_doorhinge',
        joint_angle_offset_rad=0,fixture_frame='rc_hinge_frame',hinge_world=hinge_world.tolist(),
        obstacle_geoms=obstacles,goal_kind='ROBOCASA_NATIVE_DOOR',joint_ranges={'rc_doorhinge':meta['joint_range']},
        original_handle_origin_separate_from_grasp_frame=True,robot_XML_preserved=True,
        texture_members=texture_meta['files'],qualification='UNCOMPILED_UNEXECUTED_SOURCE_PORT',
        original_kitchen_benchmark=False,new_actuators_equalities_mocap=0,**mechanical_meta)
