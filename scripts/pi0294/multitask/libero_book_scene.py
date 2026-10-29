"""Pinned LIBERO objects around the original fixed-base ARX5 source robot.

The robot, driver, cameras and integrator remain the existing source version.
This port is not the original LIBERO robot benchmark or a target-WBC task.
"""
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .contact_scene import body, numbers
from .libero_book_assets import pinned_definition
from .object_relations import pose

VERSION = 'LIBERO_PINNED_BOOK_CADDY_ARX5_SOURCE_PORT_V1'


def overlay(source_xml, card, *, definitions, asset_paths):
    if card.get('source_port') != VERSION or card.get('goal_side') not in ('left', 'right'):
        raise ValueError('EXPLICIT_PINNED_BOOK_PORT_AND_GOAL_SIDE')
    specs = {n: pinned_definition(n, definitions[n]) for n in ('black_book', 'desk_caddy')}
    required = {n+'/'+p for n, s in specs.items() for p in s['asset_files']}
    if set(asset_paths) != required or any(not Path(p).is_absolute() for p in asset_paths.values()):
        raise ValueError('COMPLETE_PREVERIFIED_MINIMUM_ASSET_PATH_MAP')
    root = ET.fromstring(source_xml)
    world, assets = root.find('worldbody'), root.find('asset')
    compiler = root.find('compiler')
    if (root.get('model') not in ('PI0291_FIXED_BASE_ARX5_TASK_SIM', 'PI0294_FIXED_BASE_ARX5_MULTITASK')
            or world is None or assets is None or compiler is None or compiler.get('angle') != 'radian'
            or compiler.get('inertiafromgeom', 'auto') not in ('auto', 'false')):
        raise ValueError('UNCHANGED_ORIGINAL_RADIAN_SOURCE_SCENE_REQUIRED')
    robot, cube = world.find("body[@name='base_link']"), world.find("body[@name='cube']")
    if robot is None or robot.find('freejoint') is not None or cube is None:
        raise ValueError('ORIGINAL_FIXED_BASE_SOURCE_ROBOT_AND_CUBE_REQUIRED')
    protected = {k: ET.tostring(root.find(k)) if root.find(k) is not None else None
                 for k in ('actuator', 'contact', 'equality', 'option')}
    original_robot = ET.tostring(robot)
    world.remove(cube)
    for marker in list(world.findall('geom')):
        if marker.get('name', '').startswith('goal_marker_'):
            if marker.get('contype') != '0' or marker.get('conaffinity') != '0':
                raise ValueError('ONLY_NONCONTACT_GOAL_MARKER_REMOVAL')
            world.remove(marker)
    collisions = {}
    for name in ('black_book', 'desk_caddy'):
        original = ET.fromstring(definitions[name])
        prefix = 'libero_'+name+'_'
        for element in original.find('asset'):
            a = deepcopy(element)
            for field in ('name', 'mesh', 'material', 'texture'):
                if field in a.attrib: a.set(field, prefix+a.attrib[field])
            if 'file' in a.attrib: a.set('file', asset_paths[name+'/'+a.attrib['file']])
            assets.append(a)
        transform = pose(card['initial_book_native_world'] if name == 'black_book' else card['caddy_world'])
        target = body(world, 'libero_'+name, transform)
        if name == 'black_book':
            ET.SubElement(target, 'joint', name='libero_book_free', type='free', damping='.0005')
            # robosuite's pinned base.xml has inertiagrouprange="0 0".
            # Reproduce this book's collision-only inertia explicitly instead
            # of changing that global compiler setting for the source robot.
            geom = specs[name]['collision_geoms'][0]
            half = np.fromstring(geom['size'], sep=' ')
            mass = float(geom['density'])*8*np.prod(half)
            inertia = mass/3*(np.sum(half**2)-half**2)
            ET.SubElement(target, 'inertial', pos=geom['pos'], quat=geom['quat'],
                          mass=numbers([mass]), diaginertia=numbers(inertia))
        collisions[name] = []
        for index, element in enumerate(original.find("worldbody/body/body[@name='object']")):
            a = deepcopy(element)
            for field in ('mesh', 'material', 'texture'):
                if field in a.attrib: a.set(field, prefix+a.attrib[field])
            a.set('name', prefix+a.get('name', a.tag+'_'+str(index)))
            if a.tag == 'geom' and a.get('group') == '0': collisions[name].append(a.get('name'))
            target.append(a)
        if name == 'black_book':
            # A separate semantic grasp frame cannot change the native body
            # origin used by LIBERO's In predicate or move its collision geom.
            geom = specs[name]['collision_geoms'][0]
            ET.SubElement(target, 'body', name='skill_book', pos=geom['pos'], quat=geom['quat'])
    if (original_robot != ET.tostring(robot) or any(v != (ET.tostring(root.find(k))
            if root.find(k) is not None else None) for k, v in protected.items())):
        raise ValueError('ROBOT_CONTROL_CONTACT_INTEGRATOR_OR_PHYSICS_CHANGED')
    root.set('model', VERSION)
    return ET.tostring(root, encoding='unicode'), dict(version=VERSION,
        object_body='skill_book', native_object_body='libero_black_book',
        grasp_geom=collisions['black_book'][0], obstacle_geoms=collisions['desk_caddy'],
        fixture_frame='libero_desk_caddy', goal_kind='LIBERO_NATIVE_SITE',
        native_goal_site='libero_desk_caddy_'+card['goal_side']+'_contain_region',
        robot_XML_preserved=True, qualification='UNEXECUTED_PINNED_SOURCE_PORT',
        native_book_origin_separate_from_grasp_frame=True)
