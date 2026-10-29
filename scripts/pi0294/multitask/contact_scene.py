"""Passive source task objects around an unchanged fixed-base source robot.

The input is the existing source XML, not the floating target/WBC model. Only
the source cube and non-contact goal markers are replaced. No new actuator,
weld, mocap, robot parameter or integration is supplied by this module.
"""
import math
import xml.etree.ElementTree as ET

import numpy as np

from go2arx5_vla.trajectory.so3 import matrix_to_quat
from .object_relations import pose, axis_rotation, book_insertion

VERSION = 'SOURCE_PASSIVE_DOOR_FREE_BOOK_CONTACT_SCENE_V1'


def numbers(values):
    return ' '.join(format(float(v), '.17g') for v in values)


def scalar(value, low, high):
    if type(value) not in (float, int) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError('BOUNDED_TASK_PHYSICAL_PARAMETER_REQUIRED')
    return float(value)


def body(parent, name, transform):
    p = pose(transform)
    xyzw = matrix_to_quat(p[:3, :3])
    return ET.SubElement(parent, 'body', name=name, pos=numbers(p[:3, 3]),
                         quat=numbers(xyzw[[3, 0, 1, 2]]))


def box(parent, name, center, half_size, **kw):
    return ET.SubElement(parent, 'geom', name=name, type='box', pos=numbers(center),
        size=numbers(half_size), friction='.8 .02 .002', condim='4', margin='0',
        solref='.008 1', **{k: str(v) for k, v in kw.items()})


def add_door(world, spec):
    width = scalar(spec['width_m'], .08, .4)
    height = scalar(spec['height_m'], .08, .5)
    thick = scalar(spec['thickness_m'], .006, .04)
    mass = scalar(spec['panel_mass_kg'], .05, 1.)
    initial = scalar(spec['initial_angle_rad'], -2., 2.)
    limits = np.asarray(spec['angle_limits_rad'], dtype=float)
    target = float(spec['target_angle_rad'])
    handle = np.asarray(spec['handle_offset_m'], dtype=float)
    radius = scalar(spec['handle_radius_m'], .005, .015)
    length = scalar(spec['handle_length_m'], .025, .08)
    if (limits.shape != (2,) or not np.isfinite(limits).all()
            or not -2.5 <= limits[0] < limits[1] <= 2.5
            or not limits[0] <= initial <= limits[1] or not limits[0] <= target <= limits[1]
            or abs(target-initial) < .17
            or handle.shape != (3,) or not np.isfinite(handle).all()
            or not radius < handle[0] < width-radius or not length/2 < handle[2] < height-length/2
            or not -.10 <= handle[1] < -thick/2-radius-.015):
        raise ValueError('FINITE_PASSIVE_DOOR_AND_ACCESSIBLE_HANDLE_GEOMETRY_REQUIRED')
    hinge = body(world, 'skill_hinge_frame', spec['hinge_world'])
    panel = body(hinge, 'skill_door', axis_rotation([0, 0, 1], initial))
    ET.SubElement(panel, 'joint', name='skill_hinge', type='hinge', axis='0 0 1', limited='true',
        range=numbers(limits-initial), damping='.02', frictionloss='.01', stiffness='0')
    box(panel, 'skill_door_panel', [width/2, 0, height/2], [width/2, thick/2, height/2],
        mass=mass, rgba='.36 .48 .65 1')
    handle_frame = np.eye(4); handle_frame[:3, 3] = handle
    grip = body(panel, 'skill_handle', handle_frame)
    ET.SubElement(grip, 'geom', name='skill_handle_geom', type='cylinder',
        size=numbers([radius, length/2]), mass='.025', friction='1.2 .02 .002',
        condim='4', margin='0', solref='.008 1', rgba='.8 .65 .2 1')
    # The rigid connector belongs to the passive articulated door, not the tool.
    ET.SubElement(panel, 'geom', name='skill_handle_connector', type='capsule',
        fromto=numbers([handle[0], -thick/2, handle[2], *handle]), size='.004',
        mass='.01', rgba='.8 .65 .2 1')
    return dict(task='DOOR', object_body='skill_handle', grasp_geom='skill_handle_geom',
        passive_joint='skill_hinge', joint_angle_offset_rad=initial,
        fixture_frame='skill_hinge_frame', obstacle_geoms=['skill_door_panel', 'skill_handle_connector'],
        new_actuators=0, new_equalities=0, new_mocap=0)


def add_book(world, spec):
    size = np.asarray(spec['book_size_m'], dtype=float)
    bounds = np.asarray(spec['cavity_bounds_shelf_m'], dtype=float)
    thickness = scalar(spec['wall_thickness_m'], .006, .03)
    mass = scalar(spec['book_mass_kg'], .03, .4)
    initial, target = pose(spec['initial_book_world']), pose(spec['target_book_world'])
    # Validate all target corners before creating any task geometry. Identity
    # actual tool/book is a geometric validation input, never an actual grasp.
    book_insertion(actual_tcp_world=initial, actual_book_world=initial, target_book_world=target,
        shelf_world=spec['shelf_world'], book_size_m=size, cavity_bounds_shelf_m=bounds,
        actual_grasp_valid=True)
    if np.any(size < .008) or np.any(size > .35) or np.any(bounds[:, 1]-bounds[:, 0] > .6):
        raise ValueError('BOUNDED_BOOK_AND_SHELF_REQUIRED')
    book = body(world, 'skill_book', initial)
    ET.SubElement(book, 'freejoint', name='skill_book_free')
    box(book, 'skill_book_geom', [0, 0, 0], size/2, mass=mass, rgba='.82 .08 .05 1')
    shelf = body(world, 'skill_shelf_frame', spec['shelf_world'])
    extent, mid = np.diff(bounds, axis=1)[:, 0], bounds.mean(axis=1)
    for side, y in (('left', bounds[1, 0]-thickness/2), ('right', bounds[1, 1]+thickness/2)):
        box(shelf, 'skill_shelf_'+side, [mid[0], y, mid[2]],
            [extent[0]/2, thickness/2, extent[2]/2], rgba='.45 .5 .58 1')
    box(shelf, 'skill_shelf_floor', [mid[0], mid[1], bounds[2, 0]-thickness/2],
        [extent[0]/2, extent[1]/2+thickness, thickness/2], rgba='.45 .5 .58 1')
    box(shelf, 'skill_shelf_back', [bounds[0, 1]+thickness/2, mid[1], mid[2]],
        [thickness/2, extent[1]/2+thickness, extent[2]/2], rgba='.45 .5 .58 1')
    return dict(task='BOOK', object_body='skill_book', grasp_geom='skill_book_geom',
        free_joint='skill_book_free', fixture_frame='skill_shelf_frame',
        support_geom='skill_shelf_floor', obstacle_geoms=['skill_shelf_left', 'skill_shelf_right',
        'skill_shelf_floor', 'skill_shelf_back'], new_actuators=0, new_equalities=0, new_mocap=0)


def overlay_source_xml(source_xml, card):
    """Return a cold source scene and task bindings; never instantiate an engine."""
    root = ET.fromstring(source_xml)
    if card.get('task') not in ('DOOR', 'BOOK') or root.get('model') not in (
            'PI0291_FIXED_BASE_ARX5_TASK_SIM', 'PI0294_FIXED_BASE_ARX5_MULTITASK'):
        raise ValueError('ONLY_EXISTING_FIXED_BASE_SOURCE_AND_NEW_TASK_CARD')
    compiler, world = root.find('compiler'), root.find('worldbody')
    if compiler is None or compiler.get('angle', 'degree') != 'radian' or world is None:
        raise ValueError('SOURCE_RADIAN_SCENE_REQUIRED_NO_ROBOT_UNIT_CHANGE')
    cube = world.find("body[@name='cube']")
    if cube is None or cube.find("freejoint[@name='cube_free']") is None:
        raise ValueError('EXACT_SOURCE_CUBE_REPLACEMENT_REQUIRED')
    protected = {tag: ET.tostring(root.find(tag)) if root.find(tag) is not None else None
                 for tag in ('actuator', 'contact', 'equality', 'option')}
    robot = world.find("body[@name='base_link']")
    if robot is None or robot.find('freejoint') is not None:
        raise ValueError('SOURCE_FIXED_BASE_ROBOT_REQUIRED')
    before_robot = ET.tostring(robot)
    world.remove(cube)
    for marker in list(world.findall('geom')):
        if marker.get('name', '').startswith('goal_marker_'):
            if marker.get('contype') != '0' or marker.get('conaffinity') != '0':
                raise ValueError('GOAL_MARKER_MUST_BE_NONCONTACT')
            world.remove(marker)
    binding = add_door(world, card['door']) if card['task'] == 'DOOR' else add_book(world, card['book'])
    if before_robot != ET.tostring(robot) or any(
            value != (ET.tostring(root.find(tag)) if root.find(tag) is not None else None)
            for tag, value in protected.items()):
        raise ValueError('SOURCE_ROBOT_EXECUTOR_OR_PHYSICS_CHANGED')
    root.set('model', VERSION+'_'+card['task'])
    return ET.tostring(root, encoding='unicode'), dict(binding, version=VERSION,
        qualification='UNEXECUTED_SOURCE_CONTACT_SCENE', robot_XML_preserved=True)


def task_only_xml(card):
    """Self-contained task-object fixture for zero-integration compiler checks.

    It has no robot, cameras or actuator and cannot establish task feasibility.
    """
    root = ET.Element('mujoco', model=VERSION+'_OBJECT_FIXTURE')
    ET.SubElement(root, 'compiler', angle='radian')
    ET.SubElement(root, 'option', timestep='.002', gravity='0 0 -9.81', integrator='implicitfast')
    world = ET.SubElement(root, 'worldbody')
    if card['task'] == 'DOOR': add_door(world, card['door'])
    elif card['task'] == 'BOOK': add_book(world, card['book'])
    else: raise ValueError('EXPLICIT_SOURCE_TASK_REQUIRED')
    return ET.tostring(root, encoding='unicode')
