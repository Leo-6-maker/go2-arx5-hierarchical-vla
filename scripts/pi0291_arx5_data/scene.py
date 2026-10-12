"""Independent fixed-base extension. The delivered WBC asset is never edited."""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

TCP_OFFSET = np.array([.1424, 0., .0001057])
DOWNWARD_ROTATION = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
ARM_NAMES = tuple(f'x5_joint{i}' for i in range(1, 7))
FINGER_NAMES = ('left_slide', 'right_slide')
PAD_INNER_GAP_AT_ZERO = .00231 - .001262


def _quat_product(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return np.array([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                     w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v])


def xml_scene(card, asset_root, parameters):
    root_path = Path(asset_root).resolve()
    source = root_path / 'go2_x5.xml'
    root = ET.parse(source).getroot()
    root.set('model', 'PI0291_FIXED_BASE_ARX5_TASK_SIM')
    root.find('compiler').set('meshdir', str(root_path / 'assets'))
    option = root.find('option')
    option.set('timestep', str(parameters['dt']))
    option.set('integrator', 'implicitfast')
    option.set('gravity', '0 0 -9.81')
    option.set('iterations', '80')
    for tag in ('sensor', 'keyframe', 'actuator'):
        old = root.find(tag)
        if old is not None:
            root.remove(old)
    world = root.find('worldbody')
    base = world.find("body[@name='base_link']")
    base.remove(base.find('freejoint'))
    # Fixed leg geometry keeps the delivered mounting and collision provenance,
    # without presenting a suspended dog as an independently controlled WBC.
    for body in base.iter('body'):
        for joint in list(body.findall('joint')):
            name = joint.get('name', '')
            if name in ARM_NAMES:
                joint.set('damping', str(parameters.get('arm_joint_damping', 3.)))
                continue
            if name.endswith('_hip_joint'):
                value, axis = (0.1 if name.startswith(('FL', 'RL')) else -0.1), np.array([1., 0., 0.])
            elif name.endswith('_thigh_joint'):
                value, axis = .8, np.array([0., 1., 0.])
            elif name.endswith('_calf_joint'):
                value, axis = -1.5, np.array([0., 1., 0.])
            else:
                raise ValueError(f'Unexpected native joint {name}')
            original = np.fromstring(body.get('quat', '1 0 0 0'), sep=' ')
            original /= np.linalg.norm(original)
            rotation = np.r_[np.cos(value/2), axis*np.sin(value/2)]
            body.set('quat', ' '.join(map(str, _quat_product(original, rotation))))
            body.remove(joint)
    link6 = base.find(".//body[@name='x5_link6']")
    for side, body_name, axis, frame_y in [('left', 'x5_link7', '0 1 0', -.0249),
                                          ('right', 'x5_link8', '0 -1 0', .0249)]:
        body = link6.find(f"body[@name='{body_name}']")
        ET.SubElement(body, 'joint', name=f'{side}_slide', type='slide', axis=axis,
                      range='0 .044', limited='true', damping='3', armature='.002', frictionloss='0')
        outward = .003 if side == 'left' else -.003
        ET.SubElement(body, 'geom', name=f'{side}_pad', type='box',
                      pos=f'.05843 {frame_y+outward} .006376', size='.02 .003 .017',
                      friction=f"{parameters['pad_friction']} .02 .002", condim='4',
                      margin='0', solref='.008 1', rgba='.10 .16 .18 1', group='3')
        ET.SubElement(body, 'site', name=f'{side}_pad_frame', pos=f'.05843 {frame_y} .006376', size='.002')
    camera_position = np.array([.1124, .10, .055])
    look_at = TCP_OFFSET + np.array([.035, 0., 0.])
    z = camera_position - look_at
    z /= np.linalg.norm(z)
    x = np.cross(np.array([1., 0., 0.]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    ET.SubElement(link6, 'camera', name='wrist_rgb', pos=' '.join(map(str, camera_position)),
                  xyaxes=' '.join(map(str, np.r_[x, y])), fovy='75')
    ET.SubElement(world, 'light', pos='0 -.4 1.8', dir='0 .2 -1', diffuse='.8 .8 .8')
    ET.SubElement(world, 'light', pos='.8 .5 1.5', dir='-.3 -.2 -1', diffuse='.6 .6 .6')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='2 2 .1', rgba='.72 .74 .76 1', friction='.8 .02 .002')
    top = float(parameters['table_top'])
    ET.SubElement(world, 'geom', name='table', type='box', pos=f'.50 0 {top-.025}',
                  size='.17 .24 .025', rgba='.69 .70 .66 1', friction='.8 .02 .002')
    gx, gy = card['goal_xy']
    for index, (x, y, sx, sy) in enumerate([(gx-.08, gy, .002, .08), (gx+.08, gy, .002, .08),
                                           (gx, gy-.08, .08, .002), (gx, gy+.08, .08, .002)]):
        ET.SubElement(world, 'geom', name=f'goal_marker_{index}', type='box',
                      pos=f'{x} {y} {top+.0002}', size=f'{sx} {sy} .0002',
                      rgba='.08 .35 .80 1', contype='0', conaffinity='0')
    sx, sy = card['source_xy']
    if card.get('mode') == 'empty_grasp':
        sx += .18
    yaw = float(card['cube_yaw'])
    cube = ET.SubElement(world, 'body', name='cube', pos=f'{sx} {sy} {top+.02}',
                         quat=f'{np.cos(yaw/2)} 0 0 {np.sin(yaw/2)}')
    ET.SubElement(cube, 'freejoint', name='cube_free')
    ET.SubElement(cube, 'geom', name='cube_geom', type='box', size='.02 .02 .02',
                  mass=str(parameters['cube_mass']), friction='.8 .02 .002', condim='4',
                  rgba='.82 .04 .04 1', margin='0', solref='.008 1')
    camera_pos = np.array([1.0, -.90, 1.10])
    look_at = np.array([.43, 0., top+.08])
    z = camera_pos - look_at
    z /= np.linalg.norm(z)
    x = np.cross(np.array([0., 0., 1.]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    ET.SubElement(world, 'camera', name='body_rgb', pos=' '.join(map(str, camera_pos)),
                  xyaxes=' '.join(map(str, np.r_[x, y])), fovy='55')
    actuators = ET.SubElement(root, 'actuator')
    for i, name in enumerate(ARM_NAMES):
        effort = 27 if i < 3 else 7
        ET.SubElement(actuators, 'motor', name=name, joint=name,
                      ctrllimited='true', ctrlrange=f'{-effort} {effort}')
    for name in FINGER_NAMES:
        force = parameters['finger_force_max_N']
        ET.SubElement(actuators, 'motor', name=name, joint=name,
                      ctrllimited='true', ctrlrange=f'{-force} {force}')
    return ET.tostring(root, encoding='unicode')
