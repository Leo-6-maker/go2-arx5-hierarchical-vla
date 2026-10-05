"""Emit a draft actuated tool URDF; CPU XML/FK checks do not load a simulator."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def number(values):
    return ' '.join(format(float(v), '.12g') for v in values)


def build():
    root = ET.Element('robot', name='draft_independent_object_tool_not_go2_arx5')
    def link(name, mass, box=None, center=(0, 0, 0)):
        node = ET.SubElement(root, 'link', name=name)
        inertial = ET.SubElement(node, 'inertial')
        ET.SubElement(inertial, 'origin', xyz=number(center), rpy='0 0 0')
        ET.SubElement(inertial, 'mass', value=number([mass]))
        moments = [1e-5]*3 if box is None else [mass*(box[(i+1)%3]**2+box[(i+2)%3]**2)/12 for i in range(3)]
        ET.SubElement(inertial, 'inertia', ixx=number(moments[:1]), iyy=number(moments[1:2]), izz=number(moments[2:]), ixy='0', ixz='0', iyz='0')
        if box is not None:
            for kind in ('visual', 'collision'):
                shape = ET.SubElement(node, kind)
                ET.SubElement(shape, 'origin', xyz=number(center), rpy='0 0 0')
                ET.SubElement(ET.SubElement(shape, 'geometry'), 'box', size=number(box))
        return node
    link('anchor', .1)
    parent = 'anchor'
    specs = [('x', 'prismatic', (1, 0, 0), -1.5, 1.5, 20., .10),
             ('y', 'prismatic', (0, 1, 0), -1.5, 1.5, 20., .10),
             ('z', 'prismatic', (0, 0, 1), .0, 1.8, 20., .10),
             ('yaw', 'revolute', (0, 0, 1), -np.pi, np.pi, 2., .5),
             ('pitch', 'revolute', (0, 1, 0), -np.pi, np.pi, 2., .5),
             ('roll', 'revolute', (1, 0, 0), -np.pi, np.pi, 2., .5)]
    for name, kind, axis, lower, upper, effort, velocity in specs:
        child = 'palm_tcp' if name == 'roll' else 'carrier_'+name
        link(child, .3 if name == 'roll' else .05, (.06, .096, .02) if name == 'roll' else None, (0, 0, -.01) if name == 'roll' else (0, 0, 0))
        joint = ET.SubElement(root, 'joint', name='tool_'+name, type=kind)
        ET.SubElement(joint, 'parent', link=parent)
        ET.SubElement(joint, 'child', link=child)
        ET.SubElement(joint, 'origin', xyz='0 0 0', rpy='0 0 0')
        ET.SubElement(joint, 'axis', xyz=number(axis))
        ET.SubElement(joint, 'limit', lower=number([lower]), upper=number([upper]), effort=number([effort]), velocity=number([velocity]))
        parent = child
    for side, sign in (('left', 1.), ('right', -1.)):
        link('finger_'+side, .03, (.02, .008, .04))
        joint = ET.SubElement(root, 'joint', name='finger_'+side, type='prismatic')
        ET.SubElement(joint, 'parent', link='palm_tcp')
        ET.SubElement(joint, 'child', link='finger_'+side)
        ET.SubElement(joint, 'origin', xyz=number([0, sign*.004, .02]), rpy='0 0 0')
        ET.SubElement(joint, 'axis', xyz=number([0, sign, 0]))
        ET.SubElement(joint, 'limit', lower='0', upper='.04', effort='10', velocity='.02')
    ET.indent(root)
    return ET.tostring(root, encoding='utf-8', xml_declaration=True)+b'\n'


def check(payload):
    root = ET.fromstring(payload)
    links = {n.attrib['name']: n for n in root.findall('link')}
    joints = root.findall('joint')
    assert len(links) == 9 and len(joints) == 8
    children = [j.find('child').attrib['link'] for j in joints]
    assert len(set(children)) == 8 and set(links)-set(children) == {'anchor'}
    reachable = {'anchor'}
    for j in joints:
        assert j.find('parent').attrib['link'] in reachable
        reachable.add(j.find('child').attrib['link'])
        limit = j.find('limit').attrib
        assert all(np.isfinite(float(v)) for v in limit.values())
        assert float(limit['lower']) < float(limit['upper']) and min(float(limit[k]) for k in ('effort', 'velocity')) > 0
    assert reachable == set(links)
    for l in links.values():
        assert float(l.find('inertial/mass').attrib['value']) > 0
        assert all(float(l.find('inertial/inertia').attrib[k]) > 0 for k in ('ixx', 'iyy', 'izz'))
    assert {n for n, l in links.items() if l.find('collision') is not None} == {'palm_tcp', 'finger_left', 'finger_right'}
    finger = {j.attrib['name']: j for j in joints[-2:]}
    for width in (0., .03, .08):
        surfaces = []
        for name, sign in (('finger_left', 1.), ('finger_right', -1.)):
            j = finger[name]
            origin = np.fromstring(j.find('origin').attrib['xyz'], sep=' ')
            axis = np.fromstring(j.find('axis').attrib['xyz'], sep=' ')
            surfaces.append((origin+axis*width/2)[1]-sign*.004)
        assert abs(surfaces[0]-surfaces[1]-width) < 1e-12
    assert np.allclose(np.fromstring(links['palm_tcp'].find('collision/origin').attrib['xyz'], sep=' ') + [0, 0, .01], 0)
    # Static FK: the palm frame is the TCP; yaw/pitch/roll axes compose Rz Ry Rx.
    r = np.eye(3)
    for j, q in zip(joints[3:6], (.4, -.35, .25)):
        a = np.fromstring(j.find('axis').attrib['xyz'], sep=' ')
        skew = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        r = r @ (np.eye(3)+np.sin(q)*skew+(1-np.cos(q))*(skew@skew))
    assert np.allclose(r @ r.T, np.eye(3)) and abs(np.linalg.det(r)-1) < 1e-12
    return dict(status='PASS_CPU_DRAFT_TOOL_URDF_TOPOLOGY_TCP_OPENING_AND_FK',
                links=9, actuated_tool_dofs=8, robot_joint_supervision=False,
                opening_m=[0, .03, .08], finger_joint_opening_rule='q_left=q_right=full_width_m/2',
                palm_tcp_semantic='palm +z face center; fingers extend local +z',
                rotational_chain='Rz(yaw) Ry(pitch) Rx(roll)', base_load_requires='fix_base_link=True',
                force_limits='URDF per DOF; future controller must also enforce vector force/torque limits',
                native_asset_load='NOT_RUN', dynamics_contact_grasp='NOT_RUN', physics_steps=0, renders=0,
                teacher_calls=0, model_queries=0, launch_allowed=False, selected_checkpoint='NONE', deployment_selected_checkpoint='NONE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    payload = build()
    result = check(payload)
    result.update(asset_sha256=hashlib.sha256(payload).hexdigest(), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.asset.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.asset.write_bytes(payload)
    args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps(result, sort_keys=True))
