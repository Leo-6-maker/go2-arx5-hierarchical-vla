"""Read-only five-object/ten-pose preparation for the fixed V5 target scope.

This module issues no native execution or data admission. Object variants are
explicit; the original robot/safety contract is applied after normalizing only
the checked object fields in a separate model copy. Old scene builders and
already running capsules remain unchanged.
"""
import copy
import hashlib
import json
import math
import struct
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .pilot1_curriculum import robot_contract, scene_xml
from .storage import digest
from .task_learning_scene_v1 import BASE_SCENE, geometry

VERSION = 'SLA_TARGET_FIVE_OBJECT_TEN_POSE_STATIC_V5'
OBJECTS = {
    'RED_CUBE40': dict(size_m=[.04, .04, .04], rgba=[.82, .04, .04, 1.]),
    'GREEN_CUBE35': dict(size_m=[.035, .035, .035], rgba=[.05, .65, .10, 1.]),
    'BLUE_BOX50_30_40': dict(size_m=[.05, .03, .04], rgba=[.04, .15, .85, 1.]),
    'YELLOW_BOX35_45_40': dict(size_m=[.035, .045, .04], rgba=[.90, .75, .04, 1.]),
    'PURPLE_BOX45_35_50': dict(size_m=[.045, .035, .05], rgba=[.60, .08, .70, 1.]),
}
# Fixed before any new student output. Local offsets remain within the old
# reachable TRAIN envelope; yaw and object geometry are new declared factors.
POSES = {
    f'POSE{i + 1:02d}': dict(offset_xy_m=[x, y], yaw_rad=math.radians(yaw))
    for i, (x, y, yaw) in enumerate([
        (-.0035, -.0035, -5), (-.0035, .0035, 5),
        (-.00175, -.0035, 0), (-.00175, .0035, -5),
        (.00025, -.0035, 5), (.00025, .0035, 0),
        (.002, -.0035, -5), (.002, .0035, 5),
        (.0035, -.0035, 0), (.0035, .0035, -5),
    ])
}


def declaration(object_id, pose_id):
    obj = OBJECTS[object_id]
    pose = POSES[pose_id]
    size = np.asarray(obj['size_m'], float)
    xyz = np.array([.26, -.014, .6 + size[2] / 2])
    xyz[:2] += pose['offset_xy_m']
    yaw = pose['yaw_rad']
    return dict(version=VERSION, object_id=object_id, pose_id=pose_id,
        object_type='BOX', size_m=size.tolist(), rgba=obj['rgba'],
        mass_kg=float(.05 * np.prod(size) / .04**3),
        initial_xyz_m=xyz.tolist(), initial_quaternion_wxyz=[math.cos(yaw/2), 0., 0., math.sin(yaw/2)],
        source_friction_and_native_safety_unchanged=True,
        tasks=['PICK_HOLD', 'PICK_PLACE'], formal_D100_admitted=False)


def catalogue():
    return dict(schema=VERSION, objects=list(OBJECTS), poses=list(POSES),
        cases=[declaration(o, p) for o in OBJECTS for p in POSES],
        D100_required_full_demonstrations=100, D300_required_full_demonstrations=300,
        production_admitted=False, student_evaluation=False,
        scope='FIVE_DECLARED_BOX_INSTANCES_NOT_UNSEEN_OBJECT_CATEGORY_GENERALIZATION')


def _number(values):
    return ' '.join(format(float(v), '.17g') for v in values)


def variant_xml(preparation, original, declared):
    expected = declaration(declared['object_id'], declared['pose_id'])
    if declared != expected:
        raise ValueError('EXACT_FIVE_OBJECT_POSE_DECLARATION')
    xml = ET.fromstring(scene_xml(dict(geometry('TRAIN01'), asset_root=preparation['asset_root']), original))
    body = xml.find("./worldbody/body[@name='task_cube']")
    geom = body.find("geom[@name='task_cube_geom']")
    body.set('pos', _number(declared['initial_xyz_m']))
    body.set('quat', _number(declared['initial_quaternion_wxyz']))
    geom.set('size', _number(np.asarray(declared['size_m']) / 2))
    geom.set('mass', format(declared['mass_kg'], '.17g'))
    geom.set('rgba', _number(declared['rgba']))
    import mujoco
    q = int(original.jnt_qposadr[original.joint('task_cube_free').id])
    for key in xml.findall('./keyframe/key'):
        index = mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_KEY, key.get('name'))
        value = original.key_qpos[index].copy()
        value[q:q+3] = declared['initial_xyz_m']
        value[q+3:q+7] = declared['initial_quaternion_wxyz']
        key.set('qpos', _number(value))
    return ET.tostring(xml, encoding='unicode')


def check_robot_and_declared_object(original, candidate, declared):
    """Check the actual variant, then reuse the unchanged robot contract."""
    from numpy.testing import assert_allclose, assert_array_equal
    if declared != declaration(declared['object_id'], declared['pose_id']):
        raise ValueError('EXACT_OBJECT_CONTRACT_DECLARATION')
    b = candidate.body('task_cube').id
    g = candidate.geom('task_cube_geom').id
    q = int(candidate.jnt_qposadr[candidate.joint('task_cube_free').id])
    size = np.asarray(declared['size_m'])
    mass = declared['mass_kg']
    support = geometry('TRAIN01')
    center = np.asarray(support['table_center_xyz_m'])
    half = np.asarray(support['table_half_size_m'])
    assert_allclose(candidate.geom('task_table').pos, center, atol=1e-15, rtol=0)
    assert_allclose(candidate.geom('task_table').size, half, atol=1e-15, rtol=0)
    yaw = POSES[declared['pose_id']]['yaw_rad']
    projected_half = np.abs(np.array([[math.cos(yaw), -math.sin(yaw)],
        [math.sin(yaw), math.cos(yaw)]])) @ (size[:2] / 2)
    xyz = np.asarray(declared['initial_xyz_m'])
    if abs(xyz[2] - size[2]/2 - center[2] - half[2]) > 1e-12 or np.any(np.abs(xyz[:2]-center[:2]) + projected_half > half[:2]):
        raise ValueError('DECLARED_OBJECT_MUST_HAVE_REAL_INITIAL_TABLE_SUPPORT')
    for i, (x, y, dx, dy) in enumerate(((center[0]-half[0], center[1], .0005, half[1]),
        (center[0]+half[0], center[1], .0005, half[1]), (center[0], center[1]-half[1], half[0], .0005),
        (center[0], center[1]+half[1], half[0], .0005))):
        name = 'task_goal_' + str(i)
        if name in [candidate.geom(k).name for k in range(candidate.ngeom)]:
            assert_allclose(candidate.geom(name).pos, [x, y, center[2]+half[2]+.0002], atol=1e-15, rtol=0)
            assert_allclose(candidate.geom(name).size, [dx, dy, .0002], atol=1e-15, rtol=0)
    expected_inertia = mass * np.array([size[1]**2 + size[2]**2,
        size[0]**2 + size[2]**2, size[0]**2 + size[1]**2]) / 12
    assert_allclose(candidate.geom_size[g], size / 2, atol=1e-15, rtol=0)
    assert_allclose(candidate.body_mass[b], mass, atol=1e-15, rtol=0)
    assert_allclose(candidate.body_inertia[b], expected_inertia, atol=1e-15, rtol=1e-14)
    assert_allclose(candidate.geom_rgba[g], declared['rgba'], atol=6e-8, rtol=0)
    assert_allclose(candidate.body_pos[b], declared['initial_xyz_m'], atol=1e-15, rtol=0)
    assert_allclose(candidate.body_quat[b], declared['initial_quaternion_wxyz'], atol=1e-15, rtol=0)
    for field in ('qpos0', 'qpos_spring'):
        assert_allclose(getattr(candidate, field)[q:q+3], declared['initial_xyz_m'], atol=1e-15, rtol=0)
        assert_allclose(getattr(candidate, field)[q+3:q+7], declared['initial_quaternion_wxyz'], atol=1e-15, rtol=0)
    for row in candidate.key_qpos:
        assert_allclose(row[q:q+3], declared['initial_xyz_m'], atol=1e-15, rtol=0)
        assert_allclose(row[q+3:q+7], declared['initial_quaternion_wxyz'], atol=1e-15, rtol=0)
    # The native collision representation still supports the unchanged box type.
    assert_array_equal(candidate.geom_type, original.geom_type)
    keep = np.arange(candidate.ngeom) != g
    assert_array_equal(candidate.geom_rgba[keep], original.geom_rgba[keep])
    normalized = copy.copy(candidate)
    for field in ('body_quat', 'body_mass', 'body_inertia'):
        getattr(normalized, field)[b] = getattr(original, field)[b]
    normalized.geom_size[g] = original.geom_size[g]
    for field in ('qpos0', 'qpos_spring'):
        getattr(normalized, field)[q+3:q+7] = getattr(original, field)[q+3:q+7]
    normalized.key_qpos[:, q+3:q+7] = original.key_qpos[:, q+3:q+7]
    frozen = robot_contract(original, normalized)
    return dict(status='ROBOT_AND_NATIVE_SAFETY_EXACT_WITH_EXPLICIT_OBJECT_VARIATION',
        normalized_original_robot_contract=frozen,
        object_declaration_sha256=hashlib.sha256(json.dumps(declared, sort_keys=True).encode()).hexdigest(),
        normalization_only_in_separate_read_only_copy=True,
        object_mass_and_inertia_declared=True, original_model_or_native_runtime_changed=False)


def compile_and_check(preparation, output, object_id, pose_id):
    """Fresh scene plus original 5703 saved startup states; no integration."""
    import mujoco
    from scripts.pi0294.task_state import ActualState, parse_state
    from scripts.pi0294.mp_client import contact_permitted
    from .native_replay import forbid_integration
    output = Path(output)
    if mujoco.__version__ != '3.2.7' or output.exists() or digest(preparation['original_model']) != BASE_SCENE:
        raise ValueError('NEW_OUTPUT_AND_PINNED_ORIGINAL_MODEL')
    for name, pin in preparation['asset_input_pins'].items():
        if digest(Path(preparation['asset_root']) / name) != pin:
            raise ValueError('FROZEN_NATIVE_XML_ASSET')
    if digest(preparation['saved_startup_journal']) != preparation['saved_startup_journal_sha256']:
        raise ValueError('ORIGINAL_CLOSED_SAVED_STARTUP_INPUT')
    declared = declaration(object_id, pose_id)
    original = mujoco.MjModel.from_binary_path(preparation['original_model'])
    text = variant_xml(preparation, original, declared)
    model = mujoco.MjModel.from_xml_string(text)
    contract = check_robot_and_declared_object(original, model, declared)
    output.mkdir()
    xml = output / 'SCENE.private.xml'; xml.write_text(text)
    binary = output / 'SCENE.private.mjb'; mujoco.mj_saveModel(model, str(binary), None)
    model = mujoco.MjModel.from_binary_path(str(binary))
    check_robot_and_declared_object(original, model, declared)
    q = int(model.jnt_qposadr[model.joint('task_cube_free').id])
    reader = ActualState(model); count = 0; first_failure = None
    with forbid_integration(mujoco) as integrations, Path(preparation['saved_startup_journal']).open('rb') as stream:
        while count < 5703:
            header = stream.read(4)
            if len(header) != 4: raise ValueError('COMPLETE_STARTUP_HEADER')
            length, = struct.unpack('<I', header)
            if not 40 <= length <= 2 * 1024**2: raise ValueError('BOUNDED_ORIGINAL_PACKET')
            raw = stream.read(length)
            if len(raw) != length: raise ValueError('COMPLETE_STARTUP_PACKET')
            packet = parse_state(raw); count += 1
            if packet['step'] != count: raise ValueError('CONTIGUOUS_ORIGINAL_STARTUP')
            hypothetical = dict(packet, qpos=packet['qpos'].copy())
            hypothetical['qpos'][q:q+3] = declared['initial_xyz_m']
            hypothetical['qpos'][q+3:q+7] = declared['initial_quaternion_wxyz']
            reader.update(hypothetical)
            for contact in reader.data.contact:
                names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) for g in (contact.geom1, contact.geom2)]
                if not contact_permitted(names, contact.dist, 'approach') and first_failure is None:
                    first_failure = dict(step=count, names=names, distance_m=float(contact.dist))
        if integrations: raise ValueError('STATIC_CHECK_MUST_NOT_INTEGRATE')
    result = dict(schema=VERSION, passes=first_failure is None, declaration=declared,
        source_sha=preparation['source_sha'], builder_sha256=digest(__file__), robot_contract=contract,
        saved_startup_checks=count, first_failure=first_failure,
        original_journal_sha256=preparation['saved_startup_journal_sha256'],
        scene_sha256=digest(binary), scene_xml_sha256=digest(xml), original_model_sha256=BASE_SCENE,
        actual_integrations=0, policy_model_loads=0, optimizer_updates=0,
        formal_demonstrations=0, native_execution_authorized=False, physics_dt_s=.002,
        static_hypothesis_not_dynamic_qualification=True)
    (output / 'STATIC_RESULT.private.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
