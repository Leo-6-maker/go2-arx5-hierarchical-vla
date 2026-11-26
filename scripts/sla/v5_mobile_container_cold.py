"""Compile and screen two fixed M_FREE container layouts without physics steps."""

import argparse
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as XML

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from v5_scene_factory import BASE


ROSTER = (('EF5_Q_MOBILE_01', 'EF5_Q_CONTAINER_01', 'red', 'left'),
          ('EF5_Q_MOBILE_02', 'EF5_Q_CONTAINER_02', 'blue', 'right'))
VARIANTS = {'initial': (.45, 0., ((0., 0.), (0., 0.))),
            'wider': (.60, np.pi / 2., ((.0235, 0.), (0., -.019))),
            'project': (1., 0., ((.0235, 0.), (0., -.019)))}
PROJECT_BOWL = 'd3ca3e1c9422c8ad2b434ca149bb1e0a1ed572384443d7e00539e1da64c4182e'


def add_bowl(xml, fixture, model_path, center, scale, yaw):
    root = XML.fromstring(xml)
    asset = root.find('asset')
    model = XML.parse(model_path).getroot()
    for item in model.findall('./asset/*'):
        element = deepcopy(item)
        if item.get('file'):
            element.set('file', str(fixture / item.get('file')))
        if element.tag == 'mesh':
            element.set('scale', ' '.join([str(scale)] * 3))
        asset.append(element)
    world = root.find('worldbody')
    bowl = XML.SubElement(world, 'body', name='ef5_static_bowl', pos=' '.join(map(str, center)), euler=f'0 0 {yaw}')
    for index, item in enumerate(model.findall('.//worldbody//geom')):
        attributes = dict(item.attrib)
        kind = attributes.pop('class')
        if kind not in ('visual', 'collision'):
            continue
        require('mesh' in attributes, 'Expected the original mesh parts')
        attributes.update(name=f'ef5_bowl_{kind}_{index}', type='mesh')
        if kind == 'visual':
            attributes.update(contype='0', conaffinity='0', group='1', mass='0')
        else:
            attributes.update(group='0', rgba='0.5 0.5 0 0.5', density='1000')
        XML.SubElement(bowl, 'geom', attributes)
    return XML.tostring(root, encoding='unicode')


def protect(base, scene):
    for name in ('nq', 'nv', 'nu', 'nkey', 'neq', 'nmocap', 'nsensor', 'ncam'):
        require(getattr(base, name) == getattr(scene, name), f'Fixture changed {name}')
    for names, count in (
        (('body_mass', 'body_inertia', 'body_parentid', 'body_pos'), base.nbody),
        (('geom_pos', 'geom_size', 'geom_type', 'geom_bodyid', 'geom_quat', 'geom_friction',
          'geom_contype', 'geom_conaffinity', 'geom_solref', 'geom_solimp'), base.ngeom),
        (('actuator_gainprm', 'actuator_biasprm', 'actuator_gear', 'actuator_ctrlrange'), base.nu),
        (('jnt_range', 'jnt_type', 'jnt_qposadr', 'jnt_axis'), base.njnt),
        (('cam_pos', 'cam_quat', 'cam_fovy'), base.ncam),
        (('mesh_vert',), base.nmeshvert),
    ):
        for field in names:
            np.testing.assert_array_equal(getattr(scene, field)[:count], getattr(base, field))
    require(scene.opt.timestep == base.opt.timestep == .002, 'Original clock changed')


def build(template_source, s1, fixture, output, *, variant):
    import mujoco as mj
    from go2arx5_vla.world_model.multitask_scene_v6 import overlay_xml, robot_contract
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from scripts.pi0292.durable import write
    from scripts.wm02.multitask_scene_audit_v6 import asset_binding

    template = read(pin(template_source, BASE))
    templates = {family: read(pin(value['path'], value['sha256'])) for family, value in template['templates'].items()}
    population = {c['id']: c for c in read(s1)['clusters']}
    scale, yaw, offsets = VARIANTS[variant]
    project_bowl = variant == 'project'
    if project_bowl:
        import sea_near_source_scene as known

        pin(known.__file__, PROJECT_BOWL)
        source_identity = PROJECT_BOWL
        extent = np.array([[-.069, -.069, 0.], [.069, .069, .04]])
    else:
        manifest = read(fixture / 'SOURCE.json')
        require(manifest['license'] == 'CC-BY-4.0' and manifest['asset'] == 'fruit_bowl/FruitBowl001',
                'Unverified external fixture')
        files = {m['file']: pin(fixture / m['file'], m['sha256']) for m in manifest['source_members']}
        extent = np.asarray(read(fixture / 'COLD_GEOMETRY_ACTUAL.json')['bounds']) * scale
        source_identity = sha(fixture / 'SOURCE.json')
    horizontal_half_size = np.max(np.abs(extent[:, :2]), axis=0)
    if yaw:
        horizontal_half_size = horizontal_half_size[::-1]
    output.mkdir(mode=0o700)
    results = []
    with forbid_integration(mj) as effects:
        for (source, identity, color, side), offset in zip(ROSTER, offsets, strict=True):
            cluster = population[source]
            spec = dict(templates[cluster['family']])
            spec['asset_root'], _ = asset_binding(spec)
            old = mj.MjModel.from_binary_path(str(pin(spec['model_path'], spec['frozen_files'][spec['model_path']])))
            xml, geometry = overlay_xml(spec, old, cluster['card'])
            baseline = mj.MjModel.from_xml_string(xml)
            frozen = robot_contract(old, baseline)
            table = baseline.geom('task_table')
            top = geometry['table_top_world_z']
            xy = np.asarray(geometry['zones'][side]['center_world_XY']) + offset
            center = [*xy, top - extent[0, 2]]
            require(np.all(np.abs(xy - table.pos[:2]) + horizontal_half_size < table.size[:2]), 'Fixture leaves the table')
            if project_bowl:
                scene_xml = XML.fromstring(xml)
                known.bowl(scene_xml.find('worldbody'), scene_xml.find('asset'), name='ef5_static_bowl', xy=xy,
                           top=top, bottom_radius=.04, rim_radius=.069, height=.04, color=known.COLORS['blue'])
                assembled = XML.tostring(scene_xml, encoding='unicode')
            else:
                assembled = add_bowl(xml, fixture, files['model.xml'], center, scale, yaw)
            model = mj.MjModel.from_xml_string(assembled)
            protect(baseline, model)
            state = mj.MjData(model)
            mj.mj_fwdPosition(model, state)
            col = [g for g in range(baseline.ngeom, model.ngeom) if model.geom_contype[g]]
            require(len(col) == (25 if project_bowl else 16), 'Original bowl collision pieces missing')
            target = model.geom(geometry['objects'][color]['geom']).id
            others = [model.geom(o['geom']).id for o in geometry['objects'].values()]
            original_contact = [g for g in range(baseline.ngeom) if baseline.geom_bodyid[g] and baseline.geom_contype[g]]
            closest, near_geom, near_piece = min((mj.mj_geomDistance(model, state, g, f, 2., None), g, f)
                                               for g in set(others + original_contact) for f in col)
            ray_id = np.array([-1], dtype=np.int32)
            ray = mj.mj_ray(model, state, np.array([*xy, top + .3]), np.array([0., 0., -1.]),
                            np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8), True, -1, ray_id)
            path = output / f'{identity}.mjb'
            mj.mj_saveModel(model, str(path), None)
            receipt = dict(id=identity, family=cluster['family'], target=color, fixture_world_xyz=center,
                           initial_clearance_mm=closest * 1e3, nearest_original_geom=model.geom(near_geom).name,
                           nearest_bowl_geom=model.geom(near_piece).name, center_ray_distance_m=ray,
                           center_ray_hit=None if ray_id[0] < 0 else model.geom(int(ray_id[0])).name,
                           scale=scale, yaw_rad=yaw, compiled_sha256=sha(path), baseline_robot_fingerprint=frozen['fingerprint'])
            if closest <= 0. or not .15 < ray < .3 or ray_id[0] not in col:
                results.append(dict(receipt, status='COLD_REJECT_INITIAL_OVERLAP_OR_NO_BOWL_FLOOR'))
                continue
            floor_z = top + .3 - ray
            address = model.joint(geometry['objects'][color]['joint']).qposadr[0]
            z_samples = np.linspace(floor_z + .023, center[2] + extent[1, 2] + .055, 32)
            distances = []
            for z in z_samples:
                state.qpos[address:address + 3] = [*xy, z]
                mj.mj_fwdPosition(model, state)
                distances.append(min(mj.mj_geomDistance(model, state, target, g, 2., None) for g in col))
            status = 'COLD_PASS' if min(distances) > 0. else 'COLD_REJECT_CUBE_INSERTION_COLLISION'
            results.append(dict(receipt, status=status, centered_insert_clearance_mm=min(distances) * 1e3,
                                fixture_floor_m=floor_z, checked_insert_heights=len(z_samples)))
    require(not effects, 'Cold fixture screening attempted native integration')
    write(output / 'ACTUAL.json', dict(rows=results, variant=variant, fixture_source_sha256=source_identity,
         source_population_sha256=sha(s1), new_native_integrations=0, native_dispatch_qualified=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('template_source', 's1', 'fixture', 'output'):
        parser.add_argument(field, type=Path)
    parser.add_argument('--variant', choices=VARIANTS, default='initial')
    build(**vars(parser.parse_args()))
