"""Prepare fixed SLA DEV inputs with original FK/contact checks; no physics."""
import json
from pathlib import Path
import sys
import time
import numpy as np
from go2arx5_vla.world_model.storage import digest, write_json
from scripts.pi0292.durable import write as write_progress
from go2arx5_vla.world_model.stable_task_scene_v1 import pinned
from go2arx5_vla.world_model.task_learning_scene_v1 import compile_and_check, geometry, scene_id
from go2arx5_vla.world_model.sla_population_v1 import DEV_OFFSETS, geometry_sha, initial_geometry


def physical_initial_geometry(model):
    """Only initial inputs, never a TEST outcome, action, image or trajectory."""
    cube = model.body('task_cube').id
    joint = int(model.body_jntadr[cube])
    address = int(model.jnt_qposadr[joint])
    table = model.geom('task_table').id
    try:
        home = model.key('home').id
    except KeyError:
        home = 0
    pose = model.key_qpos[home] if model.nkey else model.qpos0
    size = model.geom_size[model.geom('task_cube_geom').id]
    return dict(table_center_xyz_m=model.geom_pos[table].tolist(),
                table_half_size_m=model.geom_size[table].tolist(),
                object_initial_xyz_m=pose[address:address+3].tolist(),
                cube_edge_m=float(2*size[0]), cube_mass_kg=float(model.body_mass[cube]))


def same_geometry(a, b):
    # Conservative comparison fixed before any candidate/dev output: inputs
    # differing only by less than one picometre are treated as the same layout.
    return all(np.allclose(a[k], b[k], rtol=0, atol=1e-12) for k in a)


def run(spec):
    import mujoco
    if mujoco.__version__ != '3.2.7':
        raise ValueError('ORIGINAL_MUJOCO_VERSION_REQUIRED')
    R = Path(spec['registered_root'])
    declaration = pinned(spec['preregistration'])
    if declaration['schema'] != 'SLA_FIXED_DEV16_INPUTS_V1' or len(declaration['population']) != 16:
        raise ValueError('FIXED_INPUT_POPULATION_REQUIRED')
    desired = {layout: initial_geometry(geometry(layout)) for layout in DEV_OFFSETS}
    if any(row['initial_geometry_sha256'] != geometry_sha(geometry(row['layout'])) for row in declaration['population']):
        raise ValueError('CANDIDATE_INDEPENDENT_DEV_GEOMETRY_CHANGED')
    catalogs, originals, seen = [], [], set()
    for pin in spec['old_scene_catalogs']:
        catalog = pinned(pin)
        catalogs.append(dict(pin=pin, rows=len(catalog['scenes'])))
        for row in catalog['scenes']:
            p = Path(row['model_path'])
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            if digest(p) != row['model_sha256']:
                raise ValueError('OLD_INITIAL_SCENE_IDENTITY_CHANGED')
            model = mujoco.MjModel.from_binary_path(str(p))
            physical = physical_initial_geometry(model)
            originals.append(dict(scene_id=row['scene_id'], parent=row['parent'], split=row['split'],
                                  model_sha256=row['model_sha256'], physical_initial_geometry=physical))
            if any(same_geometry(physical, g) for g in desired.values()):
                raise ValueError('SLA_DEV_REUSES_OLD_ACTUAL_PHYSICAL_GEOMETRY')
    if not any(r['split'] == 'TEST' for r in originals):
        raise ValueError('OLD_TEST_INITIAL_IDENTITY_CATALOG_REQUIRED_NO_TEST_OUTCOME_READ')
    scenes, identities = {}, []
    base = spec['compiler']
    for layout in DEV_OFFSETS:
        if time.time() >= spec['deadline_epoch']:
            raise TimeoutError('FINITE_SLA_STATIC_PREPARATION')
        target = R/layout
        result = compile_and_check(dict(base, source_sha=spec['source_sha'], layout=layout,
                                         geometry=geometry(layout)), target)
        model = mujoco.MjModel.from_binary_path(str(target/'SCENE.private.mjb'))
        actual = physical_initial_geometry(model)
        if not same_geometry(actual, desired[layout]):
            raise ValueError('SLA_NEW_COMPILED_INITIAL_GEOMETRY_NOT_AS_REGISTERED')
        scenes[layout] = dict(path=str(target/'STATIC_RESULT.private.json'), sha256=digest(target/'STATIC_RESULT.private.json'))
        identities.append(dict(layout=layout, initial_geometry_sha256=geometry_sha(result['geometry']),
                               model_sha256=result['compiled']['scene_sha256'], actual=actual))
        write_progress(R/'PROGRESS.private.json', dict(completed_static_layouts=len(scenes), requested=8,
            model_calls=0, optimizer_updates=0, physics_integrations=0))
    proof = dict(schema='SLA_DEV_ACTUAL_SCENE_DISJOINTNESS_V1', passes=True,
        preregistration=spec['preregistration'], old_catalogs=catalogs,
        old_input_scenes=originals, old_TEST_initial_inputs=sum(r['split']=='TEST' for r in originals),
        new_initial_inputs=identities, geometry_comparison_atol=1e-12,
        original_physical_scenes_unchanged=True, old_TEST_scores_read=False,
        old_TEST_images_actions_trajectories_read=False, physics_integrations=0, model_calls=0,
        optimizer_updates=0, epoch=time.time())
    write_json(R/'SCENE_DISJOINTNESS.private.json', proof)
    allowlist = pinned(spec['template_allowlist'])
    allowlist = dict(allowlist, scenes=[dict(scene_id=scene_id(layout), parent=scene_id(layout), split='SLA_DEV',
        model_path=str(R/layout/'SCENE.private.mjb'), model_sha256=digest(R/layout/'SCENE.private.mjb'),
        static_check='PASSED_NO_INTEGRATION') for layout in DEV_OFFSETS])
    write_json(R/'SCENE_ALLOWLIST.private.json', allowlist)
    return dict(schema='SLA_DEV_STATIC_INPUT_PREPARATION_ACTUAL_V1', passes=True,
        scenes=scenes, scene_disjointness=dict(path=str(R/'SCENE_DISJOINTNESS.private.json'),
                                             sha256=digest(R/'SCENE_DISJOINTNESS.private.json')),
        requested_layouts=8, completed_layouts=8, saved_startup_checks_per_layout=5703,
        physics_integrations=0, model_calls=0, optimizer_updates=0, old_TEST_scores_read=False)


if __name__ == '__main__':
    spec = json.loads(Path(sys.argv[1]).read_bytes())
    value = run(spec)
    write_json(Path(spec['registered_root'])/'RESULT.private.json', value)
