"""Compare three fixed F2 blue-left carry paths without integrating a world."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
from go2arx5_vla.world_model.native_replay import forbid_integration
from go2arx5_vla.world_model.storage import digest
from scripts.pi0292.durable import read
from scripts.sla.core_hopper_carry_audit import last_stop, stopped_pairs
from scripts.sla.core_multienv_place_sweep import ArmShadow, box_separation, pose_path, scene_delta, site_pose
from scripts.wm02.multitask_stage_cards_v6 import require_unstarted_cases

CASE = 'MEV1-TRAIN-F2-NEAR_NEG-00-PLACE-blue-left'
UNOPENED = 'MEV1-TRAIN-F2-NEAR_NEG-01-PLACE-blue-left'
PRODUCER_SPEC = '45d949a883c7c10ec31c0caef9a40bdd0a4c51225d5417dac68c7b05432e870b'
PRODUCER_SOURCE = 'eccd8e2e2089bb2505c50043f959959d07094d56'
RESULT = '04d946a480c3b5746ddc5aaf332cf93d8af949f7e397d72114e01192d9e1268c'
QUERIES = '76a8cdc0954d99de3a3722ce0d22bea189c68647f0bfb701392c311800894c3e'
ANCHORS = (23453, 23703, 23953, 27203, 27703, 28203)
PATHS = ('original_goal_straight', 'y_then_x', 'x_then_y')


def sweep(shadow, qpos, goal, route):
    model, data = shadow.model, shadow.data
    data.qpos[:] = qpos
    data.qvel[:] = 0
    mujoco.mj_fwdPosition(model, data)
    grip = qpos[shadow.fingers]
    start = site_pose(data, shadow.site)
    held = np.eye(4)
    held[:3, :3] = data.geom_xmat[shadow.target].reshape(3, 3)
    held[:3, 3] = data.geom_xpos[shadow.target]
    tool_from_object = np.linalg.solve(start, held)
    turn = start.copy()
    if route == 'y_then_x':
        turn[1, 3] = goal[1, 3]
    elif route == 'x_then_y':
        turn[0, 3] = goal[0, 3]
    legs = ((start, goal),) if route == PATHS[0] else ((start, turn), (turn, goal))
    points = [start, *(p for a, b in legs for p in pose_path(a, b))]
    matched = 0
    minimum_box_gap = None
    minimum_gaps = {}
    first_problem = None
    forbidden_waypoints = 0
    for index, tcp in enumerate(points):
        out = shadow.evaluate(tcp, tcp @ tool_from_object, grip)
        if not out['fit']['matched']:
            first_problem = first_problem or dict(kind='IK', waypoint=index, fit=out['fit'])
            break
        matched += 1
        gap = 1000 * box_separation(model, data, shadow.target, shadow.other)
        minimum_box_gap = gap if minimum_box_gap is None else min(minimum_box_gap, gap)
        for name, value in out['gaps'].items():
            if name not in minimum_gaps or value['gap_mm'] < minimum_gaps[name]['gap_mm']:
                minimum_gaps[name] = dict(value, waypoint=index)
        if out['forbidden']:
            forbidden_waypoints += 1
            first_problem = first_problem or dict(kind='CONTACT', waypoint=index, pairs=out['forbidden'])
    return dict(planned=len(points), IK_matched=matched, forbidden_waypoints=forbidden_waypoints,
                minimum_held_blue_to_red_box_separation_mm=minimum_box_gap,
                minimum_pair_distances_mm=minimum_gaps,
                first_problem=first_problem,
                passed=matched == len(points) and first_problem is None and minimum_box_gap > 0)


def run(root):
    base_spec = root / 'EXECUTION_SPEC.json'
    folder = root / 'cases' / CASE
    spec = read(folder / 'EXECUTION_SPEC.json')
    terminal = folder / 'RESULT.private.json'
    native = folder / 'native-01'
    queries = native / 'LIVE_DATA/QUERIES.private.json'
    model_path = Path(spec['model_path'])
    if (digest(base_spec) != PRODUCER_SPEC or digest(terminal) != RESULT or digest(queries) != QUERIES
            or spec['source_sha'] != PRODUCER_SOURCE or spec['collection_split'] != 'TRAIN'
            or spec['panel_entry_id'] != CASE or digest(model_path) != spec['frozen_files'][str(model_path)]
            or mujoco.__version__ != '3.2.7' or mujoco.get_mjcb_control() is not None):
        raise PermissionError('Expected the exact frozen original F2 TRAIN failure and geometry runtime')
    require_unstarted_cases(spec['registry'], {UNOPENED})
    population = read_pin(read(base_spec)['compiled_population'])
    row = next(row for row in population['conditions']['TRAIN'] if row['id'] == UNOPENED)
    next_scene = next(c['scene'] for c in population['clusters'] if c['id'] == row['cluster'])
    if (root / 'cases' / UNOPENED).exists() or digest(Path(next_scene['path'])) != next_scene['sha256']:
        raise PermissionError('Original unopened TRAIN scene changed')
    model = mujoco.MjModel.from_binary_path(str(model_path))
    next_model = mujoco.MjModel.from_binary_path(next_scene['path'])
    stop = last_stop(native / 'physics.jsonl')
    contacts = stopped_pairs(model, stop, 'task_blue_cube_geom')
    if (stop['step'] != 28437 or len(contacts) != 1
            or set(contacts[0]['geoms']) != {'task_blue_cube_geom', 'task_cube_geom'}):
        raise ValueError('Original native stop no longer identifies the blue-red contact')
    captures = {x['native_step']: x for x in read(queries) if x['phase'] == 'CARRY'}
    outcomes = []
    with forbid_integration(mujoco) as attempts:
        shadows = (ArmShadow(model, 'blue'), ArmShadow(next_model, 'blue'))
        shifts = scene_delta(model, next_model)
        if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or model.jnt_qposadr[0] != 0:
            raise ValueError('Root translation must retain the original robot free joint')
        for step in ANCHORS:
            capture = captures[step]
            if not capture['actual_state']['native_grasp']:
                raise ValueError('Each retained query must be in actual grasped carry')
            qpos = np.array(capture['actual_state']['qpos'])
            next_qpos = qpos.copy()
            for slots, delta in shifts:
                next_qpos[slots] += delta
            next_qpos[:3] += shifts[1][1]
            goal = np.eye(4)
            goal[:3, :3] = capture['raw_intent']['orientation']
            goal[:3, 3] = capture['raw_intent']['goal']
            for layout, shadow, positions in zip(('original', 'unopened'), shadows, (qpos, next_qpos), strict=True):
                results = {name: sweep(shadow, positions, goal, name) for name in PATHS}
                outcomes.append(dict(layout=layout, query_tick=step, routes=results))
    if attempts:
        raise RuntimeError('Counterfactual carry sweep attempted native integration')
    require_unstarted_cases(spec['registry'], {UNOPENED})
    return dict(status='STATIC_F2_BLUE_LEFT_CARRY_SWEEP', case=CASE, unopened_case=UNOPENED,
                result_sha256=RESULT, query_sha256=QUERIES, original_scene_sha256=digest(model_path),
                unopened_scene=next_scene, stop_tick=stop['step'], original_forbidden_pairs=contacts,
                rigid_contact_projection=True, poses=outcomes, native_requests=0, native_ticks=0,
                model_generations=0, optimizer_updates=0, VAL_inputs=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('producer', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = run(args.producer)
    with args.output.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
