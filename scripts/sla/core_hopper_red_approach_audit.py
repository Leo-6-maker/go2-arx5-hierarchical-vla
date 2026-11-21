"""Check F2 red HOLD approach geometry at saved TRAIN poses without integration."""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from go2arx5_vla.world_model.multitask_carry_geometry_v6 import DistalCarryGeometry, world_yaw
from go2arx5_vla.world_model.native_replay import forbid_integration
from go2arx5_vla.world_model.storage import digest
from scripts.sla.core_hopper_carry_audit import BASE, history, read

CELL = 'CHV1-TRAIN-F2-HOLD-red'
YAW_DEGREES = (-90, -60, -30, 0, 30, 60, 90)
OFFSET = np.array([-.012, 0., 0.])


def distances(geometry, pads):
    model, data = geometry.model, geometry.data
    gaps = {}
    for part in geometry.distal:
        for name, obstacle in [('other_cube', geometry.other), ('table', geometry.table),
                               ('selected_pad' if part in pads else 'selected_nonpad', geometry.target)]:
            distance = float(mujoco.mj_geomDistance(model, data, part, obstacle, 1., geometry.distance_buffer))
            if name not in gaps or distance < gaps[name][0]:
                gaps[name] = (distance, part)
    return gaps


def analyze_case(root, entry):
    witness = history(root, entry)
    if (witness['phase'] != 'DESCEND' or witness['last_query_native_grasp']
            or not witness['pairs'] or any(set(p['bodies']) != {'x5_link6', 'task_blue_cube'} for p in witness['pairs'])):
        raise ValueError('Requires the original wrist-to-other-cube STOP before grasp')
    case = root/'cases'/entry['condition']
    spec = read(case/'EXECUTION_SPEC.json')
    query_file = case/'native-01/LIVE_DATA/QUERIES.private.json'
    queries = read(query_file)
    if digest(query_file) != witness['query_sha256'] or any(q['phase'] not in ('PREGRASP','DESCEND') for q in queries):
        raise ValueError('Unexpected capture or query phase')
    model = mujoco.MjModel.from_binary_path(spec['model_path'])
    geom = DistalCarryGeometry(model, 'red')
    pads = {model.geom(n).id for n in ('task_left_pad','task_right_pad')}
    grippers = geom.distal
    result = {}
    for deg in YAW_DEGREES:
        minima = {}
        def sample(tcp, positions, rotations, turn, translation, key, tick):
            geom.data.geom_xpos[grippers] = tcp + (positions - tcp) @ turn.T + translation
            geom.data.geom_xmat[grippers] = (turn @ rotations).reshape((-1,9))
            gaps = distances(geom, pads)
            for name, (gap, part) in gaps.items():
                values = minima.setdefault(key, {})
                if name not in values or gap*1000 < values[name]['gap_mm']:
                    values[name] = dict(gap_mm=1000*gap, geom_id=part, body=model.body(model.geom_bodyid[part]).name,
                                        native_step=tick)
        for i, q in enumerate(queries):
            actual = q['actual_state']
            geom.data.qpos[:] = actual['qpos']
            mujoco.mj_fwdPosition(model, geom.data)
            positions = geom.data.geom_xpos[grippers].copy()
            rotations = geom.data.geom_xmat[grippers].reshape((-1,3,3)).copy()
            tcp = np.asarray(actual['tcp'])[:3,3]
            orient = np.asarray(q['raw_intent']['orientation'])
            goal = np.asarray(q['raw_intent']['goal'])
            target = np.asarray(actual['object_pose'])[:3,3]
            tick = q['native_step']
            if q['phase'] == 'PREGRASP':
                # The exact turn schedule is unknown. Check intermediate headings at each observed TCP.
                for angle in np.linspace(0., deg, abs(deg)//5+1):
                    sample(tcp, positions, rotations, world_yaw(np.deg2rad(angle)), np.zeros(3),'pregrasp',tick)
                continue
            original = target + [0.,-.0017,.020] + orient @ OFFSET
            if not np.allclose(original, goal, atol=1e-9, rtol=0):
                raise ValueError('Exported intent differs from the frozen F2 grasp offset')
            turn = world_yaw(np.deg2rad(deg))
            shift = (turn - np.eye(3)) @ orient @ OFFSET
            sample(tcp, positions, rotations, turn, shift, 'descend', tick)
            if i == len(queries)-1:
                travel = goal-tcp
                for alpha in np.linspace(0.,1.,max(2,int(np.ceil(np.linalg.norm(travel)/.001))+1)):
                    sample(tcp, positions, rotations, turn, shift+alpha*travel, 'remaining_to_goal', tick)
        result[str(deg)] = minima
    return dict(condition=entry['condition'], native_result_sha256=witness['result_sha256'],
                scene_sha256=witness['scene_sha256'], queries_sha256=witness['query_sha256'],
                terminal_tick=witness['terminal_tick'], terminal_contact_mm=witness['pairs'][0]['signed_distance_mm'],
                query_count=len(queries), descend_queries=sum(q['phase']=='DESCEND' for q in queries),
                by_yaw_deg=result)


def run(root):
    started = time.monotonic()
    if mujoco.__version__!='3.2.7' or mujoco.get_mjcb_control() is not None:
        raise ValueError('Geometry audit requires the original isolated MuJoCo version and no controller callback')
    if read(root/'EXECUTION_SPEC.json')['source_sha']!=BASE:
        raise ValueError('Expected the original frozen expert')
    ids = [CELL+'-00', CELL+'-01']
    entries = {r['condition']:r for r in read(root/'RECORDS.private.json') if r['condition'] in ids}
    with forbid_integration(mujoco) as calls:
        cases = [analyze_case(root, entries[name]) for name in ids]
        if calls:
            raise RuntimeError('Native integration was attempted in geometry analysis')
    return dict(status='STATIC_SAVED_TRAIN_DISTAL_GEOMETRY', source_expert=BASE, cases=cases,
                yaw_degrees=YAW_DEGREES,pregrasp_turn_spacing_deg=5,remaining_path_sample_spacing_mm=1,
                distinct_scenes=len({c['scene_sha256'] for c in cases}), native_requests=0,
                simulation_integrations=0,model_generations=0,optimizer_updates=0,VAL_inputs=0,
                cpu_seconds=time.process_time(),wall_seconds=time.monotonic()-started)


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('original_producer',type=Path)
    cli.add_argument('output',type=Path)
    args=cli.parse_args()
    result=run(args.original_producer)
    with args.output.open('x') as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
