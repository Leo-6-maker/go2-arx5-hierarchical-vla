"""Compare fixed F2 red approach paths using saved TRAIN states and the full arm."""
import argparse
import itertools
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from go2arx5_vla.world_model.native_replay import forbid_integration
from scripts.sla.core_hopper_carry_audit import history, stopped_pairs
from scripts.sla.core_hopper_red_approach import PITCH_DEGREES, world_pitch
from scripts.sla.core_multienv_place_sweep import ArmShadow, pose_path, site_pose
from scripts.sla.core_multienv_train import PRODUCER_SOURCE, PRODUCER_SPEC
from scripts.sla.hopper_parent_custody import digest, write_once

CASES = {
    'MEV1-TRAIN-F2-NEAR_NEG-00-HOLD-red':
        '1852d281138cdac76bf9bad070742dc5713e23646e78c8b04eb75e9df5c02b27',
    'MEV1-TRAIN-F2-NEAR_NEG-00-PLACE-red-left':
        '9d730b63e12ed2b6ca15dc6d41da2dff3fef8c25df65defd0a2a59033ad43b4f',
    'MEV1-TRAIN-F2-NEAR_NEG-00-PLACE-red-right':
        '3ff9adb8b381f7ddc8b9feebf34f53e4257b9331e449b88cbb1ed304e468ca7e',
}
PLACE_REPLICATION_CASES = {
    'MEV1-TRAIN-F2-NEAR_NEG-01-PLACE-red-left':
        'b32890d1a8353f9a695dd6d1aeca2708f1988201a25e53741c7691f2a8c69797',
    'MEV1-TRAIN-F2-NEAR_NEG-01-PLACE-red-right':
        'b444fac8727d4789a2eeab72d3b9f5dfd0529304637d6f8ea96875372b8327c8',
}
# Distance away from the blue cube and waypoint height above the frozen grasp goal.
DETOURS = ((8, 30), (16, 30), (24, 30), (8, 45), (16, 45), (24, 45))
CONFIGS = ((0, 0, 0), (PITCH_DEGREES, 0, 0),
           *((PITCH_DEGREES, offset, height) for offset, height in DETOURS))
GAP_PAIRS = ('arm_to_blue', 'arm_to_table', 'nonpad_arm_to_red')


def gaps(shadow):
    model, data = shadow.model, shadow.data
    out = {}
    for part in GAP_PAIRS:
        candidates = shadow.pairs[part]
        distance, arm, _ = min((float(mujoco.mj_geomDistance(model, data, a, b, .25,
                               shadow.buffer)), a, b) for a, b in candidates)
        out[part] = dict(mm=distance*1000, link=model.body(model.geom_bodyid[arm]).name)
    contacts = [dict(geom1=int(c.geom1), geom2=int(c.geom2), dist=float(c.dist)) for c in data.contact]
    return out, stopped_pairs(model,dict(contacts=contacts),'task_cube_geom')


def pose_goals(shadow, start, turn, intent, cfg):
    pitch, offset, elevation = cfg
    rotation = world_pitch(np.deg2rad(pitch)) @ turn
    original_offset = np.array([-.012, 0., 0.])
    red, blue = [shadow.data.geom_xpos[i].copy() for i in (shadow.target, shadow.other)]
    original = red + [0., -.0017, .020] + turn @ original_offset
    if not np.allclose(original,intent,rtol=0,atol=1e-6):
        raise ValueError('The frozen grasp-goal formula differs from the recorded intent')
    endpoint = np.eye(4)
    endpoint[:3,:3] = rotation
    endpoint[:3,3] = red + [0., -.0017, .020] + rotation @ original_offset
    if not pitch:
        return [endpoint]
    rotate = start.copy()
    rotate[:3,:3] = rotation
    if not offset:
        return [rotate,endpoint]
    side = red[:2]-blue[:2]
    side /= np.linalg.norm(side)
    approach = endpoint.copy()
    approach[:2,3] += (.001*offset)*side
    approach[2,3] += .001*elevation
    return [rotate,approach,endpoint]


def compare_path(shadow, row, intent, candidate):
    model, data = shadow.model, shadow.data
    data.qpos[:] = row['actual_state']['qpos']
    data.qvel[:] = 0.
    mujoco.mj_fwdPosition(model,data)
    start = site_pose(data,shadow.site)
    turn = np.asarray(intent['orientation'])
    goals = [start,*pose_goals(shadow,start,turn,np.asarray(intent['goal']),candidate)]
    minimum, first = {},None
    points = ik_count = closing_count = 0
    pad_ids = [model.geom(name).id for name in ('task_left_pad','task_right_pad')]
    finger_start = data.qpos[shadow.fingers].copy()

    def inspect(stage, index):
        nonlocal first
        margins, bad = gaps(shadow)
        for key,value in margins.items():
            if key not in minimum or value['mm'] < minimum[key]['mm']:
                minimum[key] = dict(value, stage=stage, waypoint=index)
        if bad and first is None:
            first = dict(stage=stage, waypoint=index, reason='FORBIDDEN_CONTACT', pair=bad[0])

    inspect('START',0)
    for index,target in enumerate(itertools.chain.from_iterable(
            pose_path(a,b) for a,b in zip(goals[:-1],goals[1:]))):
        points += 1
        fit = shadow.solve(target)
        if not fit['matched']:
            first = first or dict(stage='APPROACH', waypoint=index, reason='IK', fit=fit)
            break
        ik_count += 1
        inspect('APPROACH',index)
    bilateral = False
    if ik_count == points:
        for i, aperture in enumerate(np.linspace(1.,0.,int(np.ceil(max(finger_start)/.0005))+1)):
            data.qpos[shadow.fingers] = finger_start*aperture
            mujoco.mj_fwdPosition(model,data)
            closing_count += 1
            inspect('CLOSE',i)
            if max(mujoco.mj_geomDistance(model,data,p,shadow.target,.25,shadow.buffer) for p in pad_ids) <= 0:
                bilateral = True
                break
    return dict(query_tick=row['native_step'], attempted_waypoints=points, matched_waypoints=ik_count,
                closing_samples=closing_count, bilateral_pad_touch=bilateral,
                minima=minimum, first_problem=first, passed=(
                    first is None and bilateral and all(v['mm']>0 for v in minimum.values())))


def evaluate(root, cases=CASES, configs=CONFIGS):
    if (digest(root/'EXECUTION_SPEC.json')!=PRODUCER_SPEC
            or mujoco.__version__!='3.2.7' or mujoco.get_mjcb_control() is not None):
        raise PermissionError('Expected the original TRAIN scope and geometry-only MuJoCo')
    started = time.monotonic()
    rows = {r['condition']:r for r in json.loads((root/'RECORDS.private.json').read_bytes())}
    output = {}
    with forbid_integration(mujoco) as calls:
        for cid,expected in cases.items():
            row = rows[cid]
            if row['result']['sha256'] != expected or row['post_STOP_ticks']!=0:
                raise ValueError('Original STOP outcome changed')
            witness = history(root,row,source=PRODUCER_SOURCE)
            if (witness['phase'] != 'DESCEND' or not witness['pairs']
                    or any(set(c['bodies'])!={'x5_link6','task_blue_cube'} for c in witness['pairs'])):
                raise ValueError('Only the pre-registered wrist-to-blue failure is covered')
            case = root/'cases'/cid
            spec = json.loads((case/'EXECUTION_SPEC.json').read_bytes())
            queries = case/'native-01/LIVE_DATA/QUERIES.private.json'
            if digest(queries)!=witness['query_sha256']:
                raise ValueError('Saved real queries changed')
            observed = json.loads(queries.read_bytes())
            i = next(i for i,v in enumerate(observed) if v['phase']=='DESCEND')
            anchors = observed[i-1:i+1]
            if len(anchors)!=2 or anchors[0]['phase']!='PREGRASP':
                raise ValueError('Expected the saved approach-to-descent boundary')
            shadow = ArmShadow(mujoco.MjModel.from_binary_path(spec['model_path']))
            output[cid] = dict(witness=witness,paths=[
                dict(pitch_deg=cfg[0],away_mm=cfg[1],height_mm=cfg[2],
                     anchors=[compare_path(shadow,anchor,observed[i]['raw_intent'],cfg) for anchor in anchors])
                for cfg in configs])
        if calls:
            raise RuntimeError('Unexpected native integration')
    aggregate = [sum(a['passed'] for c in output.values() for a in c['paths'][i]['anchors'])
                 for i in range(len(configs))]
    return dict(status='SAVED_TRAIN_FIXED_BASE_FULL_ARM_DESCENT_AND_CLOSURE', configs=configs,
                passed_anchor_counts=aggregate, anchors=2*len(output), cases=output,
                native_requests=0, native_integrations=0, VAL_inputs=0,
                model_generations=0, optimizer_updates=0, wall_s=time.monotonic()-started)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('producer',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--place-replication',action='store_true',
                   help='Check only the original path and the fixed 20-degree detour on the two later PLACE stops.')
    args = p.parse_args()
    if args.place_replication:
        result = evaluate(args.producer,PLACE_REPLICATION_CASES,((0,0,0),(PITCH_DEGREES,24,30)))
    else:
        result = evaluate(args.producer)
    write_once(args.output,result)
