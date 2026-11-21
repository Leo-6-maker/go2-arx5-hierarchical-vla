"""Extend the measured F2 red approach to the frozen 26 mm lift on two TRAIN layouts."""
import argparse
from pathlib import Path

import mujoco
import numpy as np

from go2arx5_vla.world_model.native_replay import forbid_integration
from scripts.pi0292.durable import read
from scripts.sla.core_multienv_place_sweep import ArmShadow, box_separation, pose_path, scene_delta, site_pose
from scripts.sla.core_multienv_red_descent import CASES, compare_path
from scripts.sla.core_multienv_red_unopened import PROOF_SHA, SCENE, SCENE_SHA
from scripts.sla.core_multienv_train import PRODUCER_SPEC
from scripts.sla.hopper_parent_custody import digest, write_once
from scripts.wm02.multitask_stage_cards_v6 import require_unstarted_cases

CANDIDATE = (20,24,30)
UNOPENED_PROOF = 'a92f17905638cecc17aeb463e34a1a20e1e8de61211c27522abf4e8a594d7add'
LIFT_M = .026


def check_distance_disagreement(shadow,waypoint):
    model,data = shadow.model,shadow.data
    pairs = shadow.pairs['arm_to_blue']
    out = []
    for arm,obstacle in pairs:
        expected = float(mujoco.mj_geomDistance(model,data,arm,obstacle,.25,shadow.buffer))
        if expected>0:
            continue
        bound = box_separation(model,data,arm,obstacle)
        pair = {arm,obstacle}
        observed = any({int(c.geom1),int(c.geom2)}==pair for c in data.contact)
        ranges = [
            dict(distmax_mm=upper*1000,
                 arm_to_blue_mm=float(mujoco.mj_geomDistance(model,data,arm,obstacle,upper,None))*1000,
                 blue_to_arm_mm=float(mujoco.mj_geomDistance(model,data,obstacle,arm,upper,None))*1000)
            for upper in (.025,.050,.100,.250)
        ]
        out.append(dict(waypoint=waypoint,geoms=[model.geom(g).name for g in (arm,obstacle)],
                        bodies=[model.body(model.geom_bodyid[g]).name for g in (arm,obstacle)],
                        returned_distance_mm=expected*1000,compiled_box_separation_mm=bound*1000 if bound is not None else None,
                        combined_geom_margin_mm=float(sum(model.geom_margin[g] for g in (arm,obstacle)))*1000,
                        active_contact=observed,distance_queries=ranges))
    return out


def lift_from_projection(shadow, approach):
    if not approach['passed']:
        raise ValueError('The previously verified closure did not replay')
    model, data = shadow.model, shadow.data
    start = site_pose(data,shadow.site)
    jaw = data.qpos[shadow.fingers].copy()
    held = np.eye(4)
    held[:3,:3] = data.geom_xmat[shadow.target].reshape(3,3)
    held[:3,3] = data.geom_xpos[shadow.target]
    if not np.allclose(held[:3,3],data.qpos[shadow.free:shadow.free+3],rtol=0,atol=1e-10):
        raise ValueError('Rigid lift projection needs the original object-centered collision shape')
    tool_from_object = np.linalg.solve(start,held)
    finish = start.copy()
    finish[2,3] += LIFT_M
    names = ('arm_to_blue','arm_to_table','nonpad_arm_to_red','held_red_to_blue','red_to_table')
    minima, first, matched = {}, None, 0
    pads = [model.geom(name).id for name in ('task_left_pad','task_right_pad')]
    maximum_pad_gap = -float('inf')
    distance_diagnostics = []
    targets = list(pose_path(start,finish))
    for index,target in enumerate(targets):
        outcome = shadow.evaluate(target,target @ tool_from_object,jaw)
        if not outcome['fit']['matched']:
            first = first or dict(reason='IK',waypoint=index,fit=outcome['fit'])
            break
        matched += 1
        for name in names:
            value = outcome['gaps'][name]
            if name not in minima or value['gap_mm']<minima[name]['gap_mm']:
                minima[name] = dict(value,waypoint=index)
        if outcome['forbidden']:
            first = first or dict(reason='FORBIDDEN_CONTACT',waypoint=index,contact=outcome['forbidden'][0])
        if outcome['gaps']['arm_to_blue']['gap_mm']<=0:
            distance_diagnostics.extend(check_distance_disagreement(shadow,index))
        for pad in pads:
            maximum_pad_gap = max(maximum_pad_gap,1000*mujoco.mj_geomDistance(
                model,data,pad,shadow.target,.25,shadow.buffer))
    return dict(approach_query=approach['query_tick'],planned_lift_waypoints=len(targets),
                matched_lift_waypoints=matched,nominal_lift_mm=1000*LIFT_M,
                object_final_height_delta_mm=1000*float(data.geom_xpos[shadow.target,2]-held[2,3]),
                minimum_clearances=minima,
                maximum_pad_gap_mm=maximum_pad_gap if matched else None,first_problem=first,
                distance_diagnostics=distance_diagnostics,
                passed=first is None and matched==len(targets) and all(v['gap_mm']>0 for v in minima.values()))


def run(root,retrospective,unopened):
    spec = read(root/'EXECUTION_SPEC.json')
    if (digest(root/'EXECUTION_SPEC.json')!=PRODUCER_SPEC or digest(retrospective)!=PROOF_SHA
            or digest(unopened)!=UNOPENED_PROOF or mujoco.__version__!='3.2.7'
            or mujoco.get_mjcb_control() is not None):
        raise PermissionError('Expected the two exact TRAIN geometry receipts and isolated MuJoCo')
    a,b = read(retrospective),read(unopened)
    index_a,index_b = (x['configs'].index(list(CANDIDATE)) for x in (a,b))
    if a['passed_anchor_counts'][index_a] != 6 or b['pass_counts'][index_b] != 6:
        raise ValueError('Candidate must have six original anchors per layout')
    ids = set(b['unopened_conditions'])
    if (len(ids)!=3 or any(not x.startswith(SCENE+'-') or (root/'cases'/x).exists() for x in ids)):
        raise ValueError('The registered next TRAIN conditions changed')
    require_unstarted_cases(spec['registry'],ids)
    new_scene = Path(b['unopened_scene']['path'])
    if digest(new_scene)!=SCENE_SHA:
        raise ValueError('New TRAIN model changed')

    outputs = []
    with forbid_integration(mujoco) as calls:
        for cid in CASES:
            c = root/'cases'/cid
            native = read(c/'EXECUTION_SPEC.json')
            scene = Path(native['model_path'])
            q = c/'native-01/LIVE_DATA/QUERIES.private.json'
            witness = a['cases'][cid]['witness']
            if digest(scene)!=witness['scene_sha256'] or digest(q)!=witness['query_sha256']:
                raise ValueError('Original geometry or anchor observations changed')
            old = mujoco.MjModel.from_binary_path(str(scene))
            rows = read(q)
            i = next(i for i,x in enumerate(rows) if x['phase']=='DESCEND')
            for name,model in (('original',old),('unopened',mujoco.MjModel.from_binary_path(str(new_scene)))):
                shadow = ArmShadow(model)
                shifts = scene_delta(old,model)
                intent = dict(rows[i]['raw_intent'],
                              goal=np.asarray(rows[i]['raw_intent']['goal'])+shifts[0][1])
                paths = []
                for row in rows[i-1:i+1]:
                    state = row['actual_state']
                    qpos = np.asarray(state['qpos']).copy()
                    for slots,delta in shifts:
                        qpos[slots] += delta
                    anchor = dict(row,actual_state=dict(state,qpos=qpos))
                    approach = compare_path(shadow,anchor,intent,CANDIDATE)
                    paths.append(lift_from_projection(shadow,approach))
                outputs.append(dict(condition=cid,layout=name,anchors=paths))
        if calls:
            raise RuntimeError('Unexpected native physics in the geometric replay')
    require_unstarted_cases(spec['registry'],ids)
    return dict(status='STATIC_FROZEN_F2_RED_LIFT',candidate=CANDIDATE,paths=outputs,
                held_object_projection='RIGID_AT_FIRST_BILATERAL_CLOSURE',
                actual_native_requests=0,actual_integrations=0,model_generations=0,
                optimizer_updates=0,VAL_inputs=0,native_grasp_lift_hold='NOT_RUN')


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('producer',type=Path)
    p.add_argument('retrospective',type=Path)
    p.add_argument('unopened',type=Path)
    p.add_argument('output',type=Path)
    a = p.parse_args()
    write_once(a.output,run(a.producer,a.retrospective,a.unopened))
