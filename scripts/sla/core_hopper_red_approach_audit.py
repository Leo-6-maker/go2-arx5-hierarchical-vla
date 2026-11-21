"""Check F2 red HOLD approach geometry at saved TRAIN poses without integration."""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from go2arx5_vla.world_model.multitask_carry_geometry_v6 import DistalCarryGeometry, world_yaw
from go2arx5_vla.world_model.native_replay import forbid_integration
from go2arx5_vla.world_model.mobile_continuous_teacher_v2 import ANGULAR_SPEED
from go2arx5_vla.world_model.multitask_scope_v6 import read_pin
from go2arx5_vla.world_model.storage import digest
from scripts.sla.core_hopper_carry_audit import BASE, history, read
from scripts.sla.core_hopper_red_approach import CELL, PITCH_DEGREES as CANDIDATE, world_pitch

YAW_DEGREES = (-90, -60, -30, 0, 30, 60, 90)
PITCH_DEGREES = (-20, -10, 0, 10, 15, 20)
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


def closing_projection(geometry, query, turn, shift, table_shift):
    """Sweep both true slides at the projected grasp goal. Stop measuring after bilateral intersection."""
    model, data = geometry.model, geometry.data
    sample = query['actual_state']
    tcp = np.asarray(sample['tcp'])[:3,3]
    goal = np.asarray(query['raw_intent']['goal'])
    joints = [model.joint(n).id for n in ('task_left_slide','task_right_slide')]
    pads = [model.geom(n).id for n in ('task_left_pad','task_right_pad')]
    slots = model.jnt_qposadr[joints]
    qpos = np.asarray(sample['qpos'])
    open_q = qpos[slots]
    if (not np.allclose(open_q, sample['finger_q'], atol=1e-8, rtol=0)
            or not np.allclose(model.jnt_range[joints], [[0.,.044],[0.,.044]])):
        raise ValueError('Finger aperture differs from the recorded slides')
    minima = {k:float('inf') for k in ('other_cube','table','selected_nonpad')}
    touch = [None, None]
    for remaining in np.linspace(1.,0.,int(np.ceil(max(open_q)/.0005))+1):
        data.qpos[:] = qpos
        data.qpos[slots] = remaining*open_q
        mujoco.mj_fwdPosition(model, data)
        data.geom_xpos[geometry.table] += table_shift
        parts = geometry.distal
        data.geom_xpos[parts] = goal + shift + (data.geom_xpos[parts]-tcp) @ turn.T
        data.geom_xmat[parts] = (turn @ data.geom_xmat[parts].reshape((-1,3,3))).reshape((-1,9))
        gaps = distances(geometry, set(pads))
        for name in minima:
            minima[name] = min(minima[name], 1000*gaps[name][0])
        pad_gaps = [float(mujoco.mj_geomDistance(model,data,pad,geometry.target,1.,geometry.distance_buffer))
                    for pad in pads]
        for i,gap in enumerate(pad_gaps):
            if gap <= 0 and touch[i] is None:
                touch[i] = 1000*remaining*open_q[i]
        if max(pad_gaps)<=0:
            break
    return dict(both_pads_intersect_red=max(pad_gaps)<=0,first_touch_slide_mm=touch,
                forbidden_min_gap_mm=minima)


def registered_shift(geometry, pilot):
    """Put the unopened scene's table in the target-relative frame of a recorded pose."""
    old = geometry.model
    base = mujoco.MjData(old)
    new = mujoco.MjData(pilot)
    mujoco.mj_fwdPosition(old, base)
    mujoco.mj_fwdPosition(pilot, new)
    relevant = [*geometry.distal,geometry.target,geometry.other,geometry.table]
    for field in ('geom_type','geom_size','geom_dataid','geom_bodyid','geom_quat','geom_pos'):
        if not np.array_equal(getattr(old,field)[relevant],getattr(pilot,field)[relevant]):
            raise ValueError('A registered collider changed size, identity or attachment')
    if not np.array_equal(old.mesh_vert,pilot.mesh_vert):
        raise ValueError('Unopened scene mesh vertices differ')
    if not np.allclose(base.geom_xmat[relevant],new.geom_xmat[relevant],atol=1e-10,rtol=0):
        raise ValueError('Unopened scene collider directions differ')
    delta = new.geom_xpos-base.geom_xpos
    if not np.allclose(delta[geometry.target],delta[geometry.other],atol=1e-10,rtol=0):
        raise ValueError('Target-relative non-target cube position changed')
    return delta[geometry.table]-delta[geometry.target]


def analyze_case(root, entry, angles, rotate, pilot=None, *, source=BASE):
    witness = history(root, entry,source=source)
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
    table_shift = np.zeros(3) if pilot is None else registered_shift(geom,pilot)
    pads = {model.geom(n).id for n in ('task_left_pad','task_right_pad')}
    grippers = geom.distal
    result = {}
    closing = {}
    for deg in angles:
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
            geom.data.geom_xpos[geom.table] += table_shift
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
                    sample(tcp, positions, rotations, rotate(np.deg2rad(angle)), np.zeros(3),'pregrasp',tick)
                continue
            original = target + [0.,-.0017,.020] + orient @ OFFSET
            if not np.allclose(original, goal, atol=1e-9, rtol=0):
                raise ValueError('Exported intent differs from the frozen F2 grasp offset')
            turn = rotate(np.deg2rad(deg))
            shift = (turn - np.eye(3)) @ orient @ OFFSET
            sample(tcp, positions, rotations, turn, shift, 'descend', tick)
            if i == len(queries)-1:
                travel = goal-tcp
                for alpha in np.linspace(0.,1.,max(2,int(np.ceil(np.linalg.norm(travel)/.001))+1)):
                    sample(tcp, positions, rotations, turn, shift+alpha*travel, 'remaining_to_goal', tick)
                if rotate is world_pitch:
                    closing[str(deg)] = closing_projection(geom,q,turn,shift,table_shift)
        result[str(deg)] = minima
    return dict(condition=entry['condition'], native_result_sha256=witness['result_sha256'],
                scene_sha256=witness['scene_sha256'], queries_sha256=witness['query_sha256'],
                terminal_tick=witness['terminal_tick'], terminal_contact_mm=witness['pairs'][0]['signed_distance_mm'],
                query_count=len(queries), descend_queries=sum(q['phase']=='DESCEND' for q in queries),
                by_angle_deg=result, closing_projection=closing,table_shift_mm=(1000*table_shift).tolist())


def run(root, axis, pilot=False):
    started = time.monotonic()
    if mujoco.__version__!='3.2.7' or mujoco.get_mjcb_control() is not None:
        raise ValueError('Geometry audit requires the original isolated MuJoCo version and no controller callback')
    spec=read(root/'EXECUTION_SPEC.json')
    if spec['source_sha']!=BASE:
        raise ValueError('Expected the original frozen expert')
    ids = [CELL+'-00', CELL+'-01']
    entries = {r['condition']:r for r in read(root/'RECORDS.private.json') if r['condition'] in ids}
    angles,rotate = (YAW_DEGREES,world_yaw) if axis=='yaw' else (PITCH_DEGREES,world_pitch)
    if pilot and axis!='pitch':
        raise ValueError('Only the identified pitch candidate has an unopened static probe')
    unopened = None
    pilot_scene = None
    candidate = CANDIDATE
    with forbid_integration(mujoco) as calls:
        if pilot:
            pop = read_pin(spec['compiled_population'])
            row = next(r for r in pop['conditions']['TRAIN'] if r['id']==CELL+'-02')
            pilot_scene = next(c['scene'] for c in pop['clusters'] if c['id']==row['scene_family'])
            if digest(Path(pilot_scene['path']))!=pilot_scene['sha256']:
                raise ValueError('Pilot numerical scene differs from its registered SHA')
            unopened=mujoco.MjModel.from_binary_path(pilot_scene['path'])
            angles=(candidate,)
        cases = [analyze_case(root, entries[name], angles, rotate, unopened) for name in ids]
        if calls:
            raise RuntimeError('Native integration was attempted in geometry analysis')
    out=dict(status='STATIC_SAVED_TRAIN_DISTAL_GEOMETRY', source_expert=BASE, cases=cases,
                axis=axis,angles_degrees=angles,pregrasp_turn_spacing_deg=5,remaining_path_sample_spacing_mm=1,
                distinct_scenes=len({c['scene_sha256'] for c in cases}), native_requests=0,
                simulation_integrations=0,model_generations=0,optimizer_updates=0,VAL_inputs=0,
                cpu_seconds=time.process_time(),wall_seconds=time.monotonic()-started)
    if pilot:
        floors=[]
        for case in cases:
            stages=case['by_angle_deg'][str(candidate)]
            closing=case['closing_projection'][str(candidate)]
            floors += [v[k]['gap_mm'] for v in stages.values() for k in ('other_cube','table','selected_nonpad')]
            floors += list(closing['forbidden_min_gap_mm'].values())
            if not closing['both_pads_intersect_red']:
                raise ValueError('No bilateral pad closure in the registered geometric projection')
        if min(floors)<=0:
            raise ValueError('The registered pose projection contains a forbidden intersection')
        out.update(status='PILOT_PITCH_GEOMETRY_VERIFIED',pilot=CELL+'-02',prior=ids,
                   pilot_scene_sha256=pilot_scene['sha256'],pitch_deg=candidate,
                   minimum_forbidden_gap_mm=min(floors),angular_command_cap_rad_s=ANGULAR_SPEED,
                   ideal_turn_lower_bound_s=float(np.deg2rad(candidate)/ANGULAR_SPEED))
    return out


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('original_producer',type=Path)
    cli.add_argument('output',type=Path)
    cli.add_argument('--axis', choices=('yaw','pitch'), default='yaw')
    cli.add_argument('--pilot', action='store_true')
    args=cli.parse_args()
    result=run(args.original_producer,args.axis,args.pilot)
    with args.output.open('x') as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
