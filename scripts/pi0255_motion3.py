"""PI0255 frozen motion registry, expert geometry, scorer and input boundary.

CPU only: this module cannot load a policy or advance a simulator.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import math
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from go2arx5_vla.trajectory.so3 import quat_slerp, quat_to_matrix

AUTHORITY = 'eac535e7c4016df39821798b111fae7ad79e8253'
TASKS = ('M1', 'M2', 'M3')
PERIOD = .05
ROOT = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0255')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def ordering(value, namespace):
    return hashlib.sha256((namespace + ':' + json.dumps(value, sort_keys=True)).encode()).hexdigest()


def transform_pose(position, quaternion, transform):
    """Full SE(3), including rotation; never subtract mixed-frame positions."""
    transform = np.asarray(transform, float)
    assert transform.shape == (4, 4) and np.isfinite(transform).all()
    assert np.allclose(transform[3], [0, 0, 0, 1], atol=1e-12)
    r = transform[:3, :3]
    assert np.allclose(r.T @ r, np.eye(3), atol=1e-8) and np.linalg.det(r) > .999999
    p = np.asarray(position, float) @ r.T + transform[:3, 3]
    return p, r @ quat_to_matrix(np.asarray(quaternion, float))


def instructions(task, command, colors):
    target = colors[command]
    if task == 'M1':
        return [f'Approach the {target} target and align the tool with its axes.',
                f'Bring the tool to the {target} marker with the pose shown by its axes.',
                f'Move to the {target} target, matching its displayed orientation.',
                f'Reach the {target} marker and match the marked tool frame.']
    if task == 'M2':
        side = ('left', 'right')[command]
        return [f'Face the orange goal and go around the grey block on the {side}.',
                f'Reach the orange goal via the {side} side of the grey obstacle.',
                f'Take the {side} route around the grey block to the orange goal.',
                f'Go to the orange goal, passing the grey obstacle on the {side}.']
    direction = ('clockwise', 'counterclockwise')[command]
    return [f'Trace the closed contour once {direction} as seen from above, starting at the black marker.',
            f'Follow one full {direction} lap of the contour from the black start marker, viewed from above.',
            f'From the black marker, traverse the contour once in the {direction} direction viewed from above.',
            f'Complete one {direction} circuit of the contour starting at black, as viewed from above.']


def semantics(instruction, task, colors):
    text = instruction.lower()
    if task == 'M1':
        matches = [i for i, name in enumerate(colors) if name in text]
        assert len(matches) == 1 and any(word in text for word in ('pose', 'axes', 'orientation', 'frame'))
        return matches[0]
    if task == 'M2':
        assert 'orange' in text and any(word in text for word in ('block', 'obstacle'))
        assert ('left' in text) != ('right' in text)
        return 0 if 'left' in text else 1
    assert 'above' in text and 'black' in text and 'contour' in text
    return 1 if 'counterclockwise' in text else 0


def freeze(output, source):
    output, source = Path(output), Path(source)
    config = json.loads((source / 'resolved_native_config_cpu.json').read_text())
    platform = json.loads((source / 'platform_source_readback.json').read_text())
    assert platform['counters']['expert_episodes_started'] == 0
    arm_origins = [np.fromstring(j['origin']['xyz'], sep=' ') for j in platform['urdf_joints']
                   if j['name'] in tuple('x5_joint' + str(i) for i in range(1, 7))]
    reach_bound = sum(float(np.linalg.norm(x)) for x in arm_origins) + float(np.linalg.norm(platform['tcp_offset_link6_m']))
    center = np.asarray(config['nominal_tcp_world'], float)
    center[2] += .025
    # ponytail: use one Cartesian factor registry; no adaptive scene search.
    candidates = [dict(x_shift=x, y_shift=y, z_shift=z, radius=radius, colors=colors)
                  for x, y, z, radius, colors in itertools.product(
                      (-.035, 0., .035), (-.025, 0., .025), (0., .025), (.045, .055),
                      (('red', 'blue'), ('blue', 'green'), ('green', 'red')))]
    candidates.sort(key=lambda value: ordering(value, 'PI0255_SCENE_V1:25501'))
    families = []
    episode_rows = []
    for i, factors in enumerate(candidates[:76]):
        split = 'PILOT' if i < 6 else 'TRAIN' if i < 46 else 'MT-VAL' if i < 56 else 'MT-TEST_COMPOSITION'
        families.append(dict(family_id=f'pi0255-family-{i:03d}', split=split, factors=factors,
                             reset_seed=2550100 + i, center_world=(center + [factors['x_shift'], factors['y_shift'], factors['z_shift']]).tolist()))
    for i in range(20):
        families.append(dict(family_id=f'pi0255-geometry-test-{i:03d}', split='MT-TEST_GEOMETRY', reset_seed=2551000 + i,
                             factors=dict(x_shift=.052 + .002 * (i % 5), y_shift=(-1 if i % 2 else 1) * (.04 + .002 * (i // 5)),
                                          z_shift=.04, radius=.064, colors=('red', 'blue')),
                             manifest_only=True))
    trained = [f for f in families if f['split'] == 'TRAIN']
    for f in families:
        split = f['split']
        for task in TASKS:
            for command in (0, 1):
                paraphrases = instructions(task, command, f['factors']['colors'])
                assert all(semantics(s, task, f['factors']['colors']) == command for s in paraphrases)
                assigned = int(ordering([f['family_id'], task, command], 'PI0255_LANGUAGE_SPLIT_V1:25502'), 16) % 2
                if split not in ('TRAIN', 'PILOT'):
                    assigned += 2
                episode_rows.append(dict(episode_id=f"{f['family_id']}-{task}-c{command}", family_id=f['family_id'],
                                         task=task, command=command, split=split, instruction=paraphrases[assigned],
                                         paraphrases=paraphrases, reset_seed=f['reset_seed'], execute=split in ('PILOT', 'TRAIN', 'MT-VAL')))
        if split == 'MT-TEST_COMPOSITION':
            for key, value in f['factors'].items():
                assert any(t['factors'][key] == value for t in trained), (key, value)
    registry = dict(schema='PI0255_MOTION3_REGISTRY_V1', authority_commit=AUTHORITY,
        scenario_namespace='PI0255_SCENE_V1', scenario_seed=25501, language_namespace='PI0255_LANGUAGE_SPLIT_V1', language_seed=25502,
        source_sha256={name:sha(source / name) for name in ('resolved_native_config_cpu.json', 'platform_source_readback.json')},
        candidate_factor_order=candidates, families=families, episodes=episode_rows,
        planned_executed_episodes=336, hard_started_cap=360, invalid_engineering_reserve=24,
        pilot_required=10, pilot_attempts_per_task=12, train_success_required=64, val_success_required=16,
        train_complete_pairs_required=24, val_complete_pairs_required=6,
        horizon=40, action_period_s=PERIOD, native_control_dt_s=.02, settling_timeout_s=5., task_timeout_s=20.,
        settling_speed_mps=.02, settling_continuous_s=.5, neutral_parallel_gripper=.5,
        settling_reference=dict(kind='REGISTERED_SLOW_LINE_FROM_MEASURED_INITIAL_POSE',delta_world_m=[.005,0,0],duration_s=5.),
        construction_reference=dict(delta_m=[.005,0,0],duration_s=2.,first_step='UNCHANGED_ENV_RESET_ZERO_ACTION_STEP_COUNTED_AS_PHYSICS'),
        geometry=dict(nominal_tcp_world=config['nominal_tcp_world'], nominal_center_world=center.tolist(),
            target_offsets_world_m=[[0,-.075,0],[0,.075,0]], target_yaw_rad=[-.2,.2],
            obstacle_offset_world_m=[.12,0,0], obstacle_half_extents_m=[.020,.020,.040],
            route_goal_offset_world_m=[.22,0,0], route_waypoint_x_offsets_m=[.075,.165],
            route_side_offset_m=.095, corridor_half_width_m=.030, corridor_entry_x_offset_m=.090, corridor_exit_x_offset_m=.150,
            forbidden_tcp_margin_m=.020, contact_force_threshold_n=.000001, contour_points=24,
            target_radius_m=.018, contour_marker_radius_m=.004, start_marker_radius_m=.008,
            axes_marker_radius_m=.006, axes_length_m=.035,
            necessary_chain_reach_bound_m=reach_bound,
            reach_bound_formula='sum(norm(arm joint origin translations)) + norm(TCP offset); necessary geometric bound, not IK proof',
            scene_classes_all_tasks=['two_pose_targets_with_axes','grey_collidable_obstacle','orange_route_goal','closed_contour','black_start_marker']),
        experts=dict(M1=dict(motion_end_s=6., nominal_end_s=8.,target_pose='WORLD_AXES_VISUAL_MARKER'),
                     M2=dict(segment_end_s=[3.,6.,10.],motion_end_s=10.,nominal_end_s=12.,approach_axis='+WORLD_X',left='+WORLD_Y'),
                     M3=dict(approach_end_s=4.,motion_end_s=16.,nominal_end_s=18.,start_angle_rad=math.pi,traversal_duration_s=12.,clockwise=-1)),
        scoring=dict(time_grid_max_gap_s=.0201,M1=dict(position_m=.035,orientation_rad=.10,sustain_s=.5),
                     M2=dict(position_m=.035,sustain_s=.5,correct_corridor_order=True,no_forbidden_contact=True),
                     M3=dict(sectors=12,cross_track_rms_m=.035,sector_hysteresis_rad=.02,maximum_backtrack_rad=.12,
                             maximum_phase_step_rad=.35,full_cycle_tolerance_rad=.08,fixed_cross_track_segment_s=[4.,16.])),
        windowing=dict(observation_period_s=.10,use_actual_sensor_times=True,desired_future_source='REGISTERED_CONTINUOUS_EXPERT_AT_ACTUAL_OBSERVATION_TIME',
            terminal_padding='EXPLICIT_ENDPOINT_HOLD_WITH_ORIGINAL_ACTIVE_DIMENSION_MASK; timestamps and padding flags recorded',
            deployable_subset='successful episode AND raw/codec/gate valid; no task-outcome-based window redraw',
            required_motion_phase_quartiles=[0,1,2,3],after_motion_hold_windows='AUDIT_RETAINED; original minimum-arc gate may reject constant-only future',
            handoff_stop='No new chunk when elapsed >= motion_end; last admitted chunk executes its registered terminal hold'),
        model_input_allowlist=['body_rgb','wrist_rgb','state_v1','language'],
        future_test='MANIFEST_ONLY_NO_RENDER_NO_EXPERT_NO_MODEL_NO_STATISTICS',
        claim='THREE_NON_CONTACT_MOTION_BEHAVIORS_NOT_CONTACT_MANIPULATION',selected_checkpoint='NONE')
    assert len([r for r in episode_rows if r['execute']]) == 336
    output.mkdir(parents=True, exist_ok=True)
    path=output/'task_registry_v1.json'
    assert not path.exists(), 'immutable registry already exists'
    write(path,registry)
    print(json.dumps(dict(status='PRE_EXECUTION_REGISTRY_MATERIALIZED',sha256=sha(path),planned=336,test_manifest_only=240)))
    return registry


def scene(registry, family):
    center = np.asarray(family['center_world'], float)
    geo = registry['geometry']
    return dict(center=center, targets=center + np.asarray(geo['target_offsets_world_m']),
                obstacle=center + geo['obstacle_offset_world_m'], goal=center + geo['route_goal_offset_world_m'],
                radius=float(family['factors']['radius']))


def expert_at(registry, family, task, command, starting_position, starting_quaternion, times):
    """One frozen expert, no resampling, no outcome-dependent corrections."""
    assert family['split'] in ('PILOT', 'TRAIN', 'MT-VAL'), 'TEST expert execution forbidden'
    data=scene(registry,family);times=np.asarray(times,float);p0=np.asarray(starting_position,float)
    q0=np.asarray(starting_quaternion,float);assert q0.shape==(4,) and np.isfinite(q0).all()
    if task=='M1':
        goal=data['targets'][command];u=np.clip(times/6.,0,1);u=u*u*(3-2*u)
        positions=p0+(goal-p0)*u[:,None]
        angle=registry['geometry']['target_yaw_rad'][command]
        target_q=np.asarray([0,0,math.sin(angle/2),math.cos(angle/2)])
        quaternions=np.asarray([quat_slerp(q0,target_q,float(t)) for t in u])
    elif task=='M2':
        side=1 if command==0 else -1;c=data['center'];geo=registry['geometry']
        waypoints=[p0,c+[geo['route_waypoint_x_offsets_m'][0],side*geo['route_side_offset_m'],0],
                   c+[geo['route_waypoint_x_offsets_m'][1],side*geo['route_side_offset_m'],0],data['goal']]
        ends=[0.,3.,6.,10.];positions=np.zeros((len(times),3))
        for i,t in enumerate(times):
            segment=min(int(np.searchsorted(ends,t,side='right'))-1,2);segment=max(segment,0)
            u=float(np.clip((t-ends[segment])/(ends[segment+1]-ends[segment]),0,1));u=u*u*(3-2*u)
            positions[i]=waypoints[segment]+u*(waypoints[segment+1]-waypoints[segment])
        quaternions=np.tile(q0,(len(times),1))
    else:
        assert task=='M3';direction=-1 if command==0 else 1
        entry=data['center']+[-data['radius'],0,0];u=np.clip(times/4.,0,1);u=u*u*(3-2*u)
        positions=p0+(entry-p0)*u[:,None]
        active=times>=4.;phase=math.pi+direction*2*math.pi*np.clip((times[active]-4.)/12.,0,1)
        positions[active]=data['center']+np.column_stack([data['radius']*np.cos(phase),data['radius']*np.sin(phase),np.zeros(len(phase))])
        quaternions=np.tile(q0,(len(times),1))
    return positions,quaternions


def sustained(times, mask, duration=.5):
    times=np.asarray(times,float);mask=np.asarray(mask,bool);start=None
    for t,okay in zip(times,mask):
        start=float(t) if okay and start is None else start if okay else None
        if start is not None and float(t)-start>=duration-1e-9:return True
    return False


def score(registry, family, task, command, timestamps, achieved_world, achieved_quaternion, contacts, clean, desired_world=None):
    times=np.asarray(timestamps,float);positions=np.asarray(achieved_world,float);quats=np.asarray(achieved_quaternion,float)
    assert len(times)>=2 and positions.shape==(len(times),3) and quats.shape==(len(times),4)
    assert np.isfinite(times).all() and np.isfinite(positions).all() and np.isfinite(quats).all()
    assert np.all(np.diff(times)>0) and times[0]>=0 and times[-1]<=20.02
    assert np.diff(times).max()<=registry['scoring']['time_grid_max_gap_s'], 'missing same-tick samples'
    assert np.max(np.abs(np.linalg.norm(quats,axis=1)-1))<1e-5
    data=scene(registry,family);result=dict(task=task,command=command,physical_clean=bool(clean),forbidden_contact=bool(np.any(contacts)))
    if desired_world is not None:
        desired_world=np.asarray(desired_world,float);assert desired_world.shape==positions.shape
        result['same_tick_tracking_rms_m']=float(np.sqrt(np.mean(np.sum((positions-desired_world)**2,axis=1))))
        result['reference_is_not_achieved']=True
    if task=='M1':
        errors=np.linalg.norm(positions-data['targets'][command],axis=1)
        angle=registry['geometry']['target_yaw_rad'][command];target=np.asarray([0,0,math.sin(angle/2),math.cos(angle/2)])
        rotation=2*np.arccos(np.clip(np.abs(quats@target),0,1))
        outcome=sustained(times,(errors<=.035)&(rotation<=.10))
        result.update(final_target_error_m=float(errors[-1]),min_target_error_m=float(errors.min()),final_orientation_error_rad=float(rotation[-1]),joint_sustained=outcome)
    elif task=='M2':
        c=data['center'];side=1 if command==0 else -1;geo=registry['geometry']
        relative=positions-c;chosen=side*relative[:,1]
        corridor=(chosen>=geo['route_side_offset_m']-geo['corridor_half_width_m'])&(chosen<=geo['route_side_offset_m']+geo['corridor_half_width_m'])
        entry=np.flatnonzero(corridor&(relative[:,0]>=geo['corridor_entry_x_offset_m'])&(relative[:,0]<=.12))
        exits=np.flatnonzero(corridor&(relative[:,0]>=geo['corridor_exit_x_offset_m']))
        order=bool(len(entry) and len(exits) and entry[0]<exits[-1])
        forbidden=np.all(np.abs(positions-data['obstacle'])<=np.asarray(geo['obstacle_half_extents_m'])+geo['forbidden_tcp_margin_m'],axis=1)
        errors=np.linalg.norm(positions-data['goal'],axis=1)
        exit_time=float(times[exits[exits>entry[0]][0]]) if order else math.inf
        hold=sustained(times,(errors<=.035)&(times>=exit_time))
        wrong=np.any((-side*relative[:,1]>.06)&(relative[:,0]>.075)&(relative[:,0]<.165))
        outcome=order and hold and not forbidden.any() and not wrong
        result.update(correct_corridor_order=order,wrong_side=bool(wrong),forbidden_volume=bool(forbidden.any()),goal_sustained=hold,final_target_error_m=float(errors[-1]))
    else:
        assert task=='M3';direction=-1 if command==0 else 1;rule=registry['scoring']['M3']
        relative=positions-data['center'];angle=np.arctan2(relative[:,1],relative[:,0]);phase=np.unwrap(angle)
        traversal=(times>=4.)&(times<=16.+1e-9);indices=np.flatnonzero(times>=4.)
        if traversal.sum()<2 or times[-1]<16.-.021:
            return dict(result,success=False,metric_complete=False,reason='DECLARED_TRAVERSAL_SEGMENT_INCOMPLETE',fixed_segment_s=[4.,16.])
        radius=np.linalg.norm(relative[:,:2],axis=1);cross=np.sqrt((radius-data['radius'])**2+relative[:,2]**2)
        progress=direction*(phase[indices]-phase[indices[0]]);steps=np.diff(progress)
        hysteresis=rule['sector_hysteresis_rad'];sector_width=2*math.pi/12
        crossed=[];next_sector=1
        for value in progress:
            if next_sector<=12 and value>=next_sector*sector_width-hysteresis:
                crossed.append(next_sector%12);next_sector+=1
        ordered=len(crossed)==12 and crossed==list(range(1,12))+[0]
        wrong=bool(np.any(steps < -rule['maximum_backtrack_rad']) or progress[-1]<-sector_width)
        skipped=bool(np.any(np.abs(steps)>rule['maximum_phase_step_rad']))
        full=float(progress.max())>=2*math.pi-rule['full_cycle_tolerance_rad']
        rms=float(np.sqrt(np.mean(cross[traversal]**2)))
        start_error=abs(float((angle[indices[0]]-math.pi+math.pi)%(2*math.pi)-math.pi))
        start_sector=start_error<=sector_width+hysteresis
        outcome=start_sector and ordered and full and not wrong and not skipped and rms<=.035
        result.update(covered_sectors=crossed,sector_order_pass=ordered,wrong_direction=wrong,skipped_sector=skipped,
                      angular_progress_rad=float(progress.max()),cross_track_rms_m=rms,cross_track_p95_m=float(np.quantile(cross[traversal],.95)),
                      cross_track_max_m=float(cross[traversal].max()),fixed_segment_s=[4.,16.],complete_cycle=full,
                      registered_start_sector=start_sector,start_sector_error_rad=start_error)
    result['success']=bool(outcome and clean and not result['forbidden_contact'])
    return result


def policy_input(example):
    """Actual dataset boundary: audit data must be supplied separately."""
    assert set(example)=={'body_rgb','wrist_rgb','state_v1','language'}, 'privileged or missing input'
    for name in ('body_rgb','wrist_rgb'):
        image=np.asarray(example[name]);assert image.dtype==np.uint8 and image.shape==(96,96,3)
    state=np.asarray(example['state_v1']);assert state.shape==(10,) and np.isfinite(state).all()
    assert isinstance(example['language'],str) and example['language']
    return dict(example)


def cpu_check(registry):
    episodes=registry['episodes'];families=registry['families'];lookup={f['family_id']:f for f in families}
    assert len({f['family_id'] for f in families})==96
    assert len([r for r in episodes if r['execute']])==336
    assert all(len({r['split'] for r in episodes if r['family_id']==f['family_id']})==1 for f in families)
    assert all(not r['execute'] for r in episodes if r['split'].startswith('MT-TEST'))
    for task in TASKS:
        assert [sum(r['task']==task and r['split']==s for r in episodes) for s in ('PILOT','TRAIN','MT-VAL')]==[12,80,20]
    family=lookup[episodes[0]['family_id']];data=scene(registry,family);p0=data['center']+[-.02,0,0];q0=np.array([0,0,0,1.])
    tests=[]
    for task in TASKS:
        duration=registry['experts'][task]['nominal_end_s'];times=np.arange(0,duration+.001,.02)
        for command in (0,1):
            p,q=expert_at(registry,family,task,command,p0,q0,times)
            result=score(registry,family,task,command,times,p,q,np.zeros(len(times),bool),True,p)
            assert result['success'],result
            tests.append(task+'_correct_'+str(command))
            other_p,other_q=expert_at(registry,family,task,1-command,p0,q0,times)
            assert not score(registry,family,task,command,times,other_p,other_q,np.zeros(len(times),bool),True)['success']
            tests.append(task+'_wrong_semantics_'+str(command))
            if task=='M3':
                skipped=p.copy();skipped[210:240]=p[300]
                assert not score(registry,family,task,command,times,skipped,q,np.zeros(len(times),bool),True)['success']
                assert score(registry,family,task,command,times,other_p,other_q,np.zeros(len(times),bool),True)['wrong_direction']
            missed=p+np.asarray([.2,.2,.2])
            assert not score(registry,family,task,command,times,missed,q,np.zeros(len(times),bool),True,p)['success']
            assert not score(registry,family,task,command,times,p,q,np.ones(len(times),bool),True,p)['success']
    transform=np.eye(4);transform[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];transform[:3,3]=[1,2,3]
    position,rotation=transform_pose([.2,.3,.4],q0,transform)
    assert np.allclose(position,[.7,2.2,3.4]) and np.allclose(rotation,transform[:3,:3])
    assert not np.allclose(position,np.array([.2,.3,.4])+transform[:3,3])
    fake=dict(body_rgb=np.zeros((96,96,3),np.uint8),wrist_rgb=np.zeros((96,96,3),np.uint8),state_v1=np.zeros(10),language='test')
    assert policy_input(fake)['language']=='test'
    for key in ('target_world','phase','source_id','expert_future','success','reward'):
        try:policy_input(dict(fake,**{key:0}))
        except AssertionError:pass
        else:raise AssertionError('forbidden input accepted')
    try:score(registry,family,'M1',0,[0,0],np.zeros((2,3)),np.tile(q0,(2,1)),[False,False],True)
    except AssertionError:pass
    else:raise AssertionError('time mismatch accepted')
    try:score(registry,family,'M1',0,[20.,20.1],np.zeros((2,3)),np.tile(q0,(2,1)),[False,False],True)
    except AssertionError:pass
    else:raise AssertionError('timeout accepted')
    return dict(status='PASS_PI0255_CPU_REGISTRY_EXPERT_SCORER_BOUNDARY',tests=tests+['rotated_SE3','mixed_frame_negative','time_negative','timeout_negative','reference_only_hit_negative','forbidden_contact_negative','false_sector_crossing_negative','actual_policy_input_privilege_negative'],
                code_sha256=sha(__file__),new_GR00T_loads=0,new_model_queries=0,physics_steps=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--cpu-check',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'design');parser.add_argument('--source-evidence',type=Path,default=ROOT/'evidence')
    args=parser.parse_args()
    if args.freeze:freeze(args.output_dir,args.source_evidence)
    elif args.cpu_check:
        path=args.output_dir/'task_registry_v1.json';receipt=cpu_check(json.loads(path.read_text()));receipt['registry_sha256']=sha(path)
        write(args.output_dir/'cpu_receipt.json',receipt);print(json.dumps(receipt))
    else:parser.print_help()
