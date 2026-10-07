"""CPU arithmetic on completed PI0280 raw evidence; never imports a model or simulator."""
import argparse,collections,hashlib,importlib.util,json,os,shutil,subprocess,time
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    if path.exists():
        history=path.parent/'history';history.mkdir(exist_ok=True)
        shutil.copy2(path,history/(path.stem+'-'+sha(path)+path.suffix))
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def stats(values):
    a=np.asarray(values,dtype=float)
    assert len(a) and np.isfinite(a).all()
    return dict(mean=float(a.mean()),rms=float(np.sqrt(np.mean(a*a))),p95=float(np.quantile(a,.95)),max=float(a.max()),final=float(a[-1]))


def quat_matrix(q):
    x,y,z,w=np.asarray(q,dtype=float)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def orientation_error(actual,reference):
    return np.arccos(np.clip((np.sum(actual*reference,axis=(-2,-1))-1)/2,-1,1))


def file_row(path):
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))


def video(directory,out,name):
    from PIL import Image,ImageDraw
    path=directory/'frames.npz'
    if not path.exists():return 'UNOBSERVED_NO_SAVED_FRAMES'
    with np.load(path,allow_pickle=False) as data:
        body,wrist,t=data['body'],data['wrist'],data['time']
    assert len(body)==len(wrist)==len(t) and len(t)>0
    fps=10 if len(t)<2 else 1/float(np.median(np.diff(t)))
    destination=out/(name+'.mp4');temporary=destination.with_suffix('.tmp.mp4')
    process=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pixel_format','rgb24',
        '-video_size','768x448','-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264',
        '-threads','4','-pix_fmt','yuv420p',str(temporary)],stdin=subprocess.PIPE)
    try:
        for left,right,stamp in zip(body,wrist,t):
            image=Image.new('RGB',(768,448),'black');draw=ImageDraw.Draw(image)
            draw.text((8,8),'DELIVERED_OLD_WBC / actual simulation / fixture / NON_REALTIME',fill='white')
            draw.text((8,30),name+' | saved sim time %.3fs'%stamp,fill='white')
            draw.text((8,52),'Body camera',fill='white');draw.text((392,52),'Wrist camera',fill='white')
            image.paste(Image.fromarray(left).resize((384,384),Image.Resampling.NEAREST),(0,64))
            image.paste(Image.fromarray(right).resize((384,384),Image.Resampling.NEAREST),(384,64))
            process.stdin.write(image.tobytes())
        process.stdin.close();assert process.wait()==0
    except BaseException:
        process.kill();process.wait();raise
    if destination.exists():
        history=out/'history';history.mkdir(exist_ok=True)
        shutil.copy2(destination,history/(destination.stem+'-'+sha(destination)+destination.suffix))
    temporary.replace(destination)
    return dict(**file_row(destination),source=file_row(path),saved_frames=len(t),
        first_sim_time_s=float(t[0]),last_sim_time_s=float(t[-1]),fps=fps,
        rendering='Only actual saved body/wrist RGB; nearest pixel enlargement; timestamps preserved')


def fixture(directory,out,ledger,make_video,config):
    result=json.loads((directory/'RESULT.json').read_text());name=directory.name
    assert result['status']!='RUNNING'
    rows=json.loads((directory/'rows.json').read_text())
    actors=[json.loads(s) for s in (directory/'actor_outputs.jsonl').read_text().splitlines()] if (directory/'actor_outputs.jsonl').exists() else []
    origin=float(result.get('initialization_s',rows[0]['time'] if rows else 0))
    controlled=[r for r in rows if r['time']>origin+1e-7]
    start=json.loads((directory/'START.json').read_text())
    attempt_ledger=[r for r in ledger if r.get('fixture')==result['fixture'] and start['time']<=r['wall']<=(directory/'RESULT.json').stat().st_mtime]
    report=dict(fixture=result['fixture'],attempt=name,execution_status=result['status'],execution_error=result.get('error'),
        controller='DELIVERED_OLD_WBC',lane='WBC_NATIVE_FIXTURE',task_success='NOT_AN_OBJECT_TASK',
        selected='NONE',deployment='NONE',raw_result=result,initialization_sim_time_s=origin,
        saved_rows=len(rows),saved_controlled_rows=len(controlled),saved_actor_output_rows=len(actors),
        ledger_counts=dict(collections.Counter(r['kind'] for r in attempt_ledger)),
        ledger_attempt_window=dict(start_unix_s=start['time'],end_RESULT_mtime_unix_s=(directory/'RESULT.json').stat().st_mtime),
        installed_orientation_tracking='UNOBSERVED_NATIVE_INSTALLED_ROTATION_NOT_RECORDED',
        object_contacts='UNOBSERVED_NOT_AN_OBJECT_TASK',object_completion='NOT_AN_OBJECT_TASK',
        metric_semantics='Saved installed=p_at(s_ref(traj_sim_time)) after step; pre-step dispatched nine-point preview is not saved. Timestamped raw reference is a distinct clock from spatial progress.',
        raw_inputs=[file_row(p) for p in directory.iterdir() if p.is_file() and p.name!='frames.npz'])
    if rows:
        report['initialization_row']=dict(sim_time_s=rows[0]['time'],FK_cache_error_m=rows[0]['FK_cache_error_m'],
            hard_max_penetration=max(rows[0]['hard_penetration'].values()),
            soft_margin_min=min(rows[0]['telemetry']['joint_limit_margin']))
    if controlled:
        t=np.array([r['time'] for r in controlled]);actual=np.array([r['TCP'] for r in controlled]);installed=np.array([r['installed'] for r in controlled])
        with np.load(directory/'reference.npz',allow_pickle=False) as reference:
            p,q,tref=reference['position'],reference['quaternion'],reference['timestamp']
        assert np.isfinite(actual).all() and np.isfinite(installed).all() and np.all(np.diff(t)>0)
        assert np.allclose(np.abs(q@q[0]),1,atol=1e-6),'Only fixed-orientation fixtures are admitted here'
        relative=t-origin;assert relative.min()>=tref[0]-1e-6 and relative.max()<=tref[-1]+1e-6
        raw=np.stack([np.interp(relative,tref,p[:,j]) for j in range(3)],axis=1)
        names=list(controlled[0]['hard_penetration']);margins=np.array([r['telemetry']['joint_limit_margin'] for r in controlled]);hard=np.array([[r['hard_penetration'][j] for j in names] for r in controlled])
        assert margins.shape==hard.shape and margins.shape[1]==len(names)
        report.update(last_saved_sim_time_s=float(t[-1]),saved_controlled_elapsed_s=float(t[-1]-origin),
            actual_to_same_tick_installed_position_m=stats(np.linalg.norm(actual-installed,axis=1)),
            actual_to_timestamped_raw_position_m=stats(np.linalg.norm(actual-raw,axis=1)),
            same_tick_installed_to_timestamped_raw_position_m=stats(np.linalg.norm(installed-raw,axis=1)),
            actual_to_raw_fixed_orientation_rad=stats(orientation_error(np.array([r['rotation'] for r in controlled]),quat_matrix(q[0]))),
            FK_cache_error_m=stats([r['FK_cache_error_m'] for r in controlled]),
            hard_penetration=dict(unit='rad_for_revolute_m_for_prismatic',rows_with_any=int(np.any(hard>0,axis=1).sum()),per_joint_max=dict(zip(names,hard.max(axis=0).tolist()))),
            inherited_soft_margins=dict(rows_with_any_negative=int(np.any(margins<0,axis=1).sum()),per_joint_min=dict(zip(names,margins.min(axis=0).tolist())),
                per_joint_negative_rows=dict(zip(names,(margins<0).sum(axis=0).tolist())),meaning='Inherited 0.9 range soft margins; negative does not imply URDF hard penetration'),
            gripper_actual_final_m=dict(zip(names[-2:],controlled[-1]['q'][-2:])),
            telemetry_flags={k:sum(bool(r['telemetry'].get(k)) for r in controlled) for k in ('invalid_state','fall_flag','joint_limit_violation')})
        if all('installed_quaternion' in r for r in controlled):
            installed_rotation=np.array([quat_matrix(r['installed_quaternion']) for r in controlled])
            report['installed_orientation_tracking']=stats(orientation_error(np.array([r['rotation'] for r in controlled]),installed_rotation))
            report['installed_to_raw_orientation_rad']=stats(orientation_error(installed_rotation,quat_matrix(q[0])))
        event=result.get('event')
        if event and 'traj_timing_err' in event:
            rule=config['resolved']['wbc']['goal_reaching']['trajectory']
            threshold=rule['terminate_timing_frac']*event['traj_L']
            cause=[event['traj_d_lat']>rule['terminate_d_lat'],abs(event['traj_timing_err'])>threshold]
            assert cause==event['termination_cause'] and event['sim_time']>rule['terminate_grace_s']
            assert len(event['preview_pos'])==rule['preview_points']==9
            report['native_stop_readback']=dict(status='PASS_ORIGINAL_THRESHOLD_AND_CAUSE_READBACK',original_thresholds=rule,
                observed_timing_error_m=event['traj_timing_err'],timing_threshold_m=threshold,
                lateral_error_m=event['traj_d_lat'],recomputed_cause=cause,
                simulator_time_s=float(t[-1]),native_trajectory_clock_s=event['sim_time'],
                native_progress_m=event['traj_s'],scheduled_progress_m=event['traj_s_ref'],path_length_m=event['traj_L'],
                interpretation='Native spatial progress reached path end while timestamped schedule was near start; timing stopper fired after original grace. No threshold/gain/path change authorized.')
    if make_video:report['video']=video(directory,out,name)
    write(out/(name+'.json'),report)
    return report


def source(root,out):
    """One finite pass over only closed source episodes; four fixed conditions always remain."""
    from scipy.spatial.transform import Rotation,Slerp
    scene=root/'design/WBC_TASK_SCENE.json'
    assert sha(scene)=='bbd41f5cabc85920ec9d1c741aa3807b70554f09841ce26598011e75882ba139'
    card=json.loads((root/'design/SOURCE_PHYSICS_CARD.json').read_text());assert card['conditions']==[10000,10001,10002,10003]
    module_path=root/'code/source_control.py';assert sha(module_path)==card['source_code_sha256']['source_control.py']
    spec=importlib.util.spec_from_file_location('source_control_cpu',module_path);control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
    tool=np.eye(4);tool[:3,:3]=Rotation.from_euler('z',np.pi/4).as_matrix();tool[2,3]=-.1034
    rows=[]
    for condition in card['conditions']:
        candidates=list((root/'source/episodes'/('PI0280-'+str(condition))).glob('attempt-*/RESULT.json'))
        closed=[p for p in candidates if json.loads(p.read_text()).get('condition')==condition]
        if not closed:
            rows.append(dict(condition=condition,status='NOT_YET_CLOSED',persistent_success=None));continue
        assert len(closed)==1,'Retain valid denominators; do not select among attempts'
        directory=closed[0].parent;result=json.loads(closed[0].read_text())
        queries=json.loads((directory/'QUERIES.json').read_text())
        with np.load(directory/'physics.npz',allow_pickle=False) as data:physics={k:data[k] for k in data.files}
        with np.load(directory/'commands.npz',allow_pickle=False) as data:commands=data['action']
        with np.load(directory/'installed.npz',allow_pickle=False) as data:installed={k:data[k] for k in data.files}
        with np.load(directory/'initial.npz',allow_pickle=False) as data:base_pose=data['articulations/panda_wristcam'][0,:7]
        base=np.eye(4);base[:3,3]=base_pose[:3];base[:3,:3]=quat_matrix(base_pose[[4,5,6,3]])
        n=len(commands);assert result['pre_roll_control_calls']==20 and n==result['control_calls']
        assert np.allclose(physics['time_s'],np.arange(len(physics['time_s']))*.01)
        predicate=physics['is_cubeA_on_cubeB']&physics['is_cubeA_static']&~physics['is_cubeA_grasped']
        assert np.array_equal(predicate,physics['success'])
        control_samples=np.arange(100,len(predicate),5);task_clock=physics['time_s'][control_samples]-1
        window=control_samples[(task_clock>=28-1e-7)&(task_clock<=30+1e-7)]
        complete=result['status']=='VALID_SOURCE_PHYSICS_COMPLETE' and n==660 and len(predicate)==3301
        persistent=bool(complete and len(window)==41 and predicate[window].all())
        first=np.flatnonzero(predicate[control_samples]);first_s=float(task_clock[first[0]]) if len(first) else None
        assert persistent==result['persistent_success'] and first_s==result['first_success_s']
        actual=physics['tcp'][np.arange(1,n+1)*5];command_rotation=commands[:,3:9].reshape(n,2,3)
        command_rotation=np.concatenate([command_rotation,np.cross(command_rotation[:,0],command_rotation[:,1])[:,None]],axis=1)
        assert actual.shape==installed['installed_target_fk'].shape
        raw_position_errors=[];raw_rotation_errors=[];raw_gripper_errors=[];query_files=[];quantization=[];horizon_position_errors=[];horizon_rotation_errors=[]
        for query in queries:
            d=directory/('q%03d'%query['index']);assert sha(d/'input.npz')==query['input_sha256'] and sha(d/'output.npz')==query['output_sha256']
            with np.load(d/'input.npz',allow_pickle=False) as data:
                assert np.allclose(data['history_times_s'],[query['control']*.05-1,query['control']*.05])
            with np.load(d/'canonical.npz',allow_pickle=False) as data:canonical={k:data[k] for k in data.files}
            with np.load(d/'output.npz',allow_pickle=False) as data:
                eef=data['decoded_eef_9d'][0];closure=data['decoded_gripper_position'][0,:,0];assert data['decoded_joint_position'].shape==(1,40,7)
            assert eef.shape==(40,9) and closure.shape==(40,)
            assert np.allclose(canonical['timestamp'],query['control']*.05+np.arange(40)/15)
            assert np.allclose(canonical['gripper'],1-closure)
            # Source native output has the recorded Panda link8 tool convention; no second relative transform.
            native=np.tile(np.eye(4),(40,1,1));native[:,:3,3]=eef[:,:3];native[:,:3,:3]=control.decode(eef[:,3:9]);converted=base@native@np.linalg.inv(tool)
            horizon_position_errors.extend(np.linalg.norm(converted[:,:3,3]-canonical['position'],axis=1).tolist())
            horizon_rotation_errors.extend(orientation_error(converted[:,:3,:3],Rotation.from_quat(canonical['quaternion']).as_matrix()).tolist())
            first8=canonical['timestamp'][:8]
            if 'replan_tick' not in query:continue
            until=min(660,20+((query['index']+1)*32+2)//3);assert until==query['replan_tick']
            assert np.allclose(first8,query['executed_first8_native_timestamps_s'])
            ticks=np.arange(query['control'],until);clock=ticks*.05
            expected=np.stack([np.interp(clock,first8,canonical['position'][:8,j]) for j in range(3)],axis=1)
            expected_rotation=Slerp(first8,Rotation.from_quat(canonical['quaternion'][:8]))(np.minimum(clock,first8[-1])).as_matrix()
            j=np.clip(np.searchsorted(first8,clock,side='right')-1,0,7)
            raw_position_errors.extend(np.linalg.norm(commands[ticks,:3]-expected,axis=1).tolist())
            raw_rotation_errors.extend(orientation_error(command_rotation[ticks],expected_rotation).tolist())
            raw_gripper_errors.extend(np.abs(commands[ticks,-1]-canonical['gripper'][j]).tolist())
            quantization.append((until-20)*.05-min(32.,(query['index']+1)*8/15))
            query_files.extend(file_row(d/name) for name in ('input.npz','output.npz','canonical.npz','GATE.json'))
        assert not raw_position_errors or max(raw_position_errors)<1e-8
        assert not raw_rotation_errors or max(raw_rotation_errors)<1e-6
        assert not raw_gripper_errors or max(raw_gripper_errors)<1e-8
        assert not horizon_position_errors or max(horizon_position_errors)<1e-6
        assert not horizon_rotation_errors or max(horizon_rotation_errors)<1e-6
        metric=dict(condition=condition,status=result['status'],domain='OOD_PANDA_SIM_NOT_OFFICIAL_DROID_BENCHMARK',full_32s=complete,
            original_result=file_row(closed[0]),task_control_calls=n-20,preroll_control_calls=20,physics_samples=len(predicate),queries=len(queries),
            original_object_predicate_exact=True,first_success_s=first_s,persistent_success=persistent,
            success_samples_28_30=int(predicate[window].sum()),required_window_samples=41,observed_window_samples=len(window),window_task_clock_s=[28,30],window_simulator_clock_s=[29,31],
            transient_success_control_samples=int(predicate[control_samples].sum()),
            on_top_28_30=int(physics['is_cubeA_on_cubeB'][window].sum()),static_28_30=int(physics['is_cubeA_static'][window].sum()),released_28_30=int((~physics['is_cubeA_grasped'][window]).sum()),
            descriptive_10Hz_window_samples=len(window[::2]),descriptive_10Hz_success_samples=int(predicate[window[::2]].sum()),
            raw_H40_to_canonical_position_m=stats(horizon_position_errors) if horizon_position_errors else 'UNOBSERVED',
            raw_H40_to_canonical_rotation_rad=stats(horizon_rotation_errors) if horizon_rotation_errors else 'UNOBSERVED',
            raw_first8_to_command_position_m=stats(raw_position_errors) if raw_position_errors else 'UNOBSERVED_NO_INSTALLED_PREFIX',
            raw_first8_to_command_rotation_rad=stats(raw_rotation_errors) if raw_rotation_errors else 'UNOBSERVED_NO_INSTALLED_PREFIX',
            raw_first8_to_command_gripper_native_fraction_max=max(raw_gripper_errors,default=0),
            time_grid_quantization_s=stats(quantization) if quantization else 'UNOBSERVED',
            command_to_installed_IK_position_m=stats(np.linalg.norm(commands[20:,:3]-installed['installed_target_fk'][20:,:3,3],axis=1)) if n>20 else 'UNOBSERVED',
            command_to_actual_endpoint_position_m=stats(np.linalg.norm(commands[20:,:3]-actual[20:,:3,3],axis=1)) if n>20 else 'UNOBSERVED',
            installed_IK_to_actual_endpoint_position_m=stats(np.linalg.norm(installed['installed_target_fk'][20:,:3,3]-actual[20:,:3,3],axis=1)) if n>20 else 'UNOBSERVED',
            command_to_actual_endpoint_rotation_rad=stats(orientation_error(actual[20:,:3,:3],command_rotation[20:])) if n>20 else 'UNOBSERVED',
            pairwise_finger_force_max_N={k:float(np.linalg.norm(physics[k],axis=1).max()) for k in ('left_force','right_force')},
            ik_hold_count=int(result['ik_holds']),new_policy_calls=0,new_physics_steps=0,new_training_updates=0,
            raw_inputs=[file_row(directory/name) for name in ('RESULT.json','QUERIES.json','physics.npz','commands.npz','installed.npz','NATIVE_INVENTORY.json','INPUT_CONTRACT_AUDIT.json')],query_files=query_files)
        write(out/('source-'+str(condition)+'.json'),metric);rows.append(metric)
    summary=dict(schema='PI0280_CPU_SOURCE_OBJECT_ATTRIBUTION_V1',observed_unix_s=time.time(),scene=file_row(scene),source_card=file_row(root/'design/SOURCE_PHYSICS_CARD.json'),
        fixed_denominator=4,closed_conditions=sum(r['status']!='NOT_YET_CLOSED' for r in rows),persistent_successes=sum(r['persistent_success'] is True for r in rows),
        full_panel_ready=all(r['status']!='NOT_YET_CLOSED' for r in rows),rows=rows,new_policy_calls=0,new_physics_steps=0,new_training_updates=0,selected='NONE',deployment='NONE')
    write(out/'SOURCE_SUMMARY.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0280'));parser.add_argument('--video',action='store_true');parser.add_argument('--source',action='store_true');parser.add_argument('--self-check',action='store_true');args=parser.parse_args()
    if args.self_check:
        assert np.allclose(quat_matrix([0,0,0,1]),np.eye(3))
        assert np.allclose(orientation_error(np.array([np.eye(3),quat_matrix([0,0,1,0])]),np.eye(3)),[0,np.pi])
        assert stats([3,4])['rms']==np.sqrt(12.5)
        print('PASS_CPU_ARITHMETIC_SELF_CHECK');return
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','Hide CUDA explicitly for this CPU-only reader'
    root=args.root;out=root/'attribution';out.mkdir(exist_ok=True)
    if args.source:source(root,out);return
    binding=json.loads((root/'evidence/WBC_BINDING.json').read_text())
    assert sha(root/'design/WBC_CONFIG.json')==binding['configuration_sha256']
    identity=[dict(**r,current_sha256=sha(r['path'])) for r in binding['files']]
    assert all(r['current_sha256']==r['sha256'] for r in identity),'Frozen WBC identity changed'
    ledger=[json.loads(s) for s in (root/'evidence/wbc_budget.jsonl').read_text().splitlines()]
    config=json.loads((root/'design/WBC_CONFIG.json').read_text())
    reports=[fixture(p,out,ledger,args.video,config) for p in sorted((root/'wbc').iterdir()) if (p/'RESULT.json').is_file()]
    summary=dict(schema='PI0280_CPU_RAW_ATTRIBUTION_V1',observed_unix_s=time.time(),implementation=file_row(Path(__file__)),
        status='COMPLETED_RAW_AUDITED_WITH_EXECUTION_OUTCOMES_RETAINED',binding=file_row(root/'evidence/WBC_BINDING.json'),
        configuration=file_row(root/'design/WBC_CONFIG.json'),frozen_WBC_identity_files=identity,
        completed_or_partial_fixtures=[r['fixture'] for r in reports],
        original_wbc_ledger=file_row(root/'evidence/wbc_budget.jsonl'),cumulative_wbc_ledger_counts=dict(collections.Counter(r['kind'] for r in ledger)),
        ledger_semantics='Reservation is written before invocation. Failed operations count; saved returned rows independently report completed telemetry. Existing ledgers untouched.',
        new_policy_calls=0,new_WBC_actor_calls=0,new_physics_steps=0,new_training_updates=0,selected='NONE',deployment='NONE')
    write(out/'SUMMARY.json',summary)
    blocked=any(r['fixture']=='vertical' and r['execution_status']=='VALID_NATIVE_TERMINATION' for r in reports)
    if blocked:write(out/'TARGET_TASK_DENOMINATORS.json',dict(status='BLOCKED_NATIVE_EXECUTOR_QUALIFICATION',rows=[dict(condition=c,lane=lane,status='NOT_EXECUTED',quality_result=None) for lane in ('WBC_EXPERT_REFERENCE','PRETRAINED_REFERENCE_REPLAY_WITH_WBC') for c in (10000,10001,10002,10003)],quality_denominator_executed=0,planned_rows=8))
    lines=['| Attempt | Status | Saved control ticks | Installed tracking final/max (mm) | Raw orientation final/max (rad) | Hard penetration rows |','|---|---|---:|---:|---:|---:|']
    for r in reports:
        position=r.get('actual_to_same_tick_installed_position_m',{});rotation=r.get('actual_to_raw_fixed_orientation_rad',{})
        lines.append('| %s | %s | %d | %.3f / %.3f | %.4f / %.4f | %s |'%(r['attempt'],r['execution_status'],r['saved_controlled_rows'],position.get('final',0)*1000,position.get('max',0)*1000,rotation.get('final',0),rotation.get('max',0),r.get('hard_penetration',{}).get('rows_with_any','UNOBSERVED')))
    (out/'FIXTURES.md').write_text('\n'.join(lines)+'\n\nFixtures do not test object success. Same-tick installed tracking and timestamped raw-reference tracking use different reference clocks. Native installed orientation is UNOBSERVED.\n')
    print(json.dumps(dict(summary=str(out/'SUMMARY.json'),fixtures=[{k:r[k] for k in ('fixture','execution_status','saved_controlled_rows','actual_to_same_tick_installed_position_m','actual_to_raw_fixed_orientation_rad','hard_penetration','inherited_soft_margins')} for r in reports])))


if __name__=='__main__':main()
