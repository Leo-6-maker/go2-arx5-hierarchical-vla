"""PI0281 finite CPU audit of sealed outputs. No model, controller, physics or rendering."""
import argparse,ast,collections,hashlib,importlib.util,json,os,sys,time,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

RELEASE='360b412f55be8aec1568054d7d039ce7de2bc597'
ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0281')
OLD=ROOT.parent/'PI0280'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def identity(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))

def save(name,v):
    p=ROOT/'audit'/name;p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        history=p.parent/'history';history.mkdir(exist_ok=True);(history/(p.stem+'-'+sha(p)+p.suffix)).write_bytes(p.read_bytes())
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def stats(a):
    a=np.asarray(a,dtype=float).reshape(-1);assert len(a) and np.isfinite(a).all()
    return dict(n=len(a),mean=float(a.mean()),rms=float(np.sqrt(np.mean(a*a))),p50=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(a.max()))

def angle(a,b):return Rotation.from_matrix(np.swapaxes(a,-1,-2)@b).magnitude()

def load_cpu():
    sys.path.insert(0,str(ROOT.parent/'PI0250/runtime/reference-317a/src'))
    from go2arx5_vla.wbc.native.kinematics import rpy_to_matrix,axis_angle_to_matrix,homogeneous
    from go2arx5_vla.contracts import TrajectoryChunkV1
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    source=ROOT.parent/'PI0265/code/pi0258_geometry.py';node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='full_fk')
    scope=dict(np=np,rpy_to_matrix=rpy_to_matrix,axis_angle_to_matrix=axis_angle_to_matrix,homogeneous=homogeneous)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
    urdf_path=ROOT.parent/'PI0276/ManiSkill/mani_skill/assets/robots/panda/panda_v3.urdf';urdf=ET.parse(urdf_path).getroot()
    contract=json.loads((OLD/'evidence/SOURCE_TCP_CONTRACT.json').read_text());assert sha(urdf_path)==contract['URDF_sha256'] and sha(source)==contract['FK_source_sha256']
    by_child={e.find('child').attrib['link']:e for e in urdf.findall('joint')};chain=[];link='panda_hand_tcp'
    while link in by_child:
        e=by_child[link];chain.append(e);link=e.find('parent').attrib['link']
    chain=chain[::-1]
    def fk(q):
        q=np.asarray(q);poses=np.tile(np.eye(4),(len(q),1,1));positions={f'panda_joint{i+1}':q[:,i] for i in range(7)}
        for e in chain:
            o=e.find('origin');rpy=np.fromstring(o.attrib.get('rpy','0 0 0'),sep=' ');xyz=np.fromstring(o.attrib.get('xyz','0 0 0'),sep=' ')
            poses=poses@homogeneous(rpy_to_matrix(rpy),xyz)
            if e.attrib['type'] in ('revolute','continuous'):
                axis=np.fromstring(e.find('axis').attrib['xyz'],sep=' ');step=np.tile(np.eye(4),(len(q),1,1));step[:,:3,:3]=Rotation.from_rotvec(positions[e.attrib['name']][:,None]*axis).as_matrix();poses=poses@step
        return poses
    check=np.array([[0,.4,0,-1.9,0,2.3,-.8]])
    scalar=scope['full_fk'](urdf,{f'panda_joint{i+1}':x for i,x in enumerate(check[0])},np.eye(4))['panda_hand_tcp']
    assert np.max(abs(fk(check)[0]-scalar))<1e-12
    spec=importlib.util.spec_from_file_location('source_control_cpu',OLD/'code/source_control.py');control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control);control.self_check()
    tool=np.eye(4);tool[:3,:3]=Rotation.from_euler('z',np.pi/4).as_matrix();tool[2,3]=-.1034
    assert np.max(abs(tool-np.array(contract['first_sample_tool_transform'])))<1e-6
    return fk,control,tool,TrajectoryChunkV1,TrajectorySafetyGateV1(),[identity(p) for p in (urdf_path,source,OLD/'code/source_control.py',OLD/'evidence/SOURCE_TCP_CONTRACT.json',OLD/'official/gr00t/data/state_action/droid_frame.py')]

def gate(chunk_type,gate,poses,closure,t,state,id):
    c=chunk_type(chunk_id=id,source='PI0281_SAVED_CPU_ONLY',observation_timestamp=t[0],point_timestamps=t,
        eef_position=poses[:,:3,3],eef_orientation=Rotation.from_matrix(poses[:,:3,:3]).as_quat(),
        source_frame='PI0276_MANISKILL_WORLD',tcp_semantic_id='PI0276_PANDA_HAND_TCP',validity_start=t[0],validity_end=t[-1],gripper=1-closure)
    return gate.evaluate(c,now=t[0],achieved_position=state[:3,3],achieved_orientation=Rotation.from_matrix(state[:3,:3]).as_quat()).as_dict()

def package(path,inp,fk,control,tool,chunk_type,posegate,id,lane,state_pose=None,base=None,now=0.):
    with np.load(path,allow_pickle=False) as v:
        eef=v['decoded_eef_9d'][0];q=v['decoded_joint_position'][0];closure=v['decoded_gripper_position'][0,:,0];noise=v['noise_uint16'].copy()
        assert eef.shape==(40,9) and q.shape==(40,7) and closure.shape==(40,) and noise.shape==(1,40,132)
    assert all(np.isfinite(v).all() for v in (eef,q,closure))
    direct=np.tile(np.eye(4),(40,1,1));direct[:,:3,:3]=control.decode(eef[:,3:9]);direct[:,:3,3]=eef[:,:3]
    joint=fk(q)@tool
    xyz=np.linalg.norm(direct[:,:3,3]-joint[:,:3,3],axis=1);rot=angle(direct[:,:3,:3],joint[:,:3,:3])
    with np.load(inp,allow_pickle=False) as v:state=v['state'];history={k:v[k].tolist() for k in v.files if k.startswith('actual_history') or k=='history_times_s'}
    state_direct=np.eye(4);state_direct[:3,:3]=control.decode(state[3:9]);state_direct[:3,3]=state[:3]
    state_fk=fk(state[10:17][None])[0]@tool
    base=np.eye(4) if base is None else base
    if state_pose is None:state_pose=base@state_direct@np.linalg.inv(tool)
    gates={}
    for label,poses in [('DIRECT_EEF',direct),('FK_DECODED_ABSOLUTE_Q',joint)]:
        world=base@poses@np.linalg.inv(tool)
        gates[label]=gate(chunk_type,posegate,world,closure,now+np.arange(40)/15,state_pose,id+'-'+label)
        gates[label+'_FIRST8']=gate(chunk_type,posegate,world[:8],closure[:8],now+np.arange(8)/15,state_pose,id+'-'+label+'-FIRST8')
    return dict(id=id,lane=lane,input=identity(inp),output=identity(path),actual_noise_array_sha256=hashlib.sha256(noise.tobytes()).hexdigest(),
        noise_dtype=str(noise.dtype),noise_shape=list(noise.shape),history=history,
        first8=dict(position_m=stats(xyz[:8]),rotation_rad=stats(rot[:8])),full40=dict(position_m=stats(xyz),rotation_rad=stats(rot)),
        horizon_position_m=xyz.tolist(),horizon_rotation_rad=rot.tolist(),shared_gripper_closure=stats(closure),
        gripper_route_difference=0.,gripper_route_difference_semantics='Both readouts use the same stock gripper output; no separate gripper modality exists for FK(q)',
        unchanged_gates=gates,state_geometry=dict(position_m=float(np.linalg.norm(state_direct[:3,3]-state_fk[:3,3])),rotation_rad=float(angle(state_direct[:3,:3],state_fk[:3,:3]))))

def audit_old():
    fk,control,tool,chunk_type,posegate,deps=load_cpu();rows=[];funnel=[];input_errors=[];native=json.loads((OLD/'design/NATIVE_CARD.json').read_text())
    for row in native['rows']:
        assert sha(Path(row['input_path']))==row['input_sha256']
        v=package(OLD/'native/outputs'/row['id']/'stock.npz',Path(row['input_path']),fk,control,tool,chunk_type,posegate,row['id'],'A2_OFFICIAL_DROID',now=row['frame']/15)
        assert v['history']['actual_history_frame_indices']==[row['frame']-15,row['frame']]
        assert np.allclose(v['history']['actual_history_timestamps_s'],[(row['frame']-15)/15,row['frame']/15]);rows.append(v)
    for condition in (10000,10001,10002,10003):
        d=OLD/'source/episodes'/('PI0280-'+str(condition))/'attempt-000';result=json.loads((d/'RESULT.json').read_text());assert result['status']=='VALID_SOURCE_PHYSICS_COMPLETE'
        queries=json.loads((d/'QUERIES.json').read_text());assert len(queries)==60
        with np.load(d/'physics.npz',allow_pickle=False) as v:physics={k:v[k] for k in v.files}
        with np.load(d/'initial.npz',allow_pickle=False) as v:pose=v['articulations/panda_wristcam'][0,:7]
        base=np.eye(4);base[:3,:3]=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix();base[:3,3]=pose[:3]
        with np.load(d/'observations.npz',allow_pickle=False) as observations:
            cameras={k:observations[k+'_raw'] for k in ('base_camera','hand_camera')}
            with np.load(d/'commands.npz',allow_pickle=False) as commands:actions=commands['action'].copy()
            for query in queries:
                index=query['control'];inp=Path(query['input']);output=Path(query['output']);assert sha(inp)==query['input_sha256'] and sha(output)==query['output_sha256']
                row=package(output,inp,fk,control,tool,chunk_type,posegate,str(condition)+'-q%03d'%query['index'],'A3_SOURCE_PANDA',physics['tcp'][index*5],base,index*.05)
                row.update(condition=condition,task_time_s=query['time_s'],query_index=query['index']);rows.append(row)
                with np.load(inp,allow_pickle=False) as v:
                    assert np.allclose(v['history_times_s'],[(index-20)*.05,index*.05])
                    assert np.array_equal(v['exterior_image_1_left'],cameras['base_camera'][[index-20,index]])
                    assert np.array_equal(v['wrist_image_left'],cameras['hand_camera'][[index-20,index]])
                    state=v['state'];true_pose=np.linalg.inv(base)@physics['tcp'][index*5]@tool
                    expected=np.r_[true_pose[:3,3],control.encode(true_pose[:3,:3]),1-(physics['qpos'][index*5,-2:].sum()-.00004)/.07996,physics['qpos'][index*5,:7]]
                    input_errors.append(float(np.max(abs(state-expected))))
                assert query['replan_tick']==control.next_tick(query['index'])
                with np.load(output,allow_pickle=False) as v:closure=v['decoded_gripper_position'][0,:,0]
                with np.load(output.with_name('canonical.npz'),allow_pickle=False) as v:
                    ts=v['timestamp'];positions=v['position'];g=v['gripper'];assert np.allclose(ts,index*.05+np.arange(40)/15)
                    ticks=np.arange(index,query['replan_tick']);clock=ticks*.05
                    assert np.max(abs(actions[ticks,:3]-np.stack([np.interp(clock,ts[:8],positions[:8,j]) for j in range(3)],axis=1)))<1e-8
                    j=np.clip(np.searchsorted(ts[:8],clock,side='right')-1,0,7)
                    assert np.array_equal(g,1-closure) and np.max(abs(actions[ticks,-1]-g[j]))<1e-8
        indices=np.arange(100,3301);t=physics['time_s'][indices]-1;grasp=physics['is_cubeA_grasped'][indices];top=physics['is_cubeA_on_cubeB'][indices];static=physics['is_cubeA_static'][indices];success=top&static&~grasp
        assert np.array_equal(success,physics['success'][indices]);z=physics['cubeA_pose'][indices,2];initial=z[0]
        def first(a):return float(t[np.flatnonzero(a)[0]]) if a.any() else None
        longest=max((sum(1 for _ in g) for flag,g in __import__('itertools').groupby(grasp) if flag),default=0)
        window=np.arange(2900,3101,5)
        funnel.append(dict(condition=condition,old_lane='PI0280_DIRECT_EEF',first_grasp_s=first(grasp),grasp_physics_samples=int(grasp.sum()),longest_grasp_s=longest*.01,
            original_lift_predicate='UNOBSERVED_NOT_RECORDED',initial_cubeA_z_m=float(initial),max_cubeA_z_m=float(z.max()),max_height_increase_m=float(z.max()-initial),
            descriptive_lift_2cm_above_task_initial_s=first(z>initial+.02),descriptive_lift_rule='Height increase >0.02m; diagnostic threshold, not original success gate',
            first_on_top_s=first(top),first_native_success_s=first(success),persistent_success=bool(success[window-100].all()),
            success_41_count=int(physics['success'][window].sum()),released_41_count=int((~physics['is_cubeA_grasped'][window]).sum()),
            released_after_any_grasp=bool(grasp.any() and (~grasp[np.flatnonzero(grasp)[0]+1:]).any()),
            max_pairwise_finger_force_N={k:float(np.linalg.norm(physics[k],axis=1).max()) for k in ('left_force','right_force')},raw_files=[identity(d/name) for name in ('RESULT.json','physics.npz','commands.npz','installed.npz','QUERIES.json')]))
    assert len(rows)==272 and len(input_errors)==240 and max(input_errors)<1e-5
    groups=[]
    for name,group in [('A2_OFFICIAL_DROID',[r for r in rows if r['lane']=='A2_OFFICIAL_DROID'])]+[(str(c),[r for r in rows if r.get('condition')==c]) for c in (10000,10001,10002,10003)]:
        groups.append(dict(group=name,packages=len(group),first8_position_m=stats([x for r in group for x in r['horizon_position_m'][:8]]),full40_position_m=stats([x for r in group for x in r['horizon_position_m']]),
            first8_rotation_rad=stats([x for r in group for x in r['horizon_rotation_rad'][:8]]),full40_rotation_rad=stats([x for r in group for x in r['horizon_rotation_rad']]),
            full40_by_horizon_position_mean_m=np.mean([r['horizon_position_m'] for r in group],axis=0).tolist(),
            gate_distribution={route:dict(collections.Counter(r['unchanged_gates'][route]['decision'] for r in group)) for route in ('DIRECT_EEF','FK_DECODED_ABSOLUTE_Q','DIRECT_EEF_FIRST8','FK_DECODED_ABSOLUTE_Q_FIRST8')},
            time_phase_bins=[dict(task_window_s=[lo,hi],packages=len(part),first8_position_m=stats([x for r in part for x in r['horizon_position_m'][:8]]),first8_rotation_rad=stats([x for r in part for x in r['horizon_rotation_rad'][:8]]))
                for lo,hi in ((0,8),(8,16),(16,24),(24,32)) for part in [[r for r in group if lo<=r.get('task_time_s',-1)<hi]] if part]))
    report=dict(schema='PI0281_INDEPENDENT_SAVED_DUAL_READOUT_V1',release=RELEASE,observed_unix_s=time.time(),status='PASS_CPU_SAVED_ABI_AND_COMMAND_AUDIT',
        package_count=272,A2_unique=32,A3_unique=240,groups=groups,rows=rows,old_object_funnel=funnel,input_max_abs_error=max(input_errors),
        definitions=dict(position='Euclidean meters same Panda/DROID frame',rotation='SO3 geodesic radians same TCP',joint='Stock decoded absolute seven Panda angles radians, never achieved/postIK targets',
            gripper='Stock DROID closure0open1closed; shared across both readouts',history='A2 indices[-15,0] at15Hz; A3 physicalcontrol[-20,0] at20Hz =1s truehistory',
            execution='8 native15Hz timestamps,20Hz interpolation through7/15 then hold until integerceil next replan; no speed scaling'),
        state_geometry_max_m=max(r['state_geometry']['position_m'] for r in rows),state_geometry_max_rad=max(r['state_geometry']['rotation_rad'] for r in rows),
        dependencies=deps+[identity(OLD/'design/BASELINE_BINDING.json'),identity(OLD/'design/NATIVE_CARD.json')],implementation=identity(Path(__file__)),new_model_calls=0,new_actor_calls=0,new_physics_steps=0,new_training_updates=0,selected='NONE',deployment='NONE')
    save('SAVED_DUAL_READOUT.json',report);save('OLD_OBJECT_FUNNEL.json',dict(rows=funnel,original_lift='UNOBSERVED',new_physics_steps=0))
    lines=['| Package group | N | XYZ mean/p95 full40 (m) | SO3 mean/p95 full40 (rad) |','|---|---:|---:|---:|']
    for g in groups:lines.append('| %s | %d | %.6f / %.6f | %.6f / %.6f |'%(g['group'],g['packages'],g['full40_position_m']['mean'],g['full40_position_m']['p95'],g['full40_rotation_rad']['mean'],g['full40_rotation_rad']['p95']))
    (ROOT/'audit/SAVED_DUAL_READOUT.md').write_text('\n'.join(lines)+'\n\nReadout mismatch is descriptive, not automatically a decode defect. Both joint and EEF are independently predicted. Original lift predicate is UNOBSERVED; measured height is retained. All272 existing output packages, no new neural or physics calls.\n')
    print(json.dumps(dict(groups=groups,input_max_abs_error=max(input_errors),funnel=funnel)))


def audit_new(only=None):
    """Finite closed-result snapshot. Unexecuted fixed slots stay null; existing receipts are versioned."""
    fk,control,tool,chunk_type,posegate,deps=load_cpu();source=[];wbc=[]
    from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix,matrix_to_quat
    lanes=('ORIGINAL_BASE_JOINT','FK_OF_JOINT_COMMAND_REPLAY','EXPERT_DROID_BRIDGE_ROUNDTRIP')
    prior_file=ROOT/'audit/NEW_RESULTS_SNAPSHOT.json'
    prior=json.loads(prior_file.read_text()) if prior_file.exists() else {}
    prior_source={(r['lane'],r['condition']):r for r in prior.get('source',[])}
    for lane in lanes:
        for condition in (10000,10001,10002,10003):
            if only and (lane,condition) not in only:
                source.append(prior_source.get((lane,condition),dict(lane=lane,condition=condition,status='NOT_YET_AUDITED',quality=None)));continue
            candidates=list((ROOT/'source/episodes'/(lane+'-'+str(condition))).glob('attempt-*/RESULT.json'))
            if not candidates:
                source.append(dict(lane=lane,condition=condition,status='NOT_YET_CLOSED',quality=None));continue
            assert len(candidates)==1,'Do not choose among valid attempts'
            d=candidates[0].parent;result=json.loads(candidates[0].read_text());report=dict(lane=lane,condition=condition,status=result['status'],result=identity(candidates[0]),quality=None)
            if (d/'physics.npz').exists():
                with np.load(d/'physics.npz',allow_pickle=False) as data:physics={k:data[k] for k in data.files}
                predicate=physics['is_cubeA_on_cubeB']&physics['is_cubeA_static']&~physics['is_cubeA_grasped'];assert np.array_equal(predicate,physics['success'])
                window=np.arange(2900,3101,5);window=window[window<len(predicate)]
                full=len(predicate)==3301 and result['control_calls']==660 and result['status']=='VALID_SOURCE_COMPLETE'
                persistent=bool(full and len(window)==41 and predicate[window].all())
                task_samples=np.arange(100,len(predicate),5);first=np.flatnonzero(predicate[task_samples]);first_s=float((task_samples[first[0]]-100)*.01) if len(first) else None
                assert persistent==result['persistent_success'] and first_s==result['first_success_s']
                report.update(full32s=full,physics_samples=len(predicate),success41=int(predicate[window].sum()),window_samples=len(window),persistent=persistent,quality=persistent,
                    persistent41_observed=bool(predicate[window].all()) if len(window)==41 else None,
                    persistent41_status='OBSERVED_ALL41' if len(window)==41 else 'UNOBSERVED_EARLY_TERMINATION',
                    task_elapsed_s=max(0,(len(predicate)-1)*.01-1),first_success_s=first_s,
                    original_lift='UNOBSERVED_NOT_RECORDED',grasp_physics_samples=int(physics['is_cubeA_grasped'][100:].sum()),max_cubeA_height_m=float(physics['cubeA_pose'][100:,2].max()) if len(predicate)>100 else None)
                if len(predicate)>100:
                    t=physics['time_s'][100:]-1;grasp=physics['is_cubeA_grasped'][100:];z=physics['cubeA_pose'][100:,2];top=physics['is_cubeA_on_cubeB'][100:]
                    def first_event(values):return float(t[np.flatnonzero(values)[0]]) if values.any() else None
                    report['object_funnel']=dict(first_grasp_s=first_event(grasp),grasp_samples=int(grasp.sum()),
                        longest_grasp_s=max((sum(1 for _ in g) for flag,g in __import__('itertools').groupby(grasp) if flag),default=0)*.01,
                        task_initial_height_m=float(z[0]),max_height_increase_m=float(z.max()-z[0]),
                        original_lift='UNOBSERVED_NOT_RECORDED',descriptive_lift_2cm_s=first_event(z>z[0]+.02),
                        descriptive_lift_rule='Measured height increase >2cm; diagnostic only',first_on_top_s=first_event(top),
                        on_top_samples=int(top.sum()),first_success_physics_s=first_event(predicate[100:]),
                        released_41=int((~physics['is_cubeA_grasped'][window]).sum()) if len(window)==41 else None,
                        static_41=int(physics['is_cubeA_static'][window].sum()) if len(window)==41 else None,
                        on_top_41=int(physics['is_cubeA_on_cubeB'][window].sum()) if len(window)==41 else None,
                        release_after_first_grasp_s=first_event((~grasp)&(np.arange(len(grasp))>np.flatnonzero(grasp)[0])) if grasp.any() else None)
                with np.load(d/'commands.npz',allow_pickle=False) as data:command=data['action']
                with np.load(d/'installed.npz',allow_pickle=False) as data:installed={k:data[k] for k in data.files}
                n=min(len(command),len(installed['action']));end=physics['tcp'][np.arange(1,n+1)*5]
                report['command_to_actual_xyz_m']=stats(np.linalg.norm(command[:n,:3]-end[:,:3,3],axis=1))
                report['installedFK_to_actual_xyz_m']=stats(np.linalg.norm(installed['installed_target_fk'][:n,:3,3]-end[:,:3,3],axis=1))
                urdf=ET.parse(ROOT.parent/'PI0276/ManiSkill/mani_skill/assets/robots/panda/panda_v3.urdf').getroot()
                jointnames=[f'panda_joint{i+1}' for i in range(7)];limits=np.array([[float(urdf.find("joint[@name='%s']/limit"%j).attrib[k]) for k in ('lower','upper')] for j in jointnames])
                actual_margin=np.minimum(physics['qpos'][:,:7]-limits[:,0],limits[:,1]-physics['qpos'][:,:7])
                penetration=max(0.,float(-actual_margin.min()))
                assert np.allclose(actual_margin.min(axis=0),result['minimum_actual_joint_hard_margin_rad'],atol=1e-7)
                assert abs(penetration-result['maximum_actual_joint_penetration_rad'])<1e-7
                report.update(actual_joint_hard_margin_min_rad=actual_margin.min(axis=0).tolist(),actual_hard_penetration_rad=penetration,
                    source_recorded_actual_hard_penetration_rad=result['maximum_actual_joint_penetration_rad'],
                    source_recorded_joint_hard_margin_min_rad=result['minimum_actual_joint_hard_margin_rad'],
                    urdf_vs_native_limit_rounding_penetration_difference_rad=penetration-result['maximum_actual_joint_penetration_rad'],
                    actual_hard_penetration_samples=int((actual_margin.min(axis=1)<0).sum()),
                    actual_hard_penetration_definition='Exact saved physical qpos versus actual Panda URDF seven arm limits; raw penetration retained in radians',
                    source_safety_semantics='Joint route rejects nonfinite or out-of-limit commanded q before dispatch. EEF replay uses unchanged commanded-pose gate; finish records actual arm penetration descriptively, without a new post-step penetration rejection. WBC prospective 1e-6 tolerance is not substituted for a source task gate.',
                    source_executor_identity=identity(ROOT/'code/source.py'))
                install_margin=np.minimum(installed['arm_target'][:,:7]-limits[:,0],limits[:,1]-installed['arm_target'][:,:7])
                report['installed_arm_target_hard_penetration_rad']=max(0.,float(-install_margin.min()))
                if install_margin.min()<0:
                    i,k=np.unravel_index(np.argmin(install_margin),install_margin.shape)
                    report['installed_arm_target_hard_penetration_peak']=dict(control_tick=int(i),task_time_s=float(i*.05-1),joint=jointnames[k],installed_q_rad=float(installed['arm_target'][i,k]),limits_rad=limits[k].tolist(),margin_rad=float(install_margin[i,k]))
                report['safety_classification']=dict(RAW='Finite commands and route-specific unchanged pre-dispatch gates; rejected future commands are not achieved states',
                    INSTALL='Recorded actuator/controller arm_target limits audited separately from raw reference',
                    ACTUAL='Physical qpos penetration is retained even when native simulation completes',
                    TASK='Original object persistent predicate only; no retrospective substitution of WBC integration gate')
                if penetration:
                    i,k=np.unravel_index(np.argmin(actual_margin),actual_margin.shape)
                    report['actual_hard_penetration_peak']=dict(physics_index=int(i),task_time_s=float(physics['time_s'][i]-1),joint=jointnames[k],actual_q_rad=float(physics['qpos'][i,k]),limits_rad=limits[k].tolist(),margin_rad=float(actual_margin[i,k]))
                if lane=='ORIGINAL_BASE_JOINT':
                    with np.load(d/'initial.npz',allow_pickle=False) as data:base_pose=data['articulations/panda_wristcam'][0,:7]
                    base=np.eye(4);base[:3,:3]=Rotation.from_quat(base_pose[[4,5,6,3]]).as_matrix();base[:3,3]=base_pose[:3]
                    commanded_fk=base@fk(installed['action'][:n,:7])
                    report['recorded_command_to_exactFK_m']=stats(np.linalg.norm(command[:n,:3]-commanded_fk[:,:3,3],axis=1))
                    assert report['recorded_command_to_exactFK_m']['max']<1e-5
                    queries=json.loads((d/'QUERIES.json').read_text());noise_equal=[];route_errors=[];new_packages=[];closure_errors=[];rejected=[]
                    with np.load(d/'observations.npz',allow_pickle=False) as observations:
                        cameras={k:observations[k+'_raw'] for k in ('base_camera','hand_camera')}
                        for query in queries:
                            qfolder=d/('q%03d'%query['index']);parent=OLD/'source/episodes'/('PI0280-'+str(condition))/'attempt-000'/('q%03d'%query['index'])/'output.npz'
                            assert sha(qfolder/'input.npz')==query['input_sha256'] and sha(qfolder/'output.npz')==query['output_sha256'] and sha(parent)==query['noise_parent_sha256']
                            with np.load(qfolder/'input.npz',allow_pickle=False) as v:
                                controltick=query['control'];assert np.allclose(v['history_times_s'],[(controltick-20)*.05,controltick*.05])
                                assert np.array_equal(v['exterior_image_1_left'],cameras['base_camera'][[controltick-20,controltick]])
                                assert np.array_equal(v['wrist_image_left'],cameras['hand_camera'][[controltick-20,controltick]])
                            with np.load(qfolder/'output.npz',allow_pickle=False) as a,np.load(parent,allow_pickle=False) as b:
                                noise_equal.append(np.array_equal(a['noise_uint16'],b['noise_uint16']));q=a['decoded_joint_position'][0];closure=a['decoded_gripper_position'][0,:,0]
                            new_packages.append(package(qfolder/'output.npz',qfolder/'input.npz',fk,control,tool,chunk_type,posegate,lane+'-'+str(condition)+'-q%03d'%query['index'],'PI0281_NATIVE_JOINT_AUDIT',physics['tcp'][controltick*5],base,controltick*.05))
                            ticks=np.arange(controltick,min(n,control.next_tick(query['index'])));j=np.clip(np.searchsorted(controltick*.05+np.arange(8)/15,ticks*.05,side='right')-1,0,7)
                            if len(ticks):
                                assert np.array_equal(installed['action'][ticks,:7],q[j])
                                route_errors.extend(np.max(abs(installed['arm_target'][ticks]-q[j]),axis=1).tolist())
                                closure_errors.extend(np.abs(installed['action'][ticks,-1]-(2*np.clip(1-closure[j],0,1)-1)).tolist())
                            if result['status']=='VALID_JOINT_HARD_LIMIT_REJECT' and query==queries[-1]:
                                tick=n;j=min(7,max(0,int(np.searchsorted(controltick*.05+np.arange(8)/15,tick*.05,side='right')-1)))
                                margins=np.minimum(q[j]-limits[:,0],limits[:,1]-q[j]);assert margins.min()<0
                                rejected=[dict(joint=jointnames[k],command_rad=float(q[j,k]),limits_rad=limits[k].tolist(),margin_rad=float(margins[k])) for k in np.flatnonzero(margins<0)]
                                report['rejected_before_dispatch']=dict(control_tick=tick,task_time_s=tick*.05-1,query_index=query['index'],native_sample=j,violations=rejected)
                    assert all(noise_equal);report.update(noise_identical_to_PI0280=True,queries=len(queries))
                    report['rawQ_to_installed_joint_max_abs_rad']=max(route_errors,default=0);report['decoded_gripper_to_actuator_max_abs']=max(closure_errors,default=0)
                    assert report['rawQ_to_installed_joint_max_abs_rad']<1e-6 and report['decoded_gripper_to_actuator_max_abs']<1e-6
                    save('NEW_JOINT_READOUT-'+str(condition)+'.json',dict(rows=new_packages,new_model_calls=0,new_physics_steps=0))
                if (d/'canonical_replay.npz').exists():
                    with np.load(d/'canonical_replay.npz',allow_pickle=False) as data:replay=data['poses'];g=data['gripper']
                    offset=20 if lane=='EXPERT_DROID_BRIDGE_ROUNDTRIP' else 0
                    prefix=min(len(replay),n-offset);report['canonical_replay_to_command_m']=stats(np.linalg.norm(replay[:prefix,:3,3]-command[offset:offset+prefix,:3],axis=1))
                    assert report['canonical_replay_to_command_m']['max']<1e-6
                    gates=json.loads((d/'GATES.json').read_text());checked=[]
                    for entry in gates:
                        start=entry['control']-offset;poses=replay[start:start+40];assert len(poses)==40
                        now=entry['control']*.05;state=physics['tcp'][entry['control']*5]
                        gate_rotations=rot6d_rows_to_matrix(poses[:,:2,:3].reshape(-1,6))
                        state_rotation=rot6d_rows_to_matrix(state[None,:2,:3].reshape(1,6))
                        chunk=chunk_type(chunk_id=lane+'-'+str(condition),source='PI0276',observation_timestamp=now,point_timestamps=now+np.arange(40)*.05,
                            eef_position=poses[:,:3,3],eef_orientation=matrix_to_quat(gate_rotations),source_frame='PI0276_MANISKILL_WORLD',tcp_semantic_id='PI0276_PANDA_HAND_TCP',validity_start=now,validity_end=now+1.95,gripper=g[start:start+40])
                        verdict=posegate.evaluate(chunk,now=now,achieved_position=state[:3,3],achieved_orientation=matrix_to_quat(state_rotation)[0]).as_dict()
                        assert verdict['decision']==entry['result']['decision']
                        for speed in ('linear_speed_mps','angular_speed_radps'):
                            assert abs(verdict['metrics'][speed]['max']-entry['result']['metrics'][speed]['max'])<1e-6
                        if verdict['decision']=='REJECT':
                            angular=angle(poses[:-1,:3,:3],poses[1:,:3,:3])/.05;j=int(np.argmax(angular));linear=np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=1)/.05
                            detail=dict(control_tick=entry['control'],task_time_s=now-1,unchanged_issues=entry['result']['issues'],
                                preview_max_angular_radps=float(angular[j]),preview_max_linear_mps=float(linear.max()),angular_peak_pair_replay_ticks=[start+j,start+j+1],
                                pair_poses=poses[j:j+2].tolist(),meaning='Original gate looks ahead over40 stored20Hz commands; rejected preview suffix was not dispatched by this replay')
                            if lane=='FK_OF_JOINT_COMMAND_REPLAY':
                                parent=json.loads((d/'PARENT.json').read_text());pd=Path(parent['folder']);assert sha(pd/'RESULT.json')==parent['result_sha256']
                                with np.load(pd/'installed.npz',allow_pickle=False) as data:literal=data['action'][start+j:start+j+2,:7]
                                detail['literal_parent_commanded_q_pair_rad']=literal.tolist();detail['parent_installed_identity']=identity(pd/'installed.npz')
                            checked.append(detail)
                    report['replay_gate_readback']=dict(status='PASS_UNCHANGED_GATE_ARRAY_RECOMPUTATION',gate_calls=len(gates),decisions=dict(collections.Counter(v['result']['decision'] for v in gates)),rejects=checked,gate_identity=identity(d/'GATES.json'))
                report['raw_inputs']=[identity(d/name) for name in ('physics.npz','commands.npz','installed.npz')]
            save('SOURCE-'+lane+'-'+str(condition)+'.json',report);source.append(report)
    if only:wbc=prior.get('wbc',[])
    for p in ([] if only else (ROOT/'wbc').rglob('RESULT.json')):
        # Native lane schemas may evolve during instrumentation; only recorded same-TCP fields are computed.
        result=json.loads(p.read_text());d=p.parent;r=dict(path=str(d),status=result.get('status'),result=identity(p),quality=None)
        rowfile=d/'rows.json'
        if rowfile.exists():
            rows=json.loads(rowfile.read_text());r['saved_rows']=len(rows);r['rows_identity']=identity(rowfile)
            valid=[x for x in rows if 'TCP' in x and 'installed' in x]
            if valid:r['position_all_saved_m']=stats([np.linalg.norm(np.array(x['TCP'])-np.array(x['installed'])) for x in valid])
            last2=[x for x in valid if 8<=x.get('time',-1)-result.get('initialization_s',0)<=10+1e-6]
            if last2:r['position_final2s_m']=stats([np.linalg.norm(np.array(x['TCP'])-np.array(x['installed'])) for x in last2])
            rotations=[x for x in last2 if 'rotation' in x and 'installed_quaternion' in x]
            r['rotation_final2s_rad']=stats([float(angle(np.array(x['rotation']),Rotation.from_quat(x['installed_quaternion']).as_matrix())) for x in rotations]) if rotations else 'UNOBSERVED_SAME_TCP_ROTATION_NOT_RECORDED'
            r['full10s']=result.get('status')=='WBC_FIXTURE_EXECUTED' and abs(result.get('actual_elapsed_s',0)-10)<1e-5
            hard=[v for x in rows for v in x.get('hard_penetration',{}).values()]
            r['hard_penetration_max_SI']=max(hard) if hard else None
            margins=[v for x in rows for v in x.get('telemetry',{}).get('joint_limit_margin',[])]
            r['soft_margin_min_mixed_SI']=min(margins) if margins else None
            position=r.get('position_final2s_m');rotation=r['rotation_final2s_rad']
            r['prospective_fixture_gate']=bool(r['full10s'] and len(last2)>=99 and isinstance(rotation,dict) and position['rms']<=.020 and position['max']<=.040 and rotation['rms']<=.15 and rotation['max']<=.30 and hard and max(hard)<=1e-6)
            r['prospective_fixture_gate_semantics']='All10s plus final2s sameTCP position/SO3 and hardpenetration<=1e-6; soft negatives retained; original native stop wins'
            if result.get('event'):r['original_native_stop']=result['event']
            proxyrows=[x for x in valid if 'rotation' in x and 'installed_quaternion' in x and 'orientation_error_proxy' in x]
            if proxyrows:
                r['sameTCP_geodesic_vs_proxy_norm_abs_rad']=stats([abs(float(angle(np.array(x['rotation']),Rotation.from_quat(x['installed_quaternion']).as_matrix()))-np.linalg.norm(x['orientation_error_proxy'])) for x in proxyrows])
            r['quality']=r['prospective_fixture_gate']
        save('WBC-'+hashlib.sha256(str(d).encode()).hexdigest()[:12]+'.json',r);wbc.append(r)
    save('NEW_RESULTS_SNAPSHOT.json',dict(status='FINITE_PROVISIONAL_SNAPSHOT_NOT_TERMINAL',observed_unix_s=time.time(),source=source,wbc=wbc,new_model_calls=0,new_physics_steps=0,new_training_updates=0,selected='NONE',deployment='NONE'))
    save('GAP_MATRIX.json',dict(status='PROVISIONAL_AS_FIXED_RESULTS_CLOSE',rows=[
        dict(gap='target_input_contract',evidence='PI0280 0/8 honest supported routes',decision='Declared EEF/gripper interface adaptation required; no fake source joints',authorization='NOT_AUTHORIZED'),
        dict(gap='directEEF_vs_FKq',evidence='SAVED_DUAL_READOUT.json all272 packages; mismatch grows with horizon',decision='Compare actual JOINT outcomes and commandedFK replay before attributing task failure to representation',authorization='DIAGNOSTIC_AUTHORIZED_ONLY'),
        dict(gap='source_bridge',evidence='expert roundtrip fixed4 slots independently audited as they close',decision='Any positive-control failure requires bridge/timing/controller review before neural tuning',authorization='NO_NEW_TRAINING'),
        dict(gap='WBC_executor',evidence='native and one proven repair fixture arrays; original stops remain',decision='Capability/semantic qualification separate from VLA task quality',authorization='EXISTING_PROTOCOL_ONLY'),
        dict(gap='next_adaptation',evidence='Pending final joint/expert/WBC outcomes; capacity and data requirement unknown',decision='Pristine initialization, native flowmatching first, preserve VLM initially; limited interface tuning versus projector+DiT only if needed',authorization='NOT_AUTHORIZED')],new_training_updates=0))
    print(json.dumps(dict(source=[{k:v[k] for k in ('lane','condition','status','quality')} for v in source],wbc=wbc)))

if __name__=='__main__':
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU audit requires hidden CUDA'
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['old','new']);p.add_argument('--only',action='append',help='Finite new-phase lane:condition subset, retaining prior unaudited snapshot rows');args=p.parse_args()
    only={(x.rsplit(':',1)[0],int(x.rsplit(':',1)[1])) for x in args.only} if args.only else None
    audit_old() if args.phase=='old' else audit_new(only)
