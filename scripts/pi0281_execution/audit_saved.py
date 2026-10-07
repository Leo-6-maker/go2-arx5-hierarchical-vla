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
    return dict(id=id,lane=lane,input=identity(inp),output=identity(path),actual_noise_array_sha256=hashlib.sha256(noise.tobytes()).hexdigest(),
        noise_dtype=str(noise.dtype),noise_shape=list(noise.shape),history=history,
        first8=dict(position_m=stats(xyz[:8]),rotation_rad=stats(rot[:8])),full40=dict(position_m=stats(xyz),rotation_rad=stats(rot)),
        horizon_position_m=xyz.tolist(),horizon_rotation_rad=rot.tolist(),shared_gripper_closure=stats(closure),
        gripper_route_difference=0.,gripper_route_difference_semantics='Both readouts use the same stock gripper output; no separate gripper modality exists for FK(q)',
        unchanged_gates=gates,state_geometry=dict(position_m=float(np.linalg.norm(state_direct[:3,3]-state_fk[:3,3])),rotation_rad=float(angle(state_direct[:3,:3],state_fk[:3,:3]))))

def audit_old():
    fk,control,tool,chunk_type,posegate,deps=load_cpu();rows=[];funnel=[];input_errors=[];native=json.loads((OLD/'design/NATIVE_CARD.json').read_text())
    for row in native['rows']:
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
            gate_distribution={route:dict(collections.Counter(r['unchanged_gates'][route]['decision'] for r in group)) for route in ('DIRECT_EEF','FK_DECODED_ABSOLUTE_Q')}))
    report=dict(schema='PI0281_INDEPENDENT_SAVED_DUAL_READOUT_V1',release=RELEASE,observed_unix_s=time.time(),status='PASS_CPU_SAVED_ABI_AND_COMMAND_AUDIT',
        package_count=272,A2_unique=32,A3_unique=240,groups=groups,rows=rows,old_object_funnel=funnel,input_max_abs_error=max(input_errors),
        definitions=dict(position='Euclidean meters same Panda/DROID frame',rotation='SO3 geodesic radians same TCP',joint='Stock decoded absolute seven Panda angles radians, never achieved/postIK targets',
            gripper='Stock DROID closure0open1closed; shared across both readouts',history='A2 indices[-15,0] at15Hz; A3 physicalcontrol[-20,0] at20Hz =1s truehistory',
            execution='8 native15Hz timestamps,20Hz interpolation through7/15 then hold until integerceil next replan; no speed scaling'),
        dependencies=deps,implementation=identity(Path(__file__)),new_model_calls=0,new_actor_calls=0,new_physics_steps=0,new_training_updates=0,selected='NONE',deployment='NONE')
    save('SAVED_DUAL_READOUT.json',report);save('OLD_OBJECT_FUNNEL.json',dict(rows=funnel,original_lift='UNOBSERVED',new_physics_steps=0))
    lines=['| Package group | N | XYZ mean/p95 full40 (m) | SO3 mean/p95 full40 (rad) |','|---|---:|---:|---:|']
    for g in groups:lines.append('| %s | %d | %.6f / %.6f | %.6f / %.6f |'%(g['group'],g['packages'],g['full40_position_m']['mean'],g['full40_position_m']['p95'],g['full40_rotation_rad']['mean'],g['full40_rotation_rad']['p95']))
    (ROOT/'audit/SAVED_DUAL_READOUT.md').write_text('\n'.join(lines)+'\n\nReadout mismatch is descriptive, not automatically a decode defect. Both joint and EEF are independently predicted. Original lift predicate is UNOBSERVED; measured height is retained. All272 existing output packages, no new neural or physics calls.\n')
    print(json.dumps(dict(groups=groups,input_max_abs_error=max(input_errors),funnel=funnel)))

if __name__=='__main__':
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU audit requires hidden CUDA'
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['old']);args=p.parse_args();audit_old()
