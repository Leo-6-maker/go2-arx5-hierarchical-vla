"""Finite PI0281 source-native joint execution and paired physical controls."""
import copy,hashlib,json,os,sys,time,subprocess
from pathlib import Path
import numpy as np
BASE=Path('/home/liuyu-nfs/Projects/go2-arx5');ROOT=BASE/'PI0281';OUT=ROOT/'source';OLD=BASE/'PI0280'
RELEASE='360b412f55be8aec1568054d7d039ce7de2bc597'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)

def freeze(p,value):
    p=Path(p)
    if p.exists():assert json.loads(p.read_text())==value,str(p)
    else:write(p,value)

def expert(seed):
    folder=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0276/episodes')/('TRAIN-'+str(seed)+'-joint')/'attempt-000'
    result=json.loads((folder/'RESULT.json').read_text());assert result['status']=='VALID_SOURCE_COMPLETE' and result['persistent_success']
    return folder

def prepare():
    from source_math import self_check,expert_bridge
    from scipy.spatial.transform import Rotation
    self_check();parents=[];noise=[];bridges=[]
    for seed in range(10000,10004):
        folder=expert(seed);files={n:digest(folder/n) for n in ['RESULT.json','commands.npz','initial.npz','installed.npz']}
        parents.append(dict(condition=seed,path=str(folder),files=files))
        with np.load(folder/'commands.npz') as data:commands=data['action'].copy()
        with np.load(folder/'initial.npz') as data:q=data['articulations/panda_wristcam'][0]
        root=np.eye(4);root[:3,3]=q[:3];root[:3,:3]=Rotation.from_quat(q[[4,5,6,3]]).as_matrix()
        poses,g,report=expert_bridge(commands,root);bridges.append(dict(condition=seed,**report))
        path=OUT/'design'/('expert_bridge_'+str(seed)+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():np.savez_compressed(path,poses=poses,gripper=g,original_commands=commands,root=root)
        else:
            with np.load(path) as data:assert np.array_equal(data['poses'],poses) and np.array_equal(data['gripper'],g)
        for index in range(60):
            p=OLD/'source/episodes'/('PI0280-'+str(seed))/'attempt-000'/('q%03d'%index)/'output.npz'
            with np.load(p) as saved:a=saved['noise_uint16'];assert a.shape==(1,40,132) and a.dtype==np.uint16
            noise.append(dict(condition=seed,index=index,path=str(p),output_sha256=digest(p),noise_sha256=hashlib.sha256(a.tobytes()).hexdigest()))
    card=dict(authority='PI0281_CORELAB_NATIVE_READOUT_WBC_REPAIR_AND_ADAPTATION_GAP_GO_V1',release=RELEASE,canonical=6040516315,
        conditions=[10000,10001,10002,10003],instruction='Stack the red cube on the green cube and release it.',
        lanes=['CONTROLLER_FIXTURES','ORIGINAL_BASE_JOINT','FK_OF_JOINT_COMMAND_REPLAY','EXPERT_DROID_BRIDGE_ROUNDTRIP'],
        fixture_seed=10000,fixture_seconds=2,fixture_command='measured nonidentity reset pose hold; gripper open0..0.5 close0.5..1.5 open1.5..2',
        task_s=32,pre_roll_s=1,control_Hz=20,physics_Hz=100,model_fps=15,H=40,prefix=8,queries_per_condition=60,
        joint_time_grid='absolute stock q zero-order hold first8 native15Hz samples; ceil(n*8/15*20) absolute replan schedule',
        joint_safety='reject nonfinite/outside actual URDF hard qlimits before actuator dispatch; exact Panda joint1..7 order; no clipping',
        source_caps=dict(starts=14,policy_attempt=272,action_head=1088,physics=41000,control=8500,rgb=8500),
        per_attempt_reservation=dict(full=dict(integrations=3380,dispatches=660,RGB_pairs=661),fixture=dict(integrations=220,dispatches=40,RGB_pairs=40),joint_model_requests=60,joint_heads=240),
        gripper='stock closure -> existing Panda fraction1-closure -> actuator2*fraction-1; inherited [-.1,1.1] admission/physical saturation recorded',
        cameras='original real128x128 base_camera_raw+hand_camera_raw; actual stationary1s history; unchanged PI0280 official preprocessing',
        FK_replay='FK of literal commanded q at each20Hz tick, original gripper timing; fresh initial state, no high-level model feedback',
        expert_roundtrip='15Hz SE3 resampling, fixed DROID calibration, original20Hz gripper event timing',
        direct_EEF='SEALED_PI0280_REUSE_NO_CORRECTION_PROVEN',NON_REALTIME=True,domain='OOD_SOURCE_PANDA_NOT_WBC_NOT_OFFICIAL_DROID_HARDWARE',
        predecessors=dict(BASELINE_BINDING=digest(OLD/'design/BASELINE_BINDING.json'),WEIGHT_LOADING=digest(OLD/'evidence/WEIGHT_LOADING.json'),
            SOURCE_TCP_CONTRACT=digest(OLD/'evidence/SOURCE_TCP_CONTRACT.json'),source_worker=digest(BASE/'PI0276/code/source_worker.py'),
            source_control=digest(OLD/'code/source_control.py'),joint_controller=digest(BASE/'PI0276/ManiSkill/mani_skill/agents/controllers/pd_joint_pos.py')),
        actual_run_command=[str(BASE/'PI0276/venv/bin/python'),'-u',str(ROOT/'code/source.py'),'run'],
        finite_resource_wrapper=[str(BASE/'PI0250/runtime/venv/bin/python'),'-u',str(ROOT/'code/source_launch.py')],
        code_sha256={n:digest(ROOT/'code'/n) for n in ['source.py','source_math.py','source_launch.py']},expert_parents=parents,noise=noise,training_updates=0,target_queries=0,hardware=0)
    freeze(OUT/'design/SOURCE_CARD.json',card);freeze(OUT/'evidence/CPU_BRIDGE_CHECK.json',dict(status='PASS_BEFORE_PHYSICS_AND_MODEL',rows=bridges,new_model_queries=0,new_physics_steps=0))
    print('SOURCE_CARD_FROZEN_CPU_CHECK_PASS',flush=True)

def guard():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    assert json.loads((BASE/'PI0279/runtime/SCIENTIFIC_PAUSE.json').read_text())['launch_allowed'] is False
    card=json.loads((OUT/'design/SOURCE_CARD.json').read_text());assert card['release']==RELEASE
    for name,sha in card['code_sha256'].items():assert digest(ROOT/'code'/name)==sha
    prior=dict(BASELINE_BINDING=OLD/'design/BASELINE_BINDING.json',WEIGHT_LOADING=OLD/'evidence/WEIGHT_LOADING.json',SOURCE_TCP_CONTRACT=OLD/'evidence/SOURCE_TCP_CONTRACT.json',source_worker=BASE/'PI0276/code/source_worker.py',source_control=OLD/'code/source_control.py',joint_controller=BASE/'PI0276/ManiSkill/mani_skill/agents/controllers/pd_joint_pos.py')
    for name,p in prior.items():assert digest(p)==card['predecessors'][name],name
    # The owner supplies publication and allocation receipts; no second worker signature grants execution.
    from budget import guard as shared_guard
    shared_guard()
    approval=json.loads((ROOT/'design/IMPLEMENTATION.json').read_text())
    assert approval['source_card_sha256']==digest(OUT/'design/SOURCE_CARD.json')
    return card

def reserve(kind,identity):
    path=OUT/'evidence/model_budget.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    limit=272 if kind=='policy_attempt' else 1088
    assert sum(row['kind']==kind for row in rows)<limit
    attempt=str(Path(identity).parent.parent)
    assert sum(row['kind']==kind and str(Path(row['id']).parent.parent)==attempt for row in rows)<(60 if kind=='policy_attempt' else 240)
    with path.open('a') as stream:stream.write(json.dumps(dict(kind=kind,id=identity,utc_ns=time.time_ns()))+'\n');stream.flush();os.fsync(stream.fileno())

def serve():
    card=guard();import torch
    sys.path.insert(0,str(OLD/'code'))
    from native import TAG,BASE as MODEL,array_hash
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7 as processing
    for row in json.loads((OLD/'evidence/PRISTINE_FILES.json').read_text())['files']:
        if not row['path'].endswith('.safetensors'):assert digest(MODEL/row['path'])==row['sha256'],row['path']
    receipt=json.loads((OLD/'evidence/WEIGHT_LOADING.json').read_text());cosmos=json.loads((OLD/'evidence/COSMOS_FILES.json').read_text())
    original=processing.build_processor
    def local(name,kwargs):
        assert name=='nvidia/Cosmos-Reason2-2B';return original(str(Path(cosmos['files'][0]['path']).parent),kwargs)
    processing.build_processor=local;torch.set_num_threads(4);torch.set_grad_enabled(False)
    policy=Gr00tPolicy(TAG,str(MODEL),device='cuda:0',strict=True);policy.model.eval();policy.model.requires_grad_(False)
    import gr00t.policy.gr00t_policy as official
    assert str(Path(official.__file__).resolve()).startswith(str(OLD/'official')+'/')
    assert policy.model.action_head.num_inference_timesteps==4 and policy.model.action_head.action_horizon==40
    def verify():
        parameters=dict(policy.model.named_parameters());assert set(parameters)==set(receipt['parameter_hashes'])
        for k,p in parameters.items():assert not p.requires_grad and array_hash(p.detach().cpu().contiguous().view(torch.uint8).numpy())==receipt['parameter_hashes'][k],k
    verify();active={};raw=[];calls=0;heads=0
    def head(*args):
        nonlocal heads
        reserve('action_head',active['id']);heads+=1
    hook=policy.model.action_head.model.register_forward_pre_hook(head);forward=policy.model.get_action
    def capture(*args,**kwargs):
        answer=forward(*args,**kwargs);raw.append(answer['action_pred'].detach().cpu().float().numpy());return answer
    policy.model.get_action=capture;print(json.dumps(dict(ready=True)),flush=True)
    try:
        for line in sys.stdin:
            request=json.loads(line);out=Path(request['output']);assert out.is_relative_to(OUT) and not out.exists()
            active['id']=str(out);reserve('policy_attempt',active['id']);calls+=1;raw.clear()
            with np.load(request['input']) as saved:
                state=saved['state'];assert state.shape==(17,) and np.isfinite(state).all()
                assert all(saved[k].shape==(2,128,128,3) and saved[k].dtype==np.uint8 for k in ['exterior_image_1_left','wrist_image_left'])
                assert np.isclose(np.diff(saved['history_times_s'])[0],1)
                observation=dict(video={k:saved[k][None] for k in ['exterior_image_1_left','wrist_image_left']},state={k:state[a:b][None,None] for k,a,b in [('eef_9d',0,9),('gripper_position',9,10),('joint_position',10,17)]},language={policy.language_key:[[card['instruction']]]})
            parent=Path(request['noise_parent']);identity=next(n for n in card['noise'] if n['path']==str(parent));assert digest(parent)==identity['output_sha256']
            with np.load(parent) as saved:noise=torch.from_numpy(saved['noise_uint16'].copy()).view(torch.bfloat16).cuda()
            randn=torch.randn;noise_calls=[]
            def actual_noise(*args,**kwargs):
                assert tuple(kwargs['size'])==tuple(noise.shape) and kwargs['dtype']==noise.dtype and torch.device(kwargs['device'])==noise.device
                noise_calls.append(1);return noise.clone()
            torch.randn=actual_noise;started=time.perf_counter()
            try:
                with torch.inference_mode():decoded,_=policy.get_action(copy.deepcopy(observation))
                torch.cuda.synchronize()
            finally:torch.randn=randn
            assert len(noise_calls)==1 and heads==4*calls
            np.savez_compressed(out,raw=raw[-1],noise_uint16=noise.detach().cpu().view(torch.uint16).numpy(),latency_s=time.perf_counter()-started,**{'decoded_'+k:v for k,v in decoded.items()})
            print(json.dumps(dict(output=str(out))),flush=True)
    finally:
        hook.remove();verify();write(OUT/'evidence/POLICY_IMMUTABLE.json',dict(effective_weights_unchanged=True,policy_attempt=calls,action_head=heads,weight_loading_sha256=digest(OLD/'evidence/WEIGHT_LOADING.json'),training_updates=0))

def setup():
    from source_math import self_check
    self_check();sys.path.insert(0,str(BASE/'PI0276/code'));import source_worker as sw
    sw.RAW=OUT;sw.LIMITS=dict(starts=14,control=8500,physics=41000,rgb=8500,teacher=0);sw.BUDGET_COUNTS=None
    from collections import Counter
    path=OUT/'evidence/budget.jsonl';prior=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    attempt_counts=Counter((x['slot'],x['kind']) for x in prior);original_reserve=sw.Ledger.reserve
    def bounded(self,kind,slot):
        fixture=slot.startswith('fixture-');caps=dict(starts=1,control=40 if fixture else 660,physics=220 if fixture else 3380,rgb=40 if fixture else 661,teacher=0)
        assert attempt_counts[(slot,kind)]<caps[kind],(slot,kind,'Attempt reservation exhausted')
        original_reserve(self,kind,slot);attempt_counts[(slot,kind)]+=1
    sw.Ledger.reserve=bounded
    return sw

def episode(sw,lane,seed,mode):
    slot=lane+'-'+str(seed);old=sw.retained(slot)
    if old:return None,old[1]
    assert not list((OUT/'episodes'/slot).glob('attempt-*')),'Replacement requires owner audit/allocation'
    from budget import reserve as shared_reserve
    fixture=lane.startswith('fixture-')
    shared_reserve('SOURCE',slot,dict(physical_starts=1,integrations=220 if fixture else 3380,dispatches=40 if fixture else 660,RGB_pairs=40 if fixture else 661,
        policy_requests=60 if lane=='ORIGINAL_BASE_JOINT' else 0,policy_heads=240 if lane=='ORIGINAL_BASE_JOINT' else 0))
    ep=sw.Episode(slot,seed,mode)
    assert not ep.arm.config.use_delta and not ep.arm.config.use_target and not ep.arm.config.normalize_action
    assert [j.name for j in ep.arm.joints]==['panda_joint%d'%i for i in range(1,8)]
    return ep,None

def step(sw,ep,target,g,joint=False):
    assert ep.control<660
    if joint:
        limits=sw.array(ep.env.agent.robot.get_qlimits())[0,:7]
        if not np.isfinite(target).all() or (target<limits[:,0]).any() or (target>limits[:,1]).any():return 'VALID_JOINT_HARD_LIMIT_REJECT'
        q=sw.array(ep.env.agent.robot.get_qpos())[0];q[:7]=target;pose=ep.command_fk(q)
        action=np.r_[target,2*np.clip(g,0,1)-1].astype(np.float32)
    else:pose=target;action=sw.native_action(pose,ep.root,g)
    ep.ledger.reserve('control',ep.slot);ep.ledger.sync();ep.observe()
    ep.commands.append(np.r_[pose[:3,3],pose[:2,:3].reshape(6),g]);ep.phase.append(-1)
    ep.env.step(action);ep.control+=1;q=sw.array(ep.env.agent.robot.get_qpos())[0];installed=dict(action=action,arm_target=sw.array(ep.arm._target_qpos)[0]);q[:7]=installed['arm_target']
    installed['installed_target_fk']=ep.command_fk(q)
    if joint:assert np.max(abs(installed['arm_target']-target))<1e-6,'Native controller changed q command'
    else:
        installed['base_target']=sw.matrix(ep.arm._target_pose);assert np.max(abs(installed['base_target']-np.linalg.inv(ep.root)@pose))<1e-5
    ep.installed.append(installed)
    return None

def finish(ep,status,lane,seed,queries=0):
    success=np.array([x['success'] for x in ep.states])[::5][20:] if lane!='CONTROLLER_FIXTURES' else np.array([])
    limits=__import__('source_worker').array(ep.env.agent.robot.get_qlimits())[0,:7]
    observed=np.array([x['qpos'][:7] for x in ep.states]);margin=np.minimum(observed-limits[:,0],limits[:,1]-observed)
    result=ep.finish(status);result.update(lane=lane,condition=seed,folder=str(ep.folder),queries=queries,
        minimum_actual_joint_hard_margin_rad=margin.min(axis=0).tolist(),maximum_actual_joint_penetration_rad=float(max(0,-margin.min())),
        pre_roll_control_calls=min(20,ep.control) if lane!='CONTROLLER_FIXTURES' else 0,task_control_calls=max(0,ep.control-20) if lane!='CONTROLLER_FIXTURES' else 40,
        persistent_success=bool(status=='VALID_SOURCE_COMPLETE' and len(success)>=601 and success[560:601].all()),
        first_success_s=float(np.flatnonzero(success)[0]*.05) if success.any() else None,NON_REALTIME=True,source_domain='OOD_SOURCE_PANDA')
    write(ep.folder/'RESULT.json',result)
    from budget import actual
    model=[json.loads(x) for x in (OUT/'evidence/model_budget.jsonl').read_text().splitlines()] if (OUT/'evidence/model_budget.jsonl').exists() else []
    own=[r for r in model if r['id'].startswith(str(ep.folder)+'/')]
    actual('SOURCE',ep.slot,dict(physical_starts=1,integrations=result['physics_completed'],dispatches=result['control_calls'],RGB_pairs=result['RGB_pairs'],
        policy_requests=sum(r['kind']=='policy_attempt' for r in own),policy_heads=sum(r['kind']=='action_head' for r in own)),status)
    return result

def fixture(sw,joint):
    lane='CONTROLLER_FIXTURES';ep,old=episode(sw,'fixture-'+('joint' if joint else 'eef'),10000,'pd_joint_pos' if joint else 'pd_ee_pose')
    if old:return old
    try:
        q=sw.array(ep.env.agent.robot.get_qpos())[0];pose=sw.matrix(ep.env.agent.tcp.pose);assert np.max(abs(ep.command_fk(q)-pose))<1e-5
        for tick in range(40):
            g=1. if tick<10 or tick>=30 else 0.;assert step(sw,ep,q[:7] if joint else pose,g,joint) is None
        gap=np.array([s['qpos'][-2:].sum() for s in ep.states]);observed=dict(gripper_qsum_min=float(gap.min()),gripper_qsum_max=float(gap.max()),nonidentity_reset_rotation=pose[:3,:3].tolist(),native_fk_error=float(np.max(abs(ep.command_fk(q)-pose))))
        status='VALID_CONTROLLER_FIXTURE_PASS' if gap.min()<.02 and gap.max()>.06 else 'VALID_CONTROLLER_FIXTURE_FAIL'
        write(ep.folder/'FIXTURE.json',observed);return finish(ep,status,lane,10000)
    except BaseException as exc:ep.finish('ENGINEERING_INCOMPLETE',repr(exc));raise

def preroll(sw,ep):
    joint=ep.mode=='pd_joint_pos';q=sw.array(ep.env.agent.robot.get_qpos())[0];pose=sw.matrix(ep.env.agent.tcp.pose)
    for _ in range(20):assert step(sw,ep,q[:7] if joint else pose,1.,joint) is None

def joint_episode(sw,server,seed,card):
    from source_math import encode,tool,native_index,next_tick,native_to_hand
    from scipy.spatial.transform import Rotation
    from go2arx5_vla.contracts import TrajectoryChunkV1
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    lane='ORIGINAL_BASE_JOINT';ep,old=episode(sw,lane,seed,'pd_joint_pos')
    if old:return old
    queries=[];status='VALID_SOURCE_COMPLETE'
    try:
        preroll(sw,ep)
        for index in range(60):
            observation=ep.observe();history=ep.inputs[ep.control-20];native=np.linalg.inv(ep.root)@sw.matrix(ep.env.agent.tcp.pose)@tool();q=sw.array(ep.env.agent.robot.get_qpos())[0]
            packet=dict(state=np.r_[native[:3,3],encode(native[:3,:3]),1-(q[-2:].sum()-.00004)/.07996,q[:7]].astype(np.float32),history_times_s=np.array([history['time_s'],observation['time_s']]),
                exterior_image_1_left=np.stack([history['base_camera_raw'],observation['base_camera_raw']]),wrist_image_left=np.stack([history['hand_camera_raw'],observation['hand_camera_raw']]))
            folder=ep.folder/('q%03d'%index);folder.mkdir();inp=folder/'input.npz';out=folder/'output.npz';np.savez_compressed(inp,**packet)
            noise_parent=OLD/'source/episodes'/('PI0280-'+str(seed))/'attempt-000'/('q%03d'%index)/'output.npz'
            server.stdin.write(json.dumps(dict(input=str(inp),output=str(out),noise_parent=str(noise_parent)))+'\n');server.stdin.flush();assert reply(server,'output')['output']==str(out)
            query=dict(index=index,control=ep.control,time_s=ep.control*.05-1,input_sha256=digest(inp),output_sha256=digest(out),noise_parent_sha256=digest(noise_parent));queries.append(query)
            with np.load(out) as data:
                joints=data['decoded_joint_position'][0];closure=data['decoded_gripper_position'][0,:,0];eef=data['decoded_eef_9d'][0]
                finite=all(np.isfinite(data[k]).all() for k in data.files if k=='raw' or k.startswith('decoded_'))
            if not finite:status='VALID_NONFINITE_REJECT';break
            assert joints.shape==(40,7) and closure.shape==(40,) and eef.shape==(40,9)
            if (closure<-.1).any() or (closure>1.1).any():status='VALID_GRIPPER_ENVELOPE_REJECT';break
            poses=native_to_hand(eef,ep.root);now=ep.control*.05;ts=now+np.arange(40)/15
            chunk=TrajectoryChunkV1(chunk_id=ep.slot+str(index),source='PI0281_STOCK_PREDICTED_EEF_AUDIT_ONLY',observation_timestamp=now,point_timestamps=ts,eef_position=poses[:,:3,3],eef_orientation=Rotation.from_matrix(poses[:,:3,:3]).as_quat(),source_frame='PI0276_MANISKILL_WORLD',tcp_semantic_id='PI0276_PANDA_HAND_TCP',validity_start=now,validity_end=ts[-1],gripper=1-closure)
            actual=sw.matrix(ep.env.agent.tcp.pose);query['predicted_EEF_gate_audit_only']=TrajectorySafetyGateV1().evaluate(chunk,now=now,achieved_position=actual[:3,3],achieved_orientation=Rotation.from_matrix(actual[:3,:3]).as_quat()).as_dict()
            np.savez_compressed(folder/'predicted_EEF_audit.npz',timestamp=ts,poses=poses,gripper=1-closure)
            until=min(660,next_tick(index))
            while ep.control<until:
                j=native_index(ep.control*.05,now);failure=step(sw,ep,joints[j],float(1-closure[j]),True)
                if failure:status=failure;break
            if status!='VALID_SOURCE_COMPLETE':break
        write(ep.folder/'QUERIES.json',queries);return finish(ep,status,lane,seed,len(queries))
    except BaseException as exc:write(ep.folder/'QUERIES.json',queries);ep.finish('ENGINEERING_INCOMPLETE',repr(exc));raise

def replay(sw,lane,seed):
    from source_math import tool,hand_to_native,native_to_hand
    ep,old=episode(sw,lane,seed,'pd_ee_pose')
    if old:return old
    try:
        if lane=='FK_OF_JOINT_COMMAND_REPLAY':
            parent,record=sw.retained('ORIGINAL_BASE_JOINT-'+str(seed))
            with np.load(parent/'installed.npz') as data:
                action=data['action'];assert action.shape[1]==8
            with np.load(parent/'commands.npz') as data:g=data['action'][:,-1].copy()
            poses=[]
            for command in action:
                q=sw.array(ep.env.agent.robot.get_qpos())[0];q[:7]=command[:7];poses.append(ep.command_fk(q))
            poses=np.asarray(poses);assert len(poses)==record['control_calls']
            with np.load(parent/'initial.npz') as data:initial={k:data[k] for k in data.files}
            actual=sw.flatten(ep.initial);assert initial.keys()==actual.keys() and max(np.max(abs(initial[k]-actual[k])) for k in initial)<=1e-6
            write(ep.folder/'PARENT.json',dict(folder=str(parent),result_sha256=digest(parent/'RESULT.json'),commanded_not_achieved=True,model_calls=0,source_status=record['status']))
            status='VALID_SOURCE_COMPLETE' if len(poses)==660 else 'VALID_TRUNCATED_SOURCE_REPLAY'
        else:
            parent=expert(seed)
            with np.load(OUT/'design'/('expert_bridge_'+str(seed)+'.npz')) as data:poses=data['poses'];g=data['gripper'];bound_root=data['root']
            assert np.max(abs(ep.root-bound_root))<1e-5
            with np.load(parent/'initial.npz') as data:initial={k:data[k] for k in data.files}
            actual=sw.flatten(ep.initial);assert initial.keys()==actual.keys() and max(np.max(abs(initial[k]-actual[k])) for k in initial)<=1e-6
            preroll(sw,ep);status='VALID_SOURCE_COMPLETE';write(ep.folder/'PARENT.json',dict(folder=str(parent),commands_sha256=digest(parent/'commands.npz'),new_teacher=0,model_calls=0,bridge_array_sha256=digest(OUT/'design'/('expert_bridge_'+str(seed)+'.npz'))))
        np.savez_compressed(ep.folder/'canonical_replay.npz',poses=poses,gripper=g)
        gates=[]
        for i,(pose,gripper) in enumerate(zip(poses,g)):
            # The EEF safety gate evaluates the commanded plan without modifying it.
            if i%10==0 and len(poses)-i>=40:
                end=min(len(poses),i+40);commands=np.c_[poses[i:end,:3,3],poses[i:end,:2,:3].reshape(-1,6),g[i:end]]
                current=sw.matrix(ep.env.agent.tcp.pose);q=sw.array(ep.env.agent.robot.get_qpos())[0];state=np.r_[current[:3,3],current[:2,:3].reshape(6),sw.opening(q)]
                gate=sw.pose_gate(commands,state,ep.control*.05,ep.slot);gates.append(dict(control=ep.control,result=gate))
                if gate['decision']=='REJECT':status='VALID_REPLAY_GATE_REJECT';break
            failure=step(sw,ep,pose,float(gripper));assert failure is None
        write(ep.folder/'GATES.json',gates);return finish(ep,status,lane,seed)
    except BaseException as exc:ep.finish('ENGINEERING_INCOMPLETE',repr(exc));raise

def reply(server,key):
    while True:
        line=server.stdout.readline();assert line,'Resident source policy exited'
        try:value=json.loads(line)
        except json.JSONDecodeError:continue
        if key in value:return value

def run():
    card=guard();sw=setup();rows=[]
    for joint in [True,False]:rows.append(fixture(sw,joint));write(OUT/'evidence/SOURCE_PROGRESS.json',dict(rows=rows))
    if any(x['status']!='VALID_CONTROLLER_FIXTURE_PASS' for x in rows):
        write(OUT/'evidence/SOURCE_RESULT.json',dict(status='BLOCKED_SOURCE_CONTROLLER_FIXTURE',rows=rows));return
    env=dict(os.environ,PYTHONPATH=str(OLD/'official')+':'+str(ROOT/'code'),HF_HOME=str(BASE/'PI0250/runtime/hf_home'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NO_ALBUMENTATIONS_UPDATE='1')
    with (OUT/'runtime/policy.log').open('a') as errors:
        server=subprocess.Popen([str(BASE/'PI0250/runtime/venv/bin/python'),'-u',str(ROOT/'code/source.py'),'serve'],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errors,text=True,bufsize=1)
        try:
            assert reply(server,'ready')['ready']
            for seed in card['conditions']:rows.append(joint_episode(sw,server,seed,card));write(OUT/'evidence/SOURCE_PROGRESS.json',dict(rows=rows))
        finally:
            server.stdin.close();code=server.wait(timeout=120);assert code==0
    for lane in ['FK_OF_JOINT_COMMAND_REPLAY','EXPERT_DROID_BRIDGE_ROUNDTRIP']:
        for seed in card['conditions']:rows.append(replay(sw,lane,seed));write(OUT/'evidence/SOURCE_PROGRESS.json',dict(rows=rows))
    write(OUT/'evidence/SOURCE_RESULT.json',dict(status='FIXED_SOURCE_PANELS_EXECUTED',rows=rows,training_updates=0,target_queries=0))

def interrupted_actuals():
    """Close attempted-event counts after an engineering interruption; never relaunch."""
    from budget import actual
    reservations=ROOT/'evidence/RESERVATIONS.jsonl'
    if not reservations.exists():return
    done=ROOT/'evidence/ACTUAL.jsonl';finished=[json.loads(x) for x in done.read_text().splitlines()] if done.exists() else []
    event=OUT/'evidence/budget.jsonl';events=[json.loads(x) for x in event.read_text().splitlines()] if event.exists() else []
    model=OUT/'evidence/model_budget.jsonl';models=[json.loads(x) for x in model.read_text().splitlines()] if model.exists() else []
    for row in map(json.loads,reservations.read_text().splitlines()):
        if row['lane']!='SOURCE' or any(x['lane']=='SOURCE' and x['uid']==row['uid'] for x in finished):continue
        uid=row['uid'];own=[x for x in events if x['slot']==uid];calls=[x for x in models if x['id'].startswith(str(OUT/'episodes'/uid)+'/')]
        counts=dict(physical_starts=sum(x['kind']=='starts' for x in own),integrations=sum(x['kind']=='physics' for x in own),dispatches=sum(x['kind']=='control' for x in own),RGB_pairs=sum(x['kind']=='rgb' for x in own),policy_requests=sum(x['kind']=='policy_attempt' for x in calls),policy_heads=sum(x['kind']=='action_head' for x in calls))
        actual('SOURCE',uid,counts,'ENGINEERING_INCOMPLETE_ATTEMPTED_EVENTS')

if __name__=='__main__':
    assert sys.argv[1] in ['prepare','serve','run']
    try:globals()[sys.argv[1]]()
    finally:
        if sys.argv[1]=='run':interrupted_actuals()
