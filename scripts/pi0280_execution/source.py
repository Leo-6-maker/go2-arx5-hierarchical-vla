"""Native frozen source policy and four bounded Panda physics episodes."""
import copy,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path('/home/liuyu-nfs/Projects/go2-arx5/PI0280')

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def guard():
    assert json.loads((ROOT.parent/'PI0279/runtime/SCIENTIFIC_PAUSE.json').read_text())['launch_allowed'] is False
    binding=json.loads((ROOT/'design/BASELINE_BINDING.json').read_text())
    assert binding['release']=='9c0172ced33c6052f7cb037b4a6d91910f9fe69e' and binding['native_steps']==4 and binding['action_horizon']==40
    assert json.loads((ROOT/'native/RESULT.json').read_text())['status']=='NATIVE_SOURCE_INFERENCE_REPRODUCED'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'

def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')

def serve():
    guard()
    import torch
    from native import TAG,BASE,array_hash
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7 as processing
    receipt=json.loads((ROOT/'evidence/WEIGHT_LOADING.json').read_text());cosmos=json.loads((ROOT/'evidence/COSMOS_FILES.json').read_text())
    original_build=processing.build_processor
    def local(name,kwargs):
        assert name=='nvidia/Cosmos-Reason2-2B';return original_build(str(Path(cosmos['files'][0]['path']).parent),kwargs)
    processing.build_processor=local;torch.set_num_threads(4);torch.set_grad_enabled(False)
    policy=Gr00tPolicy(TAG,str(BASE),device='cuda:0',strict=True);policy.model.eval();policy.model.requires_grad_(False)
    assert policy.model.action_head.num_inference_timesteps==4 and policy.model.action_head.action_horizon==40
    assert all(not p.requires_grad for p in policy.model.parameters())
    def verify():
        parameters=dict(policy.model.named_parameters());assert set(parameters)==set(receipt['parameter_hashes'])
        for k,p in parameters.items():assert array_hash(p.detach().cpu().contiguous().view(torch.uint8).numpy())==receipt['parameter_hashes'][k],k
    verify();torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    ledger=ROOT/'evidence/model_budget.jsonl';counts={'policy_attempt':0,'action_head':0}
    previous=[json.loads(line) for line in ledger.read_text().splitlines()]
    for row in previous:counts[row['kind']]+=1
    initial_counts=counts.copy()
    prior_source_calls=sum(row['kind']=='policy_attempt' and row['id'].startswith(str(ROOT/'source')+'/') for row in previous)
    active={};raw=[];noise=[]
    def charge(kind):
        assert counts[kind]<(1024 if kind=='policy_attempt' else 4096);counts[kind]+=1
        with ledger.open('a') as f:f.write(json.dumps(dict(kind=kind,id=active['id'],utc_ns=time.time_ns()))+'\n');f.flush();os.fsync(f.fileno())
    hook=policy.model.action_head.model.register_forward_pre_hook(lambda *args:charge('action_head'))
    forward=policy.model.get_action
    def capture(*a,**kw):
        answer=forward(*a,**kw);raw.append(answer['action_pred'].detach().cpu().float().numpy());return answer
    policy.model.get_action=capture
    print(json.dumps(dict(ready=True,point='PRISTINE_OFFICIAL_BASE')),flush=True)
    for line in sys.stdin:
        request=json.loads(line);out=Path(request['output']);assert not out.exists(),'First valid query must be reused'
        assert out.is_relative_to(ROOT/'source') and prior_source_calls+counts['policy_attempt']-initial_counts['policy_attempt']<320
        active['id']=str(out);charge('policy_attempt');raw.clear();noise.clear()
        with np.load(request['input']) as data:
            assert data['state'].shape==(17,) and np.isfinite(data['state']).all()
            assert all(data[k].shape==(2,128,128,3) and data[k].dtype==np.uint8 for k in ('exterior_image_1_left','wrist_image_left'))
            assert np.isclose(np.diff(data['history_times_s'])[0],1.)
            s=data['state'];observation=dict(video={k:data[k][None] for k in ('exterior_image_1_left','wrist_image_left')},state={k:s[a:b][None,None] for k,a,b in [('eef_9d',0,9),('gripper_position',9,10),('joint_position',10,17)]},language={policy.language_key:[[request['language']]]})
        randn=torch.randn
        def actual_noise(*a,**kw):
            value=randn(*a,**kw);noise.append(value.detach().cpu().view(torch.uint16).numpy().copy());return value
        torch.randn=actual_noise;start=time.perf_counter()
        try:
            with torch.inference_mode():decoded,_=policy.get_action(copy.deepcopy(observation))
            torch.cuda.synchronize()
        finally:torch.randn=randn
        assert len(noise)==1 and counts['action_head']-initial_counts['action_head']==4*(counts['policy_attempt']-initial_counts['policy_attempt'])
        np.savez_compressed(out,raw=raw[-1],noise_uint16=noise[0],latency_s=time.perf_counter()-start,**{'decoded_'+k:v for k,v in decoded.items()})
        print(json.dumps(dict(output=str(out))),flush=True)
    hook.remove();verify();write(ROOT/'evidence/SOURCE_POLICY_IMMUTABLE.json',dict(effective_weights_unchanged=True,total_counts=counts,source_counts={k:counts[k]-initial_counts[k] for k in counts},weight_loading_sha256=digest(ROOT/'evidence/WEIGHT_LOADING.json'),training_updates=0))

def run():
    guard()
    import subprocess
    from scipy.spatial.transform import Rotation,Slerp
    sys.path.insert(0,str(ROOT.parent/'PI0276/code'))
    import source_worker as sw
    from go2arx5_vla.contracts import TrajectoryChunkV1
    from go2arx5_vla.safety import TrajectorySafetyGateV1,GateDecision
    from source_control import encode,decode,next_tick,self_check
    self_check()
    assert json.loads((ROOT/'evidence/SOURCE_TCP_CONTRACT.json').read_text())['status']=='PANDA_DROID_STATIC_TCP_CONSISTENT'
    sw.RAW=ROOT/'source';sw.LIMITS=dict(starts=6,control=6000,physics=30000,rgb=6000,teacher=0);sw.BUDGET_COUNTS=None
    # Reuse actual native simulation/contacts/rendering; isolated outputs and original controller.
    card=dict(conditions=[10000,10001,10002,10003],instruction='Stack the red cube on the green cube and release it.',
        official_fps=15,H=40,prefix=8,source_control_Hz=20,pre_roll_s=1.,task_s=32.,queries_per_condition_cap=80,policy_rng_seed=0,
        input='actual Panda link8 base pose; true seven joints; real RGB now and one second ago',
        TCP_link8_from_hand=dict(rotation_z_rad=np.pi/4,translation_z_m=-.1034),
        gripper='native physical gap/max gap, 0 open 1 closed; output closure converted to original Panda actuator fraction',
        camera_binding={'base_camera_raw':'exterior_image_1_left','hand_camera_raw':'wrist_image_left'},domain='OOD_PANDA_SIM_NOT_OFFICIAL_DROID_BENCHMARK',
        time_grid='15Hz first8 samples -> 20Hz interpolation only through7/15; holdlast until ceil(n*8/15*20) controller ticks',
        time_grid_quantization_s=[0,1/20],source_caps=sw.LIMITS,
        input_gripper_formula='1-(q_left+q_right-0.00004)/0.07996',output_gripper_formula='Panda fraction=1-stock decoded DROID closure',
        source_code_sha256={name:digest(ROOT/'code'/name) for name in ['source.py','source_control.py','source_contract.py']},
        original_source_worker_sha256=digest(ROOT.parent/'PI0276/code/source_worker.py'),
        baseline_binding_sha256=digest(ROOT/'design/BASELINE_BINDING.json'),tcp_contract_sha256=digest(ROOT/'evidence/SOURCE_TCP_CONTRACT.json'),
        NON_REALTIME=True,training_updates=0)
    path=ROOT/'design/SOURCE_PHYSICS_CARD.json'
    if path.exists():assert json.loads(path.read_text())==card
    else:write(path,card)
    tool=np.eye(4);tool[:3,:3]=Rotation.from_euler('z',np.pi/4).as_matrix();tool[2,3]=-.1034
    gate=TrajectorySafetyGateV1();summary=[]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(ROOT/'official'),HF_HOME=str(ROOT.parent/'PI0250/runtime/hf_home'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    errors=(ROOT/'runtime/source-policy.log').open('a')
    server=subprocess.Popen([str(ROOT.parent/'PI0250/runtime/venv/bin/python'),'-u',str(ROOT/'code/source.py'),'serve'],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errors,text=True,bufsize=1)
    def reply(key):
        while True:
            line=server.stdout.readline();assert line,'Source policy process exited'
            try:value=json.loads(line)
            except json.JSONDecodeError:continue
            if key in value:return value
    assert reply('ready')['ready']
    try:
        for seed in card['conditions']:
            slot='PI0280-'+str(seed);old=sw.retained(slot)
            if old:summary.append(old[1]);continue
            starts=list((sw.RAW/'episodes').glob('*/attempt-*/START.json'))
            if any(p.parent.parent.name==slot for p in starts):assert len(starts)-len({p.parent.parent.name for p in starts})<2
            ep=sw.Episode(slot,seed,'pd_ee_pose');queries=[];status='VALID_SOURCE_PHYSICS_COMPLETE'
            # The reused helper has a 640-call task cap. This counted step includes 20 extra stationary history calls.
            def step(target,g):
                assert ep.control<660;ep.ledger.reserve('control',slot);ep.ledger.sync();ep.observe()
                command=np.r_[target[:3,3],target[:2,:3].reshape(6),g];action=sw.native_action(target,ep.root,g)
                ep.commands.append(command);ep.phase.append(-1);ep.env.step(action);ep.control+=1
                installed=dict(action=action,arm_target=sw.array(ep.arm._target_qpos)[0]);q=sw.array(ep.env.agent.robot.get_qpos())[0];q[:7]=installed['arm_target']
                installed['installed_target_fk']=ep.command_fk(q);installed['base_target']=sw.matrix(ep.arm._target_pose);ep.installed.append(installed)
                assert np.max(abs(installed['base_target']-np.linalg.inv(ep.root)@target))<1e-5
            try:
                hold=sw.matrix(ep.env.agent.tcp.pose)
                for _ in range(20):step(hold,1.)
                index=0
                while ep.control<660:
                    observation=ep.observe();history=ep.inputs[ep.control-20]
                    native=np.linalg.inv(ep.root)@sw.matrix(ep.env.agent.tcp.pose)@tool
                    encoded=encode(native[:3,:3])
                    q=sw.array(ep.env.agent.robot.get_qpos())[0];g=1-(q[-2:].sum()-.00004)/.07996
                    packet=dict(exterior_image_1_left=np.stack([history['base_camera_raw'],observation['base_camera_raw']]),wrist_image_left=np.stack([history['hand_camera_raw'],observation['hand_camera_raw']]),
                        state=np.r_[native[:3,3],encoded,g,q[:7]].astype(np.float32),history_times_s=np.array([float(history['time_s']),float(observation['time_s'])]))
                    assert np.allclose(packet['history_times_s'],[ep.control*.05-1,ep.control*.05]) and -.1<=g<=1.1
                    assert np.max(abs(decode(encoded)-native[:3,:3]))<1e-5
                    if index==0:write(ep.folder/'INPUT_CONTRACT_AUDIT.json',dict(status='PASS_BEFORE_FIRST_QUERY',actual_joint_count=len(q[:7]),history_times_s=packet['history_times_s'].tolist(),real_rgb_shapes={k:list(packet[k].shape) for k in ['exterior_image_1_left','wrist_image_left']},gripper_closure=float(g),seven_actual_joints=q[:7].tolist(),clock_card_sha256=digest(path),model_queries_so_far=0))
                    folder=ep.folder/('q%03d'%index);folder.mkdir();inp=folder/'input.npz';out=folder/'output.npz';np.savez_compressed(inp,**packet)
                    assert index<80;server.stdin.write(json.dumps(dict(input=str(inp),output=str(out),language=card['instruction']))+'\n');server.stdin.flush();assert reply('output')['output']==str(out)
                    query=dict(index=index,control=ep.control,time_s=ep.control*.05-1,input=str(inp),input_sha256=digest(inp),output=str(out),output_sha256=digest(out));queries.append(query)
                    with np.load(out) as data:
                        eef=data['decoded_eef_9d'][0];gripper=data['decoded_gripper_position'][0,:,0]
                        assert eef.shape==(40,9) and data['decoded_joint_position'].shape==(1,40,7) and data['decoded_gripper_position'].shape==(1,40,1)
                        finite=all(np.isfinite(data[k]).all() for k in data.files if k=='raw' or k.startswith('decoded_'))
                    if not finite:status='VALID_SOURCE_NONFINITE_FAIL';query['status']=status;break
                    with np.errstate(invalid='ignore',divide='ignore'):physical=decode(eef[:,3:9])
                    poses=np.tile(np.eye(4),(40,1,1));poses[:,:3,:3]=physical;poses[:,:3,3]=eef[:,:3];poses=ep.root@poses@np.linalg.inv(tool)
                    now=ep.control*.05;ts=now+np.arange(40)/15
                    c=TrajectoryChunkV1(chunk_id=slot+str(index),source='OFFICIAL_DROID_DECODED',observation_timestamp=now,point_timestamps=ts,eef_position=poses[:,:3,3],eef_orientation=Rotation.from_matrix(poses[:,:3,:3]).as_quat(),
                        source_frame='PI0276_MANISKILL_WORLD',tcp_semantic_id='PI0276_PANDA_HAND_TCP',validity_start=now,validity_end=ts[-1],gripper=1-gripper)
                    actual=sw.matrix(ep.env.agent.tcp.pose);verdict=gate.evaluate(c,now=now,achieved_position=actual[:3,3],achieved_orientation=Rotation.from_matrix(actual[:3,:3]).as_quat())
                    query['gate']=verdict.as_dict();write(folder/'GATE.json',verdict.as_dict())
                    np.savez_compressed(folder/'canonical.npz',timestamp=ts,position=c.eef_position,quaternion=c.eef_orientation,gripper=c.gripper)
                    if verdict.decision is GateDecision.REJECT:status='VALID_SOURCE_GATE_REJECT';break
                    if (gripper<-.1).any() or (gripper>1.1).any():status='VALID_SOURCE_GRIPPER_ENVELOPE_FAIL';break
                    until=min(660,next_tick(index));prefix_ts=ts[:8];slerp=Slerp(prefix_ts,Rotation.from_quat(c.eef_orientation[:8]))
                    query.update(executed_first8_native_timestamps_s=prefix_ts.tolist(),replan_tick=until,quantization_s=(until-20)*.05-min(32.,(index+1)*8/15))
                    while ep.control<until:
                        clock=ep.control*.05;target=np.eye(4);target[:3,3]=[np.interp(clock,prefix_ts,c.eef_position[:8,i]) for i in range(3)];target[:3,:3]=slerp(min(clock,prefix_ts[-1])).as_matrix()
                        j=min(7,max(0,int(np.searchsorted(prefix_ts,clock,side='right')-1)));step(target,float(c.gripper[j]))
                    index+=1
                write(ep.folder/'QUERIES.json',queries)
                actual=np.array([x['success'] for x in ep.states])[::5][20:]
                result=ep.finish(status);result['legacy_unshifted_persistent_success']=result['persistent_success']
                result.update(condition=seed,folder=str(ep.folder),pre_roll_control_calls=20,task_control_calls=max(0,ep.control-20),queries=len(queries),
                    persistent_success=bool(status=='VALID_SOURCE_PHYSICS_COMPLETE' and len(actual)>=601 and actual[560:601].all()),
                    first_success_s=float(np.flatnonzero(actual)[0]*.05) if actual.any() else None,NON_REALTIME=True,source_domain='OOD_PANDA_SIMULATION')
                write(ep.folder/'RESULT.json',result);summary.append(result);write(ROOT/'evidence/SOURCE_PROGRESS.json',dict(rows=summary))
            except BaseException as exc:
                write(ep.folder/'QUERIES.json',queries);ep.finish('ENGINEERING_INCOMPLETE',repr(exc));raise
    finally:
        server.stdin.close();code=server.wait(timeout=120);errors.close();assert code==0
    write(ROOT/'evidence/SOURCE_PHYSICS_RESULT.json',dict(status='FIXED_FOUR_SOURCE_CONDITIONS_EXECUTED',rows=summary,new_training_updates=0))

if __name__=='__main__':
    if sys.argv[1]=='serve':serve()
    else:assert sys.argv[1]=='run';run()
