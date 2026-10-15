# Native loop derived from the immutable MP-S2 loop; controller binaries unchanged.
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import numpy as np
from .night_runtime import validate_entry,CUTOFF,read,sha
from .task_state import ActualState,write_gripper,read_state
from .task_gripper_contract import teacher_physical_fraction,require_working_target
from .night_runtime import VERSION
from .night_runtime import require_card,require_adoption
from .task_clock import finish_pause,preinstall_clock_failure,actual_source_age


class FrameRecorder:
    """Render a read-only copy of an actual recorded native state, never step it."""
    def __init__(self,model_path,folder):
        self.model_path,self.folder=model_path,folder
        self.model=self.data=self.renderer=None

    def save(self,index,packet):
        import mujoco
        from PIL import Image
        if self.model is None:
            self.model=mujoco.MjModel.from_binary_path(str(self.model_path))
            self.data=mujoco.MjData(self.model)
            self.renderer=mujoco.Renderer(self.model,height=256,width=256)
        m,d,r=self.model,self.data,self.renderer
        d.qpos[:]=packet['qpos'];d.qvel[:]=packet['qvel'];d.ctrl[:]=packet['ctrl'];d.time=packet['time_s']
        mujoco.mj_forward(m,d)
        calibration={}
        for camera in ('body_rgb','wrist_rgb'):
            r.update_scene(d,camera=camera);image=r.render().copy()
            Image.fromarray(image).save(self.folder/'images'/f'{index:06d}-{camera}.png')
            c=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_CAMERA,camera)
            transform=np.eye(4);transform[:3,:3]=d.cam_xmat[c].reshape(3,3)@np.diag([1,-1,-1]);transform[:3,3]=d.cam_xpos[c]
            focal=128/np.tan(np.deg2rad(m.cam_fovy[c])/2)
            calibration[camera]=dict(timestamp_s=float(d.time),T_world_optical=transform.tolist(),K=[[focal,0,128],[0,focal,128],[0,0,1]])
        # Raw input states retain their actual clock. They are not desired poses.
        (self.folder/'images'/f'{index:06d}-calibration.json').write_text(json.dumps(calibration))
        return dict(index=index,native_step=packet['step'],time_s=packet['time_s'])

    def close(self):
        if self.renderer is not None:self.renderer.close()


def run(root,native,startup,episode):
    import mujoco
    import rclpy
    from rclpy.qos import qos_profile_sensor_data,QoSProfile,DurabilityPolicy
    from rosidl_runtime_py.convert import message_to_ordereddict
    from std_msgs.msg import String
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import JointState
    from ocs2_msgs.msg import MpcObservation
    from .night_geometry import xml_scene,check_frozen_robot
    from .task_plan import pilot_cards
    from .task_contacts import allowed
    from .base_intent import native_document,measured_base,rotation,ACCEL
    from go2arx5_vla.task_data.codec import pose10
    from scripts.pi0292.durable import OwnerLock,write
    root,native,startup=map(Path,(root,native,startup));manifest,parent=validate_entry(root,physical_asset=VERSION)
    card=require_card(root,manifest,episode)
    if card['role']=='DEVELOPMENT' and sum(x.get('role')=='DEVELOPMENT' for x in read(root/'NIGHT_RESOURCE_LEDGER.json')['task_attempts'])>=12:raise PermissionError('NIGHT_DEVELOPMENT_START_CAP')
    from .night_teacher import Teacher
    from .mp_client import PlannerFailure
    series=root/'factory'
    frame_cap=round(card['duration_s']/.05)+1
    step_cap=round((card['warmup_sim_s']+card['duration_s'])/.002)+3
    plan_path=series/(episode+'-FIXED_CARD.json')
    if plan_path.exists():
        if read(plan_path)['card']!=card:raise ValueError('NIGHT_FIXED_CARD_CHANGED')
    else:write(plan_path,dict(card=card,source_sha=manifest['code']['vla_commit'],failed_seed_replacement=False))
    out=series/'episodes'/episode;engineering_attempt=1
    if out.exists():raise FileExistsError('FIRST_ATTEMPT_PRESERVED_NO_RETRY')
    out.parent.mkdir(exist_ok=True)
    build_dir=root/'runtime/TASK_ADAPTER_BUILD_03';build=read(build_dir/'BUILD_RECEIPT.json')
    for name,digest in build['binaries'].items():
        if sha(build_dir/(name+'.so'))!=digest:raise ValueError('TASK_ADAPTER_CHANGED')
    if sha(Path(__file__).with_name('task_actuation.cpp'))!=build['sources']['task_actuation']:
        raise ValueError('TASK_ACTUATION_SOURCE_NOT_BUILT')
    clock=Path(manifest['dependencies']['native_clock_root']);clock_build=read(clock/'CLOCK_BUILD_RECEIPT.json')
    for path,digest in clock_build['source_pins'].items():
        if sha(native/path)!=digest:raise ValueError('FROZEN_NATIVE_SOURCE_CHANGED:'+path)
    for path,digest in clock_build['binary_sha256'].items():
        if sha(clock/path)!=digest:raise ValueError('FROZEN_NATIVE_BINARY_CHANGED')
    timing=root/'runtime/TASK_PAUSE_CLOCK_01';timing_build=read(timing/'CLOCK_BUILD_RECEIPT.json')
    if timing_build['original_clock_binaries']!=clock_build['binary_sha256']:
        raise ValueError('PAUSE_BASE_CLOCK_IDENTITY')
    if sha(timing/'rl_sim_mujoco')!=timing_build['binary_sha256']['rl_sim_mujoco']:
        raise ValueError('PAUSE_SIM_BINARY_CHANGED')
    cfg=read(startup/'startup-v4/config.json')
    for path,digest in cfg['initialization']['pinned_files'].items():
        if sha(path)!=digest:raise ValueError('QUALIFIED_NATIVE_STARTUP_CHANGED')
    lease=OwnerLock(native.parent/'PI0289_SIM_LEASE')
    owner=OwnerLock(parent/'NIGHT_RESOURCE_OWNER.lock');ledger=read(parent/'NIGHT_RESOURCE_LEDGER.json')
    if (ledger['CPU_reservations'] or ledger['main_starts']+1>72
            or ledger['main_integrations']+step_cap>4788216 or ledger['RGB_pairs']+frame_cap>184392
            or ledger['Core_CPU_s']+7200+21600>230400):raise PermissionError('CUMULATIVE_NATIVE_TASK_BUDGET')
    authority,series_ledger=require_adoption(root,manifest)
    if (series_ledger['starts']+1>72 or series_ledger['integrations']+step_cap>4788216
            or series_ledger['RGB_pairs']+frame_cap>184392 or ledger['Core_CPU_s']+7200+21600>series_ledger['CPU_cap']):
        raise PermissionError('ADOPTED_NEW_TEACHER_SERIES_BUDGET')
    cpus=sorted(os.sched_getaffinity(0))[-16:][:4]
    ticket=dict(stage=root.name+'/'+episode,kind='NATIVE_TASK',pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().split()[21],
        cpus=cpus,cores=4,cpu_s_cap=7200,physics_steps_cap=step_cap,rgb_pairs_cap=frame_cap,started_epoch=time.time())
    ledger['main_starts']+=1;ledger['main_integrations']+=step_cap;ledger['RGB_pairs']+=frame_cap
    series_ledger['starts']+=1;series_ledger['integrations']+=step_cap;series_ledger['RGB_pairs']+=frame_cap
    series_ledger['CPU_s']+=7200;series_ledger['active_ticket']=ticket
    write(series/'SERIES_LEDGER.json',series_ledger)
    ledger['CPU_reservations'][ticket['stage']]=ticket;write(parent/'NIGHT_RESOURCE_LEDGER.json',ledger);owner.close()
    os.sched_setaffinity(0,set(cpus));out.mkdir();(out/'images').mkdir()
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    (out/'tmp').mkdir();(out/'ros-logs').mkdir()
    os.environ['ROS_DOMAIN_ID']=str(card['ROS_DOMAIN_ID'])
    os.environ.update(TMPDIR=str(out/'tmp'),ROS_LOG_DIR=str(out/'ros-logs'),ROS_LOCALHOST_ONLY='1',ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST')
    write(out/'CARD.json',dict(card,task_asset_version=VERSION));write(out/'START.json',dict(ticket,task_asset_version=VERSION,engineering_attempt=engineering_attempt,source_sha=manifest['code']['vla_commit'],WBC_parameters_changed=False))
    before=resource.getrusage(resource.RUSAGE_CHILDREN);cpu0=time.process_time();wall=time.monotonic()
    processes=[];handles=[];latest={};counts={};subs=[];observed=[];commands=[];frames=[];futures=[]
    pool=ThreadPoolExecutor(max_workers=1);recorder=None;node=None
    result=dict(status='ENGINEERING_INCOMPLETE');ros_raw=(out/'ros.jsonl').open('x');sample_log=(out/'samples.jsonl').open('x');command_log=(out/'committed_intents.jsonl').open('x');pause_log=(out/'pause_intervals.jsonl').open('x')
    jsonable=lambda v: v.tolist() if isinstance(v,np.ndarray) else v
    try:
        binding=read(root/'INPUT_BINDINGS.private.json')
        text=xml_scene(card,binding['asset_root']);(out/'task-scene.private.xml').write_text(text)
        model=mujoco.MjModel.from_xml_string(text)
        original=mujoco.MjModel.from_binary_path(cfg['initialization']['compiled_model'])
        static=check_frozen_robot(original,model)
        d=mujoco.MjData(model);mujoco.mj_resetDataKeyframe(model,d,0)
        for name in ('x5_joint2','x5_joint3'):
            j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name);d.qpos[model.jnt_qposadr[j]]=.9
        mujoco.mj_forward(model,d)
        if any(not allowed(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom2)) for c in d.contact):
            raise ValueError('ZERO_STEP_INITIAL_CONTACT_REJECT')
        model_path=out/'task-scene.private.mjb';mujoco.mj_saveModel(model,str(model_path),None)
        from .night_geometry import initial_audit
        initial_proof=initial_audit(model,d,card)
        write(out/'PREEXECUTION.json',dict(initialization=initial_proof,status='EXACT_NATIVE_PROPERTIES_AND_TASK_INITIAL_CONTACTS_PASS',frozen_robot=static,model_sha256=sha(model_path),time_s=float(d.time),nq=model.nq,nu=model.nu))
        actual=ActualState(model);recorder=FrameRecorder(model_path,out)
        rclpy.init();node=rclpy.create_node('gr00t_task_'+episode.lower().replace('-','_'))
        def record(key,msg):
            latest[key]=msg;counts[key]=counts.get(key,0)+1
            ros_raw.write(json.dumps(dict(kind=key,monotonic_s=time.monotonic(),msg=message_to_ordereddict(msg)))+'\n')
        for key,topic,cls,qos in (
            ('odom','/go2_x5/slam/odom_mocap',Odometry,qos_profile_sensor_data),
            ('arm','/go2_x5/arm/state',JointState,qos_profile_sensor_data),
            ('ee','/go2_x5/ee/state',MpcObservation,10),('observation','/mobile_manipulator_mpc_observation',MpcObservation,10),
            ('mode','/go2_x5/arm/mode/state',String,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))):
            subs.append(node.create_subscription(cls,topic,lambda msg,k=key:record(k,msg),qos))
        keys=node.create_publisher(String,'/go2_x5/fsm/key',10);modes=node.create_publisher(String,'/go2_x5/arm/mode/request',10);refs=node.create_publisher(String,'/go2_x5/ee/trajectory_json',10)
        def publish(pub,value):
            msg=String();msg.data=value;pub.publish(msg)
        def control():
            try:return json.loads((out/'control.jsonl').read_text().splitlines()[-1])
            except (FileNotFoundError,IndexError,json.JSONDecodeError):return {}
        def spin(seconds,predicate=None):
            end=time.monotonic()+seconds
            while time.monotonic()<end:
                rclpy.spin_once(node,timeout_sec=.001)
                if time.time()>=min(CUTOFF,authority['new_work_cutoff_epoch']) or 4*(time.monotonic()-wall)>6900:raise TimeoutError('TASK_WINDOW_OR_BOUNDED_SERVICE_LIMIT')
                if any(p.poll() is not None for p in processes):raise RuntimeError('OWNED_NATIVE_PROCESS_EXIT')
                if predicate and predicate():return True
            return bool(predicate()) if predicate else True
        def launch(argv,name,extra=None):
            capsule=root/manifest['code']['capsule_relative']
            env=dict(os.environ,PYTHONPATH=str(capsule)+':'+str(capsule/'src')+':'+os.environ.get('PYTHONPATH',''),
                PATH=str(native.parent/'dependency-root/usr/bin')+':'+os.environ.get('PATH',''))
            env.update(extra or {});h=(out/(name+'.log')).open('x');handles.append(h)
            p=subprocess.Popen(argv,cwd=native,env=env,stdout=h,stderr=subprocess.STDOUT,start_new_session=True);processes.append(p);return p
        spin(.3)
        if any(name!=node.get_name() for name in node.get_node_names()):raise RuntimeError('TASK_ROS_DOMAIN_OCCUPIED')
        sys.path.insert(0,str(native/'install/share/go2_x5_ocs2/launch'))
        from ocs2_common import ocs2_codegen_dir
        lib=ocs2_codegen_dir('go2_x5_ocs2','arx5_ac1_floating')
        if not Path(lib).is_dir():raise ValueError('EXISTING_NATIVE_CODEGEN_REQUIRED')
        marker=out/'CONTACT_SAFETY_ARMED';marker.touch()
        env=dict(PI0288_PHYSICS_TRACE=str(out/'physics.jsonl'),PI0288_PHYSICS_CAP=str(step_cap),PI0288_CONTACT_SAFETY_MARKER=str(marker),
            PI0289_INITIALIZATION_TRACE=str(out/'initialization.jsonl'),PI0289_CONTROL_TRACE=str(out/'control.jsonl'),PI0288_COMPILED_MODEL=str(model_path),
            LD_PRELOAD=str(build_dir/'task_actuation.so')+':'+str(build_dir/'task_observer.so'),
            GR00T_TASK_GRIPPER_COMMAND=str(out/'gripper.bin'),GR00T_TASK_STATE=str(out/'state.bin'),GR00T_TASK_ERROR=str(out/'task_errors.txt'),
            GR00T_TASK_STEP_QUOTA=str(out/'step_quota'),GR00T_TASK_PAUSE_ACK=str(out/'pause_ack.json'))
        def quota(step):
            path=out/'step_quota';tmp=path.with_suffix('.pending');tmp.write_text(str(step)+'\n');os.replace(tmp,path)
        quota(2500)
        launch(['xvfb-run','-a',str(timing/'rl_sim_mujoco'),'go2_x5','scene'],'sim',env)
        params=['--ros-args','-p','use_sim_time:=true','-p','taskFile:='+str(native/'src/ctrl/go2_x5_ocs2/config/task_floating.info'),
            '-p','urdfFile:='+str(native/'src/ctrl/go2_x5_description/urdf/arx5_ac1_floating.urdf'),'-p','libFolder:='+lib]
        launch([str(native/'install/lib/ocs2_mobile_manipulator_ros/mobile_manipulator_mpc_node'),*params],'solver')
        launch([str(clock/'go2_x5_ocs2_node'),*params],'controller')
        launch([sys.executable,'-m','scripts.pi0294.persistent_teacher_receiver','--ros-args','-p','use_sim_time:=true','-p','mode:=trajectory'],
            'planner',dict(PI0294_PLANNER_TRACE=str(out/'planner.jsonl'),PI0294_EPISODE=episode))
        if not spin(30,lambda:counts.get('odom',0)>10 and counts.get('arm',0)>10):raise RuntimeError('PAUSED_NATIVE_FEEDBACK_UNAVAILABLE')
        publish(modes,'HOLD')
        if not spin(3,lambda:control().get('arm_mode')=='HOLD' and control().get('fsm')=='RLFSMStateRLLocomotion'):
            raise RuntimeError('ORIGINAL_PAUSED_STANDBY_REQUIRED')
        if control().get('running') or control().get('sim_time_s')!=0 or (out/'physics.jsonl').exists():raise ValueError('PHYSICS_BEFORE_LEGAL_STANDBY')
        from .night_geometry import actual_zero_audit
        actual_zero_audit(out,card,d.qpos)
        write(out/'BEFORE_FIRST_INTEGRATION.json',control());publish(keys,'ENTER')
        from .night_safety import JournalRangeGuard
        range_guard=JournalRangeGuard(model,out/'state.bin.journal')
        def paused(step):
            try:return read(out/'pause_ack.json')['step']==step and read_state(out/'state.bin')['step']==step
            except (FileNotFoundError,ValueError):return False
        if not spin(35,lambda:paused(2500)):raise RuntimeError('BOUNDED_NATIVE_WARMUP_UNAVAILABLE')
        range_guard.check()
        publish(modes,'WBC')
        if not spin(20,lambda:latest.get('mode',String()).data=='WBC' and 'observation' in latest and 'ee' in latest
                and 0<=actual_source_age(5.,float(latest['ee'].time))<=.025
                and 0<=actual_source_age(5.,float(latest['observation'].time))<=.025):raise RuntimeError('NATIVE_WBC_MODE_UNAVAILABLE')
        first_step=2500
        start_s=first_step*.002;teacher=None;initial_mount=None;previous_velocity=np.zeros(3);next_step=first_step;last_command_source=-1.;sequence=0
        planner_path=out/'planner.jsonl';planner_offset=0;planner_rows=[]
        def planner_events():
            nonlocal planner_offset
            if planner_path.exists():
                with planner_path.open() as f:
                    f.seek(planner_offset)
                    while True:
                        line=f.readline()
                        if not line or not line.endswith('\n'):break
                        planner_rows.append(json.loads(line));planner_offset=f.tell()
            return planner_rows
        def stop_external(stopped):
            for p in processes[1:3]:os.killpg(p.pid,signal.SIGSTOP if stopped else signal.SIGCONT)
            if stopped:
                if not spin(3,lambda:all('\nState:\tT' in Path(f'/proc/{p.pid}/status').read_text() for p in processes[1:3])):
                    raise RuntimeError('OWNED_CONTROLLER_FREEZE_ACK_MISSING')
        active_intent=None
        while len(observed)<frame_cap:
            if not spin(10,lambda:paused(next_step)):raise RuntimeError('EXACT_20HZ_PAUSE_MISSING')
            range_guard.check()
            packet=read_state(out/'state.bin')
            if packet['step']!=next_step:raise ValueError('ACTUAL_20HZ_FRAME_MISSING_NO_INTERPOLATION')
            if not spin(5,lambda:('ee' in latest and 'observation' in latest
                    and 0<=actual_source_age(packet['time_s'],float(latest['ee'].time))<=.025
                    and 0<=actual_source_age(packet['time_s'],float(latest['observation'].time))<=.025)):
                raise ValueError('ACTUAL_PAUSED_ROS_SOURCE_CLOCK_NOT_BOUND')
            stop_external(True);pause_before=read(out/'pause_ack.json');pause_started=time.monotonic()
            # One same-state transaction; no source timestamp refresh while computing.
            state=actual.update(packet)
            state['time_s']=(packet['step']-first_step)*.002
            imu=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'imu');imu_pose=np.eye(4);imu_pose[:3,:3]=actual.data.site_xmat[imu].reshape(3,3);imu_pose[:3,3]=actual.data.site_xpos[imu]
            velocity=np.zeros(6);mujoco.mj_objectVelocity(model,actual.data,mujoco.mjtObj.mjOBJ_SITE,imu,velocity,0)
            mount,measured=measured_base(imu_pose,imu_pose[:3,:3].T@velocity[3:],imu_pose[:3,:3].T@velocity[:3],[.07557,0,.05768])
            state['actual_BASE_mount']=mount.copy()
            if teacher is None:
                teacher=Teacher(card,state,actual,mount,root=root)
                write(out/'TASK_START.json',dict(time_s=start_s,native_step=first_step,actual_TCP=state['tcp'].tolist(),actual_object=state['object_pose'].tolist(),persistent_station=teacher.station.tolist()))
            teacher.observe(state,mount)
            state['teacher_stage']=teacher.stage;state['teacher_geometry']=teacher.geometry
            state['active_intent_sequence']=None if active_intent is None else active_intent['sequence']
            state['executed_prefix_index']=None if active_intent is None else (len(observed)-active_intent['index']-1)%8
            if len(observed)==frame_cap-1 or teacher.stop_reason:
                sample_log.write(json.dumps(state,default=jsonable)+'\n');sample_log.flush();observed.append(state)
                while sum(not f.done() for f in futures)>=8:time.sleep(.01)
                futures.append(pool.submit(recorder.save,len(observed)-1,packet))
                break
            query=len(observed)%8==0
            if not query:
                prefix_index=len(observed)-active_intent['index']
                opening=float(active_intent['action_EEF13'][prefix_index][9])
                write_gripper(out/'gripper.bin',len(observed),round(float(packet['time_s'])*1e9)/1e9,opening)
                command_log.write(json.dumps(dict(phase='GRIPPER_PREFIX20HZ',index=len(observed),sequence=len(observed),intent_sequence=active_intent['sequence'],prefix_index=prefix_index,source_mpc_s=float(packet['time_s']),opening=opening))+'\n');command_log.flush()
                pause_after=finish_pause(pause_before,next_step,lambda:read_state(out/'state.bin')['step'],
                    lambda:stop_external(False),lambda:spin(2,lambda:read(out/'pause_ack.json')['ack_sequence']>pause_before['ack_sequence']),
                    lambda:read(out/'pause_ack.json'))
                pause_log.write(json.dumps(dict(index=len(observed),before=pause_before,after=pause_after,query=False,
                    wall_wait_s=time.monotonic()-pause_started,non_realtime=True,owned_MPC_processes_frozen=True))+'\n');pause_log.flush()
                sample_log.write(json.dumps(state,default=jsonable)+'\n');sample_log.flush();observed.append(state)
                while sum(not f.done() for f in futures)>=8:time.sleep(.01)
                futures.append(pool.submit(recorder.save,len(observed)-1,packet))
                next_step+=25;quota(next_step);publish(keys,'ENTER');continue
            try:
                if 4*(time.monotonic()-wall)+280>7200:raise TimeoutError('MPS2_CPU_RESERVATION_CLOSEOUT')
                plan=teacher.plan(state,mount)
                state['teacher_geometry']=teacher.geometry
            except PlannerFailure as failure:
                teacher.stop_reason=("RESOURCE_INCOMPLETE: " if 'BUDGET' in str(failure) or 'CAP' in str(failure) else "VALID_PLANNING_FAILURE: ")+str(failure)
                write(out/"PLANNING_FAILURE.json",dict(reason=str(failure),evidence=failure.evidence,query_index=len(commands),native_step=packet["step"]))
                sample_log.write(json.dumps(state,default=jsonable)+"\n");sample_log.flush();observed.append(state)
                while sum(not f.done() for f in futures)>=8:time.sleep(.01)
                futures.append(pool.submit(recorder.save,len(observed)-1,packet))
                break
            if initial_mount is None:initial_mount=mount.copy()
            gripper_fraction=teacher_physical_fraction(plan['opening']);require_working_target(gripper_fraction)
            action=np.zeros((40,13));action[:,:10]=np.stack([pose10(p,g) for p,g in zip(plan['poses'],gripper_fraction)])
            action[:,10:13]=plan['BASE_soft_velocity']
            # Timestamp comes from the very same actual native packet used for
            # Teacher geometry, not a refreshed label or older asynchronous EE.
            source=round(float(packet['time_s'])*1e9)/1e9
            if (source<=last_command_source or not 0<=actual_source_age(packet['time_s'],source)<=.025):raise ValueError('ACTUAL_OBSERVATION_SOURCE_CLOCK_NOT_BOUND')
            from go2arx5_vla.contracts.trajectory import TrajectoryChunkV1
            from go2arx5_vla.safety.trajectory_gate import TrajectorySafetyGateV1
            from go2arx5_vla.trajectory.so3 import matrix_to_quat
            chunk=TrajectoryChunkV1(chunk_id=episode+'/'+str(sequence),source=VERSION,observation_timestamp=source,
                point_timestamps=source+np.arange(40)*.05,eef_position=plan['poses'][:,:3,3],
                eef_orientation=matrix_to_quat(plan['poses'][:,:3,:3]),source_frame='world',tcp_semantic_id='native_x5_ee',
                validity_start=source,validity_end=source+2.,gripper=action[:,9])
            gate=TrajectorySafetyGateV1().evaluate(chunk,now=source,achieved_position=state['tcp'][:3,3],
                achieved_orientation=matrix_to_quat(state['tcp'][:3,:3]))
            if gate.decision.value=='REJECT':raise PlannerFailure('ORIGINAL_SAFETY_GATE_REJECT',gate.as_dict())
            document=native_document(action,mount,base_valid=True,previous_velocity=previous_velocity)
            document['_pi0294']=dict(episode=episode,sequence=sequence,source_mpc_s=source,ttl_s=.4,operation='install')
            wire=json.dumps(document,separators=(',',':'))
            # A SIGSTOPped DDS peer can stall native publication under its sim
            # mutex. Re-open that owned peer before requiring a fresh ACK;
            # physical quota/flags remain paused. Compare the pre-wait counters
            # to the now-live snapshot, not a stale file during peer suspension.
            pause_after=finish_pause(pause_before,next_step,lambda:read_state(out/'state.bin')['step'],
                lambda:stop_external(False),lambda:spin(2,lambda:read(out/'pause_ack.json')['ack_sequence']>pause_before['ack_sequence']),
                lambda:read(out/'pause_ack.json'))
            pause_log.write(json.dumps(dict(index=len(observed),before=pause_before,after=pause_after,wall_wait_s=time.monotonic()-pause_started,non_realtime=True,owned_MPC_processes_frozen=True))+'\n');pause_log.flush()
            write_gripper(out/'gripper.bin',len(observed),source,float(action[0,9]));publish(refs,wire)
            intent=dict(index=len(observed),sequence=sequence,actual_observation_time_s=packet['time_s'],source_mpc_s=source,
                wire_sha256=hashlib.sha256(wire.encode()).hexdigest(),action_EEF13=action.tolist(),teacher_stage=teacher.stage,
                actual_BASE_mount=mount.tolist(),actual_BASE_planar_velocity=measured.tolist(),base_valid=True)
            from .mp_window import SavedWindow
            window=SavedWindow(action);intent['scheduled_prefix_EEF13']=window.take(8).tolist()
            intent['saved_window_after_scheduled_prefix']=window.snapshot()
            intent['canonical_safety_gate']=gate.as_dict()
            intent.update(prefix_points=8,execute_physics_steps=200,record_every_physics_steps=25,
                persistent=plan['persistent'],geometry=plan['geometry'],milestones=plan['milestones'],
                asynchronous_EE_source_s=float(latest['ee'].time),selected_MPC_policy_consumption='UNKNOWN')
            intent['source_clock']='EXISTING_NATIVE_ROS_INTEGER_NANOSECONDS_OF_SAME_ACTUAL_PACKET'
            command_log.write(json.dumps(dict(intent,phase='SUBMITTED_REQUIRES_INSTALL_ACK'))+'\n');command_log.flush();commands.append(intent)
            ref_id=intent['wire_sha256']
            def installed():
                rows=planner_events()
                if any(x['kind']=='reject' and x.get('wire_sha256')==ref_id for x in rows):raise ValueError('NATIVE_REFERENCE_TRANSACTION_REJECTED')
                return any(x['kind']=='accept' and x.get('ref_id')==ref_id for x in rows) and any(x['kind']=='installed' and x.get('ref_id')==ref_id for x in rows)
            if not spin(5,installed):raise RuntimeError('REFERENCE_INSTALL_ACK_MISSING_NO_PHYSICS_ADVANCE')
            command_log.write(json.dumps(dict(index=intent['index'],phase='INSTALL_ACK',wire_sha256=ref_id,native_step=packet['step']))+'\n');command_log.flush()
            sample_log.write(json.dumps(state,default=jsonable)+'\n');sample_log.flush();observed.append(state)
            while sum(not f.done() for f in futures)>=8:time.sleep(.01)
            futures.append(pool.submit(recorder.save,len(observed)-1,packet))
            active_intent=intent;last_command_source=source;previous_velocity=action[7,10:13].copy();sequence+=1;next_step+=25
            quota(next_step);publish(keys,'ENTER')
        if not paused(next_step):raise RuntimeError('ACTUAL_NATIVE_FINAL_PAUSE_ACK_MISSING')
        result=dict(status='RAW_TASK_COMPLETE_REQUIRES_INDEPENDENT_CPU_AUDIT' if not teacher.stop_reason else 'VALID_PLANNING_FAILURE_NO_RETRY',
            stop_reason=teacher.stop_reason,observations=len(observed),submitted_intents=len(commands),teacher_events=teacher.events,
            final_stage=teacher.stage,development_milestones=teacher.milestones)
    except PlannerFailure as exc:
        result=dict(status='VALID_PLANNING_OR_GATE_FAIL_NO_RETRY',error=str(exc),observations=len(observed),submitted_intents=len(commands),failure_evidence=exc.evidence)
    except Exception as exc:
        result=dict(status='ENGINEERING_OR_NATIVE_SAFETY_INCOMPLETE',error=type(exc).__name__+': '+str(exc),observations=len(observed),submitted_intents=len(commands))
    finally:
        for p in processes:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGCONT);os.killpg(p.pid,signal.SIGINT)
        for p in processes:
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for h in handles:h.close()
        for f in (ros_raw,sample_log,command_log,pause_log):f.close()
        try:
            for future in futures:frames.append(future.result(timeout=120))
            if recorder:pool.submit(recorder.close).result(timeout=10)
        except Exception as exc:result['recorder_error']=type(exc).__name__+': '+str(exc);result['status']='DATA_RECORDER_INCOMPLETE'
        pool.shutdown(wait=True)
        if node:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        rows=[json.loads(line) for line in (out/'physics.jsonl').read_text().splitlines()] if (out/'physics.jsonl').exists() else []
        stops=[row for row in rows if row['kind']=='stop'];samples=[row for row in rows if row['kind']=='sample']
        lower=max([row['step'] for row in samples]+[row['steps'] for row in stops]+[0])
        try:actual_steps=read_state(out/'state.bin')['step']
        except (FileNotFoundError,ValueError):actual_steps=lower
        charged=min(step_cap,max(actual_steps,lower)+(0 if stops else 3)) if max(actual_steps,lower) else 0
        safety=[row for row in stops if row['reason']!='PHYSICS_RESERVATION_CAP']
        if (out/'HIGH_RATE_RANGE_FAILURE.json').exists():
            safety=safety+[dict(reason='RAW_MECHANICAL_RANGE_EXCESS',receipt=read(out/'HIGH_RATE_RANGE_FAILURE.json'))]
        if safety:
            result['status']='VALID_NATIVE_SAFETY_FAIL_TASK_VERSION_STOPPED'
            write(out/'NATIVE_SAFETY_FAIL.json',dict(episode=episode,version=VERSION,stops=safety,source_sha=manifest['code']['vla_commit'],original_evidence_preserved=True))
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu=time.process_time()-cpu0+after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
        service=time.monotonic()-wall;cpu_charge=max(cpu,4*service)
        result.update(native_stops=stops,physics_lower_bound=lower,actual_plugin_steps=actual_steps,physics_charged=charged,
            saved_RGB_pairs=len(frames),actual_CPU_s=cpu,conservative_CPU_charged_s=cpu_charge,service_s=service,
            owned_returncodes=[p.returncode for p in processes],source_sha=manifest['code']['vla_commit'],high_level_model_queries=0,selected='NONE',deployment='NONE')
        write(out/'RESULT.json',result)
        owner=OwnerLock(parent/'NIGHT_RESOURCE_OWNER.lock');ledger=read(parent/'NIGHT_RESOURCE_LEDGER.json')
        ledger['main_integrations']-=step_cap-charged;ledger['RGB_pairs']-=frame_cap-len(frames);ledger['Core_CPU_s']+=cpu_charge
        series_ledger=read(series/'SERIES_LEDGER.json');series_ledger['integrations']-=step_cap-charged
        series_ledger['RGB_pairs']-=frame_cap-len(frames);series_ledger['CPU_s']-=7200-cpu_charge
        series_ledger.pop('active_ticket');series_ledger['attempts'].append(dict(ticket,result=result))
        write(series/'SERIES_LEDGER.json',series_ledger)
        ledger['native_safety_failures']+=int(bool(safety))
        ledger['CPU_reservations'].pop(ticket['stage']);ledger.setdefault('task_attempts',[]).append(dict(ticket,result=result,role=card['role'],teacher_version=card['teacher_version']))
        write(parent/'NIGHT_RESOURCE_LEDGER.json',ledger);owner.close();lease.close()
    print(json.dumps(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','native','startup','episode'):p.add_argument('--'+name,required=True)
    a=p.parse_args();run(a.root,a.native,a.startup,a.episode)
