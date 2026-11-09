"""One native full-body task attempt; frozen WBC, actual states and references.

No hold RMS qualification, hidden joint controller, reset during task, or model
query. All physics/CPU/camera usage spends the cumulative parent R4 ledger.
"""
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
from .task_capture_runtime import validate_entry,CUTOFF,read,sha
from .task_state import ActualState,write_gripper,read_state
from .task_gripper_contract import VERSION,teacher_physical_fraction,require_working_target
from .task_clock import finish_pause,preinstall_clock_failure,actual_source_age


class FrameRecorder:
    """Render a read-only copy of an actual recorded native state, never step it."""
    def __init__(self,model_path,folder):
        self.model_path,self.folder=model_path,folder
        self.model=self.data=self.renderer=None

    def save(self,index,packet):
        import mujoco
        from PIL import Image
        from go2arx5_vla.world_model.native_collection_performance_v1 import measure
        if self.model is None:
            with measure(self,'camera.native_model_load'):
                self.model=mujoco.MjModel.from_binary_path(str(self.model_path))
                self.data=mujoco.MjData(self.model)
            with measure(self,'camera.native_renderer_create'):
                self.renderer=mujoco.Renderer(self.model,height=256,width=256)
        m,d,r=self.model,self.data,self.renderer
        if getattr(self,'reset_each_packet',False):mujoco.mj_resetData(m,d)
        d.qpos[:]=packet['qpos'];d.qvel[:]=packet['qvel'];d.ctrl[:]=packet['ctrl'];d.time=packet['time_s']
        with measure(self,'camera.native_FK'):mujoco.mj_forward(m,d)
        calibration={}
        for camera in ('body_rgb','wrist_rgb'):
            with measure(self,'camera.native_render.'+camera):
                r.update_scene(d,camera=camera);image=r.render().copy()
            with measure(self,'camera.native_PNG.'+camera):
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
    from .task_plant import xml_scene,check_frozen_robot
    from .task_plan import pilot_cards
    from .task_contacts import allowed
    from .base_intent import native_document,measured_base,rotation,ACCEL
    from go2arx5_vla.task_data.codec import pose10
    from scripts.pi0292.durable import OwnerLock,write
    root,native,startup=map(Path,(root,native,startup));manifest,parent=validate_entry(root,physical_asset=VERSION)
    card=[c for c in pilot_cards() if c['id']==episode]
    if len(card)!=1:raise ValueError('FIXED_PILOT_CARD_REQUIRED')
    card=card[0]
    plan_path=root/'plans/FULLBODY_PILOT4_FROZEN.json'
    if plan_path.exists():
        if read(plan_path)['cards']!=pilot_cards():raise ValueError('PILOT_PLAN_CHANGED')
    else:write(plan_path,dict(cards=pilot_cards(),source_sha=manifest['code']['vla_commit'],failed_seed_replacement=False))
    out=root/'episodes'/episode;engineering_attempt=1
    while out.exists():
        prior=read(out/'RESULT.json')
        predispatched=(not prior.get('physics_charged',0) and not prior.get('submitted_intents',0)
            and not prior.get('native_stops') and prior.get('status')=='ENGINEERING_OR_NATIVE_SAFETY_INCOMPLETE')
        if not (predispatched or preinstall_clock_failure(out)):
            raise FileExistsError('FIRST_PHYSICAL_ATTEMPT_EXISTS_NO_QUALITY_RETRY')
        engineering_attempt+=1
        # W0/T2 had one original dispatch, followed by at most the protocol's
        # three reasoned clock repairs. All other predispatch caps stay three.
        maximum=4 if episode=='TASK-PILOT-W0-T2' else 3
        if engineering_attempt>maximum:raise PermissionError('BOUNDED_ENGINEERING_REPAIRS_EXHAUSTED')
        out=root/'episodes'/(episode+f'-engineering{engineering_attempt:02d}')
    build_dir=root/'runtime/TASK_ADAPTER_BUILD_03';build=read(build_dir/'BUILD_RECEIPT.json')
    for name,digest in build['binaries'].items():
        if sha(build_dir/(name+'.so'))!=digest:raise ValueError('TASK_ADAPTER_CHANGED')
    if sha(Path(__file__).with_name('task_actuation.cpp'))!=build['sources']['task_actuation']:
        raise ValueError('TASK_ACTUATION_SOURCE_NOT_BUILT')
    clock=parent/'fullbody/clock-v1';clock_build=read(clock/'CLOCK_BUILD_RECEIPT.json')
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
    owner=OwnerLock(parent/'R4_RESOURCE_OWNER.lock');ledger=read(parent/'R4_RESOURCE_LEDGER.json')
    if (ledger['CPU_reservations'] or ledger['main_starts']+1>160
            or ledger['main_integrations']+20000>2560000 or ledger['RGB_pairs']+641>102560
            or ledger['Core_CPU_s']+3600>96*3600):raise PermissionError('CUMULATIVE_NATIVE_TASK_BUDGET')
    cpus=sorted(os.sched_getaffinity(0))[-16:][:4]
    ticket=dict(stage=root.name+'/'+episode,kind='NATIVE_TASK',pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().split()[21],
        cpus=cpus,cores=4,cpu_s_cap=3600,physics_steps_cap=20000,rgb_pairs_cap=641,started_epoch=time.time())
    ledger['main_starts']+=1;ledger['main_integrations']+=20000;ledger['RGB_pairs']+=641
    ledger['CPU_reservations'][ticket['stage']]=ticket;write(parent/'R4_RESOURCE_LEDGER.json',ledger);owner.close()
    os.sched_setaffinity(0,set(cpus));out.mkdir();(out/'images').mkdir()
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    (out/'tmp').mkdir();(out/'ros-logs').mkdir()
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
        write(out/'PREEXECUTION.json',dict(status='EXACT_NATIVE_PROPERTIES_AND_TASK_INITIAL_CONTACTS_PASS',frozen_robot=static,model_sha256=sha(model_path),time_s=float(d.time),nq=model.nq,nu=model.nu))
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
                if time.time()>=CUTOFF or time.monotonic()-wall>600:raise TimeoutError('TASK_WINDOW_OR_BOUNDED_SERVICE_LIMIT')
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
        env=dict(PI0288_PHYSICS_TRACE=str(out/'physics.jsonl'),PI0288_PHYSICS_CAP='20000',PI0288_CONTACT_SAFETY_MARKER=str(marker),
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
        launch([sys.executable,'-m','scripts.pi0294.native_receiver','--ros-args','-p','use_sim_time:=true','-p','mode:=trajectory'],
            'planner',dict(PI0294_PLANNER_TRACE=str(out/'planner.jsonl'),PI0294_EPISODE=episode))
        if not spin(30,lambda:counts.get('odom',0)>10 and counts.get('arm',0)>10):raise RuntimeError('PAUSED_NATIVE_FEEDBACK_UNAVAILABLE')
        publish(modes,'HOLD')
        if not spin(3,lambda:control().get('arm_mode')=='HOLD' and control().get('fsm')=='RLFSMStateRLLocomotion'):
            raise RuntimeError('ORIGINAL_PAUSED_STANDBY_REQUIRED')
        if control().get('running') or control().get('sim_time_s')!=0 or (out/'physics.jsonl').exists():raise ValueError('PHYSICS_BEFORE_LEGAL_STANDBY')
        write(out/'BEFORE_FIRST_INTEGRATION.json',control());publish(keys,'ENTER')
        def paused(step):
            try:return read(out/'pause_ack.json')['step']==step and read_state(out/'state.bin')['step']==step
            except (FileNotFoundError,ValueError):return False
        if not spin(35,lambda:paused(2500)):raise RuntimeError('BOUNDED_NATIVE_WARMUP_UNAVAILABLE')
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
        while len(observed)<641:
            if not spin(10,lambda:paused(next_step)):raise RuntimeError('EXACT_20HZ_PAUSE_MISSING')
            packet=read_state(out/'state.bin')
            if packet['step']!=next_step:raise ValueError('ACTUAL_20HZ_FRAME_MISSING_NO_INTERPOLATION')
            if not spin(5,lambda:('ee' in latest and 'observation' in latest
                    and 0<=actual_source_age(packet['time_s'],float(latest['ee'].time))<=.025
                    and 0<=actual_source_age(packet['time_s'],float(latest['observation'].time))<=.025)):
                raise ValueError('ACTUAL_PAUSED_ROS_SOURCE_CLOCK_NOT_BOUND')
            stop_external(True);pause_before=read(out/'pause_ack.json');pause_started=time.monotonic()
            # One same-state transaction; no source timestamp refresh while computing.
            state=actual.update(packet)
            if teacher is None:
                from scripts.pi0291_arx5_data.teacher import Teacher as PickTeacher
                from .teacher import PushTeacher
                teacher=PickTeacher(card,state['object_pose']) if card['task']=='T1' else PushTeacher(card,state['object_pose'],actual)
                write(out/'TASK_START.json',dict(time_s=start_s,native_step=first_step,actual_TCP=state['tcp'].tolist(),actual_object=state['object_pose'].tolist()))
            state['time_s']=packet['time_s']-start_s
            teacher.observe(state)
            if len(observed)==640:
                sample_log.write(json.dumps(state,default=jsonable)+'\n');sample_log.flush();observed.append(state)
                futures.append(pool.submit(recorder.save,len(observed)-1,packet))
                break
            plan=teacher.plan(state)
            imu=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'imu');imu_pose=np.eye(4);imu_pose[:3,:3]=actual.data.site_xmat[imu].reshape(3,3);imu_pose[:3,3]=actual.data.site_xpos[imu]
            velocity=np.zeros(6);mujoco.mj_objectVelocity(model,actual.data,mujoco.mjtObj.mjOBJ_SITE,imu,velocity,0)
            mount,measured=measured_base(imu_pose,imu_pose[:3,:3].T@velocity[3:],imu_pose[:3,:3].T@velocity[:3],[.07557,0,.05768])
            if initial_mount is None:initial_mount=mount.copy()
            gripper_fraction=teacher_physical_fraction(plan['opening']);require_working_target(gripper_fraction)
            action=np.zeros((40,13));action[:,:10]=np.stack([pose10(p,g) for p,g in zip(plan['poses'],gripper_fraction)])
            goal=initial_mount[:3,3]+np.array([card['base_translation_preference_m'],0,0])
            yaw=np.arctan2(mount[1,0],mount[0,0]);desired=rotation(-yaw)@(goal-mount[:3,3]);desired=np.r_[np.clip(desired[:2],-.06,.06),0.]
            for i in range(40):
                previous=previous_velocity if i==0 else action[i-1,10:13]
                action[i,10:13]=previous+np.clip(desired-previous,-ACCEL*.05,ACCEL*.05)
            source=float(latest['ee'].time)
            if (source<=last_command_source or not 0<=actual_source_age(packet['time_s'],source)<=.025):raise ValueError('ACTUAL_OBSERVATION_SOURCE_CLOCK_NOT_BOUND')
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
            write_gripper(out/'gripper.bin',sequence,source,float(action[0,9]));publish(refs,wire)
            intent=dict(index=len(observed),sequence=sequence,actual_observation_time_s=packet['time_s'],source_mpc_s=source,
                wire_sha256=hashlib.sha256(wire.encode()).hexdigest(),action_EEF13=action.tolist(),teacher_stage=teacher.stage,
                actual_BASE_mount=mount.tolist(),actual_BASE_planar_velocity=measured.tolist(),base_valid=True)
            command_log.write(json.dumps(dict(intent,phase='SUBMITTED_REQUIRES_INSTALL_ACK'))+'\n');command_log.flush();commands.append(intent)
            ref_id=intent['wire_sha256']
            def installed():
                rows=planner_events()
                if any(x['kind']=='reject' and x.get('wire_sha256')==ref_id for x in rows):raise ValueError('NATIVE_REFERENCE_TRANSACTION_REJECTED')
                return any(x['kind']=='accept' and x.get('ref_id')==ref_id for x in rows) and any(x['kind']=='installed' and x.get('ref_id')==ref_id for x in rows)
            if not spin(5,installed):raise RuntimeError('REFERENCE_INSTALL_ACK_MISSING_NO_PHYSICS_ADVANCE')
            command_log.write(json.dumps(dict(index=intent['index'],phase='INSTALL_ACK',wire_sha256=ref_id,native_step=packet['step']))+'\n');command_log.flush()
            sample_log.write(json.dumps(state,default=jsonable)+'\n');sample_log.flush();observed.append(state)
            futures.append(pool.submit(recorder.save,len(observed)-1,packet))
            last_command_source=source;previous_velocity=action[0,10:13].copy();sequence+=1;next_step+=25
            quota(next_step);publish(keys,'ENTER')
        if not paused(next_step):raise RuntimeError('ACTUAL_NATIVE_FINAL_PAUSE_ACK_MISSING')
        result=dict(status='RAW_TASK_COMPLETE_REQUIRES_INDEPENDENT_CPU_AUDIT',observations=len(observed),submitted_intents=len(commands),teacher_events=teacher.events,final_stage=teacher.stage)
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
        charged=min(20000,max(actual_steps,lower)+(0 if stops else 3)) if max(actual_steps,lower) else 0
        safety=[row for row in stops if row['reason']!='PHYSICS_RESERVATION_CAP']
        if safety:
            result['status']='VALID_NATIVE_SAFETY_FAIL_TASK_VERSION_STOPPED'
            write(root/'NATIVE_SAFETY_FAIL_V2.json',dict(episode=episode,version=VERSION,stops=safety,source_sha=manifest['code']['vla_commit'],original_evidence_preserved=True))
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu=time.process_time()-cpu0+after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
        service=time.monotonic()-wall;cpu_charge=max(cpu,4*service)
        result.update(native_stops=stops,physics_lower_bound=lower,actual_plugin_steps=actual_steps,physics_charged=charged,
            saved_RGB_pairs=len(frames),actual_CPU_s=cpu,conservative_CPU_charged_s=cpu_charge,service_s=service,
            owned_returncodes=[p.returncode for p in processes],source_sha=manifest['code']['vla_commit'],high_level_model_queries=0,selected='NONE',deployment='NONE')
        write(out/'RESULT.json',result)
        owner=OwnerLock(parent/'R4_RESOURCE_OWNER.lock');ledger=read(parent/'R4_RESOURCE_LEDGER.json')
        ledger['main_integrations']-=20000-charged;ledger['RGB_pairs']-=641-len(frames);ledger['Core_CPU_s']+=cpu_charge
        ledger['CPU_reservations'].pop(ticket['stage']);ledger.setdefault('task_attempts',[]).append(dict(ticket,result=result))
        write(parent/'R4_RESOURCE_LEDGER.json',ledger);owner.close();lease.close()
    print(json.dumps(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','native','startup','episode'):p.add_argument('--'+name,required=True)
    a=p.parse_args();run(a.root,a.native,a.startup,a.episode)
