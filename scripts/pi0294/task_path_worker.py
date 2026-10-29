"""One approved S1 native attempt; original frozen binaries and safety.

Reuses the original full-body lifecycle and renderer. Costs spend only the
directly approved S1 child; closed legacy ledgers and leases remain unchanged.
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
from .task_path_runtime import validate_entry,require_card,reserve,settle,read,sha,cards
from .task_state import ActualState,write_gripper,read_state
from .task_gripper_contract import teacher_physical_fraction,require_working_target
from .task_path_teacher import VERSION,Teacher,NativePathClient,serial
from .task_gripper_contract import VERSION as PHYSICAL_VERSION
from .base_intent import eef_document
from .task_clock import finish_pause,preinstall_clock_failure,actual_source_age


def encode_record(value):
    """Preserve numeric values and bools at the native NumPy/JSON boundary."""
    return json.dumps(value, default=serial, allow_nan=False)


def publish_intent(log, intent, publish_effects):
    """Durably encode the exact command before gripper/ROS publication.

    A missing ACK stays unknown. A serialization or persistence failure cannot
    send a command that is absent from the submitted-intent journal.
    """
    line = encode_record(dict(intent, phase='SUBMISSION_INTENT_BEFORE_EFFECT'))+'\n'
    log.write(line); log.flush(); os.fsync(log.fileno())
    publish_effects()
    log.write(encode_record(dict(intent,phase='SUBMITTED_REQUIRES_INSTALL_ACK'))+'\n')
    log.flush(); os.fsync(log.fileno())
    return intent


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
            if image.shape!=(256,256,3) or image.dtype!=np.uint8:
                raise ValueError('S1_ACTUAL_DOUBLE_RGB_FORMAT')
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
    os.environ.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
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
    root,native,startup=map(Path,(root,native,startup));manifest,inputs=validate_entry(root)
    if str(native)!=inputs['native_root'] or str(startup)!=inputs['startup_root']:
        raise PermissionError('S1_EXACT_NATIVE_AND_STARTUP_BINDINGS')
    card=require_card(root,manifest,episode)
    series=root/'native-s1'
    frame_cap=round(card['duration_s']/.05)+1
    step_cap=round((card['warmup_sim_s']+card['duration_s'])/.002)+3
    plan_path=series/'FIXED_SERIES_CARDS.json'
    if plan_path.exists():
        if read(plan_path)['cards']!=cards():raise ValueError('NEW_TEACHER_FIXED_PLAN_CHANGED')
    else:raise PermissionError('S1_PRESEALED_FIXED_CARDS_REQUIRED')
    out=series/'episodes'/episode;engineering_attempt=1
    if out.exists():raise FileExistsError('FIRST_ATTEMPT_PRESERVED_NO_RETRY')
    out.parent.mkdir(exist_ok=True)
    build_dir=Path(inputs['task_adapter_build']);build=read(build_dir/'BUILD_RECEIPT.json')
    for name,digest in build['binaries'].items():
        if sha(build_dir/(name+'.so'))!=digest:raise ValueError('TASK_ADAPTER_CHANGED')
    if sha(Path(__file__).with_name('task_actuation.cpp'))!=build['sources']['task_actuation']:
        raise ValueError('TASK_ACTUATION_SOURCE_NOT_BUILT')
    clock=Path(inputs['clock_root']);clock_build=read(clock/'CLOCK_BUILD_RECEIPT.json')
    for path,digest in clock_build['source_pins'].items():
        if sha(native/path)!=digest:raise ValueError('FROZEN_NATIVE_SOURCE_CHANGED:'+path)
    for path,digest in clock_build['binary_sha256'].items():
        if sha(clock/path)!=digest:raise ValueError('FROZEN_NATIVE_BINARY_CHANGED')
    timing=Path(inputs['pause_root']);timing_build=read(timing/'CLOCK_BUILD_RECEIPT.json')
    if timing_build['original_clock_binaries']!=clock_build['binary_sha256']:
        raise ValueError('PAUSE_BASE_CLOCK_IDENTITY')
    if sha(timing/'rl_sim_mujoco')!=timing_build['binary_sha256']['rl_sim_mujoco']:
        raise ValueError('PAUSE_SIM_BINARY_CHANGED')
    cfg=read(startup/'startup-v4/config.json')
    for path,digest in cfg['initialization']['pinned_files'].items():
        if sha(path)!=digest:raise ValueError('QUALIFIED_NATIVE_STARTUP_CHANGED')
    lease=OwnerLock(native.parent/'PI0289_SIM_LEASE')
    ticket=reserve(root,card);cpus=ticket['cpus']
    os.sched_setaffinity(0,set(cpus));out.mkdir();(out/'images').mkdir()
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    (out/'tmp').mkdir();(out/'ros-logs').mkdir()
    os.environ.update(TMPDIR=str(out/'tmp'),ROS_LOG_DIR=str(out/'ros-logs'),ROS_LOCALHOST_ONLY='1',
        ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST',ROS_DOMAIN_ID=str(inputs['ROS_DOMAIN_ID']),
        MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
    write(out/'CARD.json',dict(card,task_asset_version=PHYSICAL_VERSION,teacher_version=VERSION))
    write(out/'START.json',dict(ticket,task_asset_version=PHYSICAL_VERSION,teacher_version=VERSION,
        engineering_attempt=engineering_attempt,source_sha=manifest['code']['vla_commit'],WBC_parameters_changed=False))
    before=resource.getrusage(resource.RUSAGE_CHILDREN);cpu0=time.process_time();wall=time.monotonic()
    processes=[];handles=[];latest={};counts={};subs=[];observed=[];commands=[];frames=[];futures=[]
    pool=ThreadPoolExecutor(max_workers=1);recorder=None;node=None
    result=dict(status='ENGINEERING_INCOMPLETE');ros_raw=(out/'ros.jsonl').open('x');sample_log=(out/'samples.jsonl').open('x');command_log=(out/'committed_intents.jsonl').open('x');pause_log=(out/'pause_intervals.jsonl').open('x')
    def save_frame(index,packet):
        future=pool.submit(recorder.save,index,packet);futures.append(future)
        if index==0:
            saved=future.result(timeout=15)
            write(out/'CURRENT_CAMERA_FIRST_ACTUAL_FRAME.json',dict(saved,
                status='FIRST_ACTUAL_DOUBLE_RGB_SAVED_BEFORE_TASK_ADVANCE',extra_RGB_pairs=0))
    try:
        text=xml_scene(card,inputs['asset_root']);(out/'task-scene.private.xml').write_text(text)
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
                if time.time()>=manifest['deadline_epoch']-45 or time.monotonic()-wall>600:
                    raise TimeoutError('S1_TASK_WINDOW_OR_BOUNDED_SERVICE_LIMIT')
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
        lib_source=Path(inputs['codegen_lib'])
        if not lib_source.is_dir():raise ValueError('EXISTING_NATIVE_CODEGEN_REQUIRED')
        if any(sha(path)!=pin for path,pin in inputs['codegen_files'].items()):
            raise ValueError('EXISTING_ORIGINAL_CODEGEN_SEED_CHANGED')
        # The frozen task.info requests recompilation. Preserve that exact
        # behavior, but give each owned process its own node-local cache so
        # neither the old seed nor another controller's cache is overwritten.
        libraries={}
        for role in ('solver','controller'):
            lib=out/('codegen-'+role);lib.mkdir();libraries[role]=str(lib)
            for name,pin in inputs['codegen_files'].items():
                source_path=Path(name)
                relative=source_path.relative_to(lib_source);target=lib/relative
                target.parent.mkdir(parents=True,exist_ok=True)
                body=source_path.read_bytes()
                if hashlib.sha256(body).hexdigest()!=pin:raise ValueError('CODEGEN_SEED_CHANGED_DURING_COPY')
                with target.open('xb')as f:f.write(body)
        write(out/'ORIGINAL_CODEGEN_SEEDS.json',dict(source=inputs['codegen_lib'],
            files=inputs['codegen_files'],owned_process_caches=libraries,
            original_recompileLibraries_setting_unchanged=True,old_cache_modified=False))
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
        def parameters(role):
            return ['--ros-args','-p','use_sim_time:=true','-p','taskFile:='+str(native/'src/ctrl/go2_x5_ocs2/config/task_floating.info'),
                '-p','urdfFile:='+str(native/'src/ctrl/go2_x5_description/urdf/arx5_ac1_floating.urdf'),'-p','libFolder:='+libraries[role]]
        launch([str(native/'install/lib/ocs2_mobile_manipulator_ros/mobile_manipulator_mpc_node'),*parameters('solver')],'solver')
        launch([str(clock/'go2_x5_ocs2_node'),*parameters('controller')],'controller')
        launch([sys.executable,'-m','scripts.pi0294.persistent_teacher_receiver','--ros-args','-p','use_sim_time:=true','-p','mode:=trajectory'],
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
        start_s=first_step*.002;teacher=None;next_step=first_step;last_command_source=-1.
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
        # Current native install/cancel/HOLD qualification shares this first
        # already-reserved start. No physical tick or gripper sequence is spent
        # by the two ROS qualification transactions while quota remains2500.
        packet=read_state(out/'state.bin');initial_state=actual.update(packet)
        from go2arx5_vla.task_data.codec import pose10
        hold_action=np.stack([pose10(initial_state['tcp'],teacher_physical_fraction(1.))]*40)
        hold=eef_document(hold_action);hold['_pi0294']=dict(episode=episode,sequence=0,
            source_mpc_s=5.,ttl_s=.4,operation='install')
        hold_wire=json.dumps(hold,separators=(',',':'));hold_id=hashlib.sha256(hold_wire.encode()).hexdigest()
        publish(refs,hold_wire)
        def qualification_installed():
            rows=planner_events()
            if any(r['kind']=='reject' and r.get('wire_sha256')==hold_id for r in rows):
                raise ValueError('S1_CURRENT_NATIVE_HOLD_PACKET_REJECT')
            return any(r['kind']=='accept' and r.get('ref_id')==hold_id for r in rows) and any(
                r['kind']=='installed' and r.get('ref_id')==hold_id for r in rows)
        if not spin(5,qualification_installed):raise RuntimeError('S1_CURRENT_INSTALL_ACK_NOT_PROVED')
        cancel=dict(_pi0294=dict(episode=episode,sequence=1,source_mpc_s=5.,ttl_s=.4,operation='cancel'))
        cancel_mode_count=counts.get('mode',0)
        publish(refs,json.dumps(cancel,separators=(',',':')))
        if not spin(5,lambda:any(r['kind']=='lifecycle' and r.get('reason')=='CANCEL' for r in planner_events())
                and counts.get('mode',0)>cancel_mode_count and latest.get('mode',String()).data=='HOLD'):
            raise RuntimeError('S1_CURRENT_CANCEL_AND_ORIGINAL_HOLD_NOT_PROVED')
        if not paused(2500):raise ValueError('S1_PHYSICS_ADVANCED_DURING_CURRENT_STOP_QUALIFICATION')
        reentry_mode_count=counts.get('mode',0)
        publish(modes,'WBC')
        # The native control trace is emitted on advancing simulation time.
        # At the exact pause it retains a pre-request HOLD row. Require a new
        # native mode acknowledgement for each transition, with zero new ticks.
        if not spin(5,lambda:counts.get('mode',0)>reentry_mode_count and latest.get('mode',String()).data=='WBC'):
            raise RuntimeError('S1_CURRENT_WBC_REENTRY_NOT_PROVED')
        if not paused(2500):raise ValueError('S1_PHYSICS_ADVANCED_DURING_CURRENT_REENTRY')
        write(out/'CURRENT_NATIVE_QUALIFICATION.json',dict(status='PASS_CURRENT_FEEDBACK_INSTALL_CANCEL_HOLD_AT_EXACT_PAUSE',
            native_step=2500,physics_integrations_during_qualification=0,hold_ref_id=hold_id,
            ROS_sequence_offset=2,first_gripper_sequence=0,WBC_parameters_changed=False,
            cancel_mode_count_before=cancel_mode_count,reentry_mode_count_before=reentry_mode_count,
            final_mode_count=counts.get('mode',0),last_control_trace=control()))
        while len(observed)<frame_cap:
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
            state['raw_time_s']=packet['time_s']
            state['time_s']=(packet['step']-first_step)*.002
            state['safety_valid']=bool(state['safety_valid'] and state['gripper_state_valid']
                and not state['joint_limit_penetration'])
            write(root/'control/NATIVE_OWNER_HEARTBEAT.json',dict(status='RUNNING_OWN_S1_NATIVE_ATTEMPT',
                episode=episode,pid=os.getpid(),proc_start=ticket['proc_start'],native_step=packet['step'],
                observed_epoch=time.time(),deadline_epoch=manifest['deadline_epoch']))
            state['object_table_contact']=any(
                {mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) for g in (c['geom0'],c['geom1'])}
                =={'task_table','task_cube_geom'} and c['wrench'][0]>=.05 for c in packet['contacts'])
            imu=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'imu');imu_pose=np.eye(4);imu_pose[:3,:3]=actual.data.site_xmat[imu].reshape(3,3);imu_pose[:3,3]=actual.data.site_xpos[imu]
            velocity=np.zeros(6);mujoco.mj_objectVelocity(model,actual.data,mujoco.mjtObj.mjOBJ_SITE,imu,velocity,0)
            mount,measured=measured_base(imu_pose,imu_pose[:3,:3].T@velocity[3:],imu_pose[:3,:3].T@velocity[:3],[.07557,0,.05768])
            state['actual_BASE_mount']=mount.copy()
            if teacher is None:
                client=NativePathClient(folder=out/'stage-paths',python=inputs['planner_python'],
                    pythonpath=inputs['planner_pythonpath'],code=root/manifest['code']['capsule_relative'],
                    model_path=model_path,model_sha256=sha(model_path),urdf=inputs['planner_urdf'],srdf=inputs['planner_srdf'])
                teacher=Teacher(card,actual,client,sequence_offset=2)
                write(out/'TASK_START.json',dict(time_s=start_s,native_step=first_step,
                    actual_TCP=state['tcp'].tolist(),actual_object=state['object_pose'].tolist(),BASE_supervision=False))
            teacher.observe(state)
            state['teacher_stage']=teacher.stage;state['teacher_geometry']=teacher.geometry
            state['active_intent_sequence']=None if active_intent is None else active_intent['sequence']
            state['executed_prefix_index']=None if active_intent is None else (len(observed)-active_intent['index']-1)%8
            if len(observed)==frame_cap-1 or teacher.stop_reason:
                sample_log.write(encode_record(state)+'\n');sample_log.flush();observed.append(state)
                save_frame(len(observed)-1,packet)
                break
            query=len(observed)%8==0
            if not query:
                pause_after=finish_pause(pause_before,next_step,lambda:read_state(out/'state.bin')['step'],
                    lambda:stop_external(False),lambda:spin(2,lambda:read(out/'pause_ack.json')['ack_sequence']>pause_before['ack_sequence']),
                    lambda:read(out/'pause_ack.json'))
                pause_log.write(json.dumps(dict(index=len(observed),before=pause_before,after=pause_after,query=False,
                    wall_wait_s=time.monotonic()-pause_started,non_realtime=True,owned_MPC_processes_frozen=True))+'\n');pause_log.flush()
                sample_log.write(encode_record(state)+'\n');sample_log.flush();observed.append(state)
                save_frame(len(observed)-1,packet)
                next_step+=25;quota(next_step);publish(keys,'ENTER');continue
            plan=teacher.propose();action=plan['action_EEF10'];require_working_target(action[:,9])
            # Timestamp comes from the very same actual native packet used for
            # Teacher geometry, not a refreshed label or older asynchronous EE.
            source=round(float(packet['time_s'])*1e9)/1e9
            if (source<=last_command_source or not 0<=actual_source_age(packet['time_s'],source)<=.025):raise ValueError('ACTUAL_OBSERVATION_SOURCE_CLOCK_NOT_BOUND')
            wire=plan['wire'];sequence=plan['sequence']
            if plan['source_mpc_s']!=source:raise ValueError('S1_REFERENCE_SOURCE_CLOCK_CHANGED')
            # A SIGSTOPped DDS peer can stall native publication under its sim
            # mutex. Re-open that owned peer before requiring a fresh ACK;
            # physical quota/flags remain paused. Compare the pre-wait counters
            # to the now-live snapshot, not a stale file during peer suspension.
            pause_after=finish_pause(pause_before,next_step,lambda:read_state(out/'state.bin')['step'],
                lambda:stop_external(False),lambda:spin(2,lambda:read(out/'pause_ack.json')['ack_sequence']>pause_before['ack_sequence']),
                lambda:read(out/'pause_ack.json'))
            pause_log.write(json.dumps(dict(index=len(observed),before=pause_before,after=pause_after,wall_wait_s=time.monotonic()-pause_started,non_realtime=True,owned_MPC_processes_frozen=True))+'\n');pause_log.flush()
            intent=dict(index=len(observed),sequence=sequence,actual_observation_time_s=packet['time_s'],source_mpc_s=source,
                wire_sha256=hashlib.sha256(wire.encode()).hexdigest(),action_EEF10=action.tolist(),teacher_stage=teacher.stage,
                actual_BASE_mount=mount.tolist(),actual_BASE_planar_velocity=measured.tolist(),base_valid=False,
                gripper_sequence=len(commands))
            intent.update(prefix_points=8,execute_physics_steps=200,record_every_physics_steps=25,
                path_sha256=plan['path_sha256'],geometry=teacher.geometry,milestones=teacher.milestones,
                reference_certificate=plan['reference_certificate'],object_hypothesis=plan['object_hypothesis'],
                asynchronous_EE_source_s=float(latest['ee'].time),selected_MPC_policy_consumption='UNKNOWN')
            intent['source_clock']='EXISTING_NATIVE_ROS_INTEGER_NANOSECONDS_OF_SAME_ACTUAL_PACKET'
            def publish_effects():
                write_gripper(out/'gripper.bin',len(commands),source,float(action[0,9]))
                publish(refs,wire)
            publish_intent(command_log,intent,publish_effects);commands.append(intent)
            ref_id=intent['wire_sha256']
            def installed():
                rows=planner_events()
                if any(x['kind']=='reject' and x.get('wire_sha256')==ref_id for x in rows):raise ValueError('NATIVE_REFERENCE_TRANSACTION_REJECTED')
                return any(x['kind']=='accept' and x.get('ref_id')==ref_id for x in rows) and any(x['kind']=='installed' and x.get('ref_id')==ref_id for x in rows)
            if not spin(5,installed):raise RuntimeError('REFERENCE_INSTALL_ACK_MISSING_NO_PHYSICS_ADVANCE')
            teacher.reference.acknowledge_install(ref_id=ref_id,native_step=packet['step'])
            command_log.write(json.dumps(dict(index=intent['index'],phase='INSTALL_ACK',wire_sha256=ref_id,native_step=packet['step']))+'\n');command_log.flush()
            sample_log.write(encode_record(state)+'\n');sample_log.flush();observed.append(state)
            save_frame(len(observed)-1,packet)
            active_intent=intent;last_command_source=source;next_step+=25
            quota(next_step);publish(keys,'ENTER')
        if not paused(next_step):raise RuntimeError('ACTUAL_NATIVE_FINAL_PAUSE_ACK_MISSING')
        result=dict(status='RAW_TASK_COMPLETE_REQUIRES_INDEPENDENT_CPU_AUDIT' if not teacher.stop_reason else 'VALID_TEACHER_NO_PROGRESS_FAIL',
            stop_reason=teacher.stop_reason,observations=len(observed),submitted_intents=len(commands),teacher_events=teacher.events,
            final_stage=teacher.stage,development_milestones=teacher.milestones)
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
        recorder_errors=[]
        for index,future in enumerate(futures):
            try:frames.append(future.result(timeout=120))
            except Exception as exc:recorder_errors.append(dict(index=index,error=type(exc).__name__+': '+str(exc)))
        try:
            if recorder:pool.submit(recorder.close).result(timeout=10)
        except Exception as exc:recorder_errors.append(dict(index='close',error=type(exc).__name__+': '+str(exc)))
        if recorder_errors:result['recorder_error']=recorder_errors;result['status']='DATA_RECORDER_INCOMPLETE'
        pool.shutdown(wait=True)
        if node:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        rows=[json.loads(line) for line in (out/'physics.jsonl').read_text().splitlines()] if (out/'physics.jsonl').exists() else []
        stops=[row for row in rows if row['kind']=='stop'];samples=[row for row in rows if row['kind']=='sample']
        lower=max([row['step'] for row in samples]+[row['steps'] for row in stops]+[0])
        try:actual_steps=read_state(out/'state.bin')['step']
        except (FileNotFoundError,ValueError):actual_steps=lower
        actual_lower=max(actual_steps,lower)
        charged=max(actual_lower,min(step_cap,actual_lower+(0 if stops else 3))) if actual_lower else 0
        safety=[row for row in stops if row['reason']!='PHYSICS_RESERVATION_CAP']
        if safety:
            result['status']='VALID_NATIVE_SAFETY_FAIL_TASK_VERSION_STOPPED'
            write(series/'NATIVE_SAFETY_FAIL.json',dict(episode=episode,version=VERSION,stops=safety,source_sha=manifest['code']['vla_commit'],original_evidence_preserved=True))
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu=time.process_time()-cpu0+after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
        service=time.monotonic()-wall;cpu_charge=max(cpu,4*service)+4.
        result.update(native_stops=stops,physics_lower_bound=lower,actual_plugin_steps=actual_steps,physics_charged=charged,
            saved_RGB_pairs=len(frames),RGB_pairs_charged=len(futures),actual_CPU_s=cpu,
            conservative_CPU_charged_s=cpu_charge,service_s=service,
            owned_returncodes=[p.returncode for p in processes],source_sha=manifest['code']['vla_commit'],high_level_model_queries=0,selected='NONE',deployment='NONE')
        # Final milestones can contain NumPy scalar predicates as well.
        result=json.loads(encode_record(result))
        write(out/'RESULT.json',result)
        settle(root,ticket,result);lease.close()
        write(root/'control/NATIVE_OWNER_HEARTBEAT.json',dict(status='OWN_S1_NATIVE_ATTEMPT_TERMINAL',
            episode=episode,pid=os.getpid(),result_sha256=sha(out/'RESULT.json'),
            observed_epoch=time.time(),deadline_epoch=manifest['deadline_epoch']))
    print(json.dumps(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','native','startup','episode'):p.add_argument('--'+name,required=True)
    a=p.parse_args();run(a.root,a.native,a.startup,a.episode)
