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
import struct
import subprocess
import sys
import time
import numpy as np
from .task_capture_runtime import validate_entry,CUTOFF,read,sha
from .task_state import ActualState,parse_state,write_gripper,read_state


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
    from ocs2_msgs.msg import MpcObservation
    from .task_plant import xml_scene,check_frozen_robot
    from .task_plan import pilot_cards
    from .task_contacts import allowed
    from .base_intent import native_document,measured_base,rotation,ACCEL
    from go2arx5_vla.task_data.codec import pose10
    from scripts.pi0292.durable import OwnerLock,write
    root,native,startup=map(Path,(root,native,startup));manifest,parent=validate_entry(root)
    card=[c for c in pilot_cards() if c['id']==episode]
    if len(card)!=1:raise ValueError('FIXED_PILOT_CARD_REQUIRED')
    card=card[0]
    plan_path=root/'plans/FULLBODY_PILOT4_FROZEN.json'
    if plan_path.exists():
        if read(plan_path)['cards']!=pilot_cards():raise ValueError('PILOT_PLAN_CHANGED')
    else:write(plan_path,dict(cards=pilot_cards(),source_sha=manifest['code']['vla_commit'],failed_seed_replacement=False))
    out=root/'episodes'/episode
    if out.exists():raise FileExistsError('FIRST_ATTEMPT_EXISTS_NO_QUALITY_RETRY')
    build_dir=root/'runtime/TASK_ADAPTER_BUILD_01';build=read(build_dir/'BUILD_RECEIPT.json')
    for name,digest in build['binaries'].items():
        if sha(build_dir/(name+'.so'))!=digest:raise ValueError('TASK_ADAPTER_CHANGED')
    if sha(Path(__file__).with_name('task_actuation.cpp'))!=build['sources']['task_actuation']:
        raise ValueError('TASK_ACTUATION_SOURCE_NOT_BUILT')
    clock=parent/'fullbody/clock-v1';clock_build=read(clock/'CLOCK_BUILD_RECEIPT.json')
    for path,digest in clock_build['source_pins'].items():
        if sha(native/path)!=digest:raise ValueError('FROZEN_NATIVE_SOURCE_CHANGED:'+path)
    for path,digest in clock_build['binary_sha256'].items():
        if sha(clock/path)!=digest:raise ValueError('FROZEN_NATIVE_BINARY_CHANGED')
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
    write(out/'CARD.json',card);write(out/'START.json',dict(ticket,source_sha=manifest['code']['vla_commit'],WBC_parameters_changed=False))
    before=resource.getrusage(resource.RUSAGE_CHILDREN);cpu0=time.process_time();wall=time.monotonic()
    processes=[];handles=[];latest={};counts={};subs=[];observed=[];commands=[];frames=[];futures=[]
    pool=ThreadPoolExecutor(max_workers=1);recorder=None;node=None;journal=None;journal_offset=0
    result=dict(status='ENGINEERING_INCOMPLETE');ros_raw=(out/'ros.jsonl').open('x');sample_log=(out/'samples.jsonl').open('x');command_log=(out/'committed_intents.jsonl').open('x')
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
            env=dict(os.environ,**(extra or {}));h=(out/(name+'.log')).open('x');handles.append(h)
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
            GR00T_TASK_GRIPPER_COMMAND=str(out/'gripper.bin'),GR00T_TASK_STATE=str(out/'state.bin'),GR00T_TASK_ERROR=str(out/'task_errors.txt'))
        launch(['/usr/bin/xvfb-run','-a',str(clock/'rl_sim_mujoco'),'go2_x5','scene'],'sim',env)
        params=['--ros-args','-p','use_sim_time:=true','-p','taskFile:='+str(native/'src/ctrl/go2_x5_ocs2/config/task_floating.info'),
            '-p','urdfFile:='+str(native/'src/ctrl/go2_x5_description/urdf/arx5_ac1_floating.urdf'),'-p','libFolder:='+lib]
        launch([str(native/'install/lib/ocs2_mobile_manipulator_ros/mobile_manipulator_mpc_node'),*params],'solver')
        launch([str(clock/'go2_x5_ocs2_node'),*params],'controller')
        launch([sys.executable,'-m','scripts.pi0294.native_receiver','--ros-args','-p','use_sim_time:=true','-p','mode:=trajectory'],
            'planner',dict(PI0294_PLANNER_TRACE=str(out/'planner.jsonl'),PI0294_EPISODE=episode))
        if not spin(30,lambda:counts.get('ee',0)>5):raise RuntimeError('PAUSED_NATIVE_FEEDBACK_UNAVAILABLE')
        publish(modes,'HOLD')
        if not spin(3,lambda:control().get('arm_mode')=='HOLD' and control().get('fsm')=='RLFSMStateRLLocomotion'):
            raise RuntimeError('ORIGINAL_PAUSED_STANDBY_REQUIRED')
        if control().get('running') or control().get('sim_time_s')!=0 or (out/'physics.jsonl').exists():raise ValueError('PHYSICS_BEFORE_LEGAL_STANDBY')
        write(out/'BEFORE_FIRST_INTEGRATION.json',control());publish(keys,'ENTER')
        if not spin(35,lambda:control().get('sim_time_s',0)>=5.):raise RuntimeError('BOUNDED_NATIVE_WARMUP_UNAVAILABLE')
        publish(modes,'WBC')
        if not spin(20,lambda:latest.get('mode',String()).data=='WBC' and counts.get('observation',0)>5):raise RuntimeError('NATIVE_WBC_MODE_UNAVAILABLE')
        journal=(out/'state.bin.journal').open('rb')
        def available():
            nonlocal journal_offset
            journal.seek(journal_offset);rows=[]
            while True:
                header=journal.read(4)
                if len(header)!=4:break
                size,=struct.unpack('<I',header);raw=journal.read(size)
                if len(raw)!=size:break
                journal_offset+=4+size
                packet=parse_state(raw)
                if packet['step']%25==0:rows.append(packet)
            return rows
        available() # Warmup is retained in the journal, outside task data.
        packet=read_state(out/'state.bin');first_step=((packet['step']//25)+1)*25
        start_s=first_step*.002;teacher=None;initial_mount=None;previous_velocity=np.zeros(3);next_step=first_step;last_command_source=-1.;sequence=0
        while len(observed)<641:
            spin(.001)
            for packet in available():
                if packet['step']<next_step:continue
                if packet['step']!=next_step:raise ValueError('ACTUAL_20HZ_FRAME_MISSING_NO_INTERPOLATION')
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
                action=np.zeros((40,13));action[:,:10]=np.stack([pose10(p,g) for p,g in zip(plan['poses'],plan['opening'])])
                goal=initial_mount[:3,3]+np.array([card['base_translation_preference_m'],0,0])
                yaw=np.arctan2(mount[1,0],mount[0,0]);desired=rotation(-yaw)@(goal-mount[:3,3]);desired=np.r_[np.clip(desired[:2],-.06,.06),0.]
                for i in range(40):
                    previous=previous_velocity if i==0 else action[i-1,10:13]
                    action[i,10:13]=previous+np.clip(desired-previous,-ACCEL*.05,ACCEL*.05)
                source=float(latest['ee'].time)
                if (source<=last_command_source or abs(source-packet['time_s'])>.025):raise ValueError('ACTUAL_OBSERVATION_SOURCE_CLOCK_NOT_BOUND')
                document=native_document(action,mount,base_valid=True,previous_velocity=previous_velocity)
                document['_pi0294']=dict(episode=episode,sequence=sequence,source_mpc_s=source,ttl_s=.4,operation='install')
                wire=json.dumps(document,separators=(',',':'))
                write_gripper(out/'gripper.bin',sequence,source,float(action[0,9]));publish(refs,wire)
                intent=dict(index=len(observed),sequence=sequence,actual_observation_time_s=packet['time_s'],source_mpc_s=source,
                    wire_sha256=hashlib.sha256(wire.encode()).hexdigest(),action_EEF13=action.tolist(),teacher_stage=teacher.stage,
                    actual_BASE_mount=mount.tolist(),actual_BASE_planar_velocity=measured.tolist(),base_valid=True)
                command_log.write(json.dumps(intent)+'\n');command_log.flush();commands.append(intent)
                sample_log.write(json.dumps(state,default=jsonable)+'\n');sample_log.flush();observed.append(state)
                futures.append(pool.submit(recorder.save,len(observed)-1,packet))
                last_command_source=source;previous_velocity=action[0,10:13].copy();sequence+=1;next_step+=25
                if len(observed)>=641:break
        publish(keys,'ENTER')
        if not spin(3,lambda:control().get('running') is False):raise RuntimeError('ACTUAL_NATIVE_PAUSE_ACK_MISSING')
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
        for f in (ros_raw,sample_log,command_log):f.close()
        if journal:journal.close()
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
            write(root/'NATIVE_SAFETY_FAIL.json',dict(episode=episode,stops=safety,source_sha=manifest['code']['vla_commit'],original_evidence_preserved=True))
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
