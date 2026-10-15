"""Call the existing native GetUp lifecycle; never write native leg commands."""
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import numpy as np

from .night_r1_runtime import read,sha,validate_entry,resource_lock,CUTOFF
from .night_r1_quota import bind,quota_bound
from .night_safety import JournalRangeGuard
from .task_state import read_state
from scripts.pi0292.durable import OwnerLock,write


def command_ready(row, expected, fsm):
    if (row.get('sim_time_s') != 0 or row.get('running') is not False
            or row.get('fsm') != fsm or row.get('arm_mode') != 'HOLD'):
        return False
    for key, values in [('q_target',expected['default_dof_pos'][:12]+[0,.9,.9,0,0,0]),
                        ('kp',expected['fixed_kp']),('kd',expected['fixed_kd'])]:
        actual=np.asarray(row.get(key,[]),dtype=float)
        if actual.shape!=(18,) or not np.isfinite(actual).all() or not np.allclose(actual,values,atol=1e-6,rtol=0):return False
    return True


def run(root,native,startup,episode):
    import mujoco
    import rclpy
    from rclpy.qos import qos_profile_sensor_data,QoSProfile,DurabilityPolicy
    from rosidl_runtime_py.convert import message_to_ordereddict
    from std_msgs.msg import String
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import JointState
    from ocs2_msgs.msg import MpcObservation
    from .night_geometry import xml_scene,initial_data,initial_audit,actual_zero_audit,check_frozen_robot
    from .task_plan import pilot_cards
    root,native,startup=map(Path,(root,native,startup));m,_=validate_entry(root,physical_asset='STARTUP')
    cpu_proof=read(root/'evidence/STARTUP_CPU_CONTRACTS.json')
    if cpu_proof['status']!='PASS_CPU_SUPPORTED_STARTUP_ADMISSION' or cpu_proof['startup_source_sha256']!=sha(__file__):
        raise PermissionError('R1_STARTUP_CHANGED_CONTRACT_TEST_REQUIRED')
    if episode not in ('TFNR1_STARTUP_OPEN_00','TFNR1_STARTUP_OPEN_01'):
        raise PermissionError('R1_FIXED_OPEN_STARTUP_SLOTS_ONLY')
    card=pilot_cards()[0]
    card.update(id=episode,role='STARTUP',split='DEVELOPMENT',duration_s=0,warmup_sim_s=5,
                table_top_m=.47,initial_finger_q_m=.042,ROS_DOMAIN_ID=3741 if episode.endswith('00') else 3742)
    out=root/'factory/episodes'/episode
    if out.exists():raise FileExistsError('R1_STARTUP_FIRST_ATTEMPT_PRESERVED')
    cfg=read(startup/'startup-v4/config.json')
    for name,h in cfg['core_hashes'].items():
        if sha(native/name)!=h:raise ValueError('FROZEN_NATIVE_CORE:'+name)
    for path,h in cfg['initialization']['pinned_files'].items():
        if sha(path)!=h:raise ValueError('FROZEN_STARTUP_PIN:'+path)
    clock=Path(m['dependencies']['native_clock_root']);timing=root/'runtime/TASK_PAUSE_CLOCK_01'
    clock_build=read(clock/'CLOCK_BUILD_RECEIPT.json');timing_build=read(timing/'CLOCK_BUILD_RECEIPT.json')
    assert timing_build['original_clock_binaries']==clock_build['binary_sha256']
    for path,h in clock_build['source_pins'].items():assert sha(native/path)==h
    for path,h in clock_build['binary_sha256'].items():assert sha(clock/path)==h
    assert sha(timing/'rl_sim_mujoco')==timing_build['binary_sha256']['rl_sim_mujoco']
    build_dir=root/'runtime/TASK_ADAPTER_R1_01';build=read(build_dir/'BUILD_RECEIPT.json')
    for name,h in build['binaries'].items():assert sha(build_dir/(name+'.so'))==h
    assert sha(Path(__file__).with_name('task_actuation_r1.cpp'))==build['sources']['task_actuation']
    model=mujoco.MjModel.from_xml_string(xml_scene(card,read(root/'INPUT_BINDINGS.private.json')['asset_root']))
    original=mujoco.MjModel.from_binary_path(cfg['initialization']['compiled_model'])
    static=check_frozen_robot(original,model);d=initial_data(model);initial=initial_audit(model,d,card)
    from .task_contacts import allowed
    assert all(allowed(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom2)) for c in d.contact)
    expected=read(root/'evidence/STARTUP_EXPECTED_NATIVE_COMMAND.private.json')['parameters']
    leg_nominal=np.asarray(expected['default_dof_pos'][:12])
    assert np.all(np.isfinite(leg_nominal)) and np.all(np.asarray(expected['fixed_kp'][:12])>0)
    for i,q in enumerate(leg_nominal):
        actuator=expected['joint_mapping'][i];joint=int(model.actuator_trnid[actuator,0])
        if not model.jnt_range[joint,0]<=q<=model.jnt_range[joint,1]:
            raise ValueError('SUPPORTED_NATIVE_NOMINAL_COMMAND_OUT_OF_ORIGINAL_LIMIT')
    lease=OwnerLock(native.parent/'PI0289_SIM_LEASE')
    owner=resource_lock(root);v=read(root/'R1_RESOURCE_LEDGER.json')
    assert len([x for x in v['task_attempts'] if x['role']=='STARTUP'])<4
    occupied={c for x in v['CPU_reservations'].values() for c in x['cpus']}
    cpus=[c for c in sorted(os.sched_getaffinity(0)) if c not in occupied][-8:][:4]
    cap=quota_bound(5,0)
    assert len(cpus)==4 and len(occupied)+4<=8 and v['main_starts']+1<=52 and v['main_integrations']+cap<=3458156
    assert v['Core_CPU_s']+sum(x['cpu_s_cap'] for x in v['CPU_reservations'].values())+900+21600<=read(root/'R1_ADOPTION.json')['effective_R1_CPU_cap_s']
    ticket=dict(stage=episode,kind='NATIVE_STARTUP',role='STARTUP',episode=episode,run_uuid=root.name,pid=os.getpid(),
        proc_start=Path('/proc/self/stat').read_text().split()[21],cpus=cpus,cores=4,cpu_s_cap=900,physics_steps_cap=cap,rgb_pairs_cap=0,started_epoch=time.time())
    v['main_starts']+=1;v['main_integrations']+=cap;v['CPU_reservations'][episode]=ticket;write(root/'R1_RESOURCE_LEDGER.json',v)
    out.mkdir();extra=bind(root,out,card,ticket);owner.close();os.sched_setaffinity(0,cpus)
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));(out/'tmp').mkdir();(out/'ros-logs').mkdir()
    os.environ.update(ROS_DOMAIN_ID=str(card['ROS_DOMAIN_ID']),ROS_LOCALHOST_ONLY='1',ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST',TMPDIR=str(out/'tmp'),ROS_LOG_DIR=str(out/'ros-logs'))
    model_path=out/'task-scene.private.mjb';mujoco.mj_saveModel(model,str(model_path),None)
    write(out/'PREEXECUTION.json',dict(initialization=initial,frozen_robot=static,model_sha256=sha(model_path),physics_steps=0))
    write(out/'CARD.json',card);write(out/'START.json',dict(ticket,source_sha=m['code']['vla_commit'],native_WBC_changed=False))
    before=resource.getrusage(resource.RUSAGE_CHILDREN);cpu0=time.process_time();wall=time.monotonic();processes=[];handles=[];node=None
    result=dict(status='ENGINEERING_INCOMPLETE');counts={};latest={};subs=[];raw=(out/'ros.jsonl').open('x')
    range_guard=JournalRangeGuard(model,out/'state.bin.journal')
    try:
        rclpy.init();node=rclpy.create_node('pi0294_r1_'+episode.lower())
        def record(key,msg):
            counts[key]=counts.get(key,0)+1;latest[key]=msg
            raw.write(json.dumps(dict(kind=key,monotonic_s=time.monotonic(),msg=message_to_ordereddict(msg)))+'\n')
        for key,topic,cls,qos in [('odom','/go2_x5/slam/odom_mocap',Odometry,qos_profile_sensor_data),('arm','/go2_x5/arm/state',JointState,qos_profile_sensor_data),('ee','/go2_x5/ee/state',MpcObservation,10),('observation','/mobile_manipulator_mpc_observation',MpcObservation,10),('mode','/go2_x5/arm/mode/state',String,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))]:
            subs.append(node.create_subscription(cls,topic,lambda msg,k=key:record(k,msg),qos))
        keys=node.create_publisher(String,'/go2_x5/fsm/key',10);modes=node.create_publisher(String,'/go2_x5/arm/mode/request',10)
        def control():
            try:return json.loads((out/'control.jsonl').read_text().splitlines()[-1])
            except (FileNotFoundError,IndexError,json.JSONDecodeError):return {}
        def publish(pub,key):
            msg=String();msg.data=key;pub.publish(msg)
            with (out/'STARTUP_CALLS.jsonl').open('a') as f:f.write(json.dumps(dict(value=key,monotonic_s=time.monotonic(),sim_time_s=control().get('sim_time_s'),native_control_calls=control().get('control_calls')))+'\n')
        def spin(seconds,predicate=None):
            end=time.monotonic()+seconds
            while time.monotonic()<end:
                rclpy.spin_once(node,timeout_sec=.001)
                if (out/'state.bin.journal').exists():range_guard.check()
                if time.time()>=CUTOFF or 4*(time.monotonic()-wall)+40>900:raise TimeoutError('R1_STARTUP_RESOURCE_OR_WINDOW')
                if any(p.poll() is not None for p in processes):raise RuntimeError('OWNED_NATIVE_PROCESS_EXIT')
                if predicate and predicate():return True
            return bool(predicate()) if predicate else True
        def launch(argv,name,extra_env=None):
            env=dict(os.environ);env.update(extra_env or {});h=(out/(name+'.log')).open('x');handles.append(h)
            p=subprocess.Popen(argv,cwd=native,env=env,stdout=h,stderr=subprocess.STDOUT,start_new_session=True);processes.append(p)
            with (out/'PROCESS_IDENTITIES.jsonl').open('a') as f:f.write(json.dumps(dict(role=name,pid=p.pid,start=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],monotonic_s=time.monotonic(),run_uuid=root.name))+'\n')
            return p
        def quota(step):
            assert 0<step<=cap
            temp=out/'step_quota.pending';temp.write_text(str(step)+'\n');os.replace(temp,out/'step_quota')
        def paused(step):
            try:return read(out/'pause_ack.json')['step']==step and read_state(out/'state.bin')['step']==step
            except (FileNotFoundError,ValueError):return False
        spin(.3)
        if any(n!=node.get_name() for n in node.get_node_names()):raise RuntimeError('R1_FRESH_ROS_DOMAIN_REQUIRED')
        sys.path.insert(0,str(native/'install/share/go2_x5_ocs2/launch'));from ocs2_common import ocs2_codegen_dir
        lib=ocs2_codegen_dir('go2_x5_ocs2','arx5_ac1_floating');assert Path(lib).is_dir()
        (out/'CONTACT_SAFETY_ARMED').touch();quota(2450)
        env=dict(extra,PI0288_PHYSICS_TRACE=str(out/'physics.jsonl'),PI0288_PHYSICS_CAP=str(cap),PI0288_CONTACT_SAFETY_MARKER=str(out/'CONTACT_SAFETY_ARMED'),PI0289_INITIALIZATION_TRACE=str(out/'initialization.jsonl'),PI0289_CONTROL_TRACE=str(out/'control.jsonl'),PI0288_COMPILED_MODEL=str(model_path),LD_PRELOAD=str(build_dir/'task_actuation.so')+':'+str(build_dir/'task_observer.so'),GR00T_TASK_GRIPPER_COMMAND=str(out/'gripper.bin'),GR00T_TASK_STATE=str(out/'state.bin'),GR00T_TASK_ERROR=str(out/'task_errors.txt'),GR00T_TASK_STEP_QUOTA=str(out/'step_quota'),GR00T_TASK_PAUSE_ACK=str(out/'pause_ack.json'))
        launch(['xvfb-run','-a',str(timing/'rl_sim_mujoco'),'go2_x5','scene'],'sim',env)
        params=['--ros-args','-p','use_sim_time:=true','-p','taskFile:='+str(native/'src/ctrl/go2_x5_ocs2/config/task_floating.info'),'-p','urdfFile:='+str(native/'src/ctrl/go2_x5_description/urdf/arx5_ac1_floating.urdf'),'-p','libFolder:='+lib]
        launch([str(native/'install/lib/ocs2_mobile_manipulator_ros/mobile_manipulator_mpc_node'),*params],'solver')
        launch([str(clock/'go2_x5_ocs2_node'),*params],'controller')
        if not spin(30,lambda:counts.get('odom',0)>10 and counts.get('arm',0)>10):raise RuntimeError('NATIVE_FEEDBACK_UNAVAILABLE')
        publish(modes,'HOLD')
        if not spin(3,lambda:control().get('arm_mode')=='HOLD'):raise RuntimeError('NATIVE_ARM_HOLD_ACK_MISSING')
        publish(keys,'0')
        if not spin(4,lambda:command_ready(control(),expected,'RLFSMStateGetUp')):raise RuntimeError('SUPPORTED_GETUP_COMMAND_NOT_READY')
        write(out/'GETUP_NATIVE_READY.json',control());old_calls=control()['control_calls'];publish(keys,'1')
        if not spin(10,lambda:control().get('control_calls',0)>old_calls and command_ready(control(),expected,'RLFSMStateRLLocomotion')):raise RuntimeError('NATIVE_LOCOMOTION_INITIAL_COMMAND_NOT_RETAINED')
        if (out/'physics.jsonl').exists():raise ValueError('PHYSICS_BEFORE_NATIVE_READY')
        actual_zero_audit(out,card,d.qpos);write(out/'BEFORE_FIRST_INTEGRATION.json',control());publish(keys,'ENTER')
        if not spin(35,lambda:paused(2450)):raise RuntimeError('BOUNDED_NATIVE_STARTUP_4_9S_NOT_COMPLETE')
        write(out/'BEFORE_WBC_ENTRY.json',dict(control=control(),pause=read(out/'pause_ack.json')))
        publish(modes,'WBC')
        if not spin(3,lambda:latest.get('mode',String()).data=='WBC'):raise RuntimeError('WBC_MODE_REQUEST_NOT_ACKNOWLEDGED')
        quota(2500);publish(keys,'ENTER')
        if not spin(5,lambda:paused(2500)):raise RuntimeError('BOUNDED_FIVE_SECOND_STARTUP_NOT_COMPLETE')
        if not spin(3,lambda:control().get('arm_mode')=='WBC' and latest.get('mode',String()).data=='WBC'):raise RuntimeError('ACTUAL_NATIVE_WBC_MODE_NOT_CONSUMED')
        range_guard.check();packet=read_state(out/'state.bin');assert packet['step']==2500
        write(out/'ACTUAL_STARTUP_END.json',dict(control=control(),pause=read(out/'pause_ack.json'),checked_packets=range_guard.checked,native_step=packet['step'],time_s=packet['time_s'],actual_qpos=packet['qpos'].tolist()))
        result=dict(status='RAW_COMPLETE_STARTUP_REQUIRES_INDEPENDENT_AUDIT',native_mode='WBC',full_safe_sim_s=5.,task_submissions=0,task_RGB=0)
    except Exception as exc:result=dict(status='ENGINEERING_OR_NATIVE_SAFETY_INCOMPLETE',error=type(exc).__name__+': '+str(exc),task_submissions=0,task_RGB=0)
    finally:
        for p in processes:
            if p.poll() is None:os.killpg(p.pid,signal.SIGCONT);os.killpg(p.pid,signal.SIGINT)
        for p in processes:
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for h in handles:h.close()
        raw.close()
        if node:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        rows=[json.loads(l) for l in (out/'physics.jsonl').read_text().splitlines()] if (out/'physics.jsonl').exists() else []
        stops=[x for x in rows if x.get('kind')=='stop'];lower=max([x.get('step',x.get('steps',0)) for x in rows]+[0])
        try:steps=read_state(out/'state.bin')['step']
        except (FileNotFoundError,ValueError):steps=lower
        safety=[x for x in stops if x.get('reason')!='PHYSICS_RESERVATION_CAP']
        if (out/'HIGH_RATE_RANGE_FAILURE.json').exists():safety.append(read(out/'HIGH_RATE_RANGE_FAILURE.json'))
        # Catch any unread tail even if a process exit interrupted live polling.
        try:
            if (out/'state.bin.journal').exists():range_guard.check()
        except ValueError:
            proof=read(out/'HIGH_RATE_RANGE_FAILURE.json')
            if proof not in safety:safety.append(proof)
        if safety:
            result['status']='VALID_NATIVE_OR_HIGH_RATE_SAFETY_FAIL_ALL_PHYSICS_STOP'
            write(root/'R1_SHARED_SAFETY_STOP.json',dict(episode=episode,safety=safety,old_night_failures=1,additional_failures=1,total_night_failures=2))
        after=resource.getrusage(resource.RUSAGE_CHILDREN);cpu=time.process_time()-cpu0+after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
        charged=min(cap,max(steps,lower)+(0 if safety else 3)) if max(steps,lower) else 0
        result.update(actual_integrations=steps,charged_integrations=charged,native_stops=stops,high_rate_packets=range_guard.checked,actual_CPU_s=cpu,conservative_CPU_charged_s=max(cpu,4*(time.monotonic()-wall)),owned_returncodes=[p.returncode for p in processes],owned_PIDs=[p.pid for p in processes],source_sha=m['code']['vla_commit'],new_GR00T_queries=0)
        write(out/'RESULT.json',result)
        owner=resource_lock(root);v=read(root/'R1_RESOURCE_LEDGER.json');v['main_integrations']-=cap-charged;v['Core_CPU_s']+=result['conservative_CPU_charged_s'];v['native_safety_failures']+=int(bool(safety));v['CPU_reservations'].pop(episode);v['task_attempts'].append(dict(ticket,result=result));write(root/'R1_RESOURCE_LEDGER.json',v);owner.close();lease.close()
    print(json.dumps(result),flush=True);return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    for name in ['root','native','startup','episode']:p.add_argument('--'+name,required=True)
    a=p.parse_args();run(a.root,a.native,a.startup,a.episode)
