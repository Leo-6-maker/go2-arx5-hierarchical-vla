"""One source-bound native hold/clock check; not a manipulation/plant qualification."""
import argparse
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
from scripts.pi0292.durable import OwnerLock,read,write
from .r4a import validate_scope
from .r4 import CUTOFF


def main(root,native,startup):
    import rclpy
    from rclpy.qos import qos_profile_sensor_data,QoSProfile,DurabilityPolicy
    from rosidl_runtime_py.convert import message_to_ordereddict
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String,Float32MultiArray
    from trajectory_msgs.msg import JointTrajectory
    from ocs2_msgs.msg import MpcObservation,MpcFlattenedController,MpcTargetTrajectories
    from go2arx5_vla.trajectory.so3 import quat_to_matrix,matrix_to_quat
    from .base_intent import native_document
    root,native,startup=map(Path,(root,native,startup));manifest=validate_scope(root)
    # DDS must initialize before reserving a physical launch.
    rclpy.init()
    lease=OwnerLock(native.parent/'PI0289_SIM_LEASE')
    out=root/'fullbody/native-hold-01';out.mkdir(exist_ok=False)
    clock=root/'fullbody/clock-v1';cfg=read(startup/'startup-v4/config.json')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    build=read(clock/'CLOCK_BUILD_RECEIPT.json')
    for name,digest in build['source_pins'].items():
        if sha(native/name)!=digest:raise ValueError('FROZEN_CORE_CHANGED:'+name)
    for name,digest in build['binary_sha256'].items():
        if sha(clock/name)!=digest:raise ValueError('CLOCK_BUILD_CHANGED:'+name)
    initialization=cfg['initialization']
    for path,digest in initialization['pinned_files'].items():
        if sha(Path(path))!=digest:raise ValueError('QUALIFIED_STARTUP_DEPENDENCY_CHANGED')
    audit=json.loads(subprocess.check_output([initialization['audit_binary'],initialization['compiled_model']],text=True))
    write(out/'PREEXECUTION_AUDIT.json',audit)
    if audit['invalid_joint_count'] or audit['illegal_touching_contacts'] or not audit['reset_exact'] or not audit['finite']:
        raise ValueError('PREEXECUTION_INITIALIZATION_REJECT')
    receiver=read(root/'fullbody/NATIVE_RECEIVER_CPU.json')
    if receiver['status']!='ACTUAL_NATIVE_RECEIVER_CPU_PASS':raise PermissionError('ACTUAL_RECEIVER_CPU_REQUIRED')
    sys.path.insert(0,str(native/'install/share/go2_x5_ocs2/launch'))
    from ocs2_common import ocs2_codegen_dir
    lib=ocs2_codegen_dir('go2_x5_ocs2','arx5_ac1_floating')
    if not Path(lib).is_dir():raise ValueError('EXISTING_NATIVE_CODEGEN_REQUIRED')
    owner=OwnerLock(root/'R4_RESOURCE_OWNER.lock');ledger=read(root/'R4_RESOURCE_LEDGER.json')
    if (ledger['CPU_reservations'] or ledger['main_starts']+1>160
            or ledger['main_integrations']+20000>2560000 or ledger['Core_CPU_s']+2400>96*3600):
        raise PermissionError('NATIVE_PARENT_RESERVATION_EXCEEDED')
    cpus=sorted(os.sched_getaffinity(0))[-16:][:4];os.sched_setaffinity(0,set(cpus))
    ticket=dict(stage='FULLBODY_NATIVE_HOLD_01',kind='NATIVE',pid=os.getpid(),
        proc_start=Path('/proc/self/stat').read_text().split()[21],cpus=cpus,cores=4,cpu_s_cap=2400,
        physics_steps_cap=20000,started_epoch=time.time())
    ledger['main_starts']+=1;ledger['main_integrations']+=20000
    ledger['CPU_reservations'][ticket['stage']]=ticket
    write(root/'R4_RESOURCE_LEDGER.json',ledger);write(out/'START.json',dict(ticket,source_sha=manifest['code']['vla_commit'],
        build=build,native_domain=int(os.environ['ROS_DOMAIN_ID']),plant_overlay_ready=False,
        hold_XYZ_RMS_gate_m=.015,hold_SO3_RMS_gate_rad=.05,source_clock_gate_s=.025,
        no_WBC_parameters_changed=True,original_strict_observer_sha256=sha(startup/'runtime/observer.so')));owner.close()
    children=resource.getrusage(resource.RUSAGE_CHILDREN);cpu_start=time.process_time();wall=time.monotonic()
    processes=[];handles=[];latest={};counts={};subs=[];result=dict(status='ENGINEERING_INCOMPLETE')
    node=rclpy.create_node('pi0294_fullbody_owner');raw=(out/'ros.jsonl').open('w')
    def record(key,msg):
        latest[key]=msg;counts[key]=counts.get(key,0)+1
        raw.write(json.dumps(dict(kind=key,monotonic_s=time.monotonic(),msg=message_to_ordereddict(msg)))+'\n')
    for name,topic,cls,qos in (
        ('odom','/go2_x5/slam/odom_mocap',Odometry,qos_profile_sensor_data),
        ('arm','/go2_x5/arm/state',JointState,qos_profile_sensor_data),
        ('ee','/go2_x5/ee/state',MpcObservation,10),
        ('observation','/mobile_manipulator_mpc_observation',MpcObservation,10),
        ('policy','/mobile_manipulator_mpc_policy',MpcFlattenedController,10),
        ('target','/mobile_manipulator_mpc_target',MpcTargetTrajectories,10),
        ('arm_request','/go2_x5/arm/command/request',JointTrajectory,10),
        ('base','/go2_x5/base/command',Float32MultiArray,10),
        ('mode','/go2_x5/arm/mode/state',String,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))):
        subs.append(node.create_subscription(cls,topic,lambda msg,k=name:record(k,msg),qos))
    keys=node.create_publisher(String,'/go2_x5/fsm/key',10)
    modes=node.create_publisher(String,'/go2_x5/arm/mode/request',10)
    refs=node.create_publisher(String,'/go2_x5/ee/trajectory_json',1)
    def publish(pub,value):msg=String();msg.data=value;pub.publish(msg)
    def spin(seconds,predicate=None):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            rclpy.spin_once(node,timeout_sec=.002)
            if time.time()>=CUTOFF or time.monotonic()-wall>150:raise TimeoutError('ORIGINAL_CLOCK_OR_BOUNDED_SERVICE_LIMIT')
            dead=[dict(pid=p.pid,exit=p.poll()) for p in processes if p.poll() is not None]
            if dead:raise RuntimeError('NATIVE_PROCESS_EXIT:'+json.dumps(dead))
            if predicate and predicate():return True
        return bool(predicate()) if predicate else True
    def launch(argv,name,extra=None):
        env=dict(os.environ,**(extra or {}));h=(out/(name+'.log')).open('w');handles.append(h)
        p=subprocess.Popen(argv,cwd=native,env=env,stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(p);return p
    def control():
        try:return json.loads((out/'control.jsonl').read_text().splitlines()[-1])
        except (FileNotFoundError,IndexError,json.JSONDecodeError):return {}
    try:
        spin(.4)
        if any(x!='pi0294_fullbody_owner' for x in node.get_node_names()):raise ValueError('ISOLATED_ROS_DOMAIN_OCCUPIED')
        env=dict(PI0288_PHYSICS_TRACE=str(out/'physics.jsonl'),PI0288_PHYSICS_CAP='20000',
            PI0288_CONTACT_SAFETY_MARKER=str(out/'CONTACT_SAFETY_ARMED'),
            PI0289_INITIALIZATION_TRACE=str(out/'initialization.jsonl'),PI0289_CONTROL_TRACE=str(out/'control.jsonl'),
            PI0288_COMPILED_MODEL=initialization['compiled_model'],LD_PRELOAD=str(startup/'runtime/observer.so'))
        (out/'CONTACT_SAFETY_ARMED').touch()
        launch(['xvfb-run','-a',str(clock/'rl_sim_mujoco'),'go2_x5','scene'],'sim',env)
        params=['--ros-args','-p','use_sim_time:=true','-p','taskFile:='+str(native/'src/ctrl/go2_x5_ocs2/config/task_floating.info'),
            '-p','urdfFile:='+str(native/'src/ctrl/go2_x5_description/urdf/arx5_ac1_floating.urdf'),'-p','libFolder:='+lib]
        solver=launch([str(native/'install/lib/ocs2_mobile_manipulator_ros/mobile_manipulator_mpc_node'),*params],'solver')
        controller=launch([str(clock/'go2_x5_ocs2_node'),*params],'controller')
        planner=launch([sys.executable,'-m','scripts.pi0294.native_receiver','--ros-args','-p','use_sim_time:=true','-p','mode:=trajectory'],
            'planner',dict(PI0294_PLANNER_TRACE=str(out/'planner.jsonl'),PI0294_EPISODE=out.name))
        if not spin(30,lambda:counts.get('odom',0)>10 and counts.get('arm',0)>10):raise RuntimeError('PAUSED_SIM_FEEDBACK_UNAVAILABLE')
        publish(modes,'HOLD')
        if not spin(2,lambda:control().get('arm_mode')=='HOLD' and control().get('fsm')=='RLFSMStateRLLocomotion'):
            raise RuntimeError('QUALIFIED_PAUSED_LOCOMOTION_HOLD_UNAVAILABLE')
        before=control()
        if before['running'] or before['sim_time_s']!=0 or (out/'physics.jsonl').exists():raise ValueError('PHYSICS_ADVANCED_BEFORE_LEGAL_STANDBY')
        write(out/'BEFORE_FIRST_INTEGRATION.json',before);publish(keys,'ENTER')
        if not spin(30,lambda:control().get('sim_time_s',0)>=5):raise RuntimeError('BOUNDED_FIVE_SIM_SECONDS_WARMUP_UNAVAILABLE')
        publish(modes,'WBC')
        if not spin(20,lambda:counts.get('policy',0)>3 and counts.get('arm_request',0)>20 and latest.get('mode',String()).data=='WBC'):
            raise RuntimeError('NATIVE_WBC_START_UNAVAILABLE')
        initial=list(latest['ee'].state.value[:7]);t0=float(latest['ee'].time)
        write(out/'TASK_START.json',dict(initial_tcp=initial,mpc_time_s=t0,monotonic_s=time.monotonic()))
        action=np.zeros((40,13));action[:,:3]=initial[:3];action[:,3:9]=quat_to_matrix(initial[3:]).reshape(-1)[:6];action[:,9]=1
        from go2arx5_vla.trajectory.so3 import rotation_matrix_to_rot6d_rows
        action[:,3:9]=rotation_matrix_to_rot6d_rows(quat_to_matrix(initial[3:]))
        doc=native_document(action,np.eye(4),base_valid=False,previous_velocity=[0,0,0]);issued=[];last=-1.;sequence=0
        while float(latest['ee'].time)-t0<10:
            now=float(latest['ee'].time)
            if latest['mode'].data!='WBC':raise RuntimeError('NATIVE_WBC_MODE_LOST')
            if now-last>=.30:
                packet=dict(doc,_pi0294=dict(episode=out.name,sequence=sequence,source_mpc_s=now,ttl_s=.4,operation='install'))
                wire=json.dumps(packet,separators=(',',':'));publish(refs,wire)
                issued.append(dict(source_mpc_s=now,sequence=sequence,wire_sha256=hashlib.sha256(wire.encode()).hexdigest()))
                last=now;sequence+=1
            spin(.002)
        write(out/'ISSUED.json',issued)
        # Pause the simulator first; once its actual clock is stopped freeze the entire owned ROS stack.
        publish(keys,'ENTER')
        if not spin(2,lambda:control().get('running') is False):raise RuntimeError('ACTUAL_PAUSE_ACK_MISSING')
        for p in (solver,controller,planner):os.killpg(p.pid,signal.SIGSTOP)
        paused=control();ee_time=float(latest['ee'].time);spin(.5);after=control()
        write(out/'PAUSE_CHECK.json',dict(before=paused,after=after,ee_time_before=ee_time,ee_time_after=float(latest['ee'].time),
            frozen_owned_groups=[p.pid for p in (solver,controller,planner)],scope='PAUSE_ONLY_NOT_RESUME_OR_FULL_HISTORY_QUALIFICATION'))
        if after['sim_time_s']!=paused['sim_time_s'] or float(latest['ee'].time)!=ee_time:raise ValueError('PAUSE_CLOCK_ADVANCED')
        result=dict(status='RAW_COMPLETE_REQUIRES_INDEPENDENT_CPU_SCORE',hold_seconds=float(latest['ee'].time)-t0,
            pause_clock_stable=True,GR00T_queries=0,plant_overlay_ready=False)
    except Exception as exc:result=dict(status='ENGINEERING_OR_SAFETY_INCOMPLETE',error=type(exc).__name__+': '+str(exc))
    finally:
        for p in processes:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGCONT);os.killpg(p.pid,signal.SIGINT)
        for p in processes:
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for h in handles:h.close()
        raw.close();node.destroy_node();rclpy.shutdown()
        usage=resource.getrusage(resource.RUSAGE_CHILDREN)
        cost=time.process_time()-cpu_start+usage.ru_utime+usage.ru_stime-children.ru_utime-children.ru_stime
        rows=[json.loads(x) for x in (out/'physics.jsonl').read_text().splitlines()] if (out/'physics.jsonl').exists() else []
        samples=[x for x in rows if x['kind']=='sample'];stops=[x for x in rows if x['kind']=='stop']
        lower=max([x['step'] for x in samples]+[x['steps'] for x in stops]+[0])
        charged=min(20000,lower+3) if samples and not stops else lower
        result.update(counts=counts,returncodes=[p.returncode for p in processes],physics_lower_bound=lower,
            physics_charged_conservative=charged,physics_exact='UNOBSERVED' if samples and not stops else lower,
            actual_CPU_s=cost,service_s=time.monotonic()-wall,native_stops=stops)
        write(out/'RESULT.json',result)
        owner=OwnerLock(root/'R4_RESOURCE_OWNER.lock');ledger=read(root/'R4_RESOURCE_LEDGER.json')
        ledger['main_integrations']-=20000-charged;ledger['Core_CPU_s']+=cost
        ledger['CPU_reservations'].pop(ticket['stage']);ledger.setdefault('native_attempts',[]).append(dict(ticket,result=result))
        write(root/'R4_RESOURCE_LEDGER.json',ledger);owner.close();lease.close()
    print(json.dumps(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','native','startup'):p.add_argument('--'+name,required=True)
    a=p.parse_args();main(a.root,a.native,a.startup)
