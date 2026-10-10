"""One native stack, finite reservations, raw ROS and same-time physics evidence."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

import rclpy
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
from rosidl_runtime_py.convert import message_to_ordereddict
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Float32MultiArray
from trajectory_msgs.msg import JointTrajectory
from ocs2_msgs.msg import MpcObservation, MpcFlattenedController, MpcTargetTrajectories
from go2arx5_vla.target_mujoco.reference import CONDITIONS, panel_chunks, interface_contract


def write(path, value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--root',required=True)
    p.add_argument('--mode',choices=['interface-tests','native-reference','task-reference','teacher-dev','teacher-pilot','collect'],required=True)
    p.add_argument('--condition',choices=CONDITIONS,default=CONDITIONS[0]);args=p.parse_args()
    root=Path(args.root);config=json.loads(Path(args.config).read_text());private=Path(config['private_root'])
    for name in ['runtime','runtime/tmp','runtime/ros-logs','evidence']:(root/name).mkdir(parents=True,exist_ok=True)
    lease=(private.parent/'PI0289_SIM_LEASE').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.mode not in ['interface-tests','native-reference']:
        raise RuntimeError('G1/G2 upstream receipts and verified gripper specification required before task execution')
    g0=root/'evidence/CLOCK_CONTRACT.json'
    if args.mode=='native-reference' and (not g0.exists() or json.loads(g0.read_text())['status']!='PASS'):
        raise RuntimeError('Actual G0 clock receipt required')
    for relative, expected in config['core_hashes'].items():
        if hashlib.sha256((private/relative).read_bytes()).hexdigest()!=expected:raise RuntimeError('Core pin mismatch: '+relative)
    initialization=config.get('initialization')
    audit_binary=root/'runtime/audit_model'
    sim_command=[str(private/'run.sh'),'sim']
    if initialization:
        if initialization['authorization']!='PI0289_OCS2_MUJOCO_AUTONOMOUS_BRINGUP_AND_DATA_GO_V1':raise RuntimeError('Initialization authorization missing')
        for path,expected in initialization['pinned_files'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise RuntimeError('Initialization pin mismatch: '+path)
        audit_binary=Path(initialization['audit_binary'])
        sim_command=[initialization['sim_binary'],'go2_x5','scene']
    model_path=initialization['compiled_model'] if initialization else str(private/'src/ctrl/rl_sar/src/rl_sar_zoo/go2_x5_description/mjcf/scene.xml')
    audit=subprocess.run([str(audit_binary),model_path],capture_output=True,text=True,check=True)
    model_audit=json.loads(audit.stdout)
    if initialization and (model_audit['key']!='home_plus_MPC_initial_arm_V1' or model_audit['invalid_joint_count'] or model_audit['illegal_touching_contacts'] or not model_audit['finite'] or not model_audit['reset_exact']):
        raise RuntimeError('PREEXECUTION_REJECTED: applied home contract failed')
    if not initialization and model_audit['invalid_default_joint_count']:
        raise RuntimeError('PREEXECUTION_REJECTED: native default pose exceeds raw model limits; no pose substitution authorized')
    rclpy.init();node=rclpy.create_node('pi0288_owner')
    # ponytail: fresh whole stack per episode clears native warmstart/history/queues.
    time.sleep(.5)
    graph=node.get_node_names()
    if any(n!='pi0288_owner' for n in graph):raise RuntimeError('ROS domain228 occupied: '+repr(graph))
    episode=uuid.uuid4().hex;out=root/'evidence'/episode;out.mkdir()
    ledger_path=root/'evidence/BUDGET.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'episodes':[]}
    if args.mode=='interface-tests' and any(x.get('startup_variant')==config['startup_variant'] for x in ledger['episodes']):
        raise RuntimeError('This startup variant already has its first attempt; review evidence before replacement')
    if initialization and sum(bool(x.get('technical_replacement')) for x in ledger['episodes'])>=16:
        raise RuntimeError('Technical replacement slots exhausted')
    category='setup' if args.mode=='interface-tests' else 'native_reference'
    if len(ledger['episodes'])>=180 or sum(x['category']==category for x in ledger['episodes'])>=(16 if category=='setup' else 6):raise RuntimeError('Episode category exhausted')
    if any(x['mode']==args.mode and x.get('condition')==args.condition and x.get('status') in ['PASS','QUALITY_FAIL','RUNNING'] for x in ledger['episodes']):raise RuntimeError('Do not repeat valid or active tuple')
    reservation={'physics':60000,'leg_rl':6000,'mpc_attempts':12000,'dispatch':36000,'gpu_s':60,'cpu_core_s':1200}
    caps={'physics':10000000,'leg_rl':2000000,'mpc_attempts':1000000,'dispatch':4000000,'gpu_s':57600,'cpu_core_s':921600}
    for k,v in reservation.items():
        if sum(x['reservation'][k] for x in ledger['episodes'])+v>caps[k]:raise RuntimeError('Budget cap '+k)
    entry=dict(episode=episode,category=category,mode=args.mode,condition=args.condition,reservation=reservation,status='RUNNING',out=str(out),start_wall_s=time.time())
    if initialization:entry.update(initialization_repair=True,compiled_model_runtime=True,technical_replacement=args.mode=='interface-tests',config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest())
    entry['startup_variant']=config['startup_variant']
    ledger['episodes'].append(entry);write(ledger_path,ledger)
    write(out/'START.json',entry)
    write(out/'PREEXECUTION_AUDIT.json',dict(model_audit=model_audit,initialization=initialization,core_hashes=config['core_hashes']))
    raw=(out/'ros.jsonl').open('w');latest={};counts={};subs=[];processes=[];handles=[]
    def record(key,msg):
        counts[key]=counts.get(key,0)+1;latest[key]=(time.time(),time.monotonic(),msg)
        raw.write(json.dumps(dict(kind=key,wall_s=time.time(),monotonic_s=time.monotonic(),msg=message_to_ordereddict(msg)),allow_nan=False)+'\n')
    topics=[('odom','/go2_x5/slam/odom_mocap',Odometry,qos_profile_sensor_data),('arm','/go2_x5/arm/state',JointState,qos_profile_sensor_data),
            ('ee','/go2_x5/ee/state',MpcObservation,10),('observation','/mobile_manipulator_mpc_observation',MpcObservation,10),
            ('policy','/mobile_manipulator_mpc_policy',MpcFlattenedController,10),('target','/mobile_manipulator_mpc_target',MpcTargetTrajectories,10),
            ('arm_request','/go2_x5/arm/command/request',JointTrajectory,10),('arm_target','/go2_x5/arm/command/target',JointTrajectory,10),
            ('base','/go2_x5/base/command',Float32MultiArray,10),('mode','/go2_x5/arm/mode/state',String,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))]
    for key,topic,cls,qos in topics:subs.append(node.create_subscription(cls,topic,lambda msg,k=key:record(k,msg),qos))
    keys=node.create_publisher(String,'/go2_x5/fsm/key',10);refs=node.create_publisher(String,'/go2_x5/ee/trajectory_json',1)
    start=time.monotonic();issued=[];result={}
    def spin(seconds,predicate=None):
        until=time.monotonic()+seconds
        while time.monotonic()<until:
            rclpy.spin_once(node,timeout_sec=.005)
            if time.monotonic()-start>45:raise RuntimeError('Registered service time cap')
            if any(x.poll() is not None for x in processes):raise RuntimeError('Native process exit')
            if predicate and predicate():return True
        return bool(predicate()) if predicate else True
    def key(value):
        msg=String();msg.data=value;keys.publish(msg)
        with (out/'STARTUP_COMMANDS.jsonl').open('a') as f:f.write(json.dumps(dict(key=value,monotonic_s=time.monotonic()))+'\n')
    def control_ready():
        p=out/'control.jsonl'
        if not p.exists():return False
        lines=p.read_text().splitlines()
        if not lines:return False
        try:c=json.loads(lines[-1])
        except json.JSONDecodeError:return False
        expected_fsm='RLFSMStateGetUp' if config['startup_variant'].startswith('V1_') else 'RLFSMStateRLLocomotion'
        return c['fsm']==expected_fsm and c['arm_mode']=='HOLD' and not c['running'] and c['sim_time_s']==0 and c['kp'][12:]==[40,70,70,25,25,25] and all(abs(x-y)<1e-6 for x,y in zip(c['q_target'][12:],[0,.9,.9,0,0,0]))

    try:
        env=os.environ.copy();env.update(PI0288_PHYSICS_TRACE=str(out/'physics.jsonl'),PI0288_PHYSICS_CAP='60000',PI0288_CONTACT_SAFETY_MARKER=str(out/'CONTACT_SAFETY_ARMED'))
        (out/'CONTACT_SAFETY_ARMED').touch()
        if initialization:env.update(PI0289_INITIALIZATION_TRACE=str(out/'initialization.jsonl'),PI0289_CONTROL_TRACE=str(out/'control.jsonl'),PI0288_COMPILED_MODEL=initialization['compiled_model'])
        env['LD_PRELOAD']=str(root/'runtime/observer.so')
        h=(out/'sim.log').open('w');handles.append(h)
        processes.append(subprocess.Popen(['xvfb-run','-a',*sim_command],cwd=private,env=env,stdout=h,stderr=subprocess.STDOUT,start_new_session=True))
        env=os.environ.copy();env['PI0288_PLANNER_TRACE']=str(out/'planner.jsonl')
        env['PYTHONPATH']=str(root/'project-src/scripts/pi0288_mujoco/planner_observer')+':'+env.get('PYTHONPATH','')
        h=(out/'wbc.log').open('w');handles.append(h)
        processes.append(subprocess.Popen([str(private/'run.sh'),'wbc','rviz:=false'],cwd=private,env=env,stdout=h,stderr=subprocess.STDOUT,start_new_session=True))
        if not spin(15,lambda:counts.get('odom',0)>10 and counts.get('arm',0)>10):raise RuntimeError('Feedback unavailable')
        physics_start=time.monotonic()
        if config['startup_variant'].startswith('V1_'):key('0');spin(.04)
        key('2')
        if not spin(.15,control_ready):raise RuntimeError('Paused GetUp/HOLD actual command not ready')
        if (out/'physics.jsonl').exists():raise RuntimeError('Physics ran before native standby')
        write(out/'BEFORE_FIRST_INTEGRATION.json',dict(control=json.loads((out/'control.jsonl').read_text().splitlines()[-1]),physics_integrations=0,source='ACTUAL_NATIVE_COMMAND_NOT_OBSERVER_SUBSTITUTE'))
        key('ENTER')
        if config['startup_variant'].startswith('V1_'):spin(4);key('1')
        spin(5)
        if not spin(max(0,13-(time.monotonic()-physics_start)),lambda:counts.get('ee',0)>10):raise RuntimeError('Warmup controller not ready')
        key('5')
        if not spin(max(0,15-(time.monotonic()-physics_start)),lambda:counts.get('policy',0)>3 and counts.get('arm_request',0)>20 and latest.get('mode',(None,None,None))[2].data=='WBC'):
            raise RuntimeError('Registered warmup limit: WBC not ready')
        initial=list(latest['ee'][2].state.value[:7]);t0=latest['ee'][2].time
        write(out/'TASK_START.json',dict(initial_tcp=initial,mpc_time_s=t0,wall_s=time.time(),monotonic_s=time.monotonic(),warmup_s=time.monotonic()-physics_start))
        if args.mode=='native-reference':(out/'CONTACT_SAFETY_ARMED').touch()
        chunks=panel_chunks(args.condition,initial,episode) if args.mode=='native-reference' else panel_chunks(CONDITIONS[0],initial,episode)
        duration=10
        next_chunk=0
        while latest['ee'][2].time-t0<duration:
            if latest['mode'][2].data!='WBC':raise RuntimeError('Native WBC mode lost')
            t=latest['ee'][2].time-t0
            if next_chunk<len(chunks) and t+1e-6>=chunks[next_chunk]['issue_time_s']:
                c=chunks[next_chunk];msg=String();msg.data=c['wire_json'];refs.publish(msg)
                issued.append(dict(**c,wall_s=time.time(),monotonic_s=time.monotonic(),actual_issue_mpc_s=latest['ee'][2].time))
                next_chunk+=1
            spin(.005)
        write(out/'ISSUED.json',issued)
        result=dict(status='RAW_COMPLETE',episode=episode,mode=args.mode,condition=args.condition,counts=counts,native_mode=latest['mode'][2].data,task_elapsed_s=latest['ee'][2].time-t0,topic_graph=node.get_topic_names_and_types(),pid_groups=[x.pid for x in processes],actual_source='REAL_TARGET_PHYSICS',scope='Must independently score clock/TCP/safety and reference before PASS')
    except Exception as exc:
        result=dict(status='TECHNICAL_UNCLASSIFIED',episode=episode,error=str(exc),counts=counts)
    finally:
        for x in processes:
            if x.poll() is None:os.killpg(x.pid,signal.SIGINT)
        for x in processes:
            try:x.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(x.pid,signal.SIGKILL);x.wait()
        for h in handles:h.close()
        raw.close();node.destroy_node();rclpy.shutdown()
        result.update(service_s=time.monotonic()-start,stop_wall_s=time.time(),returncodes=[x.returncode for x in processes],reset_method='FRESH_WHOLE_STACK_NO_STATE_REUSE')
        write(out/'RESULT.json',result);entry.update(status=result['status'],end_wall_s=time.time());write(ledger_path,ledger)
        print(json.dumps(result,indent=2))
    return 0 if result['status']=='RAW_COMPLETE' else 1


if __name__=='__main__':raise SystemExit(main())
