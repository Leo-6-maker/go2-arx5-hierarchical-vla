"""Bounded cold hook into the existing frozen native process graph.

No ENTER key, positive quota, model query, task rollout or prefix verdict.
Failures keep exact journals. Owned groups are waited independently of the
agent connection. Only the newly admitted caller may invoke this module.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

from .target_zero_counter import read, filehash, put
from .target_native_io_budget import ROLE, REQUESTED, need


class OwnedGroups:
    """Durable exact PID/start identities, no retry and finite group teardown."""
    def __init__(self,folder,guard):
        self.folder,self.guard=Path(folder),guard
        self.children=[];self.logs=[]
        self.closed=False

    def launch(self,name,argv,*,cwd,env):
        self.guard()
        need(not self.closed,'IO_GROUP_OWNER_ALREADY_CLOSED')
        need(name not in [n for n,p in self.children],'IO_CHILD_NAME_ONCE')
        put(self.folder/(name+'_INTENT.json'),dict(argv=argv,once=True,no_retry=True))
        log=(self.folder/(name+'.log')).open('xb',buffering=0);self.logs.append(log)
        p=subprocess.Popen(argv,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        self.children.append((name,p))
        # If this read/publication fails the child is STILL tracked and killed
        # in the surrounding finally. No lost return can start it again.
        start=Path('/proc/%d/stat'%p.pid).read_text().rsplit(')',1)[1].split()[19]
        put(self.folder/(name+'_STARTED.json'),dict(pid=p.pid,proc_start=start,argv=argv))
        return p

    def healthy(self):
        for name,p in self.children:
            need(p.poll() is None,'IO_OWNED_NATIVE_CHILD_EXIT_'+name)
        need(all(log.tell()<16*1024**2 for log in self.logs),'IO_BOUNDED_NATIVE_LOGS')

    def close(self):
        if self.closed:return
        self.closed=True
        errors=[]
        # Send the graceful signal to the entire graph before waiting, then
        # terminate any descendants in the original process group as well.
        for name,p in reversed(self.children):
            try:os.killpg(p.pid,signal.SIGINT)
            except ProcessLookupError:pass
        for name,p in reversed(self.children):
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:errors.append(name)
            # A reaped group leader does not establish that Xvfb descendants
            # exited. Kill remaining members of OUR group, never global pkill.
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            try:put(self.folder/(name+'_EXIT.json'),dict(pid=p.pid,returncode=p.returncode,waited=p.returncode is not None))
            except BaseException as exc:errors.append(name+':'+repr(exc))
        for log in self.logs:log.close()
        need(not errors,'IO_CHILD_TEARDOWN_OR_RECEIPT_FAILURE_'+repr(errors))


def run_graph(root,m,q,inputs,guard):
    root=Path(root);out=root/'evidence/graph';build=root/'evidence/build'
    code=root/m['code']['capsule_relative'];native=Path(inputs['native_root'])
    cpu0=time.process_time();usage0=resource.getrusage(resource.RUSAGE_CHILDREN)
    groups=OwnedGroups(out,guard);quota=recorder=node=lock=None;attempts=[];native_patch={}
    result=dict(status='NATIVE_ZERO_IO_ENGINEERING_INCOMPLETE',completed_stages=[])
    deadline=min(q['deadline_epoch'],time.time()+300)
    def bounded(closeout=False):
        guard(closeout=closeout)
        need(closeout or time.time()<deadline,'IO_FINITE_GRAPH_WALL')
    def ctor(label,fn):
        bounded();need(len(attempts)<REQUESTED['constructors'],'IO_CONSTRUCTOR_UPPER')
        attempts.append(label);put(out/('CONSTRUCTOR_%02d_INTENT.json'%len(attempts)),dict(label=label,once=True))
        return fn()
    try:
        # Build only OUR readonly observer against the exact sealed installed
        # include closure; original binaries/libraries are never recompiled.
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
            PYTHONDONTWRITEBYTECODE='1',LIBGL_ALWAYS_SOFTWARE='1',GALLIUM_DRIVER='llvmpipe',LP_NUM_THREADS='2',
            __EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',
            ROS_LOCALHOST_ONLY='1',ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST',ROS_DOMAIN_ID=str(inputs['ROS_DOMAIN_ID']),
            PYTHONPATH=str(code)+':'+str(code/'src')+':'+os.environ.get('PYTHONPATH',''))
        env.pop('MUJOCO_EGL_DEVICE_ID',None)
        observer=build/'target_zero_observer.so'
        compiler=groups.launch('observer-build',[inputs['compiler'],'-std=c++17','-shared','-fPIC','-O2','-pthread',
            '-I'+inputs['mujoco_include'],str(code/'scripts/pi0294/dec1/target_zero_observer.cpp'),'-ldl','-o',str(observer)],cwd=build,env=env)
        while compiler.poll() is None:
            bounded();time.sleep(.05)
        need(compiler.returncode==0 and observer.is_file(),'IO_OWN_OBSERVER_BUILD_FAILED_NO_RETRY')
        put(build/'BUILD_RETURNED.json',dict(compiler_sha256=filehash(inputs['compiler']),observer_sha256=filehash(observer),
            source_sha256=filehash(code/'scripts/pi0294/dec1/target_zero_observer.cpp'),original_WBC_rebuilt=False))
        # The compiler has already exited; use a separate group owner for the
        # persistent native graph so health checks cannot confuse the two.
        groups.close();groups=OwnedGroups(out,bounded)
        for key in ('MUJOCO_GL','PYOPENGL_PLATFORM','LIBGL_ALWAYS_SOFTWARE','GALLIUM_DRIVER','LP_NUM_THREADS',
                    '__EGL_VENDOR_LIBRARY_FILENAMES','ROS_DOMAIN_ID','ROS_LOCALHOST_ONLY','ROS_AUTOMATIC_DISCOVERY_RANGE'):
            os.environ[key]=env[key]
        os.environ.pop('MUJOCO_EGL_DEVICE_ID',None)
        import numpy as np
        import mujoco
        import rclpy
        from rclpy.qos import qos_profile_sensor_data,QoSProfile,DurabilityPolicy
        from rosidl_runtime_py.convert import message_to_ordereddict
        from std_msgs.msg import String
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import JointState
        from ocs2_msgs.msg import MpcObservation,MpcFlattenedController
        from scripts.pi0292.durable import OwnerLock
        from scripts.pi0294.task_plant import xml_scene,check_frozen_robot
        from scripts.pi0294.task_contacts import allowed
        from scripts.pi0294.task_state import read_state,ActualState
        from scripts.pi0294.task_native_worker import FrameRecorder
        from .target_zero_quota import OwnedTargetZeroQuota
        # Guard Python copy/render paths too. The native companion rejects all
        # six integration APIs BEFORE forwarding in the original sim process.
        for name in ('mj_step','mj_step1','mj_step2','mj_Euler','mj_RungeKutta','mj_implicit'):
            native_patch[name]=getattr(mujoco,name)
            def deny(*args,**kwargs):raise PermissionError('IO_NO_PYTHON_INTEGRATION_AUTHORITY')
            setattr(mujoco,name,deny)
        need(mujoco.__version__=='3.2.7','IO_EXACT_INSTALLED_MUJOCO_VERSION')
        lock=OwnerLock(native.parent/'PI0289_SIM_LEASE')
        geometry=read(root/'plans/ZERO_IO_GEOMETRY.json')
        original=ctor('ORIGINAL_FROZEN_COMPILED_MODEL',lambda:mujoco.MjModel.from_binary_path(inputs['compiled_model']))
        model=ctor('ORIGINAL_FROZEN_TASK_GEOMETRY',lambda:mujoco.MjModel.from_xml_string(xml_scene(geometry['card'],inputs['asset_root'])))
        frozen=check_frozen_robot(original,model);data=mujoco.MjData(model);mujoco.mj_resetDataKeyframe(model,data,0)
        for name in ('x5_joint2','x5_joint3'):
            j=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name);data.qpos[model.jnt_qposadr[j]]=.9
        mujoco.mj_forward(model,data)
        need(data.time==0 and all(allowed(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),
            mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,c.geom2)) for c in data.contact),'IO_ORIGINAL_INITIAL_CONTACT_RULE')
        model_path=out/'scene.private.mjb';mujoco.mj_saveModel(model,str(model_path),None)
        put(out/'FROZEN_GEOMETRY.json',dict(frozen_robot=frozen,model_sha256=filehash(model_path),nq=model.nq,nv=model.nv,nu=model.nu))
        result['completed_stages'].append('FROZEN_GEOMETRY')
        for folder in ('images','tmp','ros-logs'):(out/folder).mkdir()
        env.update(TMPDIR=str(out/'tmp'),ROS_LOG_DIR=str(out/'ros-logs'))
        # These apply to the worker's ROS/render imports as well as children.
        os.environ.update({k:v for k,v in env.items() if k in ('ROS_DOMAIN_ID','ROS_LOCALHOST_ONLY',
            'ROS_AUTOMATIC_DISCOVERY_RANGE','ROS_LOG_DIR','TMPDIR','MUJOCO_GL','PYOPENGL_PLATFORM',
            'LIBGL_ALWAYS_SOFTWARE','GALLIUM_DRIVER','LP_NUM_THREADS','__EGL_VENDOR_LIBRARY_FILENAMES')})
        rclpy.init();node=rclpy.create_node('dec1_native_zero_io_'+root.name.replace('-','_'))
        latest={};counts={};subscriptions=[];ros=(out/'ros.jsonl').open('x')
        def record(key,msg):
            value=message_to_ordereddict(msg);latest[key]=value;counts[key]=counts.get(key,0)+1
            need(ros.tell()<16*1024**2,'IO_BOUNDED_ACTUAL_ROS_TRACE')
            ros.write(json.dumps(dict(kind=key,msg=value),allow_nan=False)+'\n');ros.flush()
        for key,topic,cls,qos in (
            ('odom','/go2_x5/slam/odom_mocap',Odometry,qos_profile_sensor_data),
            ('arm','/go2_x5/arm/state',JointState,qos_profile_sensor_data),
            ('ee','/go2_x5/ee/state',MpcObservation,10),('observation','/mobile_manipulator_mpc_observation',MpcObservation,10),
            ('policy','/mobile_manipulator_mpc_policy',MpcFlattenedController,10),
            ('mode','/go2_x5/arm/mode/state',String,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))):
            subscriptions.append(node.create_subscription(cls,topic,lambda msg,k=key:record(k,msg),qos))
        mode=node.create_publisher(String,'/go2_x5/arm/mode/request',10)
        def spin(seconds,predicate=None):
            end=time.monotonic()+seconds
            while time.monotonic()<end:
                bounded();groups.healthy();rclpy.spin_once(node,timeout_sec=.001)
                if predicate and predicate():return True
            return bool(predicate()) if predicate else True
        spin(.3);need(all(n==node.get_name() for n in node.get_node_names()),'IO_ACTUAL_ROS_DOMAIN_UNOCCUPIED')
        quota=OwnedTargetZeroQuota(out,request_path=root/'REQUEST.json',request_sha256=filehash(root/'REQUEST.json'),
            clock_path=inputs['frozen_clock'],episode_id='NON_OBJECT_ZERO_IO',executor_id='ORIGINAL_FROZEN_WBC',
            run_uuid=root.name,manifest_sha256=filehash(root/'RUN_MANIFEST.json'),guard=bounded,spin=spin)
        quota.arm()
        # The exact previously qualified derivation changes ONLY CppAD cache
        # recompilation true->false. All controller numerical bytes and the
        # original config are separately pinned; do not invoke cold codegen.
        params=['--ros-args','-p','use_sim_time:=true','-p','taskFile:='+inputs['runtime_cache_only_task'],
            '-p','urdfFile:='+inputs['urdf'],'-p','libFolder:='+inputs['codegen_directory']]
        simenv=dict(env,LD_PRELOAD=str(observer)+':'+inputs['original_task_observer'],PI0288_COMPILED_MODEL=str(model_path),
            PI0289_INITIALIZATION_TRACE=str(out/'initialization.jsonl'),PI0289_CONTROL_TRACE=str(out/'control.jsonl'),
            GR00T_ZERO_IO_ROLE=ROLE,GR00T_ZERO_IO_DEADLINE_EPOCH=str(q['deadline_epoch']),
            GR00T_TASK_STEP_QUOTA=str(out/'step_quota'),GR00T_TASK_PAUSE_ACK=str(out/'pause_ack.json'),
            GR00T_TASK_STATE=str(out/'state.bin'),GR00T_TASK_ERROR=str(out/'native-errors.log'),
            GR00T_ZERO_IO_PAUSE_REQUEST=str(out/'pause-request'),GR00T_ZERO_IO_STOP=str(out/'observer-stop'),
            GR00T_ZERO_IO_STOPPED=str(out/'observer-stopped.json'))
        ctor('ORIGINAL_PAUSED_SIM',lambda:groups.launch('sim',[inputs['xvfb_run'],'-a',inputs['sim_binary'],'go2_x5','scene'],cwd=native,env=simenv))
        ctor('ORIGINAL_SOLVER',lambda:groups.launch('solver',[inputs['solver_binary'],*params],cwd=native,env=env))
        ctor('ORIGINAL_CLOCK_CONTROLLER',lambda:groups.launch('controller',[inputs['controller_binary'],*params],cwd=native,env=env))
        groups.launch('receiver',[sys.executable,'-m','scripts.pi0294.native_receiver','--ros-args','-p','use_sim_time:=true','-p','mode:=trajectory'],
            cwd=native,env=dict(env,PI0294_EPISODE='NON_OBJECT_ZERO_IO',PI0294_PLANNER_TRACE=str(out/'planner.jsonl')))
        need(spin(30,lambda:counts.get('odom',0)>10 and counts.get('arm',0)>10),'IO_ACTUAL_NATIVE_FEEDBACK_UNAVAILABLE')
        result['completed_stages'].append('ACTUAL_ODOM_AND_ARM_FEEDBACK')
        # Mode changes while NOT running precede original pause: its paused
        # branch correctly freezes FSM/history and must not be patched around.
        msg=String();msg.data='WBC';mode.publish(msg)
        mode_ready=spin(10,lambda:latest.get('mode',{}).get('data')=='WBC' and 'observation' in latest and 'ee' in latest)
        with (out/'pause-request').open('xb') as f:f.write(b'0\n');f.flush();os.fsync(f.fileno())
        need(spin(5,lambda:(out/'state.bin').exists() and (out/'pause_ack.json').exists()),'IO_ACTUAL_ZERO_STATE_PAUSE_UNAVAILABLE')
        packet=read_state(out/'state.bin');need(packet['step']==0 and packet['time_s']==0,'IO_ACTUAL_NATIVE_ZERO_INTEGRATION')
        result['completed_stages'].append('ACTUAL_ORIGINAL_PAUSE_AND_STATE')
        measured=ctor('FROZEN_ACTUAL_STATE_COPY',lambda:ActualState(model)).update(packet)
        need(measured['safety_valid'] is True and measured['gripper_state_valid'] is True
             and measured['native_abort'] is False,'IO_ACTUAL_FROZEN_FEEDBACK_RULES')
        recorder=FrameRecorder(model_path,out)
        capture=ctor('FROZEN_CAMERA_MODEL_COPY',lambda:recorder.save(0,packet))
        put(out/'ACTUAL_CAPTURE.json',dict(capture=capture,native_state_sha256=filehash(out/'state.bin'),
            qualification_only_not_training_data=True,mode_WBC_before_pause=mode_ready))
        result['completed_stages'].append('ACTUAL_DUAL_CAMERA')
        quota.stop('ZERO_IO_COMPONENT_END_NO_PHYSICAL_QUOTA');result['completed_stages'].append('ACTUAL_DURABLE_STOP')
        result.update(status='ACTUAL_ZERO_IO_FEEDBACK_AND_STOP',actual_mode_WBC_before_pause=mode_ready,
            reference_install_qualified=False,policy_prefix_samples=0,
            next_required='CURRENT_TASK_FULL_NATIVE_PREFIX_AND_ACTUAL_INSTALL_ACK',ROS_counts=counts)
    except BaseException as exc:
        result.update(error=type(exc).__name__+': '+str(exc))
    finally:
        # Stop capturing before destroying the native graph, including partial
        # startup failures. This is not a fabricated native STOP ACK.
        if quota is not None:
            if not quota.stop_attempted and (out/'state.bin').exists() and (out/'pause_ack.json').exists():
                try:quota.stop('ZERO_IO_FAILURE_CLOSEOUT')
                except BaseException as exc:result['stop_error']=repr(exc)
            try:
                with (out/'observer-stop').open('xb') as f:f.write(b'STOP\n');f.flush();os.fsync(f.fileno())
                until=time.monotonic()+2
                while not(out/'observer-stopped.json').exists() and time.monotonic()<until:time.sleep(.01)
                result['observer_stopped']=read(out/'observer-stopped.json') if (out/'observer-stopped.json').exists() else None
            except BaseException as exc:result['observer_stop_error']=repr(exc)
        try:groups.close()
        except BaseException as exc:result['teardown_error']=repr(exc);result['status']='NATIVE_ZERO_IO_ENGINEERING_INCOMPLETE'
        if recorder is not None:
            try:recorder.close()
            except BaseException as exc:result['camera_close_error']=repr(exc);result['status']='NATIVE_ZERO_IO_ENGINEERING_INCOMPLETE'
        if node is not None:
            node.destroy_node()
            if rclpy.ok():rclpy.shutdown()
        if 'ros' in locals():ros.close()
        if quota is not None:result['quota_state']=quota.state();quota.close()
        if lock is not None:lock.close()
        if native_patch:
            for name,fn in native_patch.items():setattr(mujoco,name,fn)
        usage=resource.getrusage(resource.RUSAGE_CHILDREN)
        result.update(scope=ROLE,request_id=q['request_id'],manifest_sha256=filehash(root/'RUN_MANIFEST.json'),
            source_sha=m['code']['vla_commit'],actual_constructor_attempts=len(attempts),constructor_labels=attempts,
            actual_worker_and_children_CPU_s=time.process_time()-cpu0+usage.ru_utime+usage.ru_stime-usage0.ru_utime-usage0.ru_stime,
            target_task_native_qualified=False,native_prefix_safety_qualified=False,physics_ticks=0,policy_episodes=0,
            GR00T_queries=0,training_updates=0,GPU_submissions=0,core64_cells=0)
        put(root/'evidence/NATIVE_IO_RESULT.json',result)
    return result
