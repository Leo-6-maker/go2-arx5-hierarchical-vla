"""Sequential native process adapter; no midpoint restore or alternative controller."""
from pathlib import Path
import json,os,signal,subprocess,time

def native_display_command(spec,output,environment):
    # Do not inject constructor-bearing native libraries into xvfb-run's shell,
    # xauth/Xvfb or utilities. Only the final simulator exec receives preload.
    env=dict(environment)
    for key in('LD_PRELOAD','WM02_INIT_TRACE','WM02_INIT_SEAL_REQUEST'):env.pop(key,None)
    prefix=[spec['xvfb_run'],'-a']
    if spec.get('diagnostic_debugger'):
        prefix += [spec['diagnostic_debugger'],'--batch','-ex','set pagination off','-ex','set confirm off','-ex','handle SIGINT nostop noprint pass','-ex','handle SIGTERM nostop noprint pass','-ex','run','-ex','thread apply all bt','--args']
    argv=[*prefix,'/usr/bin/env','LD_PRELOAD='+':'.join(spec[n]for n in('outer_library','observer_library','trace_library')),'WM02_INIT_TRACE='+str(Path(output)/'init.trace'),'WM02_INIT_SEAL_REQUEST='+str(Path(output)/'trace_seal_request'),spec['native_binary'],'go2_x5','scene']
    return argv,env

def owned_simulator_pid(wrapper_pid,binary):
    """Find only the exec under our display wrapper, never search other owners."""
    pending=[wrapper_pid];seen=set();found=[]
    while pending:
        pid=pending.pop()
        if pid in seen:continue
        seen.add(pid)
        if len(seen)>32:raise RuntimeError('UNEXPECTED_OWNED_DISPLAY_PROCESS_TREE')
        try:
            root=Path('/proc')/str(pid);args=(root/'cmdline').read_bytes().split(b'\0')
            # gdb may put its inferior in a distinct process group. The
            # verified parent tree and UID, rather than a global PID scan,
            # establish ownership in either case.
            if args and args[0].decode()==str(binary)and root.stat().st_uid==os.getuid():found.append(pid)
            pending.extend(int(n)for n in(root/'task'/str(pid)/'children').read_text().split())
        except(ProcessLookupError,FileNotFoundError):continue
    if len(found)!=1:raise RuntimeError('EXACT_OWNED_SIMULATOR_PID_REQUIRED')
    return found[0]

class TargetSequentialAdapter:
    def __init__(self,spec,output):
        self.spec=spec;self.output=Path(output);self.processes=[];self.handles=[];self.node=None;self.counts={};self.latest={};self.started=time.monotonic();self.commands=[];self.exit_records=[];self.reference_qualification=None;self.ros_evidence=[];self.trace_sealed=False
    def reference_phase(self):
        return self.spec['phase'].startswith('g2-')
    def dynamic_phase(self):
        return self.reference_phase()and self.spec['phase']!='g2-reference-qualification'
    def guard(self):
        if time.monotonic()-self.started>=self.spec['wall_limit_s']or time.time()>=self.spec['deadline_epoch']or any((p/'STOP.json').exists()for p in(self.output,Path(self.spec['registered_root']),Path(self.spec['campaign_root']))):raise TimeoutError('AUTO1_FINITE_ENDPOINT_OR_OPERATOR_STOP')
    def control(self):
        p=self.output/'control.jsonl'
        if not p.exists():return{}
        with p.open('rb')as f:
            f.seek(max(0,p.stat().st_size-32768));raw=f.read()
        lines=raw.splitlines()
        if raw and not raw.endswith(b'\n'):lines=lines[:-1]
        try:return json.loads(lines[-1])
        except(IndexError,json.JSONDecodeError):return{}
    def spin(self,seconds,predicate=None):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            self.guard();self.rclpy.spin_once(self.node,timeout_sec=.002)
            if any(p.poll()is not None for _,p in self.processes):raise RuntimeError('OWNED_NATIVE_PROCESS_EXIT')
            if predicate and predicate():return True
        return bool(predicate())if predicate else True
    def launch(self,argv,name,env):
        h=(self.output/(name+'.log')).open('x');self.handles.append(h)
        p=subprocess.Popen(argv,cwd=self.spec['native_root'],env=env,stdout=h,stderr=subprocess.STDOUT,start_new_session=True);self.processes.append((name,p));self.commands.append(dict(name=name,argv=list(map(str,argv)),pid=p.pid))
        from scripts.pi0292.durable import write
        write(self.output/'PROCESS_COMMANDS.private.json',self.commands)
    def publish(self,pub,text):
        msg=self.String();msg.data=text;pub.publish(msg)
    def start(self):
        # Caller has already checked scope, hashes and physical owner lock.
        os.environ.update(ROS_DOMAIN_ID=str(self.spec['ROS_DOMAIN_ID']),ROS_LOCALHOST_ONLY='1',ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST')
        import resource
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        import rclpy
        from std_msgs.msg import String
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import JointState
        from rclpy.qos import qos_profile_sensor_data,QoSProfile,DurabilityPolicy
        self.rclpy=rclpy;self.String=String;rclpy.init();self.node=rclpy.create_node('wm02_auto1_'+self.spec['phase'].replace('-','_'))
        def record(key,msg):
            self.counts[key]=self.counts.get(key,0)+1;self.latest[key]=msg
            if self.dynamic_phase()and key in('policy','arm_request','arm_target','base_command','mode','observation','ee'):
                from rosidl_runtime_py.convert import message_to_ordereddict
                row=dict(topic_key=key,received_monotonic_s=time.monotonic(),message=message_to_ordereddict(msg))
                self.ros_evidence.append(row)
                with(self.output/'ROS_CONSUMPTION.private.jsonl').open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        self.subs=[self.node.create_subscription(cls,topic,lambda msg,k=key:record(k,msg),qos_profile_sensor_data)for key,topic,cls in(('odom','/go2_x5/slam/odom_mocap',Odometry),('arm','/go2_x5/arm/state',JointState))]
        if self.reference_phase():
            from ocs2_msgs.msg import MpcObservation
            for key,topic,cls,qos in(('ee','/go2_x5/ee/state',MpcObservation,10),('observation','/mobile_manipulator_mpc_observation',MpcObservation,10),('mode','/go2_x5/arm/mode/state',String,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))):
                self.subs.append(self.node.create_subscription(cls,topic,lambda msg,k=key:record(k,msg),qos))
            self.refs=self.node.create_publisher(String,'/go2_x5/ee/trajectory_json',10)
            if self.dynamic_phase():
                from ocs2_msgs.msg import MpcFlattenedController
                from trajectory_msgs.msg import JointTrajectory
                from std_msgs.msg import Float32MultiArray
                for key,topic,cls in(('policy','/mobile_manipulator_mpc_policy',MpcFlattenedController),('arm_request','/go2_x5/arm/command/request',JointTrajectory),('arm_target','/go2_x5/arm/command/target',JointTrajectory),('base_command','/go2_x5/base/command',Float32MultiArray)):
                    self.subs.append(self.node.create_subscription(cls,topic,lambda msg,k=key:record(k,msg),10))
        self.keys=self.node.create_publisher(String,'/go2_x5/fsm/key',10);self.modes=self.node.create_publisher(String,'/go2_x5/arm/mode/request',10)
        self.spin(.3)
        if any(name!=self.node.get_name()for name in self.node.get_node_names()):raise RuntimeError('ISOLATED_ROS_DOMAIN_ALREADY_OCCUPIED')
        s=self.spec;out=self.output
        for name in('tmp','ros-logs','cache'):(out/name).mkdir()
        (out/'CONTACT_SAFETY_ARMED').touch();(out/'step_quota').write_text(str(2500 if self.reference_phase()else s['max_ticks'])+'\n')
        warmup=min(5,(s['max_ticks']+499)//500)
        # Existing C++ admits task allocation buckets; the exact bound below
        # still restricts this diagnostic to its declared few task ticks.
        (out/'quota.binding').write_text('\n'.join(('GRTQUOTA2',out.name,s['stage_identity_sha256'],out.name,str(s['max_ticks']),str(warmup),'32'if self.dynamic_phase()else'0'))+'\n')
        env=dict(os.environ,**s['native_environment']);env.update(ROS_DOMAIN_ID=str(s['ROS_DOMAIN_ID']),ROS_LOCALHOST_ONLY='1',ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST',CUDA_VISIBLE_DEVICES='',TMPDIR=str(out/'tmp'),ROS_LOG_DIR=str(out/'ros-logs'),XDG_CACHE_HOME=str(out/'cache'),XDG_CONFIG_HOME=str(out/'cache'),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
        self.native_env=env
        sim=dict(env,LD_PRELOAD=':'.join(s[n]for n in('outer_library','observer_library','trace_library')),PI0288_COMPILED_MODEL=s['model_path'],PI0288_PHYSICS_TRACE=str(out/'physics.jsonl'),PI0288_PHYSICS_CAP=str(s['max_ticks']),PI0288_CONTACT_SAFETY_MARKER=str(out/'CONTACT_SAFETY_ARMED'),PI0289_INITIALIZATION_TRACE=str(out/'initialization.jsonl'),PI0289_CONTROL_TRACE=str(out/'control.jsonl'),WM02_INIT_TRACE=str(out/'init.trace'),GR00T_TASK_GRIPPER_COMMAND=str(out/'gripper.bin'),GR00T_TASK_STATE=str(out/'state.bin'),GR00T_TASK_ERROR=str(out/'task_errors.txt'),GR00T_TASK_STEP_QUOTA=str(out/'step_quota'),GR00T_TASK_PAUSE_ACK=str(out/'pause_ack.json'),GR00T_TASK_QUOTA_BINDING=str(out/'quota.binding'),GR00T_TASK_RUN_UUID=out.name,GR00T_TASK_CARD_SHA256=s['stage_identity_sha256'],GR00T_TASK_RESERVATION_ID=out.name)
        command,display_env=native_display_command(s,out,sim);self.launch(command,'sim',display_env)
        native=Path(s['native_root']);params=['--ros-args','-p','use_sim_time:=true','-p','taskFile:='+str(native/'src/ctrl/go2_x5_ocs2/config/task_floating.info'),'-p','urdfFile:='+str(native/'src/ctrl/go2_x5_description/urdf/arx5_ac1_floating.urdf')]
        # OCS2 can regenerate cache libraries while constructing its interface.
        # Preserve canonical caches; each owned process gets its private copy.
        import shutil
        for name,key in(('solver','solver_binary'),('controller','controller_binary')):
            cache=out/'cache'/('ocs2-'+name);shutil.copytree(s['codegen_lib'],cache)
            self.launch([s[key],*params,'-p','libFolder:='+str(cache)],name,env)
        if self.reference_phase():
            import sys
            self.launch([sys.executable,'-B','-m','scripts.pi0294.native_receiver','--ros-args','-p','use_sim_time:=true','-p','mode:=trajectory'],'planner',dict(env,PI0294_PLANNER_TRACE=str(out/'planner.jsonl'),PI0294_EPISODE=out.name))
        if not self.spin(30,lambda:self.counts.get('odom',0)>10 and self.counts.get('arm',0)>10):raise RuntimeError('PAUSED_NATIVE_FEEDBACK_UNAVAILABLE')
        self.publish(self.modes,'HOLD')
        if not self.spin(5,lambda:self.control().get('arm_mode')=='HOLD'and self.control().get('fsm')=='RLFSMStateRLLocomotion'):raise RuntimeError('ORIGINAL_PAUSED_STANDBY_REQUIRED')
        c=self.control()
        if c.get('running')or c.get('sim_time_s')!=0 or(out/'physics.jsonl').exists():raise RuntimeError('UNEXPECTED_PHYSICS_BEFORE_STANDBY')
        from scripts.pi0292.durable import write
        write(out/'BEFORE_FIRST_INTEGRATION.json',c)
    def observe(self):
        from scripts.pi0294.task_state import read_state
        return read_state(self.output/'state.bin')
    def step_to(self,ticks):
        if ticks!=self.spec['max_ticks']:raise PermissionError('EXACT_REGISTERED_INITIALIZATION_QUOTA_REQUIRED')
        self.advance_to(ticks);self.seal_trace(ticks)
        return self.observe()
    def advance_to(self,ticks):
        if self.trace_sealed:raise PermissionError('TRACE_ALREADY_SEALED_NO_FURTHER_INTEGRATION')
        allowed={2500,2503}if self.spec['phase']=='g2-reference-qualification'else{self.spec['max_ticks']}
        if self.dynamic_phase():allowed={2500,2503,*range(2528,self.spec['max_ticks']+1,25)}
        if ticks not in allowed or ticks>self.spec['max_ticks']:raise PermissionError('EXACT_REGISTERED_NATIVE_PAUSE_REQUIRED')
        if self.dynamic_phase()and ticks>2503:
            from .target_reference import require_live_window
            packet=self.observe();require_live_window(self.active_reference['source_mpc_s'],packet['time_s'],ticks-packet['step'])
        p=self.output/'step_quota';pending=p.with_suffix('.pending');pending.write_text(str(ticks)+'\n');os.replace(pending,p)
        self.publish(self.keys,'ENTER')
        def paused():
            try:return json.loads((self.output/'pause_ack.json').read_text())['step']==ticks and self.observe()['step']==ticks
            except(FileNotFoundError,ValueError,KeyError):return False
        if not self.spin(35,paused):raise RuntimeError('EXACT_NATIVE_PAUSE_NOT_REACHED')
        return self.observe()
    def seal_trace(self,ticks):
        # Preserve exact evidence even when the native GUI shutdown blocks on
        # an input thread. This only closes trace admission, never physics or
        # safety. The native quota remains responsible for the actual pause.
        ack=json.loads((self.output/'pause_ack.json').read_text())
        if ack.get('running')is not False or ack.get('clock_paused')is not True:raise RuntimeError('NATIVE_PAUSE_NOT_CONFIRMED_FOR_TRACE_SEAL')
        (self.output/'trace_seal_request').write_text(json.dumps(dict(step=ticks,sim_time_s=self.observe()['time_s'])))
        def sealed():
            try:return json.loads((self.output/'init.trace.summary').read_text()).get('sealed_at_explicit_pause')is True
            except(FileNotFoundError,ValueError):return False
        if not self.spin(3,sealed):raise RuntimeError('NATIVE_PAUSE_TRACE_SEAL_NOT_ACKNOWLEDGED')
        self.trace_sealed=True
    def qualify_reference(self):
        """Original install/cancel/HOLD transaction at the native paused boundary.

        No extra physics and no claim of MPC policy consumption or task data.
        """
        import hashlib,numpy as np
        from scripts.pi0294.task_state import ActualState
        from scripts.pi0294.task_gripper_contract import teacher_physical_fraction
        from scripts.pi0294.base_intent import eef_document
        from go2arx5_vla.task_data.codec import pose10
        from scripts.pi0292.durable import write
        if self.spec['phase']!='g2-reference-qualification':raise PermissionError('EXACT_REFERENCE_TRANSACTION_SCOPE_REQUIRED')
        packet=self.observe();before=self.counts.get('mode',0)
        if packet['step']!=2500:raise RuntimeError('EXACT_G2_WARMUP_REQUIRED')
        self.publish(self.modes,'WBC')
        if not self.spin(3,lambda:self.counts.get('mode',0)>before and self.latest.get('mode',self.String()).data=='WBC'):raise RuntimeError('NATIVE_WBC_MODE_ACK_MISSING')
        # The frozen original controller rejects duplicate simulation ticks.
        # Process the mode entry on new ticks, without changing that guard or
        # any WBC math. Original quota binding explicitly allows +3 init ticks.
        self.advance_to(2503);packet=self.observe();step=packet['step'];t=packet['time_s']
        def ready():
            return(self.counts.get('mode',0)>before and self.latest.get('mode',self.String()).data=='WBC'and all(k in self.latest and 0<=t-float(self.latest[k].time)<=.025 for k in('ee','observation')))
        if not self.spin(20,ready):
            self.save_feedback_diagnostic('NATIVE_WBC_FEEDBACK_CLOCK_NOT_READY_AT_PAUSE')
            raise RuntimeError('NATIVE_WBC_FEEDBACK_CLOCK_NOT_READY_AT_PAUSE')
        import mujoco
        from .native_replay import forbid_integration
        model=mujoco.MjModel.from_binary_path(self.spec['model_path']);actual=ActualState(model)
        with forbid_integration(mujoco):state=actual.update(packet)
        action=np.stack([pose10(state['tcp'],teacher_physical_fraction(1.))]*40)
        source=float(self.latest['ee'].time);document=eef_document(action);document['_pi0294']=dict(episode=self.output.name,sequence=0,source_mpc_s=source,ttl_s=.4,operation='install')
        wire=json.dumps(document,separators=(',',':'));ref=hashlib.sha256(wire.encode()).hexdigest()
        write(self.output/'REFERENCE_PROPOSED.json',dict(wire=wire,wire_sha256=ref,action_EEF10=action.tolist(),source_mpc_s=source,native_step=step))
        def events():
            p=self.output/'planner.jsonl'
            return[json.loads(l)for l in p.read_text().splitlines()if l.strip()]if p.exists()else[]
        self.publish(self.refs,wire)
        def installed():
            rows=events()
            if any(r['kind']=='reject'and r.get('wire_sha256')==ref for r in rows):raise RuntimeError('ORIGINAL_REFERENCE_RECEIVER_REJECTED')
            return all(any(r['kind']==kind and r.get('ref_id')==ref for r in rows)for kind in('accept','installed'))
        if not self.spin(5,installed):raise RuntimeError('ORIGINAL_REFERENCE_INSTALL_ACK_MISSING')
        before=self.counts.get('mode',0);cancel=dict(_pi0294=dict(episode=self.output.name,sequence=1,source_mpc_s=source,ttl_s=.4,operation='cancel'));self.publish(self.refs,json.dumps(cancel,separators=(',',':')))
        if not self.spin(5,lambda:any(r['kind']=='lifecycle'and r.get('reason')=='CANCEL'for r in events())and self.counts.get('mode',0)>before and self.latest['mode'].data=='HOLD'):raise RuntimeError('ORIGINAL_CANCEL_AND_HOLD_ACK_MISSING')
        after=self.observe();ack=json.loads((self.output/'pause_ack.json').read_text())
        if after['step']!=step or after['time_s']!=t or ack['running']or not ack['clock_paused']:raise RuntimeError('PHYSICS_ADVANCED_DURING_PAUSED_REFERENCE_TRANSACTION')
        self.reference_qualification=dict(status='ORIGINAL_FEEDBACK_INSTALL_CANCEL_HOLD_AT_PAUSE_PASSED',native_step=step,source_mpc_s=source,installed_wire_sha256=ref,extra_integrations=0,task_reference_consumption='NOT_TESTED_NO_TASK_TICKS',events=len(events()))
        write(self.output/'REFERENCE_QUALIFICATION.json',self.reference_qualification)
        return self.reference_qualification
    def reference_events(self):
        path=self.output/'planner.jsonl'
        return [json.loads(l)for l in path.read_text().splitlines()if l.strip()]if path.exists()else[]
    def prepare_dynamic(self):
        if not self.dynamic_phase():raise PermissionError('FINITE_DYNAMIC_PHASE_REQUIRED')
        self.advance_to(2500);before=self.counts.get('mode',0);self.publish(self.modes,'WBC')
        if not self.spin(3,lambda:self.counts.get('mode',0)>before and self.latest['mode'].data=='WBC'):raise RuntimeError('NATIVE_WBC_MODE_ACK_MISSING')
        self.advance_to(2503)
        self.wait_source();self.reference_sequence=-1;self.reference_history=[]
        if self.spec['phase'].startswith('g2-standing-'):
            from go2_x5_planning.ee_trajectory import Pose
            self.base_pose_hold=Pose.from_base_state(self.latest['observation'].state.value[:6]).as_state()
        self.capture_loaded_identity()
    def wait_source(self):
        from scripts.pi0294.task_clock import actual_source_age
        t=self.observe()['time_s']
        if not self.spin(10,lambda:all(k in self.latest and 0<=actual_source_age(t,float(self.latest[k].time))<=.025 for k in('ee','observation'))):
            self.save_feedback_diagnostic('ACTUAL_DYNAMIC_SOURCE_NOT_READY');raise RuntimeError('ACTUAL_DYNAMIC_SOURCE_NOT_READY')
        if self.latest['mode'].data!='WBC':raise RuntimeError('DYNAMIC_WBC_MODE_LOST')
    def measured_task_state(self):
        import mujoco
        from .native_replay import forbid_integration
        from scripts.pi0294.task_state import ActualState
        if not hasattr(self,'actual_state'):self.actual_state=ActualState(mujoco.MjModel.from_binary_path(self.spec['model_path']))
        with forbid_integration(mujoco):return self.actual_state.update(self.observe())
    def install_reference(self,action):
        import hashlib
        from .target_reference import deadline,require_install_clock
        from scripts.pi0294.base_intent import eef_document
        from scripts.pi0294.task_state import write_gripper
        from scripts.pi0294.task_gripper_contract import require_working_target
        self.wait_source();packet=self.observe();source=float(self.latest['ee'].time);require_install_clock(source,packet['time_s'])
        require_working_target(action[:,9]);self.reference_sequence+=1
        doc=eef_document(action)
        certificate=None
        if hasattr(self,'base_pose_hold'):
            from .target_reference import measured_base_hold
            from scripts.pi0294.task_path_model import certify_chunk
            from scripts.pi0292.eval_contract import matrix10
            from .native_replay import forbid_integration
            import mujoco
            with forbid_integration(mujoco):certificate=certify_chunk(self.actual_state.model,self.measured_task_state(),matrix10(action),stage='approach',history=self.reference_history[-2:]if len(self.reference_history)>=2 else())
            doc=measured_base_hold(doc,self.base_pose_hold)
        doc['_pi0294']=dict(episode=self.output.name,sequence=self.reference_sequence,source_mpc_s=source,ttl_s=.4,operation='install')
        wire=json.dumps(doc,separators=(',',':'));ref=hashlib.sha256(wire.encode()).hexdigest()
        intent=dict(phase='proposed',sequence=self.reference_sequence,wire=wire,wire_sha256=ref,source_mpc_s=source,deadline_s=deadline(source),native_step=packet['step'],sim_time_s=packet['time_s'],wall_monotonic_s=time.monotonic(),action_EEF10=action.tolist(),measured_BASE_pose_hold=getattr(self,'base_pose_hold',None),native_reference_certificate=certificate)
        self.command_event(intent)
        write_gripper(self.output/'gripper.bin',self.reference_sequence,source,float(action[0,9]));self.publish(self.refs,wire)
        def installed():
            rows=self.reference_events()
            if any(r['kind']=='reject'and r.get('wire_sha256')==ref for r in rows):raise RuntimeError('ORIGINAL_REFERENCE_RECEIVER_REJECTED')
            return all(any(r['kind']==k and r.get('ref_id')==ref for r in rows)for k in('accept','installed'))
        if not self.spin(5,installed):raise RuntimeError('ORIGINAL_DYNAMIC_INSTALL_ACK_MISSING')
        rows=[r for r in self.reference_events()if r.get('ref_id')==ref];self.active_reference=dict(intent,receiver_events=rows)
        for kind in('accept','installed'):self.command_event(dict(phase='accepted'if kind=='accept'else'installed',wire_sha256=ref,evidence=[r for r in rows if r['kind']==kind]))
        return self.active_reference
    def command_event(self,value):
        with(self.output/'COMMANDS.private.jsonl').open('a')as f:f.write(json.dumps(value,allow_nan=False)+'\n')
    def cancel_reference(self):
        before=self.counts.get('mode',0);self.reference_sequence+=1
        source=float(self.latest['ee'].time);doc=dict(_pi0294=dict(episode=self.output.name,sequence=self.reference_sequence,source_mpc_s=source,ttl_s=.4,operation='cancel'))
        self.publish(self.refs,json.dumps(doc,separators=(',',':')))
        if not self.spin(5,lambda:any(r['kind']=='lifecycle'and r.get('reason')=='CANCEL'for r in self.reference_events())and self.counts.get('mode',0)>before and self.latest['mode'].data=='HOLD'):raise RuntimeError('DYNAMIC_CANCEL_FRESH_HOLD_ACK_MISSING')
        return dict(kind='operator_cancel',native_step=self.observe()['step'],mode='HOLD',physics_stop='NOT_IMPLIED_BY_MODE_ACK')
    def capture_loaded_identity(self):
        """Read actual mappings at a native pause, including regenerated DSOs."""
        from .storage import digest,write_json
        value={}
        for name,p in self.processes:
            pid=owned_simulator_pid(p.pid,self.spec['native_binary'])if name=='sim'else p.pid
            root=Path('/proc')/str(pid);paths=set()
            for line in(root/'maps').read_text().splitlines():
                fields=line.split(None,5)
                if len(fields)==6 and fields[5].startswith('/'):
                    path=fields[5]
                    if '(deleted)'in path:raise RuntimeError('LOADED_ARTIFACT_DELETED_IDENTITY_UNKNOWN')
                    if any(x in path for x in('/ocs2-','/candidate-','/clock-v1/','/install/','libmujoco')):paths.add(path)
            exe=os.readlink(root/'exe');value[name]=dict(pid=pid,executable=exe,executable_sha256=digest(exe),loaded_files_sha256={path:digest(path)for path in sorted(paths)})
        write_json(self.output/'LOADED_EXECUTOR.private.json',value)
        return value
    def dynamic_diagnostic(self):
        import numpy as np
        from .target_reference import trajectory_identity,require_live_window,installed_target_identities
        from go2arx5_vla.task_data.codec import pose10
        from scripts.pi0294.task_gripper_contract import teacher_physical_fraction
        from scripts.pi0292.durable import write
        self.prepare_dynamic();initial=self.measured_task_state();goal=initial['tcp'].copy()
        offset={'g2-eef-x':(0,.001),'g2-eef-y':(1,.001),'g2-standing-eef-x-v2':(0,.001),'g2-standing-eef-y-v2':(1,.001)}.get(self.spec['phase'])
        if offset:goal[offset[0],3]+=offset[1]
        # Original EEF compiler limits still apply. No BASE command labels.
        def action_at_current():
            current=self.measured_task_state()['tcp'];poses=[]
            for k in range(40):
                p=goal.copy()
                if offset:p[:3,3]=current[:3,3]+min((k+1)/8,1)*(goal[:3,3]-current[:3,3])
                poses.append(pose10(p,float(teacher_physical_fraction(1.))))
            return np.stack(poses)
        windows=[];self.install_reference(action_at_current());start=2503;installed_step=start
        while self.observe()['step']<self.spec['max_ticks']:
            before=self.observe();step=before['step']
            # V1 dynamic cadence: seven complete relative atoms (350ms),
            # preserving the original source-relative .4s receiver TTL.
            if step-installed_step>=175:self.install_reference(action_at_current());installed_step=step
            clock=require_live_window(self.active_reference['source_mpc_s'],before['time_s'],25)
            event_start=len(self.ros_evidence);after=self.advance_to(step+25);self.spin(.15)
            if self.latest['mode'].data!='WBC':raise RuntimeError('DYNAMIC_MODE_REFERENCE_MISMATCH')
            if any(r.get('reason')=='PREFIX_TTL_EXPIRED'for r in self.reference_events()):raise RuntimeError('UNEXPECTED_REFERENCE_EXPIRY')
            installed=[r for r in self.active_reference['receiver_events']if r['kind']=='installed'][-1]
            identity=trajectory_identity(installed['times'],installed['poses'],installed['inputs'])
            sent=installed_target_identities(self.reference_events(),self.active_reference['wire_sha256'])
            matched=[];arm=[]
            for row in self.ros_evidence[event_start:]:
                msg=row['message']
                if row['topic_key']=='policy':
                    target=msg['plan_target_trajectories'];key=trajectory_identity(target['time_trajectory'],[s['value']for s in target['state_trajectory']],[s['value']for s in target['input_trajectory']])
                    if key in sent and sent[key]['monotonic_s']<=row['received_monotonic_s']:matched.append(dict(received_monotonic_s=row['received_monotonic_s'],policy_init_time=msg['init_observation']['time'],trajectory_sha256=key,receiver_send_monotonic_s=sent[key]['monotonic_s'],policy_start=msg['time_trajectory'][0],policy_end=msg['time_trajectory'][-1]))
                if row['topic_key']=='arm_request':arm.append(row)
            result=dict(native_start_step=step,native_end_step=after['step'],actual_ticks=after['step']-step,clock=clock,reference_sha256=self.active_reference['wire_sha256'],target_trajectory_sha256=identity,solver_policies_with_exact_installed_target=matched,arm_request_count=len(arm),controller_selected_reference='DERIVED_ONLY_NO_SELECTED_POLICY_ID_EXPORTED',mode=self.latest['mode'].data,actual_end_sim_s=after['time_s'])
            windows.append(result);self.command_event(dict(phase='controller-consumed'if matched else'consumption_unconfirmed',evidence_level='SOLVER_TARGET_CONFIRMED_CONTROLLER_SELECTION_DERIVED'if matched else'UNCONFIRMED',**result));self.command_event(dict(phase='achieved',native_step=after['step'],sim_time_s=after['time_s'],tcp=self.measured_task_state()['tcp'].tolist()))
            from scripts.pi0292.eval_contract import matrix10
            index=(step-installed_step)//25
            self.reference_history.append(dict(end_native_step=after['step'],pose=matrix10(self.active_reference['action_EEF10'])[index]))
        self.capture_loaded_identity();cancel=self.cancel_reference();self.seal_trace(self.spec['max_ticks'])
        confirmed=sum(bool(w['solver_policies_with_exact_installed_target'])for w in windows)
        value=dict(schema='WM02_TARGET_DYNAMIC_DIAGNOSTIC_V1',phase=self.spec['phase'],actual_collection_start_step=start,actual_collection_start_s=initial['time_s'],task_native_ticks=self.observe()['step']-start,complete_task_atoms=len(windows),partial_task_ticks=0,solver_target_confirmed_atoms=confirmed,controller_selected_reference='DERIVED_ONLY_NO_SELECTED_POLICY_ID_EXPORTED',windows=windows,operator_cancel=cancel,observer_safety_stop='NOT_TESTED',full_task_success=False,BASE_labels='ABSENT',replan_cadence_s=.35,source_relative_ttl_s=.4)
        write(self.output/'DYNAMIC_DIAGNOSTIC.json',value);self.reference_qualification=value
        if confirmed!=len(windows):raise RuntimeError('SOLVER_TARGET_CONSUMPTION_NOT_CONFIRMED_FOR_EVERY_ATOM')
        return value
    def save_feedback_diagnostic(self,reason):
        from rosidl_runtime_py.convert import message_to_ordereddict
        from scripts.pi0292.durable import write
        value=dict(reason=reason,counts=self.counts,latest={k:message_to_ordereddict(v)for k,v in self.latest.items()},topics=self.node.get_topic_names_and_types(),nodes=self.node.get_node_names(),control=self.control())
        write(self.output/'REFERENCE_FEEDBACK_DIAGNOSTIC.private.json',value)
        return value
    def inspect_reference_readiness(self):
        # No ENTER, no state reconstruction, no install and no new physics.
        self.publish(self.modes,'WBC');self.spin(3)
        return self.save_feedback_diagnostic('ZERO_INTEGRATION_REFERENCE_READINESS_INSPECTION')
    def close(self):
        # Signal only the Popen-owned process groups; retain all actual exits.
        for name,p in reversed(self.processes):
            escalation=[];lookup_error=None
            if p.poll()is None:
                # xvfb-run owns the display and its cleanup. Signalling that
                # whole group first kills Xvfb under the still-running GUI.
                # Stop the exact owned simulator child; keep its display alive
                # until the wrapper reaps it and performs normal cleanup.
                child=None
                if name=='sim':
                    try:child=owned_simulator_pid(p.pid,self.spec['native_binary'])
                    except RuntimeError as exc:lookup_error=str(exc)
                for sig,wait in((signal.SIGINT,5),(signal.SIGTERM,5),(signal.SIGKILL,2)):
                    try:
                        if child is not None:
                            try:current=owned_simulator_pid(p.pid,self.spec['native_binary'])
                            except RuntimeError:
                                # The child may have exited while its wrapper
                                # is still removing the display resources.
                                p.wait(timeout=wait);break
                            if current!=child:raise RuntimeError('OWNED_SIMULATOR_IDENTITY_CHANGED')
                            os.kill(child,sig)
                        else:os.killpg(p.pid,sig)
                        escalation.append(sig.name);p.wait(timeout=wait);break
                    except ProcessLookupError:break
                    except subprocess.TimeoutExpired:continue
            self.exit_records.append(dict(name=name,pid=p.pid,exit_code=p.poll(),signals=escalation,display_child_lookup_error=lookup_error))
        for h in self.handles:h.close()
        if self.node is not None:self.node.destroy_node();self.rclpy.shutdown()
