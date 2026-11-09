"""Sequential native process adapter; no midpoint restore or alternative controller."""
from pathlib import Path
import json,os,signal,subprocess,time
from .native_collection_performance_v1 import configuration, PerformanceTrace, measure, profiled
C02_PHASE='collect-target-fusion-c02-source-state-initialization-v1'
C02_FEEDBACK_PHASE='collect-target-fusion-c02-source-state-feedback-v2'

def native_task_bucket(phase,dynamic):
    """Select only existing quota ABI buckets for the exact finite phase."""
    if phase in('collect-target-fusion-task-phase-live-skill-v8','collect-target-fusion-task-phase-live-skill-v9','collect-target-stable-start-short-v1','collect-target-stable-task-data-v1','collect-target-stable-task-eval-v1','collect-target-stable-task-closed-loop-v1','collect-target-gr00t-prefix-loop-v1','collect-target-teacher-prefix-diagnostic-v1','collect-target-task-learning-grasp-v1'):return '128'
    if phase in(C02_FEEDBACK_PHASE,'collect-target-fusion-task-phase-pose-feedback-v6','collect-target-fusion-task-phase-pose-continuation-v7','collect-target-fusion-task-phase-bias-fade-v8','collect-target-fusion-task-phase-live-skill-v8','collect-target-fusion-task-phase-live-skill-v9','collect-target-stable-start-short-v1','collect-target-stable-task-data-v1','collect-target-stable-task-eval-v1','collect-target-stable-task-closed-loop-v1','collect-target-gr00t-prefix-loop-v1','collect-target-teacher-prefix-diagnostic-v1','collect-target-task-learning-grasp-v1'):return '96'
    return '64'if phase in(C02_FEEDBACK_PHASE,C02_PHASE,'collect-target-fusion-trajectory-slow-qualification-v2','collect-target-fusion-trajectory-response-v1','collect-target-fusion-live-proposal-qualification-v3','collect-target-fusion-source-posture-qualification-v1','collect-target-fusion-source-posture-query-v3','collect-target-fusion-task-phase-initialization-v1','collect-target-fusion-task-phase-bias-calibration-v2','collect-target-fusion-task-phase-bias-continuation-v3','collect-target-fusion-task-phase-xyz-feedback-v4','collect-target-fusion-task-phase-xyz-feedback-continuation-v5')else('32'if dynamic else'0')

def native_startup_readiness(output,reference):
    """Actual owned startup logs, after controller's simulation-time origin is set."""
    required={'controller':('go2_x5_ocs2 ready; waiting for','MRT node is ready.'),'solver':('MPC node is ready.',)}
    if reference:required['planner']=('Trajectory JSON input:',)
    evidence={}
    for name,markers in required.items():
        path=Path(output)/(name+'.log')
        if not path.exists():return None
        with path.open('rb')as f:f.seek(max(0,path.stat().st_size-16384));text=f.read().decode(errors='replace')
        if not all(marker in text for marker in markers):return None
        evidence[name]=list(markers)
    return evidence


def supported_object_hypothesis(phase,stage,sample):
    """Use the existing clone-only attachment hypothesis after actual grasp."""
    if (phase in('collect-target-task-learning-grasp-v1','collect-target-pilot1-supported-close','collect-target-pilot2-supported')and stage in('close','lift','hold','release')
            and sample.get('native_grasp')is True):
        from scripts.pi0294.task_path_model import PlanningObjectMotion
        return PlanningObjectMotion(sample,mode='ACTUAL_GRASP_ATTACHMENT')
    return None

def native_display_command(spec,output,environment):
    # Do not inject constructor-bearing native libraries into xvfb-run's shell,
    # xauth/Xvfb or utilities. Only the final simulator exec receives preload.
    env=dict(environment)
    for key in('LD_PRELOAD','WM02_INIT_TRACE','WM02_INIT_SEAL_REQUEST'):env.pop(key,None)
    prefix=[spec['xvfb_run'],'-a']
    if spec.get('native_signal_trace'):
        from .storage import digest
        path=spec['native_signal_trace']
        if digest(path)!=spec['frozen_files'].get(path):raise PermissionError('PINNED_OWNED_SIGNAL_DIAGNOSTIC_REQUIRED')
        prefix += [path,'-f','-qq','-e','trace=none','-e','signal=SIGINT,SIGTERM','-o',str(Path(output)/'NATIVE_SIGNALS.private.log')]
    if spec.get('diagnostic_debugger'):
        prefix += [spec['diagnostic_debugger'],'--batch','-ex','set pagination off','-ex','set confirm off','-ex','handle SIGINT nostop noprint pass','-ex','handle SIGTERM nostop noprint pass','-ex','run','-ex','thread apply all bt','--args']
    trace_first=(spec.get('extended_trace_build')and spec.get('phase')=='collect-target-next2-contact')or(spec.get('trajectory_trace_capacity_build')and spec.get('phase')in(C02_FEEDBACK_PHASE,C02_PHASE,'collect-target-fusion-trajectory-qualification-v1','collect-target-fusion-trajectory-slow-qualification-v2','collect-target-fusion-trajectory-response-v1','collect-target-fusion-paused-query-v2','collect-target-fusion-live-proposal-qualification-v3','collect-target-fusion-source-posture-qualification-v1','collect-target-fusion-source-posture-query-v3','collect-target-fusion-task-phase-initialization-v1','collect-target-fusion-task-phase-bias-calibration-v2','collect-target-fusion-task-phase-bias-continuation-v3','collect-target-fusion-task-phase-xyz-feedback-v4','collect-target-fusion-task-phase-xyz-feedback-continuation-v5','collect-target-fusion-task-phase-pose-feedback-v6','collect-target-fusion-task-phase-pose-continuation-v7','collect-target-fusion-task-phase-bias-fade-v8','collect-target-fusion-task-phase-live-skill-v8','collect-target-fusion-task-phase-live-skill-v9','collect-target-stable-start-short-v1','collect-target-stable-task-data-v1','collect-target-stable-task-eval-v1','collect-target-stable-task-closed-loop-v1','collect-target-gr00t-prefix-loop-v1','collect-target-teacher-prefix-diagnostic-v1','collect-target-task-learning-grasp-v1'))
    order=('trace_library','outer_library','observer_library')if trace_first else('outer_library','observer_library','trace_library')
    argv=[*prefix,'/usr/bin/env','LD_PRELOAD='+':'.join(spec[n]for n in order),'WM02_INIT_TRACE='+str(Path(output)/'init.trace'),'WM02_INIT_SEAL_REQUEST='+str(Path(output)/'trace_seal_request'),spec['native_binary'],'go2_x5','scene']
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
        self.performance=PerformanceTrace() if configuration(spec) is not None else None
    def reference_phase(self):
        return self.spec['phase'].startswith(('g2-','collect-target-'))
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
    @profiled('ros.spin')
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
                with measure(self,'ros.callback.'+key):
                    with measure(self,'ros.message_conversion.'+key):
                        row=dict(topic_key=key,received_monotonic_s=time.monotonic(),message=message_to_ordereddict(msg))
                    self.ros_evidence.append(row)
                    with measure(self,'ros.journal_append.'+key):
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
        task_bucket=native_task_bucket(s.get('phase'),self.dynamic_phase())
        (out/'quota.binding').write_text('\n'.join(('GRTQUOTA2',out.name,s['stage_identity_sha256'],out.name,str(s['max_ticks']),str(warmup),task_bucket))+'\n')
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
            # Frozen native_environment may name an older capsule. Load the
            # receiver and lifecycle helper from this verified source seal.
            receiver=Path(s['source_root'])/'scripts/pi0294/native_receiver.py'
            planner_env=dict(env,PYTHONPATH=s['source_root']+':'+s['source_root']+'/src:'+env.get('PYTHONPATH',''),PI0294_PLANNER_TRACE=str(out/'planner.jsonl'),PI0294_EPISODE=out.name)
            self.launch([sys.executable,'-B',str(receiver),'--ros-args','-p','use_sim_time:=true','-p','mode:=trajectory'],'planner',planner_env)
        # OCS2 construction/codegen sets start_time_=now(). Starting physics
        # before it completes gives that process a nonzero clock origin and
        # permanently stale reference times at the next exact native pause.
        if not self.spin(30,lambda:native_startup_readiness(out,self.reference_phase())and self.counts.get('odom',0)>10 and self.counts.get('arm',0)>10):raise RuntimeError('PAUSED_NATIVE_EXECUTOR_OR_FEEDBACK_UNAVAILABLE')
        self.publish(self.modes,'HOLD')
        if not self.spin(5,lambda:self.control().get('arm_mode')=='HOLD'and self.control().get('fsm')=='RLFSMStateRLLocomotion'):raise RuntimeError('ORIGINAL_PAUSED_STANDBY_REQUIRED')
        c=self.control()
        if c.get('running')or c.get('sim_time_s')!=0 or(out/'physics.jsonl').exists():raise RuntimeError('UNEXPECTED_PHYSICS_BEFORE_STANDBY')
        from scripts.pi0292.durable import write
        write(out/'PAUSED_EXECUTOR_READINESS.json',dict(status='OWNED_EXECUTORS_CONSTRUCTED_BEFORE_FIRST_INTEGRATION',markers=native_startup_readiness(out,self.reference_phase()),native_control_time_s=c['sim_time_s'],physics_journal_exists=(out/'physics.jsonl').exists(),clock_origin_contract='CONTROLLER_CONSTRUCTOR_START_TIME_AT_NATIVE_ZERO'))
        write(out/'BEFORE_FIRST_INTEGRATION.json',c)
    @profiled('native.state_read')
    def observe(self):
        from scripts.pi0294.task_state import read_state
        until=time.monotonic()+.25
        while True:
            try:return read_state(self.output/'state.bin')
            except FileNotFoundError:
                # Same observed live-snapshot publication gap as pause_ack.
                # Never reuse an old packet or change the native step quota.
                self.guard()
                if time.monotonic()>=until:raise
                time.sleep(.002)
    def step_to(self,ticks):
        if ticks!=self.spec['max_ticks']:raise PermissionError('EXACT_REGISTERED_INITIALIZATION_QUOTA_REQUIRED')
        self.advance_initialization(ticks);self.seal_trace(ticks)
        return self.observe()
    def advance_initialization(self,ticks):
        if self.spec.get('startup_pause_stride_ticks')is None:return self.advance_to(ticks)
        if self.spec['startup_pause_stride_ticks']!=25 or not 0<ticks<=2500 or ticks%25:raise PermissionError('EXACT_FINITE_INITIALIZATION_CLOCK_WINDOWS_REQUIRED')
        # Use the same original quota/pause/ENTER seam as task atoms. No state
        # reset, synthetic transition, altered gain, dt or safety threshold.
        for target in range(25,ticks+1,25):
            state=self.advance_to(target)
            with(self.output/'INITIALIZATION_CLOCK_WINDOWS.jsonl').open('a')as f:f.write(json.dumps(dict(end_step=state['step'],sim_time_s=state['time_s'],actual_ticks=25,wall_monotonic_s=time.monotonic()))+'\n')
        return self.observe()
    @profiled('native.advance_to')
    def advance_to(self,ticks):
        if self.trace_sealed:raise PermissionError('TRACE_ALREADY_SEALED_NO_FURTHER_INTEGRATION')
        allowed={2500,2503}if self.spec['phase']=='g2-reference-qualification'else{self.spec['max_ticks']}
        if self.dynamic_phase():allowed={2500,2503,*range(2528,self.spec['max_ticks']+1,25)}
        ttl=self.spec['phase']=='g2-standing-ttl-v2'
        if ttl:allowed={2500,2503,*getattr(self,'ttl_boundaries',[])}
        # TTL restricts task-phase pauses, not the registered cold-init windows.
        if self.spec.get('startup_pause_stride_ticks')==25:allowed.update(range(25,min(2500,self.spec['max_ticks'])+1,25))
        if ticks not in allowed or ticks>self.spec['max_ticks']:raise PermissionError('EXACT_REGISTERED_NATIVE_PAUSE_REQUIRED')
        if self.dynamic_phase()and ticks>2503 and not ttl:
            from .target_reference import require_live_window
            packet=self.observe();require_live_window(self.active_reference['source_mpc_s'],packet['time_s'],ticks-packet['step'])
        p=self.output/'step_quota';pending=p.with_suffix('.pending');pending.write_text(str(ticks)+'\n');os.replace(pending,p)
        self.resume_clock_once(ticks)
        def paused():
            try:return json.loads((self.output/'pause_ack.json').read_text())['step']==ticks and self.observe()['step']==ticks
            except(FileNotFoundError,ValueError,KeyError):return False
        if not self.spin(35,paused):
            from .storage import write_json
            ack=self.output/'pause_ack.json'
            try:pause=json.loads(ack.read_text())
            except(FileNotFoundError,json.JSONDecodeError):pause=None
            write_json(self.output/'CLOCK_RESUME_FAILURE.json',dict(requested_step=ticks,
                matched_subscriptions=self.keys.get_subscription_count(),control=self.control(),
                pause_ack=pause,automatic_ENTER_retries=0,
                cause='UNRESOLVED_COMMAND_DELIVERY_OR_NATIVE_CONSUMPTION'))
            raise RuntimeError('EXACT_NATIVE_PAUSE_NOT_REACHED')
        return self.observe()
    @profiled('native.clock_command_handoff')
    def resume_clock_once(self,ticks):
        # ENTER toggles the original native clock. Never retry that write when
        # consumption is unknown. Topic matching is a separate startup barrier;
        # odometry reception alone did not establish that this publisher matched.
        def matched():
            count=self.keys.get_subscription_count()
            if count>1:raise RuntimeError('ISOLATED_CLOCK_KEY_HAS_MULTIPLE_SUBSCRIBERS')
            return count==1
        if not self.spin(3,matched):raise RuntimeError('OWNED_CLOCK_KEY_SUBSCRIBER_NOT_MATCHED')
        with(self.output/'CLOCK_RESUME_REQUESTS.jsonl').open('a')as stream:
            stream.write(json.dumps(dict(requested_step=ticks,matched_subscriptions=1,
                monotonic_s=time.monotonic(),publication_count=1,
                consumption_evidence='EXACT_NATIVE_PAUSE_REQUIRED_SEPARATELY'))+'\n')
        self.publish(self.keys,'ENTER')
    def seal_trace(self,ticks):
        # Preserve exact evidence even when the native GUI shutdown blocks on
        # an input thread. This only closes trace admission, never physics or
        # safety. The native quota remains responsible for the actual pause.
        snapshot={}
        def confirmed_pause():
            # The measured local block filesystem can briefly hide a replaced
            # live snapshot. Retry reads only; never grant another physics tick.
            try:ack=json.loads((self.output/'pause_ack.json').read_text());packet=self.observe()
            except(FileNotFoundError,json.JSONDecodeError):return False
            if (ack.get('step')!=ticks or packet['step']!=ticks
                    or ack.get('running')is not False or ack.get('clock_paused')is not True
                    or abs(ack.get('time_s',float('inf'))-packet['time_s'])>1e-8):
                raise RuntimeError('NATIVE_PAUSE_NOT_CONFIRMED_FOR_TRACE_SEAL')
            snapshot.update(packet);return True
        if not self.spin(3,confirmed_pause):raise RuntimeError('NATIVE_PAUSE_SNAPSHOT_NOT_READABLE_FOR_TRACE_SEAL')
        request=self.output/'trace_seal_request';pending=request.with_suffix('.pending')
        pending.write_text(json.dumps(dict(step=ticks,sim_time_s=snapshot['time_s'])));os.replace(pending,request)
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
        from scripts.pi0294.task_clock import actual_source_age
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
            return(self.counts.get('mode',0)>before and self.latest.get('mode',self.String()).data=='WBC'and all(k in self.latest and 0<=actual_source_age(t,float(self.latest[k].time))<=.025 for k in('ee','observation')))
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
        events=self.reference_events
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
    @profiled('reference.events_read')
    def reference_events(self):
        if self.spec.get('phase') in (C02_FEEDBACK_PHASE,C02_PHASE,'collect-target-fusion-task-phase-pose-continuation-v7','collect-target-fusion-task-phase-bias-fade-v8','collect-target-fusion-task-phase-live-skill-v8','collect-target-fusion-task-phase-live-skill-v9','collect-target-stable-start-short-v1','collect-target-stable-task-data-v1','collect-target-stable-task-eval-v1','collect-target-stable-task-closed-loop-v1','collect-target-gr00t-prefix-loop-v1','collect-target-teacher-prefix-diagnostic-v1','collect-target-task-learning-grasp-v1'):
            from .planner_event_log import PlannerEventLog
            if not hasattr(self,'_planner_event_log'):
                self._planner_event_log=PlannerEventLog(self.output/'planner.jsonl')
            return self._planner_event_log.read()
        path=self.output/'planner.jsonl'
        if not path.exists():return[]
        raw=path.read_bytes()
        # The receiver appends large records. A concurrent read may end inside
        # one write; consume only newline-committed records, retaining strict
        # parsing for any malformed complete record.
        if raw and not raw.endswith(b'\n'):raw=raw[:raw.rfind(b'\n')+1]
        return[json.loads(line)for line in raw.splitlines()if line.strip()]
    def prepare_dynamic(self):
        if not self.dynamic_phase():raise PermissionError('FINITE_DYNAMIC_PHASE_REQUIRED')
        self.advance_initialization(2500);before=self.counts.get('mode',0);self.publish(self.modes,'WBC')
        if not self.spin(3,lambda:self.counts.get('mode',0)>before and self.latest['mode'].data=='WBC'):raise RuntimeError('NATIVE_WBC_MODE_ACK_MISSING')
        self.advance_to(2503)
        self.wait_source();self.reference_sequence=-1;self.reference_history=[]
        if self.spec['phase'].startswith(('g2-standing-','collect-target-')):
            from go2_x5_planning.ee_trajectory import Pose
            self.base_pose_hold=Pose.from_base_state(self.latest['observation'].state.value[:6]).as_state()
        self.capture_loaded_identity()
    @profiled('ros.wait_source')
    def wait_source(self):
        from scripts.pi0294.task_clock import actual_source_age
        t=self.observe()['time_s']
        if not self.spin(10,lambda:all(k in self.latest and 0<=actual_source_age(t,float(self.latest[k].time))<=.025 for k in('ee','observation'))):
            self.save_feedback_diagnostic('ACTUAL_DYNAMIC_SOURCE_NOT_READY');raise RuntimeError('ACTUAL_DYNAMIC_SOURCE_NOT_READY')
        if self.latest['mode'].data!='WBC':raise RuntimeError('DYNAMIC_WBC_MODE_LOST')
    @profiled('native.measured_task_state_FK')
    def measured_task_state(self):
        import mujoco
        from .native_replay import forbid_integration
        from scripts.pi0294.task_state import ActualState
        if not hasattr(self,'actual_state'):self.actual_state=ActualState(mujoco.MjModel.from_binary_path(self.spec['model_path']))
        with forbid_integration(mujoco):return self.actual_state.update(self.observe())
    @profiled('reference.install_and_ACK')
    def install_reference(self,action):
        import hashlib
        if self.spec.get('phase')=='collect-target-task-learning-grasp-v1':
            from .task_learning_grasp_native_v1 import validate_adapter_install
            self.command_event(dict(phase='task-learning-grasp-full-H40-envelope',**validate_adapter_install(self,action)))
        if self.spec.get('phase')=='collect-target-teacher-prefix-diagnostic-v1':
            from .target_teacher_prefix_diagnostic_v1 import validate_adapter_install
            self.command_event(dict(phase='target-teacher-prefix-full-H40-envelope',
                **validate_adapter_install(self,action)))
        if self.spec.get('phase')=='collect-target-gr00t-prefix-loop-v1':
            from .target_gr00t_native_loop_v1 import validate_adapter_install
            self.command_event(dict(phase='target-gr00t-prefix-full-H40-envelope',
                **validate_adapter_install(self,action)))
        if self.spec.get('phase')=='collect-target-stable-task-closed-loop-v1':
            from .stable_task_closed_loop_v1 import validate_adapter_install
            self.command_event(dict(phase='stable-task-closed-loop-full-H40-envelope',
                **validate_adapter_install(self,action)))
        if self.spec.get('phase')=='collect-target-stable-task-eval-v1':
            from .stable_task_evaluation_v1 import validate_adapter_install
            self.command_event(dict(phase='stable-task-candidate-full-H40-envelope',
                **validate_adapter_install(self,action)))
        if self.spec.get('phase')=='collect-target-stable-task-data-v1':
            from .stable_task_data_v1 import validate_adapter_install
            self.command_event(dict(phase='stable-task-data-full-H40-envelope',
                **validate_adapter_install(self,action)))
        if self.spec.get('phase')=='collect-target-stable-start-short-v1':
            from .stable_start_short_task_v1 import validate_adapter_install
            self.command_event(dict(phase='stable-start-short-full-H40-envelope',
                **validate_adapter_install(self,action)))
        if self.spec.get('phase')==C02_FEEDBACK_PHASE:
            from .fusion_c02_source_state_feedback_v2 import validate_install
            self.command_event(dict(phase='c02-feedback-full-H40-envelope',**validate_install(self,action)))
        if self.spec.get('phase')==C02_PHASE:
            from .fusion_c02_source_state_initialization_v1 import validate_install
            self.command_event(dict(phase='c02-source-state-full-H40-envelope',**validate_install(self,action)))
        if self.spec.get('phase')=='collect-target-fusion-task-phase-live-skill-v9':
            from .fusion_task_phase_live_skill_native_v9 import validate_adapter_install
            self.command_event(dict(phase='graded-live-skill-v9-full-H40-envelope',
                **validate_adapter_install(self,action)))
        if self.spec.get('phase')=='collect-target-fusion-task-phase-live-skill-v8':
            from .fusion_task_phase_live_skill_native_v8 import validate_adapter_install
            self.command_event(dict(phase='graded-live-skill-full-H40-envelope',
                **validate_adapter_install(self,action)))
        if self.spec.get('phase')=='collect-target-fusion-task-phase-bias-fade-v8':
            from .fusion_task_phase_bias_fade_native_v8 import validate_adapter_install
            self.command_event(dict(phase='task-phase-bias-fade-full-H40-envelope',
                **validate_adapter_install(self,action)))
        if self.spec.get('phase')=='collect-target-fusion-task-phase-pose-continuation-v7':
            from .fusion_task_phase_pose_continuation_native_v7 import validate_adapter_install
            self.command_event(dict(phase='task-phase-finite-pose-continuation-full-H40-envelope',
                **validate_adapter_install(self,action)))
        if self.spec.get('phase')=='collect-target-fusion-task-phase-pose-feedback-v6':
            from .fusion_task_phase_pose_feedback_native_v6 import validate_adapter_install
            self.command_event(dict(phase='task-phase-finite-xyz-so3-feedback-full-H40-envelope',
                **validate_adapter_install(self,action)))
        if self.spec.get('phase')=='collect-target-fusion-task-phase-xyz-feedback-continuation-v5':
            from .fusion_task_phase_xyz_feedback_continuation_native_v5 import validate_adapter_install
            self.command_event(dict(phase='task-phase-graded-finite-xyz-feedback-full-H40-envelope',
                                    **validate_adapter_install(self,action)))
        if self.spec.get('phase')=='collect-target-fusion-task-phase-xyz-feedback-v4':
            from .fusion_task_phase_xyz_feedback_native_v4 import validate_adapter_install
            self.command_event(dict(phase='task-phase-finite-xyz-feedback-full-H40-envelope',
                                    **validate_adapter_install(self,action)))
        if self.spec.get('phase')=='collect-target-fusion-task-phase-bias-continuation-v3':
            from .fusion_task_phase_bias_continuation_native_v3 import validate_adapter_install
            self.command_event(dict(phase='task-phase-graded-frozen-bias-full-H40-envelope',
                                    **validate_adapter_install(self,action)))
        if self.spec.get('phase')=='collect-target-fusion-task-phase-bias-calibration-v2':
            from .fusion_task_phase_bias_calibration_native_v2 import validate_adapter_install
            self.command_event(dict(phase='task-phase-frozen-bias-calibration-full-H40-envelope',
                                    **validate_adapter_install(self,action)))
        if self.spec.get('phase')=='collect-target-fusion-task-phase-initialization-v1':
            from .fusion_task_phase_initialization_native import validate_adapter_install
            self.command_event(dict(phase='task-phase-calibration-full-H40-envelope',
                                    **validate_adapter_install(self,action)))
        if self.spec.get('phase')=='collect-target-fusion-paused-query-v2':
            from .fusion_paused_query_native import validate_adapter_install
            self.command_event(dict(phase='paused-query-hold-only-envelope',
                                    **validate_adapter_install(self,action)))
        if self.spec.get('phase')=='collect-target-fusion-trajectory-response-v1':
            from .fusion_trajectory_response_native import validate_adapter_install
            self.command_event(dict(phase='trajectory-response-full-H40-envelope',
                                    **validate_adapter_install(self,action)))
        if self.spec.get('phase')in('collect-target-fusion-trajectory-qualification-v1','collect-target-fusion-trajectory-slow-qualification-v2','collect-target-fusion-live-proposal-qualification-v3','collect-target-fusion-source-posture-qualification-v1','collect-target-fusion-source-posture-query-v3'):
            from .fusion_trajectory_qualification_native import validate_adapter_install
            self.command_event(dict(phase='trajectory-qualification-full-H40-envelope',
                                    **validate_adapter_install(self,action)))
        if self.spec.get('phase')=='collect-target-fusion-reference-qualification-v1':
            from .fusion_reference_qualification_native import validate_adapter_install
            self.command_event(dict(phase='new-reference-qualification-full-H40-envelope',
                                    **validate_adapter_install(self,action)))
        if hasattr(self,'fusion_initial_tcp'):
            from .fusion_contract import validate_local_preview
            envelope=validate_local_preview(action,initial_tcp=self.fusion_initial_tcp,
                chosen_action=self.fusion_selected_action,renewed=self.fusion_install_action_count>0,
                base_pose=self.base_pose_hold,frozen_base_pose=self.fusion_base_pose_hold)
            self.command_event(dict(phase='fusion-full-final-reference-envelope',**envelope))
            self.fusion_install_action_count+=1
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
            stage='approach'
            if self.spec['phase']=='collect-target-next2-contact':
                stage={'CLOSE':'close','SUPPORTED_HOLD':'hold','RELEASE':'release','WITHDRAW':'detach'}.get(getattr(self,'next2_contact_stage',None),'approach')
                if getattr(self,'next2_contact_stage',None)=='WITHDRAW'and not getattr(self,'next2_contact_manipulation_started',False):stage='approach'
            if self.spec['phase']in('collect-target-pilot1-supported-close','collect-target-pilot2-supported'):
                stage={'CLOSE':'close','SUPPORTED_HOLD':'hold','RELEASE':'release','EARLY_RELEASE':'release','OPEN_HOLD':'release'}.get(getattr(self,'pilot1_supported_stage',None),'approach')
            if self.spec['phase']=='collect-target-task-learning-grasp-v1':
                stage={'CLOSE':'close','LIFT':'lift','HOLD':'hold'}.get(getattr(self,'task_learning_stage',None),'approach')
            try:
                with forbid_integration(mujoco):
                    sample=self.measured_task_state();motion=supported_object_hypothesis(self.spec['phase'],stage,sample)
                    certificate=certify_chunk(self.actual_state.model,sample,matrix10(action),stage=stage,history=self.reference_history[-2:]if len(self.reference_history)>=2 else(),planning_object_motion=motion)
                    if motion is not None:certificate['planning_object_motion']=dict(motion.evidence(),source_native_step=sample['native_step'])
            except ValueError as error:
                self.command_event(dict(phase='reference-certificate-rejected',native_step=packet['step'],source_mpc_s=source,stage=stage,action_EEF10=action.tolist(),reason=str(error),contact_details=getattr(error,'native_contact_details',None),submitted_or_executed=False))
                raise
            doc=measured_base_hold(doc,self.base_pose_hold)
        doc['_pi0294']=dict(episode=self.output.name,sequence=self.reference_sequence,source_mpc_s=source,ttl_s=.4,operation='install')
        wire=json.dumps(doc,separators=(',',':'));ref=hashlib.sha256(wire.encode()).hexdigest()
        intent=dict(phase='proposed',sequence=self.reference_sequence,wire=wire,wire_sha256=ref,source_mpc_s=source,deadline_s=deadline(source),native_step=packet['step'],sim_time_s=packet['time_s'],wall_monotonic_s=time.monotonic(),action_EEF10=action.tolist(),measured_BASE_pose_hold=getattr(self,'base_pose_hold',None),native_reference_certificate=certificate)
        self.command_event(intent)
        if self.spec.get('phase')not in(C02_FEEDBACK_PHASE,C02_PHASE,'collect-target-fusion-trajectory-qualification-v1','collect-target-fusion-trajectory-slow-qualification-v2','collect-target-fusion-trajectory-response-v1','collect-target-fusion-paused-query-v2','collect-target-fusion-live-proposal-qualification-v3','collect-target-fusion-source-posture-qualification-v1','collect-target-fusion-source-posture-query-v3','collect-target-fusion-task-phase-initialization-v1','collect-target-fusion-task-phase-bias-calibration-v2','collect-target-fusion-task-phase-bias-continuation-v3','collect-target-fusion-task-phase-xyz-feedback-v4','collect-target-fusion-task-phase-xyz-feedback-continuation-v5','collect-target-fusion-task-phase-pose-feedback-v6','collect-target-fusion-task-phase-pose-continuation-v7','collect-target-fusion-task-phase-bias-fade-v8','collect-target-fusion-task-phase-live-skill-v8','collect-target-fusion-task-phase-live-skill-v9','collect-target-stable-start-short-v1','collect-target-stable-task-data-v1','collect-target-stable-task-eval-v1','collect-target-stable-task-closed-loop-v1','collect-target-gr00t-prefix-loop-v1','collect-target-teacher-prefix-diagnostic-v1','collect-target-task-learning-grasp-v1'):
            write_gripper(self.output/'gripper.bin',self.reference_sequence,source,float(action[0,9]))
        self.publish(self.refs,wire)
        def installed():
            rows=self.reference_events()
            if any(r['kind']=='reject'and r.get('wire_sha256')==ref for r in rows):raise RuntimeError('ORIGINAL_REFERENCE_RECEIVER_REJECTED')
            return all(any(r['kind']==k and r.get('ref_id')==ref for r in rows)for k in('accept','installed'))
        if not self.spin(5,installed):raise RuntimeError('ORIGINAL_DYNAMIC_INSTALL_ACK_MISSING')
        rows=[r for r in self.reference_events()if r.get('ref_id')==ref];self.active_reference=dict(intent,receiver_events=rows)
        for kind in('accept','installed'):self.command_event(dict(phase='accepted'if kind=='accept'else'installed',wire_sha256=ref,evidence=[r for r in rows if r['kind']==kind]))
        return self.active_reference
    @profiled('native.command_journal')
    def command_event(self,value):
        with(self.output/'COMMANDS.private.jsonl').open('a')as f:f.write(json.dumps(value,allow_nan=False)+'\n')
    def cancel_reference(self):
        before=self.counts.get('mode',0);self.reference_sequence+=1
        source=float(self.latest['ee'].time);doc=dict(_pi0294=dict(episode=self.output.name,sequence=self.reference_sequence,source_mpc_s=source,ttl_s=.4,operation='cancel'))
        self.publish(self.refs,json.dumps(doc,separators=(',',':')))
        if not self.spin(5,lambda:any(r['kind']=='lifecycle'and r.get('reason')=='CANCEL'for r in self.reference_events())and self.counts.get('mode',0)>before and self.latest['mode'].data=='HOLD'):raise RuntimeError('DYNAMIC_CANCEL_FRESH_HOLD_ACK_MISSING')
        return dict(kind='operator_cancel',native_step=self.observe()['step'],mode='HOLD',physics_stop='NOT_IMPLIED_BY_MODE_ACK')
    @profiled('native.loaded_identity')
    def capture_loaded_identity(self):
        """Read actual mappings at a native pause, including regenerated DSOs."""
        from .storage import digest,write_json
        from .target_reference import relevant_loaded_path
        value={}
        for name,p in self.processes:
            pid=owned_simulator_pid(p.pid,self.spec['native_binary'])if name=='sim'else p.pid
            root=Path('/proc')/str(pid);paths=set()
            for line in(root/'maps').read_text().splitlines():
                fields=line.split(None,5)
                if len(fields)==6 and fields[5].startswith('/'):
                    path=fields[5]
                    if relevant_loaded_path(path):paths.add(path)
            exe=os.readlink(root/'exe');value[name]=dict(pid=pid,executable=exe,executable_sha256=digest(exe),loaded_files_sha256={path:digest(path)for path in sorted(paths)})
        index=getattr(self,'loaded_identity_snapshots',0)
        write_json(self.output/f'LOADED_EXECUTOR_{index:02d}.private.json',value)
        self.loaded_identity_snapshots=index+1
        return value
    def dynamic_diagnostic(self):
        if self.spec.get('phase')=='collect-target-task-learning-grasp-v1':
            from .task_learning_grasp_native_v1 import run
            return run(self)
        if self.spec.get('phase')=='collect-target-teacher-prefix-diagnostic-v1':
            from .target_teacher_prefix_diagnostic_v1 import run
            return run(self)
        if self.spec.get('phase')=='collect-target-gr00t-prefix-loop-v1':
            from .target_gr00t_native_loop_v1 import run
            return run(self)
        if self.spec.get('phase')=='collect-target-stable-task-closed-loop-v1':
            from .stable_task_closed_loop_v1 import run
            return run(self)
        if self.spec.get('phase')=='collect-target-stable-task-eval-v1':
            from .stable_task_evaluation_v1 import run
            return run(self)
        if self.spec.get('phase')=='collect-target-stable-task-data-v1':
            from .stable_task_data_v1 import run
            return run(self)
        if self.spec.get('phase')=='collect-target-stable-start-short-v1':
            from .stable_start_short_task_v1 import run
            return run(self)
        if self.spec.get('phase')==C02_FEEDBACK_PHASE:
            from .fusion_c02_source_state_feedback_v2 import run
            return run(self)
        if self.spec.get('phase')==C02_PHASE:
            from .fusion_c02_source_state_initialization_v1 import run
            return run(self)
        if self.spec['phase']=='collect-target-fusion-task-phase-live-skill-v9':
            from .fusion_task_phase_live_skill_native_v9 import run
            return run(self)
        if self.spec['phase']=='collect-target-fusion-task-phase-live-skill-v8':
            from .fusion_task_phase_live_skill_native_v8 import run
            return run(self)
        if self.spec['phase']=='collect-target-fusion-task-phase-bias-fade-v8':
            from .fusion_task_phase_bias_fade_native_v8 import run
            return run(self)
        if self.spec['phase']=='collect-target-fusion-task-phase-pose-continuation-v7':
            from .fusion_task_phase_pose_continuation_native_v7 import run
            return run(self)
        if self.spec['phase']=='collect-target-fusion-task-phase-pose-feedback-v6':
            from .fusion_task_phase_pose_feedback_native_v6 import run
            return run(self)
        if self.spec['phase']=='collect-target-fusion-task-phase-xyz-feedback-continuation-v5':
            from .fusion_task_phase_xyz_feedback_continuation_native_v5 import run
            return run(self)
        if self.spec['phase']=='collect-target-fusion-task-phase-xyz-feedback-v4':
            from .fusion_task_phase_xyz_feedback_native_v4 import run
            return run(self)
        if self.spec['phase']=='collect-target-fusion-task-phase-bias-continuation-v3':
            from .fusion_task_phase_bias_continuation_native_v3 import run
            return run(self)
        if self.spec['phase']=='collect-target-fusion-task-phase-bias-calibration-v2':
            from .fusion_task_phase_bias_calibration_native_v2 import run
            return run(self)
        if self.spec['phase']=='collect-target-fusion-task-phase-initialization-v1':
            from .fusion_task_phase_initialization_native import run
            return run(self)
        if self.spec['phase']=='collect-target-fusion-source-posture-query-v3':
            from .fusion_source_posture_paused_query_v3 import run
            return run(self)
        if self.spec['phase']=='collect-target-fusion-source-posture-qualification-v1':
            from .fusion_source_posture_qualification_native import run
            return run(self)
        if self.spec['phase']=='collect-target-fusion-live-proposal-qualification-v3':
            from .fusion_live_proposal_qualification_native import run
            return run(self)
        if self.spec['phase']=='collect-target-fusion-paused-query-v2':
            from .fusion_paused_query_native import run
            return run(self)
        if self.spec['phase']=='collect-target-fusion-trajectory-response-v1':
            from .fusion_trajectory_response_native import run
            return run(self)
        if self.spec['phase']in('collect-target-fusion-trajectory-qualification-v1','collect-target-fusion-trajectory-slow-qualification-v2'):
            from .fusion_trajectory_qualification_native import run
            return run(self)
        if self.spec['phase']=='collect-target-fusion-reference-qualification-v1':
            from .fusion_reference_qualification_native import run
            return run(self)
        if self.spec['phase']in('collect-target-pilot1-supported-close','collect-target-pilot2-supported'):
            from .pilot1_supported_close import run
            return run(self)
        if self.spec['phase']=='collect-target-pilot1-contact':
            from .pilot1_contact import run
            return run(self)
        if self.spec['phase']=='collect-target-next2-contact':
            from .next2_contact import run
            return run(self)
        import numpy as np
        from .target_reference import trajectory_identity,require_live_window,installed_target_identities,fixed_goal_poses
        from go2arx5_vla.task_data.codec import pose10
        from scripts.pi0294.task_gripper_contract import teacher_physical_fraction
        from scripts.pi0292.durable import write
        self.prepare_dynamic();initial=self.measured_task_state();goal=initial['tcp'].copy();episode=None;late_decision=False;decision_applied=False
        next2=self.spec['phase']=='collect-target-next2'
        pilot1=self.spec['phase']=='collect-target-pilot1'
        next1=self.spec['phase']in('collect-target-next1','collect-target-next2','collect-target-pilot1');decision_atom=8;contact=False
        offset={'g2-eef-x':(0,.001),'g2-eef-y':(1,.001),'g2-standing-eef-x-v2':(0,.001),'g2-standing-eef-y-v2':(1,.001)}.get(self.spec['phase'])
        if next1:
            from .next1_episode import episode_config,assess
            if next2:
                from .next2_episode import episode_config
            if pilot1:
                from .pilot1_episode import episode_config
            episode=episode_config(self.spec);self.next1_samples=[initial];contact=episode['purpose']=='CONTACT_GRADED'
            if episode.get('schema')=='WM02_FUSION_LOCAL_CALIBRATION_EPISODE_V1':
                self.fusion_initial_tcp=initial['tcp'].copy();self.fusion_base_pose_hold=list(self.base_pose_hold)
                self.fusion_selected_action=0;self.fusion_install_action_count=0
            if contact:self.contact_geometry_samples=[]
            late_decision=not episode['purpose'].startswith('STEADY_');decision_atom=episode['screen_atom']
            decision_goal=goal.copy();decision_goal[:3,3]+=np.asarray(episode['offset_xyz_m'])
        elif self.spec['phase']=='collect-target-night1':
            from .target_manifest import episode_config
            episode=episode_config(self.spec)
            from .target_manifest import DECISION_RULE
            late_decision=episode['reference_rule']==DECISION_RULE
            decision_goal=goal.copy();decision_goal[:3,3]+=np.asarray(episode['offset_xyz_m'])
            if not late_decision:goal=decision_goal
        elif self.spec['phase'].startswith('collect-target-'):
            index=int(self.spec['phase'][-2:]);offset={3:(0,.001),4:(0,-.001),5:(1,.001),6:(1,-.001)}.get(index)
            if self.spec['phase'].startswith('collect-target-parallel-'):offset=(0,.001 if index==1 else-.001)
        if offset:goal[offset[0],3]+=offset[1]
        # Original EEF compiler limits still apply. No BASE command labels.
        def action_at_current(first_decision=False):
            if contact and decision_applied:
                from .next1_contact import chunk
                poses=chunk(self.contact_path,(self.observe()['step']-2503)//25-decision_atom)
            else:poses=fixed_goal_poses(initial['tcp'],goal,renewed=hasattr(self,'active_reference')and not first_decision)
            return np.stack([pose10(p,float(teacher_physical_fraction(1.)))for p in poses])
        windows=[];self.install_reference(action_at_current());start=2503;installed_step=start
        while self.observe()['step']<self.spec['max_ticks']:
            before=self.observe();step=before['step']
            if next1 and step==start+episode['prehold_atoms']*25:
                # Refresh the same fixed goal at the sampled-history boundary.
                # This is an installed renewal, never an achieved-state reanchor.
                self.install_reference(action_at_current());installed_step=step
            if next1 and step==start+episode['screen_atom']*25:
                value=assess(self,episode)
                if late_decision and not value['qualified']:raise RuntimeError('UNSTABLE_DECISION_PREFIX_ACTION_NOT_INSTALLED')
            if late_decision and step==start+decision_atom*25 and not decision_applied:
                if next2:
                    from .next2_episode import predict_before_install
                    predict_before_install(self,episode)
                if pilot1:
                    from .pilot1_episode import choose
                    chosen=choose(self,episode)
                    episode=dict(episode,offset_xyz_m=chosen.tolist())
                    decision_goal=initial['tcp'].copy();decision_goal[:3,3]+=chosen
                # The burn prefix receives a hold-only H40. Reveal/install the
                # chosen small goal at this actual pause, before its response.
                if contact:
                    from .next1_contact import planned_path
                    if episode['contact_grade']=='DESCEND_20_OPEN':
                        from scripts.pi0294.persistent_teacher import geometry
                        state=self.measured_task_state()
                        if not geometry(self.actual_state,state,'T1')['precontact_geometry_ready']:raise RuntimeError('CURRENT_ACTUAL_PRECONTACT_GEOMETRY_REQUIRED_NO_RESET_SUBSTITUTION')
                    self.contact_path=planned_path(initial['tcp'],self.measured_task_state()['object_pose'],episode['contact_grade'])
                    write(self.output/'CONTACT_PLANNED_REFERENCE.private.json',dict(grade=episode['contact_grade'],path=self.contact_path.tolist(),planned_reference_only=True,achieved_state_reanchor=False))
                goal=decision_goal;decision_applied=True
                self.command_event(dict(phase='decision-boundary',native_step=step,sim_time_s=before['time_s'],reference_rule=episode['reference_rule'],offset_xyz_m=episode['offset_xyz_m'],goal_pose=goal.tolist(),future_response_observed=False))
                self.install_reference(action_at_current(first_decision=True));installed_step=step
            # V1 dynamic cadence: seven complete relative atoms (350ms),
            # preserving the original source-relative .4s receiver TTL.
            if step-installed_step>=175:self.install_reference(action_at_current());installed_step=step
            clock=require_live_window(self.active_reference['source_mpc_s'],before['time_s'],25)
            event_start=len(self.ros_evidence);after=self.advance_to(step+25);self.spin(.15)
            if next1:self.next1_samples.append(self.measured_task_state())
            if contact:
                from scripts.pi0294.persistent_teacher import geometry
                state=self.next1_samples[-1];g=geometry(self.actual_state,state,'T1');row=dict(native_step=state['native_step'],**{k:v for k,v in g.items()if k not in('actual_pad_corners_world','actual_object_world')});self.contact_geometry_samples.append(row);self.command_event(dict(phase='contact-geometry',**row))
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
        if next1 and not hasattr(self,'next1_screen'):assess(self,episode)
        if pilot1:
            from .pilot1_episode import score
            score(self,episode)
        if contact:
            samples=self.next1_samples
            write(self.output/'CONTACT_GRADE_RESULT.private.json',dict(grade=episode['contact_grade'],completed=True,geometry=self.contact_geometry_samples,any_precontact=any(g['precontact_geometry_ready']for g in self.contact_geometry_samples),precontact_end_qualified=len(self.contact_geometry_samples)>=4 and all(g['precontact_geometry_ready']for g in self.contact_geometry_samples[-4:]),native_steps=[s['native_step']for s in samples],TCP_world_xyz_m=[s['tcp'][:3,3].tolist()for s in samples],object_xyz_m=[s['object_pose'][:3,3].tolist()for s in samples],left_object_contact=sum(bool(s['left_object_contact'])for s in samples),right_object_contact=sum(bool(s['right_object_contact'])for s in samples),native_grasp=sum(bool(s['native_grasp'])for s in samples),safety_valid=all(s['safety_valid']for s in samples),full_grasp_release_success=False,reference_path_end_reached_in_time=len(windows)-decision_atom>=len(self.contact_path)-1))
        self.capture_loaded_identity();cancel=self.cancel_reference();self.seal_trace(self.spec['max_ticks'])
        confirmed=sum(bool(w['solver_policies_with_exact_installed_target'])for w in windows)
        value=dict(schema='WM02_TARGET_DYNAMIC_DIAGNOSTIC_V1',phase=self.spec['phase'],actual_collection_start_step=start,actual_collection_start_s=initial['time_s'],task_native_ticks=self.observe()['step']-start,complete_task_atoms=len(windows),partial_task_ticks=0,solver_target_confirmed_atoms=confirmed,controller_selected_reference='DERIVED_ONLY_NO_SELECTED_POLICY_ID_EXPORTED',windows=windows,operator_cancel=cancel,observer_safety_stop='NOT_TESTED',full_task_success=False,BASE_velocity_labels='ABSENT',BASE_pose_reference='MEASURED_INITIAL_MPC_POSE_HOLD'if hasattr(self,'base_pose_hold')else'ABSENT',replan_cadence_s=.35,source_relative_ttl_s=.4)
        if episode:value.update(reference_rule=episode['reference_rule'],late_decision_applied=decision_applied,decision_atom=decision_atom if late_decision else None)
        if next1:value.update(sample_start_step=start+episode['prehold_atoms']*25,steady_prefix=self.next1_screen)
        write(self.output/'DYNAMIC_DIAGNOSTIC.json',value);self.reference_qualification=value
        if confirmed!=len(windows):raise RuntimeError('SOLVER_TARGET_CONSUMPTION_NOT_CONFIRMED_FOR_EVERY_ATOM')
        return value
    def ttl_diagnostic(self):
        import numpy as np
        from .target_reference import ttl_pause_points
        from go2arx5_vla.task_data.codec import pose10
        from scripts.pi0294.task_gripper_contract import teacher_physical_fraction
        from scripts.pi0292.durable import write
        if self.spec['phase']!='g2-standing-ttl-v2':raise PermissionError('EXACT_BOUNDED_TTL_DIAGNOSTIC_REQUIRED')
        self.prepare_dynamic();state=self.measured_task_state()
        self.install_reference(np.stack([pose10(state['tcp'],float(teacher_physical_fraction(1.)))]*40))
        self.ttl_boundaries=ttl_pause_points(self.active_reference['source_mpc_s']);samples=[]
        def expired():return any(r['kind']=='lifecycle'and r.get('reason')=='PREFIX_TTL_EXPIRED'for r in self.reference_events())
        for label,step in zip(('before','at','after','final'),self.ttl_boundaries):
            if label in('after','final')and(self.latest['mode'].data!='HOLD'or not expired()):raise RuntimeError('NO_POST_DEADLINE_TICKS_WITHOUT_FRESH_EXPIRY_HOLD_ACK')
            self.advance_to(step)
            if label=='at':self.spin(2,lambda:expired()and self.latest['mode'].data=='HOLD')
            else:self.spin(.1)
            row=dict(label=label,step=self.observe()['step'],sim_time_s=self.observe()['time_s'],source_mpc_s=self.active_reference['source_mpc_s'],deadline_s=self.active_reference['deadline_s'],last_observation_s=float(self.latest['observation'].time),mode=self.latest['mode'].data,expiry_recorded=expired())
            samples.append(row);write(self.output/f'TTL_{label.upper()}.json',row)
            if label=='before'and(expired()or row['mode']!='WBC'):raise RuntimeError('REFERENCE_EXPIRED_BEFORE_DEADLINE')
            if label=='at'and(not expired()or row['mode']!='HOLD'):raise RuntimeError('TTL_EXPIRY_NOT_OBSERVED_AT_EXACT_DEADLINE')
        self.seal_trace(self.spec['max_ticks']);value=dict(schema='WM02_NATIVE_TTL_DIAGNOSTIC_V1',samples=samples,learning_atoms=0,reason='Mode transition at source TTL is a diagnostic, not constant-reference supervision',HOLD_stops_physics=False,observer_safety_STOP='NOT_TRIGGERED_INTENTIONALLY')
        write(self.output/'TTL_DIAGNOSTIC.json',value);self.reference_qualification=value;return value
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
        from .native_collection_performance_v1 import close_resources
        try:
            with measure(self,'native.close'):
                self._close_native()
        finally:
            close_resources(self)
    def _close_native(self):
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
