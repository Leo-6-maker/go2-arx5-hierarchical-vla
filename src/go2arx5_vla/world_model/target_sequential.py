"""Sequential native process adapter; no midpoint restore or alternative controller."""
from pathlib import Path
import json,os,signal,subprocess,time

def native_display_command(spec,output,environment):
    # Do not inject constructor-bearing native libraries into xvfb-run's shell,
    # xauth/Xvfb or utilities. Only the final simulator exec receives preload.
    env=dict(environment)
    for key in('LD_PRELOAD','WM02_INIT_TRACE','WM02_INIT_SEAL_REQUEST'):env.pop(key,None)
    argv=[spec['xvfb_run'],'-a','/usr/bin/env','LD_PRELOAD='+':'.join(spec[n]for n in('outer_library','observer_library','trace_library')),'WM02_INIT_TRACE='+str(Path(output)/'init.trace'),'WM02_INIT_SEAL_REQUEST='+str(Path(output)/'trace_seal_request'),spec['native_binary'],'go2_x5','scene']
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
            if args and args[0].decode()==str(binary)and os.getpgid(pid)==wrapper_pid:found.append(pid)
            pending.extend(int(n)for n in(root/'task'/str(pid)/'children').read_text().split())
        except(ProcessLookupError,FileNotFoundError):continue
    if len(found)!=1:raise RuntimeError('EXACT_OWNED_SIMULATOR_PID_REQUIRED')
    return found[0]

class TargetSequentialAdapter:
    def __init__(self,spec,output):
        self.spec=spec;self.output=Path(output);self.processes=[];self.handles=[];self.node=None;self.counts={};self.started=time.monotonic();self.commands=[];self.exit_records=[]
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
        from rclpy.qos import qos_profile_sensor_data
        self.rclpy=rclpy;self.String=String;rclpy.init();self.node=rclpy.create_node('wm02_auto1_'+self.spec['phase'].replace('-','_'))
        def record(key,msg):self.counts[key]=self.counts.get(key,0)+1
        self.subs=[self.node.create_subscription(cls,topic,lambda msg,k=key:record(k,msg),qos_profile_sensor_data)for key,topic,cls in(('odom','/go2_x5/slam/odom_mocap',Odometry),('arm','/go2_x5/arm/state',JointState))]
        self.keys=self.node.create_publisher(String,'/go2_x5/fsm/key',10);self.modes=self.node.create_publisher(String,'/go2_x5/arm/mode/request',10)
        self.spin(.3)
        if any(name!=self.node.get_name()for name in self.node.get_node_names()):raise RuntimeError('ISOLATED_ROS_DOMAIN_ALREADY_OCCUPIED')
        s=self.spec;out=self.output
        for name in('tmp','ros-logs','cache'):(out/name).mkdir()
        (out/'CONTACT_SAFETY_ARMED').touch();(out/'step_quota').write_text(str(s['max_ticks'])+'\n')
        warmup=(s['max_ticks']+499)//500
        (out/'quota.binding').write_text('\n'.join(('GRTQUOTA2',out.name,s['stage_identity_sha256'],out.name,str(s['max_ticks']),str(warmup),'0'))+'\n')
        env=dict(os.environ,**s['native_environment']);env.update(ROS_DOMAIN_ID=str(s['ROS_DOMAIN_ID']),ROS_LOCALHOST_ONLY='1',ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST',CUDA_VISIBLE_DEVICES='',TMPDIR=str(out/'tmp'),ROS_LOG_DIR=str(out/'ros-logs'),XDG_CACHE_HOME=str(out/'cache'),XDG_CONFIG_HOME=str(out/'cache'),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
        sim=dict(env,LD_PRELOAD=':'.join(s[n]for n in('outer_library','observer_library','trace_library')),PI0288_COMPILED_MODEL=s['model_path'],PI0288_PHYSICS_TRACE=str(out/'physics.jsonl'),PI0288_PHYSICS_CAP=str(s['max_ticks']),PI0288_CONTACT_SAFETY_MARKER=str(out/'CONTACT_SAFETY_ARMED'),PI0289_INITIALIZATION_TRACE=str(out/'initialization.jsonl'),PI0289_CONTROL_TRACE=str(out/'control.jsonl'),WM02_INIT_TRACE=str(out/'init.trace'),GR00T_TASK_GRIPPER_COMMAND=str(out/'gripper.bin'),GR00T_TASK_STATE=str(out/'state.bin'),GR00T_TASK_ERROR=str(out/'task_errors.txt'),GR00T_TASK_STEP_QUOTA=str(out/'step_quota'),GR00T_TASK_PAUSE_ACK=str(out/'pause_ack.json'),GR00T_TASK_QUOTA_BINDING=str(out/'quota.binding'),GR00T_TASK_RUN_UUID=out.name,GR00T_TASK_CARD_SHA256=s['stage_identity_sha256'],GR00T_TASK_RESERVATION_ID=out.name)
        command,display_env=native_display_command(s,out,sim);self.launch(command,'sim',display_env)
        native=Path(s['native_root']);params=['--ros-args','-p','use_sim_time:=true','-p','taskFile:='+str(native/'src/ctrl/go2_x5_ocs2/config/task_floating.info'),'-p','urdfFile:='+str(native/'src/ctrl/go2_x5_description/urdf/arx5_ac1_floating.urdf')]
        # OCS2 can regenerate cache libraries while constructing its interface.
        # Preserve canonical caches; each owned process gets its private copy.
        import shutil
        for name,key in(('solver','solver_binary'),('controller','controller_binary')):
            cache=out/'cache'/('ocs2-'+name);shutil.copytree(s['codegen_lib'],cache)
            self.launch([s[key],*params,'-p','libFolder:='+str(cache)],name,env)
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
        self.publish(self.keys,'ENTER')
        def paused():
            try:return json.loads((self.output/'pause_ack.json').read_text())['step']==ticks and self.observe()['step']==ticks
            except(FileNotFoundError,ValueError,KeyError):return False
        if not self.spin(35,paused):raise RuntimeError('EXACT_NATIVE_PAUSE_NOT_REACHED')
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
        return self.observe()
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
