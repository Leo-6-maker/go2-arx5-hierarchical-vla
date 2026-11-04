"""Sequential native process adapter; no midpoint restore or alternative controller."""
from pathlib import Path
import json,os,signal,subprocess,time

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
        self.launch(['xvfb-run','-a',s['native_binary'],'go2_x5','scene'],'sim',sim)
        native=Path(s['native_root']);params=['--ros-args','-p','use_sim_time:=true','-p','taskFile:='+str(native/'src/ctrl/go2_x5_ocs2/config/task_floating.info'),'-p','urdfFile:='+str(native/'src/ctrl/go2_x5_description/urdf/arx5_ac1_floating.urdf'),'-p','libFolder:='+s['codegen_lib']]
        self.launch([s['solver_binary'],*params],'solver',env);self.launch([s['controller_binary'],*params],'controller',env)
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
        return self.observe()
    def close(self):
        # Signal only the Popen-owned process groups; retain all actual exits.
        for name,p in reversed(self.processes):
            escalation=[]
            if p.poll()is None:
                for sig,wait in((signal.SIGINT,5),(signal.SIGTERM,5),(signal.SIGKILL,2)):
                    try:os.killpg(p.pid,sig);escalation.append(sig.name);p.wait(timeout=wait);break
                    except ProcessLookupError:break
                    except subprocess.TimeoutExpired:continue
            self.exit_records.append(dict(name=name,pid=p.pid,exit_code=p.poll(),signals=escalation))
        for h in self.handles:h.close()
        if self.node is not None:self.node.destroy_node();self.rclpy.shutdown()
