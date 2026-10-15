"""Versioned native world-reference receiver; original interpolation and costs remain."""
import hashlib
import json
import math
import os
from pathlib import Path
import time


def bind_packet(document, *, episode, sequence, now, trajectory_type, ee, base):
    meta=document['_pi0294']
    source=meta['source_mpc_s'];ttl=meta['ttl_s'];seq=meta['sequence']
    if (meta['episode']!=episode or type(seq) is not int or seq!=sequence+1
            or isinstance(source,bool) or not isinstance(source,(int,float))
            or not math.isfinite(source) or ttl!=.4 or not 0<=now-source<=.025):
        raise ValueError('IDENTITY_SEQUENCE_OR_SOURCE_CLOCK_REJECT')
    if meta.get('operation')=='cancel':return None,source,seq
    if meta.get('operation')!='install' or 'waypoints' in document or 'ee' not in document:
        raise ValueError('EXPLICIT_WORLD_EEF_REQUIRED')
    for part in ('ee','base'):
        if part not in document:continue
        p=document[part];points=p['waypoints']
        if (p['frame']!='world' or p.get('start_delay',0)!=0 or p.get('lead_in',0)!=0
                or len(points)!=41 or any(abs(x['time']-k*.05)>1e-10 for k,x in enumerate(points))):
            raise ValueError('OBSERVATION_BOUND_CLOSED_WORLD_GRID_REQUIRED')
    bound=trajectory_type.from_dict(document).bind(ee,base,source)
    # Native bind stores measured start_pose even for a zero-lead world path.
    # Preserve action[0], rather than silently replacing it with a new observation.
    bound.ee.start_pose=None
    if bound.base is not None:bound.base.start_pose=None
    return bound,source,seq


def main():
    import rclpy
    from go2_x5_planning.ee_trajectory import TaskTrajectory
    from go2_x5_planning.ee_trajectory_node import EETrajectoryManager
    from std_msgs.msg import String
    original_init=EETrajectoryManager.__init__
    original_send=EETrajectoryManager._send
    original_observation=EETrajectoryManager.on_observation
    trace=Path(os.environ['PI0294_PLANNER_TRACE'])
    episode=os.environ['PI0294_EPISODE']
    from .teacher_qualification import prefix_boundary_pause
    from .task_state import read_state
    from .task_capture_runtime import read
    output=trace.parent

    def record(kind,**kw):
        with trace.open('a') as f:f.write(json.dumps(dict(kind=kind,monotonic_s=time.monotonic(),**kw),allow_nan=False)+'\n')

    def hold(self,reason):
        self.trajectory_started=False;self.bound_start=None;self._r4_deadline=None;self._r4_pending=None
        msg=String();msg.data='HOLD';self._r4_mode.publish(msg)
        record('lifecycle',reason=reason,action='ORIGINAL_NATIVE_HOLD')

    def accept(self,msg):
        try:
            if self.observation is None or self.ee_state is None or not self.reference_enabled:
                raise ValueError('MEASURED_WBC_INPUT_REQUIRED')
            document=json.loads(msg.data);source=document['_pi0294']['source_mpc_s']
            now=float(self.observation.time)
            # DDS may deliver measured EE before the matching MPC observation.
            # Wait for that observation; never accept against a future timestamp.
            if (isinstance(source,(float,int)) and not isinstance(source,bool)
                    and math.isfinite(source) and 0<source-now<=.025):
                if self._r4_pending is not None:raise ValueError('PENDING_TRANSACTION_OCCUPIED')
                self._r4_pending=msg;record('waiting_for_source_observation',source_mpc_s=source,receiver_mpc_s=now)
                return
            bound,source,sequence=bind_packet(document,episode=episode,
                sequence=self._r4_sequence,now=float(self.observation.time),trajectory_type=TaskTrajectory,
                ee=self.ee_state,base=self.base_state())
        except (KeyError,ValueError,TypeError,OverflowError,RecursionError) as exc:
            record('reject',error=str(exc),wire_sha256=hashlib.sha256(msg.data.encode()).hexdigest())
            hold(self,'INVALID_PACKET');return
        self._r4_sequence=sequence
        if bound is None:hold(self,'CANCEL');return
        self.trajectory_source=self.trajectory=bound;self.bound_start=source
        self.goal=self.base_goal=None;self.trajectory_started=True
        self._r4_deadline=source+.4;self._r4_ref=hashlib.sha256(msg.data.encode()).hexdigest()
        self._r4_boundary_step=round((source+.4)/.002)
        record('accept',ref_id=self._r4_ref,raw=msg.data,source_mpc_s=source,
            receiver_mpc_s=float(self.observation.time),deadline=self._r4_deadline)
        self.publish_trajectory()

    def observation(self,msg):
        original_observation(self,msg)
        if self._r4_pending is not None:
            pending=self._r4_pending
            source=json.loads(pending.data)['_pi0294']['source_mpc_s']
            if float(msg.time)>=source:
                self._r4_pending=None;accept(self,pending)

    def send(self,msg):
        original_send(self,msg)
        record('installed',ref_id=getattr(self,'_r4_ref',None),bound_start=self.bound_start,
            observation_time_s=None if self.observation is None else float(self.observation.time),
            times=list(msg.time_trajectory),poses=[list(s.value) for s in msg.state_trajectory],
            inputs=[list(s.value) for s in msg.input_trajectory])

    def expiry(self):
        current=(self.observation is not None,self.ee_state is not None,bool(self.reference_enabled))
        if current!=getattr(self,'_r1_previous_readiness',None):
            record('receiver_input_readiness',has_observation=current[0],has_measured_EE=current[1],reference_enabled=current[2],
                ready=all(current),receiver_mpc_s=None if self.observation is None else float(self.observation.time))
            self._r1_previous_readiness=current
        if (self._r4_deadline is not None and self.observation is not None
                and float(self.observation.time)>=self._r4_deadline-1e-10):
            try:
                boundary=prefix_boundary_pause(now=float(self.observation.time),deadline=self._r4_deadline,
                    pause=read(output/'pause_ack.json'),packet=read_state(output/'state.bin'),expected_step=self._r4_boundary_step)
            except (FileNotFoundError,ValueError,KeyError):boundary=False
            if boundary:
                if self._r4_boundary_recorded!=self._r4_ref:
                    record('prefix_boundary_paused',ref_id=self._r4_ref,time_s=float(self.observation.time),
                        step=self._r4_boundary_step,TTL_extended=False,further_physics_allowed=False)
                    self._r4_boundary_recorded=self._r4_ref
                return
            hold(self,'PREFIX_TTL_EXPIRED')

    def init(self):
        original_init(self);self._r4_sequence=-1;self._r4_deadline=None;self._r4_pending=None;self._r4_boundary_recorded=None
        self._r4_mode=self.create_publisher(String,'/go2_x5/arm/mode/request',10)
        self._r4_timer=self.create_timer(.002,lambda:expiry(self))

    EETrajectoryManager.__init__=init
    EETrajectoryManager.on_trajectory_json=accept
    EETrajectoryManager.on_observation=observation
    EETrajectoryManager._send=send
    rclpy.init();node=EETrajectoryManager()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':main()
