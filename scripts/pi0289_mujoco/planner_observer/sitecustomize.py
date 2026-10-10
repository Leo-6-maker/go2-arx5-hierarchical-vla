"""Original planner observation plus explicit session/expiry guard; no trajectory changes."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import time

if os.environ.get('PI0289_EPISODE'):
    # Reuse the original acceptance/install observer and native parser/interpolation.
    runpy.run_path(str(Path(__file__).resolve().parents[2]/'pi0288_mujoco/planner_observer/sitecustomize.py'))
    from go2_x5_planning.ee_trajectory_node import EETrajectoryManager
    from std_msgs.msg import String
    original_init=EETrajectoryManager.__init__
    original_accept=EETrajectoryManager.on_trajectory_json
    original_send=EETrajectoryManager._send
    trace=Path(os.environ['PI0288_PLANNER_TRACE'])

    def record(self,reason,**extra):
        with trace.open('a') as f:f.write(json.dumps(dict(kind='lifecycle',reason=reason,wall_s=time.time(),monotonic_s=time.monotonic(),**extra))+'\n')

    def hold(self,reason):
        msg=String();msg.data='HOLD';self._pi0289_mode_pub.publish(msg)
        self._pi0289_duration=None
        self._pi0289_deadline=None
        record(self,reason,action='ORIGINAL_NATIVE_HOLD_REQUEST')

    def tick(self):
        if self._pi0289_duration is not None and self.observation is not None:
            if self._pi0289_deadline is not None and self.observation.time>self._pi0289_deadline:
                hold(self,'REFERENCE_EXPIRED')

    def send(self,msg):
        original_send(self,msg)
        if self._pi0289_duration is not None and self.bound_start is not None:
            self._pi0289_deadline=self.trajectory.end_time

    def init(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        self._pi0289_sequence=-1;self._pi0289_duration=None;self._pi0289_deadline=None
        self._pi0289_mode_pub=self.create_publisher(String,'/go2_x5/arm/mode/request',10)
        self._pi0289_timer=self.create_timer(.02,lambda:tick(self))

    def accept(self,msg):
        try:
            document=json.loads(msg.data);meta=document['_pi0289']
            if not isinstance(document.get('waypoints'),list):raise ValueError('REGISTERED_TOP_LEVEL_EEF_ONLY')
            if meta['episode']!=os.environ['PI0289_EPISODE']:raise ValueError('WRONG_EPISODE')
            sequence=meta['sequence'];issued=meta['issued_monotonic_s']
            if isinstance(sequence,bool) or not isinstance(sequence,int) or sequence<=self._pi0289_sequence:raise ValueError('STALE_SEQUENCE')
            if isinstance(issued,bool) or not isinstance(issued,(float,int)) or not 0<=time.monotonic()-issued<=.5:raise ValueError('STALE_OR_FUTURE_PACKET')
        except (KeyError,TypeError,ValueError) as exc:
            record(self,'REJECT_PACKET',detail=str(exc),wire_sha256=hashlib.sha256(msg.data.encode()).hexdigest())
            hold(self,'INVALID_REFERENCE_PACKET');return
        before=self.trajectory_source
        original_accept(self,msg)
        if self.trajectory_source is not before:
            self._pi0289_sequence=sequence
            # Same native relative end-time; timer uses actual native bound_start/MPC observation.
            self._pi0289_duration=self.trajectory_source.total_duration
            self._pi0289_deadline=None
            record(self,'ACCEPT_PACKET',sequence=sequence,duration_s=self._pi0289_duration)

    EETrajectoryManager.__init__=init
    EETrajectoryManager.on_trajectory_json=accept
    EETrajectoryManager._send=send
