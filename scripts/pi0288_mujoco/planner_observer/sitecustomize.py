"""Record native reference acceptance/binding without replacing its algorithm."""
import hashlib
import json
import os
from pathlib import Path
import time

if os.environ.get('PI0288_PLANNER_TRACE'):
    from go2_x5_planning.ee_trajectory_node import EETrajectoryManager

    path = Path(os.environ['PI0288_PLANNER_TRACE'])
    original_accept = EETrajectoryManager.on_trajectory_json
    original_send = EETrajectoryManager._send

    def record(row):
        row.update(wall_s=time.time(), monotonic_s=time.monotonic(), pid=os.getpid())
        with path.open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')

    def accept(self, msg):
        before = self.trajectory_source
        original_accept(self, msg)
        accepted = self.trajectory_source is not before
        if accepted:
            self._pi0288_ref = hashlib.sha256(msg.data.encode()).hexdigest()
        record(dict(kind='accept', raw=msg.data, accepted=accepted,
                    ref_id=hashlib.sha256(msg.data.encode()).hexdigest(),
                    observation_time_s=None if self.observation is None else float(self.observation.time)))

    def send(self, msg):
        original_send(self, msg)
        record(dict(kind='installed', ref_id=getattr(self, '_pi0288_ref', None),
                    bound_start=self.bound_start, reference_enabled=self.reference_enabled,
                    observation_time_s=None if self.observation is None else float(self.observation.time),
                    times=list(msg.time_trajectory), poses=[list(s.value) for s in msg.state_trajectory],
                    inputs=[list(s.value) for s in msg.input_trajectory]))

    EETrajectoryManager.on_trajectory_json = accept
    EETrajectoryManager._send = send
