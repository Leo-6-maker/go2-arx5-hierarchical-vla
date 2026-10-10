"""Actual delivered parser/callback CPU comparison; no simulator or controller query."""
import json
import os
from pathlib import Path
import runpy
import time
from types import SimpleNamespace
import rclpy
from go2_x5_planning.ee_trajectory_node import EETrajectoryManager
from go2_x5_planning.ee_trajectory import TaskTrajectory,Pose
from go2arx5_vla.target_mujoco.reference import CONDITIONS,native_document

def main():
    rclpy.init();node=EETrajectoryManager();holds=[]
    node._pi0289_mode_pub=SimpleNamespace(publish=lambda msg:holds.append(msg.data))
    initial=[.4,0,.6,0,0,0,1]
    node.observation=SimpleNamespace(time=1.0)
    try:
        for sequence,condition in enumerate(CONDITIONS):
            document=native_document(condition,initial,h40=condition==CONDITIONS[-1])
            before=TaskTrajectory.from_json(json.dumps(document)).bind(Pose.from_state(initial),start_time=1.0)
            document['_pi0289']=dict(episode=os.environ['PI0289_EPISODE'],sequence=sequence,issued_monotonic_s=time.monotonic())
            node.on_trajectory_json(SimpleNamespace(data=json.dumps(document)))
            after=node.trajectory_source.bind(Pose.from_state(initial),start_time=1.0)
            a=before.window(1.0,2.0,.05);b=after.window(1.0,2.0,.05)
            assert a[0]==b[0] and [p.as_state() for p in a[1]]==[p.as_state() for p in b[1]]
        source=node.trajectory_source
        for meta in [dict(episode='old',sequence=6,issued_monotonic_s=time.monotonic()),dict(episode=os.environ['PI0289_EPISODE'],sequence=6,issued_monotonic_s=time.monotonic()-1),dict(episode=os.environ['PI0289_EPISODE'],sequence=5,issued_monotonic_s=time.monotonic()),dict(episode=os.environ['PI0289_EPISODE'],sequence=True,issued_monotonic_s=time.monotonic())]:
            document['_pi0289']=meta;count=len(holds)
            node.on_trajectory_json(SimpleNamespace(data=json.dumps(document)))
            assert len(holds)==count+1 and holds[-1]=='HOLD' and node.trajectory_source is source
        document['_pi0289']=dict(episode=os.environ['PI0289_EPISODE'],sequence=6,issued_monotonic_s=time.monotonic())
        node.on_trajectory_json(SimpleNamespace(data=json.dumps(document)))
        node.trajectory=source.bind(Pose.from_state(initial),start_time=1.0);node.bound_start=1.0
        node.publish_trajectory();deadline=node._pi0289_deadline;assert deadline==node.trajectory.end_time
        node.bound_start=None;node.observation.time=deadline+.001
        # Exercise the same registered timer callback after native end-time clears bound_start.
        node._pi0289_timer.callback();assert holds[-1]=='HOLD' and node._pi0289_deadline is None
        receipt=dict(status='ACTUAL_NATIVE_CALLBACK_CPU_PASS',panels_byte_equal=6,negative_cases=4,expiry_after_native_bound_clear='PASS',physics_integrations=0,WBC_forwards=0,GR00T_forwards=0,not_live_qualification=True)
        Path(os.environ['PI0289_CPU_RECEIPT']).write_text(json.dumps(receipt,indent=2))
        print(json.dumps(receipt))
    finally:node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
