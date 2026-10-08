"""The unchanged original gate; usable without loading the simulator."""
import numpy as np


def original_gate(command,state,now,slot):
    from go2arx5_vla.contracts.trajectory import TrajectoryChunkV1
    from go2arx5_vla.safety import TrajectorySafetyGateV1
    from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix,matrix_to_quat
    rotations=rot6d_rows_to_matrix(command[:,3:9])
    chunk=TrajectoryChunkV1(chunk_id=slot,source='PI0285',observation_timestamp=now,
        point_timestamps=now+np.arange(40)*.05,eef_position=command[:,:3],eef_orientation=matrix_to_quat(rotations),
        source_frame='PI0276_MANISKILL_WORLD',tcp_semantic_id='PI0276_PANDA_HAND_TCP',validity_start=now,validity_end=now+1.95,gripper=command[:,-1])
    return TrajectorySafetyGateV1().evaluate(chunk,now=now,achieved_position=state[:3],achieved_orientation=matrix_to_quat(rot6d_rows_to_matrix(state[None,3:9]))[0]).as_dict()
