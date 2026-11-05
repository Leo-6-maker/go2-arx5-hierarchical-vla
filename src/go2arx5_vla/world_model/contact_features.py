"""Past observed contact features; NumPy only for native replay sealing."""
import numpy as np


def state_feature(row):
    memory=row['actual_gripper_internal_memory']['values']
    return np.r_[row['qpos'],row['qvel'],row['ctrl'],row['TCP_m'],
                 np.asarray(row['object_pose'])[:3,3],row['finger_q'],
                 max(e['max_pad_cube_normal_force_N'] for e in row['native_contact_events']),
                 memory['target'],memory['opening'],memory['expiry']-row['native_end_step']*.002]

