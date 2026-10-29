"""Native EEF10 row encoding must survive a persistent Prefix8 boundary."""
import numpy as np

from go2arx5_vla.task_data.codec import pose10
from go2arx5_vla.trajectory.so3 import rot6d_to_matrix
from .task_path_teacher import executed_reference_history
from .task_path_reference import rates


def path():
    poses=np.repeat(np.eye(4)[None],48,axis=0)
    for index,pose in enumerate(poses):
        angle=.35+index*.01;c,s=np.cos(angle),np.sin(angle)
        pose[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]]
        pose[:3,3]=[.3+index*.002,.02,.7]
    return poses


def test_non_symmetric_executed_rotations_roundtrip_without_an_invented_join():
    poses=path();intervals=[dict(action_EEF10=pose10(pose,1),end_native_step=2500+(i+1)*25)
        for i,pose in enumerate(poses[:8])]
    before=np.stack([v['action_EEF10'].copy()for v in intervals])
    history=executed_reference_history(intervals)
    assert [v['end_native_step']for v in history]==[2675,2700]
    assert np.allclose(np.stack([v['pose']for v in history]),poses[6:8],atol=1e-14,rtol=0)
    assert np.array_equal(before,np.stack([v['action_EEF10']for v in intervals]))
    linear,angular=rates(np.concatenate([[v['pose']for v in history],poses[8:48]]),np.arange(42)*.05)
    assert linear<.06 and angular<.3
    wrong=[v['pose'].copy()for v in history]
    for pose,record in zip(wrong,intervals[-2:]):pose[:3,:3]=rot6d_to_matrix(record['action_EEF10'][3:9])
    assert rates(np.concatenate([wrong,poses[8:48]]),np.arange(42)*.05)[1]>.3


def test_initial_reference_has_no_fabricated_history():
    assert executed_reference_history([])==[]
