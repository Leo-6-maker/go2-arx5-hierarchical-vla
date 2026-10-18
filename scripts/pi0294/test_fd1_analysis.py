"""Fixtures for paired-bank grain and non-causal diagnostic claims."""
import unittest

from scripts.pi0294.fd1_analysis import compare_banks,range_of
from scripts.pi0294.fd1_endpoints import native_object_pose,in_zone,sample_divergence
import numpy as np


class FD1AnalysisTests(unittest.TestCase):
    def test_empty_range_is_missing_not_zero(self):
        self.assertIsNone(range_of([]))
        self.assertEqual(range_of([2,4]),dict(min=2,max=4))

    def pair(self):
        rows=[];noise={}
        for bank in [1,2]:
            name='attempt'+str(bank)
            commands=[dict(sequence=i,trajectory_sha256='action'+str(bank),camera_hashes={'body':str(i*bank)},
                           input_state_eef10=[i*bank],action_sha256=name+str(i)) for i in [0,1]]
            rows.append(dict(attempt_id=name,commands=commands,adjudication=dict(scene_group='scene',task='T2',status='SUCCESS' if bank==1 else 'VALID_TASK_FAIL'),
                actual_events=dict(final_goal_center_error_m=.1,minimum_goal_center_error_20Hz_m=.01),
                unchanged_original_score=dict(hold_subpredicates={'released':41})))
            noise[(name,0)]=dict(attempt_id=name,seed=bank,noise_sha256='noise'+str(bank),namespace='bank'+str(bank),action_sha256=name+'0',arrays_sha256='arrays')
        return rows,noise

    def test_divergence_and_unique_cause_are_separate(self):
        rows,noise=self.pair()
        r=compare_banks(*rows,noise)
        self.assertTrue(r['initial_two_camera_pixels_hashes_identical'])
        self.assertTrue(r['initial_state_eef10_identical'])
        self.assertEqual(r['first_recorded_H40_action_divergence_s'],0)
        self.assertEqual(r['first_recorded_camera_divergence_s'],.4)
        self.assertEqual(r['first_recorded_EEF_input_state_divergence_s'],.4)
        self.assertFalse(r['noise_is_unique_causal_mechanism_proven'])

    def test_noise_join_cannot_silently_use_other_action(self):
        rows,noise=self.pair()
        noise[('attempt1',0)]['action_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'NOISE_RECEIPT_JOIN'):
            compare_banks(*rows,noise)

    def test_native_free_joint_endpoint_not_last_20Hz_frame(self):
        q=np.zeros(15);q[-7:-4]=[.45,.1,.46];q[-4]=1.
        pose=native_object_pose(q)
        np.testing.assert_array_equal(pose[:3,3],[.45,.1,.46])
        np.testing.assert_array_equal(pose[:3,:3],np.eye(3))
        self.assertTrue(in_zone(pose,[.45,.1]))
        self.assertFalse(in_zone(pose,[.55,.1]))
        with self.assertRaisesRegex(ValueError,'QPOS'):
            native_object_pose(np.zeros(16))

    def test_actual_camera_grain_is_fifty_ms_not_query_grain(self):
        def row(value):
            pose=np.eye(4);pose[0,3]=value
            return dict(execution=dict(actual_tcp=pose.tolist(),actual_gripper_gap_m=0.),
                        policy_obs={c:dict(sha256=str(value)) for c in ['body_rgb','wrist_rgb']})
        r=sample_divergence([row(0),row(.001)],[row(0),row(.002)])
        self.assertEqual(r['first_original_20Hz_camera_difference_s'],.05)
        self.assertEqual(r['first_original_20Hz_TCP_or_gripper_difference_s'],.05)
