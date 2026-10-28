import unittest
import numpy as np

from .mp1.t2_coverage import window_counts,analyze_episode
from .bank import PHASES,STAGE_PHASE


class T2CoverageTests(unittest.TestCase):
    def test_overlapping_H40_windows_are_not_independent_examples(self):
        counts=window_counts(['push']*20+['detach']*20+['withdraw']*20,[0,10,20])
        self.assertEqual(counts['anchor_stage_counts']['push'],2)
        self.assertEqual(counts['windows_containing_any_stage']['detach'],3)
        self.assertEqual(counts['windows_containing_any_stage']['withdraw'],2)
        self.assertEqual(counts['multi_stage_windows'],3)
        self.assertTrue(counts['containing_any_stage_counts_overlap'])
        with self.assertRaises(ValueError):window_counts(['push']*40,[1])
        with self.assertRaises(ValueError):window_counts(['unknown']*40,[0])

    def test_raw_stages_must_equal_frozen_manifest_phase_membership(self):
        stages=['approach']*200+['push']*200+['detach']*30+['withdraw']*70+['hold']*140
        row=dict(id='fixture',split='TRAIN',task='T2',windows=601,sha256='fixture',samples_sha256='fixture',group='scene',family='F0',seed=1,
            stage_anchors={p:[a for a in range(601) if STAGE_PHASE[stages[a]]==p] for p in PHASES},unmapped_stages={})
        packet=dict(valid_window=np.ones(601,dtype=bool),label_representation=np.asarray('ABSOLUTE_COMMITTED_EEF10'),committed_absolute10=np.zeros((640,10)),anchor_time_s=np.arange(601)/20)
        samples=[dict(privileged=dict(teacher_stage=s),supervision=dict(valid=True,actually_committed=True)) for s in stages]
        samples.append(dict(privileged=dict(teacher_stage='hold'),supervision=None))
        card=dict(id='fixture',split='TRAIN',task='T2',initial_state_sha256='fixture',source_xy=[0.,0.],goal_xy=[.2,0.],cube_yaw=0.)
        events=[dict(previous='approach',stage='push',time_s=10.),dict(previous='push',stage='detach',time_s=20.),dict(previous='detach',stage='withdraw',time_s=21.5),dict(previous='withdraw',stage='hold',time_s=25.)]
        result,actual=analyze_episode(row,packet,samples,events,card)
        self.assertEqual(result['exact_stage_windows']['anchor_stage_counts']['detach'],30)
        self.assertEqual(result['first_hold_s'],25.);self.assertEqual(len(actual),640)
        self.assertTrue(result['terminal_observation_supervision_absent_not_a_training_action'])
        samples[-1]['supervision']={}
        with self.assertRaises(ValueError):analyze_episode(row,packet,samples,events,card)
        samples[-1]['supervision']=None
        row['stage_anchors']['withdraw'].pop()
        with self.assertRaises(ValueError):analyze_episode(row,packet,samples,events,card)
        with self.assertRaises(PermissionError):analyze_episode(dict(row,split='DEV'),packet,samples,events,card)
