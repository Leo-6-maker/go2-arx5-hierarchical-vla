import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .mp1.correction_event_exposure import correction_inventory,summarize_correction_visits
from .native_dataset import sha


def fixture(root):
    n=64;states=np.tile([0.,0.,.5,1.,0.,0.,0.,1.,0.,1.],(n,1))
    commands=states.copy();commands[7:40,9]=0
    valid=np.isin(np.arange(n),[0,8]);p=root/'labels.npz'
    np.savez(p,committed_absolute10=commands,actual_state10=states,valid_window=valid,
        rgb_sha256=np.asarray([['a'*64,'b'*64]]*n),label_representation='ABSOLUTE_COMMITTED_EEF10')
    row=dict(id='C1-T1',task='T1',split='TRAIN',source_student_episode='parent1',
        execution_actor='ORIGINAL_QUALIFIED_SOURCE_EXPERT',file=p.name,sha256=sha(p),
        expert_commands=n,actual_ACKs=8,legal_anchors=[0,8],windows=2,origin='E0_STATE_AGGREGATION_ONLY',
        anchor_tick=600,source_result_sha256='c'*64,full_recovery=True,full_recovery_success=True)
    return row,p


class CorrectionEvents(unittest.TestCase):
    def test_actual_gripper_events_do_not_impute_phase_or_withdraw(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);row,_=fixture(root);v=correction_inventory(root,row)
            self.assertEqual(v['events'],dict(close=[7],open=[40]))
            self.assertEqual(v['detach_and_withdraw_events'],'UNKNOWN_NOT_ZERO')
            self.assertEqual(set(v['qualified_window_fingerprints']),{'0','8'})
            self.assertEqual(len(set(v['qualified_window_fingerprints'].values())),2)

    def test_repeated_visits_actors_and_segment_events_are_distinct_grains(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);row,_=fixture(root);inv=correction_inventory(root,row)
            v=summarize_correction_visits({'C':inv},{'C':{0:3,8:2}})
            self.assertEqual(v['sample_visits'],5);self.assertEqual(v['independent_student_parents'],1)
            self.assertEqual(v['actor_visits'],dict(STUDENT_AT_CORRECTION_ANCHOR=3,EXPERT_AFTER_CORRECTION_ANCHOR=2))
            self.assertEqual(v['gripper_event_totals']['close']['prefix8_visit_hits'],3)
            self.assertEqual(v['gripper_event_totals']['open']['late_H40_visit_hits'],2)
            self.assertEqual(v['gripper_event_totals']['open']['occurrences_uncovered_in_sampled_prefix8'],1)

    def test_holes_in_legal_anchors_not_silently_treated_as_trainable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);row,_=fixture(root);inv=correction_inventory(root,row)
            with self.assertRaisesRegex(ValueError,'LEGAL_CORRECTION_ANCHORS'):
                summarize_correction_visits({'C':inv},{'C':{1:1}})
            with self.assertRaisesRegex(ValueError,'LEFT_JOIN'):
                summarize_correction_visits({'C':inv},{})

    def test_dev_T2_or_unqualified_actor_rejected_before_io(self):
        row=dict(split='TRAIN',task='T1',source_student_episode='p',execution_actor='ORIGINAL_QUALIFIED_SOURCE_EXPERT')
        for change in [dict(split='DEV'),dict(task='T2'),dict(execution_actor='STUDENT'),dict(source_student_episode=None)]:
            with self.subTest(change=change),self.assertRaises(PermissionError):
                correction_inventory('/missing',dict(row,**change))

    def test_tampered_numeric_or_unknown_shape_and_window_masks_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);row,p=fixture(root);raw=p.read_bytes()
            for mode in ['nonfinite','mask','ACK','short_state','representation']:
                p.write_bytes(raw)
                with np.load(p,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
                bad=copy.deepcopy(row)
                if mode=='nonfinite':arrays['committed_absolute10'][0,0]=np.nan
                elif mode=='mask':arrays['valid_window'][1]=True
                elif mode=='ACK':bad['actual_ACKs']=7
                elif mode=='short_state':arrays['actual_state10']=arrays['actual_state10'][:2]
                else:arrays['label_representation']=np.asarray('UNEXECUTED_DESIRED')
                np.savez(p,**arrays);bad['sha256']=sha(p)
                with self.subTest(mode=mode),self.assertRaises(ValueError):correction_inventory(root,bad)

    def test_path_escape_and_changed_numeric_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);row,p=fixture(root)
            with self.assertRaises(PermissionError):correction_inventory(root,dict(row,file='../outside'))
            p.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'NUMERIC_HASH'):correction_inventory(root,row)


if __name__=='__main__':unittest.main()
