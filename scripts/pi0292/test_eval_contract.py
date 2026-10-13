import unittest
import numpy as np

from .eval_contract import uniform_train_ids,stage_anchors,noise_seed,matrix10,gate_action,error_metrics,summarize_records


class TestEvalContract(unittest.TestCase):
    def test_uniform_endpoints_and_no_replacement_anchors(self):
        values=uniform_train_ids([f'T291{i:03d}' for i in reversed(range(64))])
        self.assertEqual(len(set(values)),16)
        self.assertEqual(values[0],'T291000');self.assertEqual(values[-1],'T291063')
        samples=[{'privileged':{'teacher_stage':s}} for s in ['close','close','lift','carry','release']]
        anchors,missing=stage_anchors(samples)
        self.assertEqual(anchors[0],dict(anchor=0,stages=['start','close']))
        self.assertEqual(missing,['hold']);self.assertEqual(len(anchors),4)

    def test_native_row_rotation_not_column_shortcut(self):
        angle=.4;c,s=np.cos(angle),np.sin(angle)
        rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        action=np.r_[.1,.2,.3,rotation[:2].reshape(-1),.7]
        np.testing.assert_allclose(matrix10(action)[:3,:3],rotation,atol=1e-15)
        self.assertGreater(np.max(np.abs(matrix10(action)[:3,:3]-rotation.T)),.1)

    def test_gate_is_noncorrective_and_rejected_errors_retained(self):
        state=np.r_[.4,0.,.5,1.,0.,0.,0.,1.,0.,1.]
        expert=np.repeat(state[None],40,axis=0)
        accepted=gate_action(expert,state,0.,'expert')
        self.assertNotEqual(accepted['decision'],'REJECT')
        prediction=expert.copy();prediction[10,0]+=2.
        before=prediction.copy()
        rejected=gate_action(prediction,state,0.,'model')
        self.assertEqual(rejected['decision'],'REJECT')
        np.testing.assert_array_equal(before,prediction)
        errors=error_metrics(prediction,expert,state)
        self.assertGreater(errors['H40']['XYZ_euclidean_m']['mean'],0.)
        self.assertEqual(errors['prefix8']['XYZ_euclidean_m']['mean'],0.)

    def test_invalid_orientation_and_gripper_fail_without_projection(self):
        state=np.r_[.4,0.,.5,1.,0.,0.,0.,1.,0.,1.]
        actions=np.repeat(state[None],40,axis=0);actions[0,3:9]=0
        self.assertEqual(gate_action(actions,state,0.,'bad')['decision'],'REJECT')
        actions=np.repeat(state[None],40,axis=0);actions[0,9]=1.01
        self.assertEqual(gate_action(actions,state,0.,'bad')['decision'],'REJECT')

    def test_frozen_paired_noise_identity(self):
        self.assertEqual(noise_seed(29201,'T291000',0),noise_seed(29201,'T291000',0))
        self.assertNotEqual(noise_seed(29201,'T291000',0),noise_seed(29202,'T291000',0))
        self.assertNotEqual(noise_seed(29201,'T291000',0),noise_seed(29201,'T291000',1))
        with self.assertRaises(ValueError):noise_seed(0,'T291000',0)

    def test_macro_keeps_rejected_and_incomplete_rows(self):
        state=np.r_[.4,0.,.5,1.,0.,0.,0.,1.,0.,1.]
        expert=np.repeat(state[None],40,axis=0);prediction=expert.copy();prediction[:,0]+=1
        row=dict(id='T291000',split='TRAIN',gate=dict(decision='REJECT',issues=[dict(code='TEST')]),
            errors=error_metrics(prediction,expert,state))
        summary=summarize_records([row])['TRAIN']
        self.assertAlmostEqual(summary['episode_macro']['H40']['XYZ_euclidean_m']['mean'],1)
        self.assertEqual(summary['decisions'],{'REJECT':1})
        prediction[0,0]=np.nan;row['errors']=error_metrics(prediction,expert,state)
        summary=summarize_records([row])['TRAIN']
        self.assertIsNone(summary['episode_macro']['H40']['XYZ_euclidean_m']['mean'])
        self.assertEqual(summary['episodes']['T291000']['H40']['XYZ_euclidean_m']['expected_points'],40)



if __name__=='__main__':unittest.main()
