import dataclasses,unittest
import numpy as np
from go2arx5_vla.world_model.fusion_contract import ActionSemantics,decode
from go2arx5_vla.world_model.fusion_proposals import source_execution_schedule,frozen_inverse_normalization_audit,target_gripper_stroke_audit


class FrozenSourceProposal(unittest.TestCase):
    def test_issue_zero_and_post_command_samples_are_not_shifted_or_preview_labels(self):
        value=source_execution_schedule(7.406)
        self.assertEqual(value['command_issue_timestamps_s'][0],7.406)
        self.assertAlmostEqual(value['committed_command_issue_s'][-1],7.756)
        self.assertAlmostEqual(value['expected_post_command_sample_s'][0],7.456)
        self.assertAlmostEqual(value['expected_post_command_sample_s'][-1],7.806)
        self.assertFalse(value['measured_source_outcomes']);self.assertEqual(value['committed_native_integrations'],200)
        s=ActionSemantics('WORLD','ABSOLUTE','QUAT_XYZW','OPEN_FRACTION','m',.05,8,'x5_ee','1'*64,
            point_time_origin='OBSERVATION_INCLUSIVE_ISSUE')
        args=dict(positions=np.zeros((40,3)),orientations=np.tile([0,0,0,1],(40,1)),gripper=np.ones(40),
            timestamps=value['command_issue_timestamps_s'],observation_timestamp=7.406,world_from_source=np.eye(4),
            observed_tcp_world=np.eye(4),semantics=s,provenance='SEMANTICS_UNRESOLVED',proposal_id='fixture-only')
        chunk=decode(**args);np.testing.assert_array_equal(chunk.point_timestamps,args['timestamps'])
        self.assertEqual(chunk.provenance['point_time_origin'],'OBSERVATION_INCLUSIVE_ISSUE')
        with self.assertRaises(ValueError):decode(**dict(args,semantics=dataclasses.replace(s,point_time_origin='OBSERVATION_PLUS_DT')))
        with self.assertRaises(ValueError):source_execution_schedule(7.406,committed_points=16)

    def test_inverse_clip_accounting_preserves_original_raw_and_physical_actions(self):
        raw=np.zeros((1,40,132));physical=np.zeros((40,10));raw[0,0,0]=-1.2;raw[0,1,9]=1.5;raw[0,2,60]=3.
        saved=raw.copy();v=frozen_inverse_normalization_audit(raw,physical)
        self.assertEqual(v['outside_unit_components'],2);self.assertEqual(v['affected_points'],2)
        self.assertEqual(v['outside_unit_components_by_channel'][9],1)
        np.testing.assert_array_equal(raw,saved);np.testing.assert_array_equal(physical,0)
        self.assertFalse(v['fusion_clipping_applied'])

    def test_source_full_stroke_is_rejected_not_silently_mapped_as_teacher_logical(self):
        values=np.array([0.,.5,1.]);saved=values.copy();v=target_gripper_stroke_audit(values)
        self.assertEqual(v['outside_working_stroke_indices'],[0,2]);self.assertFalse(v['target_stroke_compatible'])
        self.assertFalse(v['rescaled_or_clamped']);np.testing.assert_array_equal(values,saved)
        self.assertTrue(target_gripper_stroke_audit(np.array([.002/.044,.5,.042/.044]))['target_stroke_compatible'])


if __name__=='__main__':unittest.main()
