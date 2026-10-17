import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .mp1 import t2_config as config
from .mp1.science import Ledger
from .mp1.recovery_config import limits
from .mp1.t2_shadow import summarize
from .mp1.t2_figure import chart_rows


class SavedT2ShadowTests(unittest.TestCase):
    def test_no_neural_training_live_or_recovery_authority(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);stage=dict(schema=config.SCHEMA,stage_caps=config.CAPS)
            (root/'MP1_STAGE_BINDING.json').write_text(json.dumps(stage))
            (root/'RESOURCE_LEDGER.json').write_text(json.dumps(dict(reservations={})))
            ledger=Ledger(root,'core')
            for key in ['main_starts','recovery_forks','live_and_recovery_integrations','generation_queries','optimizer_updates']:
                with self.assertRaises(PermissionError):ledger.charge(key,1)
            with ledger.steps('shadow_integrations',21000,'saved-only') as meter:meter['actual']=21000
            with self.assertRaises(PermissionError):
                with ledger.steps('shadow_integrations',1,'over'):pass
            with self.assertRaises(PermissionError):limits(stage)
        self.assertEqual(config.CPU_CAPS,dict(core=3600,hopper=0));self.assertEqual(len(config.ATTEMPTS),2)

    def test_all_overlapping_original_collision_rules_are_retained(self):
        poses=np.repeat(np.eye(4)[None],40,axis=0);poses[:,2,3]=.5
        check=dict(illegal_pairs=[dict(original_rule='ARM_TABLE_OR_FLOOR',signed_distance_m=-.005)],requested_TCP_at_end=poses[0].tolist(),actual_TCP_at_end=poses[0].tolist())
        result=dict(attempt_id='fixture',states=[],collision_checks=[check],same_state_comparison=dict(
            source_actual_TCP=poses[0].tolist(),model_H40_poses=poses.tolist(),expert_H40_poses=poses.tolist(),model_H40_gripper=[0.]*40,expert_H40_gripper=[0.]*40))
        value=summarize(result)
        self.assertEqual(value['collision_checks'][0]['illegal_pairs'][0]['all_original_rules_triggered'],['ARM_TABLE_OR_FLOOR','PENETRATION_STOP'])
        self.assertNotIn('model_H40_poses',value['same_state_comparison'])
        self.assertEqual(value['signed_prefix_XYZ_model_minus_expert_m'],[[0.,0.,0.]]*8)

    def test_figure_uses_only_exact_native_diagnostics_and_saved_events(self):
        labels=['MODEL','SAME_STATE_EXPERT','EXPERT_XYZ_ONLY','EXPERT_SO3_ONLY','EXPERT_GRIPPER_ONLY']
        rows=[dict(attempt_id=identifier,maximum_actual_highrate_difference=0,
            collision_checks=[dict(candidate=label,decision='REJECT' if label=='MODEL' else 'PASS',minimum_reported_illegal_arm_surface_contact_distance_m=-.00002) for label in labels]) for identifier in config.ATTEMPTS]
        saved=[dict(attempt_id=i,original_duration_s=18,progress=dict(first_saved_robot_contact_s=11,first_goal_zone_all_corners_s=16)) for i in config.ATTEMPTS]
        a=dict(status='ACTUAL_EXACT_HISTORY_SHADOW_ATTRIBUTION_COMPLETE',rows=rows);p=dict(rows=saved)
        self.assertEqual([r['depth_mm'] for r in chart_rows(a,p)],[.02,.02])
        rows[0]['maximum_actual_highrate_difference']=1e-8
        with self.assertRaises(ValueError):chart_rows(a,p)
