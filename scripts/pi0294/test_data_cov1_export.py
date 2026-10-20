"""Synthetic saved data, not physical task or official processor evidence."""
from copy import deepcopy
import unittest

import numpy as np

from scripts.pi0294.data_cov1.plan import candidate,slots,capture_cards
from scripts.pi0294.data_cov1.export import packet,horizon_kinematics,event_index
from scripts.pi0294.pv1.export import packet as legacy_packet
from scripts.pi0294.test_pv1_export import fixture


def new_fixture():
    old,samples=fixture()
    from scripts.pi0294.scene_cards import task_card
    card=task_card(candidate(**slots()[0],index=0),'T1')
    card.update(id=card['id']+'-C04',nominal_clearance_m=.004)
    for s in samples:
        s['policy_obs']['instruction']=card['instruction'];s['causal_history']['episode_id']=card['id']
    return card,samples


class DataCov1Export(unittest.TestCase):
    def test_all_event_timestamps_match_native_flags_without_turning_them_into_task_success(self):
        from scripts.pi0294.test_pv1_event_exposure import fixture as physical_fixture
        data,card=physical_fixture()
        data['contact_audit'][699:850,3]=.49
        before=deepcopy(data);result=event_index(data,card)
        self.assertEqual(result['event_times_s']['contact_start'],[1.])
        self.assertEqual(result['event_times_s']['contact_end'],[2.])
        self.assertEqual(result['event_times_s']['native_grasp_end_after_10cm_transport'],[1.8])
        np.testing.assert_allclose(result['event_times_s']['lift_2cm_while_native_grasp'],[1.4],atol=1e-12,rtol=0)
        self.assertEqual(result['terminal_41_observation_times_s'],(np.arange(600,641)/20).tolist())
        self.assertNotIn('task_success',result)
        for k in data:self.assertTrue(np.array_equal(data[k],before[k]))

    def test_new_scope_uses_same_actual_committed_packet_but_old_PV1_rejects(self):
        card,samples=new_fixture();out=packet(samples,card)
        self.assertEqual(out['actual_state10'].shape,(641,10))
        self.assertEqual(out['committed_absolute10'].shape,(640,10))
        self.assertEqual(int(out['valid_window'].sum()),601)
        self.assertAlmostEqual(out['committed_absolute10'][0,2]-out['actual_state10'][0,2],.002)
        with self.assertRaises(PermissionError):legacy_packet(samples,card)
        for field,value in [('split','DEV'),('nominal_clearance_m',.012),('instruction','wrong task')]:
            bad=dict(card,**{field:value})
            with self.assertRaises((PermissionError,ValueError)):packet(samples,bad)
        bad=deepcopy(samples);bad[20]['causal_history']['episode_id']='OLD_PV1_PARENT'
        with self.assertRaises(PermissionError):packet(bad,card)

    def test_h8_boundary_and_tail_are_diagnostic_not_an_alternative_gate(self):
        card,samples=new_fixture();before=deepcopy(samples)
        report=horizon_kinematics(samples,card)
        self.assertEqual(report['legal_windows'],601)
        self.assertTrue(report['diagnostic_only_not_a_shorter_safety_gate'])
        for region in report['regions'].values():
            self.assertEqual(len(region['linear_speed_mps']['per_window_max']),601)
            self.assertAlmostEqual(region['linear_speed_mps']['maximum'],.0002)
        self.assertEqual(samples,before)


if __name__=='__main__':unittest.main()
