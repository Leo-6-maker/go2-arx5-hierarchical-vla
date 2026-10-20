"""Synthetic saved data, not physical task or official processor evidence."""
from copy import deepcopy
import unittest

import numpy as np

from scripts.pi0294.data_cov1.plan import candidate,slots,capture_cards
from scripts.pi0294.data_cov1.export import packet,horizon_kinematics
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
