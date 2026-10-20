"""Pure CPU metadata fixtures, not static or dynamic qualification."""
from copy import deepcopy
import unittest

import numpy as np

from scripts.pi0294.data_cov1.plan import candidate,slots,stage_groups,capture_cards,protocol_card,validate_group
from scripts.pi0294.data_cov1.gates import first_four,stage_gate,next_stage
from scripts.pi0294.pv1.capture import capture_cards as legacy_cards


def groups():return [candidate(**s,index=0) for s in slots()]


def admitted(cards):
    return [dict(id=c['id'],scene_group=c['group_id'],task=c['task'],status='ADMITTED',
        task_success=True,admitted=True,raw_admitted=True,codec_admitted=True,
        native_faults=0,unknown_effects=[]) for c in cards]


class DataCov1Plan(unittest.TestCase):
    def test_crossed_finite_population_distance_and_legacy_namespace_rejection(self):
        own=groups();self.assertEqual(len(own),144)
        for stage,count in [('P24',24),('P72',72),('P144',144)]:
            selected=stage_groups(own,stage);self.assertEqual(len(selected),count)
            self.assertEqual(sum(g['direction']=='-Y' for g in selected),count//2)
        g=own[0];self.assertEqual(g,candidate(**slots()[0],index=0))
        self.assertAlmostEqual(np.linalg.norm(np.array(g['goal_xy'])-g['source_xy']),g['euclidean_distance_m'])
        with self.assertRaises(PermissionError):candidate(**slots()[0],index=32)
        with self.assertRaises(PermissionError):legacy_cards(own,'COVERAGE_FIRST')
        bad=deepcopy(g);bad['visual']['wrist_camera_translation'][0]=.001
        with self.assertRaises(PermissionError):validate_group(bad)

    def test_first_four_are_the_original_reverse_gap_groups_and_included_in_48(self):
        cards=capture_cards(groups(),'P24');self.assertEqual(len(cards),48)
        self.assertEqual([(c['appearance_profile'],c['direction'],c['distance_bin'],c['task']) for c in cards[:4]],
            [('AP0','-Y','NEAR','T1'),('AP0','-Y','NEAR','T2'),('AP2','-Y','FAR','T1'),('AP2','-Y','FAR','T2')])
        rows=admitted(cards[:4]);self.assertTrue(first_four(groups(),rows)['count_gate_pass'])
        rows[0].update(status='CODEC_REJECT',admitted=False,codec_admitted=False)
        self.assertTrue(first_four(groups(),rows)['count_gate_pass'])
        rows[1].update(status='CODEC_REJECT',admitted=False,codec_admitted=False)
        self.assertFalse(first_four(groups(),rows)['count_gate_pass'])

    def test_cumulative_gate_never_hides_stratum_failures_or_repeats(self):
        own=groups();cards=capture_cards(own,'P24');rows=admitted(cards)
        gate=stage_gate(own,'P24',rows);self.assertTrue(gate['count_gate_pass'])
        self.assertEqual(next_stage('P24',gate,'PILOT_ONLY',fresh_resources_pass=True,pv1_priority_clear=True)['reason'],'APPROVED_SCOPE_TERMINAL')
        self.assertEqual(next_stage('P24',gate,'FULL_CONDITIONAL_CAMPAIGN',fresh_resources_pass=True,pv1_priority_clear=True)['next_stage'],'P72')
        self.assertEqual(next_stage('P24',gate,'FULL_CONDITIONAL_CAMPAIGN',fresh_resources_pass=True,pv1_priority_clear=False)['reason'],'PV1_PRIORITY')
        lost={c['id'] for c in cards if c['appearance_profile']=='AP0' and c['direction']=='-Y' and c['task']=='T2'}
        for r in rows:
            if r['id'] in lost:r.update(status='CODEC_REJECT',admitted=False,codec_admitted=False)
        gate=stage_gate(own,'P24',rows)
        self.assertTrue(gate['checks']['full_admission_each'])
        self.assertFalse(gate['checks']['priority_reverse_appearance'])
        self.assertFalse(gate['count_gate_pass'])
        with self.assertRaises(PermissionError):stage_gate(own,'P24',rows+rows[:1])

    def test_unknown_missing_relabelled_or_non_admitted_raw_never_pass(self):
        own=groups();cards=capture_cards(own,'P24');rows=admitted(cards)
        self.assertFalse(stage_gate(own,'P24',rows[:-1])['count_gate_pass'])
        rows[0]['unknown_effects']=['unsettled'];self.assertFalse(stage_gate(own,'P24',rows)['count_gate_pass'])
        rows=admitted(cards);rows[0]['raw_admitted']=False
        with self.assertRaises(PermissionError):stage_gate(own,'P24',rows)
        rows=admitted(cards);rows[0]['native_faults']=1
        with self.assertRaises(PermissionError):stage_gate(own,'P24',rows)
        self.assertEqual(protocol_card()['stages'][-1]['proposed_cumulative_limits']['all_integrations'],9216000)


if __name__=='__main__':unittest.main()
