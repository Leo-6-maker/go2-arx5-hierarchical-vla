"""Fail-closed observer and finite same-state mechanism card regressions."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.pi0294.mp1 import anchor_dynamic_config as config
from scripts.pi0294.mp1.failure_replay import observe_prefix
from scripts.pi0294.mp1.anchor_copy_ablation import ablate,compare_branch


class NativeCopyAblationTests(unittest.TestCase):
    def test_card_is_single_state_zero_neural_finite_shadow_engineering(self):
        card=json.loads((Path(__file__).resolve().parents[2]/'reports/PI0294/design/N1_ANCHOR_NATIVE_COPY_ABLATION_V1.json').read_text())
        spec=config.specification(card)
        self.assertTrue(config.is_diagnostic(card));self.assertTrue(spec['ablation_only'])
        self.assertEqual(card['caps'],spec['caps']);self.assertEqual(card['CPU_core_s'],spec['CPU_core_s'])
        self.assertEqual(spec['caps']['main_starts'],2);self.assertEqual(spec['caps']['shadow_integrations'],5600)
        self.assertEqual(spec['caps']['generation_queries'],0);self.assertEqual(spec['wall_s'],1800)
        self.assertFalse(card['single_state_pass_can_qualify_all_eight'])
        self.assertEqual(card['comparison_absolute_tolerance'],1e-10)

    def test_default_observer_keeps_original_replay_without_new_calls(self):
        with patch('scripts.pi0294.mp1.failure_replay.fingerprint') as fingerprint:
            observe_prefix(object(),26,np.zeros((8,4,4)),np.zeros(8),None)
        fingerprint.assert_not_called()

    def test_observer_detects_source_and_action_mutations(self):
        for mutation in ['state','action']:
            poses=np.zeros((8,4,4));opening=np.zeros(8)
            def observer(engine,sequence,p,g):
                self.assertEqual(sequence,26)
                if mutation=='action':g[0]=1
            with patch('scripts.pi0294.mp1.failure_replay.fingerprint',side_effect=['a','b' if mutation=='state' else 'a']):
                with self.subTest(mutation=mutation),self.assertRaises(RuntimeError):
                    observe_prefix(object(),26,poses,opening,observer)

    def test_observer_without_mutation_is_called_once_at_exact_sequence(self):
        calls=[]
        with patch('scripts.pi0294.mp1.failure_replay.fingerprint',return_value='same'):
            observe_prefix(object(),26,np.zeros((8,4,4)),np.zeros(8),lambda *args:calls.append(args))
        self.assertEqual(len(calls),1);self.assertEqual(calls[0][1],26)

    def test_native_branch_precedes_restore_and_each_copy_is_closed(self):
        events=[];copies=[]
        def candidate(label):
            obj=SimpleNamespace(close=lambda:events.append('close-'+label));copies.append(obj);return obj
        def inspect(engine,record,p,g,ledger,reference,output,label):
            events.append(label);return dict(branch=label,fullstate_parity=label=='FULL_NATIVE_COPY')
        with patch('scripts.pi0294.mp1.anchor_copy_ablation.load_owned_snapshot',return_value=({}, {}, {})), \
                patch('scripts.pi0294.mp1.anchor_copy_ablation.clone',side_effect=lambda e:candidate('copy')), \
                patch('scripts.pi0294.mp1.anchor_copy_ablation.restore_and_check',side_effect=lambda *a:events.append('restore')), \
                patch('scripts.pi0294.mp1.anchor_copy_ablation.compare_branch',side_effect=inspect), \
                patch('scripts.pi0294.mp1.anchor_copy_ablation.write'):
            charges=[];ledger=SimpleNamespace(charge=lambda *args:charges.append(args))
            rows=ablate(object(),dict(card={}),object(),object(),np.zeros((8,4,4)),np.zeros(8),ledger,object(),Path('/not-written'),make_engine=lambda *a:candidate('fresh'))
        self.assertEqual(charges,[('main_starts',1)]);self.assertEqual(len(rows),3)
        self.assertEqual(events[0],'FULL_NATIVE_COPY');self.assertEqual(events.count('restore'),2)
        self.assertEqual(sum(x.startswith('close-') for x in events),3)


if __name__=='__main__':unittest.main()
