import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.pi0294.mp1 import anchor_dynamic_config as config
from scripts.pi0294.mp1 import anchor_native_gate as native
from scripts.pi0294.mp1.anchor_dynamic import TraceMismatch
from scripts.pi0294.mp1.failure_config import ATTEMPTS


class NativeHistoryGateTests(unittest.TestCase):
    def test_exact_eight_card_has_no_inference_or_integration_state_injection(self):
        repo=Path(__file__).resolve().parents[2]
        card=json.loads((repo/'reports/PI0294/design/N1_EIGHT_NATIVE_HISTORY_COPY_PARITY_V1.json').read_text())
        spec=config.specification(card)
        self.assertTrue(spec['native_gate_only']);self.assertTrue(config.is_diagnostic(card))
        self.assertEqual(card['caps'],spec['caps']);self.assertEqual(spec['caps']['main_starts'],4)
        self.assertEqual(card['selected_states'],8);self.assertEqual(spec['parents'],4)
        history=sum(x['main_steps'] for x in ATTEMPTS.values());rejected=sum(x['offset'] for x in ATTEMPTS.values())
        self.assertEqual(history,27600);self.assertEqual(history+2*rejected+800,29202)
        self.assertEqual(card['exact_total_shadow_ticks'],29202)
        self.assertEqual(spec['caps']['generation_queries'],0)
        tree=ast.parse(Path(native.__file__).read_text())
        calls={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        attributes={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        self.assertNotIn('restore_and_check',calls);self.assertNotIn('mj_setState',attributes)
        self.assertIn('replay_case',calls);self.assertIn('inspect_prefix',calls)

    def test_rejected_full_rows_preserved_before_strict_comparison(self):
        identifier='PI0294-MP1-S-BC-512-TRAIN4-F1-000-T1'
        record=dict(id=identifier+'-S026',original_attempt_id=identifier,sequence=26,observation_sha256='a'*64)
        check=dict(decision='REJECT',shadow_integrations=63,first_violation_offset_ticks=63,
            predicted_native_safety=['original'],live_state_unchanged=True,actions_modified=False)
        for difference in [0.,2e-10]:
            with self.subTest(difference=difference),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);old=root/'old';(old/'evidence').mkdir(parents=True);output=root/'new';output.mkdir()
                np.savez_compressed(old/'evidence'/(identifier+'-rejected-shadow.npz'),observed_rows=np.zeros((63,3)))
                rows=np.zeros((63,3));rows[0,1]=difference
                with patch.object(native,'load_owned_snapshot',return_value=(dict(observed=np.zeros(3)),{},{})), \
                        patch.object(native,'observed_state',return_value=np.zeros(3)), \
                        patch.object(native,'inspect_prefix',return_value=(check,rows)), \
                        patch.object(native,'read',return_value=[dict(decision='REJECT',predicted_native_safety=['original'])]*27):
                    invoke=lambda:native.qualify_native(SimpleNamespace(),record,[record],old,root,np.zeros((8,4,4)),np.zeros(8),object(),output)
                    if difference:
                        with self.assertRaises(TraceMismatch):invoke()
                    else:self.assertEqual(invoke()['maximum_fullstate_trace_difference'],0.)
                self.assertTrue((output/(record['id']+'.npz')).exists())


if __name__=='__main__':unittest.main()
