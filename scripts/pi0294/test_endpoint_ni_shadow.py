import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.pi0294.mp1 import anchor_dynamic_config as config,endpoint_ni_shadow as s
from scripts.pi0294.native_dataset import sha


class EndpointShadowTests(unittest.TestCase):
    def test_separate_bounded_card_preserves_old_gate(self):
        repo=Path(__file__).resolve().parents[2]
        card=json.loads((repo/'reports/PI0294/design/N1_ENDPOINT_NI16_MATCHED_SHADOWS_V1.json').read_text())
        spec=config.specification(card);self.assertTrue(config.is_diagnostic(card));self.assertTrue(spec['endpoint_only'])
        self.assertEqual(spec['caps'],card['caps']);self.assertEqual(spec['caps']['generation_queries'],0)
        self.assertEqual(card['maximum_total_actual_shadow_ticks'],29202+16*200)
        self.assertNotEqual(spec['child'],config.NATIVE_GATE_CHILD)
        self.assertEqual(config.specification(dict(schema=config.NATIVE_GATE_SCHEMA))['caps']['shadow_integrations'],29600)
        tree=ast.parse(Path(s.__file__).read_text());calls={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        for name in ['qualify_native','replay_case','inspect_prefix','gate_action']:self.assertIn(name,calls)
        for name in ['restore_and_check','load','fixed_draw']:self.assertNotIn(name,calls)

    def test_numerical_reject_zero_physics_accept_and_warn_use_original_prefix(self):
        identifier=next(iter(s.delivery.ni.selected_ids()));record=dict(id=identifier,task='T1',original_attempt_id='parent')
        model='S-BC-2048'
        for decision in ['REJECT','ACCEPT','WARN']:
            with self.subTest(decision=decision),tempfile.TemporaryDirectory() as folder:
                folder=Path(folder);output=folder/'output';output.mkdir();path=folder/s.delivery.filename(model,identifier)
                action=np.zeros((40,10));action[:,3]=1;action[:,7]=1
                np.savez(path,physical_H40=action)
                row=dict(id=identifier,model=model,arrays_sha256=sha(path),original_query_uuid='q',noise={'sha256':'n'})
                check=dict(live_state_unchanged=True,actions_modified=False,decision='PASS',shadow_integrations=200)
                with patch.object(s,'load_owned_snapshot',return_value=({}, {},dict(payload=dict(state_eef10=np.zeros(10)),tick=100))), \
                        patch.object(s,'gate_action',return_value=dict(decision=decision)), \
                        patch.object(s,'inspect_prefix',return_value=(check,np.zeros((200,3)))) as inspect:
                    value=s.endpoint_prefix(object(),record,folder,folder,row,object(),output)
                    self.assertFalse(value['task_success_claimed']);self.assertEqual(value['new_live_integrations'],0)
                    self.assertEqual(inspect.call_count,0 if decision=='REJECT' else 1)
                    self.assertEqual(value['actual_shadow_ticks'],0 if decision=='REJECT' else 200)
                    if decision!='REJECT':self.assertEqual(inspect.call_args.args[1].shape,(8,4,4))


if __name__=='__main__':unittest.main()
