"""CPU-only closeout preserves original fallback and rejects altered authority."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import deadline_guard as guard


class AuthorityTests(unittest.TestCase):
    def test_original_and_pinned_closeout(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'evidence').mkdir()
            ledger=dict(CPU_s=1000,unmetered_dependency_setup_CPU_s_upper=600)
            (root/'RESOURCE_ACCOUNTING.json').write_text(json.dumps(ledger))
            with patch.dict(os.environ,{},clear=True):
                self.assertEqual(guard.resource_bounds(root),(guard.DEADLINE,171200))
            authority=dict(run_uuid=root.name,source_sha='57a02b4923e5001df5fd4b3b5958c372b9cff541',
                original_deadline_epoch=guard.DEADLINE,human_authorized=True,
                physics_model_PBS_WBC_deletion_allowed=False,T0_epoch=100,deadline_epoch=3700,
                new_CPU_budget_s=14400,total_output_limit_bytes=40*1024**3,
                CPU_ledger_baseline_s=900,auxiliary_unmetered_CPU_upper_s=60)
            raw=json.dumps(authority).encode()
            path=root/'evidence/DATA_CLOSEOUT_AUTHORIZATION.json'
            path.write_bytes(raw)
            with patch.dict(os.environ,{'PI0291_CLOSEOUT_AUTHORIZATION_SHA256':hashlib.sha256(raw).hexdigest()}):
                self.assertEqual(guard.resource_bounds(root),(3700,14240))
                path.write_bytes(raw+b' ')
                with self.assertRaises(AssertionError):
                    guard.resource_bounds(root)


if __name__=='__main__':
    unittest.main()
