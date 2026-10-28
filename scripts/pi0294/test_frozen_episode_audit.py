import json
from pathlib import Path
import tempfile
import unittest
from .mp1.audit import require_frozen_attempt


class FrozenEpisodeAuditTests(unittest.TestCase):
    def test_other_live_reservation_does_not_unfreeze_selected_terminal_case(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'episodes/terminal';folder.mkdir(parents=True);(root/'worker_processes').mkdir()
            (folder/'MP1_ADJUDICATION.json').write_text(json.dumps(dict(attempt_id='terminal',status='ENGINEERING_INCOMPLETE')))
            (root/'RESOURCE_LEDGER.json').write_text(json.dumps(dict(reservations={'other':dict(maximum=16000)})))
            (root/'worker_processes/terminal.json').write_text(json.dumps(dict(pid=2147483647,proc_start='unmatched-fixture')))
            self.assertEqual(require_frozen_attempt(root,'terminal')['attempt_id'],'terminal')
            (root/'RESOURCE_LEDGER.json').write_text(json.dumps(dict(reservations={'terminal':dict(maximum=16000)})))
            with self.assertRaises(PermissionError):require_frozen_attempt(root,'terminal')
            with self.assertRaises(PermissionError):require_frozen_attempt(root,'../terminal')

    def test_actual_live_worker_identity_rejects_audit(self):
        import os
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'episodes/live';folder.mkdir(parents=True);(root/'worker_processes').mkdir()
            (folder/'MP1_ADJUDICATION.json').write_text(json.dumps(dict(attempt_id='live')))
            (root/'RESOURCE_LEDGER.json').write_text(json.dumps(dict(reservations={})))
            (root/'worker_processes/live.json').write_text(json.dumps(dict(pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().split()[21])))
            with self.assertRaises(PermissionError):require_frozen_attempt(root,'live')
