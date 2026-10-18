from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.pi0294.run_admission import validate
from scripts.pi0294.pv1.runtime import validate_runtime


class RuntimeRoutingContracts(unittest.TestCase):
    def test_same_relay_uses_new_typed_guard_not_old_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'PV1_STAGE_BINDING.json').write_text('{}')
            with patch('scripts.pi0294.pv1.runtime.validate_runtime',return_value={'fixture_only':True}) as typed:
                self.assertEqual(validate(root,profile='hopper-neural'),{'fixture_only':True})
                typed.assert_called_once_with(root,'hopper-neural',closeout=False)
                for extra in (dict(hostname='FORGED'),dict(uid=123),dict(now=0),dict(verify_source=False)):
                    with self.assertRaises(PermissionError):validate(root,profile='hopper-neural',**extra)
                (root/'MP1_STAGE_BINDING.json').write_text('{}')
                with self.assertRaises(PermissionError):validate(root,profile='hopper-neural')

    def test_only_new_scope_entry_profiles_and_native_clock(self):
        for profile,kwargs in [('cpu-data',{}),('hopper-neural',dict(now=0))]:
            with self.assertRaises(PermissionError):validate_runtime('/not-an-entry',profile,**kwargs)
