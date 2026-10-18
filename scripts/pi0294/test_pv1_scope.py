"""Admission rejects invalid identity before any native import or side effect."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.pi0294.pv1.scope import validate


class PV1ScopeTests(unittest.TestCase):
    def test_capsule_and_seal_escape_rejected_before_identity(self):
        for code,seal in [('/outside','seal.json'),('../outside','seal.json'),('capsule','../seal.json')]:
            with tempfile.TemporaryDirectory() as tmp:
                m=dict(code=dict(capsule_relative=code,source_seal_relative=seal))
                def read(p):return m if Path(p).name=='RUN_MANIFEST.json' else {}
                with patch('scripts.pi0294.pv1.scope.read',side_effect=read):
                    with self.assertRaisesRegex(PermissionError,'CAPSULE_MUST_STAY'):
                        validate(tmp)

    def test_wrong_owner_is_not_a_cpu_or_model_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            m=dict(schema='LEGACY_N1_NOT_PV1',code=dict(capsule_relative='capsule',source_seal_relative='seal.json'))
            def read(p):return m if Path(p).name=='RUN_MANIFEST.json' else {}
            with patch('scripts.pi0294.pv1.scope.read',side_effect=read):
                with self.assertRaisesRegex(PermissionError,'NATIVE_CPU_CHILD_IDENTITY'):
                    validate(tmp)
