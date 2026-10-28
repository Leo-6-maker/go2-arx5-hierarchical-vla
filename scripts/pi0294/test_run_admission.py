"""Admission rejects changed authority before any workload side effect."""
import json
from pathlib import Path
import tempfile
import unittest

from .run_admission import AUTHORITY, CUTOFF, DEADLINE, GO_COMMENT, RUN, T0, contained, sha, validate


class AdmissionTests(unittest.TestCase):
    def test_registered_source_and_host(self):
        with tempfile.TemporaryDirectory() as d:
            parent = Path(d)
            root = parent / RUN
            (root / 'code').mkdir(parents=True)
            def write(name, value):
                path = root / name
                path.write_text(json.dumps(value))
                return path
            source = root / 'code' / 'example.py'
            source.write_text('VALUE = 1\n')
            seal = write('SOURCE_SEAL.json', dict(source_sha='a' * 40, files={'example.py': sha(source)}))
            registry = write('REGISTRY.json', dict(entries=[dict(role='RUN', path=str(root), executor_node='node')]))
            lease = write('LEASE.json', dict(run_uuid=RUN, owner='EXISTING_LOCAL_PC', status='ACTIVE',
                         protocol_ref=AUTHORITY, deadline_epoch=DEADLINE))
            binding = write('OUTPUT_BINDING.private.json', dict(run_root=str(root), role='RUN',
                registered_parent=str(parent), registry=str(registry), lease=str(lease)))
            value = dict(schema='RUN_MANIFEST_V1', run_uuid=RUN, pi='PI0294', owner='EXISTING_LOCAL_PC',
                authority=dict(protocol_ref=AUTHORITY, go_comment=GO_COMMENT),
                code=dict(vla_commit='a' * 40, source_seal_sha256=sha(seal)),
                execution=dict(profile='4080-simulation', uid=123, hostnames=['node'], private_path_role='RUN',
                               output_binding_sha256=sha(binding)),
                limits=dict(T0_epoch=T0, new_work_cutoff_epoch=CUTOFF, deadline_epoch=DEADLINE,
                            reset_on_resume=False))
            write('RUN_MANIFEST.json', value)
            kwargs = dict(profile='4080-simulation', now=T0 + 1, hostname='node', uid=123)
            self.assertEqual(validate(root, **kwargs), value)
            with self.assertRaisesRegex(PermissionError, 'HOST_PROFILE'):
                validate(root, **dict(kwargs, hostname='other'))
            capsule = root / 'capsules' / ('a' * 40)
            (capsule / 'code').mkdir(parents=True)
            copied = capsule / 'code' / 'example.py'
            copied.write_bytes(source.read_bytes())
            relocated = capsule / 'SOURCE_SEAL.json'
            relocated.write_bytes(seal.read_bytes())
            value['code'].update(capsule_relative=str((capsule / 'code').relative_to(root)),
                                 source_seal_relative=str(relocated.relative_to(root)))
            write('RUN_MANIFEST.json', value)
            self.assertEqual(validate(root, **kwargs), value)
            source.write_text('VALUE = 2\n')
            self.assertEqual(validate(root, **kwargs), value)
            copied.write_text('VALUE = 2\n')
            with self.assertRaisesRegex(PermissionError, 'SOURCE_MEMBER'):
                validate(root, **kwargs)

    def test_escape(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(PermissionError):
                contained(Path(d).parent / 'other', d)

    def test_clock_and_authority_fail_before_binding(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / RUN
            root.mkdir()
            value = dict(schema='RUN_MANIFEST_V1', run_uuid=RUN, pi='PI0294', owner='EXISTING_LOCAL_PC',
                authority=dict(protocol_ref=AUTHORITY, go_comment=GO_COMMENT),
                limits=dict(T0_epoch=T0, new_work_cutoff_epoch=CUTOFF, deadline_epoch=DEADLINE,
                            reset_on_resume=False))
            (root / 'RUN_MANIFEST.json').write_text(json.dumps(value))
            for now in (T0 - 1, CUTOFF, DEADLINE):
                with self.subTest(now=now), self.assertRaisesRegex(PermissionError, 'TIME_WINDOW'):
                    validate(root, profile='4080-simulation', now=now)
            value['limits']['reset_on_resume'] = True
            (root / 'RUN_MANIFEST.json').write_text(json.dumps(value))
            with self.assertRaisesRegex(PermissionError, 'AUTHORITY'):
                validate(root, profile='4080-simulation', now=T0 + 1)
