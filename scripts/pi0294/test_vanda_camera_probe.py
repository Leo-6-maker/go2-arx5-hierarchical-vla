import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.operations import vanda_camera_probe as probe


class VandaCameraProbeTests(unittest.TestCase):
    def test_native_bundle_cannot_expand_active_CPU_work(self):
        with patch.object(probe.os, 'sched_getaffinity', return_value=set(range(36))), \
                patch.object(probe.os, 'sched_setaffinity') as affinity, \
                patch.object(probe.resource, 'setrlimit') as limit:
            probe.constrain_cpu()
            affinity.assert_called_once_with(0, {0, 1, 2, 3})
            limit.assert_called_once_with(probe.resource.RLIMIT_CPU, (3600, 3600))

    def fixture(self, root):
        def put(name, value):
            p = root / name
            p.write_text(json.dumps(value))
            return str(p)
        binding = dict(run_uuid='fixture', run_root=str(root), role='CAMERA',
                       registry=put('registry.json', dict(entries=[dict(role='CAMERA', path=str(root))])),
                       lease=put('lease.json', dict(status='ACTIVE', run_uuid='fixture')))
        binding_path = put('OUTPUT_BINDING.private.json', binding)
        capsule = root / 'probe.py'
        capsule.write_text('fixture')
        m = dict(output_root=str(root), registered_runs_root=str(root.parent), run_uuid='fixture',
                 owner_uid=os.getuid(), owner='EXISTING_LOCAL_PC', scope=probe.SCOPE,
                 authority=dict(protocol_ref=probe.AUTHORITY, parent_run='parent'),
                 execution=dict(profile='vanda-camera-qualification', output_binding_sha256=probe.sha(binding_path)),
                 deadline_epoch=1000, limits=dict(ngpus=1, active_cpus=4, walltime_s=900, RGB_images=4,
                     physics_steps=0, WBC_launches=0, model_queries=0, optimizer_updates=0),
                 parent_adoption=put('parent.json', dict(run_uuid='parent', authority_release=probe.AUTHORITY,
                                                       new_experiment_cutoff_epoch=1000)),
                 sealed_files={str(capsule): probe.sha(capsule)}, project='fixture')
        path = put('RUN_MANIFEST.json', m)
        return path, m

    def test_bound_manifest_and_changed_input_rejected(self):
        with tempfile.TemporaryDirectory() as d, patch.object(probe.time, 'time', return_value=10):
            root = Path(d)
            path, m = self.fixture(root)
            self.assertEqual(probe.validate(path), m)
            (root / 'probe.py').write_text('changed')
            with self.assertRaisesRegex(AssertionError, 'SEALED_INPUT'):
                probe.validate(path)

    def test_physics_scope_and_closed_lease_rejected(self):
        with tempfile.TemporaryDirectory() as d, patch.object(probe.time, 'time', return_value=10):
            root = Path(d)
            path, m = self.fixture(root)
            m['limits']['physics_steps'] = 1
            Path(path).write_text(json.dumps(m))
            with self.assertRaisesRegex(AssertionError, 'LIMITS'):
                probe.validate(path)
            m['limits']['physics_steps'] = 0
            Path(path).write_text(json.dumps(m))
            (root / 'lease.json').write_text(json.dumps(dict(run_uuid='fixture', status='CLOSED')))
            with self.assertRaisesRegex(AssertionError, 'LEASE'):
                probe.validate(path)

    def test_container_requires_fresh_matching_allocation(self):
        with tempfile.TemporaryDirectory() as d, patch.object(probe.socket, 'gethostname', return_value='GN-A40-1'), \
                patch.dict(os.environ, PBS_JOBID='123.site'), patch.object(probe.time, 'time', return_value=150):
            root = Path(d)
            (root / 'SUBMISSION.private.json').write_text(json.dumps(dict(job_id='123.site')))
            admission = dict(job_id='123.site', project='project', run_uuid='run', epoch=100)
            p = root / 'NATIVE_ALLOCATION_ADMISSION.private.json'
            p.write_text(json.dumps(admission))
            m = dict(output_root=d, project='project', run_uuid='run')
            probe.require_render_allocation(m)
            admission['epoch'] = 0
            p.write_text(json.dumps(admission))
            with self.assertRaisesRegex(AssertionError, 'FRESH_NATIVE_ADMISSION'):
                probe.require_render_allocation(m)

    def n1_fixture(self, directory):
        parent = Path(directory) / '20261016T191445Z-localpc-mp1'
        root = parent / 'children/vanda-egl-cameras-n1-03'
        root.mkdir(parents=True)
        path, m = self.fixture(root)
        adoption = dict(run_uuid=parent.name, authority_release=probe.AUTHORITY,
                        T0_epoch=1792178085, new_experiment_cutoff_epoch=2000)
        parent_path = parent / 'MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json'
        parent_path.write_text(json.dumps(adoption));m['parent_adoption'] = str(parent_path)
        m['authority']['parent_run'] = parent.name
        first = parent / 'authority/N1/ADOPTION_DELTA.json'
        first.parent.mkdir(parents=True)
        first.write_text(json.dumps(dict(authority_release=probe.N1_RELEASE,
            parent_authority_release=probe.AUTHORITY, reset_on_resume=False, new_competing_owner=False,
            autonomy_started_epoch=1, new_submission_and_candidate_cutoff_epoch=900,
            new_experiment_cutoff_epoch=1000)))
        first.chmod(0o400);delta_sha = probe.sha(first)
        card = root / 'N1_QUALIFICATION_CARD.json'
        source = Path(probe.__file__).resolve().parents[2] / 'reports/PI0294/design/N1_VANDA_CAMERA_ONLY_NEW_WINDOW_V1.json'
        card.write_bytes(source.read_bytes());card.chmod(0o400)
        m['sealed_files'].update({str(first): delta_sha, str(card): probe.sha(card)})
        m['execution']['n1_camera_window'] = dict(release=probe.N1_RELEASE,
            adoption_delta_sha256=delta_sha, created_epoch=10, deadline_epoch=1000,
            maximum_queue_and_execution_window_s=43200, original_N1_T0_epoch=1)
        lease = json.loads((root / 'lease.json').read_text());lease['deadline_epoch'] = 1000
        (root / 'lease.json').write_text(json.dumps(lease))
        old = parent / 'children/vanda-egl-cameras-02';old.mkdir()
        old_lease = parent / 'old-camera-lease.json';old_lease.write_text(json.dumps(dict(status='CLOSED')))
        (old / 'OUTPUT_BINDING.private.json').write_text(json.dumps(dict(lease=str(old_lease))))
        Path(path).write_text(json.dumps(m))
        return path, m, first, delta_sha, old_lease

    def test_n1_new_camera_window_uses_original_delta_and_closed_old_lease(self):
        with tempfile.TemporaryDirectory() as d, patch.object(probe.time, 'time', return_value=11):
            path, m, first, digest, old = self.n1_fixture(d)
            with patch.object(probe, 'N1_DELTA_SHA', digest):
                self.assertEqual(probe.validate(path), m)
                old.write_text(json.dumps(dict(status='ACTIVE')))
                with self.assertRaisesRegex(AssertionError, 'OLD_CAMERA_LEASE'):
                    probe.validate(path)

    def test_n1_window_cannot_exceed_cutoff_extend_lease_or_omit_authority(self):
        with tempfile.TemporaryDirectory() as d, patch.object(probe.time, 'time', return_value=11):
            path, m, first, digest, old = self.n1_fixture(d)
            with patch.object(probe, 'N1_DELTA_SHA', digest):
                for key, value in [('deadline_epoch', 1001), ('maximum_queue_and_execution_window_s', 43201),
                                   ('original_N1_T0_epoch', 2)]:
                    changed = deepcopy(m);changed['execution']['n1_camera_window'][key] = value
                    Path(path).write_text(json.dumps(changed))
                    with self.subTest(field=key), self.assertRaises(AssertionError):
                        probe.validate(path)
                changed = deepcopy(m);changed['execution'].pop('n1_camera_window')
                Path(path).write_text(json.dumps(changed))
                with self.assertRaisesRegex(AssertionError, 'WINDOW_REQUIRED'):
                    probe.validate(path)

    def test_n1_changed_first_adoption_is_not_a_fresh_window(self):
        with tempfile.TemporaryDirectory() as d, patch.object(probe.time, 'time', return_value=11):
            path, m, first, digest, old = self.n1_fixture(d)
            first.chmod(0o600);first.write_text('{}');first.chmod(0o400)
            with patch.object(probe, 'N1_DELTA_SHA', digest), self.assertRaisesRegex(AssertionError, 'FIRST_ADOPTION_HASH'):
                probe.validate(path)

    def test_container_mounts_existing_inputs_readonly_and_only_child_output_writable(self):
        with tempfile.TemporaryDirectory() as d:
            path, m, first, digest, old = self.n1_fixture(d)
            root = Path(m['output_root']);m.update(assets=str(root.parent.parent / 'frozen-assets'),
                                                 image=str(root.parent.parent / 'frozen-image.sif'))
            mounts = probe.container_binds(m)
            self.assertEqual(mounts[-1], str(root) + ':' + str(root) + ':rw')
            self.assertTrue(all(x.endswith(':ro') for x in mounts[:-1]))
            self.assertIn(str(old) + ':' + str(old) + ':ro', mounts)
            self.assertIn(m['assets'] + ':' + m['assets'] + ':ro', mounts)

    def test_render_path_has_no_integrator_model_or_WBC_launcher(self):
        import inspect
        calls = [n for n in ast.walk(ast.parse(inspect.getsource(probe.render))) if isinstance(n, ast.Call)]
        names = [ast.unparse(n.func) for n in calls]
        self.assertFalse(any(x in names for x in ['mujoco.mj_step', 'mujoco.mj_step1', 'mujoco.mj_step2',
                                                   'subprocess.run', 'subprocess.Popen']))
        self.assertNotIn('torch', inspect.getsource(probe.render))


if __name__ == '__main__':
    unittest.main()
