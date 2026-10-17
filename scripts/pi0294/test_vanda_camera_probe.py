import ast
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

    def test_render_path_has_no_integrator_model_or_WBC_launcher(self):
        import inspect
        calls = [n for n in ast.walk(ast.parse(inspect.getsource(probe.render))) if isinstance(n, ast.Call)]
        names = [ast.unparse(n.func) for n in calls]
        self.assertFalse(any(x in names for x in ['mujoco.mj_step', 'mujoco.mj_step1', 'mujoco.mj_step2',
                                                   'subprocess.run', 'subprocess.Popen']))
        self.assertNotIn('torch', inspect.getsource(probe.render))


if __name__ == '__main__':
    unittest.main()
