"""Small synthetic software fixtures only; no native qualification or network."""

import copy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from go2arx5_vla.world_model.contact_timing_full_history import (
    ARMS, BLIND_SOURCE, FEATURE_RECIPE, HISTORY_DIMENSION, HORIZONS,
    ContactTimingFullHistory, contact_replay, features, full_history_features,
)
from go2arx5_vla.world_model.storage import digest


def small_data():
    rng = np.random.default_rng(811)
    history = rng.normal(size=(18, HISTORY_DIMENSION)).astype(np.float32)
    rows = history.reshape(-1, 9, 103)
    rows[:, :, 94] = np.abs(rows[:, :, 94])
    rows[:, :, 99:] = rng.integers(0, 2, size=(18, 9, 4))
    rows[:, -1, 97] = .25  # Deliberately differs from the actual current reference.
    current = np.full(18, .8, dtype=np.float32)
    return dict(history=history, plan=rng.uniform(.4, .95, size=(18, 64)).astype(np.float32),
        blind_plan=np.repeat(current[:, None], 64, axis=1), anchor_executed_gripper=current,
        events=rng.integers(0, 2, size=(18, 6, 4)).astype(np.float32),
        valid=np.ones((18, 6), dtype=bool), split=np.repeat('TRAIN', 18), horizons=HORIZONS)


class FullHistoryFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = small_data()
        cls.models = {arm: ContactTimingFullHistory(arm).fit(cls.data) for arm in ARMS}

    def test_all_original_927_observed_values_are_preserved_in_order(self):
        for arm in ARMS:
            for h in HORIZONS:
                x = features(self.data, h, arm)
                self.assertEqual(x.shape, (18, 927 + h))
                np.testing.assert_array_equal(x[:, :927], self.data['history'])
                future = self.data['plan' if arm == ARMS[0] else 'blind_plan']
                np.testing.assert_array_equal(x[:, 927:], future[:, :h])
        self.assertNotEqual(features(self.data, 1, ARMS[1])[0, -1], .25)

    def test_only_within_horizon_candidate_prefix_affects_ac_head(self):
        before = self.models[ARMS[0]].predict(self.data)
        for j, h in enumerate(HORIZONS):
            altered = copy.deepcopy(self.data)
            altered['plan'][:, h:] += 3.
            np.testing.assert_array_equal(before[:, j], self.models[ARMS[0]].predict(altered)[:, j])
            altered['plan'][:, h:] = np.nan
            np.testing.assert_array_equal(features(self.data, h, ARMS[0]), features(altered, h, ARMS[0]))
            altered['plan'][:, :h] += .1
            self.assertFalse(np.array_equal(features(self.data, h, ARMS[0]), features(altered, h, ARMS[0])))

    def test_blind_ignores_candidate_including_base_and_uses_executed_not_internal_opening(self):
        altered = copy.deepcopy(self.data)
        altered['plan'][:] = np.nan
        np.testing.assert_array_equal(self.models[ARMS[1]].predict(self.data),
                                      self.models[ARMS[1]].predict(altered))
        for h in HORIZONS:
            np.testing.assert_array_equal(features(altered, h, ARMS[1])[:, 927:],
                np.broadcast_to(self.data['anchor_executed_gripper'][:, None], (18, h)))

    def test_nonconstant_or_wrong_anchor_blind_plan_is_rejected(self):
        for mode in ['nonconstant', 'wrong_anchor', 'nonfinite_anchor']:
            altered = copy.deepcopy(self.data)
            if mode == 'nonconstant':
                altered['blind_plan'][0, -1] += .1
            elif mode == 'wrong_anchor':
                altered['blind_plan'][:] = .25
            else:
                altered['anchor_executed_gripper'][0] = np.nan
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'RAW_ANCHOR_BOUND'):
                self.models[ARMS[1]].predict(altered)

    def test_fit_uses_own_valid_train_only_and_invalid_labels_are_not_fit(self):
        data = small_data()
        data['valid'][-3:, 0] = False
        data['history'][-3:, :90] += 10000
        a = ContactTimingFullHistory(ARMS[0]).fit(data)
        b = ContactTimingFullHistory(ARMS[1]).fit(data)
        np.testing.assert_array_equal(a.heads[1]['mean'], features(data, 1, ARMS[0], np.arange(15)).mean(0))
        np.testing.assert_array_equal(b.heads[1]['mean'], features(data, 1, ARMS[1], np.arange(15)).mean(0))
        self.assertNotEqual(a.heads[1]['mean'][-1], b.heads[1]['mean'][-1])
        altered = copy.deepcopy(data)
        altered['events'][-3:, 0] = 1 - altered['events'][-3:, 0]
        clone = ContactTimingFullHistory(ARMS[0]).fit(altered)
        for name in a.heads[1]:
            np.testing.assert_array_equal(a.heads[1][name], clone.heads[1][name])
        data['split'][0] = 'DEV'
        with self.assertRaisesRegex(ValueError, 'TRAIN_ONLY'):
            ContactTimingFullHistory(ARMS[0]).fit(data)

    def test_exact_arrays_predictions_reload_and_old_recipe_not_accepted(self):
        from go2arx5_vla.world_model.contact_timing_baseline import ContactTimingFIR
        with tempfile.TemporaryDirectory() as tmp:
            for arm, model in self.models.items():
                path = Path(tmp) / (arm + '.npz')
                model.save(path)
                restored = ContactTimingFullHistory.load(path)
                for name, value in model.arrays().items():
                    np.testing.assert_array_equal(value, restored.arrays()[name])
                np.testing.assert_array_equal(model.predict(self.data, return_unclipped=True),
                                              restored.predict(self.data, return_unclipped=True))
                with self.assertRaises(ValueError):
                    ContactTimingFIR.load(path)
                with self.assertRaises(FileExistsError):
                    model.save(path)
            arrays = self.models[ARMS[0]].arrays()
            arrays['scale_1'][0] = 0
            bad = Path(tmp) / 'corrupt.npz'
            np.savez_compressed(bad, **arrays)
            with self.assertRaisesRegex(ValueError, 'VALID_TRAIN'):
                ContactTimingFullHistory.load(bad)


class FullHistoryCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        # Reuse the established tiny native-ABI fixture, explicitly synthetic.
        old = runpy.run_path(str(Path(__file__).with_name('test_contact_timing_baseline.py')))
        cls.episode = staticmethod(old['episode'])
        spec = old['ContactTimingCLI']().fixture(cls.root)
        from scripts.wm02 import fit_contact_timing_baseline as prior
        original = cls.root / 'prior-FIR90'
        original.mkdir()
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''):
            prior.run(spec, original)
        from scripts.wm02 import fit_contact_timing_full_history as cli
        spec.update(profile=cli.PROFILE, kind=cli.KIND, recipe=cli.RECIPE,
            execution_site='CORE_FUSION_CONTACT_FULL_HISTORY_CPU',
            feature_recipe=FEATURE_RECIPE, observed_history_features=927,
            blind_source=BLIND_SOURCE, deadline_epoch=time.time() + 300,
            frozen_fir=dict(freeze=dict(path=str(original / 'TRAIN_FROZEN.private.json'),
                sha256=digest(original / 'TRAIN_FROZEN.private.json')),
                models=[dict(arm=arm, path=str(original / (arm + '.npz')),
                    sha256=digest(original / (arm + '.npz'))) for arm in ARMS]))
        cls.spec = spec

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def output(self, name):
        path = self.root / name
        path.mkdir()
        return path

    def test_raw_anchor_current_reference_binding_and_no_future_observation_leak(self):
        ep = self.episode()
        for row in ep['samples']:
            row['actual_gripper_internal_memory']['values']['opening'] = .25
        data = contact_replay([ep])
        self.assertEqual(data['history'][0].reshape(9, 103)[-1, 97], .25)
        self.assertEqual(data['blind_plan'][0, 0], np.float32(ep['samples'][8]['executed_reference_EEF10'][-1]))
        self.assertNotEqual(data['blind_plan'][0, 0], .25)
        altered = copy.deepcopy(ep)
        altered['samples'][9]['qpos'][0] += 500
        for tick in altered['samples'][9]['native_contact_events']:
            tick['native_grasp'] = not tick['native_grasp']
        after = contact_replay([altered])
        np.testing.assert_array_equal(features(data, 1, ARMS[0])[:1], features(after, 1, ARMS[0])[:1])
        self.assertNotEqual(data['events'][0, 0, 1], after['events'][0, 0, 1])
        fake = copy.deepcopy(data)
        fake['blind_plan'][:] = .25
        with patch('go2arx5_vla.world_model.contact_timing_full_history._original_contact_replay', return_value=fake):
            with self.assertRaisesRegex(ValueError, 'RAW_ANCHOR_EXECUTED'):
                contact_replay([ep])
        with self.assertRaisesRegex(ValueError, 'WHOLE_PARENT'):
            contact_replay([self.episode('test', 'TEST', 'X')])

    def test_two_models_exact_reload_before_exposed_read_and_old_comparators_retained(self):
        from scripts.wm02 import fit_contact_timing_full_history as cli
        output = self.output('full927')
        original, opened = cli._bundle, []
        exposed = {self.spec['matched_training_package']['path'], self.spec['evaluation_bundles'][0]['path']}
        def observed(item):
            if item['path'] in exposed:
                freeze = json.loads((output / 'TRAIN_FROZEN.private.json').read_text())
                self.assertEqual(freeze['schema'], 'WM02_CONTACT_TIMING_FULL927_TRAIN_FREEZE_V1')
                self.assertEqual(len(freeze['arms']), 2)
                self.assertTrue(all(v['exact_reload'] for v in freeze['arms']))
                self.assertFalse(freeze['DEV_read'])
                opened.append(item['path'])
            return original(item)
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''), patch.object(cli, '_bundle', observed):
            result = cli.run(copy.deepcopy(self.spec), output)
        self.assertEqual(len(opened), 2)
        self.assertEqual(result['observed_history_features'], 927)
        self.assertEqual(result['completed_model_fits'], 2)
        self.assertEqual(result['closed_form_head_solves'], 12)
        self.assertEqual(result['main_optimizer_updates'], 0)
        self.assertEqual(result['native_integrations'], 0)
        self.assertEqual(len(result['frozen_FIR90_comparators']), 2)
        self.assertTrue(result['frozen_neural_comparators'][0]['training_exposure_matches_baseline'])
        self.assertTrue(result['raw_anchor_executed_reference_binding_verified'])
        self.assertFalse(result['architecture_objective_or_fitted_normalizer_matches_neural'])
        self.assertEqual(result['frozen_FIR90_comparators'][0]['observed_history_features'], 90)
        self.assertFalse(result['new_independent_TEST'])

    def test_second_fit_failure_never_opens_exposed_or_old_comparator_data(self):
        from scripts.wm02 import fit_contact_timing_full_history as cli
        output = self.output('failed-second-fit')
        original, fit = cli._bundle, ContactTimingFullHistory.fit
        def training_only(item):
            self.assertIn('TRAIN_REPLAY', item['path'])
            return original(item)
        def fail_blind(model, data):
            if model.arm == ARMS[1]:
                raise FloatingPointError('SYNTHETIC_SECOND_MODEL_FAILURE')
            return fit(model, data)
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''), patch.object(cli, '_bundle', training_only), \
                patch.object(ContactTimingFullHistory, 'fit', fail_blind), patch.object(cli, '_frozen_fir') as old:
            with self.assertRaisesRegex(FloatingPointError, 'SECOND_MODEL_FAILURE'):
                cli.run(self.spec, output)
            old.assert_not_called()
        self.assertTrue((output / (ARMS[0] + '.npz')).is_file())
        self.assertFalse((output / 'TRAIN_FROZEN.private.json').exists())

    def test_original_fir_exposure_and_hash_cannot_be_relabelled(self):
        from scripts.wm02 import fit_contact_timing_full_history as cli
        old_path = Path(self.spec['frozen_fir']['freeze']['path'])
        bad = json.loads(old_path.read_text())
        bad['training_manifest_sha256'] = '0' * 64
        path = self.root / 'wrong-FIR-freeze.json'
        path.write_text(json.dumps(bad))
        spec = copy.deepcopy(self.spec)
        spec['frozen_fir']['freeze'] = dict(path=str(path), sha256=digest(path))
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''):
            with self.assertRaisesRegex(ValueError, 'EXACT_SAME_FROZEN_TRAIN'):
                cli.run(spec, self.output('wrong-FIR-exposure'))

    def test_stale_recipe_or_gpu_is_not_accepted_and_numpy_cli_imports_no_models(self):
        from scripts.wm02 import fit_contact_timing_full_history as cli
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''):
            for key, value in [('device', 'cuda'), ('observed_history_features', 90),
                               ('recipe', 'WM02_CONTACT_TIMING_MATCHED_TRAIN_FIR_V1')]:
                with self.subTest(key=key), self.assertRaises(PermissionError):
                    cli.run(dict(self.spec, **{key:value}), self.root / 'never-created')
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / 'STOP.json').write_text('{}')
                with self.assertRaisesRegex(TimeoutError, 'ENDPOINT_OR_STOP'):
                    cli.run(dict(self.spec, registered_root=str(root)), root / 'fit')
        script = '''
import builtins,sys,json
from pathlib import Path
from contextlib import contextmanager
original=builtins.__import__
def guarded(name,*a,**kw):
    if name.split('.')[0] in ('torch','mujoco','transformers'):
        raise ImportError('MODEL_OR_PHYSICS_IMPORT_FORBIDDEN_IN_FULL927_NUMPY_CLI')
    return original(name,*a,**kw)
builtins.__import__=guarded
from go2arx5_vla.world_model import runtime
@contextmanager
def synthetic_admission(path,run_id,profile,output):
    spec=json.loads(Path(path).read_text())
    if spec['source_sha']!='SYNTHETIC_NOT_NATIVE':raise ValueError('FIXTURE_ONLY')
    Path(output).mkdir()
    yield spec
runtime.admitted_run=synthetic_admission
from scripts.wm02.fit_contact_timing_full_history import main
raise SystemExit(main(sys.argv[1:]))
'''
        result = subprocess.run([sys.executable, '-B', '-c', script, '--help'],
            check=True, capture_output=True, text=True)
        self.assertIn('--spec', result.stdout)
        spec_path = self.root / 'FULL927_SPEC.json'
        spec_path.write_text(json.dumps(self.spec))
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''):
            result = subprocess.run([sys.executable, '-B', '-c', script,
                '--spec', str(spec_path), '--output', str(self.root / 'no-model-imports')],
                check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(result.stdout)['main_optimizer_updates'], 0)


class FullHistoryAdmission(unittest.TestCase):
    def test_runtime_uses_new_scope_and_checks_all_pinned_inputs_without_GPU_or_physics(self):
        from go2arx5_vla.world_model.runtime import admitted_run
        from scripts.wm02.fit_contact_timing_full_history import PROFILE
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = dict(profile=PROFILE, host=os.uname().nodename, uid=os.getuid(),
                source_root=str(Path(__file__).parents[2]), registered_root=str(root),
                deadline_epoch=time.time()+60, execution_site='CORE_FUSION_CONTACT_FULL_HISTORY_CPU',
                python=str(root/'python'), data=str(root/'data'), cache=str(root/'cache'),
                logs=str(root/'logs'), tmp=str(root/'tmp'), training_manifest=str(root/'TRAIN'),
                evaluation_bundles=[dict(path=str(root/'DEV'))],
                matched_training_package=dict(path=str(root/'MIXED')),
                frozen_fir=dict(freeze=dict(path=str(root/'FIR_FREEZE')),
                    models=[dict(path=str(root/'FIR_AC')), dict(path=str(root/'FIR_BLIND'))]),
                frozen_models=[dict(path=str(root/'NN'), checkpoint=str(root/'NN_CHECKPOINT'))])
            path = root/'SPEC.json'
            path.write_text(json.dumps(spec))
            with patch('go2arx5_vla.world_model.pilot1_storage.local_path') as local, patch(
                    'go2arx5_vla.world_model.fusion_scope.validate_contact_full_history_cpu',
                    side_effect=PermissionError('SCOPE_REACHED_BEFORE_ANY_EFFECT')) as scope:
                with self.assertRaisesRegex(PermissionError, 'SCOPE_REACHED'):
                    with admitted_run(path, 'fit', PROFILE, root/'fit'):
                        self.fail('NO_RUNTIME_EFFECT_EXPECTED')
                scope.assert_called_once_with(spec)
                self.assertEqual(local.call_count, 15)
                self.assertFalse((root/'fit').exists())
                scope.reset_mock()
                local.reset_mock()
                for kwargs in [dict(gpu=True), dict(physics=True)]:
                    with self.subTest(kwargs=kwargs), self.assertRaisesRegex(PermissionError, 'NO_CUDA_OR_PHYSICS|does not admit'):
                        with admitted_run(path, 'fit', PROFILE, root/'fit', **kwargs):
                            self.fail('NO_RUNTIME_EFFECT_EXPECTED')
                scope.assert_not_called()
                local.assert_not_called()

    def test_new_recipe_uses_same_owner_validation_and_old_recipe_stays_separate(self):
        from go2arx5_vla.world_model.fusion_scope import (
            validate_contact_full_history_cpu, validate_contact_timing_cpu,
        )
        from scripts.wm02.fit_contact_timing_full_history import PROFILE, KIND, RECIPE
        spec = dict(execution_site='CORE_FUSION_CONTACT_FULL_HISTORY_CPU', profile=PROFILE,
            kind=KIND, recipe=RECIPE, physics=False, device='cpu', numeric_mode='NUMPY_FLOAT64',
            ridge=1., arms=list(ARMS), information='PREDECLARED_PROGRAM_CONDITIONAL',
            feature_recipe=FEATURE_RECIPE, observed_history_features=927, blind_source=BLIND_SOURCE,
            GR00T_updates=0, WBC_changes=False)
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''), patch(
                'go2arx5_vla.world_model.fusion_scope.validate', return_value='original-owner') as owner:
            self.assertEqual(validate_contact_full_history_cpu(spec), 'original-owner')
            owner.assert_called_once_with(spec)
            with self.assertRaises(PermissionError):
                validate_contact_timing_cpu(spec)
            for key, value in [('physics', True), ('device', 'cuda'), ('ridge', .1),
                               ('ridge', True), ('GR00T_updates', 1), ('observed_history_features', 90),
                               ('blind_source', 'INTERNAL_MEMORY_OPENING')]:
                with self.subTest(key=key), self.assertRaises(PermissionError):
                    validate_contact_full_history_cpu(dict(spec, **{key:value}))
            owner.side_effect = PermissionError('CLOSED_OWNER_NOT_REOPENED')
            with self.assertRaisesRegex(PermissionError, 'CLOSED_OWNER'):
                validate_contact_full_history_cpu(spec)


if __name__ == '__main__':
    unittest.main()
