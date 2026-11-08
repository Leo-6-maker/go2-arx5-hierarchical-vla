"""Synthetic ABI/causality checks; these do not qualify native contact."""

import copy
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from go2arx5_vla.world_model.contact_timing_baseline import (
    ARMS, HORIZONS, ContactTimingFIR, causal_features, contact_replay,
)
from go2arx5_vla.world_model.pilot2_contact_model import planned_openings
from go2arx5_vla.world_model.pilot2_supported import PLAN, stage_for
from go2arx5_vla.world_model.storage import digest


def episode(run='train', split='TRAIN', parent='T', close=56, program='CLOSE_RELEASE'):
    source = Path(__file__).parents[2] / 'experiments/wm02/pilot2_supported_collection_v1.json'
    e = json.loads(source.read_text())
    e.update(split=split, parent=parent, close_start_atom=close, command_program=program)
    plan, rows = planned_openings(e), []
    for i, command in enumerate(plan):
        start = 2503 + 25 * i
        release = close + 56 if program == 'CLOSE_RELEASE' else 1000
        grasp = program != 'OPEN_ONLY' and close + 12 <= i < release + 4
        contact = program != 'OPEN_ONLY' and close + 11 <= i < release + 6
        finger = .018 if grasp else .042
        pose = np.eye(4)
        pose[0, 3] = i * .000001
        ticks = [dict(native_step=s, native_grasp=grasp, table_support=True,
            contact_positive=contact, max_pad_cube_normal_force_N=.2 if contact else 0.,
            finger_q=[finger, finger], stage=stage_for(i, e)) for s in range(start + 1, start + 26)]
        rows.append(dict(native_start_step=start, native_end_step=start + 25,
            qpos=[0.] * 34, qvel=[0.] * 32, ctrl=[0.] * 20, TCP_m=[i * .000001, 0., .7],
            object_pose=pose.tolist(), finger_q=[finger, finger], contact_any=contact,
            native_contact_events=ticks, stage=stage_for(i, e),
            executed_reference_EEF10=[0., 0., .7, 1., 0., 0., 0., 1., 0., command],
            actual_gripper_internal_memory=dict(status='ACTUAL_OWNED_CHILD_GLOBALS_READ_ONLY',
                values=dict(steps=start + 25, target=[command * .044] * 2, opening=command,
                            expiry=(start + 25) * .002 + .35))))
    return dict(run_id=run, parent=parent, split=split, episode=e, samples=rows,
        declared_episode_schema='WM02_PILOT2_SUPPORTED_EPISODE_V1',
        program_known_from_anchor=8, policy_decision_sha256=None)


class CausalContactTiming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episodes = [episode(), episode('train2', close=64, program='CLOSE_KEEP')]
        cls.data = contact_replay(cls.episodes)
        cls.models = {arm: ContactTimingFIR(arm).fit(cls.data) for arm in ARMS}

    def test_future_suffix_cannot_change_earlier_forecasts_but_commands_inside_can(self):
        data, model = self.data, self.models[ARMS[0]]
        before = model.predict(data['history'][:4], data['plan'][:4])
        for j, horizon in enumerate(HORIZONS):
            with self.subTest(horizon=horizon):
                changed = data['plan'][:4].copy()
                changed[:, horizon:] += .13
                after = model.predict(data['history'][:4], changed)
                np.testing.assert_array_equal(before[:, j], after[:, j])
                x = causal_features(data['history'][:4], changed, horizon, ARMS[0])
                changed[:, :horizon] += .01
                self.assertFalse(np.array_equal(x, causal_features(data['history'][:4], changed, horizon, ARMS[0])))

    def test_blind_head_is_independent_of_all_future_program_values(self):
        model = self.models[ARMS[1]]
        changed = self.data['plan'][:8] + .2
        np.testing.assert_array_equal(model.predict(self.data['history'][:8], self.data['plan'][:8]),
                                      model.predict(self.data['history'][:8], changed))

    def test_clock_translation_phase_and_privileged_object_are_not_features(self):
        a = contact_replay([episode('a', close=56)])
        b = contact_replay([episode('b', close=64)])
        ia, ib = 52 - 8, 60 - 8
        for horizon in HORIZONS:
            np.testing.assert_array_equal(
                causal_features(a['history'][ia:ia+1], a['plan'][ia:ia+1], horizon, ARMS[0]),
                causal_features(b['history'][ib:ib+1], b['plan'][ib:ib+1], horizon, ARMS[0]))
        changed = copy.deepcopy(self.episodes[0])
        for row in changed['samples']:
            row['stage'] = 'PRIVILEGED_UNUSED_DIAGNOSTIC'
            row['TCP_m'][0] += .4
            row['object_pose'][0][3] -= .6
        c = contact_replay([changed])
        np.testing.assert_array_equal(
            causal_features(a['history'], a['plan'], 16, ARMS[0]),
            causal_features(c['history'], c['plan'], 16, ARMS[0]))

    def test_future_observation_changes_labels_not_anchor_features(self):
        a = contact_replay([self.episodes[0]])
        altered = copy.deepcopy(self.episodes[0])
        for tick in altered['samples'][9]['native_contact_events']:
            tick['native_grasp'] = not tick['native_grasp']
        b = contact_replay([altered])
        np.testing.assert_array_equal(a['history'][0], b['history'][0])
        np.testing.assert_array_equal(a['plan'][0], b['plan'][0])
        self.assertNotEqual(a['events'][0, 0, 1], b['events'][0, 0, 1])

    def test_adaptive_program_knowledge_only_starts_at_real_paused_choice(self):
        ep = copy.deepcopy(self.episodes[0])
        ep.update(declared_episode_schema='WM02_PILOT2_SUPPORTED_POLICY_EPISODE_V1',
                  program_known_from_anchor=111, policy_decision_sha256='a' * 64)
        ep['episode']['purpose'] = 'SUPPORTED_POLICY_EVALUATION'
        data = contact_replay([ep])
        self.assertEqual(data['identities'][0]['anchor_native_step'], 5303)
        self.assertEqual(len(data['history']), 56)
        ep['program_known_from_anchor'] = 8
        with self.assertRaisesRegex(ValueError, 'PAUSED_POLICY_PROGRAM_KNOWLEDGE'):
            contact_replay([ep])
        static = copy.deepcopy(self.episodes[0])
        static.pop('declared_episode_schema')
        with self.assertRaisesRegex(ValueError, 'PROGRAM_PROVENANCE'):
            contact_replay([static])

    def test_dev_test_and_disguised_field_layout_cannot_enter_fit(self):
        data = contact_replay([self.episodes[0], episode('dev', 'DEV', 'V')])
        with self.assertRaisesRegex(ValueError, 'TRAIN_ONLY'):
            ContactTimingFIR(ARMS[0]).fit(data)
        with self.assertRaisesRegex(ValueError, 'WHOLE_PARENT'):
            contact_replay([episode('test', 'TEST', 'X')])
        bad = copy.deepcopy(self.episodes[0])
        bad['samples'][0]['qpos'].append(0.)
        bad['samples'][0]['qvel'].pop()
        with self.assertRaisesRegex(ValueError, 'FIELD_OFFSETS'):
            contact_replay([bad])

    def test_unexecuted_or_fallback_suffix_remains_masked(self):
        ep = copy.deepcopy(self.episodes[0])
        before = contact_replay([ep])
        ep['samples'][65]['executed_reference_EEF10'][-1] = .95
        after = contact_replay([ep])
        self.assertGreater(before['valid'].sum(), after['valid'].sum())
        np.testing.assert_array_equal(before['history'][0], after['history'][0])
        changed = copy.deepcopy(after)
        changed['events'][~changed['valid']] = 1 - changed['events'][~changed['valid']]
        first = ContactTimingFIR(ARMS[0]).fit(after).arrays()
        second = ContactTimingFIR(ARMS[0]).fit(changed).arrays()
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])

    def test_exact_reload_and_schema_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for arm, model in self.models.items():
                path = root / (arm + '.npz')
                model.save(path)
                clone = ContactTimingFIR.load(path)
                np.testing.assert_array_equal(model.predict(self.data['history'], self.data['plan']),
                                              clone.predict(self.data['history'], self.data['plan']))
            arrays = self.models[ARMS[0]].arrays()
            arrays['scale_1'][0] = 0
            bad = root / 'bad.npz'
            with bad.open('wb') as stream:
                np.savez_compressed(stream, **arrays)
            with self.assertRaisesRegex(ValueError, 'TRAIN_FITTED'):
                ContactTimingFIR.load(bad)


class ContactTimingCLI(unittest.TestCase):
    def fixture(self, root):
        binary, scene = root / 'native', root / 'scene'
        binary.write_bytes(b'SYNTHETIC_NOT_NATIVE')
        scene.write_bytes(b'SYNTHETIC_NOT_NATIVE')
        training = [episode('train1', close=56), episode('train2', close=64, program='CLOSE_KEEP')]
        descriptors = []
        for ep in training:
            rr = root / ep['run_id']
            (rr / 'native-01').mkdir(parents=True)
            ep['episode']['run'] = str(rr)
            ep_path = rr / 'EPISODE_MANIFEST.private.json'
            ep_path.write_text(json.dumps(ep['episode']))
            spec = dict(pilot2_plan_sha=PLAN, episode_manifest_sha256=digest(ep_path),
                        native_binary=str(binary), model_path=str(scene), source_sha='SYNTHETIC_NOT_NATIVE')
            spec_path = rr / 'EXECUTION_SPEC.json'
            spec_path.write_text(json.dumps(spec))
            native = dict(status='NATIVE_STAGE_PASSED', physics_stops=[], shutdown_clean=True,
                actual_target_integrations=6703, native_outer_guard_rejection_count=0,
                native_binary_sha256=digest(binary), scene_sha256=digest(scene), source_sha=spec['source_sha'])
            npath = rr / 'native-01/RESULT.json'
            npath.write_text(json.dumps(native))
            score = dict(same_native_process=True, physics_resets=0, state_teleports=0,
                         actual_task_atoms=168, goal_success=False, first_blocker=None, samples=ep['samples'])
            spath = rr / 'native-01/SUPPORTED_CLOSE_RESULT.private.json'
            spath.write_text(json.dumps(score))
            descriptors.append(dict(run=str(rr), spec_sha256=digest(spec_path), episode_sha256=digest(ep_path),
                                    result_sha256=digest(npath), score_sha256=digest(spath)))
            ep.update(native_result_sha256=digest(npath), score_sha256=digest(spath),
                      source_sha=spec['source_sha'], native_binary_sha256=digest(binary), scene_sha256=digest(scene))
        manifest = root / 'TRAIN_MANIFEST.json'
        manifest.write_text(json.dumps(dict(schema='WM02_PILOT2_SUPPORTED_BUNDLE_MANIFEST_V1',
                                           plan_sha=PLAN, episodes=descriptors)))
        dev = episode('dev', 'DEV', 'V')
        evaluation = training + [episode('holdout', close=72), dev]
        bundle = root / 'FULL_BUNDLE.gz'
        bundle.write_bytes(gzip.compress(json.dumps(dict(schema='WM02_PILOT2_SUPPORTED_ACTUAL_BUNDLE_V1',
            plan_sha=PLAN, complete_MPC_WBC_hidden_state=False, free_rollout_qualified=False,
            episodes=evaluation)).encode(), mtime=0))
        matched = root / 'MATCHED_TRAIN_PACKAGE.gz'
        matched.write_bytes(gzip.compress(json.dumps(dict(schema='WM02_PILOT2_SUPPORTED_ACTUAL_BUNDLE_V1',
            plan_sha=PLAN, complete_MPC_WBC_hidden_state=False, free_rollout_qualified=False,
            episodes=training + [dev])).encode(), mtime=0))
        from scripts.wm02.fit_contact_timing_baseline import PROFILE, KIND, RECIPE
        item = dict(path=str(bundle), sha256=digest(bundle))
        spec = dict(profile=PROFILE, kind=KIND, recipe=RECIPE, device='cpu', physics=False,
            numeric_mode='NUMPY_FLOAT64', arms=list(ARMS), ridge=1.,
            information='PREDECLARED_PROGRAM_CONDITIONAL', GR00T_updates=0, WBC_changes=False,
            deadline_epoch=time.time() + 120, registered_root=str(root), source_sha='SYNTHETIC_NOT_NATIVE',
            training_manifest=str(manifest), training_manifest_sha256=digest(manifest),
            expected_TRAIN_episodes=2, expected_TRAIN_parents=1,
            matched_training_package=dict(path=str(matched), sha256=digest(matched)),
            evaluation_bundles=[dict(item, name='SYNTHETIC_EXPOSED')], frozen_models=[])
        portable = root / 'FROZEN_PREFIX.npz'
        arrays = dict(mean=np.zeros(991), scale=np.ones(991), horizon_count=np.array(6),
            architecture_code=np.array(1), forecast_horizons=np.array(HORIZONS),
            prefix_mask=np.arange(64)[None] < np.array(HORIZONS)[:, None])
        for index, insize, outsize in [(0, 991, 1), (2, 1, 1), (4, 1, 60)]:
            arrays[f'prefix_network.{index}.weight'] = np.zeros((outsize, insize))
            arrays[f'prefix_network.{index}.bias'] = np.zeros(outsize)
        with portable.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        checkpoint = root / 'SYNTHETIC_CHECKPOINT_BYTES'
        checkpoint.write_bytes(b'SYNTHETIC_SOFTWARE_TEST_NOT_TRAINED')
        spec['frozen_models'] = [dict(name='SYNTHETIC_FROZEN_PREFIX', arm=ARMS[0], path=str(portable),
            sha256=digest(portable), checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
            training_bundle_sha256=digest(matched))]
        return spec

    def test_real_entry_and_train_freeze_precede_any_exposed_data_read_without_torch(self):
        from scripts.wm02 import fit_contact_timing_baseline as cli
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''):
            root = Path(tmp)
            spec = self.fixture(root)
            output = root / 'fit'
            output.mkdir()
            original = cli._bundle
            reads = []
            def observed_read(item):
                if item['path'] in [spec['matched_training_package']['path'], spec['evaluation_bundles'][0]['path']]:
                    self.assertTrue((output / 'TRAIN_FROZEN.private.json').is_file())
                    reads.append(item['path'])
                return original(item)
            with patch.object(cli, '_bundle', observed_read):
                result = cli.run(spec, output)
            self.assertEqual(len(reads), 2)
            self.assertEqual(result['completed_model_fits'], 2)
            self.assertEqual(result['closed_form_head_solves'], 12)
            self.assertEqual(result['main_optimizer_updates'], 0)
            self.assertEqual(result['new_real_transitions'], 0)
            self.assertEqual(result['native_integrations'], 0)
            self.assertFalse(result['TEST_read'])
            self.assertFalse(result['native_control_gain_evaluated'])
            self.assertEqual(result['INSTALLED_REFERENCE_ONLY'], 'NOT_RUN')
            self.assertTrue(result['frozen_neural_comparators'][0]['training_exposure_matches_baseline'])
            ids = result['results']['SYNTHETIC_EXPOSED']['models'][ARMS[0]]['strata']
            self.assertGreater(ids['TRAIN_TIMING_HOLDOUT_EXPOSED']['event_forecast_metrics']['1']['NATURAL']['valid_forecasts'], 0)
            self.assertGreater(ids['DEV_EXPOSED']['event_forecast_metrics']['1']['NATURAL']['valid_forecasts'], 0)
            self.assertNotIn('TCP_delta_RMSE_mm', ids['DEV_EXPOSED']['event_forecast_metrics']['1']['NATURAL'])
            spec_path = root / 'SPEC.json'
            spec_path.write_text(json.dumps(spec))
            script = '''
import builtins,sys,json
from contextlib import contextmanager
from pathlib import Path
original=builtins.__import__
def guarded(name,*a,**kw):
    if name=='torch' or name.startswith('torch.'):
        raise ImportError('TORCH_FORBIDDEN_IN_NUMPY_CONTACT_BASELINE')
    return original(name,*a,**kw)
builtins.__import__=guarded
from go2arx5_vla.world_model import runtime
@contextmanager
def synthetic_admission(path,run_id,profile,output):
    value=json.loads(Path(path).read_text())
    if value['source_sha']!='SYNTHETIC_NOT_NATIVE':raise ValueError('SOFTWARE_FIXTURE_ONLY')
    Path(output).mkdir()
    yield value
runtime.admitted_run=synthetic_admission
from scripts.wm02.fit_contact_timing_baseline import main
raise SystemExit(main(sys.argv[1:]))
'''
            p = subprocess.run([sys.executable, '-B', '-c', script, '--help'], capture_output=True, text=True, check=True)
            self.assertIn('--spec', p.stdout)
            p = subprocess.run([sys.executable, '-B', '-c', script, '--spec', str(spec_path),
                '--output', str(root / 'subprocess-fit')], capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(p.stdout)['main_optimizer_updates'], 0)

    def test_second_fit_failure_does_not_open_exposed_data_or_write_freeze(self):
        from scripts.wm02 import fit_contact_timing_baseline as cli
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''):
            root = Path(tmp)
            spec = self.fixture(root)
            output = root / 'fit'
            output.mkdir()
            original = cli._bundle
            def training_read_only(item):
                self.assertIn('TRAIN_REPLAY', item['path'])
                return original(item)
            fit = ContactTimingFIR.fit
            def fail_blind(model, data):
                if model.arm == ARMS[1]:
                    raise FloatingPointError('SYNTHETIC_SECOND_FIT_FAILURE')
                return fit(model, data)
            with patch.object(cli, '_bundle', training_read_only), patch.object(ContactTimingFIR, 'fit', fail_blind):
                with self.assertRaisesRegex(FloatingPointError, 'SECOND_FIT_FAILURE'):
                    cli.run(spec, output)
            self.assertTrue((output / (ARMS[0] + '.npz')).is_file())
            self.assertFalse((output / 'TRAIN_FROZEN.private.json').exists())

    def test_dev_cannot_be_placed_in_training_manifest_and_input_hash_is_bound(self):
        from scripts.wm02 import fit_contact_timing_baseline as cli
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''):
            root = Path(tmp)
            spec = self.fixture(root)
            path = root / 'train1/EPISODE_MANIFEST.private.json'
            ep = json.loads(path.read_text())
            ep['split'] = 'DEV'
            path.write_text(json.dumps(ep))
            with self.assertRaisesRegex(ValueError, 'INPUT_CHANGED'):
                cli._training_inputs(spec)
            manifest = json.loads(Path(spec['training_manifest']).read_text())
            manifest['episodes'][0]['episode_sha256'] = digest(path)
            Path(spec['training_manifest']).write_text(json.dumps(manifest))
            spec['training_manifest_sha256'] = digest(spec['training_manifest'])
            with self.assertRaisesRegex(ValueError, 'TRAIN_ONLY'):
                cli._training_inputs(spec)

    def test_stop_or_gpu_recipe_cannot_start_fit(self):
        from scripts.wm02.fit_contact_timing_baseline import run
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''):
            root = Path(tmp)
            spec = self.fixture(root)
            output = root / 'fit'
            output.mkdir()
            (root / 'STOP.json').write_text('{}')
            with self.assertRaisesRegex(TimeoutError, 'ENDPOINT_OR_STOP'):
                run(spec, output)
            with self.assertRaisesRegex(PermissionError, 'CPU_CONTACT_TIMING'):
                run(dict(spec, device='cuda'), output)


if __name__ == '__main__':
    unittest.main()
