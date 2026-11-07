"""Local synthetic-only CLI checks; no admission, real fit or native execution."""

import copy
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from go2arx5_vla.world_model.fusion_structured import ARMS, StructuredResponse
from go2arx5_vla.world_model.pilot2_mean_rule import fit as fit_rule
from go2arx5_vla.world_model.storage import digest
from scripts.wm02 import train_fusion_structured as training
from tests.world_model.test_fusion_structured import synthetic_cohort


def synthetic_inputs(root):
    """Match the production schema and gain constant with labeled fake evidence."""
    meta, data = synthetic_cohort()
    # Deliberately synthetic independent-restart displacement gaps make the
    # frozen half-TRAIN-p90-gap threshold exactly the real preregistered value.
    gap = 2. * training.FROZEN_TRAIN_MIN_GAIN_MM * .001
    sign = data['repeat_index'] * 2 - 1
    data['state'][:, 9:, 86] += sign[:, None] * gap * .5 * np.arange(1, 17)[None] / 16.
    sealed = root / 'SYNTHETIC_ONLY_REPLAY_JOB'
    (sealed / 'control').mkdir(parents=True)
    (sealed / 'control/LEASE.json').write_text(json.dumps(dict(status='CLOSED', synthetic_only=True)))
    replay_dir = sealed / 'replay-01'
    replay_dir.mkdir()
    replay = replay_dir / 'REAL_RESPONSE_WINDOWS.npz'
    with replay.open('xb') as stream:
        np.savez_compressed(stream, **data)
    metadata = root / 'SYNTHETIC_ONLY_METADATA.json'
    metadata.write_text(json.dumps(meta))
    task = root / 'SYNTHETIC_ONLY_TASK.json'
    task.write_text(json.dumps(dict(success_tolerance_m=training.ORIGINAL_TASK_TOLERANCE_M,
                                   fixture='SYNTHETIC_ONLY_NOT_NATIVE')))
    train = np.flatnonzero(data['split'] == 'TRAIN')
    records = [dict(parent=str(data['parent'][i]), split='TRAIN', run=str(data['run_id'][i]),
                    action=int(data['observed_action'][i]), repeat=int(data['repeat_index'][i]),
                    response_xy_m=(data['state'][i, -1, 86:88] - data['state'][i, 8, 86:88]).tolist())
               for i in train]
    rule = root / 'SYNTHETIC_ONLY_TRAIN_RULE.json'
    rule.write_text(json.dumps(fit_rule(records, sorted(set(data['parent'][train])), digest(task))))
    report = replay_dir / 'RESULT.private.json'
    report.write_text(json.dumps(dict(
        schema='WM02_FUSION_REAL_RESPONSE_REPLAY_V1', replay_sha256=digest(replay),
        eligible_windows=48, real_post_choice_transitions=768,
        physical_parents=dict(zip(data['parent'].tolist(), data['split'].tolist())),
        identities=[dict(run=str(data['run_id'][i]), parent=str(data['parent'][i]),
                         split=str(data['split'][i])) for i in range(48)],
        excluded=[], TEST_read=False, expert_action_labels=False, synthetic_only=True)))
    spec = dict(profile=training.PROFILE, kind=training.KIND, device='cpu', physics=False,
                numeric_mode='NUMPY_FLOAT64', arms=list(ARMS),
                min_lopo_gain_mm=training.FROZEN_TRAIN_MIN_GAIN_MM,
                source_sha='SYNTHETIC_ONLY_NOT_NATIVE', GR00T_updates=0, WBC_changes=False,
                registered_root=str(root), deadline_epoch=time.time()+60.)
    for name, path in [('replay', replay), ('teacher_metadata', metadata),
                       ('training_replay_result', report), ('task', task),
                       ('train_mean_response_model', rule)]:
        spec[name], spec[name+'_sha256'] = str(path), digest(path)
    output = root / 'SYNTHETIC_ONLY_FIT'
    output.mkdir()
    return spec, output, data


class StructuredTrainingCLI(unittest.TestCase):
    def test_both_train_models_freeze_and_reload_before_dev_and_report_actual_cost(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''):
            spec, output, data = synthetic_inputs(Path(tmp))
            events = []
            verify, truth = training._verify_reload, training._truth

            def checked_reload(original, restored, sample, train):
                self.assertTrue(np.all(sample['split'][train] == 'TRAIN'))
                result = verify(original, restored, sample, train)
                events.append(('verified', original.arm))
                return result

            def checked_truth(sample, ids):
                split = str(sample['split'][ids[0]])
                if split == 'DEV':
                    self.assertEqual(events[:2], [('verified', arm) for arm in ARMS])
                    frozen = json.loads((output / 'TRAIN_FROZEN.private.json').read_text())
                    self.assertTrue(frozen['both_arms_exactly_reloaded'])
                    self.assertFalse(frozen['DEV_evaluated'])
                events.append(('truth', split))
                return truth(sample, ids)

            with patch.object(training, '_verify_reload', side_effect=checked_reload), \
                    patch.object(training, '_truth', side_effect=checked_truth):
                result = training.run(spec, output)
            self.assertEqual(events, [('verified', arm) for arm in ARMS]
                             + [('truth', 'TRAIN'), ('truth', 'DEV')])
            self.assertEqual(result['optimizer_updates'], 0)
            self.assertFalse(result['neural_training'])
            self.assertEqual(result['native_integrations'], 0)
            self.assertEqual((result['actual_TRAIN_windows'], result['actual_DEV_windows']), (32, 16))
            self.assertEqual(result['completed_model_fits'], 2)
            self.assertIn(result['completed_closed_form_ridge_solves'], (40, 41, 42))
            self.assertFalse(result['DEV_used_for_fit_normalization_or_selection'])
            self.assertFalse(result['TASK_TRANSFER_GAIN'])
            self.assertGreaterEqual(result['CPU_s'], 0.)
            self.assertGreater(result['wall_s'], 0.)
            self.assertEqual(result['TRAIN_freeze_sha256'], digest(output/'TRAIN_FROZEN.private.json'))
            for row in result['results']:
                self.assertTrue(row['restore']['array_serialization_exact'])
                self.assertEqual(row['restore']['TRAIN_candidate_reload_max_error_m'], 0.)
                self.assertEqual(set(row['absolute']), {'TRAIN', 'DEV'})
                self.assertEqual(len(row['selection']['folds']), 4)
                self.assertEqual(row['action_effect']['DEV']['independent_restart_pairs'], 12)
                if row['selected_ridge'] == 'inf':
                    self.assertEqual(row['static_minus_model_pooled_RMSE_mm'], dict(TRAIN=0., DEV=0.))
                model = StructuredResponse.load(row['model_path'])
                with np.load(output/row['arm']/'DEV_PREDICTIONS.private.npz', allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['prediction'], model.predict(data, saved['ids']))
            baseline = result['baseline_metrics']['TRAIN_MEAN_ACTION_RESPONSE']['DEV']
            self.assertEqual(set(baseline['by_parent']), {'D0', 'D1'})
            self.assertEqual(set(baseline['by_action']), {'0', '1', '2', '3'})
            before = digest(output/'RESULT.private.json')
            training.record_status(output, progress=dict(stage='observer-update'))
            training.record_status(output, progress=dict(stage='observer-update-again'))
            self.assertEqual(digest(output/'RESULT.private.json'), before)
            self.assertEqual(json.loads((output/'PROGRESS.private.json').read_text())['stage'], 'observer-update-again')

    def test_failed_second_arm_reload_prevents_dev_and_success_result(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''):
            spec, output, _ = synthetic_inputs(Path(tmp))
            original_load = StructuredResponse.load

            def broken_load(path):
                if ARMS[1] in str(path):
                    raise ValueError('SYNTHETIC_SECOND_ARM_RELOAD_FAILURE')
                return original_load(path)

            with patch.object(StructuredResponse, 'load', side_effect=broken_load), \
                    patch.object(training, '_truth') as truth:
                with self.assertRaisesRegex(ValueError, 'SECOND_ARM_RELOAD_FAILURE'):
                    training.run(spec, output)
                truth.assert_not_called()
            self.assertFalse((output/'TRAIN_FROZEN.private.json').exists())
            self.assertFalse((output/'BASELINES.private.json').exists())
            self.assertFalse((output/'RESULT.private.json').exists())
            self.assertTrue((output/ARMS[0]/'TRAIN_FIT.private.json').exists())

    def test_tampered_files_unsealed_or_mismatched_replay_and_recipe_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''):
            spec, output, _ = synthetic_inputs(Path(tmp))
            for change in (dict(min_lopo_gain_mm=0.), dict(min_lopo_gain_mm=.00001),
                           dict(arms=list(reversed(ARMS))), dict(device='cuda'),
                           dict(numeric_mode='FP32_TF32_OFF'), dict(WBC_changes=True),
                           dict(replay_sha256='tampered')):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    training.run(dict(spec, **change), output)
            report_path = Path(spec['training_replay_result'])
            original = json.loads(report_path.read_text())
            for key, value in [('eligible_windows', 47), ('excluded', ['failure']), ('TEST_read', True)]:
                changed = dict(original, **{key: value})
                report_path.write_text(json.dumps(changed))
                varied = dict(spec, training_replay_result_sha256=digest(report_path))
                with self.subTest(key=key), self.assertRaises(PermissionError):
                    training.run(varied, output)
            changed = copy.deepcopy(original)
            changed['identities'][0]['run'] = 'OTHER_SYNTHETIC_RUN'
            report_path.write_text(json.dumps(changed))
            varied = dict(spec, training_replay_result_sha256=digest(report_path))
            with self.assertRaisesRegex(ValueError, 'IDENTITIES_DIFFER'):
                training.run(varied, output)
            report_path.write_text(json.dumps(original))
            lease = report_path.parent.parent/'control/LEASE.json'
            lease.write_text(json.dumps(dict(status='ACTIVE')))
            with self.assertRaises(PermissionError):
                training.run(spec, output)
            self.assertFalse((output/'RESULT.private.json').exists())

    def test_frozen_gain_rule_is_bound_to_same_train_replay_and_original_task(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''):
            spec, output, _ = synthetic_inputs(Path(tmp))
            rule_path = Path(spec['train_mean_response_model'])
            rule = json.loads(rule_path.read_text())
            rule['same_action_restart_gap_m'][0][0] += .001
            rule_path.write_text(json.dumps(rule))
            spec['train_mean_response_model_sha256'] = digest(rule_path)
            with self.assertRaisesRegex(ValueError, 'RULE_DIFFERS_FROM_REPLAY'):
                training.run(spec, output)
            self.assertFalse((output/'TRAIN_FROZEN.private.json').exists())


if __name__ == '__main__':
    unittest.main()
