import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.pi0292.eval_contract import error_metrics, gate_action
from scripts.pi0294.native_dataset import sha
from .mp1.probe_compare import PRECISION, load_batch, compare_batches


def fixture(root, offset=0., training_entry=False):
    folder = Path(root)
    folder.mkdir()
    windows, records, observations = [], [], {}
    state = np.array([0., 0., .5, 1., 0., 0., 0., 1., 0., 1.])
    expert = np.tile(state, (40, 1))
    for i in range(2):
        identifier = 'TRAIN-T1-A' + str(i)
        noise = dict(shape=[1, 40, 132], dtype='torch.float32', sha256='c'*64)
        precision = dict(PRECISION, training_flag_on_entry=training_entry)
        prediction = expert.copy()
        prediction[:, 0] += offset
        path = folder / (identifier + '.npz')
        np.savez(path, physical_H40=prediction, expert_absolute10=expert, state=state,
                 noise_seed=np.asarray(i, dtype=np.uint64),
                 noise_fingerprint_json=np.asarray(json.dumps(noise)),
                 inference_precision_json=np.asarray(json.dumps(precision)),
                 observation_fingerprint=np.asarray('d'*64),
                 checkpoint_files_fingerprint=np.asarray('a'*64))
        window = dict(id=identifier, episode_id='TRAIN-T1', anchor=i, split='TRAIN', task='T1',
            stage='hold', noise_seed=i, state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
            absolute_H40_sha256=hashlib.sha256(expert.tobytes()).hexdigest())
        windows.append(window)
        records.append({k:window[k] for k in ['id', 'episode_id', 'anchor', 'split', 'task']} |
            dict(seed=i, noise=noise, inference_precision=precision, arrays_sha256=sha(path),
                 errors=error_metrics(prediction, expert, state), gate=gate_action(prediction, state, i*.05, identifier)))
        observations[identifier] = 'd'*64
    (folder / 'ON_EXPERT_RESULTS.json').write_text(json.dumps(dict(
        complete=True, status='ACTUAL_NATIVE_TRAIN_PROBES_NOT_TASK_SUCCESS', records=records)))
    return dict(TRAIN_only=True, count=2, windows=windows), observations


class ProbeCompareTest(unittest.TestCase):
    def test_same_effective_precision_true_episode_macro_keeps_rejected_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, obs = fixture(root/'parent', offset=5.)
            fixture(root/'candidate', offset=4., training_entry=True)
            a = load_batch(root/'parent', manifest, 'a'*64, obs)
            b = load_batch(root/'candidate', manifest, 'a'*64, obs)
            result = compare_batches(a, b)
            self.assertEqual(result['windows'], 2)
            self.assertEqual(result['distinct_episodes'], 1)
            summary = result['summaries']['T1']['candidate']
            self.assertEqual(summary['decisions'], dict(REJECT=2))
            self.assertEqual(summary['episode_macro']['prefix8']['XYZ_euclidean_m']['episodes'], 1)
            self.assertAlmostEqual(summary['episode_macro']['prefix8']['XYZ_euclidean_m']['mean'], 4.)
            self.assertFalse(result['rejected_or_nonfinite_predictions_excluded'])

    def test_changed_seed_precision_metadata_or_source_state_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, obs = fixture(root/'saved')
            for field, value in [('noise_seed', 99), ('state_sha256', 'b'*64)]:
                changed = copy.deepcopy(manifest)
                changed['windows'][0][field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    load_batch(root/'saved', changed, 'a'*64, obs)
            with self.assertRaises(ValueError):
                load_batch(root/'saved', manifest, 'b'*64, obs)
            with self.assertRaises(ValueError):
                load_batch(root/'saved', manifest, 'a'*64, dict(obs, **{next(iter(obs)):'b'*64}))
            p = root/'saved/ON_EXPERT_RESULTS.json'
            value = json.loads(p.read_text())
            value['records'][0]['inference_precision']['cuda_matmul_allow_tf32'] = True
            p.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'PRECISION'):
                load_batch(root/'saved', manifest, 'a'*64, obs)

    def test_incomplete_duplicate_wrong_metrics_and_array_changes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, obs = fixture(root/'saved')
            p = root/'saved/ON_EXPERT_RESULTS.json'
            original = json.loads(p.read_text())
            for change in ['incomplete', 'duplicate', 'metrics']:
                value = copy.deepcopy(original)
                if change == 'incomplete':
                    value['complete'] = False
                elif change == 'duplicate':
                    value['records'].append(value['records'][0])
                else:
                    value['records'][0]['errors']['prefix8']['gripper_MAE'] = 99.
                p.write_text(json.dumps(value))
                with self.subTest(change=change), self.assertRaises((ValueError, PermissionError)):
                    load_batch(root/'saved', manifest, 'a'*64, obs)
            p.write_text(json.dumps(original))
            (root/'saved'/ (manifest['windows'][0]['id']+'.npz')).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'HASH_OR_PATH'):
                load_batch(root/'saved', manifest, 'a'*64, obs)

    def test_pair_noise_or_context_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, obs = fixture(root/'saved')
            a = load_batch(root/'saved', manifest, 'a'*64, obs)
            b = copy.deepcopy(a)
            b['signatures'][next(iter(b['signatures']))]['noise']['sha256'] = 'e'*64
            with self.assertRaisesRegex(ValueError, 'NOISE_CHANGED'):
                compare_batches(a, b)
            b = copy.deepcopy(a)
            b['rows'][0]['stage'] = 'approach'
            with self.assertRaisesRegex(ValueError, 'CONTEXT_CHANGED'):
                compare_batches(a, b)

    def test_dev_manifest_or_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, obs = fixture(root/'saved')
            manifest['windows'][0]['split'] = 'DEV'
            with self.assertRaises(PermissionError):
                load_batch(root/'saved', manifest, 'a'*64, obs)
            manifest['windows'][0]['split'] = 'TRAIN'
            identifier = manifest['windows'][0]['id']
            p = root/'saved'/(identifier+'.npz')
            target = root/'outside.npz'
            p.rename(target)
            p.symlink_to(target)
            with self.assertRaises(ValueError):
                load_batch(root/'saved', manifest, 'a'*64, obs)

    def test_nonfinite_prediction_remains_in_denominator_with_incomplete_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, obs = fixture(root/'parent')
            fixture(root/'candidate')
            records_path = root/'candidate/ON_EXPERT_RESULTS.json'
            value = json.loads(records_path.read_text())
            for record in value['records']:
                path = root/'candidate'/(record['id']+'.npz')
                with np.load(path, allow_pickle=False) as saved:
                    arrays = {k:saved[k].copy() for k in saved.files}
                arrays['physical_H40'][:, 0] = np.nan
                np.savez(path, **arrays)
                record['arrays_sha256'] = sha(path)
                record['errors'] = error_metrics(arrays['physical_H40'], arrays['expert_absolute10'], arrays['state'])
                record['gate'] = gate_action(arrays['physical_H40'], arrays['state'], record['anchor']*.05, record['id'])
            records_path.write_text(json.dumps(value))
            result = compare_batches(load_batch(root/'parent', manifest, 'a'*64, obs),
                                     load_batch(root/'candidate', manifest, 'a'*64, obs))
            summary = result['summaries']['ALL']['candidate']
            self.assertEqual(summary['queries'], 2)
            self.assertEqual(summary['decisions'], dict(REJECT=2))
            metric = summary['episode_macro']['prefix8']['XYZ_euclidean_m']
            self.assertFalse(metric['complete'])
            self.assertIsNone(metric['mean'])


if __name__ == '__main__':
    unittest.main()
