import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

from .checkpoint_artifacts import REQUIRED, inventory
from .main_capacity import GIB, calculate
from .train_adapter import CyclicSampler, sha


class TestCheckpointInventory(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.recipe = dict(max_steps=4, effective_batch=32)
        self.make_checkpoint()

    def tearDown(self):
        self.temp.cleanup()

    def put(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def make_checkpoint(self):
        self.put('statistics.json', {'synthetic': True})
        self.stats = sha(self.root / 'statistics.json')
        self.put('trainer_state.json', dict(global_step=4))
        self.put('SAMPLER_STATE.json', dict(recipe=self.recipe,
            sampler=CyclicSampler(601, stop=128).state_dict(128),
            statistics_sha256=self.stats, manifest_sha256='m' * 64,
            worker_trace=dict(status='PASS_ACTUAL_CONSUMED_WORKER_IDENTITIES', start=0,
                              stop=128, count=128, prefetch_excluded=True)))
        # Opaque dummy bytes deliberately cannot be deserialized as real state.
        for name in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
            (self.root / name).write_bytes(b'SYNTHETIC_TEST_NOT_A_CHECKPOINT')
        self.put('config.json', {'fixture_only': True})
        self.put('processor/processor_config.json', dict(processor_kwargs=dict(
            max_state_dim=132, max_action_dim=132, max_action_horizon=40,
            use_percentiles=False, use_mean_std=False, clip_outliers=False, use_relative_action=True,
            state_dropout_prob=0., exclude_state=False, use_albumentations=False,
            image_crop_size=[256, 256], image_target_size=[256, 256], crop_fraction=1.)))
        self.put('processor/statistics.json', {'synthetic': True})
        self.put('processor/embodiment_id.json', {'new_embodiment': 10})
        save_file({'weight':np.arange(6, dtype=np.float32).reshape(2, 3)}, self.root / 'model.safetensors')
        self.refresh_receipt()

    def refresh_receipt(self):
        self.put('CHECKPOINT_COMPLETE.json', dict(global_step=4,
            status='FULL_STATE_SAVED_MODEL_ARCHIVE_HASH_PENDING',
            required_files={name:sha(self.root / name) for name in REQUIRED}))

    def inspect(self):
        return inventory(self.root, expected_step=4, recipe=self.recipe,
                         statistics_sha256=self.stats, manifest_sha256='m' * 64)

    def test_inventory_hashes_every_file_without_model_or_pickle_execution(self):
        result = self.inspect()
        self.assertEqual(result['global_step'], 4)
        self.assertEqual(result['consumed_samples'], 128)
        self.assertEqual(result['model_tensors']['weight']['shape'], [2, 3])
        self.assertFalse(result['optimizer_pickle_loaded'])
        self.assertFalse(result['model_loaded'])
        self.assertIn('NOT_INDEPENDENT_RESUME_PROOF', result['status'])
        self.assertEqual(result['total_bytes'], sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file()))

    def test_missing_or_modified_optimizer_rejected(self):
        (self.root / 'optimizer.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'FULL_STATE_FILE_MISSING_OR_CHANGED'):
            self.inspect()

    def test_processor_is_required_for_full_resume_artifact(self):
        (self.root / 'processor/processor_config.json').unlink()
        with self.assertRaisesRegex(ValueError, 'PROCESSOR_CONFIG_MISSING'):
            self.inspect()

    def test_processor_stats_and_embodiment_are_not_optional(self):
        for name in ('statistics.json', 'embodiment_id.json'):
            with self.subTest(name=name):
                path = self.root / 'processor' / name
                content = path.read_bytes()
                path.unlink()
                with self.assertRaisesRegex(ValueError, 'PROCESSOR_CONFIG_MISSING'):
                    self.inspect()
                path.write_bytes(content)

    def test_saved_processor_cannot_refit_stats_or_change_route(self):
        self.put('processor/statistics.json', {'synthetic': 'changed'})
        with self.assertRaisesRegex(ValueError, 'PROCESSOR_STATISTICS_CHANGED'):
            self.inspect()
        self.put('processor/statistics.json', {'synthetic': True})
        self.put('processor/embodiment_id.json', {'new_embodiment': 3})
        with self.assertRaisesRegex(ValueError, 'PROCESSOR_EMBODIMENT_ROUTE_CHANGED'):
            self.inspect()

    def test_unregistered_file_never_enters_archive_inventory(self):
        (self.root / 'credentials.txt').write_text('dummy fixture, not a credential')
        with self.assertRaisesRegex(ValueError, 'UNREGISTERED_CHECKPOINT_FILE_REFUSED'):
            self.inspect()

    def test_processor_normalization_change_is_refused(self):
        path = self.root / 'processor/processor_config.json'
        value = json.loads(path.read_text())
        value['processor_kwargs']['use_percentiles'] = True
        self.put('processor/processor_config.json', value)
        with self.assertRaisesRegex(ValueError, 'SAVED_PROCESSOR_INTERFACE_CHANGED'):
            self.inspect()

    def test_cursor_is_consumed_not_prefetched(self):
        state = json.loads((self.root / 'SAMPLER_STATE.json').read_text())
        state['sampler']['cursor'] += 8
        self.put('SAMPLER_STATE.json', state)
        self.refresh_receipt()
        with self.assertRaisesRegex(ValueError, 'CONSUMED_SAMPLE_CURSOR_CHANGED'):
            self.inspect()

    def test_statistics_identity_rejected_even_with_resaved_file_hashes(self):
        self.put('statistics.json', {'synthetic': 'refitted'})
        self.refresh_receipt()
        with self.assertRaisesRegex(ValueError, 'FROZEN_STATISTICS_BYTES_CHANGED'):
            self.inspect()

    def test_sharded_index_must_match_every_actual_tensor(self):
        (self.root / 'model.safetensors').rename(self.root / 'model-00001-of-00002.safetensors')
        save_file({'second':np.ones(2, dtype=np.float32)}, self.root / 'model-00002-of-00002.safetensors')
        self.put('model.safetensors.index.json', dict(weight_map={
            'weight':'model-00001-of-00002.safetensors', 'second':'model-00002-of-00002.safetensors'}))
        self.assertEqual(len(self.inspect()['model_tensors']), 2)
        self.put('model.safetensors.index.json', dict(weight_map={
            'wrong_key':'model-00001-of-00002.safetensors', 'second':'model-00002-of-00002.safetensors'}))
        with self.assertRaisesRegex(ValueError, 'WEIGHT_KEY_OR_SHARD_MAPPING_CHANGED'):
            self.inspect()

    def test_missing_shard_and_symlink_rejected(self):
        self.put('model.safetensors.index.json', dict(weight_map={'weight':'missing.safetensors'}))
        with self.assertRaisesRegex(ValueError, 'SHARD_SET_INCOMPLETE'):
            self.inspect()
        (self.root / 'outside').symlink_to(self.root / 'statistics.json')
        with self.assertRaisesRegex(ValueError, 'CHECKPOINT_SYMLINK_REFUSED'):
            self.inspect()


class TestMainCapacity(unittest.TestCase):
    def setUp(self):
        recipe = dict(max_steps=4, effective_batch=32, workers=2, learning_rate=1e-4,
                      seed=29200, eval_strategy='no')
        self.inputs = dict(
            smoke_gate=dict(status='TRAINER_MODEL_INTERFACE_QUALIFIED'),
            smoke_runs=[dict(global_step=step, recipe=copy.deepcopy(recipe),
                frozen_backbone_unchanged=True, changed_trainable_parameters=['fixture']) for step in (4, 2, 4)],
            step_records=[dict(step=i, cursor=i*32, epoch=100+i*2) for i in range(1, 5)],
            wrapper=dict(status='SMOKE_COMPLETE_NO_MAIN_STARTED', steps=[
                dict(mode=mode, wall_s=50, exit=0) for mode in ('smoke-continuous', 'smoke-first', 'smoke-resume')]),
            ledger=dict(main_updates=0, model_smoke_updates=8, GPU_allocated_s=200,
                CPU_s=1000, CPU_upper_s=660, stages=[], PBS_submissions=[
                    dict(status='SUBMIT_FAILED_PRESERVED', GPU_reserved_s=0, CPU_reserved_s=0),
                    dict(status='RECONCILED_TERMINAL', GPU_reserved_s=0, CPU_reserved_s=0)]),
            checkpoint_bytes=26*GIB, owned_bytes=80*GIB, filesystem_free_bytes=1000*GIB,
            quota_free_lower_bound_bytes=150*GIB, extra_peak_bytes=8*GIB,
            now=2000, new_work_cutoff=82000)

    def plan(self):
        return calculate(**self.inputs)

    def test_actual_rate_plan_preserves_full_goal_and_does_not_authorize(self):
        result = self.plan()
        self.assertEqual(result['blockers'], [])
        self.assertFalse(result['authorizes_execution'])
        self.assertEqual(result['estimated_full_4096_main_seconds'], 10390)
        self.assertEqual(result['sample_exposures'], 131072)
        self.assertEqual(result['evaluation_reserve_seconds'], 10800)
        self.assertEqual(result['checkpoint_copies_in_forecast'], 2)
        self.assertEqual(result['required_free_headroom_bytes'], 80*GIB)
        self.assertTrue(result['main_and_eval_share_next_allocation'])

    def test_slow_throughput_does_not_shorten_training_or_eval(self):
        for record in self.inputs['step_records']:
            record['epoch'] = 100 + record['step'] * 5
        result = self.plan()
        self.assertIn('FULL_4096_EXCEEDS_MAIN_4_5_HOUR_SHARE', result['blockers'])
        self.assertIn('COMBINED_MAIN_EVAL_EXCEEDS_REMAINING_GPU_TIME', result['blockers'])
        self.assertEqual(result['main_optimizer_updates'], 4096)

    def test_real_quota_not_shared_df_controls_capacity(self):
        self.inputs['quota_free_lower_bound_bytes'] = 30*GIB
        self.assertIn('PERSONAL_QUOTA_FREE_MARGIN', self.plan()['blockers'])
        self.inputs['quota_free_lower_bound_bytes'] = None
        self.assertIn('PERSONAL_QUOTA_UNKNOWN_REQUIRES_SEPARATE_BOUNDED_WRITE_AND_OWNER_ADMISSION', self.plan()['blockers'])

    def test_deadline_and_existing_outputs_do_not_reset(self):
        self.inputs['new_work_cutoff'] = 3000
        self.inputs['owned_bytes'] = 179*GIB
        result = self.plan()
        self.assertIn('COMBINED_ALLOCATION_EXCEEDS_ORIGINAL_NEW_WORK_CUTOFF', result['blockers'])
        self.assertIn('OWNED_PEAK_EXCEEDS_180_GIB', result['blockers'])

    def test_pending_allocation_and_fourth_submission_refused(self):
        self.inputs['ledger']['PBS_submissions'][-1]['GPU_reserved_s'] = 1800
        with self.assertRaisesRegex(ValueError, 'NOT_RECONCILED'):
            self.plan()
        self.inputs['ledger']['PBS_submissions'].append(dict(status='SUBMIT_FAILED_PRESERVED'))
        with self.assertRaisesRegex(ValueError, 'NO_SUBMISSION_REMAINING'):
            self.plan()

    def test_no_main_from_missing_or_failed_smoke(self):
        self.inputs['smoke_gate']['status'] = 'MODEL_SMOKE_FAILED'
        with self.assertRaisesRegex(ValueError, 'ACTUAL_SMOKE_GATE_REQUIRED'):
            self.plan()

    def test_nonfinite_capacity_cannot_pass_comparisons(self):
        for key in ('owned_bytes', 'extra_peak_bytes', 'quota_free_lower_bound_bytes', 'now'):
            with self.subTest(key=key):
                inputs=copy.deepcopy(self.inputs)
                inputs[key]=float('nan')
                with self.assertRaises(ValueError):
                    calculate(**inputs)

    def test_wrong_sampler_count_or_warmup_observation_rejected(self):
        self.inputs['step_records'][1]['cursor'] = 512
        with self.assertRaisesRegex(ValueError, 'ACTUAL_EFFECTIVE_BATCH_32_REQUIRED'):
            self.plan()
        self.inputs['step_records'][1]['cursor'] = 64
        self.inputs['step_records'][1]['epoch'] = self.inputs['step_records'][0]['epoch']
        with self.assertRaisesRegex(ValueError, 'STEP_INTERVAL'):
            self.plan()


if __name__ == '__main__':
    unittest.main()
