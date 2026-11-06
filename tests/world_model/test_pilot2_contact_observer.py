import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from tests.world_model.test_pilot2_contact_model import episode
from go2arx5_vla.world_model.pilot2_contact_model import replay
from go2arx5_vla.world_model.pilot2_supported import stage_for
from go2arx5_vla.world_model.pilot2_contact_observer import prepare, record


class Adapter:
    def __init__(self, root, config, step):
        self.output = root; self.spec = dict(source_sha='SYNTHETIC_OBSERVER_TEST_NOT_NATIVE', contact_forecast_observer=config)
        self.step = step; self.calls = 0; self.move = False
    def observe(self):
        self.calls += 1
        return dict(step=self.step+(self.calls-1 if self.move else 0))
    def install_reference(self, *args): raise AssertionError('READ_ONLY_OBSERVER_MUST_NOT_INSTALL_ACTIONS')
    def advance_to(self, *args): raise AssertionError('READ_ONLY_OBSERVER_MUST_NOT_INTEGRATE')


class ContactForecastObserver(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.ep = episode()
        for i, row in enumerate(self.ep['samples']): row['stage'] = stage_for(i, self.ep['episode'])
        data = replay([self.ep]); dim = data['history'].shape[1]+64; rng = np.random.default_rng(123)
        arrays = dict(mean=np.zeros(dim), scale=np.ones(dim), horizon_count=np.array(6))
        for index, shape in [(0, (8, dim)), (2, (8, 8)), (4, (60, 8))]:
            arrays[f'network.{index}.weight'] = rng.normal(0, .01, shape)
            arrays[f'network.{index}.bias'] = np.zeros(shape[0])
        self.path = self.root/'model.npz'; np.savez(self.path, **arrays)
        self.config = dict(schema='WM02_PILOT2_PAUSED_CONTACT_FORECAST_V1', anchors=[52, 80, 111],
            models=[dict(name='FROZEN_TEST', path=str(self.path), sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())])
    def tearDown(self): self.tmp.cleanup()
    def adapter(self, anchor=52): return Adapter(self.root, copy.deepcopy(self.config), self.ep['samples'][anchor]['native_end_step'])

    def test_forecast_uses_same_past_as_replay_and_does_not_advance(self):
        a = self.adapter(); self.assertIs(prepare(a, self.ep['episode']), record); self.assertEqual(a.calls, 0)
        point = record(a, self.ep['episode'], self.ep['samples'][:53]); data = replay([self.ep])
        i = next(i for i, v in enumerate(data['identities']) if v['anchor_native_step'] == a.step)
        x = np.r_[data['history'][i], data['plan'][i]].astype('<f4')
        self.assertEqual(point['input_sha256'], hashlib.sha256(x.tobytes()).hexdigest())
        self.assertEqual(point['native_step_before'], point['native_step_after'])
        self.assertFalse(point['future_measured_response_observed']); self.assertEqual(a.calls, 2)
        self.assertEqual(len((self.root/'CONTACT_FORECASTS.private.jsonl').read_text().splitlines()), 1)
        with self.assertRaisesRegex(FileExistsError, 'ALREADY_RECORDED'):
            record(a, self.ep['episode'], self.ep['samples'][:53])

    def test_future_achieved_observations_cannot_enter_forecast(self):
        a = self.adapter(); prepare(a, self.ep['episode']); first = record(a, self.ep['episode'], self.ep['samples'][:53])
        changed = copy.deepcopy(self.ep); changed['samples'][80]['TCP_m'][0] += 100
        other = self.root/'other'; other.mkdir(); b = Adapter(other, copy.deepcopy(self.config), a.step)
        prepare(b, changed['episode']); second = record(b, changed['episode'], changed['samples'][:53])
        self.assertEqual(first['input_sha256'], second['input_sha256']); self.assertEqual(first['predictions'], second['predictions'])

    def test_no_future_program_arm_keeps_same_history_and_only_current_command(self):
        a = self.adapter(); model = copy.deepcopy(a.spec['contact_forecast_observer']['models'][0])
        model.update(name='BLIND_TEST', arm='NO_FUTURE_PROGRAM'); a.spec['contact_forecast_observer']['models'].append(model)
        prepare(a, self.ep['episode']); point = record(a, self.ep['episode'], self.ep['samples'][:53]); data = replay([self.ep])
        i = next(i for i, v in enumerate(data['identities']) if v['anchor_native_step'] == a.step)
        x = np.r_[data['history'][i], data['blind_plan'][i]].astype('<f4')
        self.assertEqual(point['predictions']['BLIND_TEST']['input_sha256'], hashlib.sha256(x.tobytes()).hexdigest())
        self.assertNotEqual(point['predictions']['BLIND_TEST']['input_sha256'], point['predictions']['FROZEN_TEST']['input_sha256'])

    def test_changed_model_or_unknown_policy_program_is_rejected_before_physics(self):
        a = self.adapter(); self.path.write_bytes(self.path.read_bytes()+b'changed')
        with self.assertRaisesRegex(PermissionError, 'MODEL_CHANGED'): prepare(a, self.ep['episode'])
        self.assertEqual(a.calls, 0)
        policy = dict(self.ep['episode'], schema='WM02_PILOT2_SUPPORTED_POLICY_EPISODE_V1', purpose='SUPPORTED_POLICY_EVALUATION')
        with self.assertRaisesRegex(PermissionError, 'STATIC_CONTACT_PROGRAM_ONLY'): prepare(a, policy)
        self.assertFalse((self.root/'CONTACT_FORECASTS.private.jsonl').exists())

    def test_physics_movement_during_forecast_cannot_be_sealed(self):
        a = self.adapter(); prepare(a, self.ep['episode']); a.move = True
        with self.assertRaisesRegex(RuntimeError, 'PHYSICS_ADVANCED'): record(a, self.ep['episode'], self.ep['samples'][:53])
        self.assertFalse((self.root/'CONTACT_FORECASTS.private.jsonl').exists())

    def test_prefix_horizon_identity_must_match_logged_times(self):
        with np.load(self.path) as z: arrays = {k: z[k].copy() for k in z.files}
        arrays = {('prefix_'+k if k.startswith('network.') else k): v for k, v in arrays.items()}
        horizons = np.array([1, 2, 3, 4, 5, 6]); arrays.update(architecture_code=np.array(1),
            forecast_horizons=horizons, prefix_mask=(np.arange(64)[None] < horizons[:, None]).astype(np.float32))
        np.savez(self.path, **arrays); a = self.adapter()
        a.spec['contact_forecast_observer']['models'][0]['sha256'] = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, 'EXACT_LOGGED_FORECAST_HORIZONS'): prepare(a, self.ep['episode'])
        self.assertEqual(a.calls, 0)

    def test_safe_fallback_has_no_forecast_of_unexecuted_normal_program(self):
        a = self.adapter(111); prepare(a, self.ep['episode']); rows = copy.deepcopy(self.ep['samples'][:112]); rows[-1]['stage'] = 'EARLY_RELEASE'
        point = record(a, self.ep['episode'], rows)
        self.assertTrue(point['prefix_past_safe_fallback']); self.assertEqual(point['predictions'], {})
        self.assertEqual(point['skipped_reason'], 'PAST_SAFE_EARLY_RELEASE')
        self.assertFalse(point['horizon_within_declared_task'][-1])


if __name__ == '__main__': unittest.main()
