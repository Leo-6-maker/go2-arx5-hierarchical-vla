import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import json
import tempfile
import time
import numpy as np
import torch
from go2arx5_vla.world_model.pilot2_contact_neural import DirectContactWM, PrefixContactWM
from go2arx5_vla.world_model.pilot2_contact_numpy import predict, export_arrays


HORIZONS = (1, 4, 8, 16, 32, 64)


class ContactCommandPrefixes(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(104)
        self.model = PrefixContactWM(7, 64, HORIZONS, hidden=12)
        self.train = torch.randn(20, 71)
        self.model.fit_normalizer(self.train)
        self.arrays = export_arrays({k: v.detach().numpy().copy()
            for k, v in self.model.state_dict().items()}, 6)

    def test_matching_parameters_initialization_and_train_normalizer(self):
        torch.manual_seed(104); old = DirectContactWM(7, 64, 6, hidden=12)
        old.fit_normalizer(self.train)
        self.assertEqual(sum(x.numel() for x in old.parameters()), sum(x.numel() for x in self.model.parameters()))
        for name, value in old.named_parameters():
            self.assertTrue(torch.equal(value, dict(self.model.named_parameters())[name]))
        self.assertTrue(torch.equal(old.mean, self.model.mean))
        self.assertTrue(torch.equal(old.scale, self.model.scale))

    def test_later_commands_cannot_change_earlier_torch_or_numpy_forecasts(self):
        original = self.train[:4].clone()
        for j, horizon in enumerate(HORIZONS):
            with self.subTest(horizon=horizon):
                changed = original.clone(); changed[:, 7+horizon:] += 17.
                a, b = [self.model(x).detach().numpy() for x in (original, changed)]
                np.testing.assert_array_equal(a[:, j], b[:, j])
                a, b = [predict(self.arrays, x.numpy()) for x in (original, changed)]
                np.testing.assert_array_equal(a[:, j], b[:, j])

    def test_gradients_respect_the_forecast_prefix(self):
        for j, horizon in enumerate(HORIZONS):
            x = self.train[:2].clone().requires_grad_()
            grad = torch.autograd.grad(self.model(x)[:, j].sum(), x)[0]
            self.assertTrue(torch.equal(grad[:, 7+horizon:], torch.zeros_like(grad[:, 7+horizon:])))
            self.assertGreater(float(grad[:, 7:7+horizon].abs().max()), 0.)

    def test_commands_inside_the_prefix_remain_available(self):
        x = self.train[:2].clone(); changed = x.clone(); changed[:, 7] += 5.
        self.assertGreater(float((self.model(x)[:, 0]-self.model(changed)[:, 0]).abs().max().detach()), 1e-5)

    def test_numpy_matches_torch_and_legacy_reader_fails_closed(self):
        expected = self.model(self.train).detach().numpy()
        np.testing.assert_allclose(predict(self.arrays, self.train.numpy()), expected, rtol=1e-5, atol=1e-6)
        self.assertNotIn('network.0.weight', self.arrays)
        self.assertIn('prefix_network.0.weight', self.arrays)
        bad = {k: v.copy() for k, v in self.arrays.items()}; bad['prefix_mask'][0, -1] = 1
        with self.assertRaisesRegex(ValueError, 'EXACT_FORECAST_PREFIX'):
            predict(bad, self.train.numpy())
        bad = dict(self.arrays, architecture_code=np.array(0))
        with self.assertRaisesRegex(ValueError, 'EXACT_FORECAST_PREFIX'):
            predict(bad, self.train.numpy())

    def test_full_plan_counterexample_is_removed_without_changing_weights(self):
        old = DirectContactWM(1, 64, 6, hidden=2)
        with torch.no_grad():
            for p in old.parameters(): p.zero_()
            old.network[0].weight[0, -1] = 1
            old.network[2].weight[0, 0] = 1
            old.network[4].weight[7, 0] = 1
        new = PrefixContactWM(1, 64, HORIZONS, hidden=2)
        new.network.load_state_dict(old.network.state_dict())
        a = torch.zeros(1, 65); b = a.clone(); b[:, -1] = 1
        self.assertGreater(float((old(a)-old(b))[0, 0, 7].abs().detach()), .1)
        self.assertEqual(float((new(a)-new(b))[0, 0, 7].detach()), 0.)

    def test_invalid_or_reordered_horizons_are_rejected(self):
        for horizons in [(), (4, 1), (1, 1), (1, 65), (1., 4)]:
            with self.assertRaisesRegex(ValueError, 'DECLARED_FORECAST_PREFIXES'):
                PrefixContactWM(1, 64, horizons)

    def test_actual_worker_routes_prefix_recipe_and_closes_only_fresh_lease(self):
        from scripts.wm02.pilot1_neural_worker import main
        from go2arx5_vla.world_model.storage import digest
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); lease = root/'LEASE.json'; spec = root/'SPEC.json'
            spec.write_text(json.dumps(dict(registered_root=str(root), lease=str(lease),
                source_sha='SYNTHETIC_WORKER_ROUTE_TEST_NOT_SITE_ADMISSION',
                kind='CONTACT_WM', contact_recipe='WM02_PILOT2_PREFIX_CONTACT_V1',
                wall_limit_s=30, deadline_epoch=time.time()+60)))
            lease.write_text(json.dumps(dict(status='ACTIVE', run_uuid=root.name, spec_sha256=digest(spec))))
            with patch('scripts.wm02.pilot1_neural_worker.subprocess.run', return_value=SimpleNamespace(returncode=0)) as child:
                self.assertEqual(main(['--spec', str(spec)]), 0)
                self.assertEqual(child.call_args.args[0][3], 'scripts.wm02.train_pilot2_contact_wm')
                self.assertEqual(json.loads(lease.read_text())['status'], 'CLOSED')
                self.assertEqual(json.loads((root/'WORKER_EXIT.private.json').read_text())['exit_code'], 0)
                with self.assertRaisesRegex(PermissionError, 'EXACT_FRESH_PILOT1_WORKER_REQUIRED'):
                    main(['--spec', str(spec)])
                self.assertEqual(child.call_count, 1)


if __name__ == '__main__': unittest.main()
