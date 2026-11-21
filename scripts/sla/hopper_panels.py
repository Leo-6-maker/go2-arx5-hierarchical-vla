"""Replay sealed engineering and scientific panels using their original data and noise."""
from pathlib import Path

import numpy as np

from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_contract import DOMAINS, HORIZONS, canonical_hash
from go2arx5_vla.sla_adapter.scale20k_evaluation import bytes_sha, measure
from go2arx5_vla.sla_adapter.scale20k_training import prepare_host_batch
from scripts.sla import freeze_adapt5k_panels as original
from scripts.sla import freeze_adapt5k_performance_panel as engineering
from scripts.sla.evaluate_adapt5k import diagnostic_metrics, selections
from scripts.sla.hopper_inputs import inputs, relocate
from scripts.sla.hopper_parent_custody import digest

ENGINEERING_PANEL = '068b73058d28b741c9ddeba6bcc0997e5ffb1453ceb349a7e4c2baa5f2bf4f57'
FORMAL_PANEL = '3136404c06f890a9dd830b2cf5f40e487ef436ede16e805e14ffc0cb06b5eefd'
EVAL_INPUT = '4a99630a10f497417ddc1d1cf873a7b0339677b2f0b32c1f7b46937ec0c69e04'
PANEL_PATH = 'sla-adapt5k-gt-v2-batch-offline-panel-48ad2421-20261120-01/panel/OFFLINE_PANEL.private.json'


class FixedPanels:
    def __init__(self, train_assets, eval_assets):
        from openpi_client.image_tools import resize_with_pad
        from go2arx5_vla.sla_adapter.model_v1 import Tokenizer

        e = read_pin(dict(path=str(eval_assets/PANEL_PATH), sha256=ENGINEERING_PANEL))
        if e['eval_input']['sha256'] != EVAL_INPUT or e['original_panel']['sha256'] != FORMAL_PANEL:
            raise ValueError('Panel spec or population differs from the pre-registered endpoints')
        self.formal = read_pin(relocate(e['original_panel'], eval_assets))
        spec = read_pin(relocate(e['eval_input'], eval_assets))
        if (spec['panel'] != e['original_panel'] or self.formal['bindings'] != original.bindings(spec)
                or e['input_and_noise_sha256'] != canonical_hash(e['rows'])):
            raise ValueError('Fixed formal population or engineering sample order changed')
        dev = relocate(spec['original_source_dev'], eval_assets)
        dev['target'] = relocate(spec['original_source_dev']['target'], train_assets)
        moved = dict(spec, original_source_dev=dev, target_D1=relocate(spec['target_D1'], train_assets))
        self.datasets = original.datasets(moved, resize_with_pad)
        tokens = inputs(train_assets)['tokenizer']
        if digest(Path(tokens['path'])) != e['tokenizer_sha256'] or tokens['sha256'] != e['tokenizer_sha256']:
            raise ValueError('Panel tokenizer differs from the fixed training tokenizer')
        self.tokenizer = Tokenizer(tokens['path'])
        self.engineering = e['rows']
        for domain, chosen in engineering.selected_rows(self.formal).items():
            for row, (ordinal, original_row) in zip(self.engineering[domain], chosen, strict=True):
                if row['ordinal'] != ordinal or any(row[k] != v for k, v in original_row.items()):
                    raise ValueError('Engineering input does not match its original window')
        noise = relocate(e['noise'], eval_assets)
        if digest(Path(noise['path'])) != noise['sha256']:
            raise ValueError('Engineering noise bank changed')
        with np.load(noise['path'], allow_pickle=False) as archive:
            self.noise = {key: archive[key] for key in archive.files}

    def rows(self, mode):
        if mode == 'engineering':
            return {d: [(r['ordinal'], r) for r in self.engineering[d]] for d in DOMAINS}
        if mode in ('formal', 'formal-restore'):
            return selections(self.formal, mode)
        raise ValueError('Unexpected evaluation panel')

    def examples(self, mode):
        import jax

        for kind, rows in self.rows(mode).items():
            domain = 'target' if kind == 'target_phase' else kind
            dataset = self.datasets[domain]
            for ordinal, row in rows:
                if dataset.identity(row['index']) != row['identity']:
                    raise ValueError('Original evaluation window identity mismatch')
                host = prepare_host_batch(dataset, self.tokenizer, domain, (ordinal,), (row['index'],), training=False)
                shape = (1, HORIZONS[domain], 32)
                if mode == 'engineering':
                    noise = self.noise[row['noise_key']]
                    if (noise.shape != shape or bytes_sha(noise) != row['noise_sha256'] or
                            canonical_hash(engineering.input_hash(host)) != row['input_sha256']):
                        raise ValueError('Engineering observation or random input changed')
                else:
                    noise = jax.random.normal(jax.random.PRNGKey(row['noise_id']), shape)
                yield kind, domain, ordinal, row, host, noise

    def evaluate(self, mode, graphs, state, frozen, *, before_generation, after_generation, archive):
        import flax.nnx as nnx
        import jax
        from openpi.models import pi0
        from go2arx5_vla.sla_adapter import batch_v2, full_h40_v6
        from scripts.sla.mfree_c20_vectorfield_v3 import extracted_fields

        fields, _ = extracted_fields(batch_v2, pi0)
        samplers = {d: full_h40_v6.make_sampler(graphs[d], fields, rematerialize=False, precision='highest')
                    for d in DOMAINS}
        params = state['parameters']
        fitted = {d: nnx.State.merge(params['shared'], params[d]) for d in DOMAINS}
        for kind, domain, ordinal, row, host, noise in self.examples(mode):
            dataset = self.datasets[domain]
            obs, truth = batch_v2.device_batch(host)
            before_generation(kind, ordinal)
            prediction = np.asarray(samplers[domain](fitted[domain], frozen, obs, jax.device_put(noise)))
            if prediction.shape != noise.shape or not np.isfinite(prediction).all():
                raise ValueError('Incomplete or nonfinite model output')
            after_generation(kind, ordinal)
            physical, metrics = measure(domain, dataset, row, prediction[0], np.asarray(truth[0]), self.formal)
            if not np.isfinite(physical).all():
                raise ValueError('Nonfinite decoded output')
            if kind == 'target_phase' or (mode == 'engineering' and domain == 'target'):
                metrics['phase'] = diagnostic_metrics(dataset, row, physical, metrics)
            archive(kind, ordinal, row, prediction[0], physical, metrics)
