"""Single-device micro1 x accumulation32 using the pinned model and AdamW."""
import hashlib
import json
from pathlib import Path
import time
import os
import numpy as np

from .checkpoint_v1 import save, restore, sha
from .codec_v1 import Quantiles
from .config_v1 import SEED, learning_rates, visits
from .dataset_v1 import TargetDataset
from .model_v1 import load_parent, observation, Tokenizer, frozen_fingerprint


def read_pin(pin):
    p = Path(pin['path'])
    if sha(p) != pin['sha256']:
        raise ValueError('SLA_RUNTIME_INPUT_PIN_CHANGED')
    return json.loads(p.read_bytes())


def data(spec):
    from openpi_client.image_tools import resize_with_pad
    q = read_pin(spec['target_audit'])
    if q.get('schema') != 'SLA_TARGET_CODEC_DATA_ACTUAL_V1' or not q['passes'] or q['expert_windows'] != 1450:
        raise ValueError('SLA_ALL_WINDOW_DATA_QUALIFICATION_REQUIRED')
    root = Path(spec['target_audit']['path']).parent
    stats = read_pin(dict(path=str(root/'STATISTICS.private.json'), sha256=q['statistics_sha256']))
    frames = read_pin(dict(path=str(root/'FRAMES.private.json'), sha256=q['frames_sha256']))
    d = TargetDataset(spec['dataset'], spec['camera_audit'], frames['episodes'], resize_with_pad=resize_with_pad,
                      statistics={k: Quantiles(stats[k]['q01'], stats[k]['q99']) for k in ('state', 'actions')})
    if d.original.manifest_sha != q['source_dataset_sha256'] or sha(spec['tokenizer']['path']) != spec['tokenizer']['sha256']:
        raise ValueError('SLA_EXACT_DATASET_AND_TOKENIZER_REQUIRED')
    return d


def probe_indices(d):
    """Eight fixed TRAIN examples, selected by stage before any model output."""
    wanted = ['PREGRASP', 'DESCEND', 'CLOSE', 'LIFT', 'HOLD', 'PREGRASP', 'DESCEND', 'CLOSE']
    result = []
    for phase in wanted:
        for i, (ep, anchor) in enumerate(d.original.rows):
            if i not in result and ep['expert_stage_audit_only'][anchor] == phase:
                result.append(i)
                break
    if len(result) != 8:
        raise ValueError('FIXED_EIGHT_TRAIN_PROBES_MUST_COVER_ALL_PHASES')
    return result


class Engine:
    def dataset_for_spec(self, spec):
        return data(spec)

    def visits_for_dataset(self, dataset):
        return visits(dataset.original.manifest)

    def learning_rate_values_for_spec(self, spec):
        return learning_rates()

    def __init__(self, spec, output):
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp
        import optax
        from openpi.training.optimizer import AdamW
        from .fm_v1 import loss

        if len(jax.devices()) != 1 or jax.devices()[0].platform != 'gpu':
            raise ValueError('ONE_ACTUAL_CUDA_DEVICE_REQUIRED_NO_CPU_TRAINING')
        rule = read_pin(spec['initial_function_rule'])
        if os.environ.get('XLA_FLAGS') != rule['XLA_FLAGS']:
            raise ValueError('REGISTERED_SLA_DETERMINISTIC_COMPUTE_MODE_REQUIRED')
        jax.config.update('jax_compilation_cache_dir', spec['compilation_cache_dir'])
        self.spec, self.output = spec, Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.counts = dict(actual_parent_loads=0, forward_backward_attempts=0, completed_forward_backward=0,
                           optimizer_updates=0, attempted_training_visits=0, completed_training_visits=0,
                           ordinary_RGB_generation_attempts=0, ordinary_RGB_generations=0, physical_requests=0)
        asset = read_pin(spec['parent_manifest'])
        if (asset.get('repo') != 'haofuly/SLA_base' or not asset.get('all_files_verified')
                or asset.get('revision') != '5bbdf3012ab643554aa2eb83ddeae9ace3dce476'
                or Path(spec['parent_params']).resolve() != (Path(spec['parent_manifest']['path']).parent/'model/params').resolve()):
            raise ValueError('EXACT_VERIFIED_SLA_BASE_PARENT_REQUIRED')
        self.dataset = self.dataset_for_spec(spec)
        self.tokenizer = Tokenizer(spec['tokenizer']['path'])
        self.visit_map = self.visits_for_dataset(self.dataset)
        self.learning_rate_values = self.learning_rate_values_for_spec(spec)
        self.step = self.cursor = 0
        self.rng = jax.device_put(jax.random.PRNGKey(SEED))
        started = time.time()
        self.graph, self.active, self.frozen, self.inventory = load_parent(spec['parent_params'])
        self.counts['actual_parent_loads'] += 1
        self.load_wall_s = time.time()-started
        (self.output/'PARAMETERS.private.json').write_text(json.dumps(self.inventory, indent=2))
        self.initial_frozen = self.inventory['frozen_fingerprint']
        source = Path(spec['source_root'])
        adapter_source = {str(p.relative_to(source)): sha(p) for p in sorted((source/'src/go2arx5_vla/sla_adapter').glob('*.py'))}
        self.bindings = dict(parent_manifest=spec['parent_manifest'], tokenizer=spec['tokenizer']['sha256'],
                             target_audit=spec['target_audit']['sha256'], visit_map_sha256=hashlib.sha256(self.visit_map.tobytes()).hexdigest(),
                             lr_sha256=hashlib.sha256(''.join(v.hex()+'\n' for v in self.learning_rate_values).encode('ascii')).hexdigest(),
                             seed=SEED, frozen_fingerprint=self.initial_frozen,
                             upstream_sha=spec['upstream_sha'], codec_version='SLA_NATIVE_TASK_EEF7_H40_V1',
                             adapter_source=adapter_source,
                             initial_function_rule=spec['initial_function_rule']['sha256'], XLA_FLAGS=rule['XLA_FLAGS'],
                             recipe='S1_FM32_MICRO1_ACCUM32_DROPOUT_TOKEN_ONLY_V1',
                             optimizer=dict(b1=.9, b2=.95, eps=1e-8, weight_decay=1e-5, clip_norm=1.))
        rate = jnp.asarray(self.learning_rate_values, dtype=jnp.float32)
        self.tx = AdamW(weight_decay=1e-5, clip_gradient_norm=1.).create(lambda count: rate[jnp.minimum(count, len(rate)-1)])
        self.opt_state = self.tx.init(self.active)
        graph = self.graph

        def objective(active, frozen, rng, obs, actions):
            model = nnx.merge(graph, active, frozen)
            # Differentiate the upstream implementation itself. The separate
            # dimensional diagnostic never supplies optimizer gradients.
            return jnp.mean(model.compute_loss(rng, obs, actions, train=False))

        self.grad = jax.jit(jax.value_and_grad(objective))
        self.diagnostics = jax.jit(lambda active, frozen, rng, obs, actions:
                                   loss(nnx.merge(graph, active, frozen), rng, obs, actions))
        self.accumulate = jax.jit(lambda acc, grad: jax.tree.map(lambda a, g: a+g/32, acc, grad), donate_argnums=(0,))

        def apply(active, optimizer, gradients):
            updates, optimizer = self.tx.update(gradients, optimizer, active)
            return optax.apply_updates(active, updates), optimizer, optax.global_norm(gradients)

        self.apply = jax.jit(apply, donate_argnums=(0, 1))
        self.infer = jax.jit(lambda active, frozen, obs, noise:
                             nnx.merge(graph, active, frozen).sample_actions(jax.random.PRNGKey(0), obs, noise=noise, num_steps=10))

    def checkpoint(self, name):
        return save(self.output/name, (self.active, self.opt_state, self.rng), step=self.step, cursor=self.cursor, bindings=self.bindings)

    def resume(self, pin):
        if sha(Path(pin['path'])/'COMPLETE.json') != pin['sha256']:
            raise ValueError('SLA_RESUME_COMPLETE_IDENTITY_CHANGED')
        (self.active, self.opt_state, self.rng), self.step, self.cursor = restore(
            pin['path'], (self.active, self.opt_state, self.rng), bindings=self.bindings)

    def generate(self, index, noise_id):
        import jax
        item = self.dataset[index]
        obs, _ = observation(item, self.tokenizer)
        noise = jax.random.normal(jax.random.PRNGKey(2026111200+noise_id), (1, 40, 32))
        self.counts['ordinary_RGB_generation_attempts'] += 1
        result = self.infer(self.active, self.frozen, obs, noise)
        jax.block_until_ready(result)
        normalized = np.asarray(result)
        if normalized.shape != (1, 40, 32) or not np.isfinite(normalized).all():
            raise ValueError('ORDINARY_SLA_FULL_FINITE_H40_32_REQUIRED')
        active7 = self.dataset.statistics['actions'].unnormalize(normalized[0, :, :7])
        state, *_ = self.dataset.original.raw(index)
        identity = self.dataset.original.identity(index)
        from .codec_v1 import pad32
        physical = self.dataset.codecs[identity['episode_id']].decode(state, pad32(active7))
        self.counts['ordinary_RGB_generations'] += 1
        return normalized, physical

    def update(self):
        import jax
        import jax.numpy as jnp
        from openpi.training.utils import train_rng
        start = time.time()
        accumulator = jax.tree.map(jnp.zeros_like, self.active)
        logs = []
        for micro in range(32):
            cursor = self.cursor+micro
            index = int(self.visit_map[cursor])
            drop = bool(np.random.Generator(np.random.PCG64(SEED+1000000+cursor)).random() < .25)
            obs, action = observation(self.dataset[index], self.tokenizer, drop_state=drop)
            key = train_rng(self.rng, self.step, micro, 32)
            self.counts['forward_backward_attempts'] += 1
            self.counts['attempted_training_visits'] += 1
            loss, gradient = self.grad(self.active, self.frozen, key, obs, action)
            jax.block_until_ready(gradient)
            self.counts['completed_forward_backward'] += 1
            self.counts['completed_training_visits'] += 1
            if not np.isfinite(float(loss)):
                raise ValueError('NONFINITE_SLA_FM_LOSS')
            active_loss = padding_loss = None
            if micro == 0 and (self.step < 2 or (self.step+1) % 50 == 0):
                self.counts['diagnostic_forward_attempts'] = self.counts.get('diagnostic_forward_attempts', 0)+1
                diagnostic_loss, (active, padding) = self.diagnostics(self.active, self.frozen, key, obs, action)
                jax.block_until_ready(diagnostic_loss)
                self.counts['diagnostic_forward_calls'] = self.counts.get('diagnostic_forward_calls', 0)+1
                self.counts['diagnostic_TRAIN_visits'] = self.counts.get('diagnostic_TRAIN_visits', 0)+1
                if abs(float(diagnostic_loss)-float(loss)) > read_pin(self.spec['initial_function_rule'])['diagnostic_forward_decomposition_max_loss_abs']:
                    raise ValueError('FM_DIMENSION_DIAGNOSTIC_NOT_NUMERICALLY_EQUIVALENT')
                active_loss, padding_loss = float(active), float(padding)
            if micro == 0 and self.step < 2:
                norm = jax.tree.map(lambda v: jnp.linalg.norm(v), gradient).to_pure_dict()
                from .model_v1 import flat
                logs.append(dict(first_micro_gradient_norms={('/'.join(p)): float(v) for p, v in flat(norm).items()}))
            accumulator = self.accumulate(accumulator, gradient)
            logs.append(dict(loss=float(loss), active7=active_loss, padding25=padding_loss, drop_state=drop))
        self.active, self.opt_state, norm = self.apply(self.active, self.opt_state, accumulator)
        jax.block_until_ready(self.active)
        self.counts['optimizer_updates'] += 1
        self.step += 1
        self.cursor += 32
        row = dict(step=self.step, cursor=self.cursor, used_lr=self.learning_rate_values[self.step-1],
                   next_lr=self.learning_rate_values[self.step] if self.step < len(self.learning_rate_values) else None,
                   grad_norm_before_clip=float(norm), micro=logs, wall_s=time.time()-start,
                   GPU_memory=jax.devices()[0].memory_stats())
        with (self.output/'updates.private.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        return row

    def residency(self):
        import jax
        leaves = jax.tree.leaves((self.active, self.opt_state))
        if not all(hasattr(a, 'devices') and all(d.platform == 'gpu' for d in a.devices()) for a in leaves):
            raise ValueError('SLA_TRAINING_STATE_NOT_GPU_RESIDENT')
        return dict(training_state_leaves=len(leaves), all_GPU_resident=True, CPU_training_activation_offload=False,
                    GPU_memory=jax.devices()[0].memory_stats())

    def verify_frozen(self):
        actual = frozen_fingerprint(self.frozen)
        if actual != self.initial_frozen:
            raise ValueError('FROZEN_SLA_DENSE_PARAMETERS_CHANGED')
        return actual
