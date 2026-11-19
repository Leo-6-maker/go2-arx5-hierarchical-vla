"""Matched D4 target continuation with a source-only gradient into shared LoRA."""
import hashlib
from pathlib import Path
import time

import numpy as np

from .fixed500_contract import (
    J_PLAN as PLAN,
    J_RECIPE as RECIPE,
    J_SCHEMA as SCHEMA,
    J_ARMS as ARMS,
    J_B_PACKAGE as B_PACKAGE,
    J_B_CHECKPOINT as B_CHECKPOINT,
    J_SOURCE_CHECKPOINT as SOURCE_CHECKPOINT,
    J_SOURCE_MANIFEST as SOURCE_MANIFEST,
    J_SOURCE_BANK as SOURCE_BANK,
    J_SOURCE_STATS as SOURCE_STATS,
    J_SOURCE_SEED as SOURCE_SEED,
    J_SOURCE_BATCH as SOURCE_BATCH,
    J_SOURCE_WEIGHT as SOURCE_WEIGHT,
    joint_package_contract as package_contract,
)
from . import batch_v2, multisource_d4, source_batch_v6
from .checkpoint_v1 import sha
from .config_v1 import SEED, trainable
from .model_v1 import configuration, flat, observation
from .multitask_initialization_v6 import shared_key
from .provider_multitask_v6 import export_payload
from .provider_v1 import active_rows
from .runtime_v1 import read_pin
from .source_native_v6 import SourceDataset
from .source_training_v6 import RECIPE as S1000_RECIPE



def source_dropout(cursor):
    return bool(np.random.Generator(np.random.PCG64(SOURCE_SEED + 1000000 + cursor)).random() < .25)


def combine_domain_gradients(target, source, shared_indices):
    import jax

    leaves, treedef = jax.tree.flatten(target)
    for index, src in zip(shared_indices, source, strict=True):
        leaves[index] = leaves[index]+SOURCE_WEIGHT*src
    return jax.tree.unflatten(treedef, leaves)


def load_rows(root, rows):
    for row in rows:
        file = root/row['file']
        if not file.resolve().is_relative_to(root.resolve()) or sha(file) != row['sha256']:
            raise ValueError('JOINT_PINNED_ADAPTER_ARRAY_CHANGED')
        array = np.load(file, allow_pickle=False)
        if list(array.shape) != row['shape'] or str(array.dtype) != row['dtype'] or not np.isfinite(array).all():
            raise ValueError('JOINT_ADAPTER_ARRAY_INCOMPATIBLE')
        yield array


class DomainSourceDataset(SourceDataset):
    def __init__(self, root, *, raw_root, **kwargs):
        super().__init__(root, SOURCE_MANIFEST, **kwargs)
        self.raw_root = Path(raw_root).resolve(strict=True) if raw_root else None
        self.checked_raw = set()
        if self.manifest['statistics']['sha256'] != SOURCE_STATS or self.manifest['TEST_labels_read']:
            raise ValueError('SEALED_SOURCE_TRAIN_STATS_AND_TEST_ISOLATION_REQUIRED')

    def raw_path(self, episode):
        original = Path(super().raw_path(episode))
        if self.raw_root is None:
            return original
        path = (self.raw_root/(episode+'.hdf5')).resolve(strict=True)
        if episode not in self.checked_raw:
            if not path.is_relative_to(self.raw_root) or sha(path) != self.episodes[episode]['raw_sha256']:
                raise ValueError('PORTABLE_SOURCE_BYTES_DIFFER_FROM_ORIGINAL')
            self.checked_raw.add(episode)
        return path


class SourceView:
    """H50 graph and original source-private I/O over the live shared target leaves."""
    def __init__(self, engine):
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp
        import optax
        from openpi.models import pi0
        from openpi_client.image_tools import resize_with_pad
        from scripts.sla.close_goal_g1_v1 import make_sampler
        from scripts.sla.mfree_c20_vectorfield_v3 import extracted_fields

        self.engine = engine
        self.spec = dict(engine.spec, dataset=engine.spec['source_dataset'])
        self.counts = engine.source_counts
        keys, structure = jax.tree_util.tree_flatten_with_path((engine.active,))
        names = [jax.tree_util.keystr(key) for key, _ in keys]
        self.shared_indices = tuple(i for i, name in enumerate(names) if shared_key(name))
        private_names = [name for name in names if not shared_key(name)]
        if len(self.shared_indices) != 20 or len(private_names) != 8:
            raise ValueError('JOINT_REQUIRES_EXACT_20_SHARED_AND_8_PRIVATE_LEAVES')
        pin = engine.spec['S1000_source_checkpoint']
        source_path = Path(pin['path'])
        receipt = read_pin(dict(path=str(source_path/'COMPLETE.json'), sha256=SOURCE_CHECKPOINT))
        if (pin['sha256'] != SOURCE_CHECKPOINT or (receipt['step'], receipt['cursor']) != (1000, 32000)
                or receipt['bindings']['recipe'] != S1000_RECIPE
                or receipt['bindings']['parent_manifest']['sha256'] != engine.spec['parent_manifest']['sha256']
                or receipt['bindings']['statistics_sha256'] != SOURCE_STATS
                or receipt['bindings']['source_manifest_sha256'] != SOURCE_MANIFEST):
            raise ValueError('ORIGINAL_S1000_SOURCE_PRIVATE_PARAMETERS_REQUIRED')
        source_rows, _ = active_rows(receipt, engine.active)
        by_name = {row['key']:row for row in source_rows}
        private_rows = [by_name[name] for name in private_names]
        self.private = tuple(jax.device_put(v) for v in load_rows(source_path, private_rows))
        self.private_pin = tuple((row['key'], row['sha256']) for row in private_rows)
        self.private_fingerprint = self.fingerprint_private()
        self.dataset = DomainSourceDataset(self.spec['dataset'], raw_root=self.spec.get('source_raw_root'),
                                           split='TRAIN', resize_with_pad=resize_with_pad)
        self.dev = DomainSourceDataset(self.spec['dataset'], raw_root=self.spec.get('source_raw_root'),
                                       split='SOURCE_DEV', resize_with_pad=resize_with_pad)
        self.tokenizer = source_batch_v6.SourceTokenizer(engine.spec['tokenizer']['path'])
        visit_file = Path(engine.spec['source_visit_bank'])
        if sha(visit_file) != SOURCE_BANK:
            raise ValueError('ORIGINAL_FIXED_FOUR_THOUSAND_SOURCE_VISITS_REQUIRED')
        self.visits = np.load(visit_file, allow_pickle=False)
        if (self.visits.shape != (4000,) or not np.issubdtype(self.visits.dtype, np.integer)
                or self.visits.min() < 0 or self.visits.max() >= len(self.dataset)):
            raise ValueError('SOURCE_VISIT_BANK_OUTSIDE_TRAIN')
        self.rng = jax.device_put(jax.random.PRNGKey(SOURCE_SEED))
        self.draw = jax.jit(source_batch_v6.draws)
        abstract = nnx.eval_shape(configuration(action_horizon=50).create, jax.random.PRNGKey(SEED))
        graph, source_active, source_frozen = nnx.split(abstract, nnx.All(nnx.Param, lambda path, _: trainable(path)), ...)
        for abstract_state, live_state in ((source_active, engine.active), (source_frozen, engine.frozen)):
            shape_keys = lambda state: {p: tuple(v.shape) for p, v in flat(state.to_pure_dict()).items()}
            if shape_keys(abstract_state) != shape_keys(live_state):
                raise ValueError('H40_AND_H50_GRAPH_STATE_SCHEMA_INCOMPATIBLE')

        def source_active_state(shared, source_private):
            shared, private = iter(shared), iter(source_private)
            leaves = [next(shared) if shared_key(name) else next(private) for name in names]
            return jax.tree_util.tree_unflatten(structure, leaves)[0]

        fields, _ = extracted_fields(batch_v2, pi0)

        def objective(shared, source_private, frozen, obs, actions, noise, tau):
            active = source_active_state(shared, source_private)
            with jax.default_matmul_precision('highest'):
                t = tau[:,None,None]
                field = fields['train_field'](nnx.merge(graph, active, frozen), obs, t*noise+(1-t)*actions, tau)
                error = (field-(noise-actions))**2
                return jnp.mean(error), (jnp.mean(error[...,:14]), jnp.mean(error[...,14:]))

        self.grad = jax.jit(jax.value_and_grad(objective, has_aux=True))
        self.metrics = jax.jit(lambda a, b, g: jnp.r_[(a+b)/2, optax.global_norm(g)])
        self.mean = jax.jit(lambda left, right: jax.tree.map(lambda a,b:(a+b)/2, left, right))

        self.combine = jax.jit(lambda target, source: combine_domain_gradients(target, source, self.shared_indices),
                               donate_argnums=(0,))
        sampler = make_sampler(graph, fields, cached=True, precision=None)
        self.infer = lambda shared, private, frozen, obs, noise: sampler(
            source_active_state(shared, private), frozen, obs, noise)[0]

    def fingerprint_private(self):
        return [hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for v in self.private]

    def shared(self):
        import jax

        active = jax.tree.leaves(self.engine.active)
        return tuple(active[i] for i in self.shared_indices)

    def gradient(self, cursor):
        import jax.numpy as jnp
        from .batch_v2 import device_batch

        noise, tau = self.draw(self.rng, np.arange(cursor, cursor+SOURCE_BATCH, dtype=np.uint32))
        shared = self.shared()
        parts, gradients, logs = [], [], []
        for offset in (0, 4):
            if time.time() >= self.engine.spec['deadline_epoch']-120:
                raise TimeoutError('FINITE_J_SOURCE_BEFORE_UPDATE_COMMIT')
            host = source_batch_v6.host_batch(self.dataset, self.tokenizer, self.visits, cursor+offset, 4,
                                             dropper=source_dropout)
            self.counts['attempted_training_visits'] += 4
            self.counts['backward_attempts'] += 1
            obs, actions = device_batch(host)
            ((loss, detail), grad) = self.grad(shared, self.private, self.engine.frozen, obs, actions,
                                              noise[offset:offset+4], tau[offset:offset+4])
            parts.append(jnp.stack((loss, *detail)))
            gradients.append(grad)
            logs.append(dict(indices=host['indices'], first_absolute_cursor=cursor+offset,
                             state_token_dropped=host['state_token_dropped']))
        averaged = self.mean(*gradients)
        metrics = np.asarray(self.metrics(*parts, averaged))
        if not np.isfinite(metrics).all() or metrics[-1] <= 0:
            raise ValueError('REAL_FINITE_NONZERO_SOURCE_SHARED_GRADIENT_REQUIRED')
        self.counts['completed_training_visits'] += SOURCE_BATCH
        self.counts['completed_backwards'] += 2
        return averaged, dict(micro=logs, FM32=float(metrics[0]), FM14=float(metrics[1]),
                              padding18=float(metrics[2]), shared_gradient_norm=float(metrics[3]),
                              source_gradient_on_target_IO_leaves=0)

    def generate(self, index, noise_id, *, dataset):
        import jax

        obs, _ = observation(dataset[index], self.tokenizer)
        noise = jax.random.normal(jax.random.PRNGKey(noise_id), (1, 50, 32))
        self.counts['ordinary_RGB_generation_attempts'] += 1
        normalized = np.asarray(self.infer(self.shared(), self.private, self.engine.frozen, obs, noise))
        if normalized.shape != (1, 50, 32) or not np.isfinite(normalized).all():
            raise ValueError('FINITE_FULL_SOURCE_H50_REQUIRED')
        self.counts['ordinary_RGB_generations'] += 1
        return normalized, dataset.statistics['actions'].unnormalize(normalized[0,:,:14])


class Engine(multisource_d4.TransferEngine):
    recipe = RECIPE
    arms = ARMS
    source_arm = None

    def __init__(self, spec, output):
        import jax

        package = spec['B500_package']
        root = Path(package['path'])
        original = read_pin(dict(path=str(root/'MANIFEST.private.json'), sha256=B_PACKAGE))
        multisource_d4.package_contract(original)
        if (spec['plan_commit'] != PLAN or package['sha256'] != B_PACKAGE
                or original['original_complete_sha256'] != B_CHECKPOINT or original['bindings']['arm'] != 'D4-B'):
            raise ValueError('EXACT_TARGET_B500_ALL_PARAMETER_START_REQUIRED')
        super().__init__(spec, output)
        prior = original['bindings']
        common = ('tokenizer', 'target_audit', 'visit_map_sha256', 'lr_sha256', 'seed', 'frozen_fingerprint',
                  'upstream_sha', 'initial_function_rule', 'XLA_FLAGS', 'optimizer', 'microbatch', 'accumulation_steps',
                  'effective_batch', 'random_stream', 'sample_cursor', 'training_matmul_precision', 'aux_samples_per_update',
                  'full_sampling_steps', 'derived_statistics_sha256', 'dataset_manifest_sha256', 'full_XYZ_lambda',
                  'full_gripper_speed_lambda')
        if (any(self.bindings[k] != prior[k] for k in common)
                or self.bindings['parent_manifest']['sha256'] != prior['parent_manifest']['sha256']):
            raise ValueError('JOINT_REQUIRES_ORIGINAL_MATCHED_D4_TRAINING_CONFIGURATION')
        rows, structure = active_rows(dict(arrays=original['adapter_arrays']), self.active)
        self.active, = jax.tree_util.tree_unflatten(structure, [jax.device_put(v) for v in load_rows(root, rows)])
        self.opt_state = self.tx.init(self.active)
        self.source_counts = dict(attempted_training_visits=0, completed_training_visits=0, backward_attempts=0,
                                  completed_backwards=0, ordinary_RGB_generation_attempts=0, ordinary_RGB_generations=0)
        self.source = SourceView(self)
        source_noise, source_flow = self.source.draw(self.source.rng, np.arange(SOURCE_BATCH, dtype=np.uint32))
        first_source_draw = {name: hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
                             for name, value in (('noise', source_noise), ('flow_time', source_flow))}
        self.bindings.update(B500_package_sha256=B_PACKAGE, B500_checkpoint_sha256=B_CHECKPOINT,
                             S1000_checkpoint_sha256=SOURCE_CHECKPOINT, joint_source_MANIFEST_sha256=SOURCE_MANIFEST,
                             joint_source_statistics_sha256=SOURCE_STATS, joint_source_visit_bank_sha256=SOURCE_BANK,
                             joint_source_rng_seed=SOURCE_SEED, joint_source_batch=self.source_batch,
                             joint_source_weight=SOURCE_WEIGHT if self.source_batch else 0.,
                             joint_source_frozen_leaves=dict(self.source.private_pin),
                             joint_source_initial_draw_sha256=first_source_draw,
                             joint_source_dropout_sha256=hashlib.sha256(bytes(source_dropout(i) for i in range(4000))).hexdigest())
        self.initial_match = dict(B500_adapter_leaves=len(rows), source_frozen_adapter_leaves=len(self.source.private),
                                  source_shared_adapter_leaves=len(self.source.shared_indices), fresh_Adam=True,
                                  source_private_on_device_sha256=self.source.private_fingerprint,
                                  source_first_draw_sha256=first_source_draw)

    @property
    def source_batch(self):
        return SOURCE_BATCH if self.spec['arm'] == 'J1' else 0

    @property
    def source_visits(self):
        return self.step*self.source_batch

    def combine_gradient(self, accumulator):
        if not self.source_batch:
            return accumulator, {}
        source, metrics = self.source.gradient(self.source_visits)
        if (time.time() >= self.spec['deadline_epoch']-120
                or (Path(self.spec['campaign_root'])/'STOP.json').exists()):
            raise TimeoutError('FINITE_J_BEFORE_JOINT_OPTIMIZER_COMMIT')
        return self.source.combine(accumulator, source), dict(source=metrics, source_cursor=self.source_visits+SOURCE_BATCH)

    def match_initial(self, initial, complete):
        if not self.source_batch:
            return
        root = Path(self.spec['campaign_root'])/'J0'
        from scripts.pi0292.durable import read
        result = read(root/'RESULT.private.json')
        if (result['exit_code'] or not read(root/'RESTORE.private.json')['passed']
                or not read(root/'SOURCE_RESTORE.private.json')['passed']):
            raise ValueError('MATCHED_J0_ENDPOINT_AND_BOTH_RESTORES_REQUIRED')
        original = read(root/'INITIAL.private.json')
        pin = original['checkpoint']
        if pin != result['initial_checkpoint']:
            raise ValueError('J0_STEP_ZERO_IDENTITY_CHANGED_AFTER_SEAL')
        control = read_pin(dict(path=str(Path(pin['path'])/'COMPLETE.json'), sha256=pin['sha256']))
        fields = ('initial_noise_sha256', 'initial_flow_time_sha256', 'first32', 'visit_map_sha256', 'historical_match')
        if (control['bindings']['arm'] != 'J0' or control['bindings']['goal_plan_sha'] != PLAN
                or (control['step'], control['cursor']) != (0, 0)
                or control['arrays'] != complete['arrays']
                or any(original[key] != initial[key] for key in fields)):
            raise ValueError('J0_J1_FRESH_ADAM_RNG_AND_ALL_MODEL_BYTES_MUST_MATCH')
        initial['matched_J0_state_array_count'] = len(control['arrays'])


def export(package, checkpoint, *, engine):
    if (engine.source_visits != engine.source_batch * 500
            or engine.source_counts['completed_training_visits'] != engine.source_visits):
        raise ValueError('SEALED_J_ENDPOINT_REQUIRES_ACTUAL_SOURCE_OPTIMIZATION_VISITS')
    stats = engine.dataset.root/engine.dataset.manifest['statistics']['file']
    return export_payload(package, checkpoint, engine=engine, statistics=stats,
                          schema=SCHEMA, validate=package_contract, step=500, cursor=16000)


class Provider(multisource_d4.Provider):
    def validate_package_contract(self, manifest):
        package_contract(manifest)
