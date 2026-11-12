"""Ordinary RGB SLA inference: parent + all deployed adapters + saved assets.

No optimizer, training feature bank, target dataset reader or spatial teacher
is loaded by Provider. Raw H40 is returned before the existing endpoint basis.
"""
import json
from pathlib import Path
import shutil
import numpy as np

from .checkpoint_v1 import sha
from .codec_v1 import Codec, Quantiles, pad32
from .dataset_v1 import images224
from .model_v1 import load_parent, observation, Tokenizer


def active_rows(manifest, template):
    import jax
    leaves, structure = jax.tree_util.tree_flatten_with_path((template,))
    rows = manifest['arrays'][:len(leaves)]
    if len(rows) != len(leaves):
        raise ValueError('DEPLOYMENT_ADAPTER_LEAVES_MISSING')
    for (key, value), row in zip(leaves, rows):
        if (jax.tree_util.keystr(key) != row['key'] or list(value.shape) != row['shape']
                or str(value.dtype) != row['dtype']):
            raise ValueError('DEPLOYMENT_ADAPTER_TREE_MISMATCH')
    if len(manifest['arrays']) > len(rows) and manifest['arrays'][len(rows)]['key'].startswith('[0]'):
        raise ValueError('DEPLOYMENT_WOULD_DROP_A_TRAINABLE_PARAMETER')
    return rows, structure


def export(package, checkpoint, *, active_template, statistics, tokenizer, parent_manifest, target_audit):
    """Cold export only; preserves the full training checkpoint separately."""
    package, checkpoint = Path(package), Path(checkpoint)
    complete = json.loads((checkpoint/'COMPLETE.json').read_bytes())
    if complete['schema'] != 'SLA_COMPLETE_PEFT_STATE_V1' or complete['step'] not in (4, 250, 500):
        raise ValueError('COMPLETE_QUAL4_OR_FIXED_CANDIDATE_ENDPOINT_REQUIRED')
    if complete['bindings']['parent_manifest'] != parent_manifest or sha(tokenizer) != complete['bindings']['tokenizer']:
        raise ValueError('DEPLOYMENT_PARENT_OR_TOKENIZER_CHANGED')
    if sha(target_audit['path']) != target_audit['sha256'] or target_audit['sha256'] != complete['bindings']['target_audit']:
        raise ValueError('DEPLOYMENT_ORIGINAL_TARGET_AUDIT_CHANGED')
    audited = json.loads(Path(target_audit['path']).read_bytes())
    if sha(statistics) != audited['statistics_sha256']:
        raise ValueError('DEPLOYMENT_ORIGINAL_SLA_STATISTICS_CHANGED')
    rows, _ = active_rows(complete, active_template)
    package.mkdir(parents=True, exist_ok=False)
    (package/'adapters').mkdir()
    copied = []
    for row in rows:
        file = checkpoint/row['file']
        if sha(file) != row['sha256']:
            raise ValueError('DEPLOYMENT_TRAINED_PARAMETER_CHANGED')
        shutil.copyfile(file, package/'adapters'/row['file'])
        copied.append(dict(row, file='adapters/'+row['file']))
    shutil.copyfile(statistics, package/'STATISTICS.json')
    shutil.copyfile(tokenizer, package/'paligemma_tokenizer.model')
    value = dict(schema='SLA_ORDINARY_RGB_INFERENCE_PACKAGE_V1', step=complete['step'], cursor=complete['cursor'],
                 qualification_only=complete['step'] == 4, parent_manifest=parent_manifest,
                 original_complete_sha256=sha(checkpoint/'COMPLETE.json'), bindings=complete['bindings'],
                 adapter_arrays=copied, statistics_sha256=sha(package/'STATISTICS.json'),
                 tokenizer_sha256=sha(package/'paligemma_tokenizer.model'),
                 camera_keys=['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb'], image_masks=[True, True, False],
                 image_size=[224, 224], sampler_steps=10, action_horizon=40, label_interval_s=.05,
                 codec='EPISODE_FIXED_MOUNT_LEFT_SO3_RELATIVE_EEF7', endpoint_basis_applied=False,
                 teacher_required=False, training_cache_required=False, optimizer_required=False,
                 physical_execution_admitted=False)
    (package/'MANIFEST.private.json').write_text(json.dumps(value, indent=2))
    return value


class Provider:
    def __init__(self, package, *, manifest_sha256, parent_params):
        import flax.nnx as nnx
        import jax
        from openpi_client.image_tools import resize_with_pad
        self.package = Path(package)
        if sha(self.package/'MANIFEST.private.json') != manifest_sha256:
            raise ValueError('DEPLOYMENT_PACKAGE_MANIFEST_PIN_CHANGED')
        self.manifest = json.loads((self.package/'MANIFEST.private.json').read_bytes())
        m = self.manifest
        if m.get('schema') != 'SLA_ORDINARY_RGB_INFERENCE_PACKAGE_V1' or m['teacher_required'] or m['training_cache_required']:
            raise ValueError('TEACHER_FREE_ORDINARY_RGB_PACKAGE_REQUIRED')
        if (sha(self.package/'STATISTICS.json') != m['statistics_sha256']
                or sha(self.package/'paligemma_tokenizer.model') != m['tokenizer_sha256']):
            raise ValueError('DEPLOYMENT_STATISTICS_OR_TOKENIZER_MISMATCH')
        # Statistics are copied from the exact audit bound to the checkpoint,
        # rather than refitted, inferred from GR00T or accepted by shape alone.
        parent = Path(m['parent_manifest']['path'])
        if sha(parent) != m['parent_manifest']['sha256']:
            raise ValueError('DEPLOYMENT_PARENT_MANIFEST_MISMATCH')
        original = json.loads(parent.read_bytes())
        if original.get('repo') != 'haofuly/SLA_base' or original.get('revision') != '5bbdf3012ab643554aa2eb83ddeae9ace3dce476' or not original.get('all_files_verified'):
            raise ValueError('DEPLOYMENT_ORIGINAL_SLA_BASE_REQUIRED')
        if Path(parent_params).resolve() != (parent.parent/'model/params').resolve():
            raise ValueError('DEPLOYMENT_FIXED_PARENT_PATH_REQUIRED')
        self.graph, self.active, self.frozen, self.inventory = load_parent(parent_params)
        if self.inventory['frozen_fingerprint'] != m['bindings']['frozen_fingerprint']:
            raise ValueError('DEPLOYMENT_FROZEN_DENSE_IDENTITY_MISMATCH')
        # Reuse the template checker by restoring the tuple[0] adapter prefix.
        rows, structure = active_rows(dict(arrays=m['adapter_arrays']), self.active)
        leaves = []
        for row in rows:
            p = self.package/row['file']
            if not p.resolve().is_relative_to(self.package.resolve()) or sha(p) != row['sha256']:
                raise ValueError('DEPLOYMENT_ADAPTER_FILE_MISMATCH')
            value = np.load(p, allow_pickle=False)
            if list(value.shape) != row['shape'] or str(value.dtype) != row['dtype']:
                raise ValueError('DEPLOYMENT_ADAPTER_SHAPE_OR_DTYPE_MISMATCH')
            leaves.append(jax.device_put(value))
        self.active, = jax.tree_util.tree_unflatten(structure, leaves)
        stats = json.loads((self.package/'STATISTICS.json').read_bytes())
        self.statistics = {k: Quantiles(stats[k]['q01'], stats[k]['q99']) for k in ('state', 'actions')}
        self.tokenizer = Tokenizer(self.package/'paligemma_tokenizer.model')
        self.resize = resize_with_pad
        graph = self.graph
        self.infer = jax.jit(lambda active, frozen, obs, noise:
                             nnx.merge(graph, active, frozen).sample_actions(jax.random.PRNGKey(0), obs, noise=noise, num_steps=10))

    def predict(self, state10, images, prompt, T_policy_from_mount, *, noise_id):
        import jax
        codec = Codec(T_policy_from_mount)
        # State encoding is independent of the dummy action. No future state or
        # privileged scene/Teacher information is passed to the observation.
        dummy = np.broadcast_to(np.asarray(state10), (40, 10)).copy()
        dummy[:, 9] = .5
        state7, _ = codec.encode(state10, dummy)
        item = images224(images, self.resize)
        item.update(state=pad32(self.statistics['state'].normalize(state7)).astype(np.float32),
                    actions=np.zeros((40, 32), np.float32), prompt=prompt)
        obs, _ = observation(item, self.tokenizer)
        noise = jax.random.normal(jax.random.PRNGKey(2026111200+noise_id), (1, 40, 32))
        normalized = self.infer(self.active, self.frozen, obs, noise)
        jax.block_until_ready(normalized)
        normalized = np.asarray(normalized)
        if normalized.shape != (1, 40, 32) or not np.isfinite(normalized).all():
            raise ValueError('DEPLOYMENT_FULL_FINITE_H40_32_REQUIRED')
        physical = codec.decode(state10, pad32(self.statistics['actions'].unnormalize(normalized[0, :, :7])))
        return dict(normalized=normalized, physical_H40=physical, raw_before_endpoint_basis=True,
                    state_token_dropout=False, teacher_calls=0, optimizer_updates=0)
