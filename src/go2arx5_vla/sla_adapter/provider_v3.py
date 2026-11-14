"""Distinct ordinary-RGB logit package; V1 providers reject its schema."""
import json
from pathlib import Path
import shutil
import numpy as np

from .checkpoint_v1 import sha
from .codec_v1 import Quantiles, pad32
from .dataset_v1 import images224
from .gripper_v2 import EPSILON, VERSION, GripLogitCodec
from .model_v1 import Tokenizer, load_parent, observation
from .provider_v1 import active_rows

SCHEMA = 'SLA_GRIP_LOGIT_ORDINARY_RGB_INFERENCE_PACKAGE_V3'
DECODE_ORDER = 'INVERSE_LATENT_NORMALIZATION_THEN_SIGMOID_ONCE_THEN_ORIGINAL_NATIVE_NOMINAL_MAPPING'


def contract(value):
    if (value.get('schema') != SCHEMA or value.get('codec') != VERSION
            or value.get('decode_order') != DECODE_ORDER or value.get('label_epsilon') != EPSILON
            or value.get('bindings', {}).get('codec_version') != VERSION
            or value.get('state_gripper_transformed') is not False
            or value.get('teacher_required') is not False
            or value.get('training_cache_required') is not False
            or value.get('optimizer_required') is not False):
        raise ValueError('DISTINCT_TEACHER_FREE_SINGLE_SIGMOID_LOGIT_PACKAGE_REQUIRED')


def export(package, checkpoint, *, active_template, statistics, tokenizer, parent_manifest,
           action_representation_audit):
    package, checkpoint = Path(package), Path(checkpoint)
    complete = json.loads((checkpoint / 'COMPLETE.json').read_bytes())
    bindings = complete['bindings']
    if (complete['schema'] != 'SLA_COMPLETE_PEFT_STATE_V1' or complete['step'] not in (4, 500)
            or complete['cursor'] != complete['step'] * 32
            or bindings.get('codec_version') != VERSION
            or bindings.get('recipe') != 'SLA_EFFECTIVE32_DEVICE_REDUCTION_V3'
            or bindings.get('action_representation_audit') != action_representation_audit
            or bindings['parent_manifest'] != parent_manifest
            or sha(tokenizer) != bindings['tokenizer']):
        raise ValueError('ACTUAL_COMPLETE_LOGIT_ENDPOINT_AND_FULL_BINDINGS_REQUIRED')
    audit_path = Path(action_representation_audit['path'])
    if sha(audit_path) != action_representation_audit['sha256']:
        raise ValueError('LOGIT_REPRESENTATION_AUDIT_CHANGED')
    audit = json.loads(audit_path.read_bytes())
    if (audit['schema'] != 'SLA_GRIP_LOGIT_TRAIN100_DERIVATION_V2' or not audit['passes']
            or audit['original_target_audit']['sha256'] != bindings['target_audit']
            or audit['statistics_sha256'] != bindings['derived_statistics_sha256']
            or sha(statistics) != audit['statistics_sha256']):
        raise ValueError('EXACT_TRAIN_ONLY_LOGIT_STATISTICS_REQUIRED')
    return write_verified_package(package, checkpoint, complete=complete, active_template=active_template,
                                  statistics=statistics, tokenizer=tokenizer, parent_manifest=parent_manifest)


def write_verified_package(package, checkpoint, *, complete, active_template, statistics, tokenizer, parent_manifest):
    """Copy already validated endpoint assets; callers enforce their recipe."""
    package, checkpoint = Path(package), Path(checkpoint)
    bindings = complete['bindings']
    rows, _ = active_rows(complete, active_template)
    package.mkdir(parents=True, exist_ok=False)
    (package / 'adapters').mkdir()
    copied = []
    for row in rows:
        source = checkpoint / row['file']
        if not source.resolve().is_relative_to(checkpoint.resolve()) or sha(source) != row['sha256']:
            raise ValueError('TRAINED_LOGIT_PARAMETER_CHANGED')
        shutil.copyfile(source, package / 'adapters' / row['file'])
        copied.append(dict(row, file='adapters/' + row['file']))
    shutil.copyfile(statistics, package / 'STATISTICS.json')
    shutil.copyfile(tokenizer, package / 'paligemma_tokenizer.model')
    result = dict(schema=SCHEMA, codec=VERSION, decode_order=DECODE_ORDER, label_epsilon=EPSILON,
                  state_gripper_transformed=False, original_label_roundtrip_exact=False,
                  step=complete['step'], cursor=complete['cursor'], qualification_only=complete['step'] == 4,
                  bindings=bindings, parent_manifest=parent_manifest,
                  original_complete_sha256=sha(checkpoint / 'COMPLETE.json'), adapter_arrays=copied,
                  statistics_sha256=sha(package / 'STATISTICS.json'), tokenizer_sha256=sha(package / 'paligemma_tokenizer.model'),
                  camera_keys=['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb'],
                  image_masks=[True, True, False], image_size=[224, 224], sampler_steps=10,
                  action_horizon=40, label_interval_s=.05, endpoint_basis_applied=False,
                  teacher_required=False, training_cache_required=False, optimizer_required=False,
                  physical_execution_admitted=False)
    contract(result)
    (package / 'MANIFEST.private.json').write_text(json.dumps(result, indent=2))
    return result


class Provider:
    def __init__(self, package, *, manifest_sha256, parent_params, parent_manifest_path=None):
        import flax.nnx as nnx
        import jax
        from openpi_client.image_tools import resize_with_pad
        self.package = Path(package)
        if sha(self.package / 'MANIFEST.private.json') != manifest_sha256:
            raise ValueError('LOGIT_PACKAGE_MANIFEST_CHANGED')
        self.manifest = json.loads((self.package / 'MANIFEST.private.json').read_bytes())
        contract(self.manifest)
        m = self.manifest
        if len(jax.devices()) != 1 or jax.devices()[0].platform != 'gpu':
            raise ValueError('ORDINARY_LOGIT_INFERENCE_REQUIRES_ONE_ACTUAL_CUDA_DEVICE')
        for name, key in (('STATISTICS.json', 'statistics_sha256'), ('paligemma_tokenizer.model', 'tokenizer_sha256')):
            if sha(self.package / name) != m[key]:
                raise ValueError('COPIED_LOGIT_DEPLOYMENT_ASSET_CHANGED')
        if (m['statistics_sha256'] != m['bindings']['derived_statistics_sha256']
                or m['tokenizer_sha256'] != m['bindings']['tokenizer']):
            raise ValueError('LOGIT_ASSETS_MUST_MATCH_TRAINING_BINDINGS')
        parent = Path(parent_manifest_path or m['parent_manifest']['path'])
        if sha(parent) != m['parent_manifest']['sha256'] or Path(parent_params).resolve() != (parent.parent / 'model/params').resolve():
            raise ValueError('FIXED_PARENT_BYTES_AND_EXPLICIT_SITE_PATH_REQUIRED')
        asset = json.loads(parent.read_bytes())
        if (asset.get('repo') != 'haofuly/SLA_base' or not asset.get('all_files_verified')
                or asset.get('revision') != '5bbdf3012ab643554aa2eb83ddeae9ace3dce476'):
            raise ValueError('PINNED_SLA_BASE_REQUIRED')
        self.graph, self.active, self.frozen, self.inventory = load_parent(parent_params)
        if self.inventory['frozen_fingerprint'] != m['bindings']['frozen_fingerprint']:
            raise ValueError('LOGIT_PROVIDER_FROZEN_PARAMETER_MISMATCH')
        rows, structure = active_rows(dict(arrays=m['adapter_arrays']), self.active)
        leaves = []
        for row in rows:
            path = self.package / row['file']
            if not path.resolve().is_relative_to(self.package.resolve()) or sha(path) != row['sha256']:
                raise ValueError('LOGIT_ADAPTER_BYTES_CHANGED')
            value = np.load(path, allow_pickle=False)
            if list(value.shape) != row['shape'] or str(value.dtype) != row['dtype']:
                raise ValueError('LOGIT_ADAPTER_SHAPE_OR_DTYPE_CHANGED')
            leaves.append(jax.device_put(value))
        self.active, = jax.tree_util.tree_unflatten(structure, leaves)
        stats = json.loads((self.package / 'STATISTICS.json').read_bytes())
        if stats.get('codec_version') != VERSION or stats.get('epsilon') != EPSILON:
            raise ValueError('LOGIT_STATISTICS_VERSION_REQUIRED')
        self.statistics = {key: Quantiles(stats[key]['q01'], stats[key]['q99']) for key in ('state', 'actions')}
        self.tokenizer = Tokenizer(self.package / 'paligemma_tokenizer.model')
        self.resize = resize_with_pad
        graph = self.graph
        self.infer = jax.jit(lambda active, frozen, obs, noise:
                             nnx.merge(graph, active, frozen).sample_actions(jax.random.PRNGKey(0), obs, noise=noise, num_steps=10))

    def predict(self, state10, images, prompt, T_policy_from_mount, *, noise_id):
        import jax
        codec = GripLogitCodec(T_policy_from_mount)
        dummy = np.broadcast_to(np.asarray(state10), (40, 10)).copy()
        dummy[:, 9] = .5
        state7, _ = codec.encode(state10, dummy)
        item = images224(images, self.resize)
        item.update(state=pad32(self.statistics['state'].normalize(state7)).astype(np.float32),
                    actions=np.zeros((40, 32), np.float32), prompt=prompt)
        obs, _ = observation(item, self.tokenizer)
        noise = jax.random.normal(jax.random.PRNGKey(2026111200 + noise_id), (1, 40, 32))
        normalized = self.infer(self.active, self.frozen, obs, noise)
        jax.block_until_ready(normalized)
        normalized = np.asarray(normalized)
        if normalized.shape != (1, 40, 32) or not np.isfinite(normalized).all():
            raise ValueError('FINITE_FULL_H40_LOGIT_CONTAINER_REQUIRED')
        physical = codec.decode(state10, pad32(self.statistics['actions'].unnormalize(normalized[0, :, :7])))
        return dict(normalized=normalized, physical_H40=physical, codec=VERSION,
                    raw_before_endpoint_basis=True, state_token_dropout=False,
                    teacher_calls=0, optimizer_updates=0)
