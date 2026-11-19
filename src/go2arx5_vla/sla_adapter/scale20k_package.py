"""Export a sealed SCALE20K Go2 endpoint with only target inference state."""
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from scripts.pi0292.durable import read, write
from .checkpoint_v1 import sha
from .scale20k_checkpoint import is_digest, read_complete
from .scale20k_contract import PLAN, RECIPE, cursors


SCHEMA = 'SLA_SCALE20K_TARGET_RGB_PACKAGE_V1'
SCOPE = 'SLA_SCALE20K_NATIVE_G7'
ENDPOINTS = {'SCALE-TGT5K': ('SCALE-TGT5K', 5000), 'SCALE-MIX5K': ('SCALE-MIX20K', 5000),
             'SCALE-MIX20K': ('SCALE-MIX20K', 20000)}
CODEC = 'DENSE_REFERENCE_G7_V6'
TARGET_PREFIXES = ("['parameters']['shared']", "['parameters']['target']")


def contract(manifest):
    arm, step = manifest['arm'], manifest['step']
    rows = manifest['arrays']
    keys = [row['key'] for row in rows]
    hashes = ('checkpoint_sha256', 'offline_evaluation_sha256', 'offline_restore_sha256', 'statistics_sha256',
              'tokenizer_sha256', 'frozen_fingerprint', 'partition_sha256', 'initial_state_sha256',
              'panel_sha256', 'target_data_manifest_sha256', 'target_admission_sha256')
    if (manifest['schema'] != SCHEMA or manifest['plan_sha'] != PLAN or manifest['recipe'] != RECIPE
            or manifest['codec'] != CODEC or (arm, step) not in ENDPOINTS.values() or manifest['cursors'] != cursors(arm, step)
            or not all(is_digest(manifest[key]) for key in hashes) or not is_digest(manifest['parent_manifest']['sha256'])
            or manifest['inference'] != dict(sampler='ten_cached_flow_steps', matmul='highest', cameras='body_and_left_wrist',
                                            horizon=40, interval_s=.05, state_token_dropout=False)
            or any(manifest[k] is not False for k in ('hardware_allowed', 'native_admission', 'optimizer_required',
                                                     'source_data_required', 'teacher_required'))
            or not rows or len(set(keys)) != len(keys) or len({row['file'] for row in rows}) != len(rows)
            or not all(any(key.startswith(prefix) for key in keys) for prefix in TARGET_PREFIXES)
            or any(not row['key'].startswith(TARGET_PREFIXES) or row['dtype'] != 'float32'
                   or not is_digest(row['sha256']) or Path(row['file']).parts != ('parameters', Path(row['file']).name)
                   for row in rows)):
        raise ValueError('SCALE20K_SEALED_GO2_ONLY_TARGET_DEPLOYMENT_PACKAGE_REQUIRED')


def checkpoint_parameters(receipt):
    rows = [row for row in receipt['arrays'] if row['key'].startswith(TARGET_PREFIXES)]
    if not rows or any(row['file'] != f'{index:06d}.npy' for index, row in enumerate(receipt['arrays'])):
        raise ValueError('SCALE20K_DEPLOYMENT_REQUIRES_ORIGINAL_COMMITTED_TENSOR_INDEX')
    return rows


def checked_copy(source, destination, expected):
    with Path(source).open('rb') as data, Path(destination).open('xb') as out:
        shutil.copyfileobj(data, out, 1024**2)
        out.flush()
        os.fsync(out.fileno())
    if sha(destination) != expected:
        raise ValueError('SCALE20K_DEPLOYMENT_MEMBER_BYTES_CHANGED')


def export(path, spec):
    from .multitask_data_v6 import C20_STATISTICS
    from .runtime_v1 import read_pin
    from .scale20k_evaluation import PER_DOMAIN, RESTORE
    from .scale20k_contract import DOMAINS

    path = Path(path)
    arm, step = spec['arm'], spec['step']
    trained = read_pin(spec['model_at_start'])
    binding = trained['binding']
    checkpoint, complete = read_complete(spec['checkpoint'], arm, binding)
    evaluate = read_pin(spec['evaluation'])
    restore = read_pin(spec['restore'])
    if (path.exists() or (arm, step) not in ENDPOINTS.values() or complete['step'] != step
            or evaluate['status'] != 'ENDPOINT_PREDICTIONS_SEALED' or restore['status'] != 'FRESH_RGB48_RESTORED_EXACT'
            or any(r['exit_code'] or r['arm'] != arm or r['step'] != step or r['checkpoint'] != spec['checkpoint']
                   or r['optimization_updates'] or r['new_native_requests'] or r['old_TEST_access'] for r in (evaluate,restore))
            or evaluate['panel'] != restore['panel'] or evaluate['source_sha'] != restore['source_sha']
            or evaluate['new_model_generations'] != len(DOMAINS)*PER_DOMAIN
            or restore['new_model_generations'] != len(DOMAINS)*len(RESTORE)
            or any(len(evaluate['rows'][d]) != PER_DOMAIN or len(restore['rows'][d]) != len(RESTORE) for d in DOMAINS)
            or spec['statistics']['sha256'] != C20_STATISTICS
            or spec['tokenizer']['sha256'] != binding['tokenizer_sha256']
            or spec['parent_manifest']['sha256'] != binding['parent_manifest_sha256']):
        raise ValueError('SCALE20K_FORMAL_ENDPOINT_AND_FRESH_THREE_DOMAIN_RESTORATION_REQUIRED')

    staging = Path(tempfile.mkdtemp(prefix=path.name+'.pending.', dir=path.parent))
    (staging/'parameters').mkdir()
    rows = []
    for row in checkpoint_parameters(complete):
        file = checkpoint/row['file']
        if not file.resolve(strict=True).is_relative_to(checkpoint):
            raise ValueError('SCALE20K_ORIGINAL_TARGET_TENSOR_PATH_REQUIRED')
        relative = f"parameters/{row['file']}"
        checked_copy(file, staging/relative, row['sha256'])
        rows.append(dict(row, file=relative))
    for pin, name in ((spec['statistics'], 'STATISTICS.json'), (spec['tokenizer'], 'paligemma_tokenizer.model')):
        checked_copy(pin['path'], staging/name, pin['sha256'])
    manifest = dict(schema=SCHEMA, plan_sha=PLAN, recipe=RECIPE, codec=CODEC, arm=arm, step=step,
        cursors=complete['cursors'], checkpoint_sha256=spec['checkpoint']['sha256'], arrays=rows,
        parent_manifest=spec['parent_manifest'], partition_sha256=binding['partition_sha256'],
        initial_state_sha256=binding['initial_state_sha256'], frozen_fingerprint=trained['frozen_fingerprint'],
        panel_sha256=evaluate['panel']['sha256'], target_data_manifest_sha256=binding['data_manifest_sha256']['target'],
        target_admission_sha256=binding['target_admission_sha256'],
        offline_evaluation_sha256=spec['evaluation']['sha256'], offline_restore_sha256=spec['restore']['sha256'],
        statistics_sha256=spec['statistics']['sha256'], tokenizer_sha256=spec['tokenizer']['sha256'],
        inference=dict(sampler='ten_cached_flow_steps', matmul='highest', cameras='body_and_left_wrist',
                       horizon=40, interval_s=.05, state_token_dropout=False),
        hardware_allowed=False, native_admission=False, optimizer_required=False,
        source_data_required=False, teacher_required=False)
    contract(manifest)
    write(staging/'MANIFEST.private.json', manifest)
    os.rename(staging, path)
    fd = os.open(path.parent, os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return dict(path=str(path), sha256=sha(path/'MANIFEST.private.json'))


def read_weights(package, rows, template):
    import jax

    leaves, structure = jax.tree_util.tree_flatten_with_path({'parameters': template})
    restored = []
    for (key, leaf), row in zip(leaves, rows, strict=True):
        path = package/row['file']
        if (jax.tree_util.keystr(key) != row['key'] or not path.resolve(strict=True).is_relative_to(package)
                or sha(path) != row['sha256']):
            raise ValueError('SCALE20K_EXACT_ORDERED_TARGET_DEPLOYMENT_WEIGHT_REQUIRED')
        value = np.load(path, allow_pickle=False)
        if (list(value.shape) != row['shape'] or value.shape != leaf.shape or value.dtype != leaf.dtype
                or str(value.dtype) != row['dtype'] or not np.isfinite(value).all()):
            raise ValueError('SCALE20K_DEPLOYMENT_WEIGHT_SHAPE_DTYPE_OR_FINITE_MISMATCH')
        restored.append(jax.device_put(value))
    return jax.tree_util.tree_unflatten(structure, restored)['parameters']


class Provider:
    @staticmethod
    def read_manifest(root, manifest_sha256):
        return read_package(root,manifest_sha256)

    @staticmethod
    def load_statistics(root, manifest):
        from .codec_v1 import Quantiles
        from .multitask_data_v6 import C20_STATISTICS
        from .runtime_v1 import read_pin

        stats = read_pin(dict(path=str(root/'STATISTICS.json'),sha256=C20_STATISTICS))
        if manifest['statistics_sha256'] != C20_STATISTICS:
            raise ValueError('SCALE20K_TARGET_DEPLOYMENT_C20_STATISTICS_CHANGED')
        return {key:Quantiles(stats[key]['q01'],stats[key]['q99']) for key in ('state','actions')}

    def __init__(self, package, *, manifest_sha256, parent_params, parent_manifest_path=None):
        import flax.nnx as nnx
        import jax
        from openpi.models import pi0
        from openpi_client.image_tools import resize_with_pad
        from scripts.sla.mfree_c20_vectorfield_v3 import extracted_fields
        from . import batch_v2, full_h40_v6, scale20k_model
        from .model_v1 import Tokenizer
        from .scale20k_contract import canonical_hash

        root = Path(package).resolve(strict=True)
        self.manifest = manifest = self.read_manifest(root,manifest_sha256)
        if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
            raise ValueError('SCALE20K_TARGET_DEPLOYMENT_REQUIRES_EXACT_SINGLE_GPU')
        token = root/'paligemma_tokenizer.model'
        if sha(token) != manifest['tokenizer_sha256']:
            raise ValueError('SCALE20K_TARGET_DEPLOYMENT_TOKENIZER_CHANGED')
        self.statistics = self.load_statistics(root,manifest)
        self.tokenizer, self.resize = Tokenizer(token), resize_with_pad
        parent = manifest['parent_manifest']
        if parent_manifest_path is not None:
            parent = dict(parent, path=str(parent_manifest_path))
        graphs, training, self.frozen, model = scale20k_model.load(parent, parent_params)
        if (manifest['frozen_fingerprint'] != model['frozen_fingerprint']
                or manifest['partition_sha256'] != canonical_hash(model['inventory'])
                or manifest['initial_state_sha256'] != canonical_hash([model['common_shared_fingerprint'], model['common_private_fingerprint']])):
            raise ValueError('SCALE20K_TARGET_DEPLOYMENT_FROZEN_PARENT_OR_PARTITION_CHANGED')
        self.parameters = read_weights(root, manifest['arrays'], {key:training[key] for key in ('shared', 'target')})
        self.active = nnx.State.merge(*self.parameters.values())
        fields, _ = extracted_fields(batch_v2, pi0)
        self.infer = full_h40_v6.make_sampler(graphs['target'], fields, rematerialize=False, precision='highest')

    def predict(self, state10, images, prompt, T_policy_from_mount, *, noise_id, previous_reference):
        import jax
        from .codec_v1 import pad32
        from .dataset_v1 import images224
        from .dense_g7_v6 import label_margin_report, previous_command_context
        from .gripper_v2 import GripLogitCodec
        from .model_v1 import observation

        previous_command_context(previous_reference)
        if type(noise_id) is not int or not 0 <= noise_id < 2**32:
            raise ValueError('SCALE20K_REQUEST_REQUIRES_FIXED_UINT32_NOISE')
        codec = GripLogitCodec(T_policy_from_mount)
        nominal = np.broadcast_to(np.asarray(state10), (40,10)).copy()
        nominal[:,9] = .5
        state7, _ = codec.encode(state10, nominal)
        item = images224(images, self.resize)
        item.update(state=pad32(self.statistics['state'].normalize(state7)).astype(np.float32),
                    actions=np.zeros((40,32), np.float32), prompt=prompt)
        obs, _ = observation(item, self.tokenizer)
        noise = jax.random.normal(jax.random.PRNGKey(noise_id), (1,40,32))
        normalized = np.asarray(self.infer(self.active, self.frozen, obs, noise))
        if normalized.shape != (1,40,32) or not np.isfinite(normalized).all():
            raise ValueError('SCALE20K_TARGET_DEPLOYMENT_FINITE_FULL_H40_REQUIRED')
        physical = codec.decode(state10, pad32(self.statistics['actions'].unnormalize(normalized[0,:,:7])))
        return dict(normalized=normalized, physical_H40=physical, codec=CODEC,
                    G7_gripper_speed=label_margin_report(physical, previous_reference), raw_before_endpoint_basis=True,
                    state_token_dropout=False, teacher_calls=0, optimizer_updates=0)

    def attach(self, package, manifest):
        import flax.nnx as nnx

        root = Path(package).resolve(strict=True)
        shared = ('statistics_sha256', 'tokenizer_sha256', 'frozen_fingerprint', 'partition_sha256',
                  'initial_state_sha256', 'panel_sha256', 'target_data_manifest_sha256', 'target_admission_sha256')
        if (any(manifest[key] != self.manifest[key] for key in shared)
                or manifest['parent_manifest']['sha256'] != self.manifest['parent_manifest']['sha256']
                or sha(root/'STATISTICS.json') != manifest['statistics_sha256']
                or sha(root/'paligemma_tokenizer.model') != manifest['tokenizer_sha256']):
            raise ValueError('SCALE20K_RESIDENT_ENDPOINTS_MUST_SHARE_PARENT_TARGET_BANK_AND_IO')
        other = object.__new__(type(self))
        other.__dict__ = dict(self.__dict__, manifest=manifest)
        other.parameters = read_weights(root, manifest['arrays'], self.parameters)
        other.active = nnx.State.merge(*other.parameters.values())
        return other


def read_package(path, expected_sha256):
    path = Path(path)/'MANIFEST.private.json'
    if sha(path) != expected_sha256:
        raise ValueError('SCALE20K_GO2_DEPLOYMENT_MANIFEST_CHANGED')
    manifest = read(path)
    contract(manifest)
    return manifest


class Pool:
    """Keep one frozen parent and select the pinned endpoint per request."""
    def __init__(self, spec):
        packages, bindings = spec['packages'], spec['request_bindings_by_arm']
        if (spec['scope'] != SCOPE or spec['plan_commit'] != PLAN or spec['physics'] is not False
                or spec['new_physical_requests_budget'] or not packages or set(packages) != set(bindings)
                or not set(packages).issubset(ENDPOINTS)):
            raise PermissionError('SCALE20K_EXPLICIT_NONINTEGRATING_ENDPOINT_WORKER_REQUIRED')
        manifests = {}
        for name, pin in packages.items():
            m = read_package(pin['path'], pin['sha256'])
            req = bindings[name]
            if ((m['arm'], m['step']) != ENDPOINTS[name] or req['arm'] != name
                    or req['policy_package_sha256'] != pin['sha256'] or req['statistics_sha256'] != m['statistics_sha256']
                    or m['parent_manifest']['sha256'] != spec['parent_manifest']['sha256']):
                raise ValueError('SCALE20K_RUNTIME_ENDPOINT_AND_REQUEST_BINDING_CHANGED')
            manifests[name] = m
        first = next(iter(packages))
        pin = packages[first]
        parent = Provider(pin['path'], manifest_sha256=pin['sha256'], parent_params=spec['parent_params'],
                          parent_manifest_path=spec['parent_manifest']['path'])
        self.providers = {name: parent if name == first else parent.attach(pin['path'], manifests[name])
                          for name, pin in packages.items()}
        self.selected = None

    def select(self, arm):
        if arm not in self.providers:
            raise ValueError('SCALE20K_PREREGISTERED_ENDPOINT_REQUIRED')
        self.selected = arm

    def predict(self, *args, **kwargs):
        if self.selected is None:
            raise ValueError('SCALE20K_ENDPOINT_NOT_SELECTED')
        return self.providers[self.selected].predict(*args, **kwargs)


if __name__ == '__main__':
    import argparse
    import json
    from .runtime_v1 import read_pin

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    parser.add_argument('package', type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.package, read_pin(dict(path=args.spec, sha256=args.sha256))), allow_nan=False))
