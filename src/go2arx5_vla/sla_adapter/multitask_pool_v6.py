"""Finite explicit G7 arm selection over one verified parent."""
import json
from pathlib import Path

import numpy as np

from .checkpoint_v1 import sha
from .multitask_contract_v6 import ARMS, contract
from .provider_multitask_v6 import Provider
from .provider_v1 import active_rows
from .resident_close_pair_v1 import shared_contract
from .runtime_v1 import read_pin


def attach(parent, pin, *, check=contract, training_keys=('dataset_manifest_sha256', 'compiled_population_sha256',
                                                           'visit_map_sha256', 'lr_sha256')):
    import jax

    root = Path(pin['path'])
    manifest = read_pin(dict(path=str(root/'MANIFEST.private.json'),sha256=pin['sha256']))
    check(manifest)
    shared_contract(parent.manifest,manifest)
    for key in training_keys:
        if parent.manifest['bindings'][key] != manifest['bindings'][key]:
            raise ValueError('EXACT_SHARED_G7_TRAINING_IDENTITY:'+key)
    other = type(parent).__new__(type(parent))
    other.__dict__ = dict(parent.__dict__,package=root,manifest=manifest)
    for file, key in (('STATISTICS.json','statistics_sha256'),('paligemma_tokenizer.model','tokenizer_sha256')):
        if sha(root/file) != manifest[key]:
            raise ValueError('G7_RESIDENT_TARGET_ASSET_CHANGED:'+file)
    other.validate_statistics_metadata(json.loads((root/'STATISTICS.json').read_bytes()))
    rows, structure = active_rows(dict(arrays=manifest['adapter_arrays']),parent.active)
    values = []
    for row in rows:
        path = root/row['file']
        if not path.resolve().is_relative_to(root.resolve()) or sha(path) != row['sha256']:
            raise ValueError('G7_RESIDENT_ADAPTER_BYTES_CHANGED')
        value = np.load(path,allow_pickle=False)
        if list(value.shape) != row['shape'] or str(value.dtype) != row['dtype']:
            raise ValueError('G7_RESIDENT_ADAPTER_SHAPE_OR_DTYPE_CHANGED')
        values.append(jax.device_put(value))
    other.active, = jax.tree_util.tree_unflatten(structure,values)
    return other


class Pool:
    def __init__(self,spec):
        from go2arx5_vla.world_model import multisource_d4_native as d4

        is_d4 = spec.get('scope') == d4.SCOPE
        arms, provider_cls, attach_options = ARMS, Provider, {}
        if is_d4:
            from .multisource_d4 import Provider as D4Provider, package_contract

            arms, provider_cls = d4.ARMS, D4Provider
            attach_options = dict(check=package_contract,
                                  training_keys=('dataset_manifest_sha256', 'visit_map_sha256', 'lr_sha256'))
        packages = spec['packages']
        if (not packages or not set(packages).issubset(arms) or set(packages)!=set(spec['request_bindings_by_arm'])
                or (is_d4 and (set(packages) != set(arms)
                              or any(packages[a]['sha256'] != d4.SEALED[a]['package'] for a in arms)))):
            raise ValueError('EXPLICIT_PREREGISTERED_G7_ARMS_REQUIRED')
        first = next(arm for arm in arms if arm in packages)
        pin = packages[first]
        parent = provider_cls(pin['path'],manifest_sha256=pin['sha256'],parent_params=spec['parent_params'],
                              parent_manifest_path=spec['parent_manifest']['path'])
        self.providers = {arm:parent if arm==first else attach(parent, package, **attach_options)
                          for arm,package in packages.items()}
        for arm, provider in self.providers.items():
            request = spec['request_bindings_by_arm'][arm]
            if (provider.manifest['bindings']['arm']!=arm or request['arm']!=arm
                    or request['policy_package_sha256'] != packages[arm]['sha256']
                    or request['statistics_sha256'] != provider.manifest['statistics_sha256']):
                raise ValueError('G7_ARM_AND_REQUEST_PACKAGE_BINDINGS_MUST_MATCH')
        self.selected = None

    def select(self,arm):
        if arm not in self.providers:
            raise ValueError('ONLY_PREREGISTERED_G7_ARMS_MAY_SERVE')
        self.selected = arm

    def predict(self,*args,**kwargs):
        if self.selected is None:
            raise ValueError('EXPLICIT_G7_ARM_REQUIRED_BEFORE_INFERENCE')
        return self.providers[self.selected].predict(*args,**kwargs)
