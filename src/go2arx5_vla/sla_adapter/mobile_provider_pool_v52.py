"""Explicit B0/B1/B2 serving with one unchanged foundation model.

The pool never selects a policy from observations or scores. Its caller must
name the preregistered arm, and ordinary-RGB restoration must qualify each arm.
"""
import json
from pathlib import Path
import numpy as np
from .checkpoint_v1 import sha
from .provider_time_shift7_v1 import Provider as FrozenProvider
from .provider_mobile_v52 import Provider as SuccessorProvider, PARENT
from .resident_close_pair_v1 import shared_contract


def attach(one, package, arm):
    import jax
    from .provider_v1 import active_rows
    root=Path(package['path'])
    if sha(root/'MANIFEST.private.json')!=package['sha256']:
        raise ValueError('SEALED_V52_POOL_PACKAGE_REQUIRED')
    manifest=json.loads((root/'MANIFEST.private.json').read_bytes())
    two=SuccessorProvider.__new__(SuccessorProvider)
    two.__dict__=one.__dict__.copy();two.package=root;two.manifest=manifest
    two.validate_package_contract(manifest);shared_contract(one.manifest,manifest)
    if manifest['bindings']['endpoint_lambda']!={'B1':0.,'B2':.1}[arm]:
        raise ValueError('PREREGISTERED_ARM_AND_ENDPOINT_LOSS_MUST_MATCH')
    for filename,key in(('STATISTICS.json','statistics_sha256'),('paligemma_tokenizer.model','tokenizer_sha256')):
        if sha(root/filename)!=manifest[key]:raise ValueError('POOL_FROZEN_ASSET_BYTES_CHANGED')
    two.validate_statistics_metadata(json.loads((root/'STATISTICS.json').read_bytes()))
    rows,structure=active_rows(dict(arrays=manifest['adapter_arrays']),one.active);leaves=[]
    if len(rows)!=28:raise ValueError('UNCHANGED_28_ARRAY_POOL')
    for row in rows:
        path=root/row['file']
        if not path.resolve().is_relative_to(root.resolve())or sha(path)!=row['sha256']:
            raise ValueError('POOL_ADAPTER_BYTES_CHANGED')
        value=np.load(path,allow_pickle=False)
        if list(value.shape)!=row['shape']or str(value.dtype)!=row['dtype']:
            raise ValueError('POOL_ADAPTER_SHAPE_OR_DTYPE_CHANGED')
        leaves.append(jax.device_put(value))
    two.active,=jax.tree_util.tree_unflatten(structure,leaves)
    return two


class Pool:
    def __init__(self,spec):
        packages=spec['packages']
        if set(packages)!={'B0','B1','B2'}or packages['B0']['sha256']!=PARENT:
            raise ValueError('EXACT_FROZEN_B0_AND_MATCHED_B1_B2_REQUIRED')
        p=packages['B0']
        one=FrozenProvider(p['path'],manifest_sha256=p['sha256'],parent_params=spec['parent_params'],parent_manifest_path=spec['parent_manifest']['path'])
        self.providers={'B0':one,**{arm:attach(one,packages[arm],arm)for arm in('B1','B2')}}
        self.selected=None

    def select(self,arm):
        if arm not in self.providers:raise ValueError('ONLY_PREREGISTERED_ARM_SELECTION')
        self.selected=arm

    def predict(self,*args,**kwargs):
        if self.selected is None:raise RuntimeError('EXPLICIT_ARM_REQUIRED_BEFORE_INFERENCE')
        return self.providers[self.selected].predict(*args,**kwargs)
