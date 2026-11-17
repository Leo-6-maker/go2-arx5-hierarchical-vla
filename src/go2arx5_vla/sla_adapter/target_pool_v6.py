"""Explicit frozen B1 / matched FM500 / full-H40-loss500 shared-base pool."""
import json
from pathlib import Path
import numpy as np
from .checkpoint_v1 import sha
from .target_contract_v6 import B1,contract
from .resident_close_pair_v1 import shared_contract


def attach(one,package,arm):
    import jax
    from .provider_v1 import active_rows
    from .provider_target_v6 import Provider
    root=Path(package['path'])
    if sha(root/'MANIFEST.private.json')!=package['sha256']:raise ValueError('SEALED_V6_TARGET_PACKAGE_REQUIRED')
    manifest=json.loads((root/'MANIFEST.private.json').read_bytes());contract(manifest)
    if manifest['bindings']['full_H40_lambda']!={'T-FM':0.,'T-H40':.1}[arm]:raise ValueError('EXACT_V6_ARM_LAMBDA_REQUIRED')
    shared_contract(one.manifest,manifest)
    two=Provider.__new__(Provider);two.__dict__=one.__dict__.copy();two.package=root;two.manifest=manifest
    for name,key in(('STATISTICS.json','statistics_sha256'),('paligemma_tokenizer.model','tokenizer_sha256')):
        if sha(root/name)!=manifest[key]:raise ValueError('V6_SHARED_POOL_FROZEN_ASSET_CHANGED')
    two.validate_statistics_metadata(json.loads((root/'STATISTICS.json').read_bytes()))
    rows,structure=active_rows(dict(arrays=manifest['adapter_arrays']),one.active);values=[]
    if len(rows)!=28:raise ValueError('ALL28_TARGET_ACTIVE_ARRAYS_REQUIRED')
    for row in rows:
        path=root/row['file']
        if not path.resolve().is_relative_to(root.resolve())or sha(path)!=row['sha256']:raise ValueError('V6_TARGET_ADAPTER_BYTES_CHANGED')
        value=np.load(path,allow_pickle=False)
        if list(value.shape)!=row['shape']or str(value.dtype)!=row['dtype']:raise ValueError('V6_TARGET_ADAPTER_SHAPE_DTYPE_CHANGED')
        values.append(jax.device_put(value))
    two.active,=jax.tree_util.tree_unflatten(structure,values)
    return two


class Pool:
    def __init__(self,spec):
        from .provider_mobile_v52 import Provider
        packages=spec['packages']
        if set(packages)!={'T-B1','T-FM','T-H40'}or packages['T-B1']['sha256']!=B1:
            raise ValueError('EXACT_V6_TARGET_THREE_ARM_POOL_REQUIRED')
        p=packages['T-B1'];one=Provider(p['path'],manifest_sha256=p['sha256'],parent_params=spec['parent_params'],parent_manifest_path=spec['parent_manifest']['path'])
        self.providers={'T-B1':one,**{arm:attach(one,packages[arm],arm)for arm in('T-FM','T-H40')}};self.selected=None

    def select(self,arm):
        if arm not in self.providers:raise ValueError('EXPLICIT_V6_TARGET_ARM_REQUIRED')
        self.selected=arm

    def predict(self,*args,**kwargs):
        if self.selected is None:raise ValueError('NO_STATE_BASED_POLICY_SELECTION')
        return self.providers[self.selected].predict(*args,**kwargs)
