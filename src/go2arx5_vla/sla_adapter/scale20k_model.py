"""Load one SLA_base weight set shared by three native-horizon graph views."""
import gc
import hashlib
import math
from pathlib import Path

import numpy as np

from .checkpoint_v1 import sha
from .model_v1 import _restore_for_placement, flat, frozen_fingerprint
from .runtime_v1 import read_pin
from .scale20k_contract import DOMAINS, PARENT_REVISION, parameter_partition, scale_graph_config


INITIAL_SEED = 2026111920


def abstract_views():
    import flax.nnx as nnx
    import jax

    graphs, template = {},None
    for domain in DOMAINS:
        model = nnx.eval_shape(scale_graph_config(domain).create,jax.random.PRNGKey(INITIAL_SEED))
        graph,state = nnx.split(model)
        arrays = flat(state.to_pure_dict())
        if template is None:
            template = arrays
        if ({path: (x.shape,x.dtype) for path,x in arrays.items()}
                != {path: (x.shape,x.dtype) for path,x in template.items()}):
            raise ValueError('SCALE20K_HORIZON_VIEWS_MUST_SHARE_EXACT_PARAMETER_SHAPES')
        graphs[domain] = graph
    return graphs,template,parameter_partition(template)


def parent_asset(manifest, params):
    asset = read_pin(manifest)
    root = Path(manifest['path']).parent
    if (asset['repo'] != 'haofuly/SLA_base' or asset['revision'] != PARENT_REVISION
            or asset['all_files_verified'] is not True or Path(params).resolve(strict=True) != (root/'model/params').resolve(strict=True)):
        raise ValueError('SCALE20K_VERIFIED_SLA_BASE_PARENT_REQUIRED')
    return asset


def load(manifest, params):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from flax.traverse_util import unflatten_dict

    parent_asset(manifest,params)
    graphs,template,partition = abstract_views()
    frozen_names = set(partition['frozen'])
    shared_names = set(partition['vlm_lora']) | set(partition['action_expert'])
    private_names = set(partition['domain_template'])

    def placement(path):
        key = tuple(path)
        if key[0] == 'params':
            key = key[1:]
        if key[-1] == 'value':
            key = key[:-1]
        return 'float32' if key in shared_names or key in private_names else 'bfloat16'

    parent = flat(_restore_for_placement(params,dtype_for_path=placement))
    if any(path[0] != 'align_projector' for path in set(parent)-set(template)):
        raise ValueError('SCALE20K_UNEXPECTED_PARENT_WEIGHTS')
    initialized, inventory = {},[]
    for path,reference in template.items():
        if path in parent:
            host = parent.pop(path)
            if host.shape != reference.shape:
                raise ValueError('SCALE20K_DENSE_PARENT_SHAPE_MISMATCH:'+str(path))
            leaf = jax.device_put(host)
        else:
            if path not in partition['vlm_lora']:
                raise ValueError('SCALE20K_MISSING_DENSE_PARENT_WEIGHT:'+str(path))
            seed = int.from_bytes(hashlib.sha256('/'.join(path).encode()).digest()[:4],'little')
            key = jax.random.fold_in(jax.random.PRNGKey(INITIAL_SEED),seed)
            leaf = (jnp.zeros(reference.shape,jnp.float32) if path[-1].endswith('lora_b')
                    else jax.random.normal(key,reference.shape)*.01)
        jax.block_until_ready(leaf)
        initialized[path] = leaf
        group = next(group for group,paths in partition.items() if path in paths)
        inventory.append(dict(path='/'.join(path),group=group,shape=list(leaf.shape),dtype=str(leaf.dtype),numel=math.prod(leaf.shape)))
    del parent
    gc.collect()

    # Materialize the three private states using the original matched I/O bytes.
    # No previously trained projection, adapter, or optimizer enters this tree.
    model = nnx.eval_shape(scale_graph_config('target').create,jax.random.PRNGKey(INITIAL_SEED))
    _,state = nnx.split(model)
    state.replace_by_pure_dict(unflatten_dict(initialized))
    _,shared,private,frozen = nnx.split(nnx.merge(graphs['target'],state),lambda p,_:tuple(p) in shared_names,
                                        lambda p,_:tuple(p) in private_names,...)
    if set(flat(frozen.to_pure_dict())) != frozen_names:
        raise ValueError('SCALE20K_RUNTIME_FROZEN_TREE_DISAGREES_WITH_PARTITION')
    parameters = dict(shared=shared,**{domain:jax.tree.map(jnp.copy,private) for domain in DOMAINS})
    proof = dict(parent_manifest_sha256=sha(manifest['path']),parent_revision=PARENT_REVISION,inventory=inventory,
        frozen_fingerprint=frozen_fingerprint(frozen),common_shared_fingerprint=frozen_fingerprint(shared),
        common_private_fingerprint=frozen_fingerprint(private),dtype_recipe='FP32 trained parameters and gradients; BF16 frozen model storage; highest matmul')
    return graphs,parameters,frozen,proof
