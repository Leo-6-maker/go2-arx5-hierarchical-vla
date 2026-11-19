"""Durable SCALE20K state at one global, joint optimizer boundary."""
import os
from pathlib import Path
import tempfile

import numpy as np

from scripts.pi0292.durable import read, write
from .checkpoint_v1 import sha
from .scale20k_contract import DOMAINS, PLAN, RECIPE, bank_sha256, cursors, learning_rates


SCHEMA = 'SLA_SCALE20K_COMPLETE_STATE_V1'


def binding_check(arm, binding):
    keys = ('parent_manifest_sha256','partition_sha256','initial_state_sha256','optimizer_sha256')
    if (binding['recipe'] != RECIPE or binding['plan_sha'] != PLAN or binding['arm'] != arm
            or binding['learning_rates_sha256'] != bank_sha256(learning_rates())
            or set(binding['data_manifest_sha256']) != set(DOMAINS)
            or set(binding['visit_bank_sha256']) != set(DOMAINS)
            or not all(is_digest(binding[key]) for key in keys)
            or not all(is_digest(x) for key in ('data_manifest_sha256','visit_bank_sha256') for x in binding[key].values())):
        raise ValueError('SCALE20K_EXACT_MODEL_DATA_RANDOMNESS_AND_OPTIMIZER_BINDING_REQUIRED')


def is_digest(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def training_tree(parameters, optimizer, rng):
    if set(parameters) != {'shared',*DOMAINS} or set(optimizer) != set(parameters):
        raise ValueError('SCALE20K_CHECKPOINT_REQUIRES_ALL_TRAINABLE_PARAMETERS_AND_MOMENTS')
    return {'parameters':parameters,'optimizer':optimizer,'rng':rng}


def save(path, *, arm, step, parameters, optimizer, rng, binding):
    binding_check(arm,binding)
    committed = cursors(arm,step)
    metadata = dict(schema=SCHEMA,binding=binding,step=step,cursors=committed,arrays=None,
                    training_full_generations=step*4,physics=False)
    return seal_arrays(path,training_tree(parameters,optimizer,rng),metadata)


def seal_arrays(path, tree, metadata):
    import jax

    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    staging = Path(tempfile.mkdtemp(prefix=path.name+'.pending.',dir=path.parent))
    leaves,_ = jax.tree_util.tree_flatten_with_path(tree)
    records = []
    for index,(key,leaf) in enumerate(leaves):
        value = np.asarray(leaf)
        if value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
            raise ValueError('SCALE20K_CHECKPOINT_EXPECTS_FINITE_FLOAT_INT_OR_RNG_ARRAY')
        name = f'{index:06d}.npy'
        file = staging/name
        with file.open('xb') as output:
            np.save(output,value,allow_pickle=False)
            output.flush()
            os.fsync(output.fileno())
        records.append(dict(key=jax.tree_util.keystr(key),file=name,dtype=str(value.dtype),shape=list(value.shape),sha256=sha(file)))
    complete = dict(metadata,arrays=records)
    write(staging/'COMPLETE.json',complete)
    os.rename(staging,path)
    fd = os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return dict(path=str(path),sha256=sha(path/'COMPLETE.json'))


def read_complete(pin, arm, binding):
    binding_check(arm,binding)
    root = Path(pin['path']).resolve(strict=True)
    file = root/'COMPLETE.json'
    if sha(file) != pin['sha256']:
        raise ValueError('SCALE20K_SEALED_COMPLETE_CHECKPOINT_CHANGED')
    receipt = read(file)
    step = receipt['step']
    expected = cursors(arm,step)
    if (receipt['schema'] != SCHEMA or receipt['binding'] != binding or receipt['cursors'] != expected
            or receipt['training_full_generations'] != step*4 or receipt['physics'] is not False):
        raise ValueError('SCALE20K_CHECKPOINT_GLOBAL_STEP_OR_DOMAIN_CURSOR_MISMATCH')
    return root,receipt


def verified_arrays(root, receipt, leaves):
    import jax

    for index,((key,template),row) in enumerate(zip(leaves,receipt['arrays'],strict=True)):
        name = f'{index:06d}.npy'
        member = root/name
        if (row['file'] != name or not member.resolve(strict=True).is_relative_to(root)
                or sha(member) != row['sha256'] or jax.tree_util.keystr(key) != row['key']):
            raise ValueError('SCALE20K_CHECKPOINT_ARRAY_IDENTITY_MISMATCH')
        value = np.load(member,allow_pickle=False)
        if (row['dtype'] != str(value.dtype) or row['shape'] != list(value.shape) or value.shape != template.shape
                or value.dtype != template.dtype or not np.isfinite(value).all()):
            raise ValueError('SCALE20K_CHECKPOINT_ARRAY_SHAPE_DTYPE_OR_FINITE_MISMATCH')
        yield key, value


def restore(pin, *, arm, parameters, optimizer, rng, binding):
    root, receipt = read_complete(pin, arm, binding)
    theory = training_tree(parameters,optimizer,rng)
    return restore_arrays(root, receipt, theory),receipt['step'],receipt['cursors']


def restore_arrays(root, receipt, template):
    import jax

    leaves,structure = jax.tree_util.tree_flatten_with_path(template)
    restored = [jax.device_put(value) for _,value in verified_arrays(root, receipt, leaves)]
    return jax.tree_util.tree_unflatten(structure,restored)


def restore_inference(pin, *, arm, parameters, abstract_optimizer, rng, binding):
    """Verify the whole state and put only trained parameter arrays on the inference device."""
    import jax

    root, receipt = read_complete(pin, arm, binding)
    theory = training_tree(parameters, abstract_optimizer, rng)
    leaves, _ = jax.tree_util.tree_flatten_with_path(theory)
    _, structure = jax.tree_util.tree_flatten(parameters)
    parameter_key = jax.tree_util.DictKey('parameters')
    restored = [jax.device_put(value) for key, value in verified_arrays(root, receipt, leaves)
                if key[0] == parameter_key]
    return jax.tree_util.tree_unflatten(structure, restored)
