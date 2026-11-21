"""Reconstruct the training graph from the frozen base and restore the full ADAPT2K."""
import json
from pathlib import Path

import numpy as np

from go2arx5_vla.sla_adapter import adapt5k_checkpoint, scale20k_model
from go2arx5_vla.sla_adapter.adapt5k_training import OPTIMIZER
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.model_v1 import Tokenizer
from go2arx5_vla.sla_adapter.scale20k_contract import DOMAINS, bank_sha256, canonical_hash, make_adamw
from scripts.sla.hopper_inputs import BATCH, PARENT, PLAN, STEPS, check_samples, inputs, parent_identity, phase_rates


def bank_package(root, expected_sha, marker, last):
    folder=root/'banks.verified'
    receipt=folder/'BANKS.private.json'
    if sha(receipt)!=expected_sha:
        raise ValueError('Unrecognized initial bank seal')
    config=json.loads(receipt.read_bytes())
    keys={'target','robodojo','libero','robodojo_noise_cursors','libero_noise_cursors','rates','rng'}
    if (config['plan']!=PLAN or config['parent']!=PARENT or config['ancestor']!=marker['progress']
            or set(config['arrays'])!=keys or config['sources_frozen_through']!=STEPS):
        raise ValueError('Bank ancestry or membership changed')
    values={}
    for name,entry in config['arrays'].items():
        arr=np.load(folder/(name+'.npy'),mmap_mode='r',allow_pickle=False)
        if (bank_sha256(arr)!=entry['sha256'] or list(arr.shape)!=entry['shape']
                or str(arr.dtype)!=entry['dtype']):
            raise ValueError('Changed bank '+name)
        values[name]=arr
    clocks=marker['progress']['lineage_cursors']
    for domain in DOMAINS:
        if values[domain].shape!=(STEPS*BATCH[domain],) or values[domain].dtype!=np.int64:
            raise ValueError('Wrong visit bank shape for '+domain)
        if domain!='target':
            noise=values[domain+'_noise_cursors']
            if not np.array_equal(noise,np.arange(clocks[domain],clocks[domain]+len(values[domain]),dtype='<u4')):
                raise ValueError('Absolute noise clocks changed')
    if not np.array_equal(values['rates'],phase_rates(last['learning_rate'])):
        raise ValueError('20K rates diverge from the last measured parent update')
    return values,config


def restore(root, expected_banks, *, check_device_arrays=False):
    import jax

    root=Path(root)
    site=inputs(root/'assets.verified')
    parent=root/'parent.verified'
    marker,last=parent_identity(parent,root/'banks.verified/PARENT_UPDATES.private.jsonl')
    banks,receipt=bank_package(root,expected_banks,marker,last)
    datasets,samples=check_samples(site)
    if samples!=receipt['checked_train_samples']:
        raise ValueError('Destination samples differ from the bank producer')
    binding=marker['binding']
    if (sha(site['tokenizer']['path'])!=binding['tokenizer_sha256']
            or site['data']['target']['statistics']['sha256']!=binding['target_state_sha256']
            or site['data']['robodojo']['manifest']['sha256']!=binding['R200_membership_sha256']
            or site['data']['libero']['manifest']['sha256']!=binding['LIBERO_membership_sha256']
            or canonical_hash(OPTIMIZER)!=binding['optimizer_sha256']):
        raise ValueError('Frozen parent representation or optimizer differs')
    for d in DOMAINS:
        a=banks[d]
        if a.min()<0 or a.max()>=len(datasets[d]):
            raise ValueError('Visit bank exceeds its training data for '+d)
    graphs,params,frozen,model=scale20k_model.load(site['parent_manifest'],site['parent_params'])
    if (model['parent_manifest_sha256']!=binding['parent_manifest_sha256']
            or canonical_hash(model['inventory'])!=binding['partition_sha256']):
        raise ValueError('Frozen model or parameter partition differs from ADAPT2K')
    init,_=make_adamw(**OPTIMIZER)
    state,progress=adapt5k_checkpoint.restore(dict(path=str(parent),sha256=PARENT),parameters=params,
              optimizer=jax.eval_shape(init,params),rng=jax.random.PRNGKey(scale20k_model.INITIAL_SEED),binding=binding)
    del params
    jax.block_until_ready(state)
    if progress!=marker['progress'] or not np.array_equal(np.asarray(state['rng']),banks['rng']):
        raise ValueError('Restored parent random key or cursors differ from the frozen bank')
    counts={k:int(np.asarray(v[0].count)) for k,v in state['optimizer'].items()}
    if set(counts)!={'shared',*DOMAINS} or any(v!=progress['lineage_total_updates'] for v in counts.values()):
        raise ValueError('Restored Adam count differs from lineage')
    verified=0
    if check_device_arrays:
        leaves=jax.tree_util.tree_flatten_with_path(state)[0]
        for (key,array),row in zip(leaves,marker['arrays'],strict=True):
            saved=np.load(parent/row['file'],allow_pickle=False,mmap_mode='r')
            actual=np.asarray(array)
            if (jax.tree_util.keystr(key)!=row['key'] or saved.dtype!=actual.dtype or saved.shape!=actual.shape
                    or not np.array_equal(saved.reshape(-1).view(np.uint8),actual.reshape(-1).view(np.uint8))):
                raise ValueError('Reloaded array differs at '+row['key'])
            verified+=1
    proof=dict(parent=PARENT,plan=PLAN,banks_sha256=expected_banks,ancestor=progress,
               restored_arrays=len(jax.tree.leaves(state)),device_readback_exact=verified,adam_counts=counts,
               frozen_model_fingerprint=model['frozen_fingerprint'],sample_readback=samples,
               last_actual_lr=last['learning_rate'],rates_sha256=receipt['arrays']['rates']['sha256'],
               optimizer_updates=0,optimization_visits=0,native=0,generations=0)
    return graphs,frozen,state,banks,datasets,Tokenizer(site['tokenizer']['path']),proof
