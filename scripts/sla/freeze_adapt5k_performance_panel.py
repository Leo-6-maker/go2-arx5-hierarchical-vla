"""Pin the Go2/Source DEV inputs and noise before the ADAPT batch/two-GPU comparison."""
import argparse
import os
from pathlib import Path

import numpy as np

from go2arx5_vla.sla_adapter import adapt5k_contract as contract
from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.model_v1 import Tokenizer
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.scale20k_contract import DOMAINS, HORIZONS, canonical_hash
from go2arx5_vla.sla_adapter.scale20k_evaluation import bytes_sha
from go2arx5_vla.sla_adapter.scale20k_training import prepare_host_batch
from scripts.pi0292.durable import write
from scripts.sla import freeze_adapt5k_panels as original


SCHEMA = 'SLA_ADAPT5K_BATCH_FIXED_OFFLINE_PANEL_V1'
COUNTS = dict(target=32,robodojo=8,libero=8)


def selected_rows(panel):
    if (panel['schema'] != original.SCHEMA or panel['plan'] != contract.PLAN or panel['old_TEST_access']
            or panel['source_splits'] != dict.fromkeys(DOMAINS[1:],'SOURCE_DEV')):
        raise ValueError('ADAPT_BATCH_REQUIRES_FROZEN_D1_AND_SOURCE_DEV_PANEL')
    phase = list(enumerate(panel['phase_train']))
    due = [(i,r) for i,r in phase if r['phase_bucket']=='P10_DUE_CLOSE']
    not_due = [(i,r) for i,r in phase if r['phase_bucket']=='NOT_DUE_CLOSE'][:8]
    if len(due) != 24 or len(not_due) != 8:
        raise ValueError('ADAPT_BATCH_REQUIRES_ALL_FROZEN_24_DUE_AND_FIRST_8_NOT_DUE')
    picked = dict(target=due+not_due)
    picked.update({d:list(enumerate(panel['formal'][d][:8])) for d in DOMAINS[1:]})
    if any(len(rows)!=COUNTS[d] or len({r['index'] for _,r in rows})!=COUNTS[d] for d,rows in picked.items()):
        raise ValueError('ADAPT_BATCH_OFFLINE_REQUIRES_48_UNIQUE_WINDOWS_IN_FIXED_ORDER')
    return picked


def input_hash(value):
    if isinstance(value,dict):
        return {key:input_hash(item) for key,item in value.items()}
    array = np.asarray(value)
    return dict(shape=list(array.shape),dtype=str(array.dtype),sha256=bytes_sha(array))


def prepare(eval_spec_pin, tokenizer_pin, root, resize_with_pad):
    import jax

    spec = read_pin(eval_spec_pin)
    panel = read_pin(spec['panel'])
    if (jax.default_backend()!='cpu' or len(jax.devices())!=1 or root.stat().st_uid!=os.getuid()
            or any(root.iterdir()) or panel['bindings']!=original.bindings(spec)
            or sha(tokenizer_pin['path'])!=tokenizer_pin['sha256']):
        raise PermissionError('ADAPT_BATCH_INPUT_FREEZE_REQUIRES_CPU_AND_EXACT_REAL_SITE_ASSETS')
    selected = selected_rows(panel)
    ds = original.datasets(spec,resize_with_pad)
    tokenizer = Tokenizer(tokenizer_pin['path'])
    rows, noise = {},{}
    for domain, samples in selected.items():
        items = []
        for ordinal,row in samples:
            index = row['index']
            if ds[domain].identity(index) != row['identity']:
                raise ValueError(f'ADAPT_BATCH_OFFLINE_INPUT_IDENTITY_CHANGED:{domain}:{ordinal}')
            batch = prepare_host_batch(ds[domain],tokenizer,domain,(ordinal,),(index,),training=False)
            key = f'{domain}_{ordinal:03d}'
            n = np.asarray(jax.random.normal(jax.random.PRNGKey(row['noise_id']),(1,HORIZONS[domain],32),
                                             dtype=np.float32))
            noise[key] = n
            items.append(dict(row,ordinal=ordinal,input_sha256=canonical_hash(input_hash(batch)),
                              noise_key=key,noise_sha256=bytes_sha(n)))
        rows[domain] = items
    noise_file = root/'NOISE.private.npz'
    with noise_file.open('xb') as stream:
        np.savez(stream,**noise)
        stream.flush()
        os.fsync(stream.fileno())
    result = dict(schema=SCHEMA,plan=contract.PLAN,original_panel=spec['panel'],eval_input=eval_spec_pin,
                  tokenizer_sha256=tokenizer_pin['sha256'],rows=rows,counts=COUNTS,
                  input_and_noise_sha256=canonical_hash(rows),noise=dict(path=str(noise_file),sha256=sha(noise_file)),
                  ordinary_per_variant=48,independent_restore_per_variant=48,optimizer_updates=0,
                  model_generations=0,native_requests=0,old_TEST_access=False)
    write(root/'OFFLINE_PANEL.private.json',result)
    return result


def main():
    from openpi_client.image_tools import resize_with_pad

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    parser.add_argument('sha256')
    args = parser.parse_args()
    card = read_pin(dict(path=args.spec,sha256=args.sha256))
    prepare(card['eval_input'],card['tokenizer'],Path(card['registered_root']).resolve(strict=True),resize_with_pad)


if __name__ == '__main__':
    main()
