"""Bind the exact parent, unpack site paths and freeze the H200 visit and random clocks."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from go2arx5_vla.sla_adapter import adapt5k_checkpoint, adapt5k_contract, adapt5k_data
from go2arx5_vla.sla_adapter.scale20k_contract import DOMAINS, bank_sha256, canonical_hash, visits
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from scripts.sla.preflight_adapt5k_data import observation_id

PLAN = '171b5bedb290258954aac140157743e92e3b152e'
PARENT = '2bff547b6f2c3aca0e7e2fd8a790ce6b30ef1c5f33b004c6897e3083647358d5'
D2_SPEC = 'e1189c2288a68d430486d28f3612795cc4a408f2f2656ead34042bfb1127abf3'
LEDGER = '97824d4daab254134f9b3aabe1436a96c6b1e42fbcf9943c173ef95af260ae43'
STEPS = 20000
BASE = Path('/scratch/projects/CFP04/CFP04-CF-007/e1192548/GR00T-WBC/runs/WM02')
SPEC_NAME = 'sla-adapt5k-d1-single-lineage-vanda-20261120-01/slaadp2250-02/train.SPEC.private.json'
COUNTS = dict(target=2031, robodojo=210398, libero=209146)
BATCH = adapt5k_contract.BATCH


def relocate(value, base):
    """Move archived VANDA paths to an immutable local mount; preserve lookup keys."""
    if isinstance(value, dict):
        return {k: relocate(v, base) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate(v, base) for v in value]
    if isinstance(value, str) and Path(value).is_relative_to(BASE):
        return str(base / Path(value).relative_to(BASE))
    return value


def locations(spec, base):
    return {key: relocate(spec[key], base) for key in
            ('data', 'train_data_proof', 'parent_manifest', 'tokenizer', 'parent_params')}


def inputs(base):
    original = read_pin(dict(path=str(base/SPEC_NAME),sha256=D2_SPEC))
    return locations(original,base)


def check_samples(spec):
    from openpi_client.image_tools import resize_with_pad

    datasets = adapt5k_data.readers(spec['data'],resize_with_pad)
    receipt = read_pin(spec['train_data_proof'])
    ref = receipt['vanda']
    if (ref['input_sha256'] != receipt['core']['input_sha256'] or ref['total_windows'] != COUNTS
            or ref['source_tasks'] != receipt['core']['source_tasks']
            or {d:len(reader) for d,reader in datasets.items()} != COUNTS):
        raise ValueError('Reference data membership changed')
    actual = {}
    for domain in DOMAINS:
        reader = datasets[domain]
        actual[domain] = [dict(index=row['index'],identity=reader.identity(row['index']),
                              inputs=observation_id(reader[row['index']])) for row in ref['inputs'][domain]]
        if canonical_hash(actual[domain]) != canonical_hash(ref['inputs'][domain]):
            raise ValueError('Training reader sample differs in '+domain)
    if canonical_hash(actual) != ref['input_sha256']:
        raise ValueError('Training reader hash differs')
    return datasets,dict(sample_sha256=ref['input_sha256'],counts={d:len(v) for d,v in actual.items()},
                         windows=COUNTS,episodes=ref['independent_episodes'])


def parent_identity(parent, ledger):
    pin = dict(path=str(parent),sha256=PARENT)
    marker = read_pin(dict(path=str(parent/'COMPLETE.json'),sha256=PARENT))
    _, original = adapt5k_checkpoint.read_complete(pin,marker['binding'])
    clock = original['progress']
    data = ledger.read_bytes()
    if hashlib.sha256(data).hexdigest() != LEDGER:
        raise ValueError('Last parent updates ledger digest changed')
    last = json.loads(data.splitlines()[-1])
    if (last['phase_step'] != 2000 or clock['phase_step'] != last['phase_step']
            or clock['lineage_cursors'] != last['lineage_cursors'] or clock['lineage_total_updates'] != 2500
            or last['learning_rate'] != adapt5k_contract.phase_rates()[1999]):
        raise ValueError('Incorrect parent phase, cursors or learning rate')
    return original,last


def phase_rates(last_lr):
    phase = np.arange(STEPS,dtype='<f8')
    return 1e-6 + .5*(last_lr-1e-6)*(1+np.cos(np.pi*phase/(STEPS-1)))


def stage_target(dataset, start, seed):
    return adapt5k_contract.hierarchical_bank(dataset,stage_start=start,phase_stop=STEPS,seed=seed)


def freeze_banks(datasets, parent, ledger, seed, out):
    marker,last = parent_identity(parent,ledger)
    progress = marker['progress']
    target,source,libero = (datasets[d] for d in DOMAINS)
    catalogues = dict(robodojo=[dict(episode=r['id'],task=r['task'],split=r['split'],
                       admitted=r['valid'],window_count=r['windows']) for r in source.episodes.values()],
                      libero=list(libero.episodes.values()))
    rng_entry = next(r for r in marker['arrays'] if r['key']=="['rng']")
    rng_path = parent/rng_entry['file']
    if hashlib.sha256(rng_path.read_bytes()).hexdigest() != rng_entry['sha256']:
        raise ValueError('Parent random key changed')
    rng = np.load(rng_path,allow_pickle=False)
    arrays = dict(rates=phase_rates(last['learning_rate']),rng=rng)
    arrays['target'],report = stage_target(target,0,seed)
    if (report['new_quota_per_update'],report['independent_new_episodes']) != (12,3):
        raise ValueError('Initial D2 quota changed')
    for domain,rows in catalogues.items():
        begin = progress['phase_cursors'][domain]
        bank = visits(rows,domain)
        count = STEPS*BATCH[domain]
        arrays[domain] = bank[begin:begin+count]
        absolute = progress['lineage_cursors'][domain]
        arrays[domain+'_noise_cursors'] = np.arange(absolute,absolute+count,dtype='<u4')
        if len(arrays[domain]) != count or arrays[domain].max() >= len(datasets[domain]):
            raise ValueError('Inherited source sampler does not cover the complete H200 plan')
    result = {}
    for key,value in arrays.items():
        with (out/(key+'.npy')).open('xb') as f:
            np.save(f,value,allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        result[key] = dict(sha256=bank_sha256(value),shape=list(value.shape),dtype=str(value.dtype))
    return dict(schema='HOPPER_ADAPT20K_INITIAL_BANKS_V1',plan=PLAN,parent=PARENT,parent_ledger=LEDGER,
                ancestor=progress,parent_rng_array_sha256=rng_entry['sha256'],lr_start=last['learning_rate'],
                lr_end=float(arrays['rates'][-1]),arrays=result,target=report,sources_frozen_through=STEPS)
