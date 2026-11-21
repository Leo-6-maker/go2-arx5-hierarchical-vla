"""Exercise the current training reader on a sealed Core TRAIN prefix on CPU."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from openpi_client.image_tools import resize_with_pad

from go2arx5_vla.sla_adapter.adapt5k_data import AdaptTargetDataset
from scripts.sla.hopper_curriculum import WithCoreTrain
from scripts.sla.hopper_inputs import stage_target
from scripts.sla.hopper_parent_custody import digest, write_once
from scripts.sla.hopper_site import pinned
from scripts.sla.preflight_adapt5k_data import observation_id


def snapshot(d2, receipt, population):
    pin = dict(path=receipt['path'], sha256=receipt['sha256'])
    seal = pinned(pin)
    if seal['episode_sha256'] != receipt['membership'] or seal['windows'] != receipt['windows']:
        raise ValueError('Catalog and sealed admission disagree')
    return WithCoreTrain(d2, Path(pin['path']).parent, pin['sha256'], population)


def window(view, index):
    item = view[index]
    mask = {'base_0_rgb': True, 'left_wrist_0_rgb': True, 'right_wrist_0_rgb': False}
    if (item['image_mask'] != mask or item['state'].shape != (32,)
            or item['actions'].shape != (40, 32)
            or not np.isfinite(item['state']).all() or not np.isfinite(item['actions']).all()):
        raise ValueError('Wrong student image mask, state or actions')
    for name, enabled in mask.items():
        image = item['image'][name]
        if enabled and (image.shape != (224, 224, 3) or image.dtype != np.uint8):
            raise ValueError('Incomplete RGB capture: ' + name)
    ref = view.reference(index)
    age = ref['query_native_s'] - ref['source_native_s']
    if not np.isfinite(list(ref.values())).all() or age <= 0:
        raise ValueError('Invalid previous-command clock')
    identity = view.identity(index)
    record = [identity, ref, observation_id(item)]
    return json.dumps(record, allow_nan=False, sort_keys=True, separators=(',', ':')).encode(), age


def check(config):
    target_spec = pinned(config['d2'])
    target = target_spec['data']['target']
    d2 = AdaptTargetDataset(target['legacy'], target['captures'], resize_with_pad=resize_with_pad)
    d2.bind_state(target['statistics'])
    population = Path(config['population']['path'])
    pinned(config['population'])
    prior = pinned(config['prior_catalog'])
    receipt = pinned(config['catalog'])
    if (receipt['count'] <= prior['count'] or receipt['membership'][:prior['count']] != prior['membership']
            or len(receipt['membership']) != receipt['count']):
        raise ValueError('Expected a strictly extended published TRAIN prefix')
    previous = snapshot(d2, prior, population)
    view = snapshot(d2, receipt, population)
    if not len(view) == len(d2) + receipt['windows'] or len(previous) != len(d2) + prior['windows']:
        raise ValueError('Wrong window denominator')
    if any(view.identity(i) != d2.identity(i) for i in range(len(d2))):
        raise ValueError('Existing D2 window moved')

    stream = hashlib.sha256()
    ages = []
    conditions = Counter()
    for i in range(len(d2), len(view)):
        record, age = window(view, i)
        if i < len(previous) and window(previous, i)[0] != record:
            raise ValueError('An earlier sealed TRAIN input or reference changed')
        ages.append(age)
        conditions[view.identity(i)['episode_id']] += 1
        stream.update(record + b'\n')

    # This sampler uses a local RNG. The bank exists only in this CPU process.
    bank, counts = stage_target(view, 2000, target_spec['target_seed'])
    blocks = bank.reshape(18000, 64)
    quota = min(48, 4 * view.new_episodes)
    if (bank.min() < 0 or bank.max() >= len(view) or counts['new_quota_per_update'] != quota
            or not np.all((blocks >= view.original_windows).sum(axis=1) == quota)):
        raise ValueError('A target bank block violates its registered quota')
    simulated = dict(legacy=int(np.sum(bank < view.original_windows)),
                     d2_additions=int(np.sum((bank >= view.original_windows) & (bank < len(d2)))),
                     core_new=int(np.sum(bank >= len(d2))))
    return dict(status='CORE_TRAIN_CPU_READBACK', plan=config['plan'], source_members=config['source_members'],
                input_catalog_sha256=config['catalog']['sha256'], seal_sha256=receipt['sha256'],
                previous_seal_sha256=prior['sha256'], d2_identities=len(d2),
                unchanged_prior_full_inputs=len(previous)-len(d2), checked_new_windows=len(ages),
                windows_by_core_episode=dict(conditions), go2_windows=len(view),
                go2_episodes=len(view.manifest['episodes']), actual_reference_age_s=[min(ages), max(ages)],
                input_digest=stream.hexdigest(), hypothetical_phase2000=dict(seed=target_spec['target_seed'],
                blocks=len(blocks), slots_per_block=64, nonlegacy_per_block=quota, simulated_slots=simulated,
                bank_sha256=counts['sha256']), optimization_visits=0, optimizer_updates=0,
                model_generations=0, native_requests=0, native_ticks=0, course_switches=0)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('config_sha256')
    args = p.parse_args()
    cfg = pinned(dict(path=str(args.root/'CONFIG.private.json'), sha256=args.config_sha256))
    if digest(Path(__file__)) != cfg['worker_sha256']:
        raise ValueError('Readback worker changed')
    for name, expected in cfg['source_members'].items():
        if digest(Path(cfg['reader_source'])/name) != expected:
            raise ValueError('Registered reading or sampling source changed')
    start = time.time()
    result = check(cfg)
    write_once(args.root/'RESULT.private.json', dict(result, start_epoch=start, elapsed_s=time.time()-start))
