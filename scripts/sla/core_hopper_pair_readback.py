"""Read an exported MAIN state and compare the two registered sampling paths on CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
from openpi_client.image_tools import resize_with_pad

from go2arx5_vla.sla_adapter import adapt5k_data, hopper_serial_reference
from go2arx5_vla.sla_adapter.scale20k_contract import bank_sha256
from scripts.sla import hopper_checkpoint as checkpoint
from scripts.sla.hopper_curriculum import WithCoreTrain
from scripts.sla.hopper_engine import HopperEngine
from scripts.sla.hopper_inputs import BATCH, D2_SPEC, PARENT, PLAN, phase_rates, stage_target
from scripts.sla.hopper_parent_custody import digest, write_once
from scripts.sla.hopper_site import DEADLINE, pinned
from scripts.sla.hopper_training import continuation


def array(path):
    return np.load(path, allow_pickle=False, mmap_mode='r')


def read_backup(cfg):
    receipt = pinned(cfg['backup'])
    folder = Path(receipt['custody'])
    marker = pinned(dict(path=str(folder/'COMPLETE.json'), sha256=receipt['checkpoint']))
    bound = marker['binding']
    expected = checkpoint.progress(bound, receipt['phase'])
    if (marker['schema'] != checkpoint.SCHEMA or marker['progress'] != expected
            or bound['role'] != 'MAIN' or bound['source'] != cfg['training_source']
            or bound['parent'] != PARENT or bound['target_manifest'] != D2_SPEC
            or receipt['checkpoint'] != cfg['checkpoint'] or receipt['event_sha256'] != cfg['event']):
        raise ValueError('Hopper and VANDA checkpoint identity or clocks differ')
    rows = marker['arrays']
    if ([r['file'] for r in rows] != [f'{i:06d}.npy' for i in range(140)]
            or {p.name for p in folder.iterdir()} != {'COMPLETE.json', *(r['file'] for r in rows)}):
        raise ValueError('Expected the complete 140-array training state')
    leaves = {}
    for row in rows:
        path = folder/row['file']
        if digest(path) != row['sha256'] or row['key'] in leaves:
            raise ValueError('Changed or duplicated training leaf')
        leaves[row['key']] = path
    counts = {d: int(array(leaves[f"['optimizer']['{d}'][0].count"]).item()) for d in ('shared', *BATCH)}
    rng = array(leaves["['rng']"])
    if set(counts.values()) != {expected['lineage_updates']} or bank_sha256(rng) != bound['rng_sha256']:
        raise ValueError('Adam counts or RNG diverge from the saved phase')
    return bound, rng, dict(phase=receipt['phase'], checkpoint=receipt['checkpoint'],
                            event=receipt['event_sha256'], leaves=len(leaves), adam=counts, clocks=expected)


def check_pair(cfg):
    old, rng, saved = read_backup(cfg)
    bank_path = Path(cfg['banks'])
    proof = pinned(dict(path=str(bank_path/'BANKS.private.json'), sha256=checkpoint.BANKS))
    banks = {}
    for name, ref in proof['arrays'].items():
        val = array(bank_path/(name+'.npy'))
        if bank_sha256(val) != ref['sha256'] or list(val.shape) != ref['shape'] or str(val.dtype) != ref['dtype']:
            raise ValueError('Changed frozen bank: ' + name)
        banks[name] = val
    if (not np.array_equal(rng, banks['rng']) or old['target_bank_sha256'] != bank_sha256(banks['target'])
            or old['rates_sha256'] != bank_sha256(banks['rates'])
            or not np.array_equal(banks['rates'], phase_rates(proof['lr_start']))):
        raise ValueError('Restored random key, target bank or rate schedule changed')
    spec = pinned(cfg['d2'])
    target = spec['data']['target']
    d2 = adapt5k_data.AdaptTargetDataset(target['legacy'], target['captures'], resize_with_pad=resize_with_pad)
    d2.bind_state(target['statistics'])
    view = WithCoreTrain(d2, Path(cfg['snapshot']), cfg['seal'], Path(cfg['population']))
    start, stop = 2000, 4000
    main_indices, stats = stage_target(view, start, spec['target_seed'])
    main_sha = bank_sha256(main_indices)
    if main_sha != cfg['core_probe_bank'] or stats['sha256'] != main_sha:
        raise ValueError('VANDA replay and prior Core sample sequence differ')
    main = dict(old, target_origin=start, target_manifest=view.seal['manifest_sha256'], target_bank_sha256=main_sha)
    control = dict(old, role='CONTROL2000', role_start=start)
    continuation(old, main, start)
    continuation(old, control, start)

    # Use the deployed index method without constructing a model or optimizer.
    left = SimpleNamespace(banks=dict(banks, target=main_indices), target_origin=start, proof=proof)
    right = SimpleNamespace(banks=banks, target_origin=0, proof=proof)
    chains = {d:hashlib.sha256() for d in ('robodojo','libero')}
    samples = {}
    main_core = main_legacy = control_legacy = 0
    for step in range(start, stop):
        for domain, width in BATCH.items():
            li, lc = HopperEngine.indices(left, step, domain)
            ri, rc = HopperEngine.indices(right, step, domain)
            first = proof['ancestor']['lineage_cursors'][domain] + step*width
            if not lc == rc == first:
                raise ValueError('Absolute dropout, noise or auxiliary clock differs: ' + domain)
            if domain == 'target':
                if (li.min()<0 or li.max()>=len(view) or ri.min()<0 or ri.max()>=len(d2)
                        or int(np.sum(li < d2.original_windows))!=64-stats['new_quota_per_update']
                        or int(np.sum(ri < d2.original_windows))!=64-proof['target']['new_quota_per_update']):
                    raise ValueError('Go2 MAIN or CONTROL quota differs')
                main_legacy += int(np.sum(li < d2.original_windows))
                main_core += int(np.sum(li >= len(d2)))
                control_legacy += int(np.sum(ri < d2.original_windows))
                continue
            actual = banks[domain+'_noise_cursors'][step*width:(step+1)*width]
            if not np.array_equal(li, ri) or not np.array_equal(actual, np.arange(first, first+width)):
                raise ValueError('Source indices or noise positions differ: ' + domain)
            chains[domain].update(li.tobytes())
            if step in (start, stop-1):
                draw = hopper_serial_reference.frozen_draws(domain, width)
                a = tuple(np.asarray(t) for t in draw(rng, lc))
                b = tuple(np.asarray(t) for t in draw(banks['rng'], rc))
                if any(x.tobytes()!=y.tobytes() for x,y in zip(a,b,strict=True)):
                    raise ValueError('Matched CPU noise or FM time differs: '+domain)
                samples[f'{domain}/{step}'] = dict(first_cursor=lc, noise_sha256=bank_sha256(a[0]),
                                                   tau_sha256=bank_sha256(a[1]), visits=width)
    return dict(status='CPU_PAIRED_SAMPLER_READBACK', phase500=saved, snapshot_episodes=view.seal['count'],
                snapshot_seal=cfg['seal'], core_probe_bank=main_sha, candidate_boundary=start, stop_exclusive=stop,
                source_visits_per_domain=(stop-start)*32, source_sample_index_sha256={k:v.hexdigest() for k,v in chains.items()},
                sampled_noise_tau_cpu=samples, conditional_main=dict(legacy=main_legacy, new_core=main_core,
                    inherited_additions=(stop-start)*64-main_legacy-main_core),
                conditional_control=dict(legacy=control_legacy,inherited_additions=(stop-start)*64-control_legacy,new_core=0),
                first_lr=float(banks['rates'][start]), last_lr=float(banks['rates'][stop-1]),
                candidate_end_main=checkpoint.progress(main,stop), candidate_end_control=checkpoint.progress(control,stop),
                updates=0, optimization_visits=0, model_generations=0, native_requests=0, course_decisions=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('config_sha256')
    args = parser.parse_args()
    cfg = pinned(dict(path=str(args.config), sha256=args.config_sha256))
    job = os.environ['PBS_JOBID']
    live = json.loads(subprocess.check_output(['/opt/pbs/bin/qstat','-f','-F','json',job],
                                              text=True,timeout=20))['Jobs'][job]
    res = live['Resource_List']
    if (live['job_state']!='R' or live['project']!=cfg['project'] or int(res.get('ngpus',0))!=0
            or int(res['ncpus'])!=2 or live['Job_Owner'].split('@')[0]!=os.environ['USER']
            or os.environ['JAX_PLATFORMS']!='cpu' or os.environ.get('CUDA_VISIBLE_DEVICES')
            or cfg['plan']!=PLAN or time.time()>=DEADLINE or digest(Path(__file__))!=cfg['worker']):
        raise PermissionError('Expected the bounded site CPU allocation and registered worker')
    start = time.time()
    result = check_pair(cfg)
    write_once(args.config.parent/'RESULT.private.json', dict(result, job=job, start_epoch=start,
               elapsed_s=time.time()-start, worker=cfg['worker'], training_source=cfg['training_source']))
