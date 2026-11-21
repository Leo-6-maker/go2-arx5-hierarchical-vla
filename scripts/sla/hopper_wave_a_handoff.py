"""Certify a cumulative Wave A seal on Hopper without changing the live course."""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import re
import time

from core_hopper_mixed_readback import inputs
from hopper_mixed_promotion import promote
from hopper_reader_compat import compare

from scripts.pi0292.durable import OwnerLock, read
from scripts.sla.core_multienv_train import PLAN as WAVE_PLAN
from scripts.sla.hopper_parent_custody import PLAN, digest, sync_dir, write_once
from scripts.sla.hopper_site import DEADLINE, pinned
from scripts.sla.hopper_snapshot_relay import verified


def pin(path):
    return dict(path=str(path), sha256=digest(path))


def catalog(folder, population, sha):
    seal, manifest, inventory = verified(folder, population, sha)
    return dict(path=str(folder/'SEALED.private.json'), sha256=sha, membership=seal['episode_sha256'],
                count=seal['count'], windows=manifest['windows']), inventory


def place_verified(base, source, key, salt, inventory):
    final = base/'mixed-verified'/key
    if final.is_dir():
        return final
    stage = final.with_name(key+'.incoming.'+salt)
    stage.mkdir(mode=0o700, exist_ok=True)
    for name in inventory:
        old, new = source/name, stage/name
        new.parent.mkdir(exist_ok=True)
        if new.is_symlink() or (new.exists() and not new.samefile(old)):
            raise PermissionError('Existing staging member changed')
        if not new.exists():
            os.link(old,new)
    for folder in stage.iterdir():
        if folder.is_dir():
            sync_dir(folder)
    sync_dir(stage)
    stage.rename(final)
    sync_dir(final.parent)
    return final


def handoff(base, core_pin, stage_pin, previous_key):
    if (os.uname().nodename != 'hopper-l-01' or os.environ.get('PBS_JOBID')
            or os.environ.get('CUDA_VISIBLE_DEVICES') or time.time() >= DEADLINE):
        raise PermissionError('Expected the bounded nontraining Hopper data role')
    reader = base/'mixed-reader-multienv-a2e8f7a0-02'
    original = base/'qualification-source-b9581196'
    scheduler = base/'scheduler-multienv-73617aa2'
    population = reader/'source/population.json'
    view = base/'foundation-layout-v2'
    ready = base/'train-ready'
    if any((p/'STOP.json').exists() for p in (base, view, base/'HOPPER-ADAPT20K-V1')):
        raise PermissionError('Original data or training lineage was stopped')
    prior_file = reader/'verified-inputs'/(previous_key+'.json')
    prior = read(prior_file)
    core = pinned(core_pin)
    stage = pinned(stage_pin)
    key = stage['key']
    seal = stage['seal']
    if (not re.fullmatch(r'TRAIN-[0-9]{3}', key) or not re.fullmatch(r'TRAIN-[0-9]{3}', previous_key)
            or int(key[6:]) != int(previous_key[6:])+4
            or core['status'] != 'CORE_REAL_READER_PASS' or core['probe_only']
            or any(core[key] for key in ('updates','native_requests','model_generations'))
            or stage['status'] != 'HOPPER_STAGED_READER_REQUIRED'
            or Path(stage['folder']) != base/'wave-a-train-staging-3988eb4a/staged'/key
            or core['actual']['seal'] != seal or core['actual']['episodes'] != stage['count']
            or core['actual']['windows'] != stage['windows'] or core['old_prefix_windows'] != prior['actual']['windows']):
        raise PermissionError('Expected an actual extension of the original cold-verified TRAIN prefix')
    with closing(OwnerLock(scheduler/'data-ingress.lock')):
        proof = reader/'verified-inputs'/(key+'.json')
        if proof.is_file():
            old = read(proof)
            if old['reader_result_sha256'] != core_pin['sha256'] or old['actual'] != core['actual']:
                raise ValueError('Prior immutable certification differs')
            return promote(view,original,reader,population,pin(proof))
        profile = compare(original/'source',reader/'source')
        cfg = dict(mode='source',capsule=str(reader),d2=pin(reader/'LOCAL_INPUT.private.json'),population=pin(population))
        before, _ = catalog(ready/previous_key,population,prior['actual']['seal'])
        source = Path(stage['folder'])
        after, inventory = catalog(source,population,seal)
        previous = inputs(cfg,before,None)
        actual = inputs(cfg,after,previous)
        if (previous != prior['actual'] or actual != core['actual']
                or set(actual['cases'])-set(previous['cases']) != set(core['new_cases'])):
            raise ValueError('Hopper decoded inputs differ from the Core reader or prior immutable prefix')
        result = reader/(key+'_HOPPER_COLD_READBACK.private.json')
        cpu = dict(status='ACTUAL_HOPPER_COLD_READER_MATCH',actual=actual,reader=profile,
                   previous_windows=previous['windows'],members=len(inventory),
                   bytes=sum(inventory.values()),model_generations=0,native_requests=0,
                   optimizer_updates=0,optimization_visits=0)
        write_once(result,cpu)
        final = place_verified(base,source,key,seal,inventory)
        catalog(final,population,seal)
        destination = dict(plan=PLAN,status='HOPPER_INGRESS_STAGED',folder=str(final),
            count=actual['episodes'],windows=actual['windows'],hf_revision=stage['revision'],
            manifest_sha256=actual['manifest'],seal_sha256=seal,members=cpu['members'],bytes=cpu['bytes'])
        write_once(proof,dict(plan=PLAN,canonical_plan=WAVE_PLAN,status='MIXED_TRAIN_CORE_READ_AND_HOPPER_STAGED',
            qualified_reader=profile['actual_reader_runtime_revision'],reader_site='corelab-4080s',
            reader_result_sha256=core_pin['sha256'],original_prefix_windows=prior['original_prefix_windows'],
            actual=actual,destination=destination,hopper_result=pin(result),optimizer_updates=0,
            model_generations=0,native_requests=0,course_switches=0))
        certificate = promote(view,original,reader,population,pin(proof))
        return dict(status='HOPPER_WAVE_A_READY',key=key,certificate=certificate,
                    ready=str(ready/key),confirmed_members=cpu['members'],bytes=cpu['bytes'],
                    previous_windows=cpu['previous_windows'],updates=0,optimization_visits=0,
                    new_CPU_jobs=0,new_GPU_jobs=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('core', type=Path)
    parser.add_argument('core_sha')
    parser.add_argument('stage', type=Path)
    parser.add_argument('stage_sha')
    parser.add_argument('previous_key')
    args = parser.parse_args()
    os.umask(0o077)
    result = handoff(args.base,dict(path=str(args.core),sha256=args.core_sha),
                    dict(path=str(args.stage),sha256=args.stage_sha),args.previous_key)
    print(json.dumps(result))
