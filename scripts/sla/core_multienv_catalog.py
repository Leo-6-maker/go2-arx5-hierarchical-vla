"""Seal the immutable TRAIN-059 prefix plus individually admitted Wave A examples."""
import argparse
from contextlib import closing
import os
from pathlib import Path
import shutil
import tempfile
import time

from scripts.pi0292.durable import OwnerLock, read, write
from scripts.sla import core_multienv_train as contract
from scripts.sla.hopper_parent_custody import PLAN, digest, sync_dir, write_once
from scripts.sla.hopper_site import pinned
from scripts.sla.hopper_snapshot_relay import verified


def producer_binding(config):
    pin = config['producer']
    producer = pinned(pin)
    pop = pinned(producer['compiled_population'])
    authority = pinned(producer['sim_only_project_authorization'])
    card_registry = contract.registry(pin['sha256'],producer,pop)
    cards = contract.registered_cards(card_registry)
    root = Path(producer['registered_root'])
    lease = read(root/'control/LEASE.json')
    if (Path(pin['path'])!=root/'EXECUTION_SPEC.json' or producer['scope']!='SLA_CORE_MULTIENV_V1_EXPERT'
            or authority['compiled_population']!=producer['compiled_population']
            or not set(cards)<=set(authority['case_ids'])
            or config['deadline']!=authority['deadline_epoch'] or producer['deadline_epoch']!=config['deadline']
            or lease['spec_sha256']!=pin['sha256'] or lease['owner']!=producer['owner']):
        raise ValueError('Wave A producer identity or site deadline changed')
    return root, cards, card_registry


def admission_rows(producer, cards):
    rows = producer/'RECORDS.private.json'
    admissions = []
    paid = set()
    for row in read(rows) if rows.is_file() else ():
        if row['split']!='TRAIN':
            continue
        cid = row['condition']
        card = cards[cid]
        if cid in paid or row['requests']!=1:
            raise ValueError('Conflicting paid Wave A TRAIN request')
        paid.add(cid)
        if not row['admitted']:
            continue
        case = producer/'cases'/cid
        folder = producer/'admission/TRAIN'/cid
        ep = pinned(row['episode'])
        native = pinned(row['result'])
        source = read(case/'EXECUTION_SPEC.json')
        lease = read(case/'control/LEASE.json')
        if (Path(row['result']['path'])!=case/'RESULT.private.json'
                or Path(row['episode']['path'])!=folder/'EPISODE.private.json'
                or not row['task_and_mobile_success'] or row['hard_STOP'] or row['technical_failure']
                or lease['status']!='CLOSED' or lease['worker_exit_code']
                or lease['spec_sha256']!=digest(case/'EXECUTION_SPEC.json')
                or not source['expert_training_eligible'] or source['collection_split']!='TRAIN'
                or source['panel_entry_id']!=cid or source['source_sha']!=contract.PRODUCER_SOURCE
                or source['frozen_files'][source['model_path']]!=card['scene_sha256']
                or native['source_sha']!=contract.PRODUCER_SOURCE or native['physical_requests']!=1
                or native['condition']['collection_episode']!=cid or native['model_generations'] or native['optimizer_updates']
                or native['actual_native_integrations']!=row['ticks'] or not 0<row['ticks']<=card['native_max_ticks']
                or ep['id']!=cid or ep['split']!='TRAIN' or ep['individual_pass'] is not True
                or ep['native_result_sha256']!=row['result']['sha256'] or ep['scene_sha256']!=card['scene_sha256']
                or ep['windows']!=row['episode']['windows'] or ep['windows']*10!=row['episode']['confirmed_P10_atoms']):
            raise ValueError('Wave A candidate lacks an admitted native terminal: '+cid)
        proof = dict(id=cid,instruction=ep['task'],windows=ep['windows'],confirmed_P10_atoms=10*ep['windows'],
                     scene_sha256=ep['scene_sha256'],reset=ep['reset'],native_result_sha256=ep['native_result_sha256'],
                     episode_sha256=row['episode']['sha256'],native_ticks=row['ticks'],admitted=True,
                     producer_spec=contract.PRODUCER_SPEC,producer_source=contract.PRODUCER_SOURCE)
        admissions.append((ep, proof, folder.parent))
    return admissions, len(paid)


def freeze(root, prefix, population, manifest, proof, hashes, registry, rows):
    from scripts.sla.core_hopper_snapshot import object_file

    target = root/'snapshots'/f'TRAIN-{contract.PREFIX_COUNT+len(rows):03d}'
    selected = [*hashes,*(entry[1]['episode_sha256'] for entry in rows)]
    if not target.exists():
        incoming = Path(tempfile.mkdtemp(prefix=target.name+'.incoming.',dir=target.parent))
        all_episodes = [(episode,prefix) for episode in manifest['episodes']]
        all_episodes.extend((episode,origin) for episode,_,origin in rows)
        for ep, origin in all_episodes:
            (incoming/ep['id']).mkdir()
            for ref in [ep['labels'],*ep['frame_shards']]:
                rel = Path(ref['file'])
                src = origin/rel
                if rel.parent!=Path(ep['id']) or src.is_symlink() or digest(src)!=ref['sha256']:
                    raise ValueError('Missing or changed admitted episode object')
                obj = src if origin==prefix else object_file(src,ref['sha256'],root/'objects')
                os.link(obj,incoming/rel)
            sync_dir(incoming/ep['id'])
        members = [episode for episode,_ in all_episodes]
        windows = sum(ep['windows'] for ep in members)
        merged = dict(manifest,episodes=members,windows=windows)
        write_once(incoming/'MANIFEST.private.json',merged)
        msha = digest(incoming/'MANIFEST.private.json')
        audit = dict(proof,schema=contract.SCHEMA,canonical_plan=contract.PLAN,prefix_seal=contract.PREFIX_SEAL,
                     registered_producer=registry,cases=[*proof['cases'],*(case for _,case,_ in rows)],
                     admitted_episodes=len(selected),admitted_windows=windows,dataset_sha256=msha)
        write_once(incoming/'ACTUAL.private.json',audit)
        write_once(incoming/'SEALED.private.json',dict(plan=PLAN,split='TRAIN',count=len(selected),windows=windows,
                   manifest_sha256=msha,admission_sha256=digest(incoming/'ACTUAL.private.json'),
                   epoch=time.time(),episode_sha256=selected))
        sha = digest(incoming/'SEALED.private.json')
        verified(incoming,population,sha)
        sync_dir(incoming)
        incoming.rename(target)
        sync_dir(target.parent)
    path = target/'SEALED.private.json'
    seal = read(path)
    if seal['episode_sha256']!=selected:
        raise ValueError('Previously sealed Wave A ordinal has different members')
    receipt = dict(path=str(path),sha256=digest(path),count=seal['count'],windows=seal['windows'],
                   membership=selected,epoch=seal['epoch'],aggregate=True)
    write_once(root/'catalog'/(target.name+'.json'),receipt)
    return receipt


def run(root, config, probe):
    if (config['plan']!=contract.PLAN or config['uid']!=os.getuid() or config['host']!=os.uname().nodename
            or config['probe_only']!=probe or config['root']!=str(root)
            or os.environ.get('CUDA_VISIBLE_DEVICES') or (not probe and not os.environ.get('INVOCATION_ID'))):
        raise PermissionError('Expected the Core nonphysical data-catalog owner')
    source = Path(__file__).resolve().parents[2]
    files = pinned(config['source_members'])
    if (not files or config['source_root']!=str(source)
            or any(digest(source/name)!=expected for name,expected in files.items())):
        raise ValueError('Catalog source closure changed before local publication')
    producer, cards, registry = producer_binding(config)
    prefix = Path(config['prefix'])
    population = Path(config['old_population'])
    seal,manifest,_ = verified(prefix,population,contract.PREFIX_SEAL)
    proof = read(prefix/'ACTUAL.private.json')
    hashes = seal['episode_sha256']
    contract.check_prefix(manifest['episodes'],proof['cases'],hashes)
    with closing(OwnerLock(root/'owner.lock')):
        for name in ('catalog','snapshots','objects'):
            (root/name).mkdir(exist_ok=True)
        while True:
            rows, paid = admission_rows(producer,cards)
            entries = sorted((root/'catalog').glob('TRAIN-???.json'))
            latest = read(entries[-1]) if entries else dict(count=contract.PREFIX_COUNT,epoch=config['started_at'],
                                                             membership=hashes)
            exposed = latest['count']-contract.PREFIX_COUNT
            if latest['membership'] != [*hashes,*(r[1]['episode_sha256'] for r in rows[:exposed])]:
                raise ValueError('A previously sealed Wave A admission disappeared or moved')
            closed = (producer/'RESULT.private.json').is_file()
            ended = closed or (root/'STOP').exists() or time.time()>=config['deadline']
            while exposed<len(rows):
                pending = len(rows)-exposed
                if not (probe or ended or pending>=4 or time.time()>=latest['epoch']+21600):
                    break
                if not probe:
                    disk = shutil.disk_usage(root)
                    needed = sum((folder/ep['labels']['file']).stat().st_size +
                                 sum((folder/f['file']).stat().st_size for f in ep['frame_shards'])
                                 for ep,_,folder in rows[exposed:exposed+4])
                    floor = max(50*1024**3,int(.15*disk.total))+8*1024**3
                    if disk.free<floor+config['inflight_reserve_bytes']+needed:
                        break
                latest = freeze(root,prefix,population,manifest,proof,hashes,registry,rows[:exposed+4])
                exposed = latest['count']-contract.PREFIX_COUNT
            status = dict(epoch=time.time(),pid=os.getpid(),paid_TRAIN=paid,new_admitted=len(rows),
                          sealed_new_members=exposed,producer_closed=closed,probe_only=probe,
                          source=contract.PRODUCER_SOURCE,latest=latest,new_training_visits=0)
            write(root/'HEARTBEAT.json',status)
            if ended or probe:
                write_once(root/'TERMINAL.json',status)
                return
            time.sleep(20)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('sha')
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    run(args.root,pinned(dict(path=str(args.root/'CONFIG.json'),sha256=args.sha)),args.probe)
