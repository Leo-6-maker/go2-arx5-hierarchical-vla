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


def admission_rows(producer, cards, spec_sha=contract.PRODUCER_SPEC, producer_source=contract.PRODUCER_SOURCE):
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
                or source['panel_entry_id']!=cid or source['source_sha']!=producer_source
                or source['campaign_root']!=str(producer) or source['goal_plan_sha']!=contract.PLAN
                or source['frozen_files'][source['model_path']]!=card['scene_sha256']
                or native['source_sha']!=producer_source or native['physical_requests']!=1
                or native['condition']['collection_episode']!=cid or native['model_generations'] or native['optimizer_updates']
                or native['actual_native_integrations']!=row['ticks'] or not 0<row['ticks']<=card['native_max_ticks']
                or ep['id']!=cid or ep['split']!='TRAIN' or ep['individual_pass'] is not True
                or ep['native_result_sha256']!=row['result']['sha256'] or ep['scene_sha256']!=card['scene_sha256']
                or ep['windows']!=row['episode']['windows'] or ep['windows']*10!=row['episode']['confirmed_P10_atoms']):
            raise ValueError('Wave A candidate lacks an admitted native terminal: '+cid)
        proof = dict(id=cid,instruction=ep['task'],windows=ep['windows'],confirmed_P10_atoms=10*ep['windows'],
                     scene_sha256=ep['scene_sha256'],reset=ep['reset'],native_result_sha256=ep['native_result_sha256'],
                     episode_sha256=row['episode']['sha256'],native_ticks=row['ticks'],admitted=True,
                     producer_spec=spec_sha,producer_source=producer_source)
        admissions.append((ep, proof, folder.parent))
    return admissions, len(paid)


def freeze(root, prefix, population, manifest, proof, hashes, registry, rows, authority=None, retained=None):
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
                direct = origin in (prefix,retained) and src.stat().st_dev==incoming.stat().st_dev
                obj = src if direct else object_file(src,ref['sha256'],root/'objects')
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
        if authority is not None:
            audit.update(schema=contract.COLD_SCHEMA,cold_continuation=authority)
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
    if authority is not None:
        frozen = read(target/'ACTUAL.private.json')
        if frozen['schema']!=contract.COLD_SCHEMA or frozen['cold_continuation']!=authority:
            raise ValueError('Previously sealed Wave A ordinal has different producer provenance')
    receipt = dict(path=str(path),sha256=digest(path),count=seal['count'],windows=seal['windows'],
                   membership=selected,epoch=seal['epoch'],aggregate=True)
    write_once(root/'catalog'/(target.name+'.json'),receipt)
    return receipt


def cold_catalog(config, producer, cards, registry, prior, population):
    from scripts.sla.core_multienv_scope import order
    from scripts.sla.core_multienv_wave_suffix import closed_storage_timeout, suffix

    config = config['cold']
    pin = config['producer']
    spec = pinned(pin)
    root = Path(spec['registered_root'])
    handoff = spec['multienv_cold_handoff']
    original_pin = handoff['producer']
    original = pinned(original_pin)
    terminal = pinned(handoff['terminal'])
    records = producer/'RECORDS.private.json'
    lease = read(root/'control'/'LEASE.json')
    if (original_pin['sha256']!=contract.PRODUCER_SPEC or Path(original_pin['path'])!=producer/'EXECUTION_SPEC.json'
            or Path(handoff['terminal']['path'])!=producer/'RESULT.private.json' or terminal['exit_code']
            or terminal['source_sha']!=contract.PRODUCER_SOURCE or terminal['unreconciled']
            or read(records)!=terminal['records'] or read(producer/'control'/'LEASE.json')['status']!='CLOSED'
            or Path(pin['path'])!=root/'EXECUTION_SPEC.json' or not spec.get('multienv_payload_storage')
            or any(spec[k]!=original[k] for k in ('scope','compiled_population','registered_panel','owner',
                                                  'mobile_world_lock','registry','goal_plan_sha'))
            or spec['deadline_epoch']>original['deadline_epoch']
            or lease['status'] not in ('ACTIVE','CLOSED') or lease['spec_sha256']!=pin['sha256']
            or lease['owner']!=original['owner']):
        raise PermissionError('Expected an actual closed Wave A predecessor and registered continuation')
    retained = terminal['records']
    timeout = None
    if 'closed_storage_predecessor' in handoff:
        timeout, next_case, deadline = closed_storage_timeout(original_pin,handoff['closed_storage_predecessor'])
        if handoff['conditions'] != [next_case] or spec['deadline_epoch'] > deadline:
            raise ValueError('Cold catalog must preserve the paid storage timeout and site window')
        retained = [*retained,timeout]
    roster = suffix(order(pinned(original['compiled_population'])),retained,
                    storage_timeout=timeout['condition'] if timeout else None)['pending']
    scheduled = handoff['conditions']
    if not scheduled or scheduled!=[r['condition'] for r in roster[:len(scheduled)]]:
        raise ValueError('Cold producer changed the original next conditions')
    authority = dict(schema=contract.COLD_AUTHORITY_SCHEMA,plan=contract.PLAN,
        original_spec=contract.PRODUCER_SPEC,original_terminal=handoff['terminal']['sha256'],
        original_records=digest(records),population=registry['population'],spec=pin['sha256'],source=spec['source_sha'],
        prior_admissions=[[row[1][k] for k in ('id','episode_sha256','native_result_sha256')] for row in prior],
        conditions=[cid for cid in scheduled if cid in cards])
    if (authority!=pinned(config['authority']) or contract.fingerprint(authority) not in contract.COLD_AUTHORITIES):
        raise PermissionError('Cold provenance is absent from the immutable verifier boundary')
    before = config['baseline']
    seal, _, _ = verified(Path(before['path']).parent,population,before['sha256'])
    expected = [row[1]['episode_sha256'] for row in prior]
    if Path(before['path']).name!='SEALED.private.json' or seal['episode_sha256'][contract.PREFIX_COUNT:]!=expected:
        raise ValueError('The old collector must seal every original admission before continuing')
    baseline = dict(count=seal['count'],epoch=seal['epoch'],membership=seal['episode_sha256'],path=before['path'])
    return root, {cid:cards[cid] for cid in authority['conditions']}, authority, baseline, int(timeout is not None)


def cold_snapshot_bytes(root, prefix, retained, manifest, rows):
    required = {}
    destination = root.stat().st_dev
    episodes = [(ep,prefix) for ep in manifest['episodes']]
    episodes.extend((ep,origin) for ep,_,origin in rows)
    for ep,origin in episodes:
        for ref in [ep['labels'],*ep['frame_shards']]:
            sha = ref['sha256']
            obj = root/'objects'/sha
            file = origin/ref['file']
            st = file.stat()
            if not obj.is_file() and not (origin in (prefix,retained) and st.st_dev==destination):
                required[sha]=st.st_size
    return sum(required.values())


def close_original(root, config, producer, cards, registry, prefix, population, manifest, proof, hashes):
    """Seal the retained producer's last batch after it and its old catalog have closed."""
    from scripts.sla.core_multienv_payload import capacity

    bridge = config['close_original']
    spec = pinned(config['producer'])
    terminal = pinned(bridge['terminal'])
    old_cfg = pinned(bridge['catalog'])
    old_root = Path(old_cfg['root'])
    old_catalog = pinned(bridge['catalog_terminal'])
    if (config.get('cold') or Path(bridge['terminal']['path'])!=producer/'RESULT.private.json'
            or read(producer/'control'/'LEASE.json')['status']!='CLOSED' or terminal['exit_code']
            or terminal['unreconciled'] or terminal['source_sha']!=contract.PRODUCER_SOURCE
            or terminal['records']!=read(producer/'RECORDS.private.json')
            or old_cfg.get('probe_only') or old_cfg['producer']!=config['producer'] or root==old_root
            or Path(bridge['catalog_terminal']['path'])!=old_root/'TERMINAL.json'
            or not old_catalog['producer_closed']):
        raise PermissionError('The original collector and catalog must close before the final TRAIN seal')
    rows, paid = admission_rows(producer,cards)
    latest = old_catalog['latest']
    old_path = Path(latest['path'])
    retained = old_root/'snapshots'/f"TRAIN-{latest['count']:03d}"
    seal, _, _ = verified(retained,population,latest['sha256'])
    count = seal['count']-contract.PREFIX_COUNT
    expected = [*hashes,*(r[1]['episode_sha256'] for r in rows[:count])]
    if (old_path!=retained/'SEALED.private.json' or count<0 or seal['episode_sha256']!=expected
            or not 0 <= len(rows)-count <= 4 or paid!=old_catalog['paid_TRAIN']
            or paid!=sum(r['requests'] for r in terminal['records'] if r['split']=='TRAIN')
            or len(rows)!=terminal['admitted_train']):
        raise ValueError('Final original TRAIN batch is not a contiguous extension of the old catalog')
    if len(rows)==count:
        return dict(status='ORIGINAL_TRAIN_ALREADY_COMPLETE',latest=latest,paid_TRAIN=paid)
    volumes = capacity(root,bridge['payload_storage'],spec['multienv'])
    extra = sum((origin/ref['file']).stat().st_size for ep,_,origin in rows[count:]
                for ref in [ep['labels'],*ep['frame_shards']])
    local = volumes['local']
    if not volumes['passed'] or local['free_disk'] < local['floor']+local['reserve']+extra:
        raise PermissionError('The split-volume reserves do not cover the last original TRAIN batch')
    preserved = [(ep,witness,retained) for ep,witness,_ in rows[:count]]
    latest = freeze(root,prefix,population,manifest,proof,hashes,registry,
                    [*preserved,*rows[count:]],retained=retained)
    return dict(status='FINAL_ORIGINAL_TRAIN_SEALED',latest=latest,paid_TRAIN=paid,
                additional_admissions=len(rows)-count,additional_bytes=extra,volumes=volumes)


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
    cold = config.get('cold')
    lock_root = Path(cold['control_root']) if cold else root
    if cold:
        from scripts.sla.core_multienv_payload import payload_volume

        if lock_root.stat().st_uid!=os.getuid() or not lock_root.is_relative_to(producer.parent) or lock_root.is_symlink():
            raise PermissionError('The NFS catalog must retain a separate Core local owner lock')
        storage = pinned(cold['producer'])['multienv_payload_storage']
        payload_volume(lock_root,dict(storage,root=str(root)))
    with closing(OwnerLock(lock_root/'owner.lock')):
        for name in ('catalog','snapshots','objects'):
            (root/name).mkdir(exist_ok=True)
        if config.get('close_original'):
            if probe or (root/'STOP').exists() or time.time()>=config['deadline']:
                raise PermissionError('Expected a live bounded catalog closure role')
            result = close_original(root,config,producer,cards,registry,prefix,population,manifest,proof,hashes)
            write_once(root/'TERMINAL.json',dict(result,epoch=time.time(),pid=os.getpid(),new_training_visits=0))
            return
        baseline = dict(count=contract.PREFIX_COUNT,epoch=config['started_at'],membership=hashes)
        retained = None
        if cold:
            original_rows, original_paid = admission_rows(producer,cards)
            cold_root, cold_cards, authority, baseline, carried_timeouts = cold_catalog(
                config,producer,cards,registry,original_rows,population)
            original_paid += carried_timeouts
            retained = Path(baseline['path']).parent
            original_rows = [(ep,witness,retained) for ep,witness,_ in original_rows]
        while True:
            if cold:
                rows, paid = admission_rows(cold_root,cold_cards,authority['spec'],authority['source'])
                rows = [*original_rows,*rows]
                paid += original_paid
            else:
                rows, paid = admission_rows(producer,cards)
            entries = sorted((root/'catalog').glob('TRAIN-???.json'))
            latest = read(entries[-1]) if entries else baseline
            exposed = latest['count']-contract.PREFIX_COUNT
            if latest['membership'] != [*hashes,*(r[1]['episode_sha256'] for r in rows[:exposed])]:
                raise ValueError('A previously sealed Wave A admission disappeared or moved')
            closed = ((cold_root if cold else producer)/'RESULT.private.json').is_file()
            ended = closed or (root/'STOP').exists() or time.time()>=config['deadline']
            while exposed<len(rows):
                pending = len(rows)-exposed
                if not (probe or ended or pending>=4 or time.time()>=latest['epoch']+21600):
                    break
                if not probe:
                    disk = shutil.disk_usage(root)
                    if cold:
                        needed = cold_snapshot_bytes(root,prefix,retained,manifest,rows[:exposed+4])
                    else:
                        needed = sum((folder/ep['labels']['file']).stat().st_size +
                                     sum((folder/f['file']).stat().st_size for f in ep['frame_shards'])
                                     for ep,_,folder in rows[exposed:exposed+4])
                    floor = max(50*1024**3,int(.15*disk.total))+8*1024**3
                    if disk.free<floor+config['inflight_reserve_bytes']+needed:
                        break
                latest = freeze(root,prefix,population,manifest,proof,hashes,registry,rows[:exposed+4],
                                authority=authority if cold else None,retained=retained)
                exposed = latest['count']-contract.PREFIX_COUNT
            status = dict(epoch=time.time(),pid=os.getpid(),paid_TRAIN=paid,new_admitted=len(rows),
                          sealed_new_members=exposed,producer_closed=closed,probe_only=probe,
                          source=authority['source'] if cold else contract.PRODUCER_SOURCE,
                          latest=latest,new_training_visits=0)
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
