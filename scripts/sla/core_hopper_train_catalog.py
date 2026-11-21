"""Expose one append-only Core TRAIN snapshot sequence across the registered expert versions."""
import argparse
from contextlib import closing
import os
from pathlib import Path
import time

from scripts.pi0292.durable import OwnerLock, read, write
from scripts.sla.hopper_parent_custody import PLAN, digest, write_once
from scripts.sla.hopper_snapshot_producers import ORIGINAL, ORIGINAL_SOURCE, RED_PLACE_SPEC, REPAIRS, SERIES, check_case
from scripts.sla.hopper_site import T0, DEADLINE, pinned
from core_hopper_snapshot import publish


def sources(cfg):
    roots = {}
    all_specs = {ORIGINAL,*REPAIRS}
    before_series = all_specs-set(SERIES)
    if set(cfg['producers']) not in (before_series-{RED_PLACE_SPEC},before_series,all_specs):
        raise ValueError('Missing or unregistered TRAIN source')
    for sha, folder in cfg['producers'].items():
        root = Path(folder)
        spec = pinned(dict(path=str(root/'EXECUTION_SPEC.json'),sha256=sha))
        source = ORIGINAL_SOURCE if sha==ORIGINAL else REPAIRS[sha][1]
        if (spec['source_sha']!=source or spec['registered_root']!=str(root)
                or spec['goal_plan_sha']!=PLAN or spec['campaign_deadline_epoch']!=DEADLINE
                or spec['compiled_population']!=cfg['population'] or spec['scope']!='SLA_CORE_HOPPER_V1_EXPERT'):
            raise ValueError('Expert population or source differs from its actual scope')
        roots[sha] = dict(root=root,spec=sha,source=source)
    return roots


def admissions(roots, allowed):
    rows, origins, paid = {}, {}, set()
    for spec, origin in roots.items():
        folder = origin['root']
        records = folder/'RECORDS.private.json'
        for row in read(records) if records.is_file() else ():
            if row['split']!='TRAIN':
                continue
            eid=row['condition']
            check_case(eid,spec,origin['source'])
            if eid not in allowed or eid in paid or row['requests']!=1:
                raise ValueError('Duplicate or unregistered paid TRAIN request')
            paid.add(eid)
            if not row['admitted']:
                continue
            path=folder/'admission/TRAIN'/eid/'EPISODE.private.json'
            if (not row['task_and_mobile_success'] or row['hard_STOP'] or row['technical_failure']
                    or row['episode']['path']!=str(path) or digest(path)!=row['episode']['sha256']):
                raise ValueError('Invalid admitted terminal '+eid)
            if row['episode']['sha256'] in rows:
                raise ValueError('Duplicate episode digest')
            rows[row['episode']['sha256']]=row
            origins[eid]=origin
    return rows,origins,len(paid)


def record_seal(root, folder, previous, expected, aggregate):
    file=folder/'SEALED.private.json'
    seal=read(file)
    members=seal['episode_sha256']
    if (seal['plan']!=PLAN or seal['split']!='TRAIN' or members!=expected
            or members[:len(previous)]!=previous or not len(previous)<len(members)<=240
            or seal['count']!=len(members) or folder.name!=f'TRAIN-{len(members):03d}'
            or digest(folder/'ACTUAL.private.json')!=seal['admission_sha256']
            or digest(folder/'MANIFEST.private.json')!=seal['manifest_sha256']):
        raise ValueError('Conflicting cumulative TRAIN seal')
    receipt=dict(path=str(file),sha256=digest(file),count=len(members),windows=seal['windows'],
                 membership=members,epoch=seal['epoch'],aggregate=aggregate)
    write_once(root/'catalog'/(folder.name+'.json'),receipt)
    return receipt


def tick(root,cfg,roots,allowed):
    rows,origins,paid=admissions(roots,allowed)
    entries=sorted((root/'catalog').glob('TRAIN-???.json'))
    latest=read(entries[-1]) if entries else dict(membership=[],epoch=T0,aggregate=False,count=0)
    previous=latest['membership']
    if any(sha not in rows for sha in previous):
        raise ValueError('A previously published admission disappeared')
    if entries and pinned({'path':latest['path'],'sha256':latest['sha256']})['episode_sha256']!=previous:
        raise ValueError('Previously published seal changed')
    def admitted_order(sha):
        row=rows[sha]
        return Path(row['episode']['path']).stat().st_mtime_ns,row['condition']
    if not latest['aggregate']:
        base=[key for key,r in rows.items() if origins[r['condition']]['spec']==ORIGINAL]
        repairs=[admitted_order(key) for key,r in rows.items() if origins[r['condition']]['spec']!=ORIGINAL]
        for legacy in sorted((Path(cfg['legacy_mount']['path']).parent/'snapshots').glob('TRAIN-???')):
            n=int(legacy.name[-3:])
            if n<=len(previous):
                continue
            members=read(legacy/'SEALED.private.json')['episode_sha256']
            if n>len(base):
                break  # A base seal may have appeared after this read of the records.
            if members!=base[:n]:
                raise ValueError('Legacy membership differs from its actual producer order')
            if repairs and min(repairs)<admitted_order(members[-1]):
                break
            if any(origins[rows[key]['condition']]['spec']!=ORIGINAL for key in members):
                raise ValueError('A repair was incorrectly attributed to the original producer')
            latest=record_seal(root,legacy,previous,members,False)
            previous=members
    pending=sorted(set(rows)-set(previous),key=admitted_order)
    order=[*previous,*pending]
    closed=all((v['root']/'RESULT.private.json').is_file() for v in roots.values())
    now=time.time()
    while len(order)>len(previous) and (len(order)-len(previous)>=4 or now>=latest['epoch']+21600 or closed or now>=DEADLINE):
        keys=order[:len(previous)+4]
        batch=[rows[key] for key in keys]
        if not any(origins[r['condition']]['spec']!=ORIGINAL for r in batch):
            break  # The original owner publishes this exact prefix.
        folder=root/'snapshots'/f'TRAIN-{len(keys):03d}'
        if not folder.exists():
            publish(root,None,batch,cfg,origins=origins)
        latest=record_seal(root,folder,previous,keys,True)
        previous=keys
    return dict(epoch=time.time(),pid=os.getpid(),closed=closed,terminal_TRAIN_requests=paid,
                admitted=len(rows),published=len(previous),aggregate=latest['aggregate'],
                registered_producer_admissions={k:sum(v['spec']==k for v in origins.values()) for k in roots})


def watch(root,sha):
    cfg=pinned(dict(path=str(root/'CONFIG.json'),sha256=sha))
    if (cfg['plan']!=PLAN or cfg['T0_epoch']!=T0 or cfg['deadline_epoch']!=DEADLINE or
            cfg['uid']!=os.getuid() or os.uname().nodename!=cfg['host'] or not os.environ.get('INVOCATION_ID')):
        raise PermissionError('Wrong Core snapshot owner or deadline')
    src=Path(__file__).resolve().parents[2]
    members=pinned(cfg['source_members'])
    if not members or any(digest(src/name)!=expected for name,expected in members.items()):
        raise ValueError('Snapshot catalog source differs from the registered version')
    legacy=pinned(cfg['legacy_mount'])
    if (legacy['producer_spec_sha256']!=ORIGINAL or legacy['producer']!=cfg['producers'][ORIGINAL]
            or legacy['C20_statistics_sha256']!=cfg['C20_statistics_sha256']
            or Path(cfg['shared_objects'])!=Path(cfg['legacy_mount']['path']).parent/'objects'):
        raise ValueError('Original source or frozen statistics changed')
    roots=sources(cfg)
    allowed={r['id'] for r in pinned(cfg['population'])['conditions']['TRAIN']}
    if len(allowed)!=240:
        raise ValueError('Unexpected TRAIN population')
    with closing(OwnerLock(root/'owner.lock')):
        while not (root/'STOP').exists():
            result=tick(root,cfg,roots,allowed)
            write(root/'HEARTBEAT.json',result)
            if (result['closed'] and result['admitted']==result['published']) or time.time()>=DEADLINE:
                write_once(root/'RESULT.json',result)
                break
            time.sleep(min(40,max(.1,DEADLINE-time.time())))


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('root',type=Path)
    cli.add_argument('config_sha256')
    args=cli.parse_args()
    os.umask(0o077)
    watch(args.root,args.config_sha256)
