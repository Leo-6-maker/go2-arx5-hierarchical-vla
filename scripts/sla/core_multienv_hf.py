"""Exchange only admitted Wave A training snapshots through the existing project dataset."""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import re
import time

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.utils import EntryNotFoundError

from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0294.e1.journal import Journal
from scripts.sla import core_multienv_train as contract
from scripts.sla.core_hopper_hf_train import EP, META, REPO, SHA, download, metadata, same_remote
from scripts.sla.hopper_parent_custody import digest, sync_dir, write_once
from scripts.sla.hopper_site import pinned
from scripts.sla.hopper_snapshot_relay import verified

LANE = 'multienv-snapshots-v1'
SCHEMA = 'SLA_CORE_MULTIENV_PUBLIC_TRAIN_INDEX_V1'
COLD_SCHEMA = 'SLA_CORE_MULTIENV_PUBLIC_TRAIN_INDEX_V2'
NEW_EP = re.compile(r'MEV1-TRAIN-F[12]-(?:NEAR|FAR)_(?:NEG|POS)-\d{2}-(?:HOLD-(?:red|blue)|PLACE-(?:red|blue)-(?:left|right))/(?:LABELS|frames-\d{3})\.npz')


def eligible_key(name):
    return (re.fullmatch(r'TRAIN-\d{3}',name) is not None
            and contract.PREFIX_COUNT<int(name[6:])<=contract.MAX_COUNT)


def public_path(key, name, sha):
    if not eligible_key(key) or not SHA.fullmatch(sha):
        raise ValueError('Unexpected Wave A snapshot identity')
    if name in META:
        return f'{LANE}/{key}/{name}'
    if not (EP.fullmatch(name) or NEW_EP.fullmatch(name)):
        raise ValueError('Non-TRAIN or unexpected calibration member')
    return 'objects/'+sha+'.npz'


def full_index(folder, count, population, seal_sha):
    seal, _, sizes = verified(folder,population,seal_sha)
    proof = read(folder/'ACTUAL.private.json')
    if (proof.get('schema') not in contract.JOINED_SCHEMAS or seal['count']!=count
            or contract.fingerprint(proof['registered_producer'])!=contract.ALLOWED_REGISTRY):
        raise ValueError('An authorized new TRAIN union is required')
    index = dict(schema=SCHEMA,plan=contract.PLAN,key=f'TRAIN-{count:03d}',count=count,
                 windows=seal['windows'],seal=seal_sha,registry=contract.ALLOWED_REGISTRY,
                 files={n:digest(folder/n) for n in sizes})
    if proof['schema']==contract.COLD_SCHEMA:
        index.update(schema=COLD_SCHEMA,cold_authority=contract.fingerprint(proof['cold_continuation']))
    return index


def publish(root, catalog, population, api, head):
    for record in sorted((catalog/'catalog').glob('TRAIN-???.json')):
        key = record.stem
        if not eligible_key(key):
            raise ValueError('Unexpected Wave A catalog member')
        receipt = read(record)
        done = root/'receipts'/(key+'.json')
        if done.is_file():
            if read(done)['seal']!=receipt['sha256']:
                raise ValueError('A published immutable snapshot changed')
            continue
        folder = catalog/'snapshots'/key
        if Path(receipt['path'])!=folder/'SEALED.private.json':
            raise ValueError('Publication must originate in the managed catalog')
        index = full_index(folder,int(key[6:]),population,receipt['sha256'])
        index_path = f'{LANE}/{key}/INDEX.json'
        files = {public_path(key,n,h):(folder/n,h) for n,h in index['files'].items()}
        remote = metadata(api,[index_path],head)
        if index_path in remote:
            revision = head
        else:
            existing = metadata(api,files,head)
            ops = []
            for dst,(src,sha) in files.items():
                if dst not in existing:
                    ops.append(CommitOperationAdd(dst,str(src)))
                elif not same_remote(existing[dst],src,sha):
                    raise ValueError('The remote content object does not match its SHA')
            ops.append(CommitOperationAdd(index_path,json.dumps(index,sort_keys=True).encode()))
            def upload():
                return api.create_commit(REPO,repo_type='dataset',parent_commit=head,operations=ops,
                                         commit_message='Seal Wave A cumulative '+key).oid
            revision = Journal(root/'journal','CORE-MULTIENV-TRAIN').once(
                key,dict(index_sha=contract.fingerprint(index),seal=index['seal']),upload)
        if read(download(index_path,revision,root/'cache'))!=index:
            raise ValueError('Wave A public index readback differs')
        remote = metadata(api,files,revision)
        if remote.keys()!=files.keys() or any(not same_remote(remote[n],src,h) for n,(src,h) in files.items()):
            raise ValueError('Wave A object identities differ after publication')
        result = dict(status='PUBLIC_TRAIN_SNAPSHOT_VERIFIED',key=key,seal=index['seal'],
                      repo=REPO,revision=revision,windows=index['windows'],objects=len(files))
        write_once(done,result)
        return result
    return dict(status='WAIT_CATALOG_SEAL')


def stage(root, population, api, revision):
    try:
        keys = sorted(row.path.rsplit('/',1)[-1] for row in api.list_repo_tree(
            REPO,LANE,revision=revision,repo_type='dataset'))
    except EntryNotFoundError:
        return dict(status='WAIT_PUBLIC_SEAL')
    for key in keys:
        if not eligible_key(key):
            raise ValueError('Unexpected member in the isolated Wave A exchange lane')
        done = root/'receipts'/(key+'.json')
        if done.is_file():
            continue
        p = f'{LANE}/{key}/INDEX.json'
        idx = read(download(p,revision,root/'cache'))
        supported = ((idx['schema']==SCHEMA and 'cold_authority' not in idx)
                     or (idx['schema']==COLD_SCHEMA and idx['cold_authority'] in contract.COLD_AUTHORITIES))
        if (not supported or idx['plan']!=contract.PLAN or idx['registry']!=contract.ALLOWED_REGISTRY
                or idx['key']!=key or idx['count']!=int(key[6:])
                or not META<=idx['files'].keys() or idx['files']['SEALED.private.json']!=idx['seal']):
            raise ValueError('Wave A exchange index identity differs')
        incoming = root/'incoming'/(key+'.'+idx['seal'])
        final = root/'staged'/key
        folder = final if final.is_dir() else incoming
        folder.mkdir(parents=True,exist_ok=True)
        for name,sha in idx['files'].items():
            route = public_path(key,name,sha)
            target = folder/name
            if not target.exists():
                if folder==final:
                    raise ValueError('A staged member has disappeared')
                fetched = download(route,revision,root/'cache').resolve(strict=True)
                if digest(fetched)!=sha:
                    raise ValueError('Downloaded Wave A member differs')
                target.parent.mkdir(exist_ok=True)
                os.link(fetched,target)
        if full_index(folder,idx['count'],population,idx['seal'])!=idx:
            raise ValueError('Received Wave A inventory differs')
        if folder==incoming:
            for episode in (r for r in folder.iterdir() if r.is_dir()):
                sync_dir(episode)
            sync_dir(folder)
            incoming.rename(final)
            sync_dir(final.parent)
        result = dict(status='HOPPER_STAGED_READER_REQUIRED',key=key,repo=REPO,revision=revision,
                      seal=idx['seal'],folder=str(final),count=idx['count'],windows=idx['windows'])
        write_once(done,result)
        return result
    return dict(status='WAIT_PUBLIC_SEAL')


def run(root, cfg):
    publishing = cfg['role']=='publish'
    if (cfg['plan']!=contract.PLAN or cfg['root']!=str(root) or cfg['uid']!=os.getuid()
            or cfg['host']!=os.uname().nodename or not time.time()<cfg['deadline']
            or (publishing and not os.environ.get('INVOCATION_ID'))):
        raise PermissionError('Unexpected Wave A exchange owner or deadline')
    source = Path(__file__).resolve().parents[2]
    if any(digest(source/name)!=sha for name,sha in pinned(cfg['source_members']).items()):
        raise PermissionError('Wave A exchange source closure changed')
    if publishing:
        catalog_cfg = pinned(cfg['catalog'])
        catalog = Path(catalog_cfg['root'])
        if (catalog_cfg['probe_only'] or catalog_cfg['producer']['sha256']!=contract.PRODUCER_SPEC
                or catalog_cfg['deadline']!=cfg['deadline']):
            raise PermissionError('The catalog is not registered for Wave A publication')
    for name in ('receipts','cache','incoming','staged'):
        (root/name).mkdir(exist_ok=True)
    if (root/'STOP').exists():
        return dict(status='OPERATOR_STOP')
    population = Path(cfg['population'])
    api = HfApi()
    with closing(OwnerLock(root/'exchange.lock')):
        while not (root/'STOP').exists() and time.time()<cfg['deadline']-120:
            repo = api.repo_info(REPO,repo_type='dataset')
            if repo.id!=REPO or repo.private or (publishing and api.whoami()['name']!='leoyuer'):
                raise PermissionError('Unexpected project dataset or authorizing account')
            outcome = publish(root,catalog,population,api,repo.sha) if publishing else stage(root,population,api,repo.sha)
            write(root/'HEARTBEAT.json',dict(**outcome,epoch=time.time(),pid=os.getpid()))
            if not publishing:
                return outcome
            time.sleep(1 if outcome['status']=='PUBLIC_TRAIN_SNAPSHOT_VERIFIED' else 45)
        result = dict(status='STOP_OR_DEADLINE',epoch=time.time(),published=len(list((root/'receipts').glob('TRAIN-*.json'))))
        write_once(root/'TERMINAL.json',result)
        return result


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('sha')
    args = parser.parse_args()
    os.umask(0o077)
    print(json.dumps(run(args.root,pinned(dict(path=str(args.root/'CONFIG.json'),sha256=args.sha)))))
