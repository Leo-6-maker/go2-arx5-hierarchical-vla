"""Transfer only sealed Core TRAIN examples to VANDA through the public project dataset."""
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0294.e1.journal import Journal
from scripts.sla.hopper_parent_custody import PLAN, digest, sync_dir, write_once
from scripts.sla.hopper_site import DEADLINE, T0, pinned
from scripts.sla.hopper_snapshot_relay import POPULATION, verified

REPO = 'leoyuer/sla-core-hopper-train-v1'
META = {'SEALED.private.json','MANIFEST.private.json','ACTUAL.private.json'}
EP = re.compile(r'CHV1-TRAIN-F[12]-(?:HOLD-(?:red|blue)|PLACE-(?:red|blue)-(?:left|right))-\d{2}/(?:LABELS|frames-\d{3})\.npz')
SHA = re.compile('[0-9a-f]{64}')
CARD = b"""# Go2 + ARX5 mobile manipulation training demonstrations

This dataset contains individually admitted expert trajectories from the project's MuJoCo simulation.
Each snapshot index maps the original filenames to SHA256 digests. The three JSON receipts are
under the snapshot directory. RGB shards and action labels live under objects/<SHA256>.npz.
Each observation has an H40 supervision mask and a separate mask for commands consumed by the controller.
Snapshots are cumulative. Reuse the same content object when later snapshots repeat a trajectory.
"""


def public_path(key, name, sha):
    if not re.fullmatch(r'TRAIN-\d{3}',key) or not 1 <= int(key[6:]) <= 240 or not SHA.fullmatch(sha):
        raise ValueError('Unexpected snapshot or digest')
    if name in META:
        return f'snapshots/{key}/{name}'
    if not EP.fullmatch(name):
        raise ValueError('Unexpected public TRAIN member: '+name)
    return 'objects/'+sha+'.npz'


def metadata(api, paths, revision):
    paths = sorted(paths)
    result = {}
    for start in range(0,len(paths),64):
        rows = api.get_paths_info(REPO,paths[start:start+64],revision=revision,repo_type='dataset')
        result.update({r.path:r for r in rows})
    return result


def download(name, revision, cache):
    return Path(hf_hub_download(REPO,name,repo_type='dataset',revision=revision,cache_dir=cache))


def same_remote(row, src, sha):
    if row.size != src.stat().st_size:
        return False
    if row.lfs is not None:
        return row.lfs.sha256 == sha
    h = hashlib.sha1(b'blob '+str(row.size).encode()+b'\0')
    with src.open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):
            h.update(chunk)
    return h.hexdigest() == row.blob_id


def publish(root,cfg,api,head):
    catalog = Path(cfg['catalog'])
    rows = sorted((catalog/'catalog').glob('TRAIN-???.json'))
    for path in rows:
        key = path.stem
        done = root/'receipts'/(key+'.json')
        rec = read(path)
        if done.exists():
            if read(done)['seal'] != rec['sha256']:
                raise ValueError('Published catalog identity changed')
            continue
        folder = Path(rec['path']).parent
        seal, manifest, sizes = verified(folder,Path(cfg['population']),rec['sha256'])
        if (key != folder.name or seal['episode_sha256'] != rec['membership']
                or seal['count'] != rec['count'] or seal['windows'] != manifest['windows']):
            raise ValueError('Inconsistent TRAIN catalog')
        hashes = {n:digest(folder/n) for n in sizes}
        index = dict(schema='SLA_CORE_HOPPER_TRAIN_PUBLIC_V1',plan=PLAN,population=POPULATION,key=key,
                     count=seal['count'],windows=seal['windows'],seal=rec['sha256'],files=hashes)
        ipath = 'snapshots/'+key+'/INDEX.json'
        inventory = {public_path(key,n,h):(folder/n,h) for n,h in hashes.items()}
        if metadata(api,[ipath],head):
            revision = head
        else:
            existing = metadata(api,inventory,head)
            ops = []
            for dst,(src,h) in inventory.items():
                if dst not in existing:
                    ops.append(CommitOperationAdd(dst,str(src)))
                elif not same_remote(existing[dst],src,h):
                    raise ValueError('Conflicting public member: '+dst)
            ops.append(CommitOperationAdd(ipath,json.dumps(index,sort_keys=True).encode()))
            if not metadata(api,['README.md'],head):
                ops.append(CommitOperationAdd('README.md',CARD))
            def upload():
                return api.create_commit(REPO,repo_type='dataset',operations=ops,parent_commit=head,
                                         commit_message='Seal '+key+' simulation demonstrations').oid
            revision = Journal(root/'journal','CORE-TRAIN-PUBLIC-V1').once(key,dict(seal=rec['sha256'],key=key),upload)
        if read(download(ipath,revision,root/'cache')) != index:
            raise ValueError('Public index differs from the original TRAIN seal')
        remote = metadata(api,inventory,revision)
        if any(n not in remote or not same_remote(remote[n],src,h) for n,(src,h) in inventory.items()):
            raise ValueError('Post-commit TRAIN readback failed')
        write_once(done,dict(repository=REPO,revision=revision,key=key,seal=rec['sha256'],count=seal['count'],
                             windows=seal['windows'],files=len(hashes),bytes=sum(sizes.values())))
        return dict(status='COPIED',key=key,revision=revision)
    return dict(status='WAIT_SEAL')


def receive(root,cfg,api,revision):
    try:
        keys = sorted(r.path.rsplit('/',1)[-1] for r in
                      api.list_repo_tree(REPO,'snapshots',revision=revision,repo_type='dataset'))
    except EntryNotFoundError:
        return dict(status='WAIT_SEAL')
    for key in keys:
        if not re.fullmatch(r'TRAIN-\d{3}',key) or not 1 <= int(key[6:]) <= 240:
            raise ValueError('Unknown snapshot identity')
        done = root/'receipts'/(key+'.json')
        if done.exists():
            continue
        cache = root/'cache'
        idx = read(download(f'snapshots/{key}/INDEX.json',revision,cache))
        files = idx['files']
        if (idx['schema']!='SLA_CORE_HOPPER_TRAIN_PUBLIC_V1' or idx['plan']!=PLAN or idx['population']!=POPULATION
                or idx['key']!=key or idx['count']!=int(key[6:]) or not META<=files.keys()
                or idx['seal']!=files['SEALED.private.json']):
            raise ValueError('Invalid public dataset index')
        stage = root/'staging'/key
        stage.mkdir(parents=True,exist_ok=True)
        for name,sha in files.items():
            dst = public_path(key,name,sha)
            out = stage/name
            if not out.exists():
                blob = download(dst,revision,cache).resolve(strict=True)
                if digest(blob)!=sha:
                    raise ValueError('Downloaded TRAIN checksum mismatch')
                out.parent.mkdir(exist_ok=True)
                os.link(blob,out)
        seal,manifest,members = verified(stage,Path(cfg['population']),idx['seal'])
        if set(members)!=files.keys() or seal['count']!=idx['count'] or manifest['windows']!=idx['windows']:
            raise ValueError('Downloaded TRAIN inventory differs')
        # The running downstream reader accepts the original single-producer schema.
        mixed = 'schema' in read(stage/'ACTUAL.private.json')
        parent = root/'mixed-custody' if mixed else Path(cfg['inbox'])
        parent.mkdir(exist_ok=True)
        final = parent/key
        if final.exists():
            verified(final,Path(cfg['population']),idx['seal'])
            shutil.rmtree(stage)
        else:
            stage.rename(final)
            sync_dir(parent)
        write_once(done,dict(repository=REPO,revision=revision,key=key,seal=idx['seal'],count=idx['count'],
                             windows=idx['windows'],files=len(files),bytes=sum(members.values()),
                             status='MIXED_COLD_READER_REQUIRED' if mixed else 'VANDA_INBOX_VERIFIED'))
        return dict(status='COPIED',key=key,revision=revision,mixed=mixed)
    return dict(status='WAIT_SEAL')


def tick(root,role,sha):
    cfg=pinned(dict(path=str(root/'EXCHANGE.json'),sha256=sha))
    if (cfg['plan']!=PLAN or cfg['deadline']!=DEADLINE or cfg['T0']!=T0 or cfg['role']!=role
            or os.getuid()!=cfg['uid'] or any(digest(root/'source'/n)!=h for n,h in cfg['members'].items())):
        raise PermissionError('Wrong TRAIN exchange identity')
    if (root/'STOP').exists() or time.time()>=DEADLINE-120:
        return dict(status='STOP_LIMIT_OR_DEADLINE')
    if role=='publish' and (os.uname().nodename!=cfg['host'] or not os.environ.get('INVOCATION_ID')):
        raise PermissionError('Core publishing requires its managed service')
    if role=='receive' and not re.fullmatch(r'\d+\.stdct-mgmt-02',os.environ.get('PBS_JOBID','')):
        raise PermissionError('VANDA ingestion requires the project PBS owner')
    (root/'receipts').mkdir(exist_ok=True)
    with closing(OwnerLock(root/'exchange.lock')):
        api=HfApi();repo=api.repo_info(REPO,repo_type='dataset')
        if repo.private or repo.id!=REPO or (role=='publish' and api.whoami()['name']!='leoyuer'):
            raise PermissionError('Unexpected public repository or uploader')
        return (publish if role=='publish' else receive)(root,cfg,api,repo.sha)


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('root',type=Path)
    cli.add_argument('role',choices=('publish','receive'))
    cli.add_argument('sha')
    args=cli.parse_args()
    os.umask(0o077)
    while True:
        result=tick(args.root,args.role,args.sha)
        if args.role=='receive':
            print(json.dumps(result))
            break
        write(args.root/'HEARTBEAT.json',dict(**result,epoch=time.time(),pid=os.getpid()))
        if result['status']=='STOP_LIMIT_OR_DEADLINE':
            break
        time.sleep(1 if result['status']=='COPIED' else 90)
