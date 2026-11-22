"""Forward atomically deposited Core TRAIN snapshots through private VANDA custody."""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from scripts.pi0292.durable import OwnerLock, read, write
from scripts.sla.hopper_parent_custody import HOST, PLAN, SSH, digest, remote, sync_dir, write_once
from scripts.sla.hopper_snapshot_producers import check_admission_provenance

POPULATION = '9e483c85eb90cc2e9d1da66b9f3f8dff6606c6173b66755bcffcc4c7e5f91279'
C20 = 'f28d50e6eeea09b332d4bd159eca4333bbe23b185f95f1d336d45311e364ab3e'


def verified(folder, population, seal_sha, flush=False):
    """Verify the exact published inventory on both hosts, without model dependencies."""
    def pinned(name, expected):
        p = folder/name
        if digest(p, flush=flush) != expected:
            raise ValueError('File digest mismatch: ' + name)
        return read(p)
    if folder.is_symlink() or digest(population) != POPULATION:
        raise ValueError('Snapshot or registered population changed')
    cards = {r['condition_id']: r for r in read(population)['populations']['TRAIN']}
    seal = pinned('SEALED.private.json', seal_sha)
    manifest = pinned('MANIFEST.private.json', seal['manifest_sha256'])
    proof = pinned('ACTUAL.private.json', seal['admission_sha256'])
    episodes, cases, hashes = manifest['episodes'], proof['cases'], seal['episode_sha256']
    from scripts.sla import core_multienv_train as multienv

    joined = proof.get('schema') in multienv.JOINED_SCHEMAS
    if joined:
        additions = multienv.admitted_cards(proof,episodes,hashes)
        if cards.keys() & additions.keys():
            raise ValueError('Wave A changed the previous population')
        cards.update(additions)
    else:
        check_admission_provenance(proof)
    limit = multienv.MAX_COUNT if joined else 240
    if (seal['plan'] != PLAN or manifest['plan'] != PLAN
            or manifest['statistics_sha256'] != C20 or manifest['state'] != 'committed'
            or manifest['schema'] != 'SLA_TARGET_ADAPTATION_NEW_EXPERTS_V1'
            or manifest['representation'] != 'DENSE_REFERENCE_G7_V6'
            or not seal['split'] == manifest['split'] == 'TRAIN'
            or not 0 < len(episodes) == len(cases) == len(hashes) == seal['count'] == proof['admitted_episodes'] <= limit
            or not sum(ep['windows'] for ep in episodes) == seal['windows'] == proof['admitted_windows']
            or proof['dataset_sha256'] != seal['manifest_sha256']):
        raise ValueError('Wrong Core producer, TRAIN admission, statistics or data contract')
    inventory = {k: (folder/k).stat().st_size for k in ('SEALED.private.json', 'MANIFEST.private.json', 'ACTUAL.private.json')}
    ids, native, contents, scenes = set(), set(), set(), {}
    for ep, case, sha in zip(episodes, cases, hashes):
        eid = ep['id']
        card = cards[eid]
        members = [ep['labels'], *ep['frame_shards']]
        signature = tuple(row['sha256'] for row in members)
        scene = scenes.setdefault(card['scene_family'], ep['scene_sha256'])
        if (ep['split'] != 'TRAIN' or ep['individual_pass'] is not True or case['admitted'] is not True
                or (joined and eid in additions and ep['scene_sha256'] != card['scene_sha256'])
                or ep['label_kind'] != 'expert_query_plan_H40' or eid in ids or case['id'] != eid
                or ep['native_result_sha256'] in native or signature in contents or scene != ep['scene_sha256']
                or not ep['task'] == case['instruction'] == card['instruction'] or ep['family'] != card['family']
                or ep['reset'] != card['reset_seed'] or sha != case['episode_sha256']
                or any(case[k] != ep[k] for k in ('windows', 'reset', 'scene_sha256', 'native_result_sha256'))
                or not 0 < ep['windows']*10 == case['confirmed_P10_atoms']
                or not 0 < case['native_ticks'] <= card['native_max_ticks']):
            raise ValueError('Invalid original TRAIN admission for '+eid)
        ids.add(eid)
        native.add(ep['native_result_sha256'])
        contents.add(signature)
        if (folder/eid).is_symlink():
            raise ValueError('Symlinked episode')
        for member in members:
            name = member['file']
            path = folder/name
            if (name in inventory or Path(name).parent != Path(eid) or Path(name).as_posix() != name
                    or digest(path, flush=flush) != member['sha256']):
                raise ValueError('Unregistered, duplicated or changed capture member')
            inventory[name] = path.stat().st_size
        if flush:
            sync_dir(folder/eid)
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*')}
    if (actual != set(inventory) | ids or len(set(scenes.values())) != len(scenes)
            or len(set(hashes)) != len(episodes)):
        raise ValueError('Unexpected snapshot membership or duplicate admission')
    return seal, manifest, inventory


def receive(root, population, count, sha):
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    final = root/f'TRAIN-{count:03d}'
    incoming = root/(final.name+'.incoming.'+sha)
    with closing(OwnerLock(root/'receiver.lock')):
        folder = final if final.is_dir() else incoming
        seal, _, files = verified(folder, population, sha, flush=True)
        if count != seal['count']:
            raise ValueError('Wrong cumulative snapshot ordinal')
        if folder == incoming:
            previous = sorted(root.glob('TRAIN-[0-9][0-9][0-9]'))
            if previous:
                old = read(previous[-1]/'SEALED.private.json')
                if old['count'] >= count or seal['episode_sha256'][:old['count']] != old['episode_sha256']:
                    raise ValueError('Earlier committed TRAIN membership would be lost')
            sync_dir(folder)
            incoming.rename(final)
            sync_dir(root)
        return dict(plan=PLAN, seal_sha256=sha, manifest_sha256=seal['manifest_sha256'], count=count,
                    windows=seal['windows'], members=len(files), bytes=sum(files.values()), folder=str(final))


def transfer(folder, population, cfg, sha):
    seal, _, files = verified(folder, population, sha)
    count = seal['count']
    if folder.name != f'TRAIN-{count:03d}':
        raise ValueError('VANDA intake name differs from the sealed Core count')
    root = Path(cfg['hopper_ready'])
    incoming = root/(folder.name+'.incoming.'+sha)
    helper = Path(cfg['hopper_source'])
    env = ['env', 'PYTHONPATH='+str(helper), 'PYTHONDONTWRITEBYTECODE=1']
    check = 'set -eu; umask 077; cd "$1"; test "$(sha256sum code.sha256 | cut -d\' \' -f1)" = "$4"; sha256sum -c code.sha256 >/dev/null; if test -d "$2"; then echo existing; else mkdir -p -- "$3"; echo incoming; fi'
    decision = remote('bash', '-c', check, 'bash', str(helper), str(root/folder.name), str(incoming), cfg['code_sha256'])
    if decision == 'incoming':
        subprocess.run(['rsync', '-rlt', '--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=', '--files-from=-',
                        '-e', shlex.join(SSH), str(folder)+'/', HOST+':'+str(incoming)+'/'],
                       input='\n'.join(sorted(files))+'\n', text=True, check=True,
                       timeout=max(1, min(3600, int(cfg['deadline_epoch']-time.time()))))
    elif decision != 'existing':
        raise RuntimeError('Unknown Hopper destination state')
    reply = json.loads(remote(*(env+['python3', '-B', '-m', 'scripts.sla.hopper_snapshot_relay', 'receive',
                              str(root), str(helper/'population.json'), str(count), sha])))
    if (reply['seal_sha256'] != sha or reply['members'] != len(files) or reply['bytes'] != sum(files.values())
            or reply['windows'] != seal['windows'] or reply['count'] != count
            or reply['manifest_sha256'] != seal['manifest_sha256'] or reply['plan'] != PLAN):
        raise ValueError('VANDA and Hopper custody disagree')
    return reply


def watch(run):
    cfg = read(run/'MOUNT.private.json')
    job = os.environ.get('PBS_JOBID')
    if (not job or cfg['job'] != job or cfg['plan'] != PLAN or cfg['T0_epoch'] != 1795236112
            or cfg['deadline_epoch'] > cfg['T0_epoch']+14*86400 or time.time() >= cfg['deadline_epoch']):
        raise PermissionError('Wrong job or exhausted relay window')
    src = Path(cfg['vanda_source'])
    if digest(src/'code.sha256') != cfg['code_sha256']:
        raise ValueError('VANDA relay source binding changed')
    subprocess.run(['sha256sum', '--status', '-c', 'code.sha256'], cwd=src, check=True)
    live = json.loads(subprocess.check_output(['/opt/pbs/bin/qstat', '-f', '-F', 'json', job],
                      text=True, timeout=20))['Jobs'][job]
    res = live['Resource_List']
    if (live['job_state'] != 'R' or live['project'] != cfg['project']
            or live['Job_Owner'].split('@')[0] != os.environ['USER']
            or int(res.get('ngpus', 0)) != 0 or int(res['ncpus']) != 1
            or res['walltime'] != '06:00:00'):
        raise PermissionError('Expected the registered VANDA one-CPU transfer allocation')
    cfg = dict(cfg, deadline_epoch=min(cfg['deadline_epoch'],
               time.mktime(time.strptime(live['stime'], '%a %b %d %H:%M:%S %Y'))+21600))
    incoming, receipts = Path(cfg['vanda_inbox']), Path(cfg['receipt_root'])
    population = Path(cfg['vanda_source'])/'population.json'
    receipts.mkdir(parents=True, exist_ok=True)
    with closing(OwnerLock(receipts/'relay.lock')):
        sent = []
        while not (run/'STOP.json').exists() and time.time() < cfg['deadline_epoch']-90:
            for folder in sorted(incoming.glob('TRAIN-[0-9][0-9][0-9]')):
                if time.time() >= cfg['deadline_epoch']-90 or (run/'STOP.json').exists():
                    break
                if not (folder/'SEALED.private.json').is_file():
                    raise ValueError('Intake must publish sealed directories by atomic rename')
                sha = digest(folder/'SEALED.private.json')
                path = receipts/(folder.name+'.json')
                if path.exists():
                    if read(path)['seal_sha256'] != sha:
                        raise ValueError('VANDA custody changed after transfer')
                    continue
                result = transfer(folder, population, cfg, sha)
                write_once(path, result)
                sent.append(result)
            write(run/'HEARTBEAT.private.json', dict(job=job, pid=os.getpid(), epoch=time.time(),
                  received=len(list(receipts.glob('TRAIN-*.json'))), sent_in_job=len(sent)))
            time.sleep(min(45, max(0, cfg['deadline_epoch']-90-time.time())))
        write_once(run/'RESULT.private.json', dict(job=job, epoch=time.time(), sent=sent,
                   stop=(run/'STOP.json').exists(), new_updates=0, model_generations=0, native_requests=0))


if __name__ == '__main__':
    os.umask(0o077)
    cli = argparse.ArgumentParser()
    modes = cli.add_subparsers(dest='mode', required=True)
    watch_cli = modes.add_parser('watch')
    watch_cli.add_argument('run', type=Path)
    receiver = modes.add_parser('receive')
    receiver.add_argument('root', type=Path)
    receiver.add_argument('population', type=Path)
    receiver.add_argument('count', type=int, choices=range(1, 241), metavar='1..240')
    receiver.add_argument('seal_sha256')
    args = cli.parse_args()
    if args.mode == 'watch':
        watch(args.run)
    else:
        if not re.fullmatch('[a-f0-9]{64}', args.seal_sha256):
            raise ValueError('Invalid registered seal digest')
        print(json.dumps(receive(args.root, args.population, args.count, args.seal_sha256)))
