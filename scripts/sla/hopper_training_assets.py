"""Direct custody of immutable training assets, before sample/read and restore gates."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from hopper_parent_custody import HOST, PARENT, PLAN, SSH, digest, remote, sync_dir, write_once

D2_SPEC = 'e1189c2288a68d430486d28f3612795cc4a408f2f2656ead34042bfb1127abf3'
SPLITS = {'target': (26, 2031), 'robodojo': (200, 210398), 'libero': (1261, 209146)}


def sha_file(path, expected):
    if digest(path) != expected:
        raise ValueError('Asset identity mismatch: ' + path.name)


def inventory(base, spec_path):
    sha_file(spec_path, D2_SPEC)
    spec = json.loads(spec_path.read_bytes())
    entries = {}
    def member(path, sha):
        name = str(path.relative_to(base))
        if ('..' in Path(name).parts or not name or not re.fullmatch('[0-9a-f]{64}', sha)
                or path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(base)):
            raise ValueError('Unsupported training member: ' + name)
        meta = dict(bytes=path.stat().st_size, sha256=sha)
        if entries.setdefault(name, meta) != meta:
            raise ValueError('Conflicting training asset: ' + name)

    def manifest(ref):
        path = Path(ref['path'])
        member(path, ref['sha256'])
        sha_file(path, ref['sha256'])
        return json.loads(path.read_bytes())

    def relative(root, row):
        member(root / row.get('file', row.get('path')), row['sha256'])

    member(spec_path, D2_SPEC)
    manifest(spec['parent_training_spec'])
    manifest(spec['train_data_proof'])
    foundation = manifest(spec['parent_manifest'])
    params = Path(spec['parent_params'])
    for row in foundation['files']:
        if row['file'].startswith('params/'):
            relative(params.parent, row)
    member(Path(spec['tokenizer']['path']), spec['tokenizer']['sha256'])
    data = spec['data']
    target = data['target']
    legacy = target['legacy']
    portable = legacy['portable_inputs']
    for row in manifest(portable)['members'].values():
        relative(Path(portable['path']).parent, row)
    manifest(target['statistics'])
    for capture in target['captures']:
        manifest(capture['admission'])
    counts = {}
    go2 = []
    for ref in [legacy['manifest'], legacy['admission'], *[x['manifest'] for x in target['captures']]]:
        package = manifest(ref)
        root = Path(ref['path']).parent
        if 'statistics' in package:
            relative(root, package['statistics'])
        for ep in package['episodes']:
            if ep['split'] != 'TRAIN':
                raise ValueError('Unexpected non-TRAIN target episode')
            go2.append((ep['id'], ep['windows']))
            for row in [ep['labels'], *ep['frame_shards']]:
                relative(root, row)
    counts['target'] = (len(set(eid for eid, _ in go2)), sum(n for _, n in go2))
    source = data['robodojo']
    root = Path(source['root'])
    package = manifest(source['manifest'])
    relative(root, package['statistics'])
    manifest(source['inherited_statistics'])
    raw = manifest(source['raw_binding'])['raw']
    train = [r for r in package['episodes'] if r['valid'] and r['split'] == 'TRAIN']
    if set(raw) != {r['id'] for r in train}:
        raise ValueError('R200 RGB and numeric memberships differ')
    for ep in train:
        if raw[ep['id']]['sha256'] != ep['raw_sha256']:
            raise ValueError('R200 raw digest mismatch')
        relative(root, raw[ep['id']])
        relative(root, ep['arrays'])
    counts['robodojo'] = len(train), sum(ep['windows'] for ep in train)
    libero = data['libero']
    rows = [r for r in manifest(libero['manifest'])['rows'] if r['split'] == 'TRAIN']
    for row in rows:
        for ref in [row['data'], *row['cameras'].values()]:
            relative(Path(libero['root']), ref)
    counts['libero'] = len(rows), sum(ep['window_count'] for ep in rows)
    if counts != SPLITS:
        raise ValueError('Frozen training population mismatch: ' + repr(counts))
    return dict(plan=PLAN, parent=PARENT, input_spec_sha256=D2_SPEC, source_base=str(base),
                source_spec=str(spec_path.relative_to(base)), counts=counts, entries=entries)


def check_run(root, deadline):
    if time.time() >= deadline or (root/'STOP').exists() or (root/'STOP.json').exists():
        raise RuntimeError('Staging stopped or expired')


def verify(base, packet, expected_sha):
    sha_file(packet, expected_sha)
    manifest = json.loads(packet.read_bytes())
    if manifest['plan'] != PLAN or manifest['parent'] != PARENT or manifest['input_spec_sha256'] != D2_SPEC:
        raise ValueError('Unexpected staging plan')
    if json.loads((base/'PARENT_BYTES_VERIFIED.private.json').read_bytes())['seal'] != PARENT:
        raise ValueError('Parent custody missing')
    incoming = base/'assets.incoming'
    final = base/'assets.verified'
    tree = final if final.exists() else incoming
    entries = manifest['entries']
    observed = {str(p.relative_to(tree)) for p in tree.rglob('*') if not p.is_dir()}
    if observed != set(entries):
        raise ValueError('Transferred membership mismatch')
    for name, row in entries.items():
        check_run(base, manifest['deadline'])
        path = tree/name
        if path.stat().st_size != row['bytes'] or digest(path, flush=True) != row['sha256']:
            raise ValueError('Destination byte check failed: ' + name)
    if tree == incoming:
        for folder in sorted((p for p in tree.rglob('*') if p.is_dir()), key=lambda p:-len(p.parts)):
            sync_dir(folder)
        sync_dir(tree)
        tree.rename(final)
        sync_dir(base)
    result = dict(status='TRAIN_ASSET_BYTES_VERIFIED', job_id=manifest['job_id'], plan=PLAN,
                  parent=PARENT, source_sha=manifest['source_sha'], counts=manifest['counts'],
                  inventory_sha256=expected_sha, members=len(entries),
                  bytes=sum(r['bytes'] for r in entries.values()), training_ready=False,
                  updates=0, visits=0, generations=0, native=0)
    write_once(base/'TRAIN_ASSET_BYTES_VERIFIED.private.json', result)
    print(json.dumps(result))


def transfer(root):
    launch = json.loads((root/'MOUNT.private.json').read_bytes())
    if os.environ.get('PBS_JOBID') != launch['job_id'] or launch['plan'] != PLAN:
        raise ValueError('Invalid CPU staging owner')
    for name, sha in launch['code_members'].items():
        sha_file(root/name, sha)
    base = Path(launch['source_base'])
    site = Path(launch['hopper_root'])
    deadline = launch['deadline_epoch']
    check_run(root, deadline)
    manifest = inventory(base, base/launch['source_spec'])
    manifest.update(job_id=launch['job_id'], source_sha=launch['source_sha'], deadline=deadline)
    entries = manifest['entries']
    packet = root/'ASSET_MEMBERS.private.json'
    write_once(packet, manifest)
    for name, row in entries.items():
        check_run(root, deadline)
        sha_file(base/name, row['sha256'])
    receipt = site/launch['job_id']
    incoming = site/'assets.incoming'
    remote('bash','-c','set -eu; umask 077; test ! -e "$3"; mkdir -p -- "$1" "$2"',
           'bash',str(receipt),str(incoming),str(site/'assets.verified'))
    sync = ['rsync','-rt','--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=','-e',shlex.join(SSH)]
    subprocess.run([*sync,*[str(root/n) for n in launch['code_members']],str(packet),
                    HOST+':'+str(receipt)+'/'],check=True)
    subprocess.run([*sync,'--from0','--files-from=-',str(base)+'/',HOST+':'+str(incoming)+'/'],
                   input=b'\0'.join(os.fsencode(n) for n in entries)+b'\0',check=True)
    check_run(root, deadline)
    check=' && '.join('test "$(sha256sum '+n+' | cut -d" " -f1)" = '+h
                      for n,h in launch['code_members'].items())
    reply=remote('bash','-c','set -eu; cd "$1"; '+check+
                 ' && exec python3 -B hopper_training_assets.py verify "$2" ASSET_MEMBERS.private.json "$3"',
                 'bash',str(receipt),str(site),digest(packet))
    result=json.loads(reply)
    if result['job_id'] != launch['job_id'] or result['inventory_sha256'] != digest(packet):
        raise ValueError('Wrong destination custody receipt')
    write_once(root/'RESULT.private.json',result)


if __name__ == '__main__':
    os.umask(0o077)
    mode, *args = sys.argv[1:]
    if mode == 'transfer':
        root = Path.cwd()
        with (root/'OWNER.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                transfer(root)
            except Exception as error:
                write_once(root/'FAILURE.private.json',dict(error=str(error),job=os.environ.get('PBS_JOBID')))
                raise
    elif mode == 'verify':
        verify(Path(args[0]), Path(args[1]), args[2])
    else:
        raise ValueError('Unknown staging operation')
