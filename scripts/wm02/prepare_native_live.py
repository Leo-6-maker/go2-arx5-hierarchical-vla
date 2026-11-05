"""Bind a read-only viewer to one existing remote run over the existing SSH route.

Only private local assets/config are written. No remote file, lease or service changes.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess


def remote_read(command, code, timeout=90):
    value = subprocess.run(command, input=code.encode(), capture_output=True, timeout=timeout)
    if value.returncode:
        raise RuntimeError('READONLY_TRANSPORT_EXIT_' + str(value.returncode) + ': ' + value.stderr.decode(errors='replace')[-2000:])
    return value.stdout


def prepare(command, spec_path, cache, output):
    from go2arx5_vla.world_model.live_source import digest
    if not command or not all(isinstance(v, str) and v for v in command):
        raise ValueError('EXPLICIT_EXISTING_COMMAND_ARGV_REQUIRED')
    if Path(output).exists():
        raise FileExistsError(output)
    # The exact spec and frozen assets identify the source; this does not grant it permission to run.
    code = '''from pathlib import Path
import json,hashlib
p=Path(SPEC_PATH)
s=json.loads(p.read_text())
pin=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for key in ('model_path','native_binary'):
 if pin(s[key])!=s['frozen_files'][s[key]]:raise ValueError('FROZEN_ASSET_CHANGED:'+key)
print(json.dumps(dict(execution_spec=dict(path=str(p),sha256=pin(p)),native_output=s['registered_root']+'/native-01',episode_id=Path(s['registered_root']).name,source_sha=s['source_sha'],scene_sha256=pin(s['model_path']),native_binary_sha256=pin(s['native_binary']),model_path=s['model_path'],model_size=Path(s['model_path']).stat().st_size)))
'''.replace('SPEC_PATH', repr(str(spec_path)))
    remote = json.loads(remote_read(command, code))
    if not 1 <= remote['model_size'] <= 128 * 1024 * 1024:
        raise ValueError('FINITE_NATIVE_SCENE_SIZE_REQUIRED')
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    scene = cache / (remote['scene_sha256'] + '.mjb')
    transferred = 0
    if not scene.exists() or digest(scene) != remote['scene_sha256']:
        code = ('from pathlib import Path;import sys,gzip;'
                'sys.stdout.buffer.write(gzip.compress(Path(' + repr(remote['model_path']) + ').read_bytes(),compresslevel=1))')
        compressed = remote_read(command, code)
        raw = gzip.decompress(compressed)
        if len(raw) != remote['model_size'] or hashlib.sha256(raw).hexdigest() != remote['scene_sha256']:
            raise ValueError('TRANSFERRED_SCENE_IDENTITY_MISMATCH')
        # Immutable asset cache: a corrupt same-name entry is not overwritten.
        with scene.open('xb') as stream:
            stream.write(raw)
        transferred = len(compressed)
    del remote['model_path']
    del remote['model_size']
    config = dict(episode_id=remote['episode_id'], source_sha=remote['source_sha'],
                  scene=dict(path=str(scene.resolve()), sha256=remote['scene_sha256']),
                  remote=remote, observer_command=command)
    with Path(output).open('x') as stream:
        json.dump(config, stream, indent=2)
    return dict(status='READONLY_LIVE_BINDING_PREPARED', episode_id=remote['episode_id'],
                scene_sha256=remote['scene_sha256'], compressed_transfer_bytes=transferred,
                remote_mutations=0, new_physics_integrations=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--command-json', type=Path, required=True, help='Private JSON argv for existing SSH route ending in python3 -')
    parser.add_argument('--remote-spec', required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(json.loads(args.command_json.read_text()), args.remote_spec, args.cache, args.output)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
