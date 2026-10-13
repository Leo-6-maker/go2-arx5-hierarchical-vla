"""Read-only host-qualified navigation; never starts SSH or a workload."""
import argparse
import json
from pathlib import Path
import socket
import subprocess


def resolve(registry, role, hostname, validate=False):
    rows = [row for row in registry['entries'] if row['role'] == role]
    if len(rows) != 1:
        raise ValueError('Role must identify exactly one registered entry')
    row = dict(rows[0])
    allowed = ['corelab-amd-server', 'corelab-4080s'] if row['executor_node'] == 'BOTH_CORELAB_NODES' else registry.get('validated_hostnames', {}).get(row['executor_node'], [row['executor_node']])
    if hostname not in allowed:
        return dict(**row, status='HOST_REQUIRED', observed_on=hostname,
                    local_absence_is_not_data_loss=True, automatically_connected=False)
    if not validate:
        return dict(**row, status='REGISTERED_NOT_LIVE_VERIFIED', observed_on=hostname)
    path = Path(row['path'])
    if not path.exists():
        return dict(**row, status='UNKNOWN_PATH_ABSENT_ON_REGISTERED_HOST', observed_on=hostname)
    stat = path.stat()
    result = dict(**row, status='PATH_OBSERVED', observed_on=hostname,
                  realpath=str(path.resolve()), st_dev=stat.st_dev, st_ino=stat.st_ino)
    if row.get('expected_git_sha'):
        query = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                               capture_output=True, text=True, timeout=10)
        result['actual_git_sha'] = query.stdout.strip()
        result['status'] = 'PATH_AND_GIT_IDENTITY_PASS' if query.returncode == 0 and query.stdout.strip() == row['expected_git_sha'] else 'GIT_IDENTITY_MISMATCH_OR_UNKNOWN'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('role')
    parser.add_argument('--registry', default=str(Path(__file__).with_name('PATH_REGISTRY.json')))
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    value = resolve(json.loads(Path(args.registry).read_text()), args.role, socket.gethostname(), args.validate)
    print(json.dumps(value, indent=2))
    if value['status'] in ('UNKNOWN_PATH_ABSENT_ON_REGISTERED_HOST', 'GIT_IDENTITY_MISMATCH_OR_UNKNOWN'):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
