"""Bind the qualified metadata-reader assistance to one immutable new session."""
import argparse
import json
import os
from pathlib import Path
import resource
import time

from scripts.pi0292.durable import read, write
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.relay_io import read_available
from .directory_visibility_audit import force_server_attributes
from .science import validate_runtime, Ledger
from .plans import session_spec


def observe(directories, terminal, deadline, *, clock=time.time, sleep=time.sleep,
            synchronize=force_server_attributes):
    paths = [Path(p) for p in directories]
    if len(paths) != 2 or any(p.is_symlink() or not p.is_dir() for p in paths):
        raise PermissionError('EXACT_EXISTING_SESSION_INBOX_OUTBOX')
    began = clock()
    if not 0 < deadline - began <= 7200:
        raise PermissionError('SESSION_METADATA_READER_7200S_WALL_UPPER')
    scans = 0
    while clock() < deadline:
        if terminal():
            return dict(status='TERMINAL_MARKER_OBSERVED', scans=scans,
                        wall_s=clock() - began)
        for path in paths:
            if path.is_symlink():
                raise PermissionError('NO_PACKET_DIRECTORY_SYMLINK')
            synchronize(path)
            with os.scandir(path) as stream:
                if sum(1 for _ in stream) > 1280:
                    raise PermissionError('FIXED_PACKET_DIRECTORY_ENTRY_BOUND')
        scans += 1
        sleep(.2)
    return dict(status='ORIGINAL_SESSION_DEADLINE_OBSERVED', scans=scans,
                wall_s=clock() - began)


def run(root, label, binding_relative):
    root = Path(root)
    manifest = validate_runtime(root, 'hopper-neural')
    binding_path = root / binding_relative
    if (Path(binding_relative).is_absolute() or '..' in Path(binding_relative).parts
            or binding_path.is_symlink()
            or manifest['execution'].get('session_visibility_binding_sha256') != sha(binding_path)):
        raise PermissionError('SEALED_SESSION_VISIBILITY_BINDING')
    binding = read(binding_path)
    registry_path = root / 'plans' / label / 'REGISTRY.json'
    registry = read(registry_path)
    spec = session_spec(read(root / 'plans/SOURCE_EVALUATION_PLAN.json'), label, root=root)
    ready = read(root / 'pbs/MODEL_STATE.json')
    if (binding['run_uuid'] != root.name or binding['session'] != label
            or binding['registry_sha256'] != sha(registry_path)
            or binding['job_id'] != ready['job_id']
            or ready['status'] != 'MP1_NATIVE_MODEL_READY'
            or ready['label'] != label or binding['packet_mutations'] != 0
            or binding['maximum_CPU_s'] != 600
            or binding['deadline_epoch'] > registry['deadline_epoch']
            or binding['deadline_epoch'] > binding['created_epoch'] + min(7200, 900 * len(spec['cards']))
            or not binding['NFS_lookup_cache_observer_effect_disclosed']):
        raise PermissionError('EXACT_LIVE_FIXED_SESSION_METADATA_ASSISTANCE_ONLY')
    directory = root / 'evidence' / label / 'resident'
    output = root / 'evidence' / label / 'metadata-reader'
    output.mkdir(mode=0o700)
    started = time.process_time()
    ledger = Ledger(root, 'hopper')
    value = dict(status='FAILED_PRESERVED', packets_modified=0,
                 model_queries=0, physics_integrations=0,
                 NFS_lookup_cache_observer_effect_disclosed=True,
                 algorithm='QUALIFIED_STATX_FORCE_SYNC_AND_DIRECTORY_ENUMERATION_V2',
                 source_sha=manifest['code']['vla_commit'])
    def terminal():
        done = read_available(directory / 'STAGE_DONE.json')
        final = read_available(root / 'pbs' / binding['allocation'] / 'PBS_PROCESS_FINAL.json')
        if done is not None:
            if done != dict(run_uuid=root.name, job_id=binding['job_id'], label=label):
                raise PermissionError('SESSION_DONE_IDENTITY')
            return True
        if final is not None:
            if final['job_id'] != binding['job_id']:
                raise PermissionError('NATIVE_FINAL_IDENTITY')
            return True
        return False
    try:
        value.update(observe([directory / 'inbox', directory / 'outbox'], terminal,
                             binding['deadline_epoch']))
        return value
    finally:
        cpu = time.process_time() - started
        value['CPU_s'] = cpu
        ledger.cpu(cpu)
        write(output / 'RESULT.json', value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--binding', required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
    print(json.dumps(run(args.root, args.label, args.binding)), flush=True)
