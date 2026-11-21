"""Measure reversible compression on closed Core traces without altering original evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

GIB = 1024**3
FILES = (
    'ROS_CONSUMPTION.private.jsonl', 'init.trace', 'state.bin.journal',
    'physics.jsonl', 'init.trace.snapshots', 'planner.jsonl',
    'COMMANDS.private.jsonl', 'MOBILE_SAMPLES.private.jsonl',
)


def digest(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024**2), b''):
        h.update(chunk)
    return h.hexdigest()


def fingerprint(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns


def compress(path, packed):
    before = fingerprint(path)
    with path.open('rb') as original:
        original_sha = digest(original)
    tick = time.monotonic()
    subprocess.run(['zstd', '-q', '-T1', '-1', '-o', str(packed), '--', str(path)],
                   check=True, timeout=45)
    encoded_seconds = time.monotonic()-tick
    with tempfile.TemporaryFile() as messages:
        with subprocess.Popen(['zstd', '-q', '-T1', '-d', '-c', '--', str(packed)],
                              stdout=subprocess.PIPE, stderr=messages) as decoded:
            returned_sha = digest(decoded.stdout)
            code = decoded.wait(timeout=10)
        if code or returned_sha != original_sha or before != fingerprint(path):
            raise ValueError(f'Unchanged, byte-identical original not established for {path.name}')
    return dict(name=path.name, original_bytes=before[2], packed_bytes=packed.stat().st_size,
                original_sha256=original_sha, restored_sha256=returned_sha,
                encode_seconds=encoded_seconds, roundtrip_seconds=time.monotonic()-tick)


def measure(path, temporary):
    packed = temporary/'member.zst'
    result = compress(path, packed)
    packed.unlink()
    return result


def benchmark(producer, scratch, selectors):
    if producer.stat().st_uid != os.getuid() or scratch.stat().st_uid != os.getuid():
        raise PermissionError('Expected the original Core producer and owned benchmark directory')
    spec = json.loads((producer/'EXECUTION_SPEC.json').read_text())
    if (spec['goal_plan_sha'] != '4fe2c8cf1b69c168b2e3079e061bb7c702089cd3'
            or spec['registered_root'] != str(producer) or time.time() >= spec['campaign_deadline_epoch']):
        raise PermissionError('Outside the original Core collection scope')
    records = {row['condition']: row for row in json.loads((producer/'RECORDS.private.json').read_text())}
    if len(set(selectors)) != len(selectors) or not 1 <= len(selectors) <= 3:
        raise ValueError('Choose at most three distinct, already closed TRAIN cases')
    candidates = []
    for condition in selectors:
        row = records[condition]
        case = producer/'cases'/condition
        terminal = case/'RESULT.private.json'
        lease = json.loads((case/'control'/'LEASE.json').read_text())
        with terminal.open('rb') as stream:
            if (row['split'] != 'TRAIN' or row['requests'] != 1 or lease['status'] != 'CLOSED'
                    or row['result']['path'] != str(terminal) or row['result']['sha256'] != digest(stream)):
                raise PermissionError('Benchmark needs a sealed original TRAIN result')
        members = [case/'native-01'/name for name in FILES]
        if any(path.is_symlink() or not path.is_file() for path in members):
            raise ValueError('Expected the canonical trace paths')
        candidates.append((row, members))

    total = sum(path.stat().st_size for _, paths in candidates for path in paths)
    disk = shutil.disk_usage(scratch)
    floor = max(50*GIB, int(.15*disk.total))+8*GIB
    if total > 3*GIB or disk.free < floor+5*GIB:
        raise PermissionError('Core trace benchmark exceeds the fixed temporary budget')
    started = time.monotonic()
    results = []
    with tempfile.TemporaryDirectory(prefix='data-', dir=scratch) as temp:
        for row, members in candidates:
            result = [measure(path, Path(temp)) for path in members]
            results.append(dict(condition=row['condition'], native_ticks=row['ticks'],
                                admitted=row['admitted'], original_stop=row['first_failure'],
                                terminal_sha256=row['result']['sha256'], traces=result))
    return dict(status='BYTE_IDENTICAL_TRACE_ROUNDTRIP', codec='zstd -1 -T1',
                producer=spec['source_sha'], original_bytes=total,
                packed_bytes=sum(x['packed_bytes'] for r in results for x in r['traces']),
                seconds=time.monotonic()-started, results=results,
                optimizer_updates=0, native_requests=0, model_generations=0, original_files_rewritten=0)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('producer', type=Path)
    cli.add_argument('scratch', type=Path)
    cli.add_argument('conditions', nargs='+')
    opts = cli.parse_args()
    result = benchmark(opts.producer.resolve(strict=True), opts.scratch.resolve(strict=True), opts.conditions)
    with (opts.scratch/'RESULT.private.json').open('x') as out:
        json.dump(result, out, sort_keys=True)
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps(result))
