"""Run the original native trace readers over verified, seekable RAM-backed files."""
import argparse
from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from scripts.sla.core_hopper_carry_audit import last_stop, stopped_pairs
from scripts.sla.core_multienv_trace_archive import MAX_BYTES, MEMBERS, PLAN, SCHEMA
from scripts.sla.core_multienv_trace_storage import GIB

ARCHIVE_PAIR_SHA = '21ce84732bd67551f49bfe6a66111f1e89c7eda8eae2f813e855e8acbeef086e'
PRODUCER_SHA = '45d949a883c7c10ec31c0caef9a40bdd0a4c51225d5417dac68c7b05432e870b'
SEALS = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SEAL


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def headroom():
    line = next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    return 1024 * int(line.split()[1])


def verify_member(path, row, fd):
    if (path.is_symlink() or not path.is_file() or path.stat().st_size != row['packed_bytes']
            or sha(path) != row['packed_sha256'] or row['original_sha256'] != row['restored_sha256']):
        raise ValueError('The packed member no longer matches its sealed bytes')
    count = 0
    digest = hashlib.sha256()
    with subprocess.Popen(['zstd', '-q', '-T1', '-d', '-c', '--', str(path)], stdout=subprocess.PIPE) as p:
        try:
            with os.fdopen(os.dup(fd), 'wb') as raw:
                for data in iter(lambda: p.stdout.read(1024**2), b''):
                    count += len(data)
                    if count > row['original_bytes']:
                        raise ValueError('The archive expanded beyond its sealed length')
                    digest.update(data)
                    raw.write(data)
            if p.wait(timeout=10) or count != row['original_bytes'] or digest.hexdigest() != row['original_sha256']:
                raise ValueError('Decompression does not match the original byte identity')
        except BaseException:
            p.kill()
            raise
    os.lseek(fd, 0, os.SEEK_SET)
    fcntl.fcntl(fd, fcntl.F_ADD_SEALS, SEALS)
    if fcntl.fcntl(fd, fcntl.F_GET_SEALS) != SEALS:
        raise ValueError('Decoded member failed to become immutable')


@contextmanager
def archive_view(archive, expected, scratch):
    manifest = archive/'MANIFEST.json'
    if manifest.is_symlink() or sha(manifest) != expected:
        raise PermissionError('Archive manifest identity changed')
    spec = read(manifest)
    files = spec['files']
    total = sum(row['original_bytes'] for row in files.values())
    if (spec['schema'] != SCHEMA or spec['plan'] != PLAN or spec['state'] != 'sealed'
            or set(files) != set(MEMBERS) or not 0 < total <= MAX_BYTES
            or {p.name for p in archive.iterdir()} != {'MANIFEST.json', *(n+'.zst' for n in MEMBERS)}
            or any(files[n]['name'] != n or files[n]['original_bytes'] < 0 for n in MEMBERS)):
        raise ValueError('Unexpected archive members or byte budget')
    memory = headroom()
    if memory < 13*GIB + total:
        raise PermissionError('The original Core host memory reserve would be consumed')
    with ExitStack() as cleanup:
        view = Path(cleanup.enter_context(tempfile.TemporaryDirectory(prefix='trace-view-', dir=scratch)))
        for name in MEMBERS:
            fd = os.memfd_create(name, os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
            cleanup.callback(os.close, fd)
            verify_member(archive/(name+'.zst'), files[name], fd)
            (view/name).symlink_to(f'/proc/self/fd/{fd}')
        yield view, dict(members=len(MEMBERS), RAM_file_bytes=total, packed_bytes=sum(
            row['packed_bytes'] for row in files.values()), mem_available_before=memory,
            mem_available_after_unpack=headroom(), decoded_disk_payload_bytes=0)


def reconcile(producer, archives, scratch):
    from go2arx5_vla.world_model import init_trace_reader, mobile_boundary_metrics_v1, multitask_terminal_audit_v1
    from go2arx5_vla.world_model.native_replay import forbid_integration
    import mujoco

    if (os.uname().nodename != 'corelab-4080s' or os.getuid() != 2009 or scratch.stat().st_uid != os.getuid()
            or scratch.is_symlink() or any(scratch.is_relative_to(x) for x in (producer,archives))
            or sha(producer/'EXECUTION_SPEC.json') != PRODUCER_SHA or sha(archives/'RESULT.private.json') != ARCHIVE_PAIR_SHA
            or mujoco.__version__ != '3.2.7' or mujoco.get_mjcb_control() is not None):
        raise PermissionError('Expected the exact Core source, archives, and isolated geometry runtime')
    old = read(archives/'RESULT.private.json')
    frozen = (init_trace_reader,mobile_boundary_metrics_v1,multitask_terminal_audit_v1)
    consumer_shas = {m.__name__:sha(Path(m.__file__)) for m in frozen}
    if consumer_shas != old['consumer_source'] or len(old['cases']) != 2:
        raise PermissionError('Native readers or evidence cohort changed')
    out = []
    start = time.monotonic()
    with forbid_integration(mujoco) as attempts:
        for saved in old['cases']:
            cid = saved['id']
            native = producer/'cases'/cid
            spec = read(native/'EXECUTION_SPEC.json')
            terminal = native/'RESULT.private.json'
            if sha(terminal) != saved['native_result'] or read(native/'control/LEASE.json')['status'] != 'CLOSED':
                raise PermissionError('Expected the two completed original Core results')
            actual = read(terminal)
            model = Path(spec['model_path'])
            if sha(model) != spec['frozen_files'][str(model)]:
                raise PermissionError('Original compiled scene changed')
            with archive_view(archives/(cid+'.sealed'), saved['manifest'], scratch) as (view, memory):
                trace = init_trace_reader.audit_trace(view/'init.trace',max_native_ticks=spec['max_ticks'],state_abi=(41,38,20))
                metrics = mobile_boundary_metrics_v1.metrics(view,spec['mobile_condition'])
                keys = (set(metrics)&set(actual['metrics']))-{'task_success'}
                check_fields = ('status','native_step_records','STOP_observed','input_files_sha256','full_payload_records')
                if (not all(trace[key] == actual['trace_audit'][key] for key in check_fields)
                        or len(keys) != saved['original_metrics_fields_exact']
                        or not all(metrics[key] == actual['metrics'][key] for key in keys)):
                    raise ValueError('Original native trace and metrics disagree')
                tick = actual['actual_native_integrations']
                score = multitask_terminal_audit_v1.replay(
                    spec,view/'state.bin.journal',tick,(actual['detail'] or {}).get('score'))
                evaluated = score['score']
                stop = None
                if actual['unsafe']:
                    row = last_stop(view/'physics.jsonl')
                    m = mujoco.MjModel.from_binary_path(str(model))
                    stop = dict(tick=row['step'],pairs=stopped_pairs(m,row,'task_cube_geom'),
                                post_stop_ticks=trace['integrations_after_first_STOP'])
                    if stop != saved['stop']:
                        raise ValueError('Original STOP identity changed')
                if (tick != saved['native_ticks'] or evaluated['task_success'] != saved['success']
                        or evaluated['events'] != saved['recorded_stage_events']
                        or evaluated['missing_native_intervals'] or stop != saved['stop']):
                    raise ValueError('Original condition terminal scoring changed')
                out.append(dict(condition=cid, ticks=tick, success=evaluated['task_success'],
                                matched_metrics=len(keys), matched_score_events=evaluated['events'],
                                stop=stop, archive_sha=saved['manifest'], **memory))
    if attempts or any(scratch.glob('trace-view-*')):
        raise RuntimeError('RAM readback left temporary views or attempted native integration')
    return dict(status='RAM_BACKED_ORIGINAL_TRACE_AND_SCORE_RECONCILED', cases=out,
                source_archives=ARCHIVE_PAIR_SHA, consumer_shas=consumer_shas,
                seconds=time.monotonic()-start, original_files_rewritten=0, native_requests=0,
                native_integrations=0, optimizer_updates=0, model_generations=0)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('producer', type=Path)
    p.add_argument('archives', type=Path)
    p.add_argument('scratch', type=Path)
    args = p.parse_args()
    result = reconcile(args.producer.resolve(), args.archives.resolve(), args.scratch.resolve())
    with (args.scratch/'RESULT.private.json').open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
    print(json.dumps(result))
