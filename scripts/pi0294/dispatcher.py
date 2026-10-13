"""Admitted spawn collector. Only this coordinator writes the total queue."""
from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.iteration.queue import LiveCoordinator, digest
from scripts.pi0292.durable import read, write


def _spawn_worker(reservation_path, python):
    ticket = read(reservation_path)
    os.sched_setaffinity(0, set(ticket['cpus']))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', LP_NUM_THREADS='3',
               NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
               MUJOCO_GL='egl', GALLIUM_DRIVER='llvmpipe', LIBGL_ALWAYS_SOFTWARE='1',
               __EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json',
               TOKENIZERS_PARALLELISM='false',
               PYTHONPATH=str(Path(ticket['source_root']) / 'src') + os.pathsep + ticket['source_root'])
    env.pop('MUJOCO_EGL_DEVICE_ID', None)
    # The spawn process owns a fresh interpreter and fresh rendering context.
    command = [python, '-m', 'scripts.pi0294.episode', '--root', ticket['root'],
               '--reservation', str(reservation_path), '--output', ticket['output']]
    folder = Path(ticket['output'])
    logs = Path(ticket['root']) / 'worker_logs'
    processes = Path(ticket['root']) / 'worker_processes'
    with (logs / (ticket['id'] + '.stdout')).open('xb') as out, (logs / (ticket['id'] + '.stderr')).open('xb') as err:
        child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
            stdout=out, stderr=err, cwd=ticket['source_root'], start_new_session=True)
        write(processes / (ticket['id'] + '.json'), dict(pid=child.pid, spawn_pid=os.getpid(),
            proc_start=Path('/proc' + '/' + str(child.pid) + '/stat').read_text().split()[21],
            reservation_sha256=digest(reservation_path), deadline_epoch=ticket['deadline_epoch'],
            cpus=ticket['cpus'], fixture=False))
        try:
            code = child.wait(timeout=max(.01, ticket['deadline_epoch'] - time.time()))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                code = child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait()
        write(processes / (ticket['id'] + '_EXIT.json'), dict(returncode=code, finished_epoch=time.time(),
            receipt_exists=(folder / 'WORKER_RECEIPT.json').is_file(), fixture=False))


def dispatch(root, plan, admission, *, workers=2, python=sys.executable,
             timeout_s=600, audit=True, poll_s=.2, stage_boundary=False):
    from .run_admission import validate
    validate(root, profile='4080-simulation')
    coordinator = LiveCoordinator(root, plan, admission, workers=workers,
                                  stage_boundary=stage_boundary)
    os.sched_setaffinity(0, {coordinator.admission['allowed_cpus'][3 * workers]})
    # runpy launchers name this module __main__; spawn needs an importable target.
    spawned_worker = importlib.import_module('scripts.pi0294.dispatcher')._spawn_worker
    ctx = multiprocessing.get_context('spawn')
    pending = [r['id'] for r in coordinator.plan['episodes']]
    active, audits, audit_pending = {}, {}, []
    interrupted = False

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        while pending or active or audits or audit_pending:
            expired = time.time() >= coordinator.admission['cutoff_epoch']
            if interrupted or expired or coordinator.value['status'] != 'LIVE_RUNNING':
                pending.clear()
                audit_pending.clear()
            while pending and len(active) < workers:
                identifier = pending.pop(0)
                ticket = coordinator.reserve(identifier, timeout_s=timeout_s)
                path = Path(root) / 'reservations' / (identifier + '.json')
                proc = ctx.Process(target=spawned_worker, args=(path, python), name=identifier)
                try:
                    proc.start()
                except BaseException:
                    coordinator.unknown(identifier)
                    raise
                active[identifier] = (proc, ticket)
            for identifier, (proc, ticket) in list(active.items()):
                if proc.is_alive():
                    continue
                proc.join()
                del active[identifier]
                receipt = Path(ticket['output']) / 'WORKER_RECEIPT.json'
                try:
                    if not receipt.is_file():
                        raise ValueError('MISSING_DURABLE_RECEIPT')
                    value = read(receipt)
                    coordinator.complete(value)
                except Exception:
                    coordinator.unknown(identifier)
                    continue
                if audit:
                    audit_pending.append(identifier)
            if audit_pending and not audits and coordinator.value['status'] == 'LIVE_RUNNING':
                identifier = audit_pending.pop(0)
                reservation = coordinator.reserve_auxiliary('audit-' + identifier, cpu_count=1,
                    timeout_s=600, output_bytes=1024**2, memory_bytes=2 * 1024**3)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                    OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
                    PYTHONPATH=str(Path(coordinator.admission['source_root']) / 'src')
                        + os.pathsep + coordinator.admission['source_root'])
                command = [python, '-m', 'scripts.pi0294.audit', '--root', str(root),
                    '--id', identifier, '--asset-root', coordinator.admission['asset_root'],
                    '--reservation', str(Path(root) / 'reservations' / ('aux-audit-' + identifier + '.json')),
                    '--cpus', ','.join(map(str, reservation['cpus']))]
                output = Path(root) / 'episodes' / identifier
                log = (output / 'auditor.stdout').open('xb')
                proc = subprocess.Popen(command, env=env, cwd=coordinator.admission['source_root'],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                audits[identifier] = (proc, log, reservation)
            for identifier, (proc, log, reservation) in list(audits.items()):
                if proc.poll() is None and time.time() >= reservation['deadline_epoch']:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                if proc.poll() is None:
                    continue
                log.close()
                del audits[identifier]
                receipt = Path(root) / 'episodes' / identifier / 'AUDIT.json'
                if not receipt.is_file():
                    coordinator.value['status'] = 'LIVE_STOP_UNRECONCILED_AUXILIARY_RETAINED'
                    write(coordinator.path, coordinator.value)
                    continue
                value = read(receipt)
                coordinator.complete_auxiliary('audit-' + identifier,
                    cpu_s=value['CPU_s'], output_bytes=receipt.stat().st_size + 4096)
            if not active and not audits and not pending and not audit_pending:
                break
            time.sleep(poll_s)
        if interrupted or time.time() >= coordinator.admission['cutoff_epoch']:
            coordinator.value['status'] = 'LIVE_STOP_BOUNDARY_OR_CUTOFF'
            write(coordinator.path, coordinator.value)
        if coordinator.value['reservations'] or coordinator.value['auxiliary']:
            return dict(coordinator.value)
        if len(coordinator.value['records']) == coordinator.base_records + len(coordinator.plan['episodes']):
            return coordinator.finish()
        return dict(coordinator.value)
    finally:
        for identifier, (proc, ticket) in active.items():
            process_file = Path(root) / 'worker_processes' / (identifier + '.json')
            if process_file.is_file():
                child = read(process_file)
                identity = Path('/proc') / str(child['pid']) / 'stat'
                if identity.is_file() and identity.read_text().split()[21] == child['proc_start']:
                    try:
                        os.killpg(child['pid'], signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            if proc.is_alive():
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)
            coordinator.unknown(identifier)
        for identifier, (proc, log, reservation) in audits.items():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            log.close()
            coordinator.value['status'] = 'LIVE_STOP_UNRECONCILED_AUXILIARY_RETAINED'
            write(coordinator.path, coordinator.value)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        coordinator.close()


def main():
    p = argparse.ArgumentParser()
    for name in ('root', 'plan', 'admission'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--workers', type=int, choices=(1, 2, 4), default=2)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--timeout-s', type=float, default=600)
    p.add_argument('--no-audit', action='store_true')
    p.add_argument('--stage-boundary', action='store_true')
    args = p.parse_args()
    result = dispatch(args.root, args.plan, args.admission, workers=args.workers,
        python=args.python, timeout_s=args.timeout_s, audit=not args.no_audit,
        stage_boundary=args.stage_boundary)
    print(json.dumps(dict(status=result['status'], completed=len(result['records']),
                         unresolved=len(result['reservations']), fixture=False)))
    return 0 if result['status'] == 'LIVE_TERMINAL_ALL_FIRST_ATTEMPTS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
