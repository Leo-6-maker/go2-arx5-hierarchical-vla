"""One finite H200 main/control capsule. Emit sealed checkpoint events independently of Eval."""
import argparse
from contextlib import closing
import os
from pathlib import Path
import signal
import time

from scripts.pi0292.durable import OwnerLock, write
from scripts.pi0294.e1.journal import Journal
from scripts.sla import hopper_checkpoint as checkpoint
from scripts.sla.hopper_cold_restore import restore
from scripts.sla.hopper_curriculum import WithCoreTrain, bank_at_boundary
from scripts.sla.hopper_engine import HopperEngine
from scripts.sla.hopper_inputs import BATCH, PARENT, PLAN, STEPS, D2_SPEC, SPEC_NAME
from scripts.sla.hopper_parent_custody import write_once
from scripts.sla.hopper_site import live_h200, pinned

ENDPOINTS = (0, 2000, 5000, 10000, STEPS)
TARGET_KEYS = {'target_origin', 'target_manifest', 'target_bank_sha256'}


def qualified_backend(selection):
    record = pinned(selection)
    if record['plan'] != PLAN or record['parent'] != PARENT or record['winner'] not in checkpoint.ENGINEERING:
        raise ValueError('Unrecognized engineering selection')
    outcome = pinned(record['candidates'][record['winner']])
    training = pinned(outcome['training'])
    if (outcome['role'] != record['winner'] or outcome['parent'] != PARENT or not outcome['qualified']
            or outcome['ordinary'] or outcome['exact_restores'] != 48 or outcome['completed'] != 48
            or training['optimizer_updates'] != 16 or training['checkpoint'] != outcome['checkpoint']):
        raise ValueError('The selected H200 backend lacks its independent full restore')
    return record['winner'], training['binding']


def continuation(previous, current, start):
    if previous == current:
        return
    differing = {k for k in previous.keys() | current.keys() if previous.get(k) != current.get(k)}
    if current['role'] == 'CONTROL2000' and previous['role'] == 'MAIN':
        if start == current['role_start'] and differing == {'role', 'role_start'}:
            return
    if current['role'] == previous['role'] == 'MAIN' and start in checkpoint.BOUNDARIES:
        if (current['target_origin'] == start and previous['target_origin'] < start
                and differing <= TARGET_KEYS and 'target_manifest' in differing):
            return
    raise ValueError('Incompatible cold handoff of optimizer, backend, target data or branch')


def initial_state(root, cfg, start, stop):
    import jax
    original = restore(root, checkpoint.BANKS, check_device_arrays=True)
    graphs, frozen, state, banks, datasets, tokenizer, proof = original
    mode, winner = qualified_backend(cfg['selection'])
    course = None if cfg['course'] is None else pinned(cfg['course'])
    origin, target = 0, D2_SPEC
    if course is not None:
        origin = course['adopted_at']
        if (course['plan'] != PLAN or origin not in checkpoint.BOUNDARIES or not origin <= start
                or course['status'] != 'ADOPTED_CORE_TRAIN'):
            raise ValueError('No committed course adoption at this boundary')
        view = WithCoreTrain(datasets['target'], Path(course['folder']), course['seal_sha256'],
                             Path(cfg['population_file']))
        target = view.seal['manifest_sha256']
        if target != course['manifest_sha256']:
            raise ValueError('Ready target identity changed after adoption')
        original_spec = pinned(dict(path=str(root/'assets.verified'/SPEC_NAME), sha256=D2_SPEC))
        fresh, _ = bank_at_boundary(Path(course['bank_root']), view, origin, original_spec['target_seed'])
        banks = dict(banks, target=fresh)
        datasets = dict(datasets, target=view)
    restored = (graphs, frozen, state, banks, datasets, tokenizer, proof)
    bound = checkpoint.binding(restored, role=cfg['role'], source=cfg['source'], role_start=cfg['role_start'],
                               fm_micro=winner['fm_micro'], aux_micro=winner['aux_micro'],
                               target_origin=origin, target_manifest=target)
    if cfg['source'] != winner['source']:
        raise ValueError('Formal training must use the source that qualified its backend')
    checkpoint.progress(bound, start)
    checkpoint.progress(bound, stop)
    prior = cfg['previous']
    if prior['sha256'] == PARENT:
        if start != 0 or cfg['role'] != 'MAIN' or origin:
            raise ValueError('Only phase zero of the main arm starts from ADAPT2K')
    else:
        path = Path(prior['path'])/'COMPLETE.json'
        header = pinned(dict(path=str(path), sha256=prior['sha256']))
        previous = header['binding']
        continuation(previous, bound, start)
        if cfg['role'] == 'CONTROL2000' and previous['role'] == 'MAIN':
            pair = pinned(cfg['first_control_trigger'])
            if pair['boundary'] != start or pair['common_parent']['sha256'] != prior['sha256'] or pair['control_target'] != target:
                raise ValueError('Wrong common parent or pre-switch target for CONTROL2000')
        template = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), state)
        del original, restored, state
        state, _ = checkpoint.resume(prior, template, phase=start, bound=previous)
        restored = (graphs, frozen, state, banks, datasets, tokenizer, proof)
    engine = HopperEngine(restored, phase=start, segment_stop=stop, fm_micro=bound['fm_micro'],
                          aux_micro=bound['aux_micro'], target_origin=origin)
    if mode != 'E1':
        engine.enable_pipeline()
    return engine, bound


def train_segment(run, line, cfg, engine, bound, guard, *, matched_endpoint):
    start, stop = engine.step, engine.segment_stop
    last = cfg['previous']
    last_phase = start
    saved_at = time.monotonic()
    rows = Journal(line/'updates', bound['role'])
    events = line/'sealed'
    events.mkdir(exist_ok=True)
    checks = line/'checkpoints'
    checks.mkdir(exist_ok=True)

    def seal():
        nonlocal last, last_phase, saved_at
        phase = engine.step
        state = dict(parameters=engine.parameters, optimizer=engine.states, rng=engine.rng)
        pin = checkpoint.save(checks/f'phase{phase:05d}', state, phase=phase, bound=bound, predecessor=last['sha256'])
        deployment = (bound['role'] == 'MAIN' and phase in ENDPOINTS) or phase == matched_endpoint
        event = dict(plan=PLAN, role=bound['role'], phase=phase, checkpoint=pin, predecessor=last['sha256'],
                     binding=bound, cursors=checkpoint.progress(bound, phase), course=cfg['course'],
                     eval_due=deployment, deployment_due=deployment, backup_due=True,
                     offline_status='OFFLINE_EVAL_PENDING' if deployment else 'NOT_A_FORMAL_ENDPOINT')
        write_once(events/f'phase{phase:05d}.json', event)
        last, last_phase, saved_at = pin, phase, time.monotonic()

    if start == 0 and last['sha256'] == PARENT:
        seal()
    mark = 'SEGMENT_COMPLETE'
    try:
        while engine.step < stop:
            try:
                guard()
            except TimeoutError:
                mark = 'SAVED_AT_ALLOCATION_OR_STOP'
                break
            phase = engine.step + 1
            key = f'phase{phase:05d}'
            # The old in-memory parameters cannot advance just by returning a prior receipt.
            if rows.path(key).exists():
                raise RuntimeError('Earlier effect after the selected seal; reconcile the suffix before replay')
            before = dict(engine.gradient_windows)
            a0, c0 = engine.attempted_aux, engine.completed_aux
            def apply():
                row = engine.update(guard)
                visits = {d: v-before[d] for d,v in engine.gradient_windows.items()}
                if visits != BATCH or engine.step != phase or engine.completed_aux-c0 != 4 or engine.attempted_aux-a0 != 4:
                    raise RuntimeError('Wrong number of scientific training examples or generated H40')
                return dict(row, visits=visits, role=bound['role'], job=cfg['job'])
            try:
                row = rows.once(key, dict(phase=phase, binding=bound, job=cfg['job']), apply)
            except Exception as exc:
                write_once(run/'PARTIAL.private.json', dict(phase=phase, job=cfg['job'], error=str(exc),
                           observed_phase=engine.step, completed_fm={d:n-before[d] for d,n in engine.gradient_windows.items()},
                           auxiliary_attempts=engine.attempted_aux-a0, completed_auxiliary=engine.completed_aux-c0))
                if engine.step == phase-1 and engine.step > last_phase:
                    seal()
                raise
            write(run/'PROGRESS.private.json', dict(phase=phase, cursors=row['phase_cursors'], epoch=time.time(), job=cfg['job']))
            if (phase == stop or phase in ENDPOINTS or phase in checkpoint.BOUNDARIES or phase == matched_endpoint
                    or phase % 100 == 0 or time.monotonic()-saved_at >= 1800):
                seal()
        if engine.step > last_phase:
            seal()
        return dict(status=mark, start=start, phase=engine.step, updates=engine.step-start, last_checkpoint=last,
                    visits=dict(engine.gradient_windows), attempted_H40=engine.attempted_aux, completed_H40=engine.completed_aux)
    finally:
        engine.close()


def worker(run, mount_sha256):
    mount = pinned(dict(path=str(run/'MOUNT.private.json'), sha256=mount_sha256))
    cfg = pinned(mount['training_segment'])
    role = cfg['role']
    if (role not in ('MAIN', 'CONTROL2000') or cfg['plan'] != PLAN or cfg['parent'] != PARENT
            or cfg['job'] != mount['job'] or cfg['source'] != mount['source_revision']):
        raise PermissionError('Wrong training segment or PBS job')
    site = live_h200(mount, 'training')
    root = Path(mount['custody'])
    line = Path(mount['lineage'])/role
    line.mkdir(exist_ok=True)
    start, stop = cfg['start'], cfg['stop']
    matched_endpoint = None
    if cfg.get('first_control_trigger'):
        pair = pinned(cfg['first_control_trigger'])
        if pair['boundary'] not in checkpoint.BOUNDARIES:
            raise ValueError('Invalid first curriculum trigger')
        matched_endpoint = pair['boundary']+2000
    if role == 'CONTROL2000' and matched_endpoint != cfg['role_start']+2000:
        raise ValueError('Control branch requires a registered trigger')
    if not start < stop or (role == 'MAIN' and any(start < k < stop for k in checkpoint.BOUNDARIES)):
        raise ValueError('A curriculum decision requires a cold boundary')
    stopped = False
    def stop_signal(*_):
        nonlocal stopped
        stopped = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop_signal)
    def guard():
        if stopped or (run/'STOP.json').exists() or (root/'STOP.json').exists() or time.time() >= site['deadline']-900:
            raise TimeoutError('Saving before the actual site or goal deadline')
    with closing(OwnerLock(line/'OWNER.lock')):
        guard()
        write_once(run/'START.private.json', dict(source=cfg['source'], job=cfg['job'], site=site, epoch=time.time()))
        import jax
        jax.config.update('jax_compilation_cache_dir', mount['compilation_cache_dir'])
        engine, bound = initial_state(root, cfg, start, stop)
        result = train_segment(run, line, cfg, engine, bound, guard, matched_endpoint=matched_endpoint)
        write_once(run/'RESULT.private.json', dict(result, site=site, source=cfg['source'], role=role))


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('run', type=Path)
    cli.add_argument('mount_sha256')
    args = cli.parse_args()
    os.umask(0o077)
    worker(args.run, args.mount_sha256)
