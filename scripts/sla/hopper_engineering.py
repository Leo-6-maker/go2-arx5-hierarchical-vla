"""Execute one fixed engineering candidate and its separate cold output/restore passes."""
import os
from pathlib import Path
import statistics
import resource
import time

import numpy as np

from scripts.pi0292.durable import read
from scripts.pi0294.e1.journal import Journal
from scripts.sla import hopper_checkpoint as checkpoint
from scripts.sla.hopper_cold_restore import restore
from scripts.sla.hopper_engine import HopperEngine
from scripts.sla.hopper_equivalence import FirstStep
from scripts.sla.hopper_inputs import BATCH, PARENT
from scripts.sla.hopper_panels import ENGINEERING_PANEL, FixedPanels
from scripts.sla.hopper_parent_custody import digest, sync_dir, write_once


FM_WIDTH = dict(E1=1, E2=4, E3=8, E4=16)


def load_pin(pin):
    path = Path(pin['path'])
    if digest(path) != pin['sha256']:
        raise ValueError('Registered scientific input or result changed')
    return read(path)


def backend(name, available_fm):
    if name in FM_WIDTH:
        return FM_WIDTH[name], 1
    if name not in ('E5', 'E6') or set(available_fm) != set(FM_WIDTH):
        raise ValueError('Auxiliary batching requires the four observed FM outcomes')
    eligible = []
    for candidate, pin in available_fm.items():
        outcome = load_pin(pin)
        if outcome['role'] != candidate or outcome['parent'] != PARENT:
            raise ValueError('Wrong candidate predecessor')
        if outcome['qualified']:
            eligible.append((outcome['warm_p50_s'], FM_WIDTH[candidate]))
    if not eligible:
        raise ValueError('No qualified FM baseline')
    return min(eligible)[1], 2 if name == 'E5' else 4


def train(work, restored, bound, guard, reference=None):
    """The allocator owns the unique group directory; leave unknown effects for reconciliation."""
    role = bound['role']
    if role not in checkpoint.ENGINEERING or (role == 'E1') != (reference is None):
        raise ValueError('Only E1 may establish the first-step loss and gradient reference')
    if any(work.iterdir()):
        raise FileExistsError('Reconcile the existing candidate before issuing work')
    engine = HopperEngine(restored, phase=0, segment_stop=16, fm_micro=bound['fm_micro'],
                          aux_micro=bound['aux_micro'])
    if role != 'E1':
        engine.enable_pipeline()
    journal = Journal(work/'updates', role)
    timings = []
    try:
        for step in range(1, 17):
            guard()
            audit = FirstStep(work/'first-step', reference) if step == 1 else None
            before = dict(engine.gradient_windows)
            aux_start = engine.completed_aux
            def perform():
                row = engine.update(guard, audit=audit)
                visits = {d: n - before[d] for d, n in engine.gradient_windows.items()}
                if visits != BATCH or engine.completed_aux - aux_start != 4 or engine.step != step:
                    raise RuntimeError('Incomplete H200 engineering update')
                return dict(row, visits=visits)
            row = journal.once(f'step-{step:02d}', dict(step=step, binding=bound), perform)
            timings.append(row['wall_s'])
        first = work/'first-step/COMPLETE.json'
        if not read(first)['passed']:
            raise ValueError('Failed component gradient or random-input comparison')
        state = dict(parameters=engine.parameters, optimizer=engine.states, rng=engine.rng)
        save_start = time.monotonic()
        cp = checkpoint.save(work/'phase00016', state, phase=16, bound=bound, predecessor=PARENT)
        result = dict(role=role, parent=PARENT, binding=bound, checkpoint=cp, optimizer_updates=16,
                      visits=dict(engine.gradient_windows), completed_aux=engine.completed_aux,
                      save_s=time.monotonic()-save_start, process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      attempted_aux=engine.attempted_aux, first_step_sha256=digest(first),
                      update_wall_s=timings, warm_p50_s=statistics.median(timings[4:]),
                      warm_p95_s=float(np.percentile(timings[4:], 95)), qualified=False)
        write_once(work/'TRAIN.private.json', result)
        return result
    except Exception as exc:
        write_once(work/'INTERRUPTION.private.json',dict(role=role, parent=PARENT, error=str(exc),
                   observed_engine_steps=engine.step, completed_fm_windows=dict(engine.gradient_windows),
                   attempted_aux=engine.attempted_aux, completed_aux=engine.completed_aux,
                   unknown_effects_require_reconciliation=True))
        raise
    finally:
        engine.close()


def outputs(work, train_record, root, eval_assets, *, site, guard, independent=None):
    """Called in a separate process for each pass. Retain every attempted generation."""
    import jax
    from go2arx5_vla.sla_adapter.scale20k_evaluation import bytes_sha

    training = load_pin(train_record)
    bound, pin = training['binding'], training['checkpoint']
    if (training['role'] not in checkpoint.ENGINEERING or training['optimizer_updates'] != 16
            or training['parent'] != PARENT or training['visits'] != {d: v*16 for d,v in BATCH.items()}
            or training['attempted_aux'] != 64 or training['completed_aux'] != 64):
        raise ValueError('Engineering outputs require the matching complete candidate')
    if any(work.iterdir()):
        raise FileExistsError('Preserve previous attempts against the generation budget')
    guard()
    restored = restore(root, checkpoint.BANKS)
    current = checkpoint.binding(restored, role=training['role'], source=bound['source'],
                                 fm_micro=bound['fm_micro'], aux_micro=bound['aux_micro'])
    template = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), restored[2])
    graphs, frozen = restored[:2]
    del restored
    state, _ = checkpoint.resume(pin, template, phase=16, bound=current, predecessor=PARENT)
    panels = FixedPanels(root/'assets.verified', eval_assets)
    expected = load_pin(independent) if independent else None
    if expected and (expected['site'] != site or expected['checkpoint'] != pin or
                     expected['ordinary'] is not True or expected['attempted'] != 48 or
                     expected['completed'] != 48 or len(expected['rows']) != 48):
        raise ValueError('Independent evaluation requires the same H200 and complete ordinary outputs')
    attempts, completed, rows = [], [], []
    def before(kind, ordinal):
        guard()
        key = f'{kind}-{ordinal:03d}'
        if key in attempts or len(attempts) >= 48:
            raise ValueError('Repeated engineering input or exhausted output group')
        write_once(work/(key+'.intent.json'), dict(checkpoint=pin['sha256'], ordinal=ordinal, kind=kind))
        attempts.append(key)
    def after(kind, ordinal):
        key = f'{kind}-{ordinal:03d}'
        completed.append(key)
        write_once(work/(key+'.generated.json'), dict(key=key))
    def archive(kind, ordinal, row, normalized, physical, metrics):
        file = work/f'{kind}-{ordinal:03d}.npz'
        with file.open('xb') as f:
            np.savez(f, normalized=normalized, decoded=physical)
            f.flush()
            os.fsync(f.fileno())
        record = dict(kind=kind, ordinal=ordinal, identity=row['identity'], input_sha256=row['input_sha256'],
                      noise_sha256=row['noise_sha256'], normalized_sha256=bytes_sha(normalized),
                      decoded_sha256=bytes_sha(physical), file=file.name, file_sha256=digest(file), metrics=metrics)
        if expected:
            prior = expected['rows'][len(rows)]
            for key in ('kind','ordinal','identity','input_sha256','noise_sha256','normalized_sha256','decoded_sha256'):
                if prior[key] != record[key]:
                    raise ValueError(f'Independent H200 restore mismatch at {kind}/{ordinal}/{key}')
            if digest(Path(independent['path']).parent/prior['file']) != prior['file_sha256']:
                raise ValueError('Original output archive changed')
        rows.append(record)
    try:
        panels.evaluate('engineering', graphs, state, frozen, before_generation=before,
                        after_generation=after, archive=archive)
        if len(rows) != 48 or len(attempts) != 48 or attempts != completed:
            raise ValueError('Incomplete engineering output group')
        sync_dir(work)
        result = dict(role=training['role'], parent=PARENT, checkpoint=pin, site=site, panel=ENGINEERING_PANEL,
                      ordinary=expected is None, attempted=48, completed=48, rows=rows,
                      exact_restores=48 if expected else 0, warm_p50_s=training['warm_p50_s'],
                      qualified=expected is not None, training=train_record)
        write_once(work/'RESULT.private.json', result)
        return result
    except Exception as exc:
        write_once(work/'INTERRUPTION.private.json',dict(error=str(exc), attempted=attempts, completed=completed,
                   archived=len(rows), role=training['role'], parent=PARENT, qualified=False))
        raise
